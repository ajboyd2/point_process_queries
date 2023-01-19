from ast import Assert
from numpy import empty
import torch
import torch.nn as nn
import torch.nn.functional as F

NORMS = (
    nn.LayerNorm,
    nn.BatchNorm1d,
    nn.BatchNorm2d,
    nn.BatchNorm3d,
    nn.GroupNorm,
    nn.InstanceNorm1d,
    nn.InstanceNorm2d,
    nn.InstanceNorm3d,
    nn.LocalResponseNorm,
)

MAX_SAMPLE_BATCH_SIZE = 1024

def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Adapted from https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317

        Args:
            logits: logits distribution shape (vocabulary size)
            top_k >0: keep only top k tokens with highest probability (top-k filtering).
            top_p >0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
    """
    logits = logits.squeeze()
    assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits.unsqueeze(0).unsqueeze(0)


class PPModel(nn.Module):

    def __init__(
        self, 
        decoder, 
        num_channels,
        dominating_rate=10000.,
        dyn_dom_buffer=4,
    ):
        """Constructor for general PPModel class.
        
        Arguments:
            decoder {torch.nn.Module} -- Neural network decoder that accepts a latent state, marks, timestamps, and times of sample points. 

        Keyword Arguments:
            time_embedding {torch.nn.Module} -- Function to transform k-dimensional timestamps into (k+1)-dimensional embedded vectors. If specified, will make encoder and decoder share this function. (default: {None})
            encoder {torch.nn.Module} -- Neural network encoder that accepts marks and timestamps and returns a single latent state (default: {None})
            aggregator {torch.nn.Module} -- Module that turns a tensor of hidden states into a latent vector (with noise added during training) (default: {None})
        """
        super().__init__()

        self.decoder = decoder
        self.num_channels = num_channels
        self.dominating_rate = dominating_rate
        self.dyn_dom_buffer = dyn_dom_buffer

    def get_states(self, marks, timestamps, old_states=None):
        """Get the hidden states that can be used to extract intensity values from."""
        
        states = self.decoder.get_states(
            marks=marks, 
            timestamps=timestamps, 
            old_states=old_states,
        )

        return {
            "state_values": states,
            "state_times": timestamps,
        }

    def get_intensity(self, state_values, state_times, timestamps, marks=None, state_marks=None, mark_mask=1.0, censoring=None):
        """Given a set of hidden states, timestamps, and latent_state get a tensor representing intensity valu[es at timestamps.
        Specify marks to get intensity values for specific channels."""

        if (state_values is None) and (state_marks is not None):
            state_values = self.get_states(state_marks, state_times)["state_values"]

        intensity_dict = self.decoder.get_intensity(
            state_values=state_values,
            state_times=state_times,
            timestamps=timestamps,
            mark_mask=mark_mask,
        )

        if marks is not None:
            intensity_dict["log_mark_intensity"] = intensity_dict["all_log_mark_intensities"].gather(dim=-1, index=marks.unsqueeze(-1)).squeeze(-1)
        
        if censoring is not None:
            masks = censoring.get_mask(timestamps)

            intensity_dict["censored_int"] = intensity_dict["all_mark_intensities"] * masks["censored_mask"]
            intensity_dict["observed_int"] = intensity_dict["all_mark_intensities"] * masks["observed_mask"]

            if censoring.overwrite_with_censored:
                intensity_dict["all_mark_intensities"] = intensity_dict["censored_int"]
                intensity_dict["all_log_mark_intensities"] = torch.log(intensity_dict["censored_int"] + 1e-12)
                intensity_dict["total_intensity"] = intensity_dict["censored_int"].sum(dim=-1)
            elif censoring.overwrite_with_observed:
                intensity_dict["all_mark_intensities"] = intensity_dict["observed_int"]
                intensity_dict["all_log_mark_intensities"] = torch.log(intensity_dict["observed_int"] + 1e-12)
                intensity_dict["total_intensity"] = intensity_dict["observed_int"].sum(dim=-1)

        return intensity_dict 
        
    def forward(self, marks, timestamps, sample_timestamps=None):
        """Encodes a(n optional) set of marks and timestamps into a latent vector, 
        then decodes corresponding intensity values for a target set of timestamps and marks 
        (as well as a sample set if specified).
        
        Arguments:
            ref_marks {torch.LongTensor} -- Tensor containing mark ids that correspond to channel embeddings. Part of the reference set to be encoded.
            ref_timestamps {torch.FloatTensor} -- Tensor containing times that correspond to the events in `ref_marks`. Part of the reference set to be encoded.
            ref_marks_bwd {torch.LongTensor} -- Tensor containing reverse mark ids that correspond to channel embeddings. Part of the reference set to be encoded.
            ref_timestamps_bwd {torch.FloatTensor} -- Tensor containing reverse times that correspond to the events in `ref_marks`. Part of the reference set to be encoded.
            tgt_marks {torch.FloatTensor} -- Tensor containing mark ids that correspond to channel embeddings. These events will be decoded and are assumed to have happened.
            tgt_timestamps {torch.FloatTensor} -- Tensor containing times that correspond to the events in `tgt_marks`. These times will be decoded and are assumed to have happened.
            context_lengths {torch.LongTensor} -- Tensor containing position ids that correspond to last events in the reference material.

        Keyword Arguments:
            sample_timestamps {torch.FloatTensor} -- Times that will have intensity values generated for. These events are _not_ assumed to have happened. (default: {None})
        
        Returns:
            dict -- Dictionary containing the produced latent vector, intermediate hidden states, and intensity values for target sequence and sample points.
        """
        return_dict = {}
        if marks is None:
            marks = torch.LongTensor([[]], device=next(self.parameters()).device)
            timestamps = torch.FloatTensor([[]], device=next(self.parameters()).device)

        # Decoding phase
        intensity_state_dict = self.get_states(
            marks=marks,
            timestamps=timestamps,
        )
        return_dict["state_dict"] = intensity_state_dict

        intensities = self.get_intensity(
            state_values=intensity_state_dict["state_values"],
            state_times=intensity_state_dict["state_times"],
            timestamps=timestamps,
            marks=marks,
        )
        return_dict["intensities"] = intensities

        # Sample intensities for objective function
        if sample_timestamps is not None:
            sample_intensities = self.get_intensity(
                state_values=intensity_state_dict["state_values"],
                state_times=intensity_state_dict["state_times"],
                timestamps=sample_timestamps,
                marks=None,
            )
            return_dict["sample_intensities"] = sample_intensities

        return return_dict

    @staticmethod
    def log_likelihood(return_dict, right_window, left_window=0.0, mask=None, reduce=True, normalize_by_window=False):
        """Computes per-batch log-likelihood from the results of a forward pass (that included a set of sample points). 
        
        Arguments:
            return_dict {dict} -- Output from a forward call where `tgt_marks` and `sample_timestamps` were not None
            right_window {float} -- Upper-most value that was considered when the sampled points were generated

        Keyword Arguments:
            left_window {float} -- Lower-most value that was considered when the sampled points were generated (default: {0})
            mask {FloatTensor} -- Mask to delineate target intensities that correspond to real events and paddings (default: {None}) 
        """

        assert("intensities" in return_dict and "log_mark_intensity" in return_dict["intensities"])
        assert("sample_intensities" in return_dict)

        if mask is None:
            mask = 1
        else:
            assert(all(x == y for x,y in zip(return_dict["intensities"]["log_mark_intensity"].shape, mask.shape)))  # make sure they are same size

        log_mark_intensity = return_dict["intensities"]["log_mark_intensity"]
        window = right_window - left_window
        if len(window.shape) > 1:
            window = window.squeeze(-1)
        if reduce:
            positive_samples = torch.where(mask, log_mark_intensity, torch.zeros_like(log_mark_intensity)).sum(dim=-1)
            #negative_samples = (right_window - left_window) * return_dict["sample_intensities"]["total_intensity"].mean(dim=-1)  # Summing and divided by number of samples
            negative_samples = window * return_dict["sample_intensities"]["total_intensity"].mean(dim=-1)  # Summing and divided by number of samples

            if normalize_by_window:
                norm = right_window.squeeze(dim=-1)
            else:
                norm = 1.0

            ll_results = {
                "log_likelihood": (((1.0 * positive_samples) - negative_samples)/norm).mean(),
                "positive_contribution": positive_samples.mean(),
                "negative_contribution": negative_samples.mean(),
            }
            return ll_results
        else:
            positive_samples = torch.where(mask, log_mark_intensity, torch.zeros_like(log_mark_intensity))
            negative_samples = return_dict["sample_intensities"]["total_intensity"]  # Summing and divided by number of samples

            return {
                "positive_contribution": positive_samples,
                "negative_contribution": negative_samples,
                "cross_entropy": -(positive_samples - return_dict["intensities"]["total_intensity"].log()),
                "batch_log_likelihood": positive_samples.sum(dim=-1) - (window * negative_samples.mean(dim=-1)),
            }


    def sample_points(self, marks, timestamps, dominating_rate=None, T=float('inf'), left_window=0.0, length_limit=float('inf'), mark_mask=1.0, top_k=0, top_p=0.0, proposal_batch_size=1024):
        assert((T < float('inf')) or (length_limit < float('inf')))
        if dominating_rate is None:
            dominating_rate = self.dominating_rate
        if marks is None:
            marks = torch.LongTensor([[]], device=next(self.parameters()).device)
            timestamps = torch.FloatTensor([[]], device=next(self.parameters()).device)

        if T < float('inf'):
            proposal_batch_size = max(min(proposal_batch_size, int(dominating_rate*(T-left_window)*5)), 10)  # dominating_rate*(T-left_window) is the expected number of proposal times to draw from [left_window, T]

        state = self.forward(marks, timestamps)
        state_values, state_times = state["state_dict"]["state_values"], state["state_dict"]["state_times"]
        
        dist = torch.distributions.Exponential(dominating_rate)
        dist.rate = dist.rate.to(state_values.device)
        last_time = left_window 
        #new_time = last_time + dist.sample(sample_shape=torch.Size((1,1))).to(state_values.device)
        new_times = last_time + dist.sample(sample_shape=torch.Size((1, proposal_batch_size))).cumsum(dim=-1)
        sampled_times = []
        sampled_marks = []
        #while (new_time < T) and (timestamps.shape[-1] < length_limit):
        while (new_times <= T).any() and (timestamps.shape[-1] < length_limit):
            new_times = new_times[new_times <= T].unsqueeze(0)
            sample_intensities = self.get_intensity(
                state_values=state_values,
                state_times=state_times,
                timestamps=new_times,
                marks=None,
                mark_mask=mark_mask,
            )

            acceptances = torch.rand_like(new_times) <= (sample_intensities["total_intensity"] / dominating_rate)
            if acceptances.any():
                idx = acceptances.squeeze(0).float().argmax()
                new_time = new_times[:, [idx]]

                if top_k > 0 or top_p > 0:
                    logits = top_k_top_p_filtering(sample_intensities["all_log_mark_intensities"][:, [idx], :], top_k=top_k, top_p=top_p)
                else:
                    logits = sample_intensities["all_log_mark_intensities"][:, [idx], :]
                mark_probs = F.softmax(logits, -1) #(sample_intensities["all_log_mark_intensities"] - sample_intensities["total_intensity"].unsqueeze(-1).log()).exp()
                mark_dist = torch.distributions.Categorical(mark_probs)
                new_mark = mark_dist.sample()
                timestamps = torch.cat((timestamps, new_time), -1)
                marks = torch.cat((marks, new_mark), -1)
                sampled_times.append(new_time.squeeze().item())
                sampled_marks.append(new_mark.squeeze().item())

                state = self.forward(marks, timestamps)
                state_values, state_times = state["state_dict"]["state_values"], state["state_dict"]["state_times"]
                last_time = new_times[:, idx].squeeze()
            else:
                last_time = new_times.max()
        
            new_times = last_time + dist.sample(sample_shape=(1, proposal_batch_size)).cumsum(dim=-1)

        assumption_violation = False
        for _ in range(5):
            eval_times = torch.rand_like(timestamps).clamp(min=1e-8)*T
            sample_intensities = self.get_intensity(
                state_values=state_values,
                state_times=state_times,
                timestamps=eval_times,
                marks=None,
            )
            if (sample_intensities["total_intensity"] > dominating_rate).any().item():
                print("DR: {}".format(dominating_rate))
                print("IN: {}".format(sample_intensities["total_intensity"].max().item()))
                assumption_violation = True
                break

        if assumption_violation:
            print("Violation in sampling assumption occurred. Redoing sample.")
            return None # self.sample_points(ref_marks, ref_timestamps, ref_marks_bwd, ref_timestamps_bwd, tgt_marks, tgt_timestamps, context_lengths, dominating_rate * 2, T)
        else:
            return (timestamps, marks)
            #     torch.cat((timestamps, torch.FloatTensor([sampled_times])), dim=-1), 
            #     torch.cat((marks, torch.LongTensor([sampled_marks])), dim=-1),
            # )
        
    def determine_mark_mask(self, new_times, sample_lens, mask_dict):
        if "temporal_mark_restrictions" in mask_dict:
            mark_masks = mask_dict["temporal_mark_restrictions"]  # (num_boundaries, num_channels)
            time_boundaries = mask_dict["time_boundaries"]  # (num_boundaries,)
            idx = (new_times.unsqueeze(-1) >= time_boundaries.unsqueeze(-2)).sum(dim=-1)
        elif "positional_mark_restrictions" in mask_dict:
            mark_masks = mask_dict["positional_mark_restrictions"]  # (num_positions, num_channels)
            idx = sample_lens.unsqueeze(-1).expand(*new_times.shape)
        else:
            raise NotImplementedError

        return F.embedding(idx, mark_masks)

    def batch_sample_points(
        self, 
        marks, 
        timestamps, 
        dominating_rate=None, 
        T=float('inf'), 
        left_window=0.0, 
        length_limit=float('inf'), 
        mark_mask=1.0,  #TODO: Make default value None instead of 1.0
        top_k=0, 
        top_p=0.0, 
        num_samples=1, 
        proposal_batch_size=1024, 
        mask_dict=None, 
        adapt_dom_rate=True,
        stop_marks=None,
        censoring=None,
    ):
        dyn_dom_buffer = self.dyn_dom_buffer
        if num_samples > MAX_SAMPLE_BATCH_SIZE:  # Split into batches
            resulting_times, resulting_marks, resulting_states = [], [], []
            remaining_samples = num_samples
            while remaining_samples > 0:
                current_batch_size = min(remaining_samples, MAX_SAMPLE_BATCH_SIZE)
                sampled_times, sampled_marks, sampled_states = self.batch_sample_points(
                    marks=marks, 
                    timestamps=timestamps, 
                    dominating_rate=dominating_rate, 
                    T=T, 
                    left_window=left_window, 
                    length_limit=length_limit, 
                    mark_mask=mark_mask,  #TODO: Make default value None instead of 1.0
                    top_k=top_k, 
                    top_p=top_p, 
                    num_samples=current_batch_size, 
                    proposal_batch_size=proposal_batch_size, 
                    mask_dict=mask_dict, 
                    adapt_dom_rate=adapt_dom_rate,
                    stop_marks=stop_marks,
                    censoring=censoring,
                )
                remaining_samples -= current_batch_size
                resulting_times.extend(sampled_times)
                resulting_marks.extend(sampled_marks)
                resulting_states.extend(sampled_states)
            return resulting_times, resulting_marks, resulting_states
        
        stop_for_marks = stop_marks is not None
        assert((T < float('inf')) or (length_limit < float('inf')) or stop_for_marks)
        if mask_dict is None:
            mask_dict = {}
        if dominating_rate is None:
            dominating_rate = self.dominating_rate
        if marks is None:
            marks = torch.tensor([[]], dtype=torch.long, device=next(self.parameters()).device)
            timestamps = torch.tensor([[]], dtype=torch.float, device=next(self.parameters()).device)
        if isinstance(left_window, torch.Tensor):
            left_window = left_window.item()
        if isinstance(T, torch.Tensor):
            T = T.item()
        if length_limit == float('inf'):
            length_limit = torch.iinfo(torch.int64).max  # Maximum Long value

        sample_lens = torch.zeros((num_samples,), dtype=torch.int64).to(next(self.parameters()).device) + timestamps.numel()
        marks, timestamps = marks.expand(num_samples, *marks.shape[1:]), timestamps.expand(num_samples, *timestamps.shape[1:])
        time_pad, mark_pad = torch.nan_to_num(torch.tensor(float('inf'), dtype=timestamps.dtype)).item(), 0
        state = self.forward(marks, timestamps)
        state_values, state_times = state["state_dict"]["state_values"], state["state_dict"]["state_times"]
        batch_idx = torch.arange(num_samples).to(state_values.device)
        finer_proposal_batch_size = max(proposal_batch_size // 4, 16)

        dist = torch.distributions.Exponential(dominating_rate)
        dist.rate = dist.rate.to(state_values.device)*0+1  # We will manually apply the scale to samples
        dominating_rate = torch.ones((num_samples,1), dtype=torch.float32).to(state_values.device)*dominating_rate
        last_time = torch.ones_like(dominating_rate) * left_window if isinstance(left_window, (int, float)) else left_window
        #new_time = last_time + dist.sample(sample_shape=torch.Size((1,1))).to(state_values.device)
        new_times = last_time + dist.sample(sample_shape=torch.Size((num_samples, proposal_batch_size))).cumsum(dim=-1)/dominating_rate
        stop_marks = stop_marks if stop_for_marks else torch.tensor([], dtype=torch.long).to(state_values.device)
        sample_hasnt_hit_stop_marks = torch.ones((num_samples,), dtype=torch.bool).to(state_values.device)

        calculate_mark_mask = isinstance(mark_mask, float) and (("temporal_mark_restrictions" in mask_dict) or ("positional_mark_restrictions" in mask_dict))
        resulting_times, resulting_marks, resulting_states = [], [], []

        if adapt_dom_rate:
            dynamic_dom_rates = torch.ones((num_samples, dyn_dom_buffer,)).to(state_values.device)*dominating_rate
            k = 0
        j = -1
        while (new_times <= T).any() and (sample_lens < length_limit).any():
            j += 1
            within_range_mask = (new_times <= T) & (sample_lens < length_limit).unsqueeze(-1) & sample_hasnt_hit_stop_marks.unsqueeze(-1)
            to_stay = within_range_mask.any(dim=-1)
            to_go = ~to_stay
            if to_go.any():
                if stop_for_marks:
                    leaving_times, leaving_marks, leaving_states = timestamps[to_go, sample_lens[to_go]-1], marks[to_go, sample_lens[to_go]-1], state_values[to_go, sample_lens[to_go]-1, ...]
                else:
                    leaving_times, leaving_marks, leaving_states = timestamps[to_go, ...], marks[to_go, ...], state_values[to_go, ...]
                resulting_times.append(leaving_times)
                resulting_marks.append(leaving_marks)
                resulting_states.append(leaving_states)
                # for i,l in enumerate(sample_lens[to_go]):
                #     resulting_times.append(leaving_times[i, :l])
                #     resulting_marks.append(leaving_marks[i, :l])
                new_times = new_times[to_stay, ...]
                sample_lens = sample_lens[to_stay]
                timestamps = timestamps[to_stay, ...]
                marks = marks[to_stay, ...]
                state_values = state_values[to_stay, ...]
                state_times = state_times[to_stay, ...]
                batch_idx = batch_idx[:timestamps.shape[0]]
                last_time = last_time[to_stay, ...]
                dominating_rate = dominating_rate[to_stay, ...]
                sample_hasnt_hit_stop_marks = sample_hasnt_hit_stop_marks[to_stay]
                if adapt_dom_rate:
                    dynamic_dom_rates = dynamic_dom_rates[to_stay, ...]
                if batch_idx.numel() == 0:
                    break  # STOP SAMPLING

            if calculate_mark_mask:
                mark_mask = self.determine_mark_mask(new_times, sample_lens, mask_dict)

            within_range_mask = (new_times <= T) & (sample_lens < length_limit).unsqueeze(-1)

            # print(dist.rate, new_times.min(), new_times.max(), sample_lens.min(), new_times.shape[0], (new_times > 3.4028e+36).any())

            sample_intensities = self.get_intensity(
                state_values=state_values,
                state_times=state_times,
                timestamps=new_times,
                marks=None,
                mark_mask=mark_mask,
                censoring=censoring,
            )
            redo_samples = torch.zeros_like(batch_idx, dtype=torch.bool)
            if adapt_dom_rate:  # Need to check and make sure that we don't break the sampling assumption
                redo_samples = (sample_intensities["total_intensity"] > dominating_rate).any(dim=-1)
                if not redo_samples.any():
                    finer_new_times = last_time + dist.sample(sample_shape=torch.Size((last_time.shape[0], finer_proposal_batch_size))).cumsum(dim=-1)/(dominating_rate*proposal_batch_size/4)
                    # print("\tFine:", dist.rate, finer_new_times.min(), finer_new_times.max(), sample_lens.min(), finer_new_times.shape[0], (finer_new_times > 3.4028e+36).any())
                    finer_mark_mask = self.determine_mark_mask(finer_new_times, sample_lens, mask_dict) if calculate_mark_mask else 1.0
                    finer_sample_intensities = self.get_intensity(  # Finer resolution check just after `last_time`
                        state_values=state_values,
                        state_times=state_times,
                        timestamps=finer_new_times,
                        marks=None,
                        mark_mask=finer_mark_mask,
                        censoring=censoring,
                    )
                    redo_samples = redo_samples | (finer_sample_intensities["total_intensity"] > dominating_rate).any(dim=-1)
            keep_samples = ~redo_samples

            # if not redo_sample:
            # print(j, new_times.shape[0], new_times.min(), new_times.max(), T, sample_lens.min(), dominating_rate.min(), dominating_rate.mean(), dominating_rate.max())
            acceptances = torch.rand_like(new_times) <= (sample_intensities["total_intensity"] / dominating_rate)  #dominating_rate)
            acceptances = acceptances & within_range_mask & keep_samples.unsqueeze(-1) # Don't accept any sampled events outside the window or that need to be redone
            samples_w_new_events = acceptances.any(dim=-1)
            if samples_w_new_events.any():
                event_idx = acceptances.int().argmax(dim=-1)
                new_time = new_times[batch_idx, event_idx].unsqueeze(-1)

                if top_k > 0 or top_p > 0:
                    logits = top_k_top_p_filtering(sample_intensities["all_log_mark_intensities"][batch_idx, event_idx, :].unsqueeze(-2), top_k=top_k, top_p=top_p)
                else:
                    logits = sample_intensities["all_log_mark_intensities"][batch_idx, event_idx, :].unsqueeze(-2)
                mark_probs = F.softmax(logits, -1)
                mark_dist = torch.distributions.Categorical(mark_probs)
                new_mark = mark_dist.sample()

                # Need to store sampled events into timestamps and marks
                # Some need to be appended, some need to overwrite previously written padded values
                to_append = (samples_w_new_events & (sample_lens == timestamps.shape[-1])).unsqueeze(-1)
                to_pad = ~to_append
                if to_append.any():
                    timestamps = torch.cat((timestamps, torch.where(to_append, new_time, time_pad)), -1)  #new_time*to_append + time_pad*to_pad), -1)
                    marks = torch.cat((marks, torch.where(to_append, new_mark, mark_pad)), -1)  #new_mark*to_append + mark_pad*to_pad), -1)

                to_overwrite = samples_w_new_events & (sample_lens < timestamps.shape[-1])
                if to_overwrite.any():
                    timestamps[to_overwrite, sample_lens[to_overwrite]] = new_time.squeeze(-1)[to_overwrite]
                    marks[to_overwrite, sample_lens[to_overwrite]] = new_mark.squeeze(-1)[to_overwrite]

                sample_lens[samples_w_new_events] += 1  # Guaranteed at least one event was either appended or overwritten
                if stop_for_marks:
                    sample_hasnt_hit_stop_marks = torch.where(
                        samples_w_new_events,
                        ~torch.isin(new_mark.squeeze(-1), stop_marks),
                        sample_hasnt_hit_stop_marks,
                    )
                state = self.forward(marks, timestamps)
                state_values, state_times = state["state_dict"]["state_values"], state["state_dict"]["state_times"]
                last_time = torch.where(
                    redo_samples.unsqueeze(-1),
                    last_time,   
                    torch.where(samples_w_new_events.unsqueeze(-1), new_time, torch.max(new_times, dim=-1, keepdim=True).values),  #new_time*samples_w_new_events.unsqueeze(-1) + (torch.max(new_times, dim=-1).values*(~samples_w_new_events)).unsqueeze(-1)
                )
            else:
                last_time = torch.where(
                    redo_samples.unsqueeze(-1),
                    last_time,   
                    torch.max(new_times, dim=-1, keepdim=True).values,
                )
        # else: # Redo sample
            # print("j={} samples left={} d_rate={} si_max={} fsi_max={}".format(j, dominating_rate.shape[0], dominating_rate.max(), sample_intensities["total_intensity"].max(), finer_sample_intensities["total_intensity"].max()))

            if adapt_dom_rate:  
                dynamic_dom_rates[:, k] = sample_intensities["total_intensity"].max(dim=1).values*100
                k = (k+1) % dynamic_dom_rates.shape[1]
                dominating_rate = torch.max(dynamic_dom_rates, dim=1, keepdim=True).values

            # print(last_time.shape, new_times.shape, proposal_batch_size, dominating_rate.shape)
            new_times = last_time + dist.sample(sample_shape=(new_times.shape[0], proposal_batch_size)).cumsum(dim=-1)/dominating_rate

        if timestamps.shape[0] > 0:  # On the chance that we hit a break after running out of samples within the loop
            if stop_for_marks:
                if (~sample_hasnt_hit_stop_marks).any():
                    resulting_times.append(timestamps[~sample_hasnt_hit_stop_marks, sample_lens-1]) 
                    resulting_marks.append(marks[~sample_hasnt_hit_stop_marks, sample_lens-1])
                    resulting_states.append(state_values[~sample_hasnt_hit_stop_marks, sample_lens-1, ...])
                    timestamps, marks, state_values = timestamps[sample_hasnt_hit_stop_marks, ...], marks[sample_hasnt_hit_stop_marks, ...], state_values[sample_hasnt_hit_stop_marks, ...]
                if timestamps.shape[0] > 0:  
                    # Hit right window limit but still have samples that haven't ended
                    # Append sequences with times=T
                    resulting_times.append(timestamps[:, -1]*0+T)
                    resulting_marks.append(marks[:,-1]*0)
                    resulting_states.append(state_values[:, -1])  # These shouldn't be used
            else:
                resulting_times.append(timestamps) 
                resulting_marks.append(marks)
                resulting_states.append(state_values)

        assumption_violation = False
        if not adapt_dom_rate:
            for _ in range(5):
                eval_times = torch.rand_like(timestamps).clamp(min=1e-8)*T
                sample_intensities = self.get_intensity(
                    state_values=state_values,
                    state_times=state_times,
                    timestamps=eval_times,
                    marks=None,
                )
                if (sample_intensities["total_intensity"] > dominating_rate).any().item():
                    print("DR: {}".format(dominating_rate))
                    print("IN: {}".format(sample_intensities["total_intensity"].max().item()))
                    assumption_violation = True
                    break

        if assumption_violation:
            print("Violation in sampling assumption occurred. Redoing sample.")
            return None # self.sample_points(ref_marks, ref_timestamps, ref_marks_bwd, ref_timestamps_bwd, tgt_marks, tgt_timestamps, context_lengths, dominating_rate * 2, T)
        else:
            return resulting_times, resulting_marks, resulting_states #(timestamps, marks)
            #     torch.cat((timestamps, torch.FloatTensor([sampled_times])), dim=-1), 
            #     torch.cat((marks, torch.LongTensor([sampled_marks])), dim=-1),
            # )

    @torch.no_grad()
    def batch_ab_sample_query(
        self, 
        marks, 
        timestamps, 
        A,
        B,
        dominating_rate=None, 
        T=float('inf'), 
        left_window=0.0, 
        num_samples=1, 
        proposal_batch_size=1024, 
        adapt_dom_rate=True,
        precision_stop=None,  # If None, then this feature is disabled. If >0.0, then the sampling of a sequence will stop once the absolute difference in upper and lower bounds is smaller than this value
        return_raw=False,
        marginal_mark_n=None,
    ):
        dyn_dom_buffer = self.dyn_dom_buffer
        if num_samples > MAX_SAMPLE_BATCH_SIZE:  # Split into batches
            resulting_a_ests, resulting_b_ests, resulting_ab_ests = [], [], []
            
            remaining_samples = num_samples
            while remaining_samples > 0:
                current_batch_size = min(remaining_samples, MAX_SAMPLE_BATCH_SIZE)
                ra, rb, rab = self.batch_ab_sample_query(
                    marks=marks, 
                    timestamps=timestamps, 
                    A=A,
                    B=B,
                    dominating_rate=dominating_rate, 
                    T=T, 
                    left_window=left_window, 
                    num_samples=current_batch_size, 
                    proposal_batch_size=1024, 
                    adapt_dom_rate=adapt_dom_rate,
                    precision_stop=precision_stop,
                    return_raw=True,
                    marginal_mark_n=marginal_mark_n,
                )
                remaining_samples -= current_batch_size
                resulting_a_ests.append(ra)
                resulting_b_ests.append(rb)
                resulting_ab_ests.append(rab)

            resulting_a_ests, resulting_b_ests, resulting_ab_ests = torch.cat(resulting_a_ests), torch.cat(resulting_b_ests), torch.cat(resulting_ab_ests)
            results = {
                "est": resulting_ab_ests.mean(),
                "is_var": resulting_ab_ests.var(unbiased=False),
                "lower_bound": resulting_a_ests.mean(),
                "upper_bound": resulting_b_ests.mean(),
            }
            results["naive_var"] = results["est"] * (1-results["est"])
            results["rel_eff"] = results["naive_var"] / results["is_var"]
            return results
        
        assert(T < float('inf'))  # TODO: Implement precision based stop
        if dominating_rate is None:
            dominating_rate = self.dominating_rate
        if marks is None:
            marks = torch.LongTensor([[]], device=next(self.parameters()).device)
            timestamps = torch.FloatTensor([[]], device=next(self.parameters()).device)
        if isinstance(left_window, torch.Tensor):
            left_window = left_window.item()
        if isinstance(T, torch.Tensor):
            T = T.item()
        if marginal_mark_n is None:
            target_sample_len = -1
            stop_after_n = False
        else:
            target_sample_len = marginal_mark_n - 1  # If we are performing calculations for k_3 then we must sample 2 elements first, then integrate out from there
            stop_after_n = True

        sample_lens = torch.zeros((num_samples,), dtype=torch.int64).to(next(self.parameters()).device) #+ timestamps.numel()
        marks, timestamps = marks.expand(num_samples, *marks.shape[1:]), timestamps.expand(num_samples, *timestamps.shape[1:])
        state = self.get_states(marks, timestamps)
        state_values, state_times, empty_state_times = state["state_values"][:, [-1], :], state["state_times"][:, [-1]], state["state_times"][:, []]  # We only need the last state, and a placeholder for the times as we are always moving ahead
        batch_idx = torch.arange(num_samples).to(state_values.device)
        finer_proposal_batch_size = max(proposal_batch_size // 4, 16)

        dist = torch.distributions.Exponential(dominating_rate)
        dist.rate = dist.rate.to(state_values.device)*0+1  # We will manually apply the scale to samples
        dominating_rate = torch.ones((num_samples,1), dtype=torch.float32).to(state_values.device)*dominating_rate
        last_time = torch.ones_like(dominating_rate) * left_window if isinstance(left_window, (int, float)) else left_window
        new_times = last_time + dist.sample(sample_shape=torch.Size((num_samples, proposal_batch_size))).cumsum(dim=-1)/dominating_rate

        if adapt_dom_rate:
            dynamic_dom_rates = torch.ones((num_samples, dyn_dom_buffer,)).to(state_values.device)*dominating_rate
            k = 0

        # resulting_a_ests, resulting_b_ests, resulting_ab_ests = [], [], []
        resulting_a_ests, resulting_b_ests = [], []
        running_a_ests = torch.zeros((num_samples,), dtype=torch.float32).to(state_values.device)  # Accumulating lower bound estimates
        running_b_ests = torch.zeros_like(running_a_ests)                                          # Accumulating upper bound estimates (or rather 1 minus the upper bound)
        # running_ab_ests = torch.zeros_like(running_a_ests)                                         # Accumulating averaged estimates
        running_ab_int = torch.zeros_like(running_a_ests)                                          # Accumulating integral of lambda_{AB}^*(t)

        if stop_after_n: # We are performing marginal mark query, thus we reuse not_AB to be all of the available marks as (1) for events 1 to n-1, we will sample any mark and (2) for the n^th event we won't actually sample so this won't matter then anyways
            not_AB = torch.tensor(list(range(self.num_channels))).to(state_values.device)
        else:
            not_AB = torch.tensor([i for i in range(self.num_channels) if (i not in A) and (i not in B)]).to(state_values.device)
        not_AB_mask = torch.zeros((self.num_channels,), dtype=torch.float32).to(state_values.device)
        not_AB_mask[not_AB] = 1.0


        j = -1
        while True: 
            j += 1

            within_range_mask = new_times <= T
            to_stay = within_range_mask.any(dim=-1)
            if (precision_stop is not None) and (precision_stop > 0.0):
                to_stay = to_stay & ((running_a_ests+running_b_ests-1).abs() > precision_stop) & (running_ab_int < 10.0)
            to_go = ~to_stay 
            if to_go.any():
                resulting_a_ests.append(running_a_ests[to_go])
                resulting_b_ests.append(running_b_ests[to_go])

                running_a_ests = running_a_ests[to_stay]
                running_b_ests = running_b_ests[to_stay]
                running_ab_int = running_ab_int[to_stay]

                new_times = new_times[to_stay, ...]
                sample_lens = sample_lens[to_stay]
                timestamps = timestamps[to_stay, ...]
                marks = marks[to_stay, ...]
                state_values = state_values[to_stay, ...]
                state_times = state_times[to_stay, ...]
                empty_state_times = empty_state_times[to_stay, ...]
                batch_idx = batch_idx[:timestamps.shape[0]]
                last_time = last_time[to_stay, ...]
                dominating_rate = dominating_rate[to_stay, ...]
                if adapt_dom_rate:
                    dynamic_dom_rates = dynamic_dom_rates[to_stay, ...]
                if batch_idx.numel() == 0:
                    break  # STOP SAMPLING

            within_range_mask = new_times <= T
            max_idx_within_range = within_range_mask.sum(dim=-1) - 1

            sample_intensities = self.get_intensity(
                state_values=state_values,
                state_times=empty_state_times,
                timestamps=new_times-state_times,
                marks=None,
                mark_mask=1.0,
            )

            last_times_intensity = self.get_intensity(
                state_values=state_values,
                state_times=empty_state_times,
                timestamps=last_time-state_times,
                marks=None,
                mark_mask=1.0,
            )["all_mark_intensities"]

            redo_samples = torch.zeros_like(batch_idx, dtype=torch.bool)
            if adapt_dom_rate:  # Need to check and make sure that we don't break the sampling assumption
                redo_samples = (sample_intensities["all_mark_intensities"][..., not_AB].sum(dim=-1) > dominating_rate).any(dim=-1)
                if not redo_samples.any():
                    finer_new_times = last_time + dist.sample(sample_shape=torch.Size((last_time.shape[0], finer_proposal_batch_size))).cumsum(dim=-1)/(dominating_rate*proposal_batch_size/4)
                    finer_sample_intensities = self.get_intensity(  # Finer resolution check just after `last_time`
                        state_values=state_values,
                        state_times=empty_state_times,
                        timestamps=finer_new_times-state_times,
                        marks=None,
                        mark_mask=1.0,
                    )
                    redo_samples = redo_samples | (finer_sample_intensities["all_mark_intensities"][..., not_AB].sum(dim=-1) > dominating_rate).any(dim=-1)
                
            keep_samples = ~redo_samples

            acceptances = torch.rand_like(new_times) <= (sample_intensities["all_mark_intensities"][..., not_AB].sum(dim=-1) / dominating_rate)  #dominating_rate)
            acceptances = acceptances & within_range_mask & keep_samples.unsqueeze(-1) # Don't accept any sampled events outside the window or that need to be redone
            not_reached_sample_len = (sample_lens < target_sample_len)
            skip_integration = redo_samples | not_reached_sample_len
            if stop_after_n:
                acceptances = acceptances & not_reached_sample_len.unsqueeze(-1)
            
            samples_w_new_events = acceptances.any(dim=-1)
            if samples_w_new_events.any():
                event_idx = torch.where(
                    samples_w_new_events,
                    acceptances.int().argmax(dim=-1),
                    max_idx_within_range,  # if no accepted events in a sequence, then take the last time that is still within the window
                )
                new_time = new_times[batch_idx, event_idx].unsqueeze(-1)

                mark_probs = sample_intensities["all_mark_intensities"][batch_idx, event_idx, :].unsqueeze(-2) * not_AB_mask.unsqueeze(0).unsqueeze(0)
                mark_probs = mark_probs / torch.sum(mark_probs, dim=-1, keepdim=True)  #F.softmax(logits, -1)
                mark_dist = torch.distributions.Categorical(mark_probs)
                new_mark = mark_dist.sample()  # (batch_size, 1)

                sample_lens[samples_w_new_events] += 1  # Guaranteed at least one event was either appended or overwritten
                
                new_state = self.get_states(new_mark, new_time-state_times, old_states=state_values[:, -1, :])
                new_state_values = new_state["state_values"][:, [-1], :]
                state_times[samples_w_new_events, ...] = new_time[samples_w_new_events, ...]
                state_values[samples_w_new_events, ...] = new_state_values[samples_w_new_events, ...]

                all_intensities = torch.cat((last_times_intensity, sample_intensities["all_mark_intensities"]), dim=1)
                all_times = torch.cat((last_time, new_times), dim=1)
                a_intensity = all_intensities[..., A].sum(dim=-1)
                b_intensity = all_intensities[..., B].sum(dim=-1)
                ab_intensity = a_intensity + b_intensity
                ab_int = running_ab_int.unsqueeze(-1) + F.pad(torch.cumulative_trapezoid(ab_intensity, x=all_times, dim=-1), (1,0), 'constant', 0.0)

                all_possible_a_ests = F.pad(torch.cumulative_trapezoid(a_intensity*torch.exp(-ab_int), x=all_times, dim=-1), (1,0), 'constant', 0.0)
                all_possible_b_ests = F.pad(torch.cumulative_trapezoid(b_intensity*torch.exp(-ab_int), x=all_times, dim=-1), (1,0), 'constant', 0.0)


                running_a_ests += torch.where(skip_integration, 0.0, all_possible_a_ests[batch_idx, event_idx])  # int_0^a += int_a^b ==> int_0^b
                running_b_ests += torch.where(skip_integration, 0.0, all_possible_b_ests[batch_idx, event_idx])
                running_ab_int = torch.where(skip_integration, running_ab_int, ab_int[batch_idx, event_idx])

                last_time = torch.where(
                    redo_samples.unsqueeze(-1),
                    last_time,   
                    torch.where(samples_w_new_events.unsqueeze(-1), new_time, torch.max(new_times, dim=-1, keepdim=True).values),  #new_time*samples_w_new_events.unsqueeze(-1) + (torch.max(new_times, dim=-1).values*(~samples_w_new_events)).unsqueeze(-1)
                )
            else:

                all_intensities = torch.cat((last_times_intensity, sample_intensities["all_mark_intensities"]), dim=1)
                all_times = torch.cat((last_time, new_times), dim=1)
                a_intensity = all_intensities[..., A].sum(dim=-1)
                b_intensity = all_intensities[..., B].sum(dim=-1)
                ab_intensity = a_intensity + b_intensity
                ab_int = running_ab_int.unsqueeze(-1) + F.pad(torch.cumulative_trapezoid(ab_intensity, x=all_times, dim=-1), (1,0), 'constant', 0.0)

                event_idx = max_idx_within_range
                all_possible_a_ests = F.pad(torch.cumulative_trapezoid(a_intensity*torch.exp(-ab_int), x=all_times, dim=-1), (1,0), 'constant', 0.0)
                all_possible_b_ests = F.pad(torch.cumulative_trapezoid(b_intensity*torch.exp(-ab_int), x=all_times, dim=-1), (1,0), 'constant', 0.0)

                running_a_ests += torch.where(skip_integration, 0.0, all_possible_a_ests[batch_idx, event_idx])  # int_0^a += int_a^b ==> int_0^b
                running_b_ests += torch.where(skip_integration, 0.0, all_possible_b_ests[batch_idx, event_idx])
                running_ab_int = torch.where(skip_integration, running_ab_int, ab_int[batch_idx, event_idx])

                last_time = torch.where(
                    redo_samples.unsqueeze(-1),
                    last_time,   
                    torch.max(new_times, dim=-1, keepdim=True).values,
                )

            if adapt_dom_rate:  
                dynamic_dom_rates[:, k] = sample_intensities["total_intensity"].max(dim=1).values*100
                k = (k+1) % dynamic_dom_rates.shape[1]
                dominating_rate = torch.max(dynamic_dom_rates, dim=1, keepdim=True).values

            # print(last_time.shape, new_times.shape, proposal_batch_size, dominating_rate.shape)
            new_times = last_time + dist.sample(sample_shape=(new_times.shape[0], proposal_batch_size)).cumsum(dim=-1)/dominating_rate

        assert(running_a_ests.shape[0] == 0)
        # resulting_a_ests, resulting_b_ests, resulting_ab_ests = torch.cat(resulting_a_ests), torch.cat(resulting_b_ests), torch.cat(resulting_ab_ests)
        resulting_a_ests, resulting_b_ests = torch.cat(resulting_a_ests), torch.cat(resulting_b_ests)
        resulting_b_ests = 1 - resulting_b_ests
        resulting_ab_ests = (resulting_a_ests+resulting_b_ests)/2#0.5 + 0.5*resulting_ab_ests

        if return_raw:
            return resulting_a_ests, resulting_b_ests, resulting_ab_ests

        results = {
            "est": resulting_ab_ests.mean(),
            "is_var": resulting_ab_ests.var(unbiased=False),
            "lower_bound": resulting_a_ests.mean(),
            "upper_bound": resulting_b_ests.mean(),
        }
        results["naive_var"] = results["est"] * (1-results["est"])
        results["rel_eff"] = results["naive_var"] / results["is_var"]

        return results 

    def get_param_groups(self):
        """Returns iterable of dictionaries specifying parameter groups.
        The first dictionary in the return value contains parameters that will be subject to weight decay.
        The second dictionary in the return value contains parameters that will not be subject to weight decay.
        
        Returns:
            (param_group, param_groups) -- Tuple containing sets of parameters, one of which has weight decay enabled, one of which has it disabled.
        """

        weight_decay_params = {'params': []}
        no_weight_decay_params = {'params': [], 'weight_decay': 0.0}
        for module_ in self.modules():
            # Doesn't make sense to decay weights for a LayerNorm, BatchNorm, etc.
            if isinstance(module_, NORMS):
                no_weight_decay_params['params'].extend([
                    p for p in module_._parameters.values() if p is not None
                ])
            else:
                # Also doesn't make sense to decay biases.
                weight_decay_params['params'].extend([
                    p for n, p in module_._parameters.items() if p is not None and n != 'bias'
                ])
                no_weight_decay_params['params'].extend([
                    p for n, p in module_._parameters.items() if p is not None and n == 'bias'
                ])

        return weight_decay_params, no_weight_decay_params

    def compensator(self, a, b, conditional_times, conditional_marks, conditional_states=None, num_int_pts=100, calculate_bounds=False):
        scalar_bounds = (isinstance(a, (float, int)) and isinstance(b, (float, int))) or ((len(a.shape) == 0) and (len(b.shape) == 0))
        if scalar_bounds:
            assert(a <= b)
        else:
            assert((a <= b).all())

        results = {}
        if conditional_states is None:
            state_dict = self.get_states(conditional_marks, conditional_times)
            conditional_states = state_dict["state_values"]
        if scalar_bounds:
            ts = torch.linspace(a, b, num_int_pts).to(next(self.parameters()).device)
            vals = self.get_intensity(
                state_values=conditional_states, #state_dict["state_values"], 
                state_times=conditional_times, #state_dict["state_times"], 
                timestamps=ts.expand(*conditional_times.shape[:-1], -1), 
                marks=None,
            )["all_log_mark_intensities"].exp()
        else:
            if len(a.shape) == 0:
                a = a.unsqueeze(0).expand(conditional_times.shape[0])
            if len(b.shape) == 0:
                b = b.unsqueeze(0).expand(conditional_times.shape[0])

            ts = torch.linspace(0, 1, num_int_pts).unsqueeze(0).to(next(self.parameters()).device)
            ts = a.unsqueeze(-1) + ts*(b-a).unsqueeze(-1)

            vals = self.get_intensity(
                state_values=conditional_states, #state_dict["state_values"], 
                state_times=conditional_times, #state_dict["state_times"], 
                timestamps=ts,#.expand(*conditional_times.shape[:-1], -1), 
                marks=None,
            )["all_log_mark_intensities"].exp()
            ts = ts.unsqueeze(-1)
            #raise NotImplementedError

        if calculate_bounds:
            delta = (b-a)/(num_int_pts-1)
            if not scalar_bounds:
                delta = delta.unsqueeze(-1)
            left_pts, right_pts = vals[..., :-1, :], vals[..., 1:, :]
            upper_lower_pts = torch.stack((left_pts, right_pts), dim=-1)
            upper_vals, lower_vals = upper_lower_pts.max(dim=-1).values, upper_lower_pts.min(dim=-1).values
            results["upper_bound"] = upper_vals.sum(dim=-2) * delta
            results["lower_bound"] = lower_vals.sum(dim=-2) * delta
            results["integral"] = upper_lower_pts.mean(dim=-1).sum(dim=-2) * delta
        else:
        #    vals[..., 0, :], vals[..., -1, :] = 0.5*vals[..., 0, :], 0.5*vals[..., -1, :]
        #    results["integral"] = torch.trapezoid(vals, dx=delta, dim=-2)#delta * vals.sum(dim=-2)
            results["integral"] = torch.trapezoid(vals, x=ts, dim=-2)#delta * vals.sum(dim=-2)
        # if calculate_bounds:
        #     assert(results["lower_bound"].sum().item() <= results["integral"].sum().item() <= results["upper_bound"].sum().item())

        return results

    def compensator_grid(self, a, b, conditional_times, conditional_marks, num_int_pts, conditional_states=None, *args, **kwargs):
        if conditional_times is None:
            conditional_times, conditional_marks = torch.FloatTensor([[]]).to(next(self.parameters()).device), torch.LongTensor([[]]).to(next(self.parameters()).device)

        if conditional_states is None:
            state_dict = self.get_states(conditional_marks, conditional_times)
            conditional_states = state_dict["state_values"]
        num_int_pts += 1  # increment as we use n+1 samples to generate n integral results
        comps = torch.zeros((*conditional_times.shape[:-1], num_int_pts, self.num_channels), dtype=torch.float32).to(conditional_times.device)
        ts = torch.linspace(a, b, num_int_pts).expand(*conditional_times.shape[:-1], -1).to(conditional_times.device)
        intensities = self.get_intensity(
            state_values=conditional_states, #state_dict["state_values"],
            state_times=conditional_times, #state_dict["state_times"],
            timestamps=ts, 
            marks=None,
        )['all_log_mark_intensities'].exp()
        width = (b - a) / (num_int_pts - 1)
        height = (intensities[..., :-1, :] + intensities[..., 1:, :]) / 2
        comps[..., 1:, :] = height * width
        comps = comps.cumsum(dim=-2)

        return comps