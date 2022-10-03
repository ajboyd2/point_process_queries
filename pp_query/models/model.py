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
        dominating_rate=1000.,
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

    def get_states(self, marks, timestamps):
        """Get the hidden states that can be used to extract intensity values from."""
        
        states = self.decoder.get_states(
            marks=marks, 
            timestamps=timestamps, 
        )

        return {
            "state_values": states,
            "state_times": timestamps,
        }

    def get_intensity(self, state_values, state_times, timestamps, marks=None, state_marks=None, mark_mask=1.0):
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

    def compensator(self, a, b, conditional_times, conditional_marks, num_int_pts=100, calculate_bounds=False):
        assert(a < b)
        results = {}
        state_dict = self.get_states(conditional_marks, conditional_times)
        ts = torch.linspace(a, b, num_int_pts).to(next(self.parameters()).device)
        vals = self.get_intensity(
            state_values=state_dict["state_values"], 
            state_times=state_dict["state_times"], 
            timestamps=ts.expand(*conditional_times.shape[:-1], -1), 
            marks=None,
        )["all_log_mark_intensities"].exp()
        if calculate_bounds:
            delta = (b-a)/(num_int_pts-1)
            left_pts, right_pts = vals[..., :-1, :], vals[..., 1:, :]
            upper_lower_pts = torch.stack((left_pts, right_pts), dim=-1)
            upper_vals, lower_vals = upper_lower_pts.max(dim=-1).values, upper_lower_pts.min(dim=-1).values
            results["upper_bound"] = upper_vals.sum(dim=-2) * delta
            results["lower_bound"] = lower_vals.sum(dim=-2) * delta
            results["integral"] = upper_lower_pts.mean(dim=-1).sum(dim=-2) * delta
        else:
        #    vals[..., 0, :], vals[..., -1, :] = 0.5*vals[..., 0, :], 0.5*vals[..., -1, :]
        #    results["integral"] = torch.trapezoid(vals, dx=delta, dim=-2)#delta * vals.sum(dim=-2)
            results["integral"] = torch.trapezoid(vals, x=torch.linspace(a, b, num_int_pts), dim=-2)#delta * vals.sum(dim=-2)
        if calculate_bounds:
            assert(results["lower_bound"].sum().item() <= results["integral"].sum().item() <= results["upper_bound"].sum().item())

        return results

    def compensator_grid(self, a, b, conditional_times, conditional_marks, num_int_pts, *args, **kwargs):
        if conditional_times is None:
            conditional_times, conditional_marks = torch.FloatTensor([[]]).to(next(self.parameters()).device), torch.LongTensor([[]]).to(next(self.parameters()).device)

        state_dict = self.get_states(conditional_marks, conditional_times)
        num_int_pts += 1  # increment as we use n+1 samples to generate n integral results
        comps = torch.zeros((*conditional_times.shape[:-1], num_int_pts, self.num_channels), dtype=torch.float32).to(conditional_times.device)
        ts = torch.linspace(a, b, num_int_pts).expand(*conditional_times.shape[:-1], -1).to(conditional_times.device)
        intensities = self.get_intensity(
            state_values=state_dict["state_values"],
            state_times=state_dict["state_times"],
            timestamps=ts, 
            marks=None,
        )['all_log_mark_intensities'].exp()
        width = (b - a) / (num_int_pts - 1)
        height = (intensities[..., :-1, :] + intensities[..., 1:, :]) / 2
        comps[..., 1:, :] = height * width
        comps = comps.cumsum(dim=-2)

        return comps