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

    def get_intensity(self, state_values, state_times, timestamps, marks=None):
        """Given a set of hidden states, timestamps, and latent_state get a tensor representing intensity values at timestamps.
        Specify marks to get intensity values for specific channels."""

        intensity_dict = self.decoder.get_intensity(
            state_values=state_values,
            state_times=state_times,
            timestamps=timestamps,
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

        
    def sample_points(self, marks, timestamps, dominating_rate, T, left_window, top_k=0, top_p=0.0):
        state = self.forward(marks, timestamps)
        state_values, state_times, latent_state = state["state_dict"]["state_values"], state["state_dict"]["state_times"], state["latent_state"]
        
        dist = torch.distributions.Exponential(dominating_rate)
        last_time = left_window 
        new_time = last_time + dist.sample(sample_shape=torch.Size((1,1))).to(torch.cuda.current_device())
        sampled_times = []
        sampled_marks = []
        while new_time < T:
            sample_intensities = self.get_intensity(
                state_values=state_values,
                state_times=state_times,
                timestamps=new_time,
                latent_state=latent_state,
                marks=None,
            )

            if torch.rand_like(new_time) <= (sample_intensities["total_intensity"] / (dominating_rate)):
                if top_k > 0 or top_p > 0:
                    logits = top_k_top_p_filtering(sample_intensities["all_log_mark_intensities"], top_k=top_k, top_p=top_p)
                else:
                    logits = sample_intensities["all_log_mark_intensities"]
                mark_probs = F.softmax(logits, -1) #(sample_intensities["all_log_mark_intensities"] - sample_intensities["total_intensity"].unsqueeze(-1).log()).exp()
                mark_dist = torch.distributions.Categorical(mark_probs)
                new_mark = mark_dist.sample()
                timestamps = torch.cat((timestamps, new_time), -1)
                marks = torch.cat((marks, new_mark), -1)
                sampled_times.append(new_time.squeeze().item())
                sampled_marks.append(new_mark.squeeze().item())

                state = self.forward(marks, timestamps)
                state_values, state_times, latent_state = state["state_dict"]["state_values"], state["state_dict"]["state_times"], state["latent_state"]
        
            new_time = new_time + dist.sample(sample_shape=torch.Size((1,1))).to(torch.cuda.current_device())

        assumption_violation = False
        for _ in range(5):
            eval_times = torch.rand_like(timestamps).clamp(min=1e-8)*T
            sample_intensities = self.get_intensity(
                state_values=state_values,
                state_times=state_times,
                timestamps=eval_times,
                latent_state=latent_state,
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
            return sampled_times, sampled_marks

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