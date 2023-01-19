import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import PPModel
from pp_query.modules.utils import xavier_truncated_normal, flatten, find_closest, ACTIVATIONS

class HawkesModel(PPModel):

    def __init__(
        self, 
        num_marks,
        bounded=False,
        int_strength=0.3,
    ):
        """Constructor for general PPModel class.
        
        Arguments:
            decoder {torch.nn.Module} -- Neural network decoder that accepts a latent state, marks, timestamps, and times of sample points. 

        Keyword Arguments:
            time_embedding {torch.nn.Module} -- Function to transform k-dimensional timestamps into (k+1)-dimensional embedded vectors. If specified, will make encoder and decoder share this function. (default: {None})
            encoder {torch.nn.Module} -- Neural network encoder that accepts marks and timestamps and returns a single latent state (default: {None})
            aggregator {torch.nn.Module} -- Module that turns a tensor of hidden states into a latent vector (with noise added during training) (default: {None})
        """
        super().__init__(decoder=None, num_channels=num_marks)

        self.num_marks = num_marks
        self.alphas = torch.nn.Embedding(
            num_embeddings=num_marks, 
            embedding_dim=num_marks,
        )
        self.deltas = torch.nn.Embedding(
            num_embeddings=num_marks,
            embedding_dim=num_marks,
        )
        self.alphas.weight.data = torch.rand_like(self.alphas.weight.data)*0.5+0.3  #torch.randn_like(self.alphas.weight.data) * 0.0001
        self.deltas.weight.data = torch.rand_like(self.deltas.weight.data)*0.4+0.8  #torch.randn_like(self.deltas.weight.data) * 0.0001
        if int_strength != 1.0:
            i = torch.eye(num_marks)
            self.alphas.weight.data = self.alphas.weight.data*i + self.alphas.weight.data*(1-i)*int_strength # lower the interaction effects on average, as by default they are too strong
        
        self.mus = torch.nn.Parameter(torch.rand(num_marks,)*0.4+0.1)     #torch.nn.Parameter(torch.randn(num_marks,) * 0.0001)
        self.s = torch.nn.Parameter(torch.rand(num_marks,)*0.0+1)       #torch.nn.Parameter(torch.randn(num_marks,) * 0.0001)
        self.bounded = bounded

    def get_states(self, tgt_marks, tgt_timestamps):
        """Get the hidden states that can be used to extract intensity values from."""

        return {
            "state_values": tgt_marks,
            "state_times": tgt_timestamps,
        }

    def get_intensity(self, state_values, state_times, timestamps, marks=None, state_marks=None, mark_mask=1.0, from_right=False, censoring=None):
        """Given a set of hidden states, timestamps, and latent_state get a tensor representing intensity values at timestamps.
        Specify marks to get intensity values for specific channels."""

        if (state_values is None) and (state_marks is not None):
            state_values = self.get_states(state_marks, state_times)["state_values"]

        batch_size, seq_len = timestamps.shape
        hist_len = state_times.shape[1]
        num_marks = self.num_marks

        mu, alpha, delta = self.mus, self.alphas(state_values), self.deltas(state_values)
        # if self.bounded:
        #     mu, alpha, delta = mu.exp(), alpha.exp(), delta.exp()
        mu = mu.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1)
        alpha = torch.transpose(alpha.unsqueeze(-1).expand(-1, -1, -1, seq_len), 1, 3).contiguous()
        delta = torch.transpose(delta.unsqueeze(-1).expand(-1, -1, -1, seq_len), 1, 3).contiguous()

        time_diffs = F.relu(timestamps.unsqueeze(2) - state_times.unsqueeze(1))
        time_diffs = time_diffs.unsqueeze(2).expand(-1, -1, num_marks, -1)
        if from_right:
            valid_terms = time_diffs >= 0
        else:
            valid_terms = time_diffs > 0

        prod = alpha * (-1 * delta * time_diffs).exp()
        prod = torch.where(valid_terms, prod, torch.zeros_like(prod))

        all_mark_intensities = mu + prod.sum(-1)

        # if not self.bounded:
        #     s = self.s.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1).exp()
        #     all_mark_intensities = s * torch.log(1 + torch.exp(all_mark_intensities / s))

        if isinstance(mark_mask, torch.FloatTensor):
            if len(mark_mask.shape) == 1:
                mark_mask = mark_mask.view(*((1,)*(len(all_mark_intensities.shape)-1)), -1)
            all_mark_intensities *= mark_mask

        all_log_mark_intensities = all_mark_intensities.log()
        total_intensity = all_mark_intensities.sum(-1)

        intensity_dict = {
            "all_log_mark_intensities": all_log_mark_intensities,
            "total_intensity": total_intensity,
            "all_mark_intensities": all_mark_intensities,
        }

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

        if marks is not None:
            intensity_dict["log_mark_intensity"] = intensity_dict["all_log_mark_intensities"].gather(dim=-1, index=marks.unsqueeze(-1)).squeeze(-1)
        
        return intensity_dict 
        
    def forward(self, tgt_marks, tgt_timestamps, sample_timestamps=None, pp_id=None):
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
            tgt_marks=tgt_marks,
            tgt_timestamps=tgt_timestamps,
        )
        return_dict["state_dict"] = intensity_state_dict

        tgt_intensities = self.get_intensity(
            state_values=intensity_state_dict["state_values"],
            state_times=intensity_state_dict["state_times"],
            timestamps=tgt_timestamps,
            marks=tgt_marks,
        )
        return_dict["tgt_intensities"] = tgt_intensities

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

    def sample_points(self, marks, timestamps, T=float('inf'), left_window=0.0, length_limit=float('inf'), mark_mask=1.0, proposal_batch_size=10):
        assert((T < float('inf')) or (length_limit < float('inf')))
        dev = next(self.parameters()).device
        if marks is None:
            marks = torch.tensor([[]], dtype=torch.long, device=dev)
            timestamps = torch.tensor([[]], torch.float, device=dev)

        state = self.forward(marks, timestamps)
        state_values, state_times = state["state_dict"]["state_values"], state["state_dict"]["state_times"]
        if isinstance(left_window, torch.Tensor):
            last_time = left_window
        else:
            last_time = torch.tensor(left_window, dtype=torch.float32, device=dev)
        last_time_placeholder = torch.tensor([[1.0]], dtype=torch.float32, device=dev)
        
        sampled_times = []
        sampled_marks = []
        while (last_time <= T).any() and (timestamps.shape[-1] < length_limit):
            dominating_rate = self.get_intensity(state_values, state_times, last_time_placeholder*last_time, mark_mask=mark_mask, from_right=True)["total_intensity"].squeeze()
            dist = torch.distributions.Exponential(dominating_rate)
            new_times = last_time + dist.sample(sample_shape=torch.Size((1, proposal_batch_size))).cumsum(dim=-1)
            if (new_times > T).all():
                break

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

                logits = sample_intensities["all_log_mark_intensities"][:, [idx], :]
                mark_probs = F.softmax(logits, -1) 
                mark_dist = torch.distributions.Categorical(mark_probs)
                new_mark = mark_dist.sample()
                timestamps = torch.cat((timestamps, new_time), -1)
                marks = torch.cat((marks, new_mark), -1)
                sampled_times.append(new_time.squeeze().item())
                sampled_marks.append(new_mark.squeeze().item())

                state = self.forward(marks, timestamps)
                state_values, state_times = state["state_dict"]["state_values"], state["state_dict"]["state_times"]
                last_time = new_times[:, idx] #.squeeze()
            else:
                last_time = new_times.max()
        
        return (timestamps, marks)
