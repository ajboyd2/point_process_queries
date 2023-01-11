import torch
import torch.nn.functional as F
import math

from tqdm import tqdm
from abc import abstractmethod

from pp_query.modules.utils import flatten, find_closest


class CensoredTimeline:
    def __init__(self, boundaries, censored_marks, total_marks, relative_start=False, device=torch.device('cpu')):
        if isinstance(boundaries, tuple):
            boundaries, censored_marks = [boundaries], [censored_marks]
        
        assert(isinstance(boundaries, list))
        for i, b in enumerate(boundaries):
            assert(len(b) == 2)
            for t in b:
                assert(isinstance(t, float))

        if len(boundaries) > 1:
            for b1, b2 in zip(boundaries[:-1], boundaries[1:]):
                assert(b1[1] <= b2[0])  # ensure no overlapping boundaries

        assert(isinstance(censored_marks, list) and len(boundaries) == len(censored_marks))
        for cm in censored_marks:
            for m in cm:
                assert(isinstance(m, int))
                assert(0 <= m < total_marks)

        self.boundaries = boundaries
        self.censored_marks = censored_marks
        self.total_marks = total_marks
        self.relative_start = relative_start

        # Create tensor of boundary timepoints and masks for censoring
        all_bounds, all_censored = [], [[]]  # Mark masks have a default empty mask for times prior to censoring
        for i in range(len(boundaries)):
            cur_cm = censored_marks[i]
            cur_begin, _ = boundaries[i]
            if i == 0:
                if cur_begin > 0.0:
                    all_bounds.append(0.0)
                    all_censored.append([])
                else:  # First boundary starts at 0.0
                    all_censored = [cur_cm]
                    continue
            else:
                _, last_end = boundaries[i-1]
                if last_end < cur_begin:
                    all_bounds.append(last_end)
                    all_censored.append([])

            all_bounds.append(cur_begin)
            all_censored.append(cur_cm)

        all_bounds.append(boundaries[-1][1])
        all_censored.append([])  # uncensored after last boundary - could be unreachable if last boundary ends with float('inf')
                
        for i in range(len(all_censored)):
            # Converts list of mark indices into list of Boolean values: ex. [1, 2, 4] w/ total marks=6 ==>  [False, True, True, False, True, False]
            all_censored[i] = [j in all_censored[i] for j in range(total_marks)]  

        self.bounds_tensor = torch.tensor(all_bounds, dtype=torch.float, device=device)
        self.censored_mask = torch.tensor(all_censored, dtype=torch.bool, device=device).float()
        self.observed_mask = 1-self.censored_mask
        self.start_time = boundaries[0][0]

        self.overwrite_with_observed, self.overwrite_with_censored = False, False

    def get_mask(self, times, time_offset=None):
        bounds_tensor = self.bounds_tensor
        if times.dim() > 1:
            for _ in range(times.dim()-1):
                bounds_tensor = bounds_tensor.unsqueeze(0)
            bounds_tensor = bounds_tensor.expand(*times.shape[:-1], -1)  # Match all but last dimension of the `times` tensor
        if time_offset is not None:
            assert(isinstance(time_offset, float))
            mask_indices = find_closest(sample_times=times, true_times=bounds_tensor+time_offset, effective_zero=time_offset)["closest_indices"]
        else:
            mask_indices = find_closest(sample_times=times, true_times=bounds_tensor)["closest_indices"]
        censored_mask = F.embedding(mask_indices, self.censored_mask)
        observed_mask = F.embedding(mask_indices, self.observed_mask)
        return {
            "censored_mask": censored_mask,
            "observed_mask": observed_mask,
        }

    def filter_sequences(self, times, marks, time_offset=None):
        masks = self.get_mask(times, time_offset=time_offset)
        to_keep = torch.gather(masks["observed_mask"], dim=-1, index=marks.unsqueeze(-1)).squeeze(-1) == 1.0
        observed_times, observed_marks = times[to_keep], marks[to_keep]
        if observed_times.dim() < times.dim():
            observed_times = observed_times.view(*((1,)*(times.dim()-1)), -1)  # Pad beginning dimensions
        if observed_marks.dim() < marks.dim():
            observed_marks = observed_marks.view(*((1,)*(marks.dim()-1)), -1) 
        return observed_times, observed_marks

    def __repr__(self):
        return str(self)

    def __str__(self):
        return "{}".format({
            "boundaries": self.bounds_tensor,
            "censored_mask": self.censored_mask,
        })

class CensoredPP:
    def __init__(
        self,
        base_process, 
        censoring,
        num_sampled_sequences, 
        use_same_seqs_for_ratio=False, 
        proposal_batch_size=1024,
        num_integral_pts=1024,
    ):
        self.base_process = base_process
        self.censoring = censoring
        self.num_sampled_sequences = num_sampled_sequences
        self.use_same_seqs_for_ratio = use_same_seqs_for_ratio
        assert(self.use_same_seqs_for_ratio)  # TODO: Support different sequences for numerator and denominator in estimates
        self.device = next(base_process.parameters()).device
        self.proposal_batch_size = proposal_batch_size
        self.num_integral_pts = num_integral_pts
        self.dominating_rate = 100.

    @torch.no_grad()
    def _generate_supporting_seqs(self, existing_times, existing_marks, max_sample_time):
        self.censoring.overwrite_with_censored = True
        sampled_times, sampled_marks, sampled_states = self.base_process.batch_sample_points(
            T=max_sample_time, 
            left_window=self.censoring.start_time, 
            timestamps=existing_times.unsqueeze(0), 
            marks=existing_marks.unsqueeze(0),
            dominating_rate=self.dominating_rate,
            # mark_mask=mark_mask,
            censoring=self.censoring,
            num_samples=self.num_sampled_sequences,
            proposal_batch_size=self.proposal_batch_size,
        )
        self.censoring.overwrite_with_censored = False
        return sampled_times, sampled_marks, sampled_states

    @torch.no_grad()
    def intensity(self, existing_times, existing_marks, eval_times, integrate_results=False, stable_mean=True):
        assert(eval_times.dim() == 1 and existing_times.dim() == 1 and existing_marks.dim() == 1)
        censoring_start, censoring_end = self.censoring.start_time, eval_times.max()
        print("sampling")
        sampled_times, sampled_marks, sampled_states = self._generate_supporting_seqs(existing_times, existing_marks, censoring_end)
        print('done sampling\n')
        grid_times = torch.linspace(
            0.0 if integrate_results else censoring_start, 
            censoring_end, 
            self.num_integral_pts, 
            device=self.device,
        ).view(*((1,)*(sampled_times[0].dim()-1)), -1)  # pad beginning dimensions

        all_times = torch.cat((eval_times.unsqueeze(0), grid_times), dim=-1)
        all_sorted_times, all_sorted_indices = all_times.sort(dim=-1)
        if stable_mean:
            items, weights = [], []
        else:
            numer_total, denom_total = 0.0, 0.0

        for st, sm, ss in zip(sampled_times, sampled_marks, sampled_states):
            padded_all_times = all_sorted_times.expand(*(st.shape[:-1]), -1)  # Match batch dimensions
            intensity_dict = self.base_process.get_intensity(ss, st, padded_all_times, censoring=self.censoring)
            observed_intensity = intensity_dict["observed_int"]

            observed_compensator = torch.cumulative_trapezoid(y=observed_intensity.sum(dim=-1), x=padded_all_times, dim=-1)
            observed_compensator = F.pad(observed_compensator, (1, 0), 'constant', 0.0).unsqueeze(-1)  # Ensure we have a beginning element for \int_0^0 ... dt

            if stable_mean:
                items.append(observed_intensity)
                weights.append(observed_compensator)
            else:
                observed_compensator = torch.exp(-observed_compensator)
                numer_total += (observed_intensity * observed_compensator).sum(dim=-3)  # (..., time_dim, mark_dim)
                denom_total += observed_compensator.sum(dim=-3)

        if stable_mean:
            items = torch.cat(items, dim=0)
            weights = torch.cat(weights, dim=0)
            norm_weights = torch.softmax(-weights, dim=0)
            all_sorted_estimates = (items * norm_weights).sum(dim=0)
        else:
            all_sorted_estimates = numer_total / denom_total

        print("all_sorted", all_sorted_estimates.shape, all_sorted_estimates.isnan().any())
        print(all_sorted_estimates[:5, 2].tolist(), all_sorted_estimates[-5:, 2].tolist())
        all_estimates = all_sorted_estimates.gather(-2, all_sorted_indices.squeeze(0).unsqueeze(-1).expand(*all_sorted_estimates.shape).argsort(-2))
        print("all_estimates", all_estimates.shape, all_estimates.isnan().any())
        eval_estimates = all_estimates[:len(eval_times), :]
        print("eval_estimates", eval_estimates.shape, eval_estimates.isnan().any())
        results = {"intensity": eval_estimates}

        if integrate_results:  # Instead of returning the \lambda(t), return \int_0^t \lambda(s) ds
            all_sorted_int_estimates = torch.cumulative_trapezoid(y=all_sorted_estimates, x=all_sorted_times.squeeze(0), dim=-2)  # -2 due to integrating each mark-specific intensity individually
            all_sorted_int_estimates = F.pad(all_sorted_int_estimates, (0, 0, 1, 0), 'constant', 0.0)
            all_int_estimates = all_sorted_int_estimates.gather(-2, all_sorted_indices.squeeze(0).unsqueeze(-1).expand(*all_sorted_int_estimates.shape).argsort(-2))
            eval_int_estimates = all_int_estimates[:len(eval_times), :]
            results["compensator"] = eval_int_estimates

        for k,v in results.items():
            print(k, v.shape, v.isnan().any())

        return results

    @torch.no_grad()
    def log_likelihood(self, existing_times, existing_marks, T):
        if isinstance(T, float):
            padded_times = F.pad(existing_times, (0, 1), 'constant', T)
        elif isinstance(T, torch.Tensor):
            assert(T.numel() == 1)
            padded_times = torch.cat((existing_times, T), dim=-1)
        else:
            raise AssertionError

        intensity_res = self.intensity(existing_times, existing_marks, padded_times, integrate_results=True)
        pos_ll = intensity_res["intensity"][:-1, :].log().gather(dim=-1, index=existing_marks.unsqueeze(-1)).squeeze(-1).sum(-1)
        neg_ll = intensity_res["compensator"][-1, :].sum(-1)
        return {
            "log_likelihood": pos_ll - neg_ll,
            "positive_contribution": pos_ll,
            "negative_contribution": neg_ll,
        }

    @torch.no_grad()
    def next_event_prediction(self, existing_times, existing_marks, num_samples=1024):
        pass

    @torch.no_grad()
    def summary_statistics(self, existing_times, existing_marks, T):
        pass

    @torch.no_grad()
    def goodness_fit_transform(self, existing_times, existing_marks, T):
        pass

    @torch.no_grad()
    def kl_div(self, other_model):
        pass

    @torch.no_grad()
    def ll_pred_experiment_pass(self, uncensored_times, uncensored_marks, T):
        uncensored_times, uncensored_marks = uncensored_times[..., uncensored_times.squeeze() <= T], uncensored_marks[..., uncensored_times.squeeze() <= T]
        observed_times, observed_marks = self.censoring.filter_sequences(times=uncensored_times, marks=uncensored_marks)
        integral_pts = torch.linspace(0, T, self.num_integral_pts, device=self.device).unsqueeze(0)
        results = {}
        
        # Likelihood of the original sequence under the base model, no changes made to either
        original_pos_ll = self.base_process.get_intensity(
            state_values=None, 
            state_times=uncensored_times, 
            timestamps=uncensored_times, 
            marks=uncensored_marks, 
            state_marks=uncensored_marks, 
            censoring=None,
        )["log_mark_intensity"].sum()
        original_neg_ll = torch.trapezoid(
            y=self.base_process.get_intensity(
                state_values=None, 
                state_times=uncensored_times, 
                timestamps=integral_pts, 
                marks=None, 
                state_marks=uncensored_marks, 
                censoring=None,
            )["total_intensity"], 
            x=integral_pts, 
            dim=-1,
        )
        results["original"] = {
            "log_likelihood": (original_pos_ll - original_neg_ll).squeeze(),
            "positive_contribution": original_pos_ll.squeeze(),
            "negative_contribution": original_neg_ll.squeeze(),
        }
        
        naive_pos_ll = self.base_process.get_intensity(
            state_values=None, 
            state_times=observed_times, 
            timestamps=observed_times, 
            marks=observed_marks, 
            state_marks=observed_marks, 
            censoring=None,
        )["log_mark_intensity"].sum()
        naive_neg_ll = torch.trapezoid(
            y=self.base_process.get_intensity(
                state_values=None, 
                state_times=observed_times, 
                timestamps=integral_pts, 
                marks=None, 
                state_marks=observed_marks, 
                censoring=None,
            )["total_intensity"],
            x=integral_pts,
            dim=-1,
        )
        results["naive"] = {
            "log_likelihood": (naive_pos_ll - naive_neg_ll).squeeze(),
            "positive_contribution": naive_pos_ll.squeeze(),
            "negative_contribution": naive_neg_ll.squeeze(),
        }

        self.censoring.overwrite_with_observed = True
        baseline_pos_ll = naive_pos_ll
        baseline_neg_ll = torch.trapezoid(
            y=self.base_process.get_intensity(
                state_values=None, 
                state_times=observed_times, 
                timestamps=integral_pts, 
                marks=None, 
                state_marks=observed_marks, 
                censoring=self.censoring,
            )["total_intensity"],
            x=integral_pts,
            dim=-1,
        )
        results["baseline"] = {
            "log_likelihood": (baseline_pos_ll - baseline_neg_ll).squeeze(),
            "positive_contribution": baseline_pos_ll.squeeze(),
            "negative_contribution": baseline_neg_ll.squeeze(),
        }
        self.censoring.overwrite_with_observed = False

        results["censored"] = self.log_likelihood(
            existing_times=observed_times.squeeze(0), 
            existing_marks=observed_marks.squeeze(0), 
            T=T,
        )

        return results

'''
class CensoredPP_orig:
    def __init__(
        self, 
        base_process, 
        observed_marks, 
        num_sampled_sequences, 
        use_same_seqs_for_ratio=False, 
        batch_size=256, 
        device=torch.device("cpu"), 
        use_tqdm=True, 
        proposal_batch_size=1024,
    ):
        assert(len(observed_marks) > 0)  # K=6, observed_marks=[0,1,2] ==> censored_marks=[3,4,5]
        assert(num_sampled_sequences > 0)

        self.base_process = base_process
        self.marks = base_process.num_channels
        self.observed_marks = torch.LongTensor(observed_marks)
        assert(len(self.observed_marks.shape) == 1)
        self.censored_marks = torch.LongTensor([i for i in range(self.marks) if i not in observed_marks])

        self.num_sampled_sequences = num_sampled_sequences
        self.use_same_seqs_for_ratio = use_same_seqs_for_ratio
        if not use_same_seqs_for_ratio:
            assert(num_sampled_sequences % 2 == 0) # we will use half of sequences for numerator and half for denominator

        self.dominating_rate = base_process.dominating_rate
        self.cond_seqs = None
        self.cond_left_window = None
        self.cond_right_window = None
        self.cond_fixed_times = None
        self.cond_fixed_marks = None

        self.batch_size = batch_size
        self.device = device
        self.use_tqdm = use_tqdm
        self.proposal_batch_size = proposal_batch_size

    @torch.no_grad()
    def gen_cond_seqs(
        self,
        right_window,  # Max time to go out until
        left_window,   # Where to start sampling
        fixed_times,   # Either times of events prior to sampling, or that are guaranteed to occur in the middle of sampling
        fixed_marks,   # Corresponding marks of `fixed_times`
    ):
        if fixed_times is None:
            fixed_times = torch.FloatTensor([[]]).to(self.device)
            fixed_marks = torch.LongTensor([[]]).to(self.device)

        if self.cond_seqs is not None:
            if (self.cond_fixed_times == fixed_times).all() and (self.cond_fixed_marks == fixed_marks).all() and (self.cond_left_window == left_window):
                if self.cond_right_window >= right_window:
                    return self.cond_seqs
                else:
                    return self.extend_cond_seqs(right_window)
        self.cond_left_window, self.cond_right_window = left_window, right_window
        self.cond_fixed_times, self.cond_fixed_marks = fixed_times, fixed_marks

        n = self.num_sampled_sequences
        # cond_times, cond_marks = [], []
        mark_mask = torch.ones((self.marks,), dtype=torch.float32).to(self.device)
        mark_mask[self.observed_marks] = 0.0  # Only sample the censored marks (to marginalize them out)
        numer_cond_times, numer_cond_marks, numer_states = self.base_process.batch_sample_points(
            T=right_window, 
            left_window=left_window, 
            timestamps=fixed_times, 
            marks=fixed_marks,
            dominating_rate=self.dominating_rate,
            mark_mask=mark_mask,
            num_samples=n // (1 if self.use_same_seqs_for_ratio else 2),
            proposal_batch_size=self.proposal_batch_size,
        )
        numer_cond_times, numer_cond_marks, _ = batch_samples(numer_cond_times, numer_cond_marks, numer_states, self.batch_size)

        if self.use_same_seqs_for_ratio:
            self.cond_seqs = {
                "numer_cond_times": numer_cond_times,
                "numer_cond_marks": numer_cond_marks,
                "denom_cond_times": numer_cond_times,
                "denom_cond_marks": numer_cond_marks,
            }
        else:
            denom_cond_times, denom_cond_marks, denom_states = self.base_process.batch_sample_points(
                T=right_window, 
                left_window=left_window, 
                timestamps=fixed_times, 
                marks=fixed_marks,
                dominating_rate=self.dominating_rate,
                mark_mask=mark_mask,
                num_samples=n//2,
                proposal_batch_size=self.proposal_batch_size,
            )

            denom_cond_times, denom_cond_marks, _ = batch_samples(denom_cond_times, denom_cond_marks, denom_states, self.batch_size)

            self.cond_seqs = {
                "numer_cond_times": numer_cond_times,
                "numer_cond_marks": numer_cond_marks,
                "denom_cond_times": denom_cond_times,
                "denom_cond_marks": denom_cond_marks,
            }

        return self.cond_seqs

    @torch.no_grad()
    def extend_cond_seqs(self, right_window):
        print("EXTENDING")
        n = self.num_sampled_sequences
        cond_times, cond_marks = self.cond_seqs["numer_cond_times"], self.cond_seqs["numer_cond_marks"]
        if not self.use_same_seqs_for_ratio:
            cond_times += self.cond_seqs["denom_cond_times"]
            cond_marks += self.cond_seqs["denom_cond_marks"]
        assert(n == len(cond_times))

        mark_mask = torch.ones((self.marks,), dtype=torch.float32).to(self.device)
        mark_mask[self.observed_marks] = 0.0  # Only sample the censored marks (to marginalize them out)
        for i in range(n):
            fixed_times, fixed_marks = cond_times[i], cond_marks[i]
            times, marks = self.base_process.sample_points(
                T=right_window, 
                left_window=self.right_window, 
                timestamps=fixed_times, 
                marks=fixed_marks,
                dominating_rate=self.dominating_rate,
                mark_mask=mark_mask,
            )
            cond_times[i], cond_marks[i] = times, marks

        if self.use_same_seqs_for_ratio:
            self.cond_seqs = {
                "numer_cond_times": cond_times,
                "numer_cond_marks": cond_marks,
                "denom_cond_times": cond_times,
                "denom_cond_marks": cond_marks,
            }
        else:
            self.cond_seqs = {
                "numer_cond_times": cond_times[:n//2],
                "numer_cond_marks": cond_marks[:n//2],
                "denom_cond_times": cond_times[n//2:],
                "denom_cond_marks": cond_marks[n//2:],
            }

        self.cond_right_window = right_window
        return self.cond_seqs

    @torch.no_grad()
    def intensity(
        self, 
        t, 
        conditional_times, 
        conditional_marks, 
        cond_seqs=None,
        censoring_start_time=0.0,
    ):
        if not isinstance(t, (int, float)):
            return torch.FloatTensor([self.intensity(_t, conditional_times, conditional_marks, cond_seqs, censoring_start_time) for _t in t])

        if conditional_times is None:
            conditional_times, conditional_marks = torch.FloatTensor([]), torch.LongTensor([])

        if t <= censoring_start_time:
            return self.base_process.intensity(t, conditional_times, conditional_marks)

        if cond_seqs is None:  # sample conditional sequences for expected values
            cond_seqs = self.gen_cond_seqs(
                right_window=t, 
                left_window=censoring_start_time, 
                fixed_times=conditional_times, 
                fixed_marks=conditional_marks,
            )  # sampled_times and sampled_marks will be included in the resulting sequences

        numer, denom = 0.0, 0.0
        pp = self.base_process
        num_seqs = self.num_sampled_sequences // (1 if self.use_same_seqs_for_ratio else 2)
        obs, cen = self.observed_marks, self.censored_marks
        for i in range(num_seqs):
            numer_times, numer_marks = cond_seqs["numer_cond_times"][i], cond_seqs["numer_cond_marks"][i]
            n = pp.intensity(t, numer_times, numer_marks) * torch.exp(-pp.compensator(censoring_start_time, t, numer_times, numer_marks)["integral"][obs].sum())

            denom_times, denom_marks = cond_seqs["denom_cond_times"][i], cond_seqs["denom_cond_marks"][i]
            d = torch.exp(-pp.compensator(censoring_start_time, t, denom_times, denom_marks)["integral"][obs].sum())

            numer += n / num_seqs
            denom += d / num_seqs

        numer[cen] *= 0.0  # zero out censored intensities as they are ill-defined here

        return numer / denom

    @torch.no_grad()
    def compensator(
        self, 
        a, 
        b, 
        conditional_times, 
        conditional_marks, 
        num_samples=1000, 
        cond_seqs=None, 
        censoring_start_time=0.0, 
    ):
        if cond_seqs is None:  # sample conditional sequences for expected values
            cond_seqs = self.gen_cond_seqs(
                right_window=b, 
                left_window=censoring_start_time, 
                fixed_times=conditional_times, 
                fixed_marks=conditional_marks,
            )  # sampled_times and sampled_marks will be included in the resulting sequences

        numer, denom = 0.0, 0.0

        ts = torch.linspace(a, b, num_samples).to(self.device)
        num_seqs = self.num_sampled_sequences // 2
        mark_mask = torch.ones((self.marks,), dtype=torch.float32).to(self.device)
        if len(self.censored_marks) > 0:
            mark_mask[self.censored_marks] = 0.0

        for numer_times, numer_marks in tqdm(zip(cond_seqs["numer_cond_times"], cond_seqs["numer_cond_marks"]), disable=not self.use_tqdm):
            intensity = self.base_process.get_intensity(
                state_values=None, 
                state_times=numer_times, 
                timestamps=ts.unsqueeze(0).expand(numer_times.shape[0], -1), 
                marks=None, 
                state_marks=numer_marks, 
                mark_mask=1.0,
            )["all_log_mark_intensities"].exp()*mark_mask.unsqueeze(0)
            numer_comp = self.base_process.compensator_grid(a, b, numer_times, numer_marks, num_samples-1) * mark_mask.unsqueeze(0) #[np.newaxis, :]
            if censoring_start_time < a:
                nc = self.base_process.compensator(censoring_start_time, a, numer_times, numer_marks)["integral"] * mark_mask
                numer_comp += nc
            numer += torch.sum(intensity * torch.exp(-torch.sum(numer_comp, dim=-1, keepdim=True)), dim=0, keepdim=True) / num_seqs #(num_seqs*numer_comp.shape[0])
            if self.use_same_seqs_for_ratio:
                denom += torch.sum(torch.exp(-torch.sum(numer_comp, dim=-1, keepdim=True)), dim=0, keepdim=True) / num_seqs #(num_seqs*numer_comp.shape[0])

        if not self.use_same_seqs_for_ratio:
            for denom_times, denom_marks in tqdm(zip(cond_seqs["denom_cond_times"], cond_seqs["denom_cond_marks"]), disable=not self.use_tqdm):
                denom_comp = self.base_process.compensator_grid(a, b, denom_times, denom_marks, num_samples-1) * mark_mask.unsqueeze(0) #[np.newaxis, :]
                if censoring_start_time < a:
                    denom_comp += self.base_process.compensator(censoring_start_time, a, denom_times, denom_marks)["integral"] * mark_mask
                denom += torch.sum(torch.exp(-torch.sum(denom_comp, dim=-1, keepdim=True)), dim=0, keepdim=True) / num_seqs #(num_seqs*denom_comp.shape[0])

        # for i in tqdm(range(num_seqs), disable=not self.use_tqdm):
        #     numer_times, numer_marks = cond_seqs["numer_cond_times"][i], cond_seqs["numer_cond_marks"][i]
        #     denom_times, denom_marks = cond_seqs["denom_cond_times"][i], cond_seqs["denom_cond_marks"][i]

        #     # intensity = np.array([self.base_process.intensity(t, numer_times, numer_marks)*mark_mask for t in ts])
        #     # intensity = self.base_process.get_intensity(ts, numer_times, numer_marks)*mark_mask.unsqueeze(0) #[np.newaxis, :]
        #     intensity = self.base_process.get_intensity(
        #         state_values=None, 
        #         state_times=numer_times, 
        #         timestamps=ts.unsqueeze(0).expand(numer_times.shape[0], -1), 
        #         marks=None, 
        #         state_marks=numer_marks, 
        #         mark_mask=1.0,
        #     )["all_log_mark_intensities"].exp()*mark_mask.unsqueeze(0)
        #     numer_comp = self.base_process.compensator_grid(a, b, numer_times, numer_marks, num_samples-1) * mark_mask.unsqueeze(0) #[np.newaxis, :]
        #     if self.use_same_seqs_for_ratio:
        #         denom_comp = numer_comp
        #     else:
        #         denom_comp = self.base_process.compensator_grid(a, b, denom_times, denom_marks, num_samples-1) * mark_mask.unsqueeze(0) #[np.newaxis, :]

        #     if censoring_start_time < a:
        #         nc = self.base_process.compensator(censoring_start_time, a, numer_times, numer_marks)["integral"] * mark_mask
        #         numer_comp += nc
        #         if self.use_same_seqs_for_ratio:
        #             denom_comp = numer_comp
        #         else:
        #             denom_comp += self.base_process.compensator(censoring_start_time, a, denom_times, denom_marks)["integral"] * mark_mask

        #     numer += torch.sum(intensity * torch.exp(-torch.sum(numer_comp, dim=-1, keepdim=True)), dim=0, keepdim=True) / (num_seqs*numer_comp.shape[0])
        #     denom += torch.sum(torch.exp(-torch.sum(denom_comp, dim=-1, keepdim=True)), dim=0, keepdim=True) / (num_seqs*denom_comp.shape[0])
        
        #numer, denom = sum(numer), sum(denom)  # each are size (num_samples, marks)
        censored_intensities = numer / denom
        # Perform trapezoidal rule to approximate \int_a^b of censored_intensity(t) dt
        return torch.trapezoid(censored_intensities, x=ts, dim=-2)

    @torch.no_grad()
    def sample(self, right_window=None, left_window=0.0, length_limit=None, sampled_times=None, sampled_marks=None, mark_mask=1.0): 
        mark_mask = torch.ones((self.marks,), dtype=torch.float32) * mark_mask
        mark_mask[self.censored_marks] = 0.0  # don't allow samples of censored marks
        return super().sample(
            marks=sampled_marks, 
            timestamps=sampled_times,
            dominating_rate=self.dominating_rate, 
            T=right_window, 
            left_window=left_window, 
            mark_mask=mark_mask,
        )
'''