from bdb import effective
from turtle import left
import torch
import torch.nn.functional as F

from tqdm import tqdm
from abc import ABC, abstractmethod


class CensoredPP:
    def __init__(self, base_process, observed_marks, num_sampled_sequences, use_same_seqs_for_ratio=False, batch_size=128, device=torch.device("cpu")):
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
        cond_times, cond_marks = [], []
        mark_mask = torch.ones((self.marks,), dtype=torch.float32).to(self.device)
        mark_mask[self.observed_marks] = 0.0  # Only sample the censored marks (to marginalize them out)
        for _ in tqdm(range(n)):
            times, marks = self.base_process.sample_points(
                T=right_window, 
                left_window=left_window, 
                timestamps=fixed_times, 
                marks=fixed_marks,
                dominating_rate=self.dominating_rate,
                mark_mask=mark_mask,
            )
            cond_times.append(times); cond_marks.append(marks)

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

        return self.cond_seqs

    @torch.no_grad()
    def extend_cond_seqs(self, right_window):
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

        numer, denom = [], []

        ts = torch.linspace(a, b, num_samples)
        num_seqs = self.num_sampled_sequences // 2
        mark_mask = torch.ones((self.marks,), dtype=torch.float32)
        if len(self.censored_marks) > 0:
            mark_mask[self.censored_marks] = 0.0
        for i in tqdm(range(num_seqs)):
            numer_times, numer_marks = cond_seqs["numer_cond_times"][i], cond_seqs["numer_cond_marks"][i]
            denom_times, denom_marks = cond_seqs["denom_cond_times"][i], cond_seqs["denom_cond_marks"][i]

            # intensity = np.array([self.base_process.intensity(t, numer_times, numer_marks)*mark_mask for t in ts])
            # intensity = self.base_process.get_intensity(ts, numer_times, numer_marks)*mark_mask.unsqueeze(0) #[np.newaxis, :]
            intensity = self.base_process.get_intensity(
                state_values=None, 
                state_times=numer_times, 
                timestamps=ts.unsqueeze(0), 
                marks=None, 
                state_marks=numer_marks, 
                mark_mask=1.0,
            )["all_log_mark_intensities"].exp()*mark_mask.unsqueeze(0)
            numer_comp = self.base_process.compensator_grid(a, b, numer_times, numer_marks, num_samples-1) * mark_mask.unsqueeze(0) #[np.newaxis, :]
            denom_comp = self.base_process.compensator_grid(a, b, denom_times, denom_marks, num_samples-1) * mark_mask.unsqueeze(0) #[np.newaxis, :]

            if censoring_start_time < a:
                numer_comp += self.base_process.compensator(censoring_start_time, a, numer_times, numer_marks)["integral"] * mark_mask
                denom_comp += self.base_process.compensator(censoring_start_time, a, denom_times, denom_marks)["integral"] * mark_mask

            numer.append(intensity * torch.exp(-torch.sum(numer_comp, dim=-1, keepdim=True)) / num_seqs)
            denom.append(torch.exp(-torch.sum(denom_comp, dim=-1, keepdim=True)) / num_seqs)
        
        numer, denom = sum(numer), sum(denom)  # each are size (num_samples, marks)
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


class Query:

    @abstractmethod
    def naive_estimate(self, model, num_sample_seq):
        pass

    # TODO: Implement both forms of estimation, currently implemented is just the trapezoidal methods
    # @abstractmethod
    # def mc_estimate(self, model, num_sample_seq, num_int_samples):
    #     pass

    # @abstractmethod
    # def trap_estimate(self, model, num_sample_seq, num_int_samples):
    #     pass

    @abstractmethod
    def estimate(self, model, num_sample_seq, num_int_samples):
        pass

    @abstractmethod
    def proposal_dist_sample(self, model, num_samples=1):
        pass

class TemporalMarkQuery(Query):

    def __init__(self, time_boundaries, mark_restrictions, batch_size=128, device=torch.device('cpu'), use_tqdm=True):
        if isinstance(time_boundaries, list):
            time_boundaries = torch.FloatTensor(time_boundaries)

        self.time_boundaries = time_boundaries.to(device)
        self.mark_restrictions = []    # list of list of restricted marks, if k is in one of these lists, then events with marks=k are not allowed in that respected time span
        for m in mark_restrictions:
            if isinstance(m, int):
                m = torch.LongTensor([m])
            elif isinstance(m, list):
                m = torch.LongTensor(m)
            else:
                assert(isinstance(m, torch.LongTensor))
            self.mark_restrictions.append(m.to(device))

        self.max_time = max(time_boundaries)
        self.num_times = len(time_boundaries)
        assert(time_boundaries[0] > 0)
        assert((time_boundaries[:-1] < time_boundaries[1:]).all())    # strictly increasing
        assert(len(time_boundaries) == len(mark_restrictions))

        self.device = device
        self.batch_size = batch_size
        self.restricted_positions = torch.BoolTensor([len(m) > 0 for m in self.mark_restrictions]).to(device)
        self.use_tqdm = use_tqdm

    @torch.no_grad()
    def naive_estimate(self, model, num_sample_seq, conditional_times=None, conditional_marks=None):
        res = 0.0    # for a sample to count as 1 in the average, it must respect _every_ mark restriction
        if (conditional_times is None) or (conditional_times.numel() == 0):
            offset = 0.0
        else:
            offset = conditional_times.max()
        for _ in tqdm(range(num_sample_seq), disable=not self.use_tqdm): 
            times, marks = model.sample_points(
                T=self.max_time+offset, 
                left_window=0+offset, 
                timestamps=conditional_times, 
                marks=conditional_marks, 
                mark_mask=1.0,
            )
            times, marks = times.squeeze(0), marks.squeeze(0)
            if conditional_times is not None:
                times = times[conditional_times.numel():] - offset
                marks = marks[conditional_times.numel():]

            for i in range(self.num_times):
                if i == 0:
                    a, b = 0.0, self.time_boundaries[0]
                else:
                    a, b = self.time_boundaries[i-1], self.time_boundaries[i]

                if torch.isin(marks[(a < times) & (times <= b)], self.mark_restrictions[i]).any():
                    break
            else:    # Executes if for loop did not break
                res += 1. / num_sample_seq
        return res

    @torch.no_grad()
    def proposal_dist_sample(self, model, conditional_times=None, conditional_marks=None, offset=None):
        last_t = 0.0
        times, marks = conditional_times, conditional_marks
        if offset is None:
            if (conditional_times is None) or (conditional_times.numel() == 0):
                offset = 0.0
            else:
                offset = conditional_times.max() #+1e-32
        for i,t in enumerate(self.time_boundaries):
            mark_mask = torch.ones((model.num_channels,), dtype=torch.float32).to(self.device)
            mark_mask[self.mark_restrictions[i]] = 0.0
            times, marks = model.sample_points(
                T=t+offset, 
                left_window=last_t+offset, 
                timestamps=times, 
                marks=marks, 
                mark_mask=mark_mask,
            )
            last_t = t
        return times.squeeze(0), marks.squeeze(0)

    @torch.no_grad()
    def estimate(self, model, num_sample_seqs, num_int_samples, conditional_times=None, conditional_marks=None, calculate_bounds=False):
        # If true, then we will need to integrate for events in that position
        time_spans = self.time_boundaries - F.pad(self.time_boundaries[:-1].unsqueeze(0), (1,0), 'constant', 0.0).squeeze(0) # Equivalent to: np.ediff1d(self.time_boundaries, to_begin=self.time_boundaries[0])
        time_norm = time_spans[self.restricted_positions].sum()    # Used to scale how many integration sample points each interval uses
        ests = []
        est_lower_bound, est_upper_bound = 0.0, 0.0
        if (conditional_times is None) or (conditional_times.numel() == 0):
            offset = 0.0
        else:
            offset = conditional_times.max()#+1e-32
        for _ in tqdm(range(num_sample_seqs), disable=not self.use_tqdm):
            times, marks = self.proposal_dist_sample(model, conditional_times=conditional_times, conditional_marks=conditional_marks)
            total_int, lower_int, upper_int = 0.0, 0.0, 0.0
            for i in range(len(time_spans)):
                if not self.restricted_positions[i]:
                    continue

                if i == 0:
                    a,b = 0.0, self.time_boundaries[0]
                else:
                    a,b = self.time_boundaries[i-1], self.time_boundaries[i]
                a = a + offset
                b = b + offset
                single_res = model.compensator(
                    a, 
                    b, 
                    conditional_times=times.unsqueeze(0), 
                    conditional_marks=marks.unsqueeze(0), 
                    num_int_pts=max(int(num_int_samples*time_spans[i]/time_norm), 2),  # At least two points are needed to integrate
                    calculate_bounds=calculate_bounds,
                )

                if calculate_bounds:
                    lower_int += single_res["lower_bound"].squeeze(0)[self.mark_restrictions[i]].sum()
                    upper_int += single_res["upper_bound"].squeeze(0)[self.mark_restrictions[i]].sum()
                total_int += single_res["integral"].squeeze(0)[self.mark_restrictions[i]].sum()
                
            ests.append(torch.exp(-total_int))
            if calculate_bounds:
                est_lower_bound += torch.exp(-lower_int) / num_sample_seqs
                est_upper_bound += torch.exp(-upper_int) / num_sample_seqs

        est = sum(ests) / num_sample_seqs
        results = {
            "est": est,
            "naive_var": est - est**2,
            "is_var": sum((e-est)**2 for e in ests) / num_sample_seqs,
        }
        results["rel_eff"] = results["naive_var"] / results["is_var"]

        if calculate_bounds:
            results["lower_est"] = est_lower_bound
            results["upper_est"] = est_upper_bound

        return results


class PositionalMarkQuery(Query):

    def __init__(self, mark_restrictions, batch_size=128, device=torch.device('cpu'), use_tqdm=True):
        self.mark_restrictions = []    # list of list of restricted marks, if k is in one of these lists, then events with marks=k are not allowed in that respected time span
        for m in mark_restrictions:
            if isinstance(m, int):
                self.mark_restrictions.append([m])
            else:
                self.mark_restrictions.append(m)
        self.max_events = len(mark_restrictions)
        self.restricted_positions = torch.BoolTensor([len(m) > 0 for m in self.mark_restrictions])    # If true, then we will need to integrate for events in that position
        self.batch_size = batch_size
        self.device = device
        self.use_tqdm = use_tqdm

    @torch.no_grad()
    def naive_estimate(self, model, num_sample_seq, conditional_times=None, conditional_marks=None):
        mark_res_array = torch.zeros((self.max_events, model.num_channels), dtype=torch.int32)
        for i,m in zip(range(self.max_events), self.mark_restrictions):
            mark_res_array[i,m] = 1

        res = 0.0    # for a sample to count as 1 in the average, it must respect _every_ mark restriction
        for _ in tqdm(range(num_sample_seq), disable=not self.use_tqdm): 
            times, marks = model.sample_points(
                left_window=0 if conditional_times is None else conditional_times.max()+1e-32, 
                length_limit=self.max_events + (0 if conditional_times is None else conditional_times.numel()), 
                timestamps=conditional_times, 
                marks=conditional_marks, 
                mark_mask=1.0,
            )

            times, marks = times.squeeze(0), marks.squeeze(0)
            marks = marks[(0 if conditional_marks is None else conditional_marks.numel()):]  # only take the sampled marks, not the conditional ones
            if mark_res_array[torch.arange(0, self.max_events), marks].sum() == 0:
                res += 1. / num_sample_seq
        return res

    @torch.no_grad()
    def proposal_dist_sample(self, model, conditional_times=None, conditional_marks=None):
        times, marks = conditional_times, conditional_marks
        for i in range(self.max_events):
            mark_mask = torch.ones((model.num_channels,), dtype=torch.float32)
            mark_mask[self.mark_restrictions[i]] = 0.0

            times, marks = model.sample_points(
                left_window=times[..., -1].item() if i > 0 else (0.0 if conditional_times is None else conditional_times.max()+1e-32), 
                length_limit=i+1 + (0 if conditional_times is None else conditional_times.numel()), 
                timestamps=times, 
                marks=marks, 
                mark_mask=mark_mask,
            )
        return times.squeeze(0), marks.squeeze(0)

    @torch.no_grad()
    def estimate(self, model, num_sample_seqs, num_int_samples, conditional_times=None, conditional_marks=None, calculate_bounds=False):
        est_lower_bound, est_upper_bound = 0.0, 0.0
        ests = []
        for _ in tqdm(range(num_sample_seqs), disable=not self.use_tqdm):
            times, marks = self.proposal_dist_sample(model, conditional_times=conditional_times, conditional_marks=conditional_marks)
            time_spans = times[(0 if conditional_times is None else conditional_times.numel()):] - F.pad(times[(0 if conditional_times is None else conditional_times.numel()):-1].unsqueeze(0), (1,0), 'constant', 0.0).squeeze(0) # equivalent to: np.ediff1d(times, to_begin=times[0])
            time_norm = time_spans[self.restricted_positions].sum()    # Used to scale how many integration sample points each interval uses

            lower_int, upper_int = 0.0, 0.0
            total_int = 0.0
            for i in range(len(times)-(0 if conditional_times is None else conditional_times.numel())):
                if not self.restricted_positions[i]:
                    continue

                if conditional_times is None:
                    if i == 0:
                        a,b = 0.0, times[0]
                    else:
                        a,b = times[i-1], times[i]
                else:
                    a,b = times[i-1+conditional_times.numel()], times[i+conditional_times.numel()]

                # if conditional_times is not None:  # Make the last event in the conditional_times effectively 0
                #     a += conditional_times.max()+1e-32
                #     b += conditional_times.max()+1e-32

                single_res = model.compensator(
                    a, 
                    b, 
                    conditional_times=times.unsqueeze(0), 
                    conditional_marks=marks.unsqueeze(0), 
                    num_int_pts=max(int(num_int_samples*time_spans[i]/time_norm), 2),  # at least two points are needed to integrate
                    calculate_bounds=calculate_bounds,
                )
                if calculate_bounds:
                    lower_int += single_res["lower_bound"].squeeze(0)[self.mark_restrictions[i]].sum()
                    upper_int += single_res["upper_bound"].squeeze(0)[self.mark_restrictions[i]].sum()
                total_int += single_res["integral"].squeeze(0)[self.mark_restrictions[i]].sum()
                
            ests.append(torch.exp(-total_int))
            if calculate_bounds:
                est_lower_bound += torch.exp(-lower_int) / num_sample_seqs
                est_upper_bound += torch.exp(-upper_int) / num_sample_seqs

        est = sum(ests) / num_sample_seqs
        results = {
            "est": est,
            "naive_var": est - est**2,
            "is_var": sum((e-est)**2 for e in ests) / num_sample_seqs,
        }
        results["rel_eff"] = results["naive_var"] / results["is_var"]

        if calculate_bounds:
            results["lower_est"] = est_lower_bound
            results["upper_est"] = est_upper_bound

        return results


class UnbiasedHittingTimeQuery(TemporalMarkQuery):

    def __init__(self, up_to, hitting_marks, batch_size=128, device=torch.device('cpu'), use_tqdm=True):
        assert(isinstance(up_to, (float, int)) and up_to > 0)
        assert(isinstance(hitting_marks, (int, list)))
        super().__init__(
            time_boundaries=[up_to], 
            mark_restrictions=[hitting_marks], 
            batch_size=batch_size, 
            device=device,
            use_tqdm=use_tqdm,
        )

    @torch.no_grad()
    def naive_estimate(self, model, num_sample_seqs, conditional_times=None, conditional_marks=None):
        return 1 - super().naive_estimate(model, num_sample_seqs, conditional_times=conditional_times, conditional_marks=conditional_marks)

    @torch.no_grad()
    def estimate(self, model, num_sample_seqs, num_int_samples, conditional_times=None, conditional_marks=None, calculate_bounds=False):
        result = super().estimate(
            model, 
            num_sample_seqs, 
            num_int_samples, 
            conditional_times=conditional_times, 
            conditional_marks=conditional_marks, 
            calculate_bounds=calculate_bounds,
        )
        result["est"] = 1 - result["est"]
        if calculate_bounds:
            result["lower_est"], result["upper_est"] = 1 - result["upper_est"], 1 - result["lower_est"]  # Need to calculate the complement and swap upper and lower bounds
        return result

