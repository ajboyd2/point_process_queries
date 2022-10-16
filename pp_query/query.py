import torch
import torch.nn.functional as F
import math

from tqdm import tqdm
from abc import abstractmethod

from pp_query.modules.utils import flatten

ADAPT_DOM_RATE = True

def batch_samples(times, marks, states, batch_size):
    return times, marks, states
    # time_pad = torch.finfo(torch.float32).max
    # mark_pad = 0
    # times, marks, states = zip(*sorted(zip(times, marks, states), key=lambda x: -x[0].shape[-1]))  # sort by lengths, decreasing
    # batches, bt, bm, bs = [], [], [], []
    # batch_amt, batch_len = 0, None
    # for i in range(len(times)):
    #     t, m, s = times[i], marks[i], states[i]
    #     if batch_len is None:
    #         batch_len = t.shape[-1]
    #     else:
    #         to_pad = batch_len - t.shape[-1]
    #         t = F.pad(t, pad=(0, to_pad), mode='constant', value=time_pad)
    #         m = F.pad(m, pad=(0, to_pad), mode='constant', value=mark_pad)
    #         s = F.pad(s, pad=(0, 0, 0, to_pad), mode='replicate')
    #     bt.append(t); bm.append(m); bs.append(s)

    #     if (batch_amt >= batch_size) or (i == len(times)-1):
    #         batches.append((
    #             torch.cat(bt, dim=0),
    #             torch.cat(bm, dim=0),
    #             torch.cat(bs, dim=0),
    #         ))
    #         bt, bm, bs = [], [], []
    #         batch_amt, batch_len = 0, None

    # return zip(*batches)

class CensoredPP:
    def __init__(self, base_process, observed_marks, num_sampled_sequences, use_same_seqs_for_ratio=False, batch_size=256, device=torch.device("cpu"), use_tqdm=True, proposal_batch_size=1024):
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

    def __init__(self, time_boundaries, mark_restrictions, batch_size=256, device=torch.device('cpu'), use_tqdm=True, proposal_batch_size=1024):
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
        self.proposal_batch_size = proposal_batch_size

    @torch.no_grad()
    def naive_estimate(self, model, num_sample_seq, conditional_times=None, conditional_marks=None):
        res = 0.0    # for a sample to count as 1 in the average, it must respect _every_ mark restriction
        if (conditional_times is None) or (conditional_times.numel() == 0):
            offset = 0.0
        else:
            offset = conditional_times.max()

        all_times, all_marks, _ = model.batch_sample_points(
            T=self.max_time+offset, 
            left_window=0+offset, 
            timestamps=conditional_times, 
            marks=conditional_marks, 
            mark_mask=1.0,
            num_samples=num_sample_seq,
            adapt_dom_rate=ADAPT_DOM_RATE,
            proposal_batch_size=self.proposal_batch_size,
        )
        all_times = flatten([times.unbind(dim=0) for times in all_times])
        all_marks = flatten([marks.unbind(dim=0) for marks in all_marks])
        for times, marks in tqdm(zip(all_times, all_marks), disable=not self.use_tqdm): 
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
    def proposal_dist_sample(self, model, conditional_times=None, conditional_marks=None, offset=None, num_samples=1):
        last_t = 0.0
        times, marks = conditional_times, conditional_marks
        if offset is None:
            if (conditional_times is None) or (conditional_times.numel() == 0):
                offset = 0.0
            else:
                offset = conditional_times.max() #+1e-32
        
        mark_mask = torch.ones((len(self.time_boundaries)+1, model.num_channels,), dtype=torch.float32).to(self.device)  # +1 to have an unrestricted final mask
        for i in range(len(self.time_boundaries)):
            mark_mask[i, self.mark_restrictions[i]] = 0.0
        mask_dict = {
            "temporal_mark_restrictions": mark_mask,
            "time_boundaries": self.time_boundaries+offset,
        }

        # for i,t in enumerate(self.time_boundaries):
            # mark_mask = torch.ones((model.num_channels,), dtype=torch.float32).to(self.device)
            # mark_mask[self.mark_restrictions[i]] = 0.0
        all_times, all_marks, all_states = model.batch_sample_points(
            T=self.time_boundaries.max()+offset, #t+offset, 
            left_window=offset, #last_t+offset, 
            timestamps=times, 
            marks=marks, 
            mark_mask=1.0,#mark_mask,
            num_samples=num_samples,
            mask_dict=mask_dict,
            adapt_dom_rate=ADAPT_DOM_RATE,
            proposal_batch_size=self.proposal_batch_size,
        )

        all_times, all_marks, all_states = batch_samples(all_times, all_marks, all_states, self.batch_size)
        # last_t = t
        return all_times, all_marks, all_states #times.squeeze(0), marks.squeeze(0)


    # @torch.no_grad()
    # def proposal_dist_sample(self, model, conditional_times=None, conditional_marks=None, offset=None):
    #     last_t = 0.0
    #     times, marks = conditional_times, conditional_marks
    #     if offset is None:
    #         if (conditional_times is None) or (conditional_times.numel() == 0):
    #             offset = 0.0
    #         else:
    #             offset = conditional_times.max() #+1e-32
        
    #     for i,t in enumerate(self.time_boundaries):
    #         mark_mask = torch.ones((model.num_channels,), dtype=torch.float32).to(self.device)
    #         mark_mask[self.mark_restrictions[i]] = 0.0
    #         times, marks = model.sample_points(
    #             T=t+offset, 
    #             left_window=last_t+offset, 
    #             timestamps=times, 
    #             marks=marks, 
    #             mark_mask=mark_mask,
    #         )
    #         last_t = t
    #     return times.squeeze(0), marks.squeeze(0)

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
            offset = conditional_times.max().item() #+1e-32
        
        # if self.batch_size > 1:
        batch_sizes = [min(self.batch_size, num_sample_seqs-i*self.batch_size) for i in range(math.ceil(num_sample_seqs / self.batch_size))]
        for batch_size in batch_sizes:
            all_times, all_marks, all_states = self.proposal_dist_sample(model, conditional_times=conditional_times, conditional_marks=conditional_marks, num_samples=batch_size)
            # else:
                # all_seqs = [self.proposal_dist_sample(model, conditional_times=conditional_times, conditional_marks=conditional_marks) for _ in range(num_sample_seqs)]
                # all_times, all_marks = zip(*all_seqs)

            for times, marks, states in tqdm(zip(all_times, all_marks, all_states), disable=not self.use_tqdm):
                # times, marks = self.proposal_dist_sample(model, conditional_times=conditional_times, conditional_marks=conditional_marks)
                total_int, lower_int, upper_int = 0.0, 0.0, 0.0
                # print(conditional_times.shape, conditional_times.numel(), times.shape, marks.shape, states.shape)
                
                for i in range(len(time_spans)):
                    if not self.restricted_positions[i]:
                        continue

                    if i == 0:
                        a,b = 0.0, self.time_boundaries[0].item()
                    else:
                        a,b = self.time_boundaries[i-1].item(), self.time_boundaries[i].item()
                    a = a + offset
                    b = b + offset
                    single_res = model.compensator(
                        a, 
                        b, 
                        conditional_times=times, #.unsqueeze(0), 
                        conditional_marks=marks, #.unsqueeze(0), 
                        conditional_states=states,
                        num_int_pts=max(int(num_int_samples*time_spans[i]/time_norm), 2),  # At least two points are needed to integrate
                        calculate_bounds=calculate_bounds,  # TODO: Use mask_dict here too to do a single compensator pass
                    )

                    if calculate_bounds:
                        lower_int += single_res["lower_bound"][:, self.mark_restrictions[i]].sum(dim=-1)
                        upper_int += single_res["upper_bound"][:, self.mark_restrictions[i]].sum(dim=-1)
                    total_int += single_res["integral"][:, self.mark_restrictions[i]].sum(dim=-1)
                    
                ests.append(torch.exp(-total_int))
                if calculate_bounds:
                    est_lower_bound += torch.exp(-lower_int).sum(dim=0) / num_sample_seqs #(num_sample_seqs*lower_int.shape[0])
                    est_upper_bound += torch.exp(-upper_int).sum(dim=0) / num_sample_seqs #(num_sample_seqs*upper_int.shape[0])

        ests = torch.cat(ests, dim=0)
        assert(ests.numel() == num_sample_seqs)
        est = ests.mean()  #.sum() / num_sample_seqs
        results = {
            "est": est,
            "naive_var": est - est**2,
            "is_var": (est-ests).pow(2).mean(), #sum((e-est)**2 for e in ests) / num_sample_seqs,
        }
        results["rel_eff"] = results["naive_var"] / results["is_var"]

        if calculate_bounds:
            results["lower_est"] = est_lower_bound
            results["upper_est"] = est_upper_bound

        return results


class PositionalMarkQuery(Query):

    def __init__(self, mark_restrictions, batch_size=256, device=torch.device('cpu'), use_tqdm=True, proposal_batch_size=1024):
        self.mark_restrictions = []    # list of list of restricted marks, if k is in one of these lists, then events with marks=k are not allowed in that respected time span
        for m in mark_restrictions:
            if isinstance(m, int):
                self.mark_restrictions.append([m])
            else:
                if isinstance(m, torch.Tensor):
                    assert(len(m.shape) == 1)
                self.mark_restrictions.append(m)
        self.max_events = len(mark_restrictions)
        self.restricted_positions = torch.BoolTensor([len(m) > 0 for m in self.mark_restrictions]).to(device)  # If true, then we will need to integrate for events in that position
        self.batch_size = batch_size
        self.device = device
        self.use_tqdm = use_tqdm
        self.proposal_batch_size = proposal_batch_size

    @torch.no_grad()
    def naive_estimate(self, model, num_sample_seqs, conditional_times=None, conditional_marks=None):
        mark_res_array = torch.zeros((self.max_events, model.num_channels), dtype=torch.int32).to(self.device)
        for i,m in zip(range(self.max_events), self.mark_restrictions):
            mark_res_array[i,m] = 1

        res = 0.0    # for a sample to count as 1 in the average, it must respect _every_ mark restriction
        indices = torch.arange(0, self.max_events).to(self.device)
        _, all_marks, _ = model.batch_sample_points(
            left_window=0 if conditional_times is None else conditional_times.max(), 
            length_limit=self.max_events + (0 if conditional_times is None else conditional_times.numel()), 
            timestamps=conditional_times, 
            marks=conditional_marks, 
            mark_mask=1.0,
            num_samples=num_sample_seqs,
            adapt_dom_rate=ADAPT_DOM_RATE,
            proposal_batch_size=self.proposal_batch_size,
        )
        all_marks = flatten([marks.unbind(dim=0) for marks in all_marks])
        for marks in tqdm(all_marks, disable=not self.use_tqdm): 
            marks = marks[(0 if conditional_marks is None else conditional_marks.numel()):]  # only take the sampled marks, not the conditional ones
            if mark_res_array[indices, marks].sum() == 0:
                res += 1. / num_sample_seqs
        return res

    @torch.no_grad()
    def proposal_dist_sample(self, model, conditional_times=None, conditional_marks=None, num_samples=1):
        cond_len = 0 if conditional_times is None else conditional_times.numel()
        mark_mask = torch.ones((self.max_events+1+cond_len, model.num_channels,), dtype=torch.float32).to(self.device)  # +1 to have an unrestricted final mask
        for i in range(cond_len, self.max_events+cond_len):
            mark_mask[i, self.mark_restrictions[i-cond_len]] = 0.0
        mask_dict = {
            "positional_mark_restrictions": mark_mask,
        }

        all_times, all_marks, all_states = model.batch_sample_points(
            left_window=0.0 if conditional_times is None else conditional_times.max(), 
            length_limit=self.max_events + (0 if conditional_times is None else conditional_times.numel()), 
            timestamps=conditional_times, 
            marks=conditional_marks, 
            mark_mask=1.0,
            num_samples=num_samples,
            mask_dict=mask_dict,
            adapt_dom_rate=ADAPT_DOM_RATE,
            proposal_batch_size=self.proposal_batch_size,
        )

        all_times, all_marks, all_states = batch_samples(all_times, all_marks, all_states, self.batch_size)

        return all_times, all_marks, all_states

    # @torch.no_grad()
    # def proposal_dist_sample(self, model, conditional_times=None, conditional_marks=None, num_samples=1):
    #     times, marks = conditional_times, conditional_marks
    #     for i in range(self.max_events):
    #         mark_mask = torch.ones((model.num_channels,), dtype=torch.float32).to(self.device)
    #         mark_mask[self.mark_restrictions[i]] = 0.0

    #         times, marks = model.sample_points(
    #             left_window=times[..., -1].item() if i > 0 else (0.0 if conditional_times is None else conditional_times.max()+1e-32), 
    #             length_limit=i+1 + (0 if conditional_times is None else conditional_times.numel()), 
    #             timestamps=times, 
    #             marks=marks, 
    #             mark_mask=mark_mask,
    #         )
    #     return times.squeeze(0), marks.squeeze(0)

    @torch.no_grad()
    def estimate(self, model, num_sample_seqs, num_int_samples, conditional_times=None, conditional_marks=None, calculate_bounds=False):
        est_lower_bound, est_upper_bound = 0.0, 0.0
        ests = []
        all_times, all_marks, all_states = self.proposal_dist_sample(model, conditional_times=conditional_times, conditional_marks=conditional_marks, num_samples=num_sample_seqs)
        num_integrable_spans = self.restricted_positions.sum().item()
        for times, marks, states in tqdm(zip(all_times, all_marks, all_states), disable=not self.use_tqdm):
            #time_spans = times[:, (0 if conditional_times is None else conditional_times.shape[-1]):] - F.pad(times[:, (0 if conditional_times is None else conditional_times.shape[-1]):-1], (1,0), 'constant', 0.0) # equivalent to: np.ediff1d(times, to_begin=times[0])
            #time_norm = time_spans[:, self.restricted_positions].sum(dim=-1)    # Used to scale how many integration sample points each interval uses
            lower_int, upper_int = 0.0, 0.0
            total_int = 0.0
            for i in range(times.shape[-1]-(0 if conditional_times is None else conditional_times.shape[-1])):
                if not self.restricted_positions[i]:
                    continue

                if conditional_times is None:
                    if i == 0:
                        a,b = torch.zeros_like(times[:, 0]), times[:, 0]
                    else:
                        a,b = times[:, i-1], times[:, i]
                else:
                    a,b = times[:, i-1+conditional_times.shape[-1]], times[:, i+conditional_times.shape[-1]]

                # if conditional_times is not None:  # Make the last event in the conditional_times effectively 0
                #     a += conditional_times.max()+1e-32
                #     b += conditional_times.max()+1e-32

                single_res = model.compensator(
                    a, 
                    b, 
                    conditional_times=times,#.unsqueeze(0), 
                    conditional_marks=marks,#.unsqueeze(0), 
                    num_int_pts=max(int(num_int_samples/num_integrable_spans), 2), #(num_int_samples*time_spans[:, i]/time_norm).long().clamp(min=2),  # at least two points are needed to integrate
                    conditional_states=states,
                    calculate_bounds=calculate_bounds,
                )
                if calculate_bounds:
                    lower_int += single_res["lower_bound"][:, self.mark_restrictions[i]].sum(dim=-1)
                    upper_int += single_res["upper_bound"][:, self.mark_restrictions[i]].sum(dim=-1)
                total_int += single_res["integral"][:, self.mark_restrictions[i]].sum(dim=-1)
                
            ests.append(torch.exp(-total_int))
            if calculate_bounds:
                est_lower_bound += torch.exp(-lower_int).sum(dim=0) / num_sample_seqs #(num_sample_seqs*lower_int.shape[0])
                est_upper_bound += torch.exp(-upper_int).sum(dim=0) / num_sample_seqs #(num_sample_seqs*upper_int.shape[0])

        ests = torch.cat(ests, dim=0)
        assert(ests.numel() == num_sample_seqs)
        est = ests.mean()  #sum(ests) / num_sample_seqs
        results = {
            "est": est,
            "naive_var": est - est**2,
            "is_var": (est - ests).pow(2).mean(), #sum((e-est)**2 for e in ests) / num_sample_seqs,
        }
        results["rel_eff"] = results["naive_var"] / results["is_var"]

        if calculate_bounds:
            results["lower_est"] = est_lower_bound
            results["upper_est"] = est_upper_bound

        return results


class UnbiasedHittingTimeQuery(TemporalMarkQuery):

    def __init__(self, up_to, hitting_marks, batch_size=256, device=torch.device('cpu'), use_tqdm=True, proposal_batch_size=1024):
        assert(isinstance(up_to, (float, int)) and up_to > 0)
        assert(isinstance(hitting_marks, (int, list)))
        super().__init__(
            time_boundaries=[up_to], 
            mark_restrictions=[hitting_marks], 
            batch_size=batch_size, 
            device=device,
            use_tqdm=use_tqdm,
            proposal_batch_size=proposal_batch_size,
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

class MarginalMarkQuery(PositionalMarkQuery):

    def __init__(
        self, 
        n, 
        marks_of_interest, 
        total_marks, 
        batch_size=256, 
        device=torch.device('cpu'), 
        use_tqdm=True, 
        proposal_batch_size=1024,
        proposal_right_window_limit=float('inf'), 
        precision_stop=1e-2, 
        default_dominating_rate=10000.,
        dynamic_buffer_size=64,
    ):
        assert(isinstance(n, int) and n > 0)
        assert(isinstance(marks_of_interest, (int, list, torch.Tensor)))
        if isinstance(marks_of_interest, int):
            marks_of_interest = [marks_of_interest]
        marks_of_disinterest = [mark for mark in range(total_marks) if mark not in marks_of_interest]
        super().__init__(
            mark_restrictions=[[]]*(n-1) + [marks_of_disinterest], 
            batch_size=batch_size, 
            device=device,
            use_tqdm=use_tqdm,
            proposal_batch_size=proposal_batch_size,
        )

        if not isinstance(marks_of_interest, torch.Tensor):
            marks_of_interest = torch.tensor(marks_of_interest)
        if not isinstance(marks_of_disinterest, torch.Tensor):
            marks_of_disinterest = torch.tensor(marks_of_disinterest)
        self.marks_of_interest = marks_of_interest
        self.marks_of_disinterest = marks_of_disinterest

        self.proposal_right_window_limit = proposal_right_window_limit
        self.precision_stop = precision_stop
        self.default_dominating_rate = default_dominating_rate 
        self.dynamic_buffer_size = dynamic_buffer_size
    
    @torch.no_grad()
    def naive_estimate(self, model, num_sample_seqs, conditional_times=None, conditional_marks=None):
        res = 0.0    # for a sample to count as 1 in the average, it must respect _every_ mark restriction
        _, all_marks, _ = model.batch_sample_points(
            left_window=0 if conditional_times is None else conditional_times.max(), 
            length_limit=self.max_events + (0 if conditional_times is None else conditional_times.numel()), 
            timestamps=conditional_times, 
            marks=conditional_marks, 
            mark_mask=1.0,
            num_samples=num_sample_seqs,
            adapt_dom_rate=ADAPT_DOM_RATE,
            proposal_batch_size=self.proposal_batch_size,
        )
        for marks in tqdm(all_marks, disable=not self.use_tqdm): 
            marks = marks[:, -1]  # we only care about the last mark here
            res += torch.isin(marks, self.marks_of_interest).sum() / num_sample_seqs

        return res.item()

    @torch.no_grad()
    def alt_estimate(self, model, num_sample_seqs, conditional_times=None, conditional_marks=None):
        old_buffer_size = model.dyn_dom_buffer
        model.dyn_dom_buffer = self.dynamic_buffer_size
        results = model.batch_ab_sample_query(
            marks=conditional_marks, 
            timestamps=conditional_times, 
            A=self.marks_of_interest,
            B=self.marks_of_disinterest,
            dominating_rate=self.default_dominating_rate, 
            T=5000., 
            left_window=0 if conditional_times.numel() == 0 else conditional_times.max(), 
            num_samples=num_sample_seqs, 
            proposal_batch_size=1024, 
            adapt_dom_rate=True,
            precision_stop=self.precision_stop,
            marginal_mark_n=self.max_events,
        )
        model.dyn_dom_buffer = old_buffer_size
        return results

class ABeforeBQuery(Query):

    def __init__(
        self, 
        A, 
        B, 
        total_marks, 
        batch_size=256, 
        device=torch.device('cpu'), 
        use_tqdm=True, 
        proposal_batch_size=1024, 
        proposal_right_window_limit=float('inf'), 
        precision_stop=1e-2, 
        default_dominating_rate=200000.,
        dynamic_buffer_size=512,
    ):
        self.A = A if isinstance(A, torch.Tensor) else torch.tensor(A, dtype=torch.long).to(device)
        self.B = B if isinstance(B, torch.Tensor) else torch.tensor(B, dtype=torch.long).to(device)
        assert((len(self.A.shape)==1) and (self.A.numel() > 0))
        assert((len(self.B.shape)==1) and (self.B.numel() > 0))
        assert((~torch.isin(self.A, self.B).any()).item())
        self.A_and_B = torch.cat((self.A, self.B), dim=0)

        self.batch_size = batch_size
        self.device = device
        self.use_tqdm = use_tqdm
        self.proposal_batch_size = proposal_batch_size

        self.query_mask = torch.ones(total_marks).to(self.device)
        self.query_mask[self.B] = 0.0

        self.estimate_mask = 1 - self.query_mask

        self.censor_mask = torch.ones(total_marks).to(self.device)
        self.censor_mask[self.A_and_B] = 0.0
        self.observed_mask = 1 - self.censor_mask

        self.proposal_right_window_limit = proposal_right_window_limit
        self.precision_stop = precision_stop
        self.default_dominating_rate = default_dominating_rate 
        self.dynamic_buffer_size = dynamic_buffer_size

        # print("A", self.A)
        # print("B", self.B)
        # print("A and B", self.A_and_B)
        # print("Query Mask", self.query_mask.tolist())
        # print("Estimate Mask", self.estimate_mask.tolist())
        # print("Censor Mask", self.censor_mask.tolist())
        # print("Observed Mask", self.observed_mask.tolist())
        # print()

    @torch.no_grad()
    def naive_estimate(self, model, num_sample_seqs, conditional_times=None, conditional_marks=None):
        _, all_marks, _ = model.batch_sample_points(
            left_window=0 if conditional_times is None else conditional_times.max(), 
            stop_marks=self.A_and_B,
            timestamps=conditional_times, 
            marks=conditional_marks, 
            mark_mask=1.0,
            num_samples=num_sample_seqs,
            adapt_dom_rate=ADAPT_DOM_RATE,
            proposal_batch_size=self.proposal_batch_size,
        )
        all_marks = torch.cat(all_marks, dim=0)
        assert(torch.isin(all_marks, self.A_and_B).all())
        res = torch.isin(all_marks, self.A).float().mean().item()  # Average amount of times A came before B
        
        return res

    @torch.no_grad()
    def proposal_dist_sample(self, model, conditional_times=None, conditional_marks=None, num_censored_seqs=1, num_query_seqs=1):
        # num_censored_seqs is the number of sequences to use to estimate the censored intensity
        # num_query_seqs is the number of sequences to use to estimate the marginal mark distribution outer query (really we just need to know how far to integrate the censored intensity)
        right_window_limit = self.proposal_right_window_limit
        query_times, _, _ = model.batch_sample_points(
            left_window=0 if conditional_times is None else conditional_times.max(), 
            stop_marks=self.A,
            T=right_window_limit,
            timestamps=conditional_times, 
            marks=conditional_marks, 
            mark_mask=self.query_mask,
            num_samples=num_query_seqs,
            adapt_dom_rate=ADAPT_DOM_RATE,
            proposal_batch_size=self.proposal_batch_size,
        )
        # query_times, query_marks = torch.cat(query_times, dim=0), torch.cat(query_marks, dim=0)
        query_times = torch.cat(query_times, dim=0)
        if right_window_limit is None:
            max_query_time = torch.minimum(query_times.max(), torch.quantile(query_times, 0.9)*3)
        else:
            max_query_time = min(right_window_limit, query_times.max())

        censored_times, censored_marks, censored_states = model.batch_sample_points(
            left_window=0 if conditional_times is None else conditional_times.max(), 
            T=max_query_time,
            timestamps=conditional_times, 
            marks=conditional_marks, 
            mark_mask=self.censor_mask,
            num_samples=num_censored_seqs,
            adapt_dom_rate=ADAPT_DOM_RATE,
            proposal_batch_size=self.proposal_batch_size,
        )

        return max_query_time, query_times, censored_times, censored_marks, censored_states
    
    @torch.no_grad()
    def estimate(self, model, num_sample_seqs, num_int_samples, pct_used_for_censoring=0.5, conditional_times=None, conditional_marks=None):
        num_censored_seqs = min(max(int(num_sample_seqs*pct_used_for_censoring), 1), num_sample_seqs-1)
        num_query_seqs = num_sample_seqs - num_censored_seqs

        max_query_time, query_times, censored_times, censored_marks, censored_states = self.proposal_dist_sample(
            model=model,
            conditional_times=conditional_times,
            conditional_marks=conditional_marks,
            num_censored_seqs=num_censored_seqs,
            num_query_seqs=num_query_seqs,
        )
        ts = torch.cat([query_times, torch.linspace(0.0, max_query_time, num_int_samples).to(self.device)], dim=0)
        ts_sorted = torch.sort(ts, dim=0)
        ts_sorted, ts_indices = ts_sorted.values, ts_sorted.indices
        query_indices = ts_indices.argsort(-1)[:query_times.numel()]

        numer, denom = 0.0, 0.0
        for ct, _, cs in zip(censored_times, censored_marks, censored_states):
            intensity = model.get_intensity(
                state_values=cs, 
                state_times=ct, 
                timestamps=ts_sorted.unsqueeze(0).expand(ct.shape[0], -1), 
                marks=None, 
                # state_marks=numer_marks, 
                mark_mask=1.0,
            )["all_log_mark_intensities"].exp()

            comp = F.pad(torch.cumulative_trapezoid((intensity*self.observed_mask.unsqueeze(0)).sum(dim=-1), x=ts_sorted, dim=-1), (1,0), 'constant', 0.0)
            numer += ((intensity*self.estimate_mask.unsqueeze(0)).sum(dim=-1) * torch.exp(-comp)).sum(dim=0) / num_censored_seqs
            denom += torch.exp(-comp).sum(dim=0) / num_censored_seqs
        
        censored_intensity = numer / denom  # shape=(len(ts_sorted),)
        comp_censored_intensity = F.pad(torch.cumulative_trapezoid(censored_intensity, x=ts_sorted, dim=-1), (1,0), 'constant', 0.0)[query_indices]
        ests = torch.exp(-comp_censored_intensity)
        est = ests.mean()
        results = {
            "est": est,
            "naive_var": est - est**2,
            "is_var": (est - ests).pow(2).mean(), #sum((e-est)**2 for e in ests) / num_sample_seqs,
        }
        results["rel_eff"] = results["naive_var"] / results["is_var"]
        return results


    @torch.no_grad()
    def alt_proposal_dist_sample(self, model, conditional_times=None, conditional_marks=None, num_sample_seqs=1):
        # num_censored_seqs is the number of sequences to use to estimate the censored intensity
        # num_query_seqs is the number of sequences to use to estimate the marginal mark distribution outer query (really we just need to know how far to integrate the censored intensity)
        right_window_limit = self.proposal_right_window_limit
        assert(right_window_limit is not None)
        sampled_times, sampled_marks, sampled_states = model.batch_sample_points(
            left_window=0 if conditional_times is None else conditional_times.max(), 
            T=right_window_limit,
            timestamps=conditional_times, 
            marks=conditional_marks, 
            mark_mask=self.censor_mask,
            num_samples=num_sample_seqs,
            adapt_dom_rate=ADAPT_DOM_RATE,
            proposal_batch_size=self.proposal_batch_size,
        )

        return sampled_times, sampled_marks, sampled_states

    @torch.no_grad()
    def alt_estimate(self, model, num_sample_seqs, num_int_samples, conditional_times=None, conditional_marks=None):
        sampled_times, sampled_marks, sampled_states = self.alt_proposal_dist_sample(
            model=model,
            conditional_times=conditional_times,
            conditional_marks=conditional_marks,
            num_sample_seqs=num_sample_seqs,
        )
        ts = torch.linspace(0.0, self.proposal_right_window_limit, num_int_samples).to(self.device)

        a_b_est, b_a_est, avg_ests = 0.0, 0.0, []
        for st, _, ss in zip(sampled_times, sampled_marks, sampled_states):
            intensity = model.get_intensity(
                state_values=ss, 
                state_times=st, 
                timestamps=ts.unsqueeze(0).expand(st.shape[0], -1), 
                marks=None, 
                # state_marks=numer_marks, 
                mark_mask=1.0,
            )["all_log_mark_intensities"].exp()

            a_intensity = intensity[..., self.A].sum(dim=-1)
            b_intensity = intensity[..., self.B].sum(dim=-1)
            ab_intensity = a_intensity + b_intensity
            comp = torch.exp(-F.pad(torch.cumulative_trapezoid(ab_intensity, x=ts, dim=-1), (1,0), 'constant', 0.0))
            a_b_est += torch.trapezoid(a_intensity * comp, x=ts, dim=-1).sum(dim=0) / num_sample_seqs
            b_a_est += torch.trapezoid(b_intensity * comp, x=ts, dim=-1).sum(dim=0) / num_sample_seqs
            avg_ests.append(torch.trapezoid((a_intensity - b_intensity) * comp, x=ts, dim=-1))

        avg_ests = 0.5+0.5*torch.cat(avg_ests)
        avg_est = avg_ests.mean()
            
        results = {
            "est": avg_est,
            "lower_bound": a_b_est,
            "upper_bound": 1 - b_a_est,
            "naive_var": avg_est*(1-avg_est),
            "is_var": (avg_est-avg_ests).pow(2).mean(),
        }
        results["rel_eff"] = results["naive_var"] / results["is_var"]
        return results

    @torch.no_grad()
    def alt_2_estimate(self, model, num_sample_seqs, conditional_times=None, conditional_marks=None):
        old_buffer_size = model.dyn_dom_buffer
        model.dyn_dom_buffer = self.dynamic_buffer_size
        results = model.batch_ab_sample_query(
            marks=conditional_marks, 
            timestamps=conditional_times, 
            A=self.A,
            B=self.B,
            dominating_rate=self.default_dominating_rate, 
            T=5000., 
            left_window=0 if conditional_times.numel() == 0 else conditional_times.max(), 
            num_samples=num_sample_seqs, 
            proposal_batch_size=1024, 
            adapt_dom_rate=True,
            precision_stop=self.precision_stop,
        )
        model.dyn_dom_buffer = old_buffer_size
        return results

'''
USING MAX
tensor([1.3966e+01, 2.8925e+01, 2.8912e+01, 2.8692e+01, 2.8581e+01, 2.5564e+01,
        1.7557e+01, 4.1577e-04, 6.0813e-05, 4.6060e-05, 6.2766e-05],
       device='cuda:0')
Pis 0.041820254165679215 0.40454307198524475

USING 90% * 10
tensor([1.3966e+01, 2.8925e+01, 2.8912e+01, 2.8692e+01, 2.8581e+01, 2.5564e+01,
        1.7557e+01, 6.0637e-03, 5.2135e-04, 3.0391e-04, 2.8301e-04],
       device='cuda:0')
Pis 0.010878360692411661 0.40458938479423523

USING 90% * 3

'''