from bdb import effective
import torch
import torch.nn.functional as F

from tqdm import tqdm
from abc import ABC, abstractmethod


class CensoredPP:
  def __init__(self, base_process, observed_marks, num_sampled_sequences, use_same_seqs_for_ratio=False):
    assert(len(observed_marks) > 0)  # K=6, observed_marks=[0,1,2] ==> censored_marks=[3,4,5]
    assert(num_sampled_sequences > 0)

    self.base_process = base_process
    self.marks = base_process.marks
    self.observed_marks = torch.LongTensor(observed_marks)
    assert(len(self.observed_marks.shape) == 1)
    self.censored_marks = torch.LongTensor([i for i in range(self.marks) if i not in observed_marks])

    self.num_sampled_sequences = num_sampled_sequences
    self.use_same_seqs_for_ratio = use_same_seqs_for_ratio
    if not use_same_seqs_for_ratio:
        assert(num_sampled_sequences % 2 == 0) # we will use half of sequences for numerator and half for denominator

    try:
      self.dom_rate = base_process.dom_rate
    except:
      pass

  def gen_cond_seqs(
      self,
      right_window,  # Max time to go out until
      left_window,   # Where to start sampling
      fixed_times,   # Either times of events prior to sampling, or that are guaranteed to occur in the middle of sampling
      fixed_marks,   # Corresponding marks of `fixed_times`
  ):
    n = self.num_sampled_sequences
    cond_times, cond_marks = [], []
    mark_mask = torch.ones((self.marks,), dtype=torch.float32)
    mark_mask[self.observed_marks] = 0.0  # Only sample the censored marks (to marginalize them out)
    for _ in range(n):
      times, marks = self.base_process.sample(
          right_window=right_window, 
          left_window=left_window, 
          sampled_times=fixed_times, 
          sampled_marks=fixed_marks,
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
      conditional_times, conditional_marks = torch.FloatTensor([]), torch.IntTensor([])

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
      n = pp.intensity(t, numer_times, numer_marks) * torch.exp(-pp.compensator(censoring_start_time, t, numer_times, numer_marks)[obs].sum())
      
      denom_times, denom_marks = cond_seqs["denom_cond_times"][i], cond_seqs["denom_cond_marks"][i]
      d = torch.exp(-pp.compensator(censoring_start_time, t, denom_times, denom_marks)[obs].sum())

      #print(numer, denom, n, d, numer_times, numer_marks, denom_times, denom_marks)
      #print()

      numer += n / num_seqs
      denom += d / num_seqs
    
    numer[cen] *= 0.0  # zero out censored intensities as they are ill-defined here

    return numer / denom

  def compensator(
      self, 
      a, 
      b, 
      conditional_times, 
      conditional_marks, 
      num_samples=100000, 
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
        intensity = self.base_process.intensity(ts, numer_times, numer_marks)*mark_mask.unsqueeze(0) #[np.newaxis, :]
        numer_comp = self.base_process.compensator_grid(a, b, numer_times, numer_marks, num_samples-1) * mark_mask.unsqueeze(0) #[np.newaxis, :]
        denom_comp = self.base_process.compensator_grid(a, b, denom_times, denom_marks, num_samples-1) * mark_mask.unsqueeze(0) #[np.newaxis, :]

    if censoring_start_time < a:
        numer_comp += self.base_process.compensator(censoring_start_time, a, numer_times, numer_marks) * mark_mask
        denom_comp += self.base_process.compensator(censoring_start_time, a, denom_times, denom_marks) * mark_mask

    numer.append(intensity * torch.exp(-torch.sum(numer_comp, dim=1, keepdim=True)) / num_seqs)
    denom.append(torch.exp(-torch.sum(denom_comp, dim=1, keepdim=True)) / num_seqs)
    
    print(len(numer), numer[0].shape)
    numer, denom = sum(numer), sum(denom)  # each are size (num_samples, marks)
    censored_intensities = numer / denom
    # Perform trapezoidal rule to approximate \int_a^b of censored_intensity(t) dt
    censored_intensities[1:-1] *= 2
    delta = (b-a)/(num_samples-1)
    return delta * censored_intensities.sum(axis=0) / 2

  def sample(self, right_window=None, left_window=0.0, length_limit=None, sampled_times=None, sampled_marks=None, mark_mask=1.0): 
    mark_mask = torch.ones((self.marks,), dtype=torch.float32) * mark_mask
    mark_mask[self.censored_marks] = 0.0  # don't allow samples of censored marks
    return super().sample(
        right_window=right_window, 
        left_window=left_window, 
        sampled_times=sampled_times, 
        sampled_marks=sampled_marks, 
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

    def __init__(self, time_boundaries, mark_restrictions, batch_size=128, device=torch.device('cpu')):
        if isinstance(time_boundaries, list):
            time_boundaries = torch.FloatTensor(time_boundaries, device=device)

        self.time_boundaries = time_boundaries
        self.mark_restrictions = []    # list of list of restricted marks, if k is in one of these lists, then events with marks=k are not allowed in that respected time span
        for m in mark_restrictions:
            if isinstance(m, int):
                m = torch.LongTensor([m], device=device)
            elif isinstance(m, list):
                m = torch.LongTensor(m, device=device)
            else:
                assert(isinstance(m, torch.LongTensor))
            self.mark_restrictions.append(m)

        self.max_time = max(time_boundaries)
        self.num_times = len(time_boundaries)
        assert(time_boundaries[0] > 0)
        assert((time_boundaries[:-1] < time_boundaries[1:]).all())    # strictly increasing
        assert(len(time_boundaries) == len(mark_restrictions))

        self.device = device
        self.batch_size = batch_size
        self.restricted_positions = torch.BoolTensor([len(m) > 0 for m in self.mark_restrictions])

    def naive_estimate(self, model, num_sample_seq):
        res = 0.0    # for a sample to count as 1 in the average, it must respect _every_ mark restriction
        for _ in range(num_sample_seq): 
            times, marks = model.sample(
                right_window=self.max_time, 
                left_window=0, 
                length_limit=None, 
                conditioned_times=None, 
                conditioned_marks=None, 
                mark_mask=1.0,
            )
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

    def proposal_dist_sample(self, model):
        last_t = 0.0
        times, marks = None, None
        for i,t in enumerate(self.time_boundaries):
            mark_mask = torch.ones((model.marks,), dtype=torch.float32)
            mark_mask[self.mark_restrictions[i]] = 0.0
            times, marks = model.sample(
                right_window=t, 
                left_window=last_t, 
                length_limit=None, 
                conditioned_times=times, 
                conditioned_marks=marks, 
                mark_mask=mark_mask,
            )
        return times, marks

    def estimate(self, model, num_sample_seqs, num_int_samples):
            # If true, then we will need to integrate for events in that position
        time_spans = self.time_boundaries - F.pad(self.time_boundaries[:-1].unsqueeze(0), (1,0), 'constant', 0.0).squeeze(0) # Equivalent to: np.ediff1d(self.time_boundaries, to_begin=self.time_boundaries[0])
        time_norm = time_spans[self.restricted_positions].sum()    # Used to scale how many integration sample points each interval uses
        est, est_sq = 0.0, 0.0
        for _ in range(num_sample_seqs):
            times, marks = self.proposal_dist_sample(model)
            total_int = 0.0
            for i in range(len(time_spans)):
                if not self.restricted_positions[i]:
                    continue

                if i == 0:
                    a,b = 0.0, self.time_boundaries[0]
                else:
                    a,b = self.time_boundaries[i-1], self.time_boundaries[i]
                single_int = model.compensator(
                    a, 
                    b, 
                    conditioned_times=times, 
                    conditioned_marks=marks, 
                    num_samples=int(num_int_samples*time_spans[i]/time_norm),
                )
                total_int += single_int[self.mark_restrictions[i]].sum()

            est += torch.exp(-total_int) / num_sample_seqs
            est_sq += torch.exp(-total_int)**2 / num_sample_seqs
        
        return est, (est - est**2) / (est_sq - est**2)


class PositionalMarkQuery(Query):

    def __init__(self, mark_restrictions, batch_size=128, device=torch.device('cpu')):
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

    def naive_estimate(self, model, num_sample_seq):
        mark_res_array = torch.zeros((self.max_events, model.marks), dtype=torch.int32)
        for i,m in zip(range(self.max_events), self.mark_restrictions):
            mark_res_array[i,m] = 1

        res = 0.0    # for a sample to count as 1 in the average, it must respect _every_ mark restriction
        for _ in range(num_sample_seq): 
            times, marks = model.sample(
                right_window=None, 
                left_window=0, 
                length_limit=self.max_events, 
                conditioned_times=None, 
                conditioned_marks=None, 
                mark_mask=1.0,
            )
            if mark_res_array[torch.arange(0, self.max_events), marks].sum() == 0:
                res += 1. / num_sample_seq
        return res

    def proposal_dist_sample(self, model):
        times, marks = None, None
        for i in range(self.max_events):
            mark_mask = torch.ones((model.marks,), dtype=torch.float32)
            mark_mask[self.mark_restrictions[i]] = 0.0
            times, marks = model.sample(
                right_window=None, 
                left_window=times[-1] if i > 0 else 0.0, 
                length_limit=i+1, 
                conditioned_times=times, 
                conditioned_marks=marks, 
                mark_mask=mark_mask,
            )
        return times, marks

    def estimate(self, model, num_sample_seqs, num_int_samples):
        est, est_sq = 0.0, 0.0
        for _ in range(num_sample_seqs):
            times, marks = self.proposal_dist_sample(model)
            time_spans = times - F.pad(times[:-1].unsqueeze(0), (1,0), 'constant', 0.0).squeeze(0) # equivalent to: np.ediff1d(times, to_begin=times[0])
            time_norm = time_spans[self.restricted_positions].sum()    # Used to scale how many integration sample points each interval uses

            total_int = 0.0
            for i in range(len(times)):
                if not self.restricted_positions[i]:
                    continue

                if i == 0:
                    a,b = 0.0, times[0]
                else:
                    a,b = times[i-1], times[i]
                single_int = model.compensator(
                    a, 
                    b, 
                    conditioned_times=times, 
                    conditioned_marks=marks, 
                    num_samples=int(num_int_samples*time_spans[i]/time_norm),
                )
                total_int += single_int[self.mark_restrictions[i]].sum()

            est += torch.exp(-total_int) / num_sample_seqs
            est_sq += torch.exp(-total_int)**2 / num_sample_seqs

        return est, (est - est**2) / (est_sq - est**2)


class UnbiasedHittingTimeQuery(TemporalMarkQuery):

    def __init__(self, up_to, hitting_marks, batch_size=128, device=torch.device('cpu')):
        assert(isinstance(up_to, (float, int)) and up_to > 0)
        assert(isinstance(hitting_marks, (int, list)))
        super().__init__(
            time_boundaries=[up_to], 
            mark_restrictions=hitting_marks, 
            batch_size=batch_size, 
            device=device,
        )

    def naive_estimate(self, model, num_sample_seqs):
        return 1 - super().naive_estimate(model, num_sample_seqs)

    def estimate(self, model, num_sample_seqs, num_int_samples):
        est, eff = super().estimate(model, num_sample_seqs, num_int_samples)
        return 1 - est, eff

