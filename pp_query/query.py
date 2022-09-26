import torch

from abc import ABC, abstractmethod

class Query:

    @abstractmethod
    def naive_estimate(self, model, num_sample_seq):
        pass

    @abstractmethod
    def mc_estimate(self, model, num_sample_seq, num_int_samples):
        pass

    @abstractmethod
    def trap_estimate(self, model, num_sample_seq, num_int_samples):
        pass

    @abstractmethod
    def proposal_dist_sample(self, model, num_samples=1):
        pass

class TemporalMarkQuery(Query):

  def __init__(self, time_boundaries, mark_restrictions, device=torch.device('cpu')):
    if isinstance(time_boundaries, list):
        time_boundaries = torch.FloatTensor(time_boundaries, device=device)
        
    self.time_boundaries = time_boundaries
    self.mark_restrictions = []  # list of list of restricted marks, if k is in one of these lists, then events with marks=k are not allowed in that respected time span
    for m in mark_restrictions:
      if isinstance(m, int):
        self.mark_restrictions.append([m])
      else:
        self.mark_restrictions.append(m)

    self.max_time = max(time_boundaries)
    self.num_times = len(time_boundaries)
    assert(time_boundaries[0] > 0)
    assert((time_boundaries[:-1] < time_boundaries[1:]).all())  # strictly increasing
    assert(len(time_boundaries) == len(mark_restrictions))

  def naive_estimate(self, model, num_sample_seq):
    res = 0.0  # for a sample to count as 1 in the average, it must respect _every_ mark restriction
    for _ in range(num_sample_seq): 
      times, marks = model.sample(
          right_window=self.max_time, 
          left_window=0, 
          length_limit=None, 
          sampled_times=None, 
          sampled_marks=None, 
          mark_mask=1.0,
      )
      for i in range(self.num_times):
        if i == 0:
          a, b = 0.0, self.time_boundaries[0]
        else:
          a, b = self.time_boundaries[i-1], self.time_boundaries[i]

        if np.isin(marks[(a < times) & (times <= b)], self.mark_restrictions[i]).any():
          break
      else:  # Executes if for loop did not break
        res += 1. / num_sample_seq
    return res

  def proposal_dist_sample(self, model):
    last_t = 0.0
    times, marks = None, None
    for i,t in enumerate(self.time_boundaries):
      mark_mask = np.ones((model.marks,)).astype(float)
      mark_mask[self.mark_restrictions[i]] = 0.0
      times, marks = model.sample(
          right_window=t, 
          left_window=last_t, 
          length_limit=None, 
          sampled_times=times, 
          sampled_marks=marks, 
          mark_mask=mark_mask,
      )
    return times, marks

  def estimate(self, model, num_sample_seqs, num_int_samples):
    restricted_positions = np.array([len(m) > 0 for m in self.mark_restrictions])  # If true, then we will need to integrate for events in that position
    time_spans = np.ediff1d(self.time_boundaries, to_begin=self.time_boundaries[0])
    time_norm = time_spans[restricted_positions].sum()  # Used to scale how many integration sample points each interval uses
    est, est_sq = 0.0, 0.0
    for _ in range(num_sample_seqs):
      times, marks = self.proposal_dist_sample(model)
      total_int = 0.0
      for i in range(len(time_spans)):
        if not restricted_positions[i]:
          continue

        if i == 0:
          a,b = 0.0, self.time_boundaries[0]
        else:
          a,b = self.time_boundaries[i-1], self.time_boundaries[i]
        single_int = model.compensator(
            a, 
            b, 
            sampled_times=times, 
            sampled_marks=marks, 
            num_samples=int(num_int_samples*time_spans[i]/time_norm),
        )
        total_int += single_int[self.mark_restrictions[i]].sum()

      est += np.exp(-total_int) / num_sample_seqs
      est_sq += np.exp(-total_int)**2 / num_sample_seqs
    
    return est, (est - est**2) / (est_sq - est**2)


class PositionalMarkQuery(Query):

  def __init__(self, mark_restrictions):
    self.mark_restrictions = []  # list of list of restricted marks, if k is in one of these lists, then events with marks=k are not allowed in that respected time span
    for m in mark_restrictions:
      if isinstance(m, int):
        self.mark_restrictions.append([m])
      else:
        self.mark_restrictions.append(m)
    self.max_events = len(mark_restrictions)

  def naive_estimate(self, model, num_sample_seq):
    mark_res_array = np.zeros((self.max_events, model.marks)).astype(int)
    for i,m in zip(range(self.max_events), self.mark_restrictions):
      mark_res_array[i,m] = 1

    res = 0.0  # for a sample to count as 1 in the average, it must respect _every_ mark restriction
    for _ in range(num_sample_seq): 
      times, marks = model.sample(
          right_window=None, 
          left_window=0, 
          length_limit=self.max_events, 
          sampled_times=None, 
          sampled_marks=None, 
          mark_mask=1.0,
      )
      if mark_res_array[np.arange(self.max_events), marks].sum() == 0:
        res += 1. / num_sample_seq
    return res

  def proposal_dist_sample(self, model):
    last_t = 0.0
    times, marks = None, None
    for i in range(self.max_events):
      mark_mask = np.ones((model.marks,)).astype(float)
      mark_mask[self.mark_restrictions[i]] = 0.0
      times, marks = model.sample(
          right_window=None, 
          left_window=times[-1] if i > 0 else 0.0, 
          length_limit=i+1, 
          sampled_times=times, 
          sampled_marks=marks, 
          mark_mask=mark_mask,
      )
    return times, marks

  def estimate(self, model, num_sample_seqs, num_int_samples):
    restricted_positions = np.array([len(m) > 0 for m in self.mark_restrictions])  # If true, then we will need to integrate for events in that position
    est, est_sq = 0.0, 0.0
    for _ in range(num_sample_seqs):
      times, marks = self.proposal_dist_sample(model)
      time_spans = np.ediff1d(times, to_begin=times[0])
      time_norm = time_spans[restricted_positions].sum()  # Used to scale how many integration sample points each interval uses

      total_int = 0.0
      for i in range(len(times)):
        if not restricted_positions[i]:
          continue

        if i == 0:
          a,b = 0.0, times[0]
        else:
          a,b = times[i-1], times[i]
        single_int = model.compensator(
            a, 
            b, 
            sampled_times=times, 
            sampled_marks=marks, 
            num_samples=int(num_int_samples*time_spans[i]/time_norm),
        )
        total_int += single_int[self.mark_restrictions[i]].sum()

      est += np.exp(-total_int) / num_sample_seqs
      est_sq += np.exp(-total_int)**2 / num_sample_seqs

    return est, (est - est**2) / (est_sq - est**2)