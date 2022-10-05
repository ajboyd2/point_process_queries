from itertools import filterfalse
import torch
import torch.nn.functional as F

from tqdm import tqdm
from time import perf_counter

from pp_query.models import get_model
from pp_query.train import load_checkpoint, setup_model_and_optim, set_random_seed, get_data
from pp_query.arguments import get_args
from pp_query.query import PositionalMarkQuery, TemporalMarkQuery, CensoredPP, UnbiasedHittingTimeQuery, MarginalMarkQuery

set_random_seed(seed=12321)

args = get_args()
args.device = torch.device("cpu")

args.train_data_percentage = 1.0
args.num_channels = 182
args.channel_embedding_size = 32
args.dec_recurrent_hidden_size = 64
args.dec_num_recurrent_layers = 1
args.dec_intensity_hidden_size = 32
args.dec_num_intensity_layers = 2
args.dec_act_func = "gelu"
args.dropout = 0.01
args.checkpoint_path = "./data/movie/nhp_models/"
args.train_epochs = 100
args.num_workers = 2
args.batch_size = 128
args.log_interval = 500
args.save_epochs = 5
args.optimizer = "adam"
args.grad_clip = 10000.0
args.lr = 0.001
args.weight_decay = 0.0
args.warmup_pct = 0.01
args.lr_decay_style = "constant"
args.train_data_path = ["./data/movie/train/"]
args.valid_data_path = ["./data/movie/valid/"]
args.test_data_path = ["./data/movie/test/"]
args.neural_hawkes = True


# Load data first as this determines the number of total marks
args.batch_size = 1
dl, _, _ = get_data(args)

m, _, _ = setup_model_and_optim(args, 1000)
load_checkpoint(args, m)

# m = get_model(
#     channel_embedding_size=10,
#     num_channels=4,
#     dec_recurrent_hidden_size=10,
#     dec_num_recurrent_layers=1,
#     dec_intensity_hidden_size=10,
#     dec_intensity_factored_heads=False,
#     dec_num_intensity_layers=1,
#     dec_intensity_use_embeddings=False,
#     dec_act_func="gelu",
#     dropout=0.0,
#     hawkes=True,
#     hawkes_bounded=True,
#     neural_hawkes=False,
#     rmtpp=False,
# )

m.eval()

print(m)

n_seqs = 1000
n_pts = 1000
b = up_to = 0.25
times, marks = torch.FloatTensor([[0.3, 0.8]]), torch.LongTensor([[0, 0]])
next_mark = 2
max_time = offset = 0.8

batch = next(iter(dl))
if args.cuda:
    batch = {k:v.cuda(torch.cuda.current_device()) for k,v in batch.items()}

times, marks = batch["tgt_times"], batch["tgt_marks"]
length = times.numel()
next_time = times[0, (length//3)].item()
next_mark = marks[0, (length//3)].item()
times, marks = times[..., :length//3], marks[..., :length//3]
max_time = times.max().item()
up_to = 0.1 #min((next_time-max_time)*10, 1.0)

tmq = UnbiasedHittingTimeQuery(up_to, [next_mark], batch_size=128, device=args.device)
cpp = CensoredPP(
    base_process=m, 
    observed_marks=[next_mark], 
    num_sampled_sequences=n_seqs, 
    use_same_seqs_for_ratio=False,
    batch_size=128,
    device=args.device,
)

scpp = CensoredPP(
    base_process=m, 
    observed_marks=[next_mark], 
    num_sampled_sequences=n_seqs, 
    use_same_seqs_for_ratio=True,
    batch_size=128,
    device=args.device,
)
print(times, marks)
print()

'''

# ts = torch.linspace(offset, offset+up_to, n_pts).unsqueeze(0)
# print(ts.min(), ts.max())
# mark_mask = torch.zeros((m.num_channels,), dtype=torch.float32)
# mark_mask[next_mark] = 1.0
# intensities = []
# with torch.no_grad():
#     for _ in tqdm(range(n_seqs)):
#         sample_times, sample_marks = tmq.proposal_dist_sample(m, conditional_times=times, conditional_marks=marks, offset=offset)
#         intensities.append(m.get_intensity(
#             state_values=None, 
#             state_times=sample_times.unsqueeze(0), 
#             timestamps=ts, 
#             marks=None, 
#             state_marks=sample_marks.unsqueeze(0), 
#             mark_mask=mark_mask,
#         )["all_log_mark_intensities"].exp().sum(dim=-1).squeeze(0))
# ts = ts.squeeze(0)
# unbiased_ests = []
# for intensity in intensities:
#     unbiased_ests.append(torch.exp(-torch.trapezoid(intensity, x=ts, dim=-1)).item())
# unbiased_est = sum(unbiased_ests) / (n_seqs)
# unbiased_est_var = sum((ue - unbiased_est)**2 for ue in unbiased_ests) / (n_seqs-1)
# print("Unbiased est        ", 1-unbiased_est, "Sample Var", unbiased_est_var, "Est Var", unbiased_est_var / n_seqs, "Eff", (unbiased_est*(1-unbiased_est)) / unbiased_est_var)
# ts = ts.unsqueeze(0)
# intensities = []
# with torch.no_grad():
#     for _ in tqdm(range(n_seqs)):
#         sample_times, sample_marks = tmq.proposal_dist_sample(m, conditional_times=times, conditional_marks=marks, offset=offset)
#         intensities.append(m.get_intensity(
#             state_values=None, 
#             state_times=sample_times.unsqueeze(0), 
#             timestamps=ts, 
#             marks=None, 
#             state_marks=sample_marks.unsqueeze(0), 
#             mark_mask=mark_mask,
#         )["all_log_mark_intensities"].exp().sum(dim=-1).squeeze(0))
# ts = ts.squeeze(0)
# numers, denoms = [], []
# for intensity in intensities:
#     cumulative_intensity = torch.exp(-F.pad(torch.cumulative_trapezoid(intensity, x=ts, dim=-1), (1,0), 'constant', 0.0))
#     numers.append(intensity * cumulative_intensity)
#     denoms.append(cumulative_intensity)
# numers, denoms = torch.stack(numers, dim=0), torch.stack(denoms, dim=0)
# censored_intensity = numers.sum(dim=0) / denoms.sum(dim=0)
# biased_est = torch.exp(-torch.trapezoid(censored_intensity, x=ts, dim=-1))
# print("Same Seqs Biased est", 1-biased_est.item())

# ts = ts.unsqueeze(0)
# intensities = []
# with torch.no_grad():
#     for _ in tqdm(range(n_seqs)):
#         sample_times, sample_marks = tmq.proposal_dist_sample(m, conditional_times=times, conditional_marks=marks, offset=offset)
#         intensities.append(m.get_intensity(
#             state_values=None, 
#             state_times=sample_times.unsqueeze(0), 
#             timestamps=ts, 
#             marks=None, 
#             state_marks=sample_marks.unsqueeze(0), 
#             mark_mask=mark_mask,
#         )["all_log_mark_intensities"].exp().sum(dim=-1).squeeze(0))
# ts = ts.squeeze(0)
# numers, denoms = [], []
# for numer_intensity, denom_intensity in zip(intensities[:n_seqs//2], intensities[n_seqs//2:]):
#     numer_cumulative_intensity = torch.exp(-F.pad(torch.cumulative_trapezoid(numer_intensity, x=ts, dim=-1), (1,0), 'constant', 0.0))
#     denom_cumulative_intensity = torch.exp(-F.pad(torch.cumulative_trapezoid(denom_intensity, x=ts, dim=-1), (1,0), 'constant', 0.0))
#     numers.append(numer_intensity * numer_cumulative_intensity)
#     denoms.append(denom_cumulative_intensity)
# numers, denoms = torch.stack(numers, dim=0), torch.stack(denoms, dim=0)
# censored_intensity = numers.sum(dim=0) / denoms.sum(dim=0)
# biased_est = torch.exp(-torch.trapezoid(censored_intensity, x=ts, dim=-1))
# print("Diff Seqs Biased est", 1-biased_est.item())

# ts = ts.unsqueeze(0)
# intensities = []
# with torch.no_grad():
#     for _ in tqdm(range(n_seqs*2)):
#         sample_times, sample_marks = tmq.proposal_dist_sample(m, conditional_times=times, conditional_marks=marks, offset=offset)
#         intensities.append(m.get_intensity(
#             state_values=None, 
#             state_times=sample_times.unsqueeze(0), 
#             timestamps=ts, 
#             marks=None, 
#             state_marks=sample_marks.unsqueeze(0), 
#             mark_mask=mark_mask,
#         )["all_log_mark_intensities"].exp().sum(dim=-1).squeeze(0))
# ts = ts.squeeze(0)

# numers, denoms = [], []
# for numer_intensity, denom_intensity in zip(intensities[:n_seqs], intensities[n_seqs:]):
#     numer_cumulative_intensity = torch.exp(-F.pad(torch.cumulative_trapezoid(numer_intensity, x=ts, dim=-1), (1,0), 'constant', 0.0))
#     denom_cumulative_intensity = torch.exp(-F.pad(torch.cumulative_trapezoid(denom_intensity, x=ts, dim=-1), (1,0), 'constant', 0.0))
#     numers.append(numer_intensity * numer_cumulative_intensity)
#     denoms.append(denom_cumulative_intensity)
# numers, denoms = torch.stack(numers, dim=0), torch.stack(denoms, dim=0)
# censored_intensity = numers.sum(dim=0) / denoms.sum(dim=0)
# biased_est = torch.exp(-torch.trapezoid(censored_intensity, x=ts, dim=-1))
# print("Diff Seqs Biased est", 1-biased_est.item())
'''



print()
t0 = perf_counter()
tn1 = tmq.naive_estimate(m, n_seqs, conditional_times=times, conditional_marks=marks)
t1 = perf_counter()
print("Tn1", (t1-t0) / n_seqs , tn1)

t0 = perf_counter()
tis = {k:v.item() for k,v in tmq.estimate(m, n_seqs, n_pts, conditional_times=times, conditional_marks=marks, calculate_bounds=True).items()}
t1 = perf_counter()
print("Tis", (t1-t0) / n_seqs , tis)

t0 = perf_counter()
tis = {k:v.item() for k,v in tmq.estimate(m, n_seqs, n_pts, conditional_times=times, conditional_marks=marks, calculate_bounds=False).items()}
t1 = perf_counter()
print("Tis", (t1-t0) / n_seqs , tis)

t0 = perf_counter()
c_est = 1-torch.exp(-cpp.compensator(max_time, max_time+up_to, conditional_times=times, conditional_marks=marks, num_samples=n_pts, cond_seqs=None, censoring_start_time=max_time).sum()).item()
t1 = perf_counter()
print("Tcd", (t1-t0) / n_seqs, c_est)

t0 = perf_counter()
sc_est = 1-torch.exp(-scpp.compensator(max_time, max_time+up_to, conditional_times=times, conditional_marks=marks, num_samples=n_pts, cond_seqs=None, censoring_start_time=max_time).sum()).item()
t1 = perf_counter()
print("Tcs", (t1-t0) / n_seqs, sc_est)


t0 = perf_counter()
naive_est = 0.0
sample_times, sample_marks, _ = m.batch_sample_points(
    left_window=0 if times is None else times.max(), 
    T=up_to + (0 if times is None else times.max()), 
    timestamps=times, 
    marks=marks, 
    mark_mask=1.0,
    proposal_batch_size=1024,
    num_samples=n_seqs,
)
for sample_mark in sample_marks:
    naive_est += (sample_mark == next_mark).any(dim=-1).sum(dim=0).item() / (n_seqs)
t1 = perf_counter()
print("Tn2",  (t1-t0) / n_seqs, naive_est)


t0 = perf_counter()
naive_est = 0.0
for i in tqdm(range(n_seqs)):
    sample_times, sample_marks = m.sample_points(
        left_window=0 if times is None else times.max(), 
        T=up_to + (0 if times is None else times.max()), 
        timestamps=times, 
        marks=marks, 
        mark_mask=1.0,
        proposal_batch_size=1024,
    )
    sample_mark = sample_marks.squeeze(0)[0 if times is None else times.numel():]
    if (sample_mark == next_mark).any():
        naive_est += 1. / n_seqs
t1 = perf_counter()
print("Tn3",  (t1-t0) / n_seqs, naive_est)

1/0

unique_marks = set(batch["tgt_marks"].squeeze().cpu().tolist())

#pmq = MarginalMarkQuery(n=1, marks_of_interest=[u for i,u in enumerate(unique_marks) if i%2==0], batch_size=128, device=args.device)  # Choose marginal set to be half of the unique marks in the sequence
A = [u for i,u in enumerate(unique_marks) if i%2==0]  #[0]
pmq = MarginalMarkQuery(n=3, marks_of_interest=A, total_marks=m.num_channels, batch_size=128, device=args.device)  # Choose marginal set to be half of the unique marks in the sequence


print()
t0 = perf_counter()
pn1 = pmq.naive_estimate(m, n_seqs, conditional_times=times, conditional_marks=marks)
t1 = perf_counter()
print("Pn1", (t1-t0) / n_seqs , pn1)

print()
t0 = perf_counter()
pis = {k:v.item() for k,v in pmq.estimate(m, n_seqs, n_pts, conditional_times=times, conditional_marks=marks, calculate_bounds=True).items()}
t1 = perf_counter()
print("Pis", (t1-t0) / n_seqs , pis)

print()
t0 = perf_counter()
pis = {k:v.item() for k,v in pmq.estimate(m, n_seqs, n_pts, conditional_times=times, conditional_marks=marks, calculate_bounds=False).items()}
t1 = perf_counter()
print("Pis", (t1-t0) / n_seqs , pis)



print()
# print()
# print(1-torch.exp(-cpp.compensator(max_time, max_time+up_to, conditional_times=times, conditional_marks=marks, num_samples=n_pts//100, cond_seqs=None, censoring_start_time=max_time).sum()).item())
# print(1-torch.exp(-cpp.compensator(max_time, max_time+up_to, conditional_times=times, conditional_marks=marks, num_samples=n_pts//10, cond_seqs=None, censoring_start_time=max_time).sum()).item())


t0 = perf_counter()
naive_est = 0.0
for i in tqdm(range(n_seqs)):
    sample_times, sample_marks, _ = m.batch_sample_points(
        left_window=0 if times is None else times.max(), 
        length_limit=3 + (0 if times is None else times.numel()), 
        timestamps=times, 
        marks=marks, 
        mark_mask=1.0,
        proposal_batch_size=1024,
        num_samples=1,
    )
    sample_mark = sample_marks[0].squeeze(0)[-1].item() #][0 if times is None else times.numel():]
    if sample_mark in A:
        naive_est += 1. / n_seqs
t1 = perf_counter()
print("Pn2", (t1-t0) / n_seqs , naive_est)



print()





# results = []
# for i, batch in tqdm(enumerate(dl)):
#     if args.cuda:
#         batch = {k:v.cuda(torch.cuda.current_device()) for k,v in batch.items()}
    
#     times, marks = batch["tgt_times"], batch["tgt_marks"]
#     length = times.numel()
#     next_time = times[0, (length//2)].item()
#     next_mark = marks[0, (length//2)].item()
#     times, marks = times[..., :length//2], marks[..., :length//2]
#     max_time = times.max().item()
#     up_to = min((next_time-max_time)*10, 1.0)
#     tqm = UnbiasedHittingTimeQuery(up_to=up_to, hitting_marks=next_mark, use_tqdm=False)
#     res = tqm.estimate(m, n_seqs, n_pts, conditional_times=times, conditional_marks=marks, calculate_bounds=True)
#     results.append(res)

#     if i > 20:
#         break

# print()
# print("Results Ordered from Least Estimate to Greatest Estimate:")
# print("=================================================================================")
# results = sorted(results, key=lambda x: x["est"])
# for i, result in enumerate(results):
#     est, eff, n_var, i_var = result["est"], result["rel_eff"], result["naive_var"], result["is_var"]
#     lb, ub = result["lower_est"], result["upper_est"]
#     print("Est: {:.5f}  LB: {:.5f}  UB: {:.5f}  Eff: {:.5e}  IS Var: {:.5e}  Naive Var: {:.5e}".format(est, lb, ub, eff, i_var, n_var))


# pmq = PositionalMarkQuery([[0,1,2], [0,1], [0]])
# tmq = TemporalMarkQuery([1.0, 2.0, 4.5], [[0], [1], [2]])

# times, marks = torch.FloatTensor([[0.3, 0.8, 1.7]]), torch.LongTensor([[0, 0, 1]])

# n = 10
# print()
# print(tmq.estimate(m, n, 100, conditional_times=times, conditional_marks=marks))
# print(tmq.naive_estimate(m, n, conditional_times=times, conditional_marks=marks))
# print()
# print(pmq.naive_estimate(m, n, conditional_times=times, conditional_marks=marks))
# print(pmq.estimate(m, n, 100, conditional_times=times, conditional_marks=marks))



''' TODO: Implement upper and lower bounds
Results Ordered from Least Estimate to Greatest Estimate:
=================================================================================
Est: 0.00111    Eff: 1.17441e+03    IS Var: 9.48216e-07    Naive Var: 1.11359e-03
Est: 0.00289    Eff: 2.12860e+02    IS Var: 1.35573e-05    Naive Var: 2.88582e-03
Est: 0.00387    Eff: 2.08000e+03    IS Var: 1.85537e-06    Naive Var: 3.85916e-03
Est: 0.00899    Eff: 1.20531e+03    IS Var: 7.39104e-06    Naive Var: 8.90851e-03
Est: 0.01307    Eff: 1.86779e+04    IS Var: 6.90488e-07    Naive Var: 1.28968e-02
Est: 0.01361    Eff: 2.93389e+04    IS Var: 4.57498e-07    Naive Var: 1.34225e-02
Est: 0.01624    Eff: 2.70595e+03    IS Var: 5.90588e-06    Naive Var: 1.59810e-02
Est: 0.01782    Eff: 1.14098e+03    IS Var: 1.53384e-05    Naive Var: 1.75008e-02
Est: 0.01885    Eff: 2.73115e+04    IS Var: 6.77316e-07    Naive Var: 1.84985e-02
Est: 0.02803    Eff: 1.72375e+04    IS Var: 1.58057e-06    Naive Var: 2.72450e-02
Est: 0.03167    Eff: 3.98251e+02    IS Var: 7.70064e-05    Naive Var: 3.06678e-02
Est: 0.03265    Eff: 4.05229e+01    IS Var: 7.79317e-04    Naive Var: 3.15802e-02
Est: 0.04046    Eff: 2.71408e+03    IS Var: 1.43035e-05    Naive Var: 3.88208e-02
Est: 0.04240    Eff: 1.71101e+04    IS Var: 2.37296e-06    Naive Var: 4.06017e-02
Est: 0.04969    Eff: 2.34390e+03    IS Var: 2.01458e-05    Naive Var: 4.72197e-02
Est: 0.05053    Eff: 2.59358e+03    IS Var: 1.84992e-05    Naive Var: 4.79792e-02
Est: 0.05829    Eff: 4.19166e+03    IS Var: 1.30962e-05    Naive Var: 5.48948e-02
Est: 0.05846    Eff: 8.29023e+04    IS Var: 6.63901e-07    Naive Var: 5.50389e-02
Est: 0.05998    Eff: 4.90364e+00    IS Var: 1.14979e-02    Naive Var: 5.63816e-02
Est: 0.06253    Eff: 1.48915e+04    IS Var: 3.93648e-06    Naive Var: 5.86199e-02
Est: 0.06284    Eff: 4.67904e+02    IS Var: 1.25869e-04    Naive Var: 5.88949e-02
Est: 0.06975    Eff: 3.63179e+01    IS Var: 1.78668e-03    Naive Var: 6.48886e-02
Est: 0.07103    Eff: 8.93427e+01    IS Var: 7.38512e-04    Naive Var: 6.59807e-02
Est: 0.08779    Eff: 2.49874e+02    IS Var: 3.20504e-04    Naive Var: 8.00856e-02
Est: 0.08842    Eff: 1.19771e+03    IS Var: 6.72997e-05    Naive Var: 8.06057e-02
Est: 0.09173    Eff: 1.51891e+03    IS Var: 5.48504e-05    Naive Var: 8.33126e-02
Est: 0.09643    Eff: 7.70933e+02    IS Var: 1.13025e-04    Naive Var: 8.71344e-02
Est: 0.09803    Eff: 1.56873e+03    IS Var: 5.63661e-05    Naive Var: 8.84230e-02
Est: 0.10522    Eff: 4.41246e+01    IS Var: 2.13362e-03    Naive Var: 9.41451e-02
Est: 0.10703    Eff: 1.99406e+03    IS Var: 4.79287e-05    Naive Var: 9.55727e-02
Est: 0.11001    Eff: 1.89305e+02    IS Var: 5.17184e-04    Naive Var: 9.79055e-02
Est: 0.12404    Eff: 4.05041e+01    IS Var: 2.68258e-03    Naive Var: 1.08655e-01
Est: 0.12408    Eff: 6.90584e+03    IS Var: 1.57379e-05    Naive Var: 1.08683e-01
Est: 0.13179    Eff: 1.45339e+01    IS Var: 7.87295e-03    Naive Var: 1.14425e-01
Est: 0.15626    Eff: 7.72538e+02    IS Var: 1.70665e-04    Naive Var: 1.31845e-01
Est: 0.16596    Eff: 4.27412e+02    IS Var: 3.23850e-04    Naive Var: 1.38417e-01
Est: 0.17616    Eff: 4.22986e+01    IS Var: 3.43100e-03    Naive Var: 1.45127e-01
Est: 0.21058    Eff: 5.76782e+01    IS Var: 2.88209e-03    Naive Var: 1.66234e-01
Est: 0.25764    Eff: 3.21647e+01    IS Var: 5.94637e-03    Naive Var: 1.91263e-01
Est: 0.35417    Eff: 5.70732e+01    IS Var: 4.00772e-03    Naive Var: 2.28733e-01
Est: 0.73236    Eff: 6.41615e+00    IS Var: 3.05495e-02    Naive Var: 1.96010e-01
Est: 0.84150    Eff: 9.50077e+00    IS Var: 1.40383e-02    Naive Var: 1.33375e-01
'''
