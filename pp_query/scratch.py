from itertools import filterfalse
import torch
import torch.nn.functional as F
import math

from tqdm import tqdm
from time import perf_counter

from pp_query.models import get_model
from pp_query.train import load_checkpoint, setup_model_and_optim, set_random_seed, get_data
from pp_query.arguments import get_args
from pp_query.query import PositionalMarkQuery, TemporalMarkQuery, UnbiasedHittingTimeQuery, MarginalMarkQuery, ABeforeBQuery
from pp_query.censor import CensoredPP, CensoredTimeline

set_random_seed(seed=123)

args = get_args()
args.cuda = True    
args.device = torch.device("cuda:0") #cpu")
# args.device = torch.device("cpu")
# args.cuda = False    
# args.device = torch.device("cpu")

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
#     num_channels=6,
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
# m = get_model(
#     channel_embedding_size=10,
#     num_channels=10,
#     dec_recurrent_hidden_size=10,
#     dec_num_recurrent_layers=1,
#     dec_intensity_hidden_size=10,
#     dec_intensity_factored_heads=False,
#     dec_num_intensity_layers=1,
#     dec_intensity_use_embeddings=False,
#     dec_act_func="gelu",
#     dropout=0.0,
#     hawkes=False,
#     hawkes_bounded=False,
#     neural_hawkes=True,
#     rmtpp=False,
# )

m.eval()
print(m)
print()
M = m.num_channels

censor_mark_pct = 0.5
# batch = next(iter(dl))

results = {
    "original": [],
    "naive": [],
    "baseline": [],
    "censored": [],
}

import pickle
# results = pickle.load(open("/home/alexjb/source/point_process_queries/pp_query/censor_experiment_results.pickle", "rb"))

# results = {k:torch.tensor(v) for k,v in results.items()}
# avg_results = {k:v.mean() for k,v in results.items()}
# rel_results = {k:((results["original"] - v) / results["original"]).mean() for k,v in results.items()}
# # rel_results = {k:((results["original"] - v) ).mean() for k,v in results.items()}

# from pprint import pprint
# print("Avg Results:")
# pprint({k:v.item() for k,v in avg_results.items()})
# print("\nRel. Avg Results:")
# pprint({k:v.item() for k,v in rel_results.items()})

# 1/0;

pickle.dump(results, open("/home/alexjb/source/point_process_queries/pp_query/censor_experiment_results2.pickle", "wb"))

for i, batch in enumerate(dl):
    if len(results["original"]) >= 20:
        break
    if args.cuda:
         batch = {k:v.to(args.device) for k,v in batch.items()}
    print(i, len(results["original"]))
    original_times, original_marks, T = batch["tgt_times"], batch["tgt_marks"], batch["T"].item()
    if T > 3.0:
        continue

    unique_marks = torch.unique(original_marks.squeeze(0))
    num_unique = unique_marks.numel()
    if num_unique <= 1:
        continue
    marks_to_censor = sorted(unique_marks[torch.randperm(num_unique)[:math.floor(num_unique*censor_mark_pct)]].tolist())
    censoring = CensoredTimeline(
        boundaries=(0.0, T), 
        # censored_marks=[list(range(100, M)), list(range(50, M)), list(range(3, M))], 
        # censored_marks=[list(range(5, M)), list(range(3, M)), list(range(2, M))], 
        censored_marks=marks_to_censor, 
        total_marks=M, 
        relative_start=False, 
        device=args.device,
    )

    # print(T, original_times)
    # print()
    # print(original_marks)
    # print()

    cpp = CensoredPP(m, censoring, num_sampled_sequences=3, use_same_seqs_for_ratio=True, proposal_batch_size=1024, num_integral_pts=1024)
    res = cpp.ll_pred_experiment_pass(uncensored_times=original_times, uncensored_marks=original_marks, T=T)
    for k,v in results.items():
        v.append(res[k]["log_likelihood"].item())

import pickle
pickle.dump(results, open("/home/alexjb/source/point_process_queries/pp_query/censor_experiment_results2.pickle", "wb"))

results = {k:torch.tensor(v) for k,v in results.items()}
avg_results = {k:v.mean() for k,v in results.items()}
rel_results = {k:((results["original"] - v).abs() / results["original"]).mean() for k,v in results.items()}

from pprint import pprint
print("Avg Results:")
pprint({k:v.item() for k,v in avg_results.items()})
print("\nRel. Avg Results:")
pprint({k:v.item() for k,v in rel_results.items()})

# times = torch.tensor([0.1, 0.7, 0.9, 1.3, 1.8, 2.3]).unsqueeze(0)
# marks = torch.tensor([  0,   0,   1,   2,   2,   2]).unsqueeze(0)
# T = torch.tensor([2.5])

# censoring = CensoredTimeline(
#     # boundaries=[(0.3, 1.0), (1.0, 1.5), (2.0, 3.0)], 
#     boundaries=[(0.0, 2.0), (2.0, 4.0)], 
#     # censored_marks=[list(range(100, M)), list(range(50, M)), list(range(3, M))], 
#     # censored_marks=[list(range(5, M)), list(range(3, M)), list(range(2, M))], 
#     censored_marks=[list(range(6, M)), [9]], 
#     total_marks=M, 
#     relative_start=False, 
#     device=args.device,
# )
# print(censoring)
# print(times)
# print(censoring.get_mask(times))
# masks = censoring.get_mask(times)
# print({k: v for k,v in m.get_intensity(None, times, times, None, marks, censoring=censoring).items()})

# cpp = CensoredPP(m, censoring, 10, True, 1024, 1024)
# intensity = cpp.intensity(times.squeeze(0), marks.squeeze(0), times.squeeze(0), True)
# print()
# print()
# print(intensity)
# print()
# print()

# st, sm, ss = m.batch_sample_points(
#     None, 
#     None, 
#     dominating_rate=100., 
#     T=2.0, 
#     left_window=0.0, 
#     length_limit=float('inf'), 
#     mark_mask=1.0,  #TODO: Make default value None instead of 1.0
#     top_k=0, 
#     top_p=0.0, 
#     num_samples=1, 
#     proposal_batch_size=1024, 
#     mask_dict=None, 
#     adapt_dom_rate=True,
#     stop_marks=None,
#     censoring=None,
# )
# st, sm, ss = st[0], sm[0], ss[0]
# # x = torch.linspace(0, 2.0, 1000).unsqueeze(0)
# # print(st, sm)
# # pos_ints = m.get_intensity(ss, st, st, sm)["log_mark_intensity"].sum()
# # neg_ints = m.get_intensity(ss, st, x)["total_intensity"]
# # neg_ints = torch.trapezoid(neg_ints, x, dim=-1)
# # print("Unchanged LL:", (pos_ints - neg_ints).item())

# # ct, cm = st[sm < 6].unsqueeze(0), sm[sm < 6].unsqueeze(0)
# # pos_ints = m.get_intensity(None, ct, ct, cm, cm)["log_mark_intensity"].sum()
# # neg_ints = m.get_intensity(None, ct, x, None, cm)["total_intensity"]
# # neg_ints = torch.trapezoid(neg_ints, x, dim=-1)
# # print("Naive LL:", (pos_ints - neg_ints).item())

# # pos_ints = m.get_intensity(None, ct, ct, cm, cm)["log_mark_intensity"].sum()
# # neg_ints = m.get_intensity(None, ct, x, None, cm)["all_mark_intensities"][..., :6].sum(dim=-1)
# # neg_ints = torch.trapezoid(neg_ints, x, dim=-1)
# # print("Baseline LL:", (pos_ints - neg_ints).item())

# # print("Censored LL:", cpp.log_likelihood(ct.squeeze(0), cm.squeeze(0), 2.0)["log_likelihood"].item())

# print(cpp.ll_pred_experiment_pass(uncensored_times=st, uncensored_marks=sm, T=2.0))
