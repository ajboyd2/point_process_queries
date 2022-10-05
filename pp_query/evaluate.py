"""
Command-line utility for visualizing a model's outputs
"""
# import matplotlib.pyplot as plt
# import seaborn as sns
from mmap import ACCESS_COPY
import numpy as np
import os
import pickle
import torch
from torch.utils.data import DataLoader
import math
from collections import defaultdict
from tqdm import tqdm
import time
import datetime

from pp_query.utils import print_log
from pp_query.models import get_model
from pp_query.train import load_checkpoint, setup_model_and_optim, set_random_seed, get_data
from pp_query.arguments import get_args
from pp_query.query import PositionalMarkQuery, TemporalMarkQuery, CensoredPP, UnbiasedHittingTimeQuery, MarginalMarkQuery

def save_results(args, results, suffix=""):
    fp = args.checkpoint_path.rstrip("/")
    if args.hitting_time_queries:
        fp += "/hitting_time_queries"
    elif args.marginal_mark_queries:
        fp += "/marginal_queries"
    elif args.a_before_b_queries:
        fp += "/a_before_b_queries"

    folders = fp.split("/")
    for i in range(len(folders)):
        if folders[i] == "":
            continue
        intermediate_path = "/".join(folders[:i+1])
        if not os.path.exists(intermediate_path):
            os.mkdir(intermediate_path)
    
    fp += "/results{}{}.pickle".format("_" if suffix != "" else "", suffix)
    with open(fp, "wb") as f:
        pickle.dump(results, f)
    print_log("Saved results at {}".format(fp))

def _setup_hitting_time_query(args, batch, model, num_seqs, use_tqdm=False):
    if args.cuda:
        batch = {k:v.cuda(torch.cuda.current_device()) for k,v in batch.items()}

    times, marks = batch["tgt_times"], batch["tgt_marks"]
    length = times.numel()
    to_condition_on = min(length-1, 5)  # length // 2
    next_time = times[0, to_condition_on].item()
    next_mark = marks[0, to_condition_on].item()
    times, marks = times[..., :to_condition_on], marks[..., :to_condition_on]
    max_time = times.max().item()
    up_to = max(min((next_time-max_time)*10, 1.0), 1e-5)

    tqm = UnbiasedHittingTimeQuery(up_to=up_to, hitting_marks=next_mark, batch_size=128, device=args.device, use_tqdm=use_tqdm)
    cpp = CensoredPP(
        base_process=model, 
        observed_marks=[next_mark], 
        num_sampled_sequences=num_seqs, 
        use_same_seqs_for_ratio=False,
        batch_size=128,
        device=args.device,
        use_tqdm=use_tqdm,
    )
    same_cpp = CensoredPP(
        base_process=model, 
        observed_marks=[next_mark], 
        num_sampled_sequences=num_seqs, 
        use_same_seqs_for_ratio=True,
        batch_size=128,
        device=args.device,
        use_tqdm=use_tqdm,
    )

    return times, marks, max_time, up_to, tqm, cpp, same_cpp

def _hitting_time_queries(args, model, dataloader, num_seqs, num_int_pts):
    results = {
        "is_est": [],
        "naive_est": [],
        "censored_est": [],
        "censored_same_est": [],
        "naive_var": [],
        "is_var": [],
        "rel_eff": [],
        "avg_is_time": 0.0,
        "avg_naive_time": 0.0,
        "avg_censored_time": 0.0,
        "avg_censored_same_time": 0.0,
    }
    if args.calculate_is_bounds:
        results["is_lower"] = []
        results["is_upper"] = []
    
    num_queries = min(args.num_queries, len(dataloader))

    print_log("Estimating {} Total Queries using {} Sequences and {} Integration Points Each".format(num_queries, num_seqs, num_int_pts))
    dl_iter = iter(dataloader)
    for i in tqdm(range(num_queries)):
        batch = next(dl_iter)
        if i == num_queries:
            break

        times, marks, max_time, up_to, tqm, cpp, same_cpp = _setup_hitting_time_query(args, batch, model, num_seqs)

        is_t0 = time.perf_counter()
        is_res = tqm.estimate(model, num_seqs, num_int_pts, conditional_times=times, conditional_marks=marks, calculate_bounds=args.calculate_is_bounds)
        is_t1 = time.perf_counter()
        is_res = {k:v.item() for k,v in is_res.items()}

        naive_t0 = time.perf_counter()
        naive_res = tqm.naive_estimate(model, num_seqs, conditional_times=times, conditional_marks=marks)
        naive_t1 = time.perf_counter()
        
        c_t0 = time.perf_counter()
        censored_res = 1 - torch.exp(-cpp.compensator(max_time, max_time+up_to,  conditional_times=times, conditional_marks=marks, num_samples=num_int_pts, cond_seqs=None, censoring_start_time=max_time).sum()).item()
        c_t1 = time.perf_counter()

        sc_t0 = time.perf_counter()
        same_censored_res = 1 - torch.exp(-same_cpp.compensator(max_time, max_time+up_to,  conditional_times=times, conditional_marks=marks, num_samples=num_int_pts, cond_seqs=None, censoring_start_time=max_time).sum()).item()
        sc_t1 = time.perf_counter()
        
        results["is_est"].append(is_res["est"])
        results["naive_var"].append(is_res["naive_var"])
        results["is_var"].append(is_res["is_var"])
        results["rel_eff"].append(is_res["rel_eff"])
        results["naive_est"].append(naive_res)
        results["censored_est"].append(censored_res)
        results["censored_same_est"].append(same_censored_res)

        if args.calculate_is_bounds:
            results["is_lower"].append(is_res["lower_est"])
            results["is_upper"].append(is_res["upper_est"])

        results["avg_is_time"] += (is_t1 - is_t0) / num_queries
        results["avg_naive_time"] += (naive_t1 - naive_t0) / num_queries
        results["avg_censored_time"] += (c_t1 - c_t0) / num_queries
        results["avg_censored_same_time"] += (sc_t1 - sc_t0) / num_queries

    return results

def _hitting_time_queries_gt(args, model, dataloader):
    num_seqs = args.gt_num_seqs
    num_int_pts = args.gt_num_int_pts
    num_queries = min(args.num_queries, len(dataloader))
    gts = []
    effs = []
    dl_iter = iter(dataloader)
    for _ in tqdm(range(num_queries)):
        batch = next(dl_iter)
        times, marks, _, _, tqm, _, _ = _setup_hitting_time_query(args, batch, model, num_seqs, use_tqdm=False)
        is_res = tqm.estimate(model, num_seqs, num_int_pts, conditional_times=times, conditional_marks=marks, calculate_bounds=False)
        gts.append(is_res["est"].item())
        effs.append(is_res["rel_eff"].item())

    return gts, effs

def hitting_time_queries(args, model, dataloader):
    file_suffix = datetime.datetime.now().strftime('%m_%d_%Y_%H_%M_%S')
    seed = args.seed
    results = {"estimates": {}, "gt": None, "gt_eff": None}

    args.seed = seed + len(args.num_seqs)*len(args.num_int_pts) + 1
    set_random_seed(args)
    results["gt"], results["gt_eff"] = _hitting_time_queries_gt(args, model, dataloader)
    save_results(args, results, file_suffix)

    if not args.just_gt:
        for i, num_seqs in enumerate(args.num_seqs):
            ns_key = "num_seqs_{}".format(num_seqs)
            results["estimates"][ns_key] = {}
            for j, num_int_pts in enumerate(args.num_int_pts):
                np_key = "num_int_pts_{}".format(num_int_pts)
                args.seed = seed + i*len(args.num_int_pts) + j
                set_random_seed(args)

                results["estimates"][ns_key][np_key] = _hitting_time_queries(args, model, dataloader, num_seqs, num_int_pts)
                save_results(args, results, file_suffix)
    
    save_results(args, results, file_suffix)
    return results


def _setup_marginal_mark_query(args, batch, model, num_seqs, use_tqdm=False):
    if args.cuda:
        batch = {k:v.cuda(torch.cuda.current_device()) for k,v in batch.items()}

    times, marks = batch["tgt_times"], batch["tgt_marks"]
    length = times.numel()
    to_condition_on = min(length-1, 5)  # length // 2
    
    marks_of_interest = torch.unique(marks[0, ...])
    accepted_marks = torch.rand(marks_of_interest.shape).to(marks.device)
    accepted_marks = accepted_marks <= max(0.3, accepted_marks.min())  # Ensures at least one mark will be accepted
    accepted_marks = marks_of_interest[accepted_marks]

    times, marks = times[..., :to_condition_on], marks[..., :to_condition_on]
    
    mmq = MarginalMarkQuery(args.marg_query_n, accepted_marks, total_marks=model.num_channels, batch_size=128, device=args.device, use_tqdm=use_tqdm)

    return times, marks, mmq

def _marginal_mark_queries(args, model, dataloader, num_seqs, num_int_pts):
    results = {
        "is_est": [],
        "naive_est": [],
        "naive_var": [],
        "is_var": [],
        "rel_eff": [],
        "avg_is_time": 0.0,
        "avg_naive_time": 0.0,
    }
    if args.calculate_is_bounds:
        results["is_lower"] = []
        results["is_upper"] = []
    
    num_queries = min(args.num_queries, len(dataloader))

    print_log("Estimating {} Total Queries using {} Sequences and {} Integration Points Each".format(num_queries, num_seqs, num_int_pts))
    dl_iter = iter(dataloader)
    for i in tqdm(range(num_queries)):
        batch = next(dl_iter)
        if i == num_queries:
            break

        times, marks, mmq = _setup_marginal_mark_query(args, batch, model, num_seqs)

        is_t0 = time.perf_counter()
        is_res = mmq.estimate(model, num_seqs, num_int_pts, conditional_times=times, conditional_marks=marks, calculate_bounds=args.calculate_is_bounds)
        is_t1 = time.perf_counter()
        is_res = {k:v.item() for k,v in is_res.items()}

        naive_t0 = time.perf_counter()
        naive_res = mmq.naive_estimate(model, num_seqs, conditional_times=times, conditional_marks=marks)
        naive_t1 = time.perf_counter()
        
        results["is_est"].append(is_res["est"])
        results["naive_var"].append(is_res["naive_var"])
        results["is_var"].append(is_res["is_var"])
        results["rel_eff"].append(is_res["rel_eff"])
        results["naive_est"].append(naive_res)

        if args.calculate_is_bounds:
            results["is_lower"].append(is_res["lower_est"])
            results["is_upper"].append(is_res["upper_est"])

        results["avg_is_time"] += (is_t1 - is_t0) / num_queries
        results["avg_naive_time"] += (naive_t1 - naive_t0) / num_queries

    return results

def _marginal_mark_queries_gt(args, model, dataloader):
    num_seqs = args.gt_num_seqs
    num_int_pts = args.gt_num_int_pts
    num_queries = min(args.num_queries, len(dataloader))
    gts = []
    effs = []
    dl_iter = iter(dataloader)
    for _ in tqdm(range(num_queries)):
        batch = next(dl_iter)
        times, marks, mmq = _setup_marginal_mark_query(args, batch, model, num_seqs, use_tqdm=False)
        is_res = mmq.estimate(model, num_seqs, num_int_pts, conditional_times=times, conditional_marks=marks, calculate_bounds=False)
        gts.append(is_res["est"].item())
        effs.append(is_res["rel_eff"].item())

    return gts, effs

def marginal_mark_queries(args, model, dataloader):
    file_suffix = datetime.datetime.now().strftime('%m_%d_%Y_%H_%M_%S')
    seed = args.seed
    results = {"estimates": {}, "gt": None, "gt_eff": None, "nth_marginal": args.marg_query_n}

    args.seed = seed + len(args.num_seqs)*len(args.num_int_pts) + 1
    set_random_seed(args)
    results["gt"], results["gt_eff"] = _marginal_mark_queries_gt(args, model, dataloader)
    save_results(args, results, file_suffix)

    if not args.just_gt:
        for i, num_seqs in enumerate(args.num_seqs):
            ns_key = "num_seqs_{}".format(num_seqs)
            results["estimates"][ns_key] = {}
            for j, num_int_pts in enumerate(args.num_int_pts):
                np_key = "num_int_pts_{}".format(num_int_pts)
                args.seed = seed + i*len(args.num_int_pts) + j
                set_random_seed(args)

                results["estimates"][ns_key][np_key] = _marginal_mark_queries(args, model, dataloader, num_seqs, num_int_pts)
                save_results(args, results, file_suffix)
    
    save_results(args, results, file_suffix)
    return results

def a_before_b_queries(args, model, dataloader):
    pass    


def main():
    print_log("Getting arguments.")
    args = get_args()

    args.evaluate = True
    args.top_k = 0
    args.top_p = 0
    args.batch_size = 1
    args.shuffle = False

    print_log("Setting seed.")
    seed = args.seed
    set_random_seed(args)

    print_log("Setting up dataloaders.")
    args.pin_test_memory = True
    # train_dataloader contains the right data for most tasks
    train_dataloader, valid_dataloader, test_dataloader = get_data(args)
    
    print_log("Setting up model, optimizer, and learning rate scheduler.")
    model, _, _ = setup_model_and_optim(args, len(train_dataloader))

    load_checkpoint(args, model)
    model.eval()

    print_log("")
    print_log("")
    print_log("Commencing Experiments")
    with torch.no_grad():
        if args.hitting_time_queries:
            hitting_time_queries(args, model, valid_dataloader)
        elif args.marginal_mark_queries:
            marginal_mark_queries(args, model, valid_dataloader)
        elif args.a_before_b_queries:
            a_before_b_queries(args, model, valid_dataloader)
    


if __name__ == "__main__":
    main()
