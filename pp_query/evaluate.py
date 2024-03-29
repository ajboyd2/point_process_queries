"""
Command-line utility for visualizing a model's outputs
"""
# import matplotlib.pyplot as plt
# import seaborn as sns
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
import random

from pp_query.utils import print_log
from pp_query.models import get_model
from pp_query.train import load_checkpoint, setup_model_and_optim, set_random_seed, get_data
from pp_query.arguments import get_args
from pp_query.query import ABeforeBQuery, PositionalMarkQuery, TemporalMarkQuery, CensoredPP, UnbiasedHittingTimeQuery, MarginalMarkQuery

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

    tqm = UnbiasedHittingTimeQuery(up_to=up_to, hitting_marks=next_mark, batch_size=args.query_batch_size, device=args.device, use_tqdm=use_tqdm, proposal_batch_size=args.proposal_batch_size)
    cpp = CensoredPP(
        base_process=model, 
        observed_marks=[next_mark], 
        num_sampled_sequences=num_seqs, 
        use_same_seqs_for_ratio=False,
        batch_size=args.query_batch_size,
        device=args.device,
        use_tqdm=use_tqdm,
        proposal_batch_size=args.proposal_batch_size,
    )
    same_cpp = CensoredPP(
        base_process=model, 
        observed_marks=[next_mark], 
        num_sampled_sequences=num_seqs, 
        use_same_seqs_for_ratio=True,
        batch_size=args.query_batch_size,
        device=args.device,
        use_tqdm=use_tqdm,
        proposal_batch_size=args.proposal_batch_size,
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

def hitting_time_queries(args, model, dataloader, results):
    if args.continue_experiments is not None:
        file_suffix = args.continue_experiments.split("/results_")[-1].replace(".pickle", "")
    else:
        file_suffix = datetime.datetime.now().strftime('%m_%d_%Y_%H_%M_%S')
    seed = args.seed
    if results is None:
        results = {"estimates": {}, "gt": None, "gt_eff": None}

    args.seed = seed + len(args.num_seqs)*len(args.num_int_pts) + 1
    set_random_seed(args)

    if (results["gt"] is None) or (results["gt_eff"] is None):
        results["gt"], results["gt_eff"] = _hitting_time_queries_gt(args, model, dataloader)
        save_results(args, results, file_suffix)
    else:
        print_log("Skipping GT Estimates.")

    if not args.just_gt:
        for i, num_seqs in enumerate(args.num_seqs):
            ns_key = "num_seqs_{}".format(num_seqs)
            results["estimates"][ns_key] = {}
            for j, num_int_pts in enumerate(args.num_int_pts):
                np_key = "num_int_pts_{}".format(num_int_pts)
                args.seed = seed + i*len(args.num_int_pts) + j
                set_random_seed(args)
                if (ns_key in results["estimates"]) and (np_key in results["estimates"][ns_key]) and (results["estimates"][ns_key][np_key] is not None):
                    print_log("Skipping {} {}".format(ns_key, np_key))
                    continue
                else:
                    results["estimates"][ns_key][np_key] = _hitting_time_queries(args, model, dataloader, num_seqs, num_int_pts)
                    save_results(args, results, file_suffix)
    
    save_results(args, results, file_suffix)
    return results

def _setup_marginal_mark_query(args, batch, model, num_seqs, keep_mark_pct, use_tqdm=False):
    if args.cuda:
        batch = {k:v.cuda(torch.cuda.current_device()) for k,v in batch.items()}

    times, marks = batch["tgt_times"], batch["tgt_marks"]
    length = times.numel()
    to_condition_on = min(length-1, 5)  # length // 2
    
    marks_of_interest = torch.unique(marks[0, ...])
    # accepted_marks = torch.rand(marks_of_interest.shape).to(marks.device)
    # accepted_marks = accepted_marks <= max(0.3, accepted_marks.min())  # Ensures at least one mark will be accepted
    num_marks_to_keep = math.ceil(keep_mark_pct*marks_of_interest.numel())
    accepted_marks = marks_of_interest[:max(num_marks_to_keep, 1)]  #[accepted_marks]
    unaccepted_marks = torch.tensor([i for i in range(model.num_channels) if i not in accepted_marks])
    assert(unaccepted_marks.numel() > 0)

    times, marks = times[..., :to_condition_on], marks[..., :to_condition_on]
    
    mmq = MarginalMarkQuery(args.marg_query_n, accepted_marks, total_marks=model.num_channels, batch_size=args.query_batch_size, device=args.device, use_tqdm=use_tqdm, proposal_batch_size=args.proposal_batch_size)
    c_mmq = MarginalMarkQuery(args.marg_query_n, unaccepted_marks, total_marks=model.num_channels, batch_size=args.query_batch_size, device=args.device, use_tqdm=use_tqdm, proposal_batch_size=args.proposal_batch_size)

    return times, marks, mmq, c_mmq

def _marginal_mark_queries(args, model, dataloader, num_seqs, num_int_pts, keep_mark_pcts):
    results = {
        "is_est": [],
        "c_is_est": [],
        "naive_est": [],
        "naive_var": [],
        "is_var": [],
        "c_is_var": [],
        "rel_eff": [],
        "c_rel_eff": [],
        "avg_is_time": 0.0,
        "avg_c_is_time": 0.0,
        "avg_naive_time": 0.0,
    }
    if args.calculate_is_bounds:
        results["is_lower"] = []
        results["is_upper"] = []
        results["c_is_lower"] = []
        results["c_is_upper"] = []
    
    num_queries = min(args.num_queries, len(dataloader))

    print_log("Estimating {} Total Queries using {} Sequences and {} Integration Points Each".format(num_queries, num_seqs, num_int_pts))
    dl_iter = iter(dataloader)
    for i in tqdm(range(num_queries)):
        batch = next(dl_iter)

        if i < args.query_start_idx:
            continue

        keep_mark_pct = keep_mark_pcts[i]        

        times, marks, mmq, c_mmq = _setup_marginal_mark_query(args, batch, model, num_seqs, keep_mark_pct)

        is_t0 = time.perf_counter()
        is_res = mmq.estimate(model, num_seqs, num_int_pts, conditional_times=times, conditional_marks=marks, calculate_bounds=args.calculate_is_bounds)
        is_t1 = time.perf_counter()
        is_res = {k:v.item() for k,v in is_res.items()}

        c_is_t0 = time.perf_counter()
        c_is_res = c_mmq.estimate(model, num_seqs, num_int_pts, conditional_times=times, conditional_marks=marks, calculate_bounds=args.calculate_is_bounds)
        c_is_t1 = time.perf_counter()
        c_is_res = {k:v.item() for k,v in c_is_res.items()}

        naive_t0 = time.perf_counter()
        naive_res = mmq.naive_estimate(model, num_seqs, conditional_times=times, conditional_marks=marks)
        naive_t1 = time.perf_counter()
        
        results["is_est"].append(is_res["est"])
        results["naive_var"].append(is_res["naive_var"])
        results["is_var"].append(is_res["is_var"])
        results["rel_eff"].append(is_res["rel_eff"])
        results["naive_est"].append(naive_res)

        results["c_is_est"].append(1-c_is_res["est"])
        results["c_is_var"].append(c_is_res["is_var"])
        results["c_rel_eff"].append(c_is_res["rel_eff"])

        if args.calculate_is_bounds:
            results["is_lower"].append(is_res["lower_est"])
            results["is_upper"].append(is_res["upper_est"])
            results["c_is_lower"].append(1-c_is_res["upper_est"])
            results["c_is_upper"].append(1-c_is_res["lower_est"])

        results["avg_is_time"] += (is_t1 - is_t0) / num_queries
        results["avg_c_is_time"] += (c_is_t1 - c_is_t0) / num_queries
        results["avg_naive_time"] += (naive_t1 - naive_t0) / num_queries

    return results

def _marginal_mark_queries_gt(args, model, dataloader, keep_mark_pcts):
    num_seqs = args.gt_num_seqs
    num_int_pts = args.gt_num_int_pts
    num_queries = min(args.num_queries, len(dataloader))
    gts = []
    effs = []
    dl_iter = iter(dataloader)
    for i in tqdm(range(num_queries)):
        batch = next(dl_iter)
        
        if i < args.query_start_idx:
            continue

        keep_mark_pct = keep_mark_pcts[i]
        times, marks, mmq, _ = _setup_marginal_mark_query(args, batch, model, num_seqs, keep_mark_pct, use_tqdm=False)
        is_res = mmq.estimate(model, num_seqs, num_int_pts, conditional_times=times, conditional_marks=marks, calculate_bounds=False)
        gts.append(is_res["est"].item())
        effs.append(is_res["rel_eff"].item())

    return gts, effs

def marginal_mark_queries(args, model, dataloader, results=None):
    file_suffix = datetime.datetime.now().strftime('%m_%d_%Y_%H_%M_%S')
    # Determine percentage of unique marks to keep for query ahead of time, so it is consistent across runs
    set_random_seed(args)
    num_queries = min(args.num_queries, len(dataloader))
    keep_mark_pcts = torch.rand((num_queries,)).tolist()
    #print_log("KEEPING THESE MARK PCTS", keep_mark_pcts)

    seed = args.seed
    if results is None:
        results = {"estimates": {}, "gt": None, "gt_eff": None, "nth_marginal": args.marg_query_n}

    args.seed = seed + len(args.num_seqs)*len(args.num_int_pts) + 1

    if not args.just_gt:
        for i, num_seqs in enumerate(args.num_seqs):
            ns_key = "num_seqs_{}".format(num_seqs)
            results["estimates"][ns_key] = {}
            for j, num_int_pts in enumerate(args.num_int_pts):
                np_key = "num_int_pts_{}".format(num_int_pts)
                args.seed = seed + i*len(args.num_int_pts) + j
                set_random_seed(args)
                if (ns_key in results["estimates"]) and (np_key in results["estimates"][ns_key]) and (results["estimates"][ns_key][np_key] is not None):
                    print_log("Skipping {} {}".format(ns_key, np_key))
                    continue
                else:
                    results["estimates"][ns_key][np_key] = _marginal_mark_queries(args, model, dataloader, num_seqs, num_int_pts, keep_mark_pcts)
                    save_results(args, results, file_suffix)

    set_random_seed(args)
    if ((results["gt"] is None) or (results["gt_eff"] is None)) and (not args.skip_gt):
        print_log("Starting GT Estimates w/ {} Samples.".format(args.gt_num_seqs))
        results["gt"], results["gt_eff"] = _marginal_mark_queries_gt(args, model, dataloader, keep_mark_pcts)
        save_results(args, results, file_suffix)
    else:
        print_log("Skipping GT Estimates.")


    save_results(args, results, file_suffix)
    return results

# def _setup_a_before_b_query(args, batch, model, num_seqs, a_pct, b_pct, use_tqdm=False):
def _setup_a_before_b_query(args, batch, model, num_seqs, A, B, use_tqdm=False):
    if args.cuda:
        batch = {k:v.cuda(torch.cuda.current_device()) for k,v in batch.items()}

    times, marks = batch["tgt_times"], batch["tgt_marks"]
    length = times.numel()
    global_max_time = times.max()
    to_condition_on = min(length-1, 5)  # length // 2
    
    marks_of_interest = torch.unique(marks[0, ...])
    num_unique_marks = marks_of_interest.numel()
    if num_unique_marks == 1:
        marks_of_interest = torch.cat((marks_of_interest, torch.tensor(list(range(20))).to(times.device)))
        marks_of_interest = torch.unique(marks_of_interest)
        num_unique_marks = marks_of_interest.numel()

    # a_num = math.floor(a_pct*num_unique_marks)
    # b_num = math.floor(b_pct*num_unique_marks)
    # a_num = min(max(a_num, 1), num_unique_marks-1)
    # b_num = min(max(b_num, a_num+1), num_unique_marks)
    # assert(1 <= a_num < b_num <= num_unique_marks)
    # A = marks_of_interest[:a_num]
    # B = marks_of_interest[a_num:b_num]

    times, marks = times[..., :to_condition_on], marks[..., :to_condition_on]
    cond_max_time = times.max()
    
    abq = ABeforeBQuery(
        A, 
        B, 
        total_marks=model.num_channels, 
        batch_size=args.query_batch_size, 
        device=args.device, 
        use_tqdm=use_tqdm, 
        proposal_batch_size=args.proposal_batch_size,
        proposal_right_window_limit=cond_max_time*args.hit_ab_scale if args.use_hit_a_b else float('inf'),
        precision_stop=args.ab_precision_stop,
        default_dominating_rate=args.default_dominating_rate,
        dynamic_buffer_size=args.dynamic_buffer_size,
    )

    return times, marks, abq

# def _a_before_b_queries(args, model, dataloader, num_seqs, num_int_pts, keep_mark_pcts):
def _a_before_b_queries(args, model, dataloader, num_seqs, num_int_pts, As, Bs):
    results = {
        "naive_est": [],
        "avg_naive_time": 0.0,
    }
    if args.use_hit_a_b:
        results["is_est"] = []
        results["is_lower"] = []
        results["is_upper"] = []
        
        results["is_var"] = []
        results["naive_var"] = []
        results["rel_eff"] = []
        results["avg_is_time"] = 0.0
    else:
        for pct in args.ab_pct_splits:
            results["is_{}_est".format(pct)] = []
            results["is_{}_var".format(pct)] = []
            results["naive_{}_var".format(pct)] = []
            results["rel_{}_eff".format(pct)] = []
            results["avg_is_{}_time".format(pct)] = 0.0    
    
    num_queries = min(args.num_queries, len(dataloader))

    print_log("Estimating {} Total Queries using {} Sequences and {} Integration Points Each".format(num_queries, num_seqs, num_int_pts))
    print_log("\tUsing {} Estimation Technique".format("Hitting Time" if args.use_hit_a_b else "Censored Marginal"))
    dl_iter = iter(dataloader)
    for i in tqdm(range(num_queries)):
        batch = next(dl_iter)

        if i < args.query_start_idx:
            continue

        # a_pct, b_pct = keep_mark_pcts[i]        
        A, B = As[i], Bs[i]

        # times, marks, abq = _setup_a_before_b_query(args, batch, model, num_seqs, a_pct, b_pct)
        times, marks, abq = _setup_a_before_b_query(args, batch, model, num_seqs, A, B)

        naive_t0 = time.perf_counter()
        naive_res = abq.naive_estimate(model, num_seqs, conditional_times=times, conditional_marks=marks)
        naive_t1 = time.perf_counter()
        results["naive_est"].append(naive_res)
        results["avg_naive_time"] += (naive_t1 - naive_t0) / num_queries

        if args.use_hit_a_b:
            is_t0 = time.perf_counter()
            is_res = abq.alt_2_estimate(model, num_seqs, conditional_times=times, conditional_marks=marks)
            # is_res = abq.alt_estimate(model, num_seqs, num_int_pts, conditional_times=times, conditional_marks=marks)
            is_t1 = time.perf_counter()
            is_res = {k:v.item() for k,v in is_res.items()}
            
            results["is_est"].append(is_res["est"])
            results["is_lower"].append(is_res["lower_bound"])
            results["is_upper"].append(is_res["upper_bound"])
            results["is_var"].append(is_res["is_var"])
            results["naive_var"].append(is_res["naive_var"])
            results["rel_eff"].append(is_res["rel_eff"])
            results["avg_is_time"] += (is_t1 - is_t0) / num_queries
        else:
            for pct in args.ab_pct_splits:
                is_t0 = time.perf_counter()
                is_res = abq.estimate(model, num_seqs, num_int_pts, pct_used_for_censoring=pct, conditional_times=times, conditional_marks=marks)
                is_t1 = time.perf_counter()
                is_res = {k:v.item() for k,v in is_res.items()}
                
                results["is_{}_est".format(pct)].append(is_res["est"])
                results["is_{}_var".format(pct)].append(is_res["is_var"])
                results["naive_{}_var".format(pct)].append(is_res["naive_var"])
                results["rel_{}_eff".format(pct)].append(is_res["rel_eff"])
                results["avg_is_{}_time".format(pct)] += (is_t1 - is_t0) / num_queries

    return results

# def _a_before_b_queries_gt(args, model, dataloader, keep_mark_pcts):
def _a_before_b_queries_gt(args, model, dataloader, As, Bs):
    num_seqs = args.gt_num_seqs
    num_int_pts = args.gt_num_int_pts
    num_queries = min(args.num_queries, len(dataloader))
    gts = []
    effs = None if args.use_naive_for_gt else []
    dl_iter = iter(dataloader)
    print("USING {} FOR GT".format("naive" if args.use_naive_for_gt else "importance sampling"))
    for i in tqdm(range(num_queries)):
        batch = next(dl_iter)
        
        if i < args.query_start_idx:
            continue

        # a_pct, b_pct = keep_mark_pcts[i]
        A, B = As[i], Bs[i]
        times, marks, abq = _setup_a_before_b_query(args, batch, model, num_seqs, A, B, use_tqdm=False)

        if args.use_naive_for_gt:
            res = abq.naive_estimate(model, num_seqs, conditional_times=times, conditional_marks=marks)
            gts.append(res)
        else:
            is_res = abq.estimate(model, num_seqs, num_int_pts, pct_used_for_censoring=args.ab_gt_pct, conditional_times=times, conditional_marks=marks)
            gts.append(is_res["est"].item())
            effs.append(is_res["rel_eff"].item())

    return gts, effs

def a_before_b_queries(args, model, dataloader, results=None):
    file_suffix = datetime.datetime.now().strftime('%m_%d_%Y_%H_%M_%S')
    # Determine percentage of unique marks to keep for query ahead of time, so it is consistent across runs
    set_random_seed(args)
    As, Bs = [], []
    n = model.num_channels
    to_A, to_B = max(n//3, 1), max(2*n//3, 2)
    num_queries = min(args.num_queries, len(dataloader))
    for _ in range(num_queries):
        possible_marks = list(range(n))
        random.shuffle(possible_marks)
        As.append(possible_marks[:to_A])
        Bs.append(possible_marks[to_A:to_B])
    # keep_mark_pcts = torch.rand((num_queries,2))  # Second dim will house (the percentage of marks for A, the percentage of marks for A and B)
    # keep_mark_pcts[:, 1] += keep_mark_pcts[:, 0]  
    # keep_mark_pcts = keep_mark_pcts.tolist()

    seed = args.seed
    if results is None:
        results = {"estimates": {}, "gt": None, "gt_eff": None}

    args.seed = seed + len(args.num_seqs)*len(args.num_int_pts) + 1
    set_random_seed(args)
    if (results["gt"] is None) and (not args.skip_gt):
        # results["gt"], results["gt_eff"] = _a_before_b_queries_gt(args, model, dataloader, keep_mark_pcts)
        results["gt"], results["gt_eff"] = _a_before_b_queries_gt(args, model, dataloader, As, Bs)
        save_results(args, results, file_suffix)
    else:
        print_log("Skipping GT Estimates.")

    if not args.just_gt:
        for i, num_seqs in enumerate(args.num_seqs):
            ns_key = "num_seqs_{}".format(num_seqs)
            results["estimates"][ns_key] = {}
            for j, num_int_pts in enumerate(args.num_int_pts):
                np_key = "num_int_pts_{}".format(num_int_pts)
                args.seed = seed + i*len(args.num_int_pts) + j
                set_random_seed(args)
                if (ns_key in results["estimates"]) and (np_key in results["estimates"][ns_key]) and (results["estimates"][ns_key][np_key] is not None):
                    print_log("Skipping {} {}".format(ns_key, np_key))
                    continue
                else:
                    # results["estimates"][ns_key][np_key] = _a_before_b_queries(args, model, dataloader, num_seqs, num_int_pts, keep_mark_pcts)
                    results["estimates"][ns_key][np_key] = _a_before_b_queries(args, model, dataloader, num_seqs, num_int_pts, As, Bs)
                    save_results(args, results, file_suffix)
    
    save_results(args, results, file_suffix)
    return results

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

    if args.continue_experiments is not None:
        partial_res = pickle.load(open(args.continue_experiments, "rb"))
    else:
        partial_res = None

    print_log("")
    print_log("")
    print_log("Commencing Experiments")
    with torch.no_grad():
        if args.hitting_time_queries:
            hitting_time_queries(args, model, valid_dataloader, partial_res)
        elif args.marginal_mark_queries:
            marginal_mark_queries(args, model, valid_dataloader, partial_res)
        elif args.a_before_b_queries:
            a_before_b_queries(args, model, valid_dataloader, partial_res)
    


if __name__ == "__main__":
    main()
