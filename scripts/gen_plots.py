import pickle
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import seaborn as sns
matplotlib.rc('font', family='serif')
matplotlib.rc('text', usetex=True)


COL_WIDTH = 3.2499999998649995
TEXT_WIDTH = 6.7499999997196145
HALF_PLOT_WIDTH = 1.4
HALF_PLOT_WIDTH_GAP = COL_WIDTH - 2*HALF_PLOT_WIDTH
FULL_PLOT_WIDTH = COL_WIDTH
PLOT_HEIGHT_SHORT = 1.1
PLOT_HEIGHT_MID = 1.4
PLOT_HEIGHT_TALL = 2.0
PLOT_HEIGHT_GAP = HALF_PLOT_WIDTH_GAP*0.3

ERR_MARKER_SIZE = 4
EFF_MARKER_SIZE = 2

Y_LABEL_FS = 9
X_LABEL_FS = 9
TITLE_FS = 11
BOLD_YLABEL = False

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def flatten(t):
    return [item for sublist in t for item in sublist]

hitting_time_query_fps = [
   ("MovieLens", "./data/movie/nhp_models/hitting_time_queries/results_10_06_2022_12_51_17.pickle"),
   ("MOOC", "./data/mooc/nhp_models/hitting_time_queries/results_10_06_2022_12_51_02.pickle"), 
   ("Taobao", "./data/taobao/nhp_models/hitting_time_queries/results_10_06_2022_13_12_51.pickle"),
]

a_before_b_query_fps = [
    ("MovieLens", "./data/movie/nhp_models/a_before_b_queries/results_10_10_2022_00_07_02.pickle"),
    ("MOOC", "./data/mooc/nhp_models/a_before_b_queries/results_10_09_2022_23_56_59.pickle"),
    ("Taobao", (
        "./data/taobao/nhp_models/a_before_b_queries/results_10_10_2022_00_07_17.pickle", 
        "./data/taobao/nhp_models/a_before_b_queries/results_10_10_2022_20_56_00.pickle",
    )),
]

marginal_mark_query_fps = [
    ("MovieLens", "./data/movie/nhp_models/marginal_queries/results_10_16_2022_22_36_18.pickle"),
    ("MOOC", "./data/mooc/nhp_models/marginal_queries/results_10_16_2022_22_36_20.pickle"),
    ("Taobao", "./data/taobao/nhp_models/marginal_queries/results_10_16_2022_22_36_26.pickle"),
]

synth_fpd = {
    "runtime": "/home/alexjb/source/point_process_queries/data/synthetic_results/df_query_interactions_01_2_1000models_10seq.pkl",
    "probs": "/home/alexjb/source/point_process_queries/data/synthetic_results/hawkes_probs_1000.pkl",
    "effs": "/home/alexjb/source/point_process_queries/data/synthetic_results/hawkes_effs_1000.pkl",
}

dir_prefix = "/home/alexjb/source/point_process_queries"

def read_real_data_results(fp, dataset):
    if isinstance(fp, tuple):
        all_res = [pickle.load(open(("" if _fp[0]!="." else dir_prefix) +_fp.lstrip("."), 'rb')) for _fp in fp]
        res, all_res = all_res[0], all_res[1:]
        keys = list(res["estimates"].keys())
        for r in all_res:
            if r["gt"] is not None:
                res["gt"].extend(r["gt"])
            if r["gt_eff"] is not None:
                res["gt_eff"].extend(r["gt_eff"])
            for key in keys:
                inner_keys = res["estimates"][key]["num_int_pts_1000"].keys()
                existing_len = len(res["estimates"][key]["num_int_pts_1000"]["is_est"])
                new_len = len(r["estimates"][key]["num_int_pts_1000"]["is_est"])
                for inner_key in inner_keys:
                    if isinstance(res["estimates"][key]["num_int_pts_1000"][inner_key], list):
                        res["estimates"][key]["num_int_pts_1000"][inner_key].extend(r["estimates"][key]["num_int_pts_1000"][inner_key])
                    else:
                        res["estimates"][key]["num_int_pts_1000"][inner_key] = ((res["estimates"][key]["num_int_pts_1000"][inner_key])*(existing_len) + (r["estimates"][key]["num_int_pts_1000"][inner_key])*(new_len)) / (existing_len+new_len)
        fp = fp[0]
    else:
        res = pickle.load(open(("" if fp[0]!="." else dir_prefix) + fp.lstrip("."), 'rb'))

    keys = list(res["estimates"].keys())#["num_seqs_{}".format(i) for i in num_seqs]
    num_seqs = [int(k.split("_")[-1]) for k in keys]
    num_seqs, keys = zip(*sorted(zip(num_seqs, keys)))

    for key in keys:
        res["estimates"][key] = res["estimates"][key]["num_int_pts_1000"]

    true_last_key = keys[-1]    
    if ("gt" in res) and (res['gt'] is not None):
        gt = np.array(res['gt'])[np.newaxis, :]
    else:
        gt = np.array(res["estimates"][keys[-1]]["is_est"])
        num_seqs, keys = num_seqs[:-1], keys[:-1]

    return dotdict({
        "gt": gt,
        "est": res["estimates"],
        "num_seqs": num_seqs,
        "keys": keys,
        "true_last_key": true_last_key,
        "dataset": dataset,
    })
    

def plot_res(ax, args, res, is_first, is_last, kind, legend=False):
    assert(kind in ('effs', 'errs', "runtime"))
    if "dataset" in res:
        ax.annotate(r"\textbf{" + res.dataset + r"}", xy=(0.5, 0.9), fontsize=Y_LABEL_FS, ha='center', xycoords='axes fraction')

    ax.set_yscale(args.yscale)
    ax.set_xscale(args.xscale)
    if is_first:
        ax.set_title(args.title, fontsize=TITLE_FS)
    if is_last:
        ax.set_xlabel(args.xlabel, fontsize=X_LABEL_FS)
    else:
        ax.set_xticklabels([])

    if BOLD_YLABEL:
        ylabel = args.ylabel.format(dataset=res.get("dataset", ""))
        ax.set_ylabel(r"\textbf{" + ylabel + r"}", fontsize=Y_LABEL_FS)
    else:
        ax.set_ylabel(args.ylabel.format(dataset=res.get("dataset", "")), fontsize=Y_LABEL_FS)
    ax.yaxis.set_label_coords(args.ylabel_pad, 0.5)  # Reduce some padding from label to ticks
    ax.set_xlim(*args.xlim)

    if kind == "errs":
        gt = res["gt"]
        methods = {
            r"$\mathrm{Naive}$": np.array([res["est"][k]["naive_est"] for k in res["keys"]]),
            r"$\mathrm{Imp.}$": np.array([res["est"][k]["is_est"] for k in res["keys"]]),
        }
        colors = ["tab:orange", "tab:blue"]
        if args.plot_bounds:
            methods[r"$\mathrm{Imp.}_\mathrm{L}$"] = np.array([res["est"][k]["is_lower"] for k in res["keys"]])
            methods[r"$\mathrm{Imp.}_\mathrm{U}$"] = np.array([res["est"][k]["is_upper"] for k in res["keys"]])
            colors.extend(["tab:green", "tab:red"])
        means = {k:(np.abs(v-gt)/gt).mean(axis=-1) for k,v in methods.items()}
        for i, (method_name, method_mean) in enumerate(means.items()):
            color = colors[i]
            ax.plot(res.num_seqs, method_mean, color=color, linestyle='-', marker='o', label=method_name, markersize=ERR_MARKER_SIZE, clip_on=False)    
    elif kind == "effs":
        if "precomputed" in res:
            gt, eff = res["precomputed"]
            gt, eff = np.array(gt), np.array(eff)
        else:
            gt = res["gt"][0, :]
            naive_var = gt*(1-gt)
            is_var = res["est"][res.true_last_key]["is_var"]
            eff = naive_var / is_var
        ax.scatter(gt[eff <= args.cutoff], eff[eff <= args.cutoff], s=EFF_MARKER_SIZE, alpha=0.5, clip_on=True)

        # Plot Reference Horizontal Lines (for 1.0 Efficiency and Avg. Runtime Ratio)
        ax.axhline(y=1.0, linestyle="--", color="gray")
        print(args.title, res.get("dataset", ""), np.median(eff), np.mean(eff))
        if "est" in res:
            rt_ratio = res["est"][res.true_last_key]["avg_is_time"] / res["est"][res.true_last_key]["avg_naive_time"]
            ax.axhline(y=rt_ratio, linestyle=(0, (1, 1)), color='tab:red')
            ax.text(ax.get_xlim()[-1]*0.95, rt_ratio, "$\\times${:.2f}".format(rt_ratio), ha="right", va="center", size=X_LABEL_FS-2,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="tab:red", lw=1))
    elif kind == "runtime":
        interactions = np.linspace(0.1, 2.0, num=20)
        query_time_mean = np.array(res["runtime"]).mean(axis=0) / 10
        df_query_time = pd.DataFrame(query_time_mean, columns = interactions,
                        index = ['Naive', 'CensoredPP (Exact)', 'CensoredPP (Approx)', 'Importance Sampling'])
        naive_res = df_query_time.loc['Naive',:]
        imp_res = df_query_time.loc['Importance Sampling',:]
        ax.plot(interactions, naive_res, color="tab:orange", linestyle='-', marker='o', label="Naive", markersize=ERR_MARKER_SIZE, clip_on=False) 
        ax.plot(interactions, imp_res, color="tab:blue", linestyle='-', marker='o', label="Imp.", markersize=ERR_MARKER_SIZE, clip_on=False) 

    if "ylim" in args:
        ax.set_ylim(*args.ylim)
    if "scale_ylim" in args:
        ax.set_ylim(ax.get_ylim()[0], ax.get_ylim()[-1]*args.scale_ylim)

    if legend:
        ax.legend(loc=args.legend_loc, prop={'size': 8})

def plot_experiment_results(args, fps):
    ds_names = [n for n,_ in fps]
    num_ds = len(ds_names)

    plot_configs = []
    for i,n in enumerate(ds_names):
        plot_configs.append(["{}_err".format(n), "gap_c_{}".format(i), "{}_eff".format(n)])
        plot_configs.append(["gap_r_{}".format(i)]*3)
    plot_configs = plot_configs[:-1]  # Don't need last padded row

    gs_kw = {
        "height_ratios": ([PLOT_HEIGHT_MID, PLOT_HEIGHT_GAP]*num_ds)[:-1],
        "width_ratios": [HALF_PLOT_WIDTH, HALF_PLOT_WIDTH_GAP, HALF_PLOT_WIDTH],
    }

    fig_height, fig_width = sum(gs_kw["height_ratios"]), sum(gs_kw["width_ratios"])
    fig, axd = plt.subplot_mosaic(
        plot_configs,
        gridspec_kw=gs_kw,
        figsize=(fig_width, fig_height),
        constrained_layout=True,
    )

    for k, ax in axd.items():
        if k.startswith('gap_'):
            ax.set_visible(False)

    for i, (n, fp) in enumerate(fps):
        res = read_real_data_results(fp, n)
        ax_err, ax_eff = axd["{}_err".format(n)], axd["{}_eff".format(n)]
        plot_res(ax_err, args.err, res, is_first=i==0, is_last=i==len(fps)-1, kind="errs", legend=i==args.err.legend_id)
        plot_res(ax_eff, args.eff, res, is_first=i==0, is_last=i==len(fps)-1, kind="effs")

    for ax in axd.values():
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    fig.subplots_adjust(
        left   = 0.0,  # the left side of the subplots of the figure
        right  = 1.0,    # the right side of the subplots of the figure
        bottom = 0.0,   # the bottom of the subplots of the figure
        top    = 1.0,      # the top of the subplots of the figure
        wspace = 0.0,   # the amount of width reserved for blank space between subplots
        hspace = 0.0,   # the amount of height reserved for white space between subplots
    )

    fig.savefig(args.dest_path, bbox_inches="tight", transparent=True)

def plot_synth_results(args, fpd):
    data = {k:pickle.load(open(v,"rb")) for k,v in fpd.items()}

    fig_width, fig_height = FULL_PLOT_WIDTH, PLOT_HEIGHT_MID
    fig, ax = plt.subplots(
        1, 1,
        figsize=(fig_width, fig_height),
        constrained_layout=True,
    )

    plot_res(ax, args.runtime, {"runtime": data["runtime"]}, is_first=True, is_last=True, kind="runtime", legend=True)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.subplots_adjust(
        left   = 0.0,  # the left side of the subplots of the figure
        right  = 1.0,    # the right side of the subplots of the figure
        bottom = 0.0,   # the bottom of the subplots of the figure
        top    = 1.0,      # the top of the subplots of the figure
        wspace = 0.0,   # the amount of width reserved for blank space between subplots
        hspace = 0.0,   # the amount of height reserved for white space between subplots
    )

    fig.savefig(args.runtime.dest_path, bbox_inches="tight", transparent=True)

    fig_width, fig_height = FULL_PLOT_WIDTH, PLOT_HEIGHT_SHORT
    fig, ax = plt.subplots(
        1, 1,
        figsize=(fig_width, fig_height),
        constrained_layout=True,
    )
    plot_res(ax, args.eff, {"precomputed": (data["probs"], data["effs"])}, is_first=True, is_last=True, kind="effs", legend=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.subplots_adjust(
        left   = 0.0,  # the left side of the subplots of the figure
        right  = 1.0,    # the right side of the subplots of the figure
        bottom = 0.0,   # the bottom of the subplots of the figure
        top    = 1.0,      # the top of the subplots of the figure
        wspace = 0.0,   # the amount of width reserved for blank space between subplots
        hspace = 0.0,   # the amount of height reserved for white space between subplots
    )

    fig.savefig(args.eff.dest_path, bbox_inches="tight", transparent=True)

if __name__ == "__main__":
    hit_args = dotdict({
        "err": dotdict({
            "title": "\\textbf{(a)} Hitting Time Query\nMean RAE v. Sample Size",
            "xlabel": r"\# Samples",
            "ylabel": r"Mean RAE",
            # "ylabel": "{dataset}", #r"Mean RAE",
            "xlim": (1.5, 1000*1.1),
            # "ylim": (0.01, 1.75),
            "yscale": 'log',
            "xscale": "log",
            "plot_bounds": False,
            "legend_id": 2,
            "legend_loc": "lower left",
            "ylabel_pad": -0.28,
        }),
        "eff": dotdict({
            "title": "\\textbf{(b)} Relative Efficiency\nfor Hitting Time Queries",
            "xlabel": r"Ground Truth Prob.",
            # "ylabel": "", #r"Rel. Eff.", #r"$\mathrm{eff}(\hat{\pi}_{\mathrm{Imp.}}, \hat{\pi}_{\mathrm{Naive}})$",
            "ylabel": r"Rel. Eff.", #r"$\mathrm{eff}(\hat{\pi}_{\mathrm{Imp.}}, \hat{\pi}_{\mathrm{Naive}})$",
            "xlim": (0.0, 1.0),
            "yscale": 'log',
            "xscale": "linear",
            "ylabel_pad": -0.21,
            'cutoff': 1e6,
        }),
        "dest_path": dir_prefix + "/data/plots/hitting_time_plots.pdf",
    })
    plot_experiment_results(hit_args, hitting_time_query_fps)
    ab_args = dotdict({
        "err": dotdict({
            "title": "\\textbf{(a)} ``A Before B'' Query\nMean RAE v. Sample Size",
            "xlabel": r"\# Samples",
            "ylabel": r"Mean RAE",
            # "ylabel": "{dataset}", #r"Mean RAE",
            "xlim": (1.5, 250*1.1),     
            "ylim": (0.01, None),
            "yscale": 'log',
            "xscale": "log",     
            "plot_bounds": False, #True,  
            "legend_id": 2,
            "legend_loc": "center left",
            "ylabel_pad": -0.28,
        }),
        "eff": dotdict({
            "title": "\\textbf{(b)} Variance Reduction\nfor ``A Before B'' Queries",
            "xlabel": r"Ground Truth Prob.",
            "ylabel": r"Var. Ratio", #r"$\mathrm{Var}_{p}[\hat{\pi}_{\mathrm{Naive}}]/\mathrm{Var}_{q}[\hat{\pi}_{\mathrm{Imp.}}]$",
            # "ylabel": "", #r"Var. Ratio", #r"$\mathrm{Var}_{p}[\hat{\pi}_{\mathrm{Naive}}]/\mathrm{Var}_{q}[\hat{\pi}_{\mathrm{Imp.}}]$",
            "xlim": (0.0, 1.0),
            "scale_ylim": 7,
            "yscale": 'log',
            "xscale": "linear",
            "ylabel_pad": -0.21,
            'cutoff': 1e7,
        }),
        "dest_path": dir_prefix + "/data/plots/a_before_b_plots.pdf",
    })
    plot_experiment_results(ab_args, a_before_b_query_fps)
    marg_args = dotdict({
        "err": dotdict({
            "title": "\\textbf{(a)} Marginal Mark Query\nMean RAE v. Sample Size",
            "xlabel": r"\# Samples",
            "ylabel": r"Mean RAE",
            # "ylabel": "{dataset}", #r"Mean RAE",
            "xlim": (1.5, 1000*1.1),     
            "ylim": (0.01, None),
            "yscale": 'log',
            "xscale": "log",     
            "plot_bounds": False, #True,  
            "legend_id": 2,
            "legend_loc": "center left",
            "ylabel_pad": -0.28,
        }),
        "eff": dotdict({
            "title": "\\textbf{(b)} Relative Efficiency for\nMarginal Mark Queries",
            "xlabel": r"Ground Truth Prob.",
            "ylabel": r"Var. Ratio", #r"$\mathrm{Var}_{p}[\hat{\pi}_{\mathrm{Naive}}]/\mathrm{Var}_{q}[\hat{\pi}_{\mathrm{Imp.}}]$",
            # "ylabel": "", #r"Var. Ratio", #r"$\mathrm{Var}_{p}[\hat{\pi}_{\mathrm{Naive}}]/\mathrm{Var}_{q}[\hat{\pi}_{\mathrm{Imp.}}]$",
            "xlim": (0.0, 1.0),
            "scale_ylim": 1,
            "yscale": 'linear',
            "xscale": "linear",
            "ylabel_pad": -0.21,
            'cutoff': 5,
        }),
        "dest_path": dir_prefix + "/data/plots/marginal_mark_plots.pdf",
    })
    plot_experiment_results(marg_args, marginal_mark_query_fps)
    synth_args = dotdict({
        "runtime": dotdict({
            "title": "Synthetic Hitting Time Query Runtime Comparison",
            "xlabel": r"Interaction Strength",
            "ylabel": r"Mean Runtime per Sample",
            # "ylabel": "{dataset}", #r"Mean RAE",
            "xlim": (0, 2.05),     
            #"ylim": (0.01, None),
            "yscale": 'log',
            "xscale": "linear",     
            "plot_bounds": False, #True,  
            "legend_id": 0,
            "legend_loc": "upper left",
            "ylabel_pad": -0.12,
            "dest_path": dir_prefix + "/data/plots/synth_runtime_plot.pdf",
        }),
        "eff": dotdict({
            "title": "Relative Efficiency for Synthetic Hitting Time Queries",
            "xlabel": r"Ground Truth Prob.",
            "ylabel": r"Rel. Eff.", #r"$\mathrm{Var}_{p}[\hat{\pi}_{\mathrm{Naive}}]/\mathrm{Var}_{q}[\hat{\pi}_{\mathrm{Imp.}}]$",
            # "ylabel": "", #r"Var. Ratio", #r"$\mathrm{Var}_{p}[\hat{\pi}_{\mathrm{Naive}}]/\mathrm{Var}_{q}[\hat{\pi}_{\mathrm{Imp.}}]$",
            "xlim": (0.0, 0.7),
            "yscale": 'log',
            "xscale": "linear",
            "ylabel_pad": -0.095,
            'cutoff': 1e6,
            "dest_path": dir_prefix + "/data/plots/synth_eff_plot.pdf",
        }),
    })
    plot_synth_results(synth_args, synth_fpd)
    