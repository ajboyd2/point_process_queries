import pickle
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import seaborn as sns
matplotlib.rc('font', family='serif')
matplotlib.rc('text', usetex=True)


TEXT_WIDTH = 6.7499999997196145
# COL_WIDTH = 3.2499999998649995
COL_WIDTH = TEXT_WIDTH 
HALF_PLOT_WIDTH = 1.4
HALF_PLOT_WIDTH_GAP = COL_WIDTH - 2*HALF_PLOT_WIDTH
THIRD_PLOT_WIDTH = 2.1 #0.85
THIRD_PLOT_GAP = (COL_WIDTH - 3*THIRD_PLOT_WIDTH) / 2
FULL_PLOT_WIDTH = COL_WIDTH
PLOT_HEIGHT_SHORT = 1.1
PLOT_HEIGHT_MID = 1.4
PLOT_HEIGHT_TALL = 2.0
PLOT_HEIGHT_GAP = HALF_PLOT_WIDTH_GAP*0.05 #HALF_PLOT_WIDTH_GAP*0.3

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

log_likelihood_fps = [
   ("MovieLens", "./data/movie/nhp_models/censored_log_likelihood/results_01_15_2023_16_17_06.pickle"),#"./data/movie/nhp_models/censored_log_likelihood/movie_censored_log_likelihood_results.pickle"),
   #("MOOC", "./data/mooc/nhp_models/censored_log_likelihood/results_01_15_2023_16_17_22.pickle"), #"./data/mooc/nhp_models/censored_log_likelihood/mooc_censored_log_likelihood_results.pickle"), 
   # Real Data ^^ Sampled Data vv
   #("MOOC", "./data/mooc/nhp_models/censored_log_likelihood/results_01_17_2023_13_09_34.pickle"),
#    ("MOOC", "./data/mooc/nhp_models/censored_log_likelihood/results_01_19_2023_15_43_21.pickle"),
   ("MOOC", "./data/mooc/nhp_models/censored_log_likelihood/results_01_23_2023_20_26_45.pickle"),
   ("Taobao", "./data/taobao/nhp_models/censored_log_likelihood/taobao_censored_log_likelihood_results.pickle"),
]

next_event_fps = [
    # ("MovieLens", "./data/movie/nhp_models/censored_next_event/results_01_15_2023_19_17_43.pickle"),#"./data/movie/nhp_models/censored_next_event/movie_censored_next_event_results.pickle"),
    ("MovieLens", "./data/movie/nhp_models/censored_next_event/results_01_30_2023_00_22_03.pickle"),
    # ("MOOC", "./data/mooc/nhp_models/censored_next_event/mooc_censored_next_event_results.pickle"),
    ("MOOC", "./data/mooc/nhp_models/censored_next_event/results_01_30_2023_00_21_56.pickle"),
    # ("Taobao", "./data/taobao/nhp_models/censored_next_event/taobao_censored_next_event_results.pickle"),
    ("Taobao", "./data/taobao/nhp_models/censored_next_event/results_01_30_2023_00_20_38.pickle"),
]
real_data = True

sample_log_likelihood_fps = [
    ("MovieLens", "./data/movie/nhp_models/censored_log_likelihood/results_01_30_2023_17_10_55.pickle"),
    ("MOOC", "./data/mooc/nhp_models/censored_log_likelihood/results_01_30_2023_17_10_25.pickle"),
    ("Taobao", "./data/taobao/nhp_models/censored_log_likelihood/results_01_30_2023_23_07_08.pickle"),
]

sample_next_event_fps = [
    ("MovieLens", "./data/movie/nhp_models/censored_next_event/results_01_30_2023_21_49_15.pickle"),
    ("MOOC", "./data/mooc/nhp_models/censored_next_event/results_01_30_2023_19_19_38.pickle"),
    ("Taobao", "./data/taobao/nhp_models/censored_next_event/results_01_31_2023_03_35_54.pickle"),
]

real_data = False
log_likelihood_fps = sample_log_likelihood_fps
next_event_fps = sample_next_event_fps

dir_prefix = "/home/alexjb/source/point_process_queries"

def read_real_data_results(fp, dataset):
    res = pickle.load(open(("" if fp[0]!="." else dir_prefix) + fp.lstrip("."), 'rb'))

    pcts = sorted(list(res.keys()))

    return dotdict({
        "results": res,
        "pcts": pcts,
        "dataset": dataset,
    })
    

def plot_res(ax, args, res, is_first, is_last, kind, metric, legend=False):
    assert(kind in ("log_likelihood", "time", "mark"))
    # if is_first: #"dataset" in res:
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

    # if BOLD_YLABEL:
    #     ylabel = args.ylabel.format(dataset=res.get("dataset", ""))
    #     ax.set_ylabel(r"\textbf{" + ylabel + r"}", fontsize=Y_LABEL_FS)
    # else:
    ax.set_ylabel(args.ylabel, fontsize=Y_LABEL_FS)
    ax.yaxis.set_label_coords(args.ylabel_pad, 0.5)  # Reduce some padding from label to ticks
    if "xlim" in args:
        ax.set_xlim(*args.xlim)

    colors = ["tab:orange", "tab:blue", "tab:green", "tab:red"]

    if kind == "log_likelihood":
        pcts = sorted(res["pcts"])
        original = [np.array(res["results"][pct]["original_ll"]) for pct in pcts]
        for i, method in enumerate(args.res_keys):
            method_res = [np.array(res["results"][pct][method]) for pct in pcts]
            if metric == "avg":
                method_res = [v.mean() for v in method_res]
            elif metric == "rel":
                method_res = [((o - v) / o).mean() for v, o in zip(method_res, original)]
            ax.plot(pcts, method_res, color=colors[i], label=method.replace("_ll", "").capitalize() if legend else None, linestyle='-', marker='o', markersize=ERR_MARKER_SIZE, clip_on=False)
    elif kind == "time":
        pcts = sorted(res["pcts"])
        true_times = [np.array(res["results"][pct]["true_time"]) for pct in pcts]
        for i, (method_name, method_key) in enumerate([("Censored", "cen_time_est"), ("Baseline", "naive_time_est")]):
            method_res = [np.array(res["results"][pct][method_key]) for pct in pcts]
            # method_res = [((y_hat - y)**2).mean() for y_hat, y in zip(method_res, true_times)]
            method_res = [np.median(abs(y_hat - y)) for y_hat, y in zip(method_res, true_times)]
            # method_res = [np.median(abs(y_hat - y) / y) for y_hat, y in zip(method_res, true_times)]
            ax.plot(pcts, method_res, color=colors[i], label=None, linestyle='-', marker='o', markersize=ERR_MARKER_SIZE, clip_on=False)
    elif kind == "mark":
        pcts = sorted(res["pcts"])
        true_marks = [np.array(res["results"][pct]["true_mark"]) for pct in pcts]
        for i, (method_name, method_key) in enumerate([("Censored", "cen_mark_dist"), ("Baseline", "naive_mark_dist")]):
            method_res = [np.array(res["results"][pct][method_key]) for pct in pcts]
            # method_res = [-np.log(p_y[np.arange(len(y)), y]+1e-4).mean() for p_y, y in zip(method_res, true_marks)]
            #method_res = [(p_y.argmax(-1) == y).mean() for p_y, y in zip(method_res, true_marks)]
            all_ranks = [np.argsort(np.argsort(-p_y, axis=-1), axis=-1) for p_y in method_res]
            ranks = [r[np.arange(len(y)), y] for r, y in zip(all_ranks, true_marks)]
            k = 10
            method_res = [(r < k).mean() for r in ranks]
            ax.plot(pcts, method_res, color=colors[i], label=None, linestyle='-', marker='o', markersize=ERR_MARKER_SIZE, clip_on=False)

        # r = res["results"]
        # r = {k:np.array(v) if isinstance(v, list) else v for k,v in r.items()}
        # for i, key in enumerate(args.y_key):
        #     ax.scatter(r[args.x_key], r[key], color=colors[i], label=key.replace("_ll", "").capitalize() if legend else None, s=EFF_MARKER_SIZE, alpha=0.5, clip_on=True)
        
    elif kind == "next_event":
        pass

    '''
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
    '''

    if "ylim" in args:
        ax.set_ylim(*args.ylim)
    if "scale_ylim" in args:
        ax.set_ylim(ax.get_ylim()[0], ax.get_ylim()[-1]*args.scale_ylim)

    if legend:
        ax.legend(loc=args.legend_loc, prop={'size': 8})

def plot_experiment_results(args, fps, next_event_fps):
    ds_names = [n for n,_ in fps]
    num_ds = len(ds_names)

    plot_configs = []
    for i,n in enumerate(ds_names):
        plot_configs.append(["{}_avg".format(n), "gap_c_{}_1".format(i), "{}_time".format(n), "gap_c_{}_2".format(i), "{}_mark".format(n)])
        plot_configs.append(["gap_r_{}".format(i)]*5)
    plot_configs = plot_configs[:-1]  # Don't need last padded row

    gs_kw = {
        "height_ratios": ([PLOT_HEIGHT_MID, PLOT_HEIGHT_GAP]*num_ds)[:-1],
        # "width_ratios": [HALF_PLOT_WIDTH, HALF_PLOT_WIDTH_GAP, HALF_PLOT_WIDTH], #[HALF_PLOT_WIDTH, HALF_PLOT_WIDTH_GAP, HALF_PLOT_WIDTH],
        "width_ratios": [THIRD_PLOT_WIDTH, THIRD_PLOT_GAP, THIRD_PLOT_WIDTH, THIRD_PLOT_GAP, THIRD_PLOT_WIDTH], #[HALF_PLOT_WIDTH, HALF_PLOT_WIDTH_GAP, HALF_PLOT_WIDTH],
        # "width_ratios": [FULL_PLOT_WIDTH], #[HALF_PLOT_WIDTH, HALF_PLOT_WIDTH_GAP, HALF_PLOT_WIDTH],
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
        #ax_avg, ax_rel = axd["{}_avg".format(n)], axd["{}_rel".format(n)]
        ax_avg = axd["{}_avg".format(n)]
        sub_args = args.ll
        plot_res(ax_avg, sub_args, res, is_first=i==0, is_last=i==len(fps)-1, kind="log_likelihood", metric="avg", legend=i==args.ll.legend_id)
        #plot_res(ax_rel, sub_args, res, is_first=i==0, is_last=i==len(fps)-1, kind=kind, metric="rel")

    for i, (n, fp) in enumerate(next_event_fps):
        res = read_real_data_results(fp, n)
        #ax_avg, ax_rel = axd["{}_avg".format(n)], axd["{}_rel".format(n)]
        ax_time = axd["{}_time".format(n)]
        sub_args = args.ne_t
        plot_res(ax_time, sub_args, res, is_first=i==0, is_last=i==len(next_event_fps)-1, kind="time", metric="mse", legend=i==args.ll.legend_id)
        ax_mark = axd["{}_mark".format(n)]
        sub_args = args.ne_m
        plot_res(ax_mark, sub_args, res, is_first=i==0, is_last=i==len(next_event_fps)-1, kind="mark", metric="ce", legend=i==args.ll.legend_id)

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

    all_res = {n: read_real_data_results(fp, n) for n, fp in fps}
    pcts = [0.2, 0.5, 0.8]

 
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
        "ll": dotdict({
            "title": "\\textbf{(a)} Mean Log\nLikelihood",
            "xlabel": r"",#Mark Censoring Percentage",
            "ylabel": r"Log $\mathcal{L}(\underline{\mathcal{H}})$",
            "res_keys": ["censored_ll", "baseline_ll", "naive_ll", "original_ll"],
            # "ylabel": "{dataset}", #r"Mean RAE",
            # "xlim": (1.5, 1000*1.1),
            # "ylim": (0.01, 1.75),
            "yscale": 'linear',
            "xscale": "linear",
            "plot_bounds": False,
            "legend_id": 2,
            "legend_loc": "upper left",
            "ylabel_pad": -0.2,#-0.28,
        }),
        "ne_t": dotdict({
            "title": "\\textbf{(b)} Median Absolute Error for\nNext Time Prediction",
            "xlabel": r"Mark Censoring Percentage",#Ground Truth Prob.",
            # "ylabel": "", #r"Rel. Eff.", #r"$\mathrm{eff}(\hat{\pi}_{\mathrm{Imp.}}, \hat{\pi}_{\mathrm{Naive}})$",
            "ylabel": r"Med. AE", #r"$\mathrm{eff}(\hat{\pi}_{\mathrm{Imp.}}, \hat{\pi}_{\mathrm{Naive}})$",
            # "xlim": (0.0, 1.0),
            "yscale": 'linear',
            "xscale": "linear",
            "ylabel_pad": -0.21,
            # 'cutoff': 1e6,
        }),
        "ne_m": dotdict({
            "title": "\\textbf{(c)} Top-10 Accuracy for\nNext Mark Prediction",
            "xlabel": r"",#Ground Truth Prob.",
            # "ylabel": "", #r"Rel. Eff.", #r"$\mathrm{eff}(\hat{\pi}_{\mathrm{Imp.}}, \hat{\pi}_{\mathrm{Naive}})$",
            "ylabel": r"Acc@10", #r"$\mathrm{eff}(\hat{\pi}_{\mathrm{Imp.}}, \hat{\pi}_{\mathrm{Naive}})$",
            #"xlim": (0.0, 1.0),
            "yscale": 'linear',
            "xscale": "linear",
            "ylabel_pad": -0.21,
            # 'cutoff': 1e6,
        }),
        "dest_path": dir_prefix + "/data/plots/{}_censored_log_likelihood_plots.pdf".format("real" if real_data else "sampled"),
    })
    plot_experiment_results(hit_args, log_likelihood_fps, next_event_fps)





    