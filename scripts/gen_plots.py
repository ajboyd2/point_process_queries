import pickle
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from matplotlib.ticker import MultipleLocator, PercentFormatter
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
#   ("MovieLens", "./data/movie/nhp_models/censored_log_likelihood/results_01_15_2023_16_17_06.pickle"),#"./data/movie/nhp_models/censored_log_likelihood/movie_censored_log_likelihood_results.pickle"),
   #("MOOC", "./data/mooc/nhp_models/censored_log_likelihood/results_01_15_2023_16_17_22.pickle"), #"./data/mooc/nhp_models/censored_log_likelihood/mooc_censored_log_likelihood_results.pickle"), 
   # Real Data ^^ Sampled Data vv
   #("MOOC", "./data/mooc/nhp_models/censored_log_likelihood/results_01_17_2023_13_09_34.pickle"),
#    ("MOOC", "./data/mooc/nhp_models/censored_log_likelihood/results_01_19_2023_15_43_21.pickle"),
#   ("MOOC", "./data/mooc/nhp_models/censored_log_likelihood/results_01_23_2023_20_26_45.pickle"),
   ("Taobao", "./data/taobao/nhp_models/censored_log_likelihood/taobao_censored_log_likelihood_results.pickle"),
   ("Reddit", "./data/reddit/nhp_models/censored_log_likelihood/results_02_05_2023_12_53_14.pickle"), #"./data/reddit/nhp_models/censored_log_likelihood/results_02_05_2023_00_52_54.pickle"),
   ("MemeTracker", "./data/meme/nhp_models/censored_log_likelihood/results_02_04_2023_22_46_17.pickle"),
   ("Email", "./data/email/nhp_models/censored_log_likelihood/results_02_04_2023_22_45_45.pickle"),
]

next_event_fps = [
    # ("MovieLens", "./data/movie/nhp_models/censored_next_event/results_01_15_2023_19_17_43.pickle"),#"./data/movie/nhp_models/censored_next_event/movie_censored_next_event_results.pickle"),
#    ("MovieLens", "./data/movie/nhp_models/censored_next_event/results_01_30_2023_00_22_03.pickle"),
    # ("MOOC", "./data/mooc/nhp_models/censored_next_event/mooc_censored_next_event_results.pickle"),
#    ("MOOC", "./data/mooc/nhp_models/censored_next_event/results_01_30_2023_00_21_56.pickle"),
    # ("Taobao", "./data/taobao/nhp_models/censored_next_event/taobao_censored_next_event_results.pickle"),
    ("Taobao", "./data/taobao/nhp_models/censored_next_event/results_01_30_2023_00_20_38.pickle"),
    ("Reddit", "./data/reddit/nhp_models/censored_next_event/results_02_05_2023_14_07_25.pickle"), #"./data/reddit/nhp_models/censored_next_event/results_02_05_2023_02_12_10.pickle"),
    ("MemeTracker", "./data/meme/nhp_models/censored_next_event/results_02_05_2023_00_03_40.pickle"),
    ("Email", "./data/email/nhp_models/censored_next_event/results_02_04_2023_23_07_54.pickle"),
]
real_data = True
total_censoring = False
if total_censoring:
    # Total Censoring vvv
    log_likelihood_fps = [
        ("MovieLens", "./data/movie/nhp_models/censored_log_likelihood/results_02_03_2023_02_09_05.pickle"),
        ("MOOC", "./data/mooc/nhp_models/censored_log_likelihood/results_02_03_2023_02_09_06.pickle"),
        ("Taobao", "./data/taobao/nhp_models/censored_log_likelihood/results_02_03_2023_02_09_00.pickle"),
    ]
    # Temp next event results
    next_event_fps = [
        ("MovieLens", "./data/movie/nhp_models/censored_next_event/results_01_30_2023_21_49_15.pickle"),
        ("MOOC", "./data/mooc/nhp_models/censored_next_event/results_01_30_2023_19_19_38.pickle"),
        ("Taobao", "./data/taobao/nhp_models/censored_next_event/results_01_31_2023_03_35_54.pickle"),
    ]

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

# real_data = False
# log_likelihood_fps = sample_log_likelihood_fps
# next_event_fps = sample_next_event_fps

dir_prefix = "/home/alexjb/source/point_process_queries"

def read_real_data_results(fp, dataset):
    res = pickle.load(open(("" if fp[0]!="." else dir_prefix) + fp.lstrip("."), 'rb'))

    pcts = sorted(list(res.keys()))

    return dotdict({
        "results": res,
        "pcts": pcts,
        "dataset": dataset,
    })
    
def format_ax(ax, args, left, top, bottom):
    if not args.full_border:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    if "title" in args and top:
        ax.set_title(args.title.format(**args), fontsize=TITLE_FS)
    if left:
        ax.set_ylabel(args.ylabel, fontsize=Y_LABEL_FS)
        ax.yaxis.set_label_coords(args.ylabel_pad, 0.5)  # Reduce some padding from label to ticks
    else:
        ax.set_ylabel("")
    if bottom:
        ax.set_xlabel(args.xlabel, fontsize=X_LABEL_FS)
        ax.tick_params(which="both", top=True, labeltop=True, bottom=False, labelbottom=False, left=False, labelleft=False)
        ax.tick_params(axis="x", pad=0.5)
        ax.spines['bottom'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
    else:
        ax.tick_params(which="both", top=False, labeltop=False, bottom=True, labelbottom=False)

    ax.tick_params(axis="y", pad=2.0)

    if "yscale" in args and args.yscale!="linear":
        ax.set_yscale(args.yscale)
    if "xscale" in args and args.xscale!="linear":
        ax.set_xscale(args.xscale)
    
    if "xlim" in args:
        ax.set_xlim(*args.xlim)
    if "ylim" in args:
        ax.set_ylim(*args.ylim)
    if args.get("flip_y", False):
        ax.set_ylim(*ax.get_ylim()[::-1])

def plot_ll(ax, args, res):
    colors = ["tab:orange", "tab:blue", "tab:green", "tab:red"]

    pcts = sorted(res["pcts"])

    censored, baseline, num_censored_marks = [], [], []
    for pct in pcts:
        censored.extend(res["results"][pct]["censored_ll"])
        baseline.extend(res["results"][pct]["baseline_ll"])
        num_censored_marks.extend(res["results"][pct]["kept_marks"])
    censored, baseline, num_censored_marks = np.array(censored), np.array(baseline), np.array(num_censored_marks)
    response = (censored-baseline) / np.log(10.)
    ax.plot(num_censored_marks, response, '.', color="tab:blue", alpha=0.4)

    ax.axhline(0, linestyle=':', color="tab:red")
    lowess = sm.nonparametric.lowess
    ax.plot(np.sort(num_censored_marks), lowess(response, num_censored_marks, frac=1./3)[:, 1], color="black")

    if res.dataset == "Taobao":
        ax.set_ylim((-6, 21)) #30))
        ticks = [-5, 0, 5, 10, 15, 20]#, 30]
    elif res.dataset == "Reddit":
        ax.set_ylim((None, 18))
        ticks = [0, 5, 10, 15]
    elif res.dataset == "MemeTracker":
        ax.set_ylim((-8, 18))
        ticks = [-5, 0, 5, 10, 15] #20]
    elif res.dataset == "Email":
        ax.set_ylim((None, 12))
        ticks = [0, 5, 10]
    # ax.set_yticks(ticks)
    # ax.set_yticklabels([r"$10^{" + str(tick) + r"}$" for tick in ticks])
    ax.yaxis.set_major_locator(MultipleLocator(10))
    ax.yaxis.set_major_formatter('$10^{{ {x:.0f} }}$')
    ax.yaxis.set_minor_locator(MultipleLocator(5))

    # print(ax.get_yticks())
    # print(ax.get_yticklabels())

def plot_time(ax, args, res):
    colors = ["tab:orange", "tab:blue", "tab:green", "tab:red"]

    pcts = sorted(res["pcts"])
    true_times = [np.array(res["results"][pct]["true_time"]) for pct in pcts]

    true_times, censored, baseline, num_censored_marks = [], [], [], []
    for pct in pcts:
        true_times.extend(res["results"][pct]["true_time"])
        censored.extend(res["results"][pct]["cen_time_est"])
        baseline.extend(res["results"][pct]["naive_time_est"])
        num_censored_marks.extend(res["results"][pct]["kept_marks"])
    true_times, censored, baseline, num_censored_marks = np.array(true_times), np.array(censored), np.array(baseline), np.array(num_censored_marks)

    sorted_num_censored_marks = np.sort(num_censored_marks)
    lowess = sm.nonparametric.lowess
    censored_err = np.abs(true_times - censored)
    baseline_err = np.abs(true_times - baseline)
    ax.plot(sorted_num_censored_marks, lowess(censored_err, num_censored_marks)[:, 1], color="tab:orange")
    ax.plot(sorted_num_censored_marks, lowess(baseline_err, num_censored_marks)[:, 1], color="tab:green")
    

def plot_mark(ax, args, res, legend=False):
    colors = ["tab:orange", "tab:blue", "tab:green", "tab:red"]

    pcts = sorted(res["pcts"])
    true_marks = [np.array(res["results"][pct]["true_mark"]) for pct in pcts]
    
    true_marks, censored, baseline, num_censored_marks = [], [], [], []
    for pct in pcts:
        true_marks.extend(res["results"][pct]["true_mark"])
        censored.extend(res["results"][pct]["cen_mark_dist"])
        baseline.extend(res["results"][pct]["naive_mark_dist"])
        num_censored_marks.extend(res["results"][pct]["kept_marks"])
    true_marks, censored, baseline, num_censored_marks = np.array(true_marks), np.array(censored), np.array(baseline), np.array(num_censored_marks)

    all_censored_ranks = np.argsort(np.argsort(-censored, axis=-1), axis=-1)
    all_baseline_ranks = np.argsort(np.argsort(-baseline, axis=-1), axis=-1)
    censored_ranks = all_censored_ranks[np.arange(len(true_marks)), true_marks]
    baseline_ranks = all_baseline_ranks[np.arange(len(true_marks)), true_marks]
    k = 10
    censored_thresh = (censored_ranks < k).astype(float)#[(r < k).mean() for r in ranks]
    baseline_thresh = (baseline_ranks < k).astype(float)#[(r < k).mean() for r in ranks]
    sorted_num_censored_marks = np.sort(num_censored_marks)
    lowess = sm.nonparametric.lowess

    ax.plot(sorted_num_censored_marks, lowess(censored_thresh, num_censored_marks)[:, 1], color="tab:orange", label="Censored" if legend else None)
    ax.plot(sorted_num_censored_marks, lowess(baseline_thresh, num_censored_marks)[:, 1], color="tab:green", label="Baseline" if legend else None)

    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_major_formatter('{x:.1f}')
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))

def plot_density(ax, args, res):
    pcts = sorted(res["pcts"])

    num_censored_marks = []
    for pct in pcts:
        num_censored_marks.extend(res["results"][pct]["kept_marks"])
    num_censored_marks = np.array(num_censored_marks)

    sns.kdeplot(x=num_censored_marks, cut=0, ax=ax, fill=True, log_scale=args.xscale=="log")
 
def plot_ll_results(args, fpd):
    datasets = {n: read_real_data_results(fp, n) for n, fp in fpd}
    num_ds = len(datasets)

    plot_configs = []
    for i,n in enumerate(datasets.keys()):
        plot_configs.append(["{}_ll".format(n), "gap_r_{}".format(i), "{}_density".format(n)])
        plot_configs.append(["gap_c_{}".format(i)]*3)
    plot_configs = plot_configs[:-1]  # Don't need last padded row
    plot_configs = list(map(list, zip(*plot_configs)))  # Transpose list as it was built by column instead of by row

    w = FULL_PLOT_WIDTH
    gap_pct, plot_pct = 0.18 / num_ds, 0.82 / num_ds
    gs_kw = {
        "height_ratios": [PLOT_HEIGHT_SHORT*0.9, PLOT_HEIGHT_GAP*1.25, PLOT_HEIGHT_SHORT*0.4],
        "width_ratios": ([w*plot_pct, w*gap_pct]*num_ds)[:-1],
    }

    fig_height, fig_width = sum(gs_kw["height_ratios"]), sum(gs_kw["width_ratios"])
    fig, axd = plt.subplot_mosaic(
        plot_configs,
        gridspec_kw=gs_kw,
        figsize=(fig_width, fig_height),
        constrained_layout=True,
        sharex=False,
    )

    for k, ax in axd.items():
        if k.startswith('gap_'):
            ax.set_visible(False)

    for i, (name, data) in enumerate(datasets.items()):
        args.ll.dataset, args.density.dataset = name, name
        ax_ll, ax_density = axd["{}_ll".format(name)], axd["{}_density".format(name)]
        plot_ll(ax=ax_ll, args=args.ll, res=data)
        plot_density(ax=ax_density, args=args.density, res=data)
        
        format_ax(ax=ax_ll, args=args.ll, left=i==0, top=True, bottom=False)
        format_ax(ax=ax_density, args=args.density, left=i==0, top=False, bottom=True)
        ax_density.set_xlim(*ax_ll.get_xlim())

    fig.subplots_adjust(
        left   = 0.0,  # the left side of the subplots of the figure
        right  = 1.0,    # the right side of the subplots of the figure
        bottom = 0.0,   # the bottom of the subplots of the figure
        top    = 1.0,      # the top of the subplots of the figure
        wspace = 0.0,   # the amount of width reserved for blank space between subplots
        hspace = 0.0,   # the amount of height reserved for white space between subplots
    )

    fig.savefig(args.dest_path, bbox_inches="tight", transparent=True)


def plot_next_event_results(args, fpd):
    datasets = {n: read_real_data_results(fp, n) for n, fp in fpd}
    num_ds = len(datasets)

    plot_configs = []
    for i,n in enumerate(datasets.keys()):
        plot_configs.append(["{}_time".format(n), "gap_r_{}".format(i), "{}_mark".format(n), "gap_r2_{}".format(i), "{}_density".format(n)])
        plot_configs.append(["gap_c_{}".format(i)]*5)
    plot_configs = plot_configs[:-1]  # Don't need last padded row
    plot_configs = list(map(list, zip(*plot_configs)))  # Transpose list as it was built by column instead of by row

    w = FULL_PLOT_WIDTH
    gap_pct, plot_pct = 0.18 / num_ds, 0.82 / num_ds
    gs_kw = {
        "height_ratios": [PLOT_HEIGHT_SHORT*0.75, PLOT_HEIGHT_GAP*0.5, PLOT_HEIGHT_SHORT*0.75, PLOT_HEIGHT_GAP*1.25, PLOT_HEIGHT_SHORT*0.4],
        "width_ratios": ([w*plot_pct, w*gap_pct]*num_ds)[:-1],
    }

    fig_height, fig_width = sum(gs_kw["height_ratios"]), sum(gs_kw["width_ratios"])
    fig, axd = plt.subplot_mosaic(
        plot_configs,
        gridspec_kw=gs_kw,
        figsize=(fig_width, fig_height),
        constrained_layout=True,
        sharex=False,
    )

    for k, ax in axd.items():
        if k.startswith('gap_'):
            ax.set_visible(False)

    for i, (name, data) in enumerate(datasets.items()):
        args.time.dataset, args.mark.dataset, args.density.dataset = name, name, name
        ax_time, ax_mark, ax_density = axd["{}_time".format(name)], axd["{}_mark".format(name)], axd["{}_density".format(name)]
        plot_time(ax=ax_time, args=args.time, res=data)
        plot_mark(ax=ax_mark, args=args.mark, res=data, legend=i==0)
        plot_density(ax=ax_density, args=args.density, res=data)
        
        format_ax(ax=ax_time, args=args.time, left=i==0, top=True, bottom=False)
        format_ax(ax=ax_mark, args=args.mark, left=i==0, top=False, bottom=False)
        format_ax(ax=ax_density, args=args.density, left=i==0, top=False, bottom=True)
        ax_mark.set_xlim(*ax_time.get_xlim())
        ax_density.set_xlim(*ax_time.get_xlim())

        if i == 0:
            ax_mark.legend(loc="lower left", prop={'size': 7})

    fig.subplots_adjust(
        left   = 0.0,  # the left side of the subplots of the figure
        right  = 1.0,    # the right side of the subplots of the figure
        bottom = 0.0,   # the bottom of the subplots of the figure
        top    = 1.0,      # the top of the subplots of the figure
        wspace = 0.0,   # the amount of width reserved for blank space between subplots
        hspace = 0.0,   # the amount of height reserved for white space between subplots
    )

    fig.savefig(args.dest_path, bbox_inches="tight", transparent=True)



if __name__ == "__main__":
    ll_args = dotdict({
        "ll": dotdict({
            "title": "{dataset}",
            "xlabel": "",
            "ylabel": "Likelihood Ratio",
            "yscale": 'linear',
            "xscale": "log",
            "ylabel_pad": -0.25,
            "full_border": False,
        }),
        "density": dotdict({
            "xlabel": r"\# Marks Censored",
            "ylabel": "Density",
            "yscale": "linear",
            "xscale": "log",
            "ylabel_pad": -0.05,
            "full_border": True,
            "ylim": (0, None),
            "flip_y": True,
        }),
        "dest_path": dir_prefix + "/data/plots/final_plots/log_likelihood_plots.pdf",
    })
    next_event_args = dotdict({
        "time": dotdict({
            "title": "{dataset}",
            "xlabel": "",
            "ylabel": r"Mean AE",
            "yscale": 'linear',
            "xscale": "log",
            "ylabel_pad": -0.21,
            "full_border": False,
        }),
        "mark": dotdict({
            "xlabel": "",
            "ylabel": r"Acc@10",
            "yscale": 'linear',
            "xscale": "log",
            "ylim": (0, 0.85),
            "ylabel_pad": -0.21,
            "full_border": False,
        }),
        "density": ll_args.density,
        "dest_path": dir_prefix + "/data/plots/final_plots/next_event_plots.pdf",
    })
    # plot_experiment_results(hit_args, log_likelihood_fps, next_event_fps)
    plot_ll_results(ll_args, log_likelihood_fps)
    plot_next_event_results(next_event_args, next_event_fps)





    