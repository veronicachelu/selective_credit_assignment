from imports import *
from utils import *
def save_fig_pdf(fig, name):
    from colabtools import fileedit
    file = '/tmp/{}.pdf'.format(name)
    with open(file, 'wb') as f:
        fig.savefig(f, facecolor='none', edgecolor='none', format='pdf', bbox_inches="tight")
    fileedit.download_file(file)  # this uses the python interface to %download_file


def save_fig_png(fig, name):
    from colabtools import fileedit
    file = '/tmp/{}.pdf'.format(name)
    with open(file, 'wb') as f:
        fig.savefig(f, facecolor='none', edgecolor='none', format='png', bbox_inches="tight")
    fileedit.download_file(file)  # this uses the python interface to %download_file


# @title plotting
OP_COLORS = (
  '#8c510a',
'#bf812d',
'#dfc27d',
'#f6e8c3',
'#c7eae5',
'#80cdc1',
'#35978f',
'#01665e',
'#a6cee3',

# # '#8c510a',
# '#bf812d',
# # '#dfc27d',
# # '#f6e8c3',
# # '#a6cee3',
# '#1f78b4',
# # '#b2df8a',
# # '#33a02c',
#     '#a6cee3',
#     '#1f78b4',
#     '#b2df8a',
#     '#33a02c',
#     '#fb9a99',
#     '#e31a1c',
#     '#fdbf6f',
#     '#ff7f00',
#     '#cab2d6',
#     '#6a3d9a',
#     '#ffff99',
#     '#b15928',
#     '#8c510a',
#     '#bf812d',
#     '#dfc27d',
#     '#f6e8c3',
#     '#c7eae5',
#     '#80cdc1',
#     '#35978f',
#     '#01665e',
#     '#a6cee3',
#     '#1f78b4',
#     '#b2df8a',
#     '#33a02c',
#     '#1b9e77',
#     '#d95f02',
)
COLORS = (
'#8c510a',
# '#bf812d',
'#dfc27d',
# '#f6e8c3',
'#c7eae5',
# '#80cdc1',
'#35978f',
# '#01665e',
#     '#a6cee3',
#     '#1f78b4',
#     '#b2df8a',
#     '#33a02c',
#     '#fb9a99',
#     '#e31a1c',
#     '#fdbf6f',
#     '#ff7f00',
#     '#cab2d6',
#     '#6a3d9a',
#     '#ffff99',
#     '#b15928',
#     '#8c510a',
#     '#bf812d',
#     '#dfc27d',
#     '#f6e8c3',
#     '#c7eae5',
#     '#80cdc1',
#     '#35978f',
#     '#01665e',
#     '#a6cee3',
#     '#1f78b4',
#     '#b2df8a',
#     '#33a02c',
#     '#1b9e77',
#     '#d95f02',
#     # '#7570b3',
#     '#253494',
#     '#2c7fb8',
#     '#41b6c4',
#     '#7fcdbb',
#     '#c7e9b4',
#     '#ffffcc',
)


def make_axis_nice(ax, fontsize, smoothing):
    ax.xaxis.label.set_fontsize(fontsize)
    ax.yaxis.label.set_fontsize(fontsize)
    ax.tick_params(axis="both", which="both", bottom="off", top="off",
                   labelbottom="on", left="off", right="off", labelleft="on")
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set(visible=True, color='black', lw=2)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set(visible=True, color='black', lw=2)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.xaxis.grid(False)
    ax.yaxis.grid(linestyle="--", lw=0.5, color="black", alpha=0.25)
    ax.tick_params(labelsize=fontsize)


def add_subplot(env_id, data, num_ticks,
                seeds, agents, data_key, subplot_title, ax, smoothing=10,
                fontsize=20, linestyle="-",
                custom_hyperparams=None, hyperparams=None):
    i = 0
    i_q = 0
    i_qet = 0
    for agent in agents:
        print(agent)
        agents_params = {}
        for k, v in hyperparams.items():
            agents_params[k] = v
            if agent in custom_hyperparams.keys() and \
                    k in custom_hyperparams[agent].keys():
                agents_params[k] = custom_hyperparams[agent][k]
        hyper = {}
        # print(agents_params.values())
        for hyper_values in itertools.product(*agents_params.values()):
            for v, k in zip(hyper_values, agents_params.keys()):
                hyper[k] = v
            print(i)
            if len(data[i]) == 0:
                i += 1
                i_q += 1 if hyper['use_only_primitive_actions'] == 1 else 0
                i_qet += 1 if hyper['use_only_primitive_actions'] == 0 else 0
                continue

            mean = np.mean(data[i], axis=0)
            data_std = np.std(data[i], axis=0)

            if len(mean) <= smoothing:
                i += 1
                i_q += 1 if hyper['use_only_primitive_actions'] == 1 else 0
                i_qet += 1 if hyper['use_only_primitive_actions'] == 0 else 0
                continue
            mean = smooth(mean, smoothing)
            x_mean = range(0, len(mean) * smoothing, smoothing)
            data_std = smooth(data_std, smoothing)
            agent_label = (
                        r'{}{}{}{}'.format("X-" if hyper["eta_f"] == 0 else "", "E" if hyper["use_prior_corr"] else r'',
                                           r'QET' if hyper['eta_q'] == 0 else r'Q',
                                           r'$_O$' if not hyper["use_only_primitive_actions"] else r'$_A$') +
                        "{}{}{}{}{}{}{}{}{}{}".format(
                            r'($\lambda={}$)'.format(hyper["lambda"]) if hyper["lambda"] is not None else r'',
                            r', w. IS/option' if hyper["option_use_post_corr"] else r'',
                            r', w. IS' if hyper["use_post_corr"] else r'',
                            r',$\epsilon={}$'.format(hyper["epsilon"]) if hyper["epsilon"] is not None and (
                                        "epsilon" in list(agents_params.keys()) and len(
                                    agents_params["epsilon"]) > 1 or "epsilon" not in list(
                                    agents_params.keys())) else '',
                            r',$\alpha_q={}$'.format(hyper["alpha_q"]) if hyper["alpha_q"] is not None and (
                                        "alpha_q" in list(agents_params.keys()) and len(
                                    agents_params["alpha_q"]) > 1 or "alpha_q" not in list(
                                    agents_params.keys())) else '',
                            r',$\alpha_z={}$'.format(hyper["alpha_z"]) if hyper["alpha_z"] is not None and len(
                                agents_params["alpha_z"]) > 1 else '',
                            r',$\alpha_f={}$'.format(hyper["alpha_f"]) if hyper["alpha_f"] is not None and (
                                        "alpha_f" in list(agents_params.keys()) and len(
                                    agents_params["alpha_f"]) > 1 or "alpha_f" not in list(
                                    agents_params.keys())) else '',
                            r',$\alpha^o_q={}$'.format(hyper["option_alpha_q"]) if hyper[
                                                                                       "option_alpha_q"] is not None and (
                                                                                               "option_alpha_q" in list(
                                                                                           agents_params.keys()) and len(
                                                                                           agents_params[
                                                                                               "option_alpha_q"]) > 1 or "option_alpha_q" not in list(
                                                                                           agents_params.keys())) else '',
                            r',$\alpha^o_z={}$'.format(hyper["option_alpha_z"]) if hyper[
                                                                                       "option_alpha_z"] is not None and (
                                                                                               "option_alpha_z" in list(
                                                                                           agents_params.keys()) and len(
                                                                                           agents_params[
                                                                                               "option_alpha_z"]) > 1 or "option_alpha_z" not in list(
                                                                                           agents_params.keys())) else '',
                            r',$\alpha^o_f={}$'.format(hyper["option_alpha_f"]) if hyper[
                                                                                       "option_alpha_f"] is not None and (
                                                                                               "option_alpha_f" in list(
                                                                                           agents_params.keys()) and len(
                                                                                           agents_params[
                                                                                               "option_alpha_f"]) > 1 or "option_alpha_f" not in list(
                                                                                           agents_params.keys())) else '',
                            r',$\beta={}$'.format(hyper["beta"]) if hyper["beta"] < 100 and len(
                                agents_params["beta"]) > 1 else '',
                            r',$\beta^o={}$'.format(hyper["option_beta"]) if hyper["option_beta"] < 100 and len(
                                agents_params["option_beta"]) > 1 else ''))
            linestyle = ':' if hyper['use_only_primitive_actions'] == 1 else '-'
            color = i_q % len(COLORS) if hyper['use_only_primitive_actions'] == 1 else i_qet % len(COLORS)
            ax.plot(x_mean, mean,
                    ls=linestyle, lw=5,
                    color=COLORS[color], alpha=1.,
                    label=agent_label)
            ax.fill_between(x=x_mean, y1=mean - data_std / np.sqrt(seeds),
                            y2=mean + data_std / np.sqrt(seeds),
                            color=COLORS[color], alpha=0.2)
            #   for s in range(seeds):
            #     if len(data[i][s]) <= smoothing:
            #       continue
            #     ax.plot(smooth(data[i][s], smoothing),
            #             linestyle, lw=3,
            #             color=COLORS[i % len(COLORS)], alpha=0.2)
            i += 1
            i_q += 1 if hyper['use_only_primitive_actions'] == 1 else 0
            i_qet += 1 if hyper['use_only_primitive_actions'] == 0 else 0
    #   ax.set_title(subplot_title, fontdict=dict(fontsize=fontsize))
    make_axis_nice(ax, fontsize)
    ax.set_xlabel("Episodes")
    ax.set_ylabel(subplot_title)
    #   ticks = list(range(len(mean)))[0::len(mean)//num_ticks]
    #   ticks.append(len(mean))
    #   ax.set_xticks(ticks)
    #   ticklabels = list(range(len(mean)))[0::len(mean)//num_ticks]
    #   ticklabels.append(len(mean))
    #   ax.set_xticklabels(["{0:1.1e}".format(int(xi)*4) for xi in ticklabels])

    ylim = ax.get_ylim()
    d = (np.minimum(fig_lims[data_key][1], ylim[1]) -
         np.maximum(fig_lims[data_key][0], ylim[0])) / 100
    ax.set_ylim(np.maximum(fig_lims[data_key][0], ylim[0]) - d,
                np.minimum(fig_lims[data_key][1], ylim[1]) + d)
    return ax


def plot_statistics(env_id, agents,
                    statistics,
                    custom_hyperparams,
                    hyperparams,
                    seeds, num_ticks=5,
                    smoothing=10,
                    fontsize=20,
                    cols=4,
                    rows=4,
                    legend_bbox_to_anchor=(0., 1.05),
                    legend_ncol=6,
                    legend_loc='upper left',
                    ):
    fig = plt.figure(figsize=(cols * 9, rows * 7))
    fig.suptitle(env_id, fontsize=fontsize)
    stats_index = 1
    handle_list = []
    label_list = []
    for stats_key, stats in statistics.items():
        if len(stats["data"]) > 0:
            ax = plt.subplot(rows, cols, stats_index)
            ax = add_subplot(agents=agents, env_id=env_id, smoothing=smoothing, hyperparams=hyperparams,
                             seeds=seeds, custom_hyperparams=custom_hyperparams,
                             ax=ax, fontsize=fontsize, data=stats["data"], subplot_title=stats["description"],
                             data_key=stats_key, num_ticks=num_ticks)
            handles, labels = ax.get_legend_handles_labels()
            for handle, label in zip(handles, labels):
                if label not in label_list:
                    handle_list.append(handle)
                    label_list.append(label)
            stats_index += 1
            fig.add_subplot(ax)
    #   print(label_list)
    fig.legend(handle_list, label_list,
               loc=legend_loc, prop={'size': fontsize},
               bbox_to_anchor=legend_bbox_to_anchor,
               ncol=legend_ncol)  # , fancybox=True, shadow=True)
    #   fig.tight_layout(pad=5)
    #   fig.subplots_adjust(top=0.80)
    fig.show()


#   save_fig_pdf(fig, 'graph')

LINESTYLES = ["-", "--", ":",  ".-"]
LINEWIDTHS = [1., 2., 3., 4., 5., 6.]


def plot_options_statistics(env_id, agents,
                            option_statistics,
                            custom_hyperparams,
                            hyperparams,
                            seeds, num_ticks=5,
                            smoothing=10,
                            fontsize=20,
                            cols=4,
                            rows=4,
                            legend_bbox_to_anchor=(0., 1.05),
                            legend_ncol=6,
                            legend_loc='upper left',
                            ):
    fig = plt.figure(figsize=(cols * 9, rows * 7))
    fig.suptitle(env_id, fontsize=fontsize)
    stats_index = 1
    handle_list = []
    label_list = []
    option_subplots = {}
    for option, statistics in enumerate(option_statistics):
        linestyle = LINESTYLES[option]
        for stats_key, stats in statistics.items():
            if len(stats["data"]) > 0:
                if option == 0:
                    ax = plt.subplot(cols, rows, stats_index)
                    option_subplots[stats_key] = ax
                    stats_index += 1
                else:
                    ax = option_subplots[stats_key]

                ax = add_subplot(agents=[r"{}[$o_{}$]".format(agent, option) for agent in agents], linestyle=linestyle,
                                 env_id=env_id,
                                 smoothing=smoothing, hyperparams=hyperparams,
                                 seeds=seeds, custom_hyperparams=custom_hyperparams,
                                 ax=ax, fontsize=fontsize, data=stats["data"], subplot_title=stats["description"],
                                 data_key=stats_key, num_ticks=num_ticks)

        for stats_key, stats in statistics.items():
            ax = option_subplots[stats_key]
            handles, labels = ax.get_legend_handles_labels()
            for handle, label in zip(handles, labels):
                if label not in label_list:
                    handle_list.append(handle)
                    label_list.append(label)
            fig.add_subplot(ax)

    #   print(label_list)
    fig.legend(handle_list, label_list,
               loc=legend_loc, prop={'size': fontsize},
               bbox_to_anchor=legend_bbox_to_anchor,
               ncol=legend_ncol)  # , fancybox=True, shadow=True)
    #   fig.tight_layout(pad=5)
    #   fig.subplots_adjust(top=0.90)
    fig.show()