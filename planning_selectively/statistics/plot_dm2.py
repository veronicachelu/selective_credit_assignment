from collections.abc import Iterable
from matplotlib.ticker import ScalarFormatter
from imports import *
from statistics import *
from utils import *
xfmt = ScalarFormatter(useMathText=True)
xfmt.set_powerlimits((-3, 3))  # Or whatever your limits are . . .

def graph_list2(_df_list, x_list, y_list, group_list, facet,
               max_over_list, pick_list,
               log=True, window=10, num_ticks=10,
               legend_ncols=6,
               legend_loc='upper center',
               fontsize=20, axis_fn=None,
               legend_bbox=(-0.7, 1.3),
               label_fn=None, facet_fn=None,
               ls_fn_list=None, alpha_fn=None,
               per_ax_legend=False):
    # window = 2
    handle_list = []
    label_list = []
    fig = plt.figure(figsize=(5, 4))
    axs = []
    plt_nr = 0
    handle_list, label_list = [], []
    for index, (_df, x, y, group, max_over, pick) in\
            enumerate(zip(_df_list, x_list, y_list, group_list,
                        max_over_list, pick_list)):
        axs_index = []
        # Support empty groups
        if not group:
            _df['dummy_group'] = 1
            group = ['dummy_group']

        # Filter.
        if pick:
            _df = _df[reduce(lambda a, b: a & b, [f(_df) for f in pick])]

            axis_fn = axis_fn if axis_fn is not None else lambda x: x
            label_fn = label_fn if label_fn is not None else lambda group, *g: "{}".format(*g)
            facet_fn = facet_fn if facet_fn is not None else lambda g_name, g_val: '{}:{}'.format(g_name, g_val)
            _fn = label_fn if label_fn is not None else lambda x: x
            ls_fn = ls_fn_list[index] if ls_fn_list[index] is not None else lambda _: '-'
            alpha_fn = alpha_fn if alpha_fn is not None else lambda _: 0.8
            # We will first mean over anything not specified above
            # Min over certain hyper-parameters
            param_set = set(facet) | set(group) | {x} | set(max_over)

            tmp = _df[list(param_set | {y})]
            print('mean over all except {}'.format(list(param_set)))
            tmp_std = tmp.groupby(list(param_set)).agg(np.std).reset_index()
            tmp_count = tmp.groupby(list(param_set)).count().reset_index()
            print('mean over all except {}'.format(list(param_set)))
            tmp = tmp.groupby(list(param_set)).agg(np.mean).reset_index()
            print('max over all except {}'.format(list(param_set - set(max_over))))
            tmp = tmp.groupby(list(param_set - set(max_over))).agg(np.max).reset_index()
            tmp = pd.merge(tmp, tmp_std, how='inner', on=list(param_set), suffixes=('', '_std'))
            tmp = pd.merge(tmp, tmp_count, how='inner', on=list(param_set), suffixes=('', '_num'))
            # Plot
            facets = [tmp[f].unique() for f in facet]
            # n_facets = list(map(len, facets))
            # w, h = np.prod(n_facets[:len(n_facets) // 2]), np.prod(n_facets[len(n_facets) // 2:])

            fn = np.log if log else lambda x: x

            for _fvalues, _df in tmp.groupby(facet):
                _fvalues = [_fvalues] if not isinstance(_fvalues, Iterable) else list(_fvalues)
                facet_fn = facet_fn if facet_fn is not None else (lambda facet, _fvalues: '; '.join(
                    [str(_x_) + ': ' + str(_y_) for _x_, _y_ in zip(facet, _fvalues)]))
                plt_nr += 1
                # if index == 0:
                if len(axs_index) > 0:
                    ax = plt.subplot(1, 1, plt_nr, sharey=axs_index[0])
                else:
                    ax = plt.subplot(1, 1, plt_nr)
                axs_index.append(ax)
                axs.append(ax)
                # else:
                #     ax = axs[plt_nr - 1]
                clr_nr = 0

                all_ys = []

                for *g, __df in _df.groupby(group):
                    xs, ys = smoothed(list(range(len(__df[x]))), __df[y], window=window)
                    xs_ticks = __df["agent_steps"][:len(xs)] * 4
                    _, ys_std = smoothed(list(range(len(__df[x]))), __df[y + "_std"], window=window)
                    _, ys_count = smoothed(list(range(len(__df[x]))), __df[y + "_num"], window=window)
                    label = label_fn(group, index, *g)
                    ax.plot(
                        xs_ticks, fn(ys),
                        color=COLORS[clr_nr % len(COLORS)],
                        label=label,
                        ls=ls_fn(*g), lw=5, alpha=1.)#alpha_fn(*g))

                    ax.fill_between(x=xs_ticks,
                        y1=fn(ys - ys_std / np.sqrt(ys_count)), y2=fn(ys + ys_std / np.sqrt(ys_count)),
                        color=COLORS[clr_nr % len(COLORS)], alpha=0.2)
                    # ax.set_xticks(list(range(len(__df[x])))[0::len(__df[x]) // num_ticks])
                    # ax.set_xticklabels(["{:0f}".format(xi) for xi in list(__df[x])[0::len(__df[x]) // num_ticks]])

                    # if not facet == ['dummy_facet']:
                    #     ax.set_title(facet_fn(facet, list(_fvalues)), fontdict=dict(fontsize=fontsize))

                    clr_nr += 1

                    all_ys.extend(ys)

                ax.xaxis.set_major_formatter(xfmt)
                ax.xaxis.get_offset_text().set_fontsize(fontsize)
                # ax.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
                make_axis_nice(ax, fontsize)
                ax.set_xlabel(axis_fn(x))
                ax.set_ylabel(axis_fn(y))

                if per_ax_legend:
                    ax.legend(loc=0, prop={'size': fontsize}, frameon=False)
                else:
                    handles, labels = ax.get_legend_handles_labels()

                    for handle, label in zip(handles, labels):
                        if label not in label_list:
                            handle_list.append(handle)
                            label_list.append(label)

    for ax in axs:
        d = ((np.max([ax.get_ylim()[1] for ax in axs]) -
              np.min([ax.get_ylim()[0] for ax in axs])) * 0.001)
        ax.set_ylim(np.min([ax.get_ylim()[0] - d for ax in axs]),
                    np.max([ax.get_ylim()[1] + d for ax in axs]))

    if not per_ax_legend:
        a = list(zip(label_list, handle_list))
        a.sort(key=lambda x: x[0])
        label_list = [x[0] for x in a]
        handle_list = [x[1] for x in a]
        leg = fig.legend(flip(handle_list, 3), flip(label_list, 3),
                   prop={'size': fontsize},
                   labelspacing=2,
                   columnspacing=0.1,
                   handletextpad=0.1,
                   handlelength=1.,
                         frameon=False,
                   loc=legend_loc, bbox_to_anchor=legend_bbox,
                   ncol=legend_ncols, fancybox=False, shadow=False)
        # for legobj in leg.legendHandles:
        #     legobj.set_linesize(2.0)
    fig.tight_layout()
    save_fig_pdf(fig, 'graph_fst_page')

def flip(items, ncol):
    return itertools.chain(*[items[i::ncol] for i in range(ncol)])
