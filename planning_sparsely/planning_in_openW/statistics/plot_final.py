import matplotlib.pyplot as plt
import numpy as np

from imports import *
from .plot_stats import *
from envs import *
import os

def plot_option(env, v_values, true_v_values, policy,
                text_values=True, path=None,
                invert_y=True, update=False,
                filename=None, save=False):
    w = env._num_cols
    h = env._num_rows
    if update:
        clear_output(wait=True)
    fig = plt.figure(figsize=(21, 7))
    ax = plt.subplot(1, 3, 1)
    normalized_values = v_values
    mmin = np.min(true_v_values)
    mmax = np.max(true_v_values)
    normalized_values = normalized_values - mmin
    normalized_values = normalized_values / mmax
    for x, y in itertools.product(range(w), range(h)):
        state_idx = env._get_state_idx(y, x)
        if invert_y:
            yy = h - y - 1

        xy = np.array([x, yy])
        val = normalized_values[state_idx]
        color = PLOT_CMAP(val)
        ax.add_patch(Rectangle(xy - 0.5, 1, 1, color=color))

        if env._matrix_mdp[y][x] == -1:
            ax.add_patch(
                patches.Rectangle(
                    (x - 0.5, yy - 0.5),  # (x,y)
                    1.0,  # width
                    1.0,  # height
                    facecolor="gray"
                )
            )
    ax.set_xticks(np.arange(-1, w + 1, 1))
    ax.set_yticks(np.arange(-1, h + 1, 1))
    fig.add_subplot(ax)

    ax = plt.subplot(1, 3, 2)
    # normalized_values = q_values
    # normalized_values = normalized_values - np.min(normalized_values)
    # normalized_values = normalized_values/np.max(normalized_values)
    # for x, y in itertools.product(range(w), range(h)):
    #     state_idx = env._get_state_idx(y, x)
    #     if invert_y:
    #         y = h-y-1

    #     xy = np.array([x, y])
    #     xy3 = np.expand_dims(xy, axis=0)

    #     for a in range(len(env.get_action_set())):
    #         val = normalized_values[state_idx,a]
    #         og_val = q_values[state_idx,a]

    #         patch_offset, txt_offset = ACT_OFFSETS[a]

    #         color = PLOT_CMAP(val)
    #         ax.add_patch(Polygon(xy3+patch_offset, True, color=color))
    normalized_values = true_v_values
    normalized_values = normalized_values - mmin
    normalized_values = normalized_values / mmax
    for x, y in itertools.product(range(w), range(h)):
        state_idx = env._get_state_idx(y, x)
        if invert_y:
            yy = h - y - 1

        xy = np.array([x, yy])
        val = normalized_values[state_idx]
        color = PLOT_CMAP(val)
        ax.add_patch(Rectangle(xy - 0.5, 1, 1, color=color))
        if env._matrix_mdp[y][x] == -1:
            ax.add_patch(
                patches.Rectangle(
                    (x - 0.5, yy - 0.5),  # (x,y)
                    1.0,  # width
                    1.0,  # height
                    facecolor="gray"
                )
            )
    ax.set_xticks(np.arange(-1, w + 1, 1))
    ax.set_yticks(np.arange(-1, h + 1, 1))
    fig.add_subplot(ax)

    ax = plt.subplot(1, 3, 3)
    for x, y in itertools.product(range(w), range(h)):
        state_idx = env._get_state_idx(y, x)
        a = np.argmax(policy[state_idx])

        if invert_y:
            yy = h - y - 1

        xy = np.array([x, yy])

        dx = 0
        dy = 0
        if a == 0:  # up
            dy = 0.3
        elif a == 1:  # right
            dx = 0.3
        elif a == 2:  # down
            dy = -0.3
        elif a == 3:  # left
            dx = -0.3

        if env._matrix_mdp[y][x] != -1:
            plt.arrow(x + 0.5, yy + 0.5, dx, dy,
                      head_width=0.15, head_length=0.15, fc='k', ec='k')
        if env._matrix_mdp[y][x] == -1:
            ax.add_patch(
                patches.Rectangle(
                    (x, yy),  # (x,y)
                    1.0,  # width
                    1.0,  # height
                    facecolor="gray"
                )
            )

    ax.set_xticks(np.arange(-1, w + 1, 1))
    ax.set_yticks(np.arange(-1, h + 1, 1))

    for i in range(env._num_cols):
        plt.axvline(i, color='k', linestyle=':')
    plt.axvline(env._num_cols, color='k', linestyle=':')

    for j in range(env._num_rows):
        plt.axhline(j, color='k', linestyle=':')
    plt.axhline(env._num_rows, color='k', linestyle=':')

    fig.add_subplot(ax)
    plt.grid()

    if save:
        path = os.path.join("./plots", path)
        filename = os.path.join(path, filename)
        if not os.path.exists(path):
            os.makedirs(path)
        plt.savefig(f"{filename}.png", bbox_inches='tight')
    else:
        plt.show()


def plot_final2(agents,
                options_stats,
                options_pi,
                options_v_pi,
                options_true_v_pi,
                env,
                exclude,
                save,
                log_every,
                option_epsilon_list,
                action_repeats_list,
                reps_list,
                custom_hyperparams,
                hyperparams,
                seeds, num_ticks=5,
                smoothing=10,
                fontsize=20,
                align_axis=True,
                legend_bbox_to_anchor=(0., 1.05),
                legend_ncol=6,
                legend_loc='upper left',
                verbose=False,
                ):
    handle_list = []
    label_list = []
    num_options = 2
    num_action_repeats = len(action_repeats_list)
    num_reps = len(reps_list)
    fig = plt.figure(figsize=(7 * num_action_repeats * num_reps, 9 * num_options))
    desc = "VE"
    axs = []
    for option_epsilon in option_epsilon_list:
        for action_repeats in action_repeats_list:
            for reps in reps_list:
                for option in range(2):
                    stats = {"data": options_stats[option],
                             "description": r"${}[o_{}]$".format(desc, option)}
                    if len(stats["data"]) > 0:
                        ax = plt.subplot(num_options, action_repeats * reps, 1 + option)
                        ax = add_subplot(agents=[r"{}[$o_{}$]".format(agent, option) for agent in agents],
                                         linestyle='-', log_every=log_every, exclude=exclude,
                                         smoothing=smoothing, hyperparams=hyperparams,
                                         seeds=seeds, custom_hyperparams=custom_hyperparams,
                                         ax=ax, fontsize=fontsize, data=stats["data"],
                                         subplot_title=stats["description"],

                                         data_key="value_errors", num_ticks=num_ticks)

                        axs.append(ax)
                        handles, labels = ax.get_legend_handles_labels()
                        for handle, label in zip(handles, labels):
                            if label not in label_list:
                                handle_list.append(handle)
                                label_list.append(label)
                        fig.add_subplot(ax)

                # fig.align_ylabels()
                # if align_axis:
                    for ax in axs:
                        # d = ((np.max([ax.get_ylim()[1] for ax in axs]) -
                        #       np.min([ax.get_ylim()[0] for ax in axs])) * 0.01)
                        # ax.set_ylim(np.min([ax.get_ylim()[0] - d for ax in axs]),
                        #             np.max([ax.get_ylim()[1] + d for ax in axs]))
                        ax.set_ylim(2, 4)

                fig.legend(handle_list, label_list,
                           loc=legend_loc, prop={'size': fontsize},
                           bbox_to_anchor=legend_bbox_to_anchor,
                           ncol=legend_ncol)
                fig.tight_layout()
                # fig.subplots_adjust(top=0.70)
                if save:
                    plt.savefig("plots/stats.png", bbox_inches='tight')
                else:
                    fig.show()

                if verbose:
                    j = 0
                    for agent in enumerate(agents):
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
                            seed = 0
                            # if hyper["option_alpha_v"] >= 0.01:
                            #     j += 1
                            #     continue
                            # for seed in range(len(options_true_v_pi[j])):
                            for option in range(2):
                                agent_label = (r'{}{}{}{}'.format("X-" if hyper["option_eta_f"] == 0 else "",
                                                                  "E" if hyper["option_use_prior_corr"] else r'',
                                                                  "E" if hyper["use_prior_corr"] else r'',
                                                                  r'QET' if hyper['option_eta_v'] == 0 else r'Q') +
                                               "{}{}{}{}{}".format(
                                                   r'($\lambda={}$)'.format(hyper["lambda"]) if hyper[
                                                                                                    "lambda"] is not None else r'',
                                                   # r', w IS/option' if hyper["option_use_post_corr"] else r'',
                                                   # r', action rep {}' if hyper["action_repeats"] else r'',
                                                   r',action_repeats={}'.format(hyper["action_repeats"]) if hyper[
                                                                                                                "action_repeats"] is not None and (
                                                                                                                        "action_repeats" in list(
                                                                                                                    agents_params.keys()) and len(
                                                                                                                    agents_params[
                                                                                                                        "action_repeats"]) > 1 or "action_repeats" not in list(
                                                                                                                    agents_params.keys())) else '',
                                                   # r',$\alpha_z={}$'.format(hyper["alpha_z"]) if hyper["alpha_z"] is not None and len(agents_params["alpha_z"]) > 1 else '',
                                                   # r',$\alpha_f={}$'.format(hyper["alpha_f"]) if hyper["alpha_f"] is not None and ("alpha_f" in list(agents_params.keys()) and len(agents_params["alpha_f"]) > 1 or "alpha_f" not in list(agents_params.keys())) else '',
                                                   r',$\alpha^o_v={}$'.format(hyper["option_alpha_v"]) if hyper[
                                                                                                              "option_alpha_v"] is not None and (
                                                                                                                      "option_alpha_v" in list(
                                                                                                                  agents_params.keys()) and len(
                                                                                                                  agents_params[
                                                                                                                      "option_alpha_v"]) > 1 or "option_alpha_v" not in list(
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
                                                   r',$c={}$'.format(hyper["c"]) if hyper["c"] < 100 and len(
                                                       agents_params["c"]) > 1 else '')
                                               )

                                plot_option(env, options_v_pi[j][seed][option],
                                            options_true_v_pi[j][seed][option],
                                            options_pi[j][seed][option],
                                            title=r'{} -- $o_{}$'.format(agent_label, option), save=save)
                            j += 1


def plot_final3(agents,
                options_stats,
                options_pi,
                options_v_pi,
                options_true_v_pi,
                env,
                save,
                hyper,
                log_every,
                option_epsilon_list,
                action_repeats_list,
                option_alpha_z_list,
                reps_list,
                d_list,
                dz_list,
                option_alpha_v_list,
                option_eta_v_list,
                seeds, num_ticks=5,
                prefix="",
                plot_zloss=False,
                smoothing=10,
                fontsize=20,
                how_many=None,
                align_axis=True,
                legend_bbox_to_anchor=(0., 1.05),
                legend_ncol=6,
                legend_loc='upper left',
                verbose=False,
                ):

    num_options = 1
    num_action_repeats = len(action_repeats_list)
    num_reps = len(reps_list)
    num_option_epsilon = len(option_epsilon_list)
    colors = {}
    linestyles = {}
    all_lr = []
    all_lr_z = []
    for lr in list(option_alpha_v_list.values()):
        all_lr.extend(lr)
    d_list_all = []
    dz_list_all = []
    for d in list(d_list.values()):
        d_list_all.extend(d)
    for dz in list(dz_list.values()):
        dz_list_all.extend(dz)
    for lr_z in list(option_alpha_z_list.values()):
        all_lr_z.extend(lr_z)
    # all_lr = list(set(all_lr))
    # for lr_ind, lr in enumerate(all_lr):
    #     for lr_z_ind, lr_z in enumerate(all_lr_z):
    #         colors[f"{lr}_{lr_z}"] = (len(all_lr_z) * lr_ind + lr_z_ind) % len(COLORS)
    all_lr = list(set(all_lr))
    d_list_all = list(set(d_list_all))
    dz_list_all = list(set(dz_list_all))
    new_all_lr = []
    new_all_lrz = []
    for lr in all_lr:
        if lr is not None:
            new_all_lr.append(lr)
        else:
            for d in d_list_all:
                lr = f"d_{d}"
                new_all_lr.append(lr)
    for lrz in all_lr_z:
        if lrz is not None:
            new_all_lrz.append(lrz)
        else:
            for dz in dz_list_all:
                lrz = f"dz_{dz}"
                new_all_lrz.append(lrz)
    l = 0
    for lr_ind, lr in enumerate(new_all_lr):
        # for lrz_ind, lrz in enumerate(new_all_lrz):
        linestyles[f"{lr}"] = l % len(LINESTYLES)
        l += 1
    l = 0
    for option_eta_v in option_eta_v_list:
        # for lr_ind, lr in enumerate(new_all_lr):
        #     for lrz_ind, lrz in enumerate(new_all_lrz):
        colors[f"{option_eta_v}"] = l % len(COLORS)
        l += 1

    for action_repeats in action_repeats_list:
        fig = plt.figure(figsize=(5 * num_reps, 4))
        handle_list = []
        label_list = []
        # fig.suptitle(f"action repeats:{action_repeats}",
        #              fontsize=fontsize)
        idx = 1
        # for option in range(num_options):
        option = 1
        axs = []
        for option_eps_idx, option_epsilon in enumerate(option_epsilon_list):
                for rep_ind, reps in enumerate(reps_list):
                    ax = plt.subplot(1, num_reps, idx)
                    idx += 1
                    if plot_zloss:
                        description1 = "$Trace loss[o_{}]$".format(option)
                    else:
                        description1 = "VE$[o_{}]$".format(option)
                    description2 = r"$\epsilon_r = {}, \epsilon_P = 0.05, \epsilon_o = {}$".format(reps, option_epsilon)
                    for option_alpha_v in option_alpha_v_list[reps]:
                        if option_alpha_v is not None:
                            d_list_temp = [None]
                        else:
                            d_list_temp = d_list[reps]
                        for d in d_list_temp:
                            for option_eta_v in option_eta_v_list:
                                for option_alpha_z in option_alpha_z_list[option_eta_v]:
                                    if option_alpha_z is not None:
                                        dz_list_temp = [None]
                                    else:
                                        dz_list_temp = dz_list[reps]
                                    for dz in dz_list_temp:
                                        option_alpha_v_temp = option_alpha_v if option_alpha_v is not None else f"d_{d}"
                                        option_alpha_z_temp = option_alpha_z if option_alpha_z is not None else f"dz_{dz}"
                                        hyper["option_alpha_v"] = [option_alpha_v_temp]
                                        hyper["option_alpha_z"] = [option_alpha_z_temp]
                                        hyper["option_eta_v"] = [option_eta_v]
                                        hyper["action_repeats"] = [action_repeats]
                                        hyper["option_epsilon"] = [option_epsilon]
                                        agent_label = (r'{}'.format(r'ET' if option_eta_v == 0 else r'TD'))
                                                       # +
                                                       # "{}{}{}{}".format(
                                                           # r'($\lambda={}$)'.format(hyper["lambda"][0]),
                                                           # r',action_repeats={}'.format(action_repeats),
                                                           # r'$\alpha^o_v={}$'.format(option_alpha_v) if option_alpha_v is not None else r'$d={}$'.format(d),
                                                           # r")" if option_eta_v == 1 else r'',
                                                           # (r',$\alpha^o_z={}$'.format(option_alpha_z) if option_alpha_z is not None else r',$d_z={}$'.format(dz)) if option_eta_v == 0 else r'',
                                                           # r")" if option_eta_v == 0 else r''))
                                        # linestyle = '-' if option_eta_v == 1 else '--'
                                        linestyle = '-' #LINESTYLES[linestyles[f"{option_alpha_v_temp}"]]
                                        data = []
                                        for seed in range(seeds):
                                            key = f"{seed}_{option_alpha_v_temp}_{option_alpha_z_temp}_{option_eta_v}_{action_repeats}_{option_epsilon}_{reps}"
                                            if key not in options_stats[option]:
                                                continue
                                            data_per_seed = options_stats[option][key] if how_many is None else options_stats[option][key][:how_many]
                                            data.append(data_per_seed)
                                        data = np.asarray(data)
                                        if len(data) == 0:
                                            continue
                                        mean = np.mean(data, axis=0)
                                        data_std = np.std(data, axis=0)
                                        x_mean, mean = smooth(range(0, len(mean)), mean, smoothing)
                                        x_mean = [x * log_every for x in x_mean]
                                        # if option_eta_v == 1:
                                        #     option_alpha_z = option_alpha_z_list[0][0]
                                        # color = colors[f"{option_alpha_v}_{option_alpha_z}"]
                                        # {option_alpha_v_temp}
                                        # _
                                        # {option_alpha_z_temp}
                                        # # _
                                        color = colors[f"{option_eta_v}"]
                                        # if log_every is not None:
                                        # x_mean = range(0, len(mean) * log_every, log_every)
                                        # else:
                                            # x_mean = range(0, len(mean), smoothing)
                                        _, data_std = smooth(range(0, len(mean)), data_std, smoothing)
                                        ax.plot(x_mean, mean,
                                                ls=linestyle, lw=5,
                                                color=COLORS[color], alpha=1.,
                                                label=agent_label)
                                        ax.fill_between(x=x_mean, y1=mean - data_std / np.sqrt(seeds),
                                                        y2=mean + data_std / np.sqrt(seeds),
                                                        color=COLORS[color], alpha=0.2)

                    make_axis_nice(ax, fontsize, smoothing)
                    ax.set_xlabel("Episodes")
                    ax.set_ylabel(description1)
                    ax.set_title(description2, fontsize=fontsize)
                    axs.append(ax)
                    handles, labels = ax.get_legend_handles_labels()

                    for handle, label in zip(handles, labels):
                        if label not in label_list:
                            handle_list.append(handle)
                            label_list.append(label)
                    fig.add_subplot(ax)
            # if align_axis:
            #     for ax in axs:
            #         aux_d = ((np.max([ax.get_ylim()[1] for ax in axs]) -
            #               np.min([ax.get_ylim()[0] for ax in axs])) * 0.001)
            #         ax.set_ylim(np.min([ax.get_ylim()[0] - aux_d for ax in axs]),
            #                     np.max([ax.get_ylim()[1] + aux_d for ax in axs]))
                    ax.set_ylim(2, 9)
        # idx = list(np.argsort(label_list))
        # label_list = label_list[idx]
        # handle_list = handle_list[idx]
        a = list(zip(label_list, handle_list))
        a.sort(key=lambda x: x[0])
        label_list = [x[0] for x in a]
        handle_list = [x[1] for x in a]
        fig.legend(handle_list, label_list,
                   loc=legend_loc, prop={'size': fontsize},
                   bbox_to_anchor=legend_bbox_to_anchor,
                   ncol=legend_ncol,
                   labelspacing=0.1,
                   columnspacing=1,
                   # handletextpad=0.1,
                   handlelength=1.5,
                   frameon=False,
                   # columnspacing=0.5
                   )
        fig.tight_layout()
        if save:
            if plot_zloss:
                fig.savefig(f"plots/{prefix}trace_loss_AR_{action_repeats}.png", bbox_inches='tight')
            else:
                fig.savefig(f"plots/{prefix}ve_AR_{action_repeats}.png", bbox_inches='tight')
        else:
            fig.show()

    if verbose:
        for action_repeats in action_repeats_list:
            for option in range(num_options):
                if option == 0:
                    for option_eps_idx, option_epsilon in enumerate(option_epsilon_list):
                        for rep_ind, reps in enumerate(reps_list):
                            for option_eta_v in option_eta_v_list:
                                for option_alpha_v in option_alpha_v_list[reps]:
                                    if option_alpha_v is not None:
                                        d_list_temp = [None]
                                    else:
                                        d_list_temp = d_list[reps]
                                    for d in d_list_temp:
                                        for option_alpha_z in option_alpha_z_list[option_eta_v]:
                                            if option_alpha_z is not None:
                                                dz_list_temp = [None]
                                            else:
                                                dz_list_temp = dz_list[reps]
                                            for dz in dz_list_temp:
                                                option_alpha_v_temp = option_alpha_v if option_alpha_v is not None else f"d_{d}"
                                                option_alpha_z_temp = option_alpha_z if option_alpha_z is not None else f"dz_{dz}"
                                                key0 = f"0_{option_alpha_v_temp}_{option_alpha_z_temp}_{option_eta_v}_{action_repeats}_{option_epsilon}_{reps}"
                                                if key0 not in options_stats[option]:
                                                    continue
                                                aux = r'ET' if option_eta_v == 0 else r'TD'
                                                # aux2 = r"" if option_eta_v == 1 else f"{option_alpha_z}"
                                                aux2 = r"" if (option_eta_v == 1 or (len(dz_list_temp) == 1 and len(option_alpha_z_list) == 1)) else f"_{option_alpha_z_temp}"
                                                plot_option(env[key0], options_v_pi[key0][option],
                                                            options_true_v_pi[key0][option],
                                                            options_pi[key0][option],
                                                            path=f"{action_repeats}/{reps}/{option_epsilon}/{d}",
                                                            filename=f"{aux}{aux2}",
                                                            save=save)
