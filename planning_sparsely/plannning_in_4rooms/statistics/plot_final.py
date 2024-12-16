# @title plot final
import matplotlib.pyplot as plt
import numpy as np
from utils import *
from imports import *
# from .plot_stats import *
from envs import *
# from plot_stats import *
from .plot_stats import *
import os

def plot_pio(env, q_pi, options_q_pi,
             interest, option_interests,
             filename, path,
             text_values=True,
             invert_y=True,
             update=False,
             title=None,
             fontsize=25,
             save=True,
             subtitle=None):
    w = env._num_cols
    h = env._num_rows

    fig = plt.figure(figsize=(9, 9))

    ax = fig.gca()

    for oo in range(len(option_interests)):
        q_values = options_q_pi[oo]
        normalized_values = q_values
        normalized_values = normalized_values - np.min(normalized_values) #+ 0.5
        normalized_values = normalized_values / np.max(normalized_values)
        normalized_values -= 0.2
        option_interest = option_interests[oo]
        for x, y in itertools.product(range(w), range(h)):
            if y == env._goal_x and x == env._goal_y:
                continue
            state_idx = env._get_state_idx(y, x)

            if invert_y:
                yy = h - y - 1

            # xy = np.array([x, yy])
            xy = np.array([x, yy])
            xy3 = np.expand_dims(xy, axis=0)

            op_offset = OP_OFFSETS[oo]
            if interest[state_idx] > 0:
                # if text_values:
                xy_text = xy
                a = np.argmax(q_values[state_idx])

                dx = 0
                dy = 0
                if a == 0:  # up
                    dy = 0.1
                elif a == 1:  # right
                    dx = 0.1
                elif a == 2:  # down
                    dy = -0.1
                elif a == 3:  # left
                    dx = -0.1
                # elif env._matrix_mdp[y][x] != -1 and option_interest[state_idx] == 0: # termination
                #     circle = plt.Circle(
                #         (x + 0.5, env._num_rows - y + 0.5 - 1), 0.025, color='k')
                #     ax.add_artist(circle)
                plt.arrow(xy_text[0] + op_offset[0] + 0.1, xy_text[1] + op_offset[1] + 0.2, dx, dy,
                          head_width=0.1, head_length=0.1,
                          fc='k',
                          ec='k')
                          # fc=OP_COLORS[oo % len(OP_COLORS)],
                          # ec=OP_COLORS[oo % len(OP_COLORS)])

                ax.text(xy_text[0] + op_offset[0] + 0., xy_text[1] + op_offset[1] + 0.1,
                        r"$o_{}$".format(int(oo)), size='xx-large',
                        # c=OP_COLORS[oo % len(OP_COLORS)])
                        c='k')
                color = PLOT_CMAP(val)
                ax.add_patch(Rectangle(xy + op_offset, 0.5, 0.5, color=color))

            if env._matrix_mdp[y][x] != -1 and interest[state_idx] == 0 and option_interest[state_idx] > 0:
                for aa in range(len(env.get_action_set())):
                    val = normalized_values[state_idx, aa]
                    patch_offset, txt_offset = ACT_OFFSETS[aa]
                    color = PLOT_CMAP(val)
                    ax.add_patch(Polygon(xy3 + patch_offset + 0.5, True, color=color))

                a = np.argmax(q_values[state_idx])

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
                # elif env._matrix_mdp[y][x] != -1 and option_interest[state_idx] == 0: # termination
                #     circle = plt.Circle(
                #         (x + 0.5, env._num_rows - y + 0.5 - 1), 0.025, color='k')
                #     ax.add_artist(circle)
                plt.arrow(x + 0.5, env._num_rows - y + 0.5 - 1, dx, dy,
                          head_width=0.15, head_length=0.15,
                          fc='k',
                          ec='k')
                          # fc=OP_COLORS[oo % len(OP_COLORS)],
                          # ec=OP_COLORS[oo % len(OP_COLORS)])

    normalized_values = q_pi
    maxim = np.max(normalized_values)
    minim = np.min(normalized_values)
    normalized_values = normalized_values - minim
    normalized_values = normalized_values / maxim
    maxim = np.max(normalized_values)
    minim = np.min(normalized_values)
    normalized_values = normalized_values - minim #+ 0.9
    normalized_values = normalized_values / maxim#+0.9)
    normalized_values -= 0.2
    # normalized_values = np.maximum(normalized_values, np.zeros_like(normalized_values))
    # normalized_values = np.minimum(normalized_values, np.ones_like(normalized_values))

    for x, y in itertools.product(range(w), range(h)):
        state_idx = env._get_state_idx(y, x)
        # o = np.argmax(policy[state_idx])
        # o_i = []
        # for option, option_interest in enumerate(option_interests):
        #     o_i.append(option_interest[state_idx])
        # q = np.where(np.asarray(o_i) == 0, -np.inf, q_pi[state_idx])
        q = q_pi[state_idx]
        if invert_y:
            yy = h - y - 1

        xy = np.array([x, yy])
        # xy4 = np.expand_dims(xy, axis=0)

        # xy_text = xy

        if interest[state_idx] > 0:
            for oo in range(len(option_interests)):
                oop_offset = OP_OFFSETS[oo]
                val = normalized_values[state_idx][oo]
                # if option_interests[oo][state_idx] == 0:
                #     ax.add_patch(Rectangle(xy+oop_offset, 0.5, 0.5, fill=None, color="k", hatch='///'))
                # else:
                #     # if text_values:
                #     #     xy_text = xy+txt_offset
                #     #     ax.text(xy_text[0], xy_text[1], '%.1f'%og_val, size='small')
                color = PLOT_CMAP2(val)
                ax.add_patch(Rectangle(xy + oop_offset, 0.5, 0.5, color=color))

            op_offset = OP_OFFSETS[np.argmax(q)]
            ax.add_patch(Rectangle(xy + op_offset, 0.5, 0.5, fill=None, color="k"))

            # Polygon(xy4+op_offset, True, color=color))
            # ax.text(xy_text[0] + 0.5, xy_text[1]+0.5, str(int(o)), size='small', color=COLORS[o % len(COLORS)])
            # color = PLOT_CMAP(val)
            # ax.add_patch(Rectangle(xy, 1, 1, color=color))
        if env._matrix_mdp[y][x] == -1:
            ax.add_patch(
                patches.Rectangle(
                    (x, env._num_rows - y - 1),  # (x,y)
                    1.0,  # width
                    1.0,  # height
                    facecolor="k"
                )
            )
        if y == env._goal_x and x == env._goal_y:
            ax.add_patch(
                patches.Rectangle(
                    (x, env._num_rows - y - 1),  # (x,y)
                    1.0,  # width
                    1.0,  # height
                    facecolor="grey"
                )
            )
            # if invert_y:
            #     yy = h - y - 1

            # xy = np.array([x, yy])
            # plt.arrow(x + 0.5, env._num_rows - y + 0.5 - 1, dx, dy,
            #           head_width=0.15, head_length=0.15, fc=OP_COLORS[oo % len(OP_COLORS)],
            #           ec=OP_COLORS[oo % len(OP_COLORS)])

            ax.text(x + 0.3, env._num_rows - y + 0.3 - 1, "G", size='xx-large',
                    c='k')

    ax.set_xlim([0, env._num_cols])
    ax.set_ylim([0, env._num_rows])

    for i in range(env._num_cols):
        plt.axvline(i, color='k', linestyle=':')
    plt.axvline(env._num_cols, color='k', linestyle=':')

    for j in range(env._num_rows):
        plt.axhline(j, color='k', linestyle=':')
    plt.axhline(env._num_rows, color='k', linestyle=':')

    # plt.grid()
    plt.xticks([])
    plt.yticks([])
    ax.set_title(subtitle, fontsize=fontsize)
    fig.add_subplot(ax)
    if save:
        path = os.path.join("./plots", path)
        filename = os.path.join(path, filename)
        if not os.path.exists(path):
            os.makedirs(path)
        plt.savefig(f"{filename}.png", bbox_inches='tight')
       # fig.savefig(f"plots/value_{filename}.png", bbox_inches='tight')
    else:
        fig.show()


def plot_pi(env, q_pi, interest, filename,
            text_values=True,
            invert_y=True,
            update=False,
            title=None,
            fontsize=25, path=None,
            save=True,
            subtitle=None):
    w = env._num_cols
    h = env._num_rows

    fig = plt.figure(figsize=(9, 9))
    ax = fig.gca()
    interest = np.ones_like(interest)
    normalized_values = q_pi
    normalized_values = normalized_values - np.min(normalized_values)  # + 0.8
    normalized_values = normalized_values / np.max(normalized_values)
    # normalized_values -= 0.4

    for x, y in itertools.product(range(w), range(h)):
        state_idx = env._get_state_idx(y, x)
        if invert_y:
            yy = h - y - 1
        xy = np.array([x, yy])
        xy3 = np.expand_dims(xy, axis=0)

        if interest[state_idx] > 0 and env._matrix_mdp[y][x] != -1 and not (y == env._goal_x and x == env._goal_y):
            for aa in range(len(env.get_action_set())):
                val = normalized_values[state_idx, aa]
                patch_offset, txt_offset = ACT_OFFSETS[aa]
                color = PLOT_CMAP(val)
                ax.add_patch(Polygon(xy3 + patch_offset + 0.5, True, color=color))

            a = np.argmax(q_pi[state_idx])

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
            plt.arrow(x + 0.5, env._num_rows - y + 0.5 - 1, dx, dy,
                      head_width=0.15, head_length=0.15, fc="k", ec="k")

        if env._matrix_mdp[y][x] == -1:
            ax.add_patch(
                patches.Rectangle(
                    (x, env._num_rows - y - 1),  # (x,y)
                    1.0,  # width
                    1.0,  # height
                    facecolor="k"
                )
            )
        if y == env._goal_x and x == env._goal_y:
            ax.add_patch(
                patches.Rectangle(
                    (x, env._num_rows - y - 1),  # (x,y)
                    1.0,  # width
                    1.0,  # height
                    facecolor="grey"
                )
            )
            # if invert_y:
            #     yy = h - y - 1

            # xy = np.array([x, yy])
            # plt.arrow(x + 0.5, env._num_rows - y + 0.5 - 1, dx, dy,
            #           head_width=0.15, head_length=0.15, fc=OP_COLORS[oo % len(OP_COLORS)],
            #           ec=OP_COLORS[oo % len(OP_COLORS)])

            ax.text(x + 0.3, env._num_rows - y + 0.3 - 1, "G", size='xx-large',
                    c='k')
    ax.set_xlim([0, env._num_cols])
    ax.set_ylim([0, env._num_rows])

    for i in range(env._num_cols):
        plt.axvline(i, color='k', linestyle=':')
    plt.axvline(env._num_cols, color='k', linestyle=':')

    for j in range(env._num_rows):
        plt.axhline(j, color='k', linestyle=':')
    plt.axhline(env._num_rows, color='k', linestyle=':')

    # plt.grid(True)
    plt.xticks([])
    plt.yticks([])
    ax.set_title(subtitle, fontsize=fontsize)
    fig.add_subplot(ax)

    if save:
        path = os.path.join("./plots", path)
        filename = os.path.join(path, filename)
        if not os.path.exists(path):
            os.makedirs(path)
        plt.savefig(f"{filename}.png", bbox_inches='tight')
        # fig.savefig(f"plots/value_{filename}.png", bbox_inches='tight')
    else:
        fig.show()


def plot_final(agents,
               options_q_star,
               q_pi,
               q_star,
               option_interests,
               interest,
               statistics,
               custom_hyperparams,
               hyperparams,
               seeds, num_ticks=5,
               smoothing=10,
               fontsize=20,
               legend_bbox_to_anchor=(0., 1.05),
               legend_ncol=6,
               legend_loc='upper left'):
    j = 0
    for agent_id, agent in enumerate(agents):
        agents_params = {}
        for k, v in hyperparams.items():
            agents_params[k] = v
            if agent in custom_hyperparams.keys() and \
                    k in custom_hyperparams[agent].keys():
                agents_params[k] = custom_hyperparams[agent][k]
        hyper = {}
        for hyper_values in itertools.product(*agents_params.values()):
            for v, k in zip(hyper_values, agents_params.keys()):
                hyper[k] = v
            seed = 0
            agent_label = (r'{}{}{}_{}'.format("X-" if hyper["eta_f"] == 0 else "",
                                               "E" if hyper["use_prior_corr"] else r'',
                                               r'QET' if hyper['eta_q'] == 0 else r'Q',
                                               r"A" if hyper["use_only_primitive_actions"] else r"O"))
                           # "{}{}{}{}{}{}{}{}{}{}".format(
                           #     r'($\lambda={}$)'.format(hyper["lambda"]) if hyper["lambda"] is not None else r'',
                           #     r', w. IS/option' if hyper["option_use_post_corr"] else r'',
                           #     r', w. IS' if hyper["use_post_corr"] else r'',
                           #     r',$\alpha_q={}$'.format(hyper["alpha_q"]) if hyper["alpha_q"] is not None and (
                           #                 "alpha_v" in list(agents_params.keys()) and len(
                           #             agents_params["alpha_v"]) > 1 or "alpha_v" not in list(
                           #             agents_params.keys())) else '',
                           #     r',$\alpha_z={}$'.format(hyper["alpha_z"]) if hyper["alpha_z"] is not None and len(
                           #         agents_params["alpha_z"]) > 1 else '',
                           #     r',$\alpha_f={}$'.format(hyper["alpha_f"]) if hyper["alpha_f"] is not None and (
                           #                 "alpha_f" in list(agents_params.keys()) and len(
                           #             agents_params["alpha_f"]) > 1 or "alpha_f" not in list(
                           #             agents_params.keys())) else '',
                           #     r',$\alpha^o_q={}$'.format(hyper["option_alpha_q"]) if hyper[
                           #                                                                "option_alpha_q"] is not None and (
                           #                                                                        "option_alpha_q" in list(
                           #                                                                    agents_params.keys()) and len(
                           #                                                                    agents_params[
                           #                                                                        "option_alpha_q"]) > 1 or "option_alpha_q" not in list(
                           #                                                                    agents_params.keys())) else '',
                           #     r',$\alpha^o_z={}$'.format(hyper["option_alpha_z"]) if hyper[
                           #                                                                "option_alpha_z"] is not None and (
                           #                                                                        "option_alpha_z" in list(
                           #                                                                    agents_params.keys()) and len(
                           #                                                                    agents_params[
                           #                                                                        "option_alpha_z"]) > 1 or "option_alpha_z" not in list(
                           #                                                                    agents_params.keys())) else '',
                           #     r',$\alpha^o_f={}$'.format(hyper["option_alpha_f"]) if hyper[
                           #                                                                "option_alpha_f"] is not None and (
                           #                                                                        "option_alpha_f" in list(
                           #                                                                    agents_params.keys()) and len(
                           #                                                                    agents_params[
                           #                                                                        "option_alpha_f"]) > 1 or "option_alpha_f" not in list(
                           #                                                                    agents_params.keys())) else '',
                           #     r',$\beta={}$'.format(hyper["beta"]) if hyper["beta"] < 100 and len(
                           #         agents_params["beta"]) > 1 else '',
                           #     r',$\beta^o={}$'.format(hyper["option_beta"]) if hyper["option_beta"] < 100 and len(
                           #         agents_params["option_beta"]) > 1 else ''))
            if hyper["use_only_primitive_actions"]:
                plot_pi(env, q_pi[j][seed],
                        interest[j][seed],
                        text_values=False,
                        title=r"{} -- $\pi_A$".format(agent_label),
                        subtitle=r"{}$_A$".format(agent_label))
            else:
                plot_pio(env, q_pi[j][seed], options_q_star[j][seed],
                         q_star[j][seed],
                         interest[j][seed],
                         option_interests[j][seed],
                         text_values=False, title=r"{} -- $\pi_O$".format(agent_label),
                         subtitle=r"{}$_O$".format(agent_label))
            j += 1


def plot_final2(agents,
               options_q_star,
               q_pi,
               q_star,
               reps_list,
               options_stats,
               options_pi,
               options_q_pi,
               options_true_q_pi,
               env,
               log_every,
               alpha_q_list,
               alpha_z_list,
               d_list,
               dz_list,
               eta_q_list,
               use_only_primitive_actions_list,
               discounts,
               save,
               seeds,
               option_interests,
               interest,
               stats,
               custom_hyperparams,
               hyper,
               prefix="",
               verbose=False,
               num_ticks=5,
               align_axis=True,
               smoothing=10,
               fontsize=20,
               legend_bbox_to_anchor=(0., 1.05),
               legend_ncol=6,
how_many=None,
               legend_loc='upper left'):
    colors = {}
    all_lr = []
    all_lrz = []
    for lr in list(alpha_q_list.values()):
        all_lr.extend(lr)
    for lrz in list(alpha_z_list.values()):
        all_lrz.extend(lrz)
    all_lr = list(set(all_lr))
    all_lrz = list(set(all_lrz))
    new_all_lr = []
    new_all_lrz = []
    d_list_all = []
    dz_list_all = []
    for d_ in list(d_list.values()):
        for d in list(d_.values()):
            d_list_all.extend(d)
    for dz_ in list(dz_list.values()):
        for dz in list(dz_.values()):
            dz_list_all.extend(dz)
    dz_list_all = list(set(dz_list_all))
    d_list_all = list(set(d_list_all))
    for lr in all_lr:
        if lr is not None:
            new_all_lr.append(lr)
        else:
            for d in d_list_all:
                lr = f"d_{d}"
                new_all_lr.append(lr)
    for lrz in all_lrz:
        if lrz is not None:
            new_all_lrz.append(lrz)
        else:
            for dz in dz_list_all:
                lrz = f"dz_{dz}"
                new_all_lrz.append(lrz)
    l = 0
    linewidths = {}
    linestyles = {}
    for lrz_ind, lrz in enumerate(new_all_lrz):
        linewidths[f"{lrz}"] = l % len(LINEWIDTHS)
        l += 1
    # for eta_q in eta_q_list:
    #     linestyles[f"{eta_q}"] = l % len(LINESTYLES)
    #     l += 1
    l = 0
    for lr_ind, lr in enumerate(new_all_lr):
        linestyles[f"{lr}"] = l % len(LINESTYLES)
        l += 1
    # l = 0
    for use_only_primitive_actions in use_only_primitive_actions_list:
        for discount in discounts[use_only_primitive_actions]:
            colors[f"{use_only_primitive_actions}_{discount}"] = l % len(OP_COLORS)
        # for option_eta_v in option_eta_v_list:
        # for lr_ind, lr in enumerate(new_all_lr):
        #     for lrz_ind, lrz in enumerate(new_all_lrz):
        # colors[f"{option_eta_v}"] = l % len(COLORS)
            l += 1
    # l = 0
    # # for eta_q_idx, eta_q in enumerate(eta_q_list):
    # for use_only_primitive_actions in use_only_primitive_actions_list:
    #     for discount in discounts[use_only_primitive_actions]:
    #         for lr_ind, lr in enumerate(new_all_lr):
    #             # for lrz_ind, lrz in enumerate(new_all_lrz):
    #             colors[f"{lr}_{use_only_primitive_actions}_{discount}"] = l % len(OP_COLORS)
    #             l += 1
        # for eta_q_idx, eta_q in enumerate(eta_q_list):
        #     colors[f"{lr}"] = (lr_ind) % len(OP_COLORS)

    handle_list = []
    label_list = []
    fig = plt.figure(figsize=(5*2, 4*len(reps_list)))
    axs = []
    description = "Episode timesteps"
    for reps_idx, reps in enumerate(reps_list):
        for eta_q_idx, eta_q in enumerate(eta_q_list):
            ax = plt.subplot(len(reps_list), 2, reps_idx * len(reps_list) + eta_q_idx + 1)
            for use_only_primitive_actions in use_only_primitive_actions_list:
                for discount in discounts[use_only_primitive_actions]:
                    for alpha_q in alpha_q_list[use_only_primitive_actions]:
                        for alpha_z in alpha_z_list[eta_q]:
                            if alpha_q is None:
                                d_temp_list = d_list[use_only_primitive_actions][eta_q]
                            else:
                                d_temp_list = [None]
                            if alpha_z is None:
                                dz_temp_list = dz_list[use_only_primitive_actions][eta_q]
                            else:
                                dz_temp_list = [None]
                            for d in d_temp_list:
                                for dz in dz_temp_list:
                                    alpha_q_temp = alpha_q if alpha_q is not None else f"d_{d}"
                                    alpha_z_temp = alpha_z if alpha_z is not None else f"dz_{dz}"
                                    hyper["alpha_q"] = [alpha_q]
                                    hyper["eta_q"] = [eta_q]
                                    hyper["use_q_for_reward"]: [True] if use_only_primitive_actions == 0 else [False]
                                    hyper["use_magic_rule"]: [True] if use_only_primitive_actions == 0 else [False]
                                    hyper["use_true_options"]: [True] if use_only_primitive_actions == 0 else [False]
                                    hyper["learn_trace_by_magic"]: [True] if use_only_primitive_actions == 0 else [False]
                                    hyper["use_only_primitive_actions"] = [use_only_primitive_actions]
                                    # agent_label = (r'{}$_{}$'.format(r'QET' if eta_q == 0 else r'Q',
                                    #                                 r"A" if use_only_primitive_actions == 1 else r"O") +
                                    #               "{}".format(
                                    #                   # r'($\lambda={}$)'.format(hyper["lambda"][0]),
                                    #                   r'($\alpha_q={}$)'.format(alpha_q) if alpha_q is not None else r'($d={}$)'.format(d)))
                                    agent_label = (r'{}$_{}$(${}$)'.format(r'QET' if eta_q == 0 else r'Q',
                                                                    r"A" if use_only_primitive_actions == 1 else r"O",
                                                   # "{}".format(
                                                       r'\gamma={}'.format(discount) if len(discounts[use_only_primitive_actions]) > 1 else r'',
                                                       # r'$\alpha_q={}$'.format(alpha_q) if alpha_q is not None else r'$d\!:\!{}$'.format(d),
                                                       # r")" if eta_q == 1 else r'',
                                                       # (r',$\alpha_z={}$'.format(alpha_z) if alpha_z is not None else r',$d_z\!:\!{}$'.format(
                                                       #     dz)) if eta_q == 0 else r'',
                                                       # r")" if eta_q == 0 else r'')
                                                   ))
                                    # linestyle = '--' if use_only_primitive_actions == 0 else '-'
                                    linestyle = LINESTYLES[linestyles[f"{alpha_q_temp}"]]
                                    linewidth = 4 #LINEWIDTHS[linewidths[f"{alpha_z_temp}"]]

                                    data = []
                                    for seed in range(seeds):
                                        key = f"{seed}_{reps}_{alpha_q_temp}_{alpha_z_temp}_{eta_q}_{use_only_primitive_actions}_{discount}"
                                        if key not in stats:
                                            print(f"key {key} does not exists")
                                            continue
                                        data_per_seed = stats[key] if how_many is None else stats[key] [:how_many]
                                        data.append(data_per_seed)
                                    data = np.asarray(data)
                                    if len(data) == 0:
                                        continue
                                    mean = np.mean(data  , axis=0)
                                    data_std = np.std(data, axis=0)
                                    x_mean, mean = smooth(range(len(mean)), mean, smoothing)
                                    _, data_std = smooth(range(len(mean)), data_std, smoothing)
                                    color = colors[f"{use_only_primitive_actions}_{discount}"]
                                    # if log_every is not None:
                                    #     x_mean = range(0, len(mean) * smoothing * log_every, smoothing * log_every)
                                    # else:
                                    #     x_mean = range(0, len(mean) * smoothing, smoothing)

                                    ax.plot(x_mean, mean,
                                            ls=linestyle, lw=linewidth,
                                            color=OP_COLORS[color], alpha=1.,
                                            label=agent_label)
                                    ax.fill_between(x=x_mean, y1=mean - data_std / np.sqrt(seeds),
                                                    y2=mean + data_std / np.sqrt(seeds),
                                                    color=OP_COLORS[color], alpha=0.2)

                                    make_axis_nice(ax, fontsize, smoothing)
                                    fig.add_subplot(ax)
                                    # ax.set_yscale('log')
                                    # ax.set_ylim(0, 3000)
                                    ax.set_xlabel("Episodes")
                                    ax.set_ylabel(description)
                                    ax.set_title(r"$\eta_q={}, \epsilon_r={}$".format(eta_q, reps), fontsize=fontsize)
                                    #if eta_q == 0 else ('-.' if use_only_primitive_actions == 0 else '--')")


            axs.append(ax)
            handles, labels = ax.get_legend_handles_labels()
            for handle, label in zip(handles, labels):
                if label not in label_list:
                    handle_list.append(handle)
                    label_list.append(label)
    if align_axis:
        max_ax = np.max([ax.get_ylim()[1] for ax in axs])
        min_ax = np.min([ax.get_ylim()[0] for ax in axs])
        aux_d = ((max_ax -
                  min_ax) * 0.001)
        for ax in axs:
            ax.set_ylim(0, max_ax)

    a = list(zip(label_list, handle_list))
    a.sort(key=lambda x: x[0])
    label_list = [x[0] for x in a]
    handle_list = [x[1] for x in a]
    fig.legend(handle_list, label_list,
           loc=legend_loc, prop={'size': fontsize},
           bbox_to_anchor=legend_bbox_to_anchor,
           ncol=legend_ncol,
           # labelspacing=0.05,
           # columnspacing=0.1,
           #     handletextpad=0.1,
               )
    fig.tight_layout()
    if save:
        fig.savefig(f"plots/{prefix}stats.png", bbox_inches='tight')
    else:
        fig.show()

    if verbose:
        for reps_idx, reps in enumerate(reps_list):
            for eta_q_idx, eta_q in enumerate(eta_q_list):
                for use_only_primitive_actions in use_only_primitive_actions_list:
                    for discount in discounts[use_only_primitive_actions]:
                        for alpha_q in alpha_q_list[use_only_primitive_actions]:
                                for alpha_z in alpha_z_list[eta_q]:
                                    if alpha_q is None:
                                        d_temp_list = d_list[use_only_primitive_actions][eta_q]
                                    else:
                                        d_temp_list = [None]
                                    if alpha_z is None:
                                        dz_temp_list = dz_list[use_only_primitive_actions][eta_q]
                                    else:
                                        dz_temp_list = [None]
                                    for d in d_temp_list:
                                        for dz in dz_temp_list:
                                            alpha_q_temp = alpha_q if alpha_q is not None else f"d_{d}"
                                            alpha_z_temp = alpha_z if alpha_z is not None else f"dz_{dz}"
                                            hyper["alpha_q"] = [alpha_q]
                                            hyper["eta_q"] = [eta_q]
                                            hyper["use_q_for_reward"]: [True] if use_only_primitive_actions == 0 else [False]
                                            hyper["use_magic_rule"]: [True] if use_only_primitive_actions == 0 else [False]
                                            hyper["use_true_options"]: [True] if use_only_primitive_actions == 0 else [False]
                                            hyper["learn_trace_by_magic"]: [True] if use_only_primitive_actions == 0 else [
                                                False]
                                            hyper["use_only_primitive_actions"] = [use_only_primitive_actions]
                                            agent_label = (r'{}$_{}$'.format(r'QET' if eta_q == 0 else r'Q',
                                                                             r"A" if use_only_primitive_actions == 1 else r"O"),
                                                           "{}".format(
                                                               # r'($\lambda={}$)'.format(hyper["lambda"][0]),
                                                               r' $\alpha_q={}$'.format(
                                                                   alpha_q) if alpha_q is not None else r'$d={}$'.format(d)))

                                            key_0 = f"0_{reps}_{alpha_q_temp}_{alpha_z_temp}_{eta_q}_{use_only_primitive_actions}_{discount}"
                                            if key_0 not in stats:
                                                continue
                                            path = f"{reps}/{d}"
                                            aux = r'QET' if eta_q == 0 else r'Q'
                                            aux2 = r'_A' if use_only_primitive_actions else r'_O'
                                            aux3 = r"" if (eta_q == 1 or (len(dz_temp_list) == 1 and len(
                                                alpha_z_list) == 1)) else f"_{alpha_z_temp}"

                                            filename = f"{aux}{aux2}{aux3}"
                                            if use_only_primitive_actions:
                                                plot_pi(env[key_0], q_pi[key_0],
                                                        interest[key_0],
                                                        text_values=False, save=save,
                                                        title=r"{} -- $\pi_A$".format(agent_label),
                                                        path=path,
                                                        filename=filename,
                                                        subtitle=r"{}".format(agent_label))
                                            else:
                                                plot_pio(env[key_0], q_pi[key_0], options_q_star[key_0],
                                                         # q_star[key_0],
                                                         interest[key_0],
                                                         option_interests[key_0],
                                                         path=path,
                                                         filename=filename,
                                                         save=save,
                                                         text_values=False, title=r"{} -- $\pi_O$".format(agent_label),
                                                         subtitle=r"{}".format(agent_label))