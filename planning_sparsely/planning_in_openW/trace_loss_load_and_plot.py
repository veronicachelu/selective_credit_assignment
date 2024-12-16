from imports import *
from utils import *
from options import *
from statistics import *
from envs import *
import os
agents = [""]
smoothing = 1
custom_hyperparams = {}
stats_all = {}
q_pi_all = {}
options_stats_all = [{}, {}, {}, {}]
options_pi_all = {}
options_v_pi_all = {}
options_true_v_pi_all = {}
env_all = {}
seeds = 1
# option_alpha_v_list = {
#                     # 1.: [1e-1, 5e-2, 1e-2],
#                     # 0.5: [1e-2, 5e-3, 1e-3],
#                     0.5: [None],
#                     0.1: [None],
#     # 0.01: [1e-3, 5e-4, 1e-4],
#                     }
# d_list = {0.5, 0.8, 0.9}
# option_eta_v_list = [1, 0]
# action_repeats_list = [1]
# option_epsilon_list = [0.2, 0.5]
# reps_list = [0.5, 0.1]
# option_alpha_z_list = {0: [7e-1],
#                        1: [5e-1]}
# seeds = 1
# option_alpha_v_list = {
#                     # 1.: [1e-1, 5e-2, 1e-2],
#                     # 0.5: [1e-2, 5e-3, 1e-3],
#                     # 0.5: [1e-2, 5e-3, 1e-3, None],
#                     # 0.1: [5e-3, 1e-3, 5e-4, None],
#                     0.5: [None],
#                     0.1: [None],
#     # 0.01: [1e-3, 5e-4, 1e-4],
#                     }
# d_list = [0.5, 0.6, 0.7]
# dz_list = [0.4, 0.5, 0.8, 0.9]
# option_eta_v_list = [1, 0]
# action_repeats_list = [1]
# option_epsilon_list = [0.2, 0.5]
# reps_list = [0.5, 0.1]
# option_alpha_z_list = {0: [7e-1],
#                        1: [7e-1]}
#
option_alpha_v_list = {
                    # 1.: [1e-1, 5e-2, 1e-2],
                    # 0.5: [1e-2, 5e-3, 1e-3],
                    # 0.5: [1e-2, 5e-3, 1e-3, None],
                    # 0.1: [5e-3, 1e-3, 5e-4, None],
                    0.5: [None],
                    1.: [None],
                    0.1: [None],
                    0.01: [None],
                    0.001: [None],
                    0.0001: [None],
                    # 0.01: [1e-3, 5e-4, 1e-4],
                    }
d_list = [0.8, 0.4]
dz_list = [0.99]
option_eta_v_list = [1, 0]
action_repeats_list = [1]
option_epsilon_list = [0.5]
reps_list = [0.01]
option_alpha_z_list = {0: [7e-1, None],
                       1: [1e-1]}

hyperparams = {"discount": [0.99],
               "use_prior_corr": [False],
               "use_post_corr": [False],
               "option_use_post_corr": [True],
               "option_use_prior_corr": [False],
               "eta_f": [1],
               "eta_x_f": [1],
               "eta_q": [1],
               "eta_z": [1],
               "option_eta_v": option_eta_v_list,
               "option_eta_z": [1],
               "option_eta_f": [1],
               "option_eta_x_f": [1],
               "lambda": [0.98],
               "alpha_q": [1e-1],
               "option_alpha_v": option_alpha_v_list,
               "option_alpha_f": [1e-1],
               "alpha_f": [1e-1],
               "alpha_z": [1e-1],
               "option_alpha_z": option_alpha_z_list,
               "beta": [np.inf],
               "c": [1],
               "epsilon": [1.0],
               "option_epsilon": option_epsilon_list,
               "action_repeats": action_repeats_list,
               }

def one_or(z, seed):
    return z[seed] if len(z) > 1 else z[0]

for seed in range(seeds):
    for option_eta_v in option_eta_v_list:
        for action_repeats in action_repeats_list:
            for option_epsilon in option_epsilon_list:
                for reps in reps_list:
                    for option_alpha_v in option_alpha_v_list[reps]:
                        if option_alpha_v is not None:
                            d_list_temp = [None]
                        else:
                            d_list_temp = d_list
                        for d in d_list_temp:
                            for option_alpha_z in option_alpha_z_list[option_eta_v]:
                                if option_alpha_z is not None:
                                    dz_list_temp = [None]
                                else:
                                    dz_list_temp = dz_list
                                for dz in dz_list_temp:
                                    option_alpha_v_temp = option_alpha_v if option_alpha_v is not None else f"d_{d}"
                                    option_alpha_z_temp = option_alpha_z if option_alpha_z is not None else f"dz_{dz}"

                                    key = f"{seed}_{option_alpha_v_temp}_{option_alpha_z_temp}_{option_eta_v}_{action_repeats}_{option_epsilon}_{reps}"
                                    filename = f"./data/data_{key}.pkl"
                                    filename_no_pkl = f"./data/data_{key}"
                                    if os.path.exists(filename):
                                        config = load_obj(filename_no_pkl)
                                        for option in range(4):
                                            options_stats_all[option][key] = one_or(config["options_stats"][option]["zloss"][0], seed)
                                        options_pi_all[key] = one_or(config["options_pi"][0], seed)
                                        options_v_pi_all[key] = one_or(config["options_v_pi"][0], seed)
                                        options_true_v_pi_all[key] = one_or(config["options_true_v_pi"][0], seed)
                                        env_all[key] = config["env"]
                                    else:
                                        print(f"{filename} does not exists")
                                        continue
                                    # if key in options_pi_all:
                                    #     for option in range(4):
                                    #         options_stats_all[option][key] = one_or(config["options_stats"][option]["value_errors"][0], seed)
                                    #     options_pi_all[key] = one_or(config["options_pi"][0], seed)
                                    #     options_v_pi_all[key] = one_or(config["options_v_pi"][0], seed)
                                    #     options_true_v_pi_all[key] = one_or(config["options_true_v_pi"][0], seed)
                                    #     env_all[key] = config["env"]
                                    # else:

plot_final3(agents=agents,
            options_stats=options_stats_all,
            options_pi=options_pi_all,
            options_v_pi=options_v_pi_all,
            options_true_v_pi=options_true_v_pi_all,
            env=env_all,
            log_every=100,
            d_list=d_list,
            dz_list=dz_list,
            plot_zloss=True,
            option_alpha_v_list=option_alpha_v_list,
            option_eta_v_list=option_eta_v_list,
            action_repeats_list=action_repeats_list,
            option_epsilon_list=option_epsilon_list,
            option_alpha_z_list=option_alpha_z_list,
            reps_list=reps_list,
            hyper=hyperparams,
            smoothing=1, fontsize=25,
            legend_loc='upper left',
            legend_bbox_to_anchor=(0., 1.05),
            align_axis=True,
            legend_ncol=12, how_many=None,
            num_ticks=2, save=True,
            seeds=seeds, verbose=False)