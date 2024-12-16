from imports import *
from utils import *
from options import *
from statistics import *
from envs import *
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
#                     1.: [1e-1, 5e-2, 1e-2],
#                     0.5: [1e-2, 5e-3, 1e-3],
#                     0.1: [5e-3, 1e-3, 5e-4],
#                     0.01: [1e-3, 5e-4, 1e-4],
#                     }
# option_eta_v_list = [0]
# action_repeats_list = [1]
# option_epsilon_list = [0., 0.2, 0.5, 0.7]
# reps_list = [1., 0.5, 0.1, 0.01]
# option_alpha_z_list = {0: [1e-1, 5e-1, 7e-1],
#                        }
option_alpha_v_list = {
                    # 1.: [1e-1, 5e-2, 1e-2],
                    # 0.5: [1e-2, 5e-3, 1e-3],
                    0.1: [1e-3, 5e-3, 1e-3],
                    # 0.01: [1e-3, 5e-4, 1e-4],
                    }
option_eta_v_list = [1, 0]
action_repeats_list = [1]
option_epsilon_list = [0.2]
reps_list = [0.1]
option_alpha_z_list = {0: [5e-1],
                       1: [5e-1]}

hyperparams = {"discount": [0.98],
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
               "lambda": [0.9],
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


for seed in range(seeds):
    for option_eta_v in option_eta_v_list:
        for option_alpha_z in option_alpha_z_list[option_eta_v]:
            for action_repeats in action_repeats_list:
                for option_epsilon in option_epsilon_list:
                    for reps in reps_list:
                        for option_alpha_v in option_alpha_v_list[reps]:
                            key = f"{seed}_{option_alpha_v}_{option_alpha_z}_{option_eta_v}_{action_repeats}_{option_epsilon}_{reps}"
                            filename = f"./data/data_{key}.pkl"
                            filename_no_pkl = f"./data/data_{key}"
                            config = load_obj(filename_no_pkl)
                            if key in options_pi_all:
                                for option in range(4):
                                    options_stats_all[option][key] = config["options_stats"][option]["value_errors"][0][0]
                                options_pi_all[key] = config["options_pi"][0][0]
                                options_v_pi_all[key] = config["options_v_pi"][0][0]
                                options_true_v_pi_all[key] = config["options_true_v_pi"][0][0]
                                env_all[key] = config["env"]
                            else:
                                for option in range(4):
                                    options_stats_all[option][key] = config["options_stats"][option]["value_errors"][0][0]
                                options_pi_all[key] = config["options_pi"][0][0]
                                options_v_pi_all[key] = config["options_v_pi"][0][0]
                                options_true_v_pi_all[key] = config["options_true_v_pi"][0][0]
                                env_all[key] = config["env"]
plot_final3(agents=agents,
            options_stats=options_stats_all,
            options_pi=options_pi_all,
            options_v_pi=options_v_pi_all,
            options_true_v_pi=options_true_v_pi_all,
            env=env_all,
            log_every=200,
            option_alpha_v_list=option_alpha_v_list,
            option_eta_v_list=option_eta_v_list,
            action_repeats_list=action_repeats_list,
            option_epsilon_list=option_epsilon_list,
            option_alpha_z_list=option_alpha_z_list,
            reps_list=reps_list,
            hyper=hyperparams,
            smoothing=1, fontsize=25,
            legend_loc='upper left',
            legend_bbox_to_anchor=(0., 1.1),
            align_axis=True,
            legend_ncol=4,
            prefix="et_",
            num_ticks=2, save=True,
            seeds=seeds, verbose=False)