# @title Options + policy over options | onehot | linear | QET
from imports import *
from runner import *
import os
seeds = 20
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
# d_list = [1.]
d_list = [1., 0.9, 0.7, 0.5]
# dz_list = [0.8]
dz_list = [0.1, 0.01, 0.001]
option_eta_v_list = [0, 1]
action_repeats_list = [1]
option_epsilon_list = [0.2, 0.5, 0.7]
reps_list = [0.001]
option_alpha_z_list = {0: [None],
                       1: [1e-1]}

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
                                            continue
                                        config = {}
                                        agents = [""]
                                        config["seed"] = seed
                                        smoothing = 1
                                        custom_hyperparams = {}
                                        hyperparams = {"discount": [0.99],
                                                       "use_prior_corr": [False],
                                                       "use_post_corr": [False],
                                                       "option_use_post_corr": [True],
                                                       "option_use_prior_corr": [False],
                                                       "eta_f": [1],
                                                       "eta_x_f": [1],
                                                       "eta_q": [1],
                                                       "eta_z": [1],
                                                       "option_eta_v": [option_eta_v],
                                                       "option_eta_z": [0],
                                                       "option_eta_f": [1],
                                                       "option_eta_x_f": [1],
                                                       "lambda": [0.98],
                                                       "alpha_q": [1e-1],
                                                       "option_alpha_v": [option_alpha_v],
                                                       "option_alpha_f": [1e-1],
                                                       "alpha_f": [1e-1],
                                                       "alpha_z": [1e-1],
                                                       "option_alpha_z": [option_alpha_z],
                                                       "beta": [np.inf],
                                                       "c": [1],
                                                       "d": [d],
                                                       "dz": [dz],
                                                       "epsilon": [1.0],
                                                       "option_epsilon": [option_epsilon],
                                                       "action_repeats": [action_repeats],
                                                       }
                                        str_in = "13,13\n" + \
                                                 "XXXXXXXXXXXXX\n" + \
                                                 "XSSSSSSSSSSSX\n" + \
                                                 "XSSSSSSSSSSSX\n" + \
                                                 "XSSSSgSSSSSSX\n" + \
                                                 "XSSSSSSSSSSSX\n" + \
                                                 "XSSSSSSSSSSSX\n" + \
                                                 "XSSSSSSSSSSSX\n" + \
                                                 "XSSSSSSSSSSSX\n" + \
                                                 "XSSSSSSSSSSSX\n" + \
                                                 "XSSSSSSSGSSSX\n" + \
                                                 "XSSSSSSSSSSSX\n" + \
                                                 "XSSSSSSSSSSSX\n" + \
                                                 "XXXXXXXXXXXXX"

                                        stats, q_pi, options_stats, options_pi, options_v_pi, options_true_v_pi, env = run_w_options(
                                            env_str_in=str_in,
                                            number_of_steps=-1,
                                            number_of_episodes=int(500),
                                            teps=0.05, reps=reps, reward=10. / reps,
                                            option_reward=1.,
                                            time_limit=-1,
                                            log_every_steps=10000,
                                            log_every_episodes=5,
                                            network="linear",
                                            obs_type=OBS_ONEHOT,
                                            agents=agents,
                                            hyperparams=hyperparams,
                                            custom_hyperparams={},
                                            seed=seed,
                                            verbose=False,
                                            plot=False)

                                        config["hyperparams"] = hyperparams
                                        config["stats"] = stats
                                        config["q_pi"] = q_pi
                                        config["options_stats"] = options_stats
                                        config["options_pi"] = options_pi
                                        config["options_v_pi"] = options_v_pi
                                        config["options_true_v_pi"] = options_true_v_pi
                                        config["env"] = env

                                        save_obj(config, filename_no_pkl)
# to_plot = [("value_errors", "VE"), ("zloss", "\delta_z"), ("td_error", "\delta")]
# plot_final(agents=agents,
#            to_plot=to_plot,
#             option_interests=option_interests,
#             option_statistics=stats,
#             options_stats=options_stats,
#             options_pi=options_pi,
#             options_v_pi=options_v_pi,
#             options_true_v_pi=options_true_v_pi,
#             env=env,
#             custom_hyperparams=custom_hyperparams,
#             hyperparams=hyperparams,
#             smoothing=smoothing, fontsize=25,
#             legend_loc='upper left',
#             legend_bbox_to_anchor=(0., 0.7),
#             align_axis=False,
#             legend_ncol=5,
#             num_ticks=2,
#             seeds=seeds, verbose=True)
# TODO: alight plots, two step sizes,