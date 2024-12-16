# @title Options + policy over options | onehot | linear | QET
from imports import *
from runner import *
import os
seeds = 1
reps_list = [1.]
smoothing = 1
eta_q_list = [0]
alpha_q_list = {
    1: [None],
    0: [None]
}
alpha_z_list = {
    1: [5e-1],
    0: [None]
}

d_list = [0.2]
dz_list = [0.01]
use_only_primitive_actions_list = [1]
discounts = {0: [0.98],
             1: [0.98]
             }
for seed in range(seeds):
    for reps in reps_list:
        for use_only_primitive_actions in use_only_primitive_actions_list:
            for discount in discounts[use_only_primitive_actions]:
                for alpha_q in alpha_q_list[use_only_primitive_actions]:
                    for eta_q in eta_q_list:
                        for alpha_z in alpha_z_list[eta_q]:
                            if alpha_q is None:
                                d_temp_list = d_list
                            else:
                                d_temp_list = [None]
                            if alpha_z is None:
                                dz_temp_list = dz_list
                            else:
                                dz_temp_list = [None]
                            for d in d_temp_list:
                                for dz in dz_temp_list:
                                    alpha_q_temp = alpha_q if alpha_q is not None else f"d_{d}"
                                    alpha_z_temp = alpha_z if alpha_z is not None else f"dz_{dz}"
                                    key = f"{seed}_{reps}_{alpha_q_temp}_{alpha_z_temp}_{eta_q}_{use_only_primitive_actions}_{discount}"
                                    filename = f"data/data_{key}.pkl"
                                    filename_no_pkl = f"data/data_{key}"
                                    if os.path.exists(filename):
                                        continue
                                    config = {}
                                    agents = [""]
                                    config["seed"] = seed
                                    custom_hyperparams = {}
                                    hyperparams = {"discount": [discount],
                                                   "intra_option_discount": [0.9],
                                                   "use_prior_corr": [False],
                                                   "use_post_corr": [False],
                                                   "option_use_post_corr": [False],
                                                   "option_use_prior_corr": [False],
                                                   "eta_f": [1],
                                                   "eta_x_f": [1],
                                                   "use_discount_magic_rule": [False],
                                                       # True if use_only_primitive_actions == 0 and eta_q == 0 else False],
                                                   "use_q_for_reward": [True if use_only_primitive_actions == 0 else False],
                                                   "use_magic_rule": [True if use_only_primitive_actions == 0 else False],
                                                   "use_true_options": [True if use_only_primitive_actions == 0 else False],
                                                   "learn_trace_by_magic": [True if use_only_primitive_actions == 0 else False],
                                                   "use_only_primitive_actions": [False if use_only_primitive_actions == 0 else True],
                                                   "alpha_q": [alpha_q],
                                                   "eta_q": [eta_q],
                                                   "eta_z": [1],
                                                   "d": [d],
                                                   "dz": [dz],
                                                   "option_eta_q": [1],
                                                   "option_eta_z": [1],
                                                   "option_eta_f": [1],
                                                   "option_eta_x_f": [1],
                                                   "lambda": [0.9],
                                                   "option_alpha_q": [1e-1],
                                                   "option_alpha_f": [1e-1],
                                                   "alpha_f": [1e-1],
                                                   "alpha_z": [alpha_z],
                                                   "option_alpha_z": [1e-1],
                                                   "beta": [np.inf],
                                                   "option_beta": [np.inf],
                                                   "epsilon": [0.5],
                                                   "option_epsilon": [0.],
                                                   }
                                    config["hyperparams"] = hyperparams
                                    str_in = "13,13\n" + \
                                             "XXXXXXXXXXXXX\n" + \
                                             "X.....X.....X\n" + \
                                             "X.....X.....X\n" + \
                                             "X.....S.....X\n" + \
                                             "X.....X.....X\n" + \
                                             "X.....X.....X\n" + \
                                             "XXSXXXX.....X\n" + \
                                             "X.....XXXGXXX\n" + \
                                             "X.....X.....X\n" + \
                                             "X.....X.....X\n" + \
                                             "X.....S.....X\n" + \
                                             "X.....X.....X\n" + \
                                             "XXXXXXXXXXXXX"

                                    (stats, interest, option_interests, q_star, q_pi, options_stats,
                                     options_q_star, options_pi, options_q_pi, options_true_q_pi,
                                     env) = run_w_options(
                                        env_str_in=str_in,
                                        number_of_steps=-1,
                                        number_of_episodes=int(100),
                                        teps=0., reps=reps, option_reps=1., reward=10. / reps,
                                        option_reward=1.,
                                        time_limit=-1,
                                        log_every_steps=10000,
                                        log_every_episodes=10,
                                        network="linear",
                                        obs_type=OBS_ONEHOT,
                                        agents=agents,
                                        eval_every=1,
                                        hyperparams=hyperparams,
                                        custom_hyperparams=custom_hyperparams,
                                        seed=seed,
                                        verbose=False,
                                        plot=False)

                                    config["stats"] = stats
                                    config["interest"] = interest
                                    config["option_interests"] = option_interests
                                    config["q_star"] = q_star
                                    config["options_q_star"] = options_q_star
                                    config["q_pi"] = q_pi
                                    config["options_stats"] = options_stats
                                    config["options_pi"] = options_pi
                                    config["options_q_pi"] = options_q_pi
                                    config["options_true_q_pi"] = options_true_q_pi
                                    config["env"] = env

                                    save_obj(config, filename_no_pkl)
                # j += 1

# statistics = {
#             "ep_timesteps": {"data": stats["ep_timesteps"],
#                                 "description": "Episode timesteps"},
#             }

# plot_statistics(env_id=r"  ", agents=agents,
#                     statistics=statistics,
#                     custom_hyperparams=custom_hyperparams,
#                     hyperparams=hyperparams,
#                     smoothing=smoothing, fontsize=20,
#                     legend_loc='upper left',
#                     legend_bbox_to_anchor=(0.1, 1.05),
#                     legend_ncol=2,
#                     num_ticks=1,
#                     cols=1, rows=1,
#                     seeds=seeds)
# plot_final(agents=agents,
#            q_pi=q_pi,
#            interest=interest,
#            q_star=q_star,
#            options_q_star=options_q_star,
#            option_interests=option_interests,
#             statistics=statistics,
#             custom_hyperparams=custom_hyperparams,
#             hyperparams=hyperparams,
#             smoothing=smoothing, fontsize=20,
#             legend_loc='upper left',
#             legend_bbox_to_anchor=(0., 1.2),
#             legend_ncol=2,
#             num_ticks=1,
#             seeds=seeds)