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
# options_stats_all = [{}, {}, {}, {}]
options_stats_all = {}
option_interests_all = {}
q_star_all = {}
options_pi_all = {}
options_q_pi_all = {}
options_true_q_pi_all = {}
env_all = {}
interest_all = {}
options_q_star_all = {}
seeds = 1
reps_list = [1.]
eta_q_list = [0]
alpha_q_list = {
    1: [None],
    0: [None]
}
alpha_z_list = {
    1: [5e-1],
    0: [None]
}
d_list = {1:
              {
                1: [0.2],
                0: [0.5, 0.7]
               },
          0:
              {
                1: [0.2],
                0: [0.2]
               }
          }
# dz_list = [0.8]
dz_list = {1: {1: [1.],
               0: [0.01, 0.1, 1.]
               },
           0: {1: [1.],
               0: [1.],
               }}
discounts = {0: [0.98],
             1: [0.9, 0.95, 0.98]
             }

use_only_primitive_actions_list = [1]

hyperparams = {"discount": [0.95],
               "intra_option_discount": [0.9],
               "use_prior_corr": [False],
               "use_post_corr": [False],
               "option_use_post_corr": [False],
               "option_use_prior_corr": [False],
               "eta_f": [1],
               "eta_x_f": [1],
               "use_q_for_reward": [True if use_only_primitive_actions == 0 else False for use_only_primitive_actions in
                                    use_only_primitive_actions_list],
               "use_magic_rule": [True if use_only_primitive_actions == 0 else False for use_only_primitive_actions in
                                  use_only_primitive_actions_list],
               "use_true_options": [True if use_only_primitive_actions == 0 else False for use_only_primitive_actions in
                                    use_only_primitive_actions_list],
               "learn_trace_by_magic": [True if use_only_primitive_actions == 0 else False for
                                        use_only_primitive_actions in use_only_primitive_actions_list],
               "use_only_primitive_actions": [False if use_only_primitive_actions == 0 else True for
                                              use_only_primitive_actions in use_only_primitive_actions_list],
               "alpha_q": alpha_q_list,
               "eta_q": eta_q_list,
               "eta_z": [1],
               "option_eta_q": [1],
               "option_eta_z": [1],
               "option_eta_f": [1],
               "option_eta_x_f": [1],
               "lambda": [0.9],
               "option_alpha_q": [0],
               "option_alpha_f": [1e-1],
               "alpha_f": [1e-1],
               "alpha_z": [5e-1],
               "option_alpha_z": [1e-1],
               "beta": [np.inf],
               "option_beta": [np.inf],
               "epsilon": [0.1],
               "option_epsilon": [0.],
               }

for seed in range(seeds):
    for reps in reps_list:
        for use_only_primitive_actions in use_only_primitive_actions_list:
            for discount in discounts[use_only_primitive_actions]:
                for discount in discounts[use_only_primitive_actions]:
                    hyperparams["discount"] = [discount]
                    for alpha_q in alpha_q_list[use_only_primitive_actions]:
                        for eta_q in eta_q_list:
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
                                        key = f"{seed}_{reps}_{alpha_q_temp}_{alpha_z_temp}_{eta_q}_{use_only_primitive_actions}_{discount}"
                                        filename = f"./data/data_{key}.pkl"
                                        filename_no_pkl = f"./data/data_{key}"
                                        if os.path.exists(filename):
                                            config = load_obj(filename_no_pkl)
                                            q_pi_all[key] = config["q_pi"][0][0]
                                            stats_all[key] = config["stats"]["eval_ep_timesteps"][0][0]
                                            option_interests_all[key] = config["option_interests"][0][0]
                                            # q_star_all[key] = config["q_star"][0][0]
                                            options_stats_all[key] = config["options_stats"][0][0]
                                            options_pi_all[key] = config["options_pi"][0][0]
                                            interest_all[key] = config["interest"][0][0]
                                            options_q_star_all[key] = config["options_q_star"][0][0]
                                            options_q_pi_all[key] = config["options_q_pi"][0][0]
                                            options_true_q_pi_all[key] = config["options_true_q_pi"][0][0]
                                            env_all[key] = config["env"]
                                        else:
                                            print(f"{filename} does not exists")
                                            continue

plot_final2(agents=agents,
            stats=stats_all,
            options_stats=options_stats_all,
            options_pi=options_pi_all,
            options_q_pi=options_q_pi_all,
            options_q_star=options_q_star_all,
            q_pi=q_pi_all,
            discounts=discounts,
            reps_list=reps_list,
            interest=interest_all,
            custom_hyperparams=custom_hyperparams,
            options_true_q_pi=options_true_q_pi_all,
            q_star=q_star_all,
            eta_q_list=eta_q_list,
            option_interests=option_interests_all,
            env=env_all,
            log_every=10,
            alpha_q_list=alpha_q_list,
            d_list=d_list,
            dz_list=dz_list,
            alpha_z_list=alpha_z_list,
            use_only_primitive_actions_list=use_only_primitive_actions_list,
            hyper=hyperparams, how_many=1000,
            smoothing=10, fontsize=20,
            legend_loc='lower left',
            legend_bbox_to_anchor=(.12, 1.0),
            legend_ncol=4,
            num_ticks=1, save=True,
            seeds=seeds, verbose=True)
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
