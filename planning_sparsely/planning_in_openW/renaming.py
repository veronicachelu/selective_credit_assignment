# import os
# from imports import *
# import pickle
# from utils import *
# from options import *
# from statistics import *
# from envs import *
# seeds = [1e-2, 5e-3, 1e-3]
# option_eta_v_list = [1, 0]
# action_repeats_list = [1, 4]
# option_epsilon_list = [0., 0.1]
# reps_list = [1., 0.5]
# option_alpha_z_list = [1e-1, 5e-1]
# option_alpha_z = 5e-1
# for seed in range(seeds):
#     for option_alpha_v in option_alpha_v_list:
#         for option_eta_v in option_eta_v_list:
#             for action_repeats in action_repeats_list:
#                 for option_epsilon in option_epsilon_list:
#                     for reps in option_epsilon_list:
#                         old_key = f"{seed}_{option_alpha_v}_{option_alpha_z}_{option_eta_v}_{action_repeats}_{option_epsilon}_{reps}"
#                         old_filename = f"./data/data_{old_key}.pkl"
#                         if os.path.exists(old_filename):
#                             new_key = f"{seed}_{option_alpha_v}_{option_alpha_z}_{option_eta_v}_{action_repeats}_{option_epsilon}_{reps}"
#                             new_filename = f"./data/data_{new_key}.pkl"
#                             os.rename(old_filename, new_filename)