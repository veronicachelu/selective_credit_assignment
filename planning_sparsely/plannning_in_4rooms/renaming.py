import os
from imports import *
import pickle
from utils import *
from options import *
from statistics import *
from envs import *
seeds = 20
reps_list = [1.0, 0.5]
smoothing = 1
eta_q_list = [1, 0]
alpha_q_list = {
    1: [1e-1, 5e-2, 1e-2, None],
    0: [1e-1, 5e-2, 1e-2, None]
}
alpha_z_list = {
    1: [5e-1],
    0: [5e-1, None]
}
d_list = [0.4, 0.5, 0.8, 0.9]
dz_list = [0.4, 0.5, 0.8, 0.9]
use_only_primitive_actions_list = [1, 0]

for seed in range(seeds):
    for reps in reps_list:
        for use_only_primitive_actions in use_only_primitive_actions_list:
            for alpha_q in alpha_q_list[use_only_primitive_actions]:
                for eta_q in eta_q_list:
                    for alpha_z in alpha_z_list[eta_q]:
                        if alpha_q is None:
                            d_temp_list = [None]
                        else:
                            d_temp_list = d_list
                        if alpha_z is None:
                            dz_temp_list = [None]
                        else:
                            dz_temp_list = dz_list
                        for d in d_temp_list:
                            for dz in dz_temp_list:
                                alpha_q_temp = alpha_q if alpha_q is not None else f"d_{d}"
                                alpha_z_temp = alpha_z if alpha_z is not None else f"dz_{dz}"
                                old_key = f"{seed}_{reps}_{alpha_q_temp}_{eta_q}_{use_only_primitive_actions}"
                                old_filename = f"./data/data_{old_key}.pkl"
                                if os.path.exists(old_filename):
                                    new_key = f"{seed}_{reps}_{alpha_q_temp}_{alpha_z_temp}_{eta_q}_{use_only_primitive_actions}"
                                    new_filename = f"./data/data_{new_key}.pkl"
                                    os.rename(old_filename, new_filename)