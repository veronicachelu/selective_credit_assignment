# @title Options + policy over options | onehot | linear | QET
from imports import *
from utils import *
from statistics import *
import os
df_iq = load_data("./flatboard/df_iqet.csv")
df_iqet = load_data("./flatboard/df_iq.csv")

tmp = [df_iqet.copy(), df_iq.copy()]
x = 'agent_steps'
y = 'mean_return'
 # Anything not specified explicitly will be averaged over (e.g., 'replica')
group = [['use_i', 'i_dep_trace_param', 'i_dep_eta', 'i_dep_et_loss'],
         ['use_i', 'i_dep_trace_param', 'i_dep_eta', 'i_dep_et_loss']]
facet = ['noisy_prob', "eta_q"]
max_over = [[], []]
pick = [[
    lambda _df: (_df['noisy_prob'] != 0.3),
    lambda _df: (_df['eta_q'] == 1),
],[
    lambda _df: (_df['noisy_prob'] != 0.3),
    lambda _df: (_df['eta_q'] == 1),
]]

graph_list(tmp, [x,x], [y,y], group,
           facet, max_over, pick,
           log=False, fontsize=21,
           window=20, label_fn=label_fn,
           facet_fn=facet_fn, axis_fn=axis_fn,
           ls_fn_list=[lambda _: "-", lambda _: "--"],
           legend_ncols=3, legend_loc='upper left',
           legend_bbox=(0.2, 1.1))