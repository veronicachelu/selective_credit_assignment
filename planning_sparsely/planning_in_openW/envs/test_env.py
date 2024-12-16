import pprint
from envs import *
from options import *
pp = pprint.PrettyPrinter(width=41, compact=True)
# @title env test
str_in = "13,13\n" + \
         "XXXXXXXXXXXXX\n" + \
         "XSSSSSSSSSSSX\n" + \
         "XSSSgSSSSSSSX\n" + \
         "XSSSSSSSSSSSX\n" + \
         "XSSSSSSSSSSSX\n" + \
         "XSSSSSSSSSSSX\n" + \
         "XSSSSSSSSSSSX\n" + \
         "XSSSSSSSSSSSX\n" + \
         "XSSSSSSSSSSSX\n" + \
         "XSSSSSSSGSSSX\n" + \
         "XSSSSSSSSSSSX\n" + \
         "XSSSSSSSSSSSX\n" + \
         "XXXXXXXXXXXXX"

env = GridWorld(seed=4321,
                str_in=str_in,
                teps=0.,
                reward=1., reps=0.1, eval=False,
                observation_type=OBS_XY, use_q_for_reward=False)
# # q_star = env.q_iteration(num_itrs=20, discount=0.8)

option_interests = []
option_pis = []
option_true_q_pis = []
option_true_v_pis = []
for option in [option_0,
               option_1,
               # option_2,
               # option_3
               ]:
    interest, policy = load_option(env, fstr=option)
    option_interests.append(interest)
    option_pis.append(policy)
    # option_q_pi = env.get_q_pi(pi=policy, discount=0.99)
    # option_v_pi = env.get_v_pi(pi=policy, discount=0.99)
    # option_true_q_pis.append(option_q_pi)
    # option_true_v_pis.append(option_v_pi)
# # interest = load_option(env, fstr=nonhallways)
# # q_star = env.q_iteration(num_itrs=20, discount=0.9)
# #                         # plot_sa_values(env, q_star,
# #                         #                    title=f'High-level q *')
# # v_star = np.max(q_star, axis=1)
# # all_states = env.get_all_states()
# # options_q_star = env.options_q_iteration(option_interests,
# #                                          num_itrs=20, v_star=v_star,
# #                                          discount=0.9)
# # options_q_pi = agent.get_options_q(all_states, agent_state,
# #                         option_interests=option_interests)
# # q_pio = agent.get_q(all_states, agent_state)
# # v_pio = np.max(q_pio, axis=-1)
# # options_true_q_pi = env.get_options_q_pi(options_pi,
# #                                         option_interests,
# #                                         v_pio,
# #                                         option_discount=agent_config["cfg"]["intra_option_discount"])
# for option, option_interest in enumerate(option_interests):
#     # option_reward_matrix = env.get_option_reward_matrix(option_interest, v_star)
#     # option_reward_matrix = np.max(option_reward_matrix, axis=(0, 1))
#     # option_v_pi = np.max(option_v_star, axis=1)
#     plot_option(env, option_true_v_pis[option],
#                 option_true_v_pis[option], option_pis[option],
#                 title=f'option {option} v_\pi, q_\pi, \pi')

#                title=f'option {option} v_\pi')
# plot_s_values(env, option_true_v_pis[option],
#                title=f'option {option} v_\pi')
# plot_sa_values(env, option_true_q_pis[option],
#                title=f'option {option} q_\pi')
# for option, option_q_pi in enumerate(options_q_pi):
#     # print(option_q_star.shape)
#     # plot_sa_values(env, option_q_star, text_values=False,
#     #             title=f'option {option} q*')
#     weights = np.tile(option_interests[option][..., None], (1, env._num_actions))
#     plot_sa_values_4(env,
#                     options_pi[option],
#                     option_interests[option],
#                     option_q_pi * weights,
#                     option_q_pi * weights,
#                     text_values=False, title=r'$o_{}$'.format(option),
#                         subtitles=[r"$q^*$", r"$q$", r"$q_\pi$", r"$v,\pi$"])
#     # option_pi_q_star = env.compute_policy_deterministic(option_q_star, eps_greedy=0.)
# # pi_q_star = env.compute_policy_deterministic(q_star, eps_greedy=0.)
# # v_star = np.max(q_star, axis=1)f
# # plot_sa_values(env, q_star, invert_y=True, invert_x=False, title='q*')
# # plot_s_values(env, v_star, title='v*')
# # d_q_star = env.get_s_stationary(pi_q_star, discount=0.9)
# # d_sa_q_star = env.get_sa_stationary(pi_q_star, discount=0.9)
# # plot_sa_values(env, d_sa_q_star, title='d_sa(pi_q*)')
# # plot_s_values(env, d_q_star, title='d_s(pi_q*)')
