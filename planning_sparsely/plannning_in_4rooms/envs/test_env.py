import pprint
from envs import *
from options import *
pp = pprint.PrettyPrinter(width=41, compact=True)
# @title env test
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

env = GridWorld(seed=4321,
                str_in=str_in,
                teps=0.1,
                reward=1., reps=1., option_reps=1., eval=False,
                observation_type=OBS_XY, use_q_for_reward=True)
# q_star = env.q_iteration(num_itrs=20, discount=0.8)

option_interests = []
for option in [option_0,
               option_1,
               option_2,
               option_3]:
    option_interests.append(load_option(env, fstr=option))
interest = load_option(env, fstr=nonhallways)
# q_star = env.q_iteration(num_itrs=20, discount=0.9)
# plot_sa_values(env, q_star,
#                    title=f'High-level q *')
# v_star = np.max(q_star, axis=1)
# all_states = env.get_all_states()
# options_q_star = env.options_q_iteration(option_interests,
#                                          num_itrs=20, v_star=v_star,
#                                          discount=0.9)
# options_q_pi = agent.get_options_q(all_states, agent_state,
#                         option_interests=option_interests)
# q_pio = agent.get_q(all_states, agent_state)
# v_pio = np.max(q_pio, axis=-1)
# options_true_q_pi = env.get_options_q_pi(options_pi,
#                                         option_interests,
#                                         v_pio,
#                                         option_discount=agent_config["cfg"]["intra_option_discount"])
# for option, option_interest in enumerate(option_interests):
# option_reward_matrix = env.get_option_reward_matrix(option_interest, v_star)
# option_reward_matrix = np.max(option_reward_matrix, axis=(0, 1))
# option_v_star = np.max(option_q_star, axis=1)
# plot_s_values(env, option_reward_matrix,
#                title=f'option {option} r*')
# for option, option_q_star in enumerate(options_q_star):
#     option_q_star = options_q_star[option]
#     # print(option_q_star.shape)
#     # plot_sa_values(env, option_q_star, text_values=False,
#     #             title=f'option {option} q*')
#     weights = np.tile(option_interests[option][..., None], (1, env._num_actions))
#     pi = (option_q_star == option_q_star.max(axis=-1, keepdims=True)).astype(jnp.float32)
#     pi /= jnp.sum(pi, axis=-1, keepdims=True)
#     plot_sa_values_4(env,
#                      pi,
#                      option_interests[option],
#                      option_q_star,  # * weights,
#                      option_q_star,  # * weights,
#                      option_q_star,  # * weights,
#                      text_values=False, title=r'$o_{}$'.format(option),
#                      subtitles=[r"$q^*$", r"$q$", r"$q_\pi$", r"$v,\pi$"])
    # option_pi_q_star = env.compute_policy_deterministic(option_q_star, eps_greedy=0.)
# pi_q_star = env.compute_policy_deterministic(q_star, eps_greedy=0.)
# v_star = np.max(q_star, axis=1)f
# plot_sa_values(env, q_star, invert_y=True, invert_x=False, title='q*')
# plot_s_values(env, v_star, title='v*')
# d_q_star = env.get_s_stationary(pi_q_star, discount=0.9)
# d_sa_q_star = env.get_sa_stationary(pi_q_star, discount=0.9)
# plot_sa_values(env, d_sa_q_star, title='d_sa(pi_q*)')
# plot_s_values(env, d_q_star, title='d_s(pi_q*)')
