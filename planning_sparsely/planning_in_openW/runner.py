# @title runner
from imports import *
from envs import *
from agents import *
def run_w_options(
        env_str_in="",
        network="linear",
        agents=None,
        number_of_episodes=-1,
        number_of_steps=1e3,
        log_every_steps=10,
        log_every_episodes=10,
        seed=1,
        dim_obs=-1,
        action_repeats=4,
        reps=1,
        teps=0.,
        reward=1,
        option_reward=1,
        eval=False,
        obs_type="one_hot",
        hyperparams=None,
        custom_hyperparams=None,
        time_limit=-1,
        verbose=False,
        plot=False,
):
    stats_keys = ["td_error", "rewards", "value_errors",
                  "values", "f_td_or_mc_errors",
                  "decision_errors", "ep_timesteps",
                  "f_exp_errors", "zloss", "rhos_post",
                  "f_var_errors", "value_weights", "options_value_errors"]
    agents_stats = defaultdict(list)
    agents_options_stats = []
    agents_final_option_pi = []
    agents_final_option_v_pi = []
    agents_final_true_option_v_pi = []
    agents_final_q_pi = []
    agents_final_true_q_pi = []
    for option in range(2):
        agents_options_stats.append(defaultdict(list))
    hyper = {}
    for agent_name in agents:
        agents_params = {}
        for k, v in hyperparams.items():
            agents_params[k] = v
            if agent_name in custom_hyperparams.keys() and \
                    k in custom_hyperparams[agent_name].keys():
                agents_params[k] = custom_hyperparams[agent_name][k]

        for hyper_values in itertools.product(*agents_params.values()):
            seeds_stats = defaultdict(list)
            seeds_options_stats = []
            seeds_final_option_pi = []
            seeds_final_option_v_pi = []
            seeds_final_true_option_v_pi = []
            seeds_q_pi = []
            seeds_true_q_pi = []
            for option in range(2):
                seeds_options_stats.append(defaultdict(list))
            for v, k in zip(hyper_values, agents_params.keys()):
                hyper[k] = v
            for seed in [seed]:
                stats = defaultdict(list)
                options_stats = []

                env = GridWorld(seed=4321 + seed, option_reward=option_reward,
                                str_in=env_str_in, time_limit=time_limit,
                                teps=teps, reps=reps, dim_obs=dim_obs,
                                reward=reward, eval=False,
                                observation_type=obs_type,
                                )

                option_interests = []
                option_pis = []
                # option_true_q_pis = []
                option_true_v_pis = []
                for option in [option_0, option_1,
                               # option_2, option_3
                               ]:
                    option_interest, option_pi = load_option(env, fstr=option)
                    option_interests.append(option_interest)
                    option_pis.append(option_pi)
                    options_stats.append(defaultdict(list))
                    # option_true_q_pis.append(env.get_q_pi(pi=option_pi, discount=hyper["discount"], option_epsilon=hyper["option_epsilon"]))
                    option_true_v_pis.append(env.get_v_pi(pi=option_pi, discount=hyper["discount"], option_epsilon=hyper["option_epsilon"]))
                interest = np.ones_like(option_interest)
                num_options = len(option_interests)

                agent_config = get_config(agent_name,
                                          interest=interest,
                                          option_interests=option_interests,
                                          option_pis=option_pis,
                                          # option_true_q_pis=option_true_q_pis,
                                          hyper=hyper)

                agent, agent_state = agent_config["init"](
                    rng_key=jax.random.PRNGKey(4321 + seed),
                    network_spec=(network or env_id),
                    action_spec=env.action_spec(),
                    option_spec=env.action_spec(num_options),
                    observation_spec=env.observation_spec(),
                    **agent_config["cfg"])

                timestep = None
                episode_timesteps = 0
                t = 0
                ep = 0
                ep_return = 0

                while (True):
                    # Reset environment and pick first action
                    if not timestep or timestep.last():
                        timestep = env.reset()
                        s = env._curr_idx
                        action, agent_state = agent.first_step(timestep, s, agent_state)

                        all_states = env.get_all_states()
                        d_mu = env.get_d_pi(agent.get_mu(all_states, agent_state), discount=hyper["discount"])

                    timestep = env.step(action)
                    next_s = env._curr_idx
                    prev_s = s
                    prev_obs = agent_state.obs
                    action, agent_state, log_dict, options_log_dict = agent.step(
                        timestep=timestep, s=s,
                        next_s=next_s, agent_state=agent_state)

                    episode_timesteps += 1
                    ep_return += timestep.reward
                    t += 1
                    s = next_s

                    if timestep.last():
                        stats["ep_timesteps"].append(episode_timesteps)
                        episode_timesteps = 0
                        ep += 1
                        stats["returns"].append(ep_return)
                        ep_return = 0

                    if t % log_every_steps == 0 and verbose:
                        print(f'ep {ep}',
                              f'steps: {t:10}',
                              f', episode_timesteps {episode_timesteps} reward: {timestep.reward}'
                              f', episode_end: {timestep.last()}',
                              # f', option: {agent_state.option}',
                              f', action: {action}',
                              f', discount: {timestep.discount}',
                              f', state {prev_s}',
                              f', next_state {next_s}')

                        for option in range(len(option_interests)):
                            print(f'option {option} action {action} td error {options_log_dict[option]["td_error"]}'
                                  f', reward {options_log_dict[option]["reward"]}'
                                  f' discount {options_log_dict[option]["discount"]}'
                                  f' trace_decay {options_log_dict[option]["trace_decay"]}'
                                  f' rho_post {options_log_dict[option]["rho_post"]}'
                                  f' rho_prior {options_log_dict[option]["rho_prior"]}'
                                  )

                    if timestep.last() and (ep == 1 or ep % log_every_episodes == 0) and verbose:
                        print(f'ep {ep} of {number_of_episodes}')

                    if (timestep.last() and (ep == 1 or ep % log_every_episodes == 0)) or \
                            (number_of_steps > -1 and t % log_every_steps == 0):
                        # Log instantaneous reward
                        stats["rewards"].append(timestep.reward)
                        # Log td errors
                        stats["td_error"].append(log_dict["td_error"])

                        # pio = agent.get_pi(all_states, agent_state)
                        # q_pio = agent.get_q(all_states, agent_state)
                        # v_pio = np.max(q_pio, axis=-1)

                        option_v_pis = agent.get_options_v(all_states, agent_state,
                                                           option_interests=option_interests)

                        for option, option_interest in enumerate(option_interests):
                            ve = np.average((option_true_v_pis[option] - option_v_pis[option]) ** 2,
                                            weights=d_mu)
                            options_stats[option]["value_errors"].append(ve)
                            options_stats[option]["td_error"].append(options_log_dict[option]["td_error"])
                            options_stats[option]["zloss"].append(options_log_dict[option]["zloss"])
                            options_stats[option]["f_td_or_mc_errors"].append(
                                options_log_dict[option]["f_td_or_mc_error"])
                            options_stats[option]["rho_post"].append(options_log_dict[option]["f_td_or_mc_error"])

                            # options_stats[option]["f_var_errors"].append(option_f_var_e)

                            if ep % log_every_episodes == 0 and plot:
                                # weights = np.tile(option_interests[option][..., None], (1, env._num_actions))
                                plot_option(env, option_v_pis[option], option_true_v_pis[option],
                                            option_pis[option],
                                            title=r'$o_{}$'.format(option))

                                # plot_s_values(env, option_v_pis[option], text_values=False, title=r'$o_{}$'.format(option))
                                # plot_sa_values_4(env,
                                #                 options_pi[option],
                                #                 option_interests[option],
                                #                 option_q_pis[option] * weights,
                                #                 true_options_q_pis[option] * weights,
                                #                 true_options_q_pis[option] * weights,
                                #                 text_values=False, title=r'$o_{}$'.format(option),
                                #                     subtitles=[r"$q^*$", r"$q$", r"$q_\pi$", r"$v,\pi$"])

                        stats["values"].append(log_dict["q"])
                        # Log the td or the mc error for the follow-on trace
                        stats["f_td_or_mc_errors"].append(log_dict["f_td_or_mc_error"])
                        stats["zloss"].append(log_dict["zloss"])
                        # Log value error

                        # weights = np.tile(interest[..., None], (1, env._num_actions))

                        # stats["rhos_prior"].append(log_dict["rho_prior"])
                        # stats["rhos_post"].append(log_dict["rho_post"])

                    if (number_of_steps > -1 and t >= number_of_steps) or \
                            (number_of_episodes > -1 and ep >= number_of_episodes):
                        break

                # seeds_q_pi.append(q_pio)
                seeds_final_option_pi.append(option_pis)
                seeds_final_option_v_pi.append(option_v_pis)
                seeds_final_true_option_v_pi.append(option_true_v_pis)
                for k in stats_keys:
                    seeds_stats[k].append(stats[k])
                    for option in range(len(option_interests)):
                        seeds_options_stats[option][k].append(options_stats[option][k])
            # agents_final_q_pi.append(seeds_q_pi)
            agents_final_option_pi.append(seeds_final_option_pi)
            agents_final_option_v_pi.append(seeds_final_option_v_pi)
            agents_final_true_option_v_pi.append(seeds_final_true_option_v_pi)
            for k in stats_keys:
                agents_stats[k].append(seeds_stats[k])
                for option in range(len(option_interests)):
                    agents_options_stats[option][k].append(seeds_options_stats[option][k])

    return agents_stats, agents_final_q_pi, agents_options_stats, agents_final_option_pi, agents_final_option_v_pi, agents_final_true_option_v_pi, env