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
        eval_every=1,
        dim_obs=-1,
        reps=1,
        option_reps=1.,
        teps=0.,
        reward=1,
        option_reward=1,
        obs_type="one_hot",
        hyperparams=None,
        custom_hyperparams=None,
        time_limit=-1,
        verbose=False,
        plot=False,
):
    stats_keys = ["td_error", "rewards", "value_errors",
                  "values", "f_td_or_mc_errors",
                  "decision_errors", "ep_timesteps", "eval_ep_timesteps",
                  "f_exp_errors", "zloss", "rhos_post",
                  "f_var_errors", "value_weights", "options_value_errors"]
    agents_stats = defaultdict(list)
    agents_options_stats = []
    agents_final_option_pi = []
    agents_final_q_star = []
    agents_final_option_interests = []
    agents_final_interest = []
    agents_final_option_q_pi = []
    agents_final_true_option_q_pi = []
    agents_final_q_pi = []
    agents_final_true_q_pi = []
    agents_final_option_qstar = []
    for option in range(4):
        agents_options_stats.append(defaultdict(list))
    hyper = {}
    for agent_name in agents:
        # print(agent_name)
        agents_params = {}
        for k, v in hyperparams.items():
            agents_params[k] = v
            if agent_name in custom_hyperparams.keys() and \
                    k in custom_hyperparams[agent_name].keys():
                agents_params[k] = custom_hyperparams[agent_name][k]

        # print(len(list(itertools.product(*agents_params.values()))))
        for hyper_values in itertools.product(*agents_params.values()):
            seeds_stats = defaultdict(list)
            seeds_options_stats = []
            seeds_final_option_qstar = []
            seeds_final_q_star = []
            seeds_final_interest = []
            seeds_final_option_interests = []
            seeds_final_option_pi = []
            seeds_final_option_q_pi = []
            seeds_final_true_option_q_pi = []
            seeds_q_pi = []
            seeds_true_q_pi = []
            for option in range(4):
                seeds_options_stats.append(defaultdict(list))
            for v, k in zip(hyper_values, agents_params.keys()):
                hyper[k] = v
            for seed in [seed]:
                stats = defaultdict(list)
                options_stats = []

                env = GridWorld(seed=4321 + seed, option_reward=option_reward,
                                str_in=env_str_in, time_limit=time_limit,
                                teps=teps, reps=reps, dim_obs=dim_obs,
                                reward=reward, eval=False, option_reps=option_reps,
                                observation_type=obs_type,
                                use_q_for_reward=hyper["use_q_for_reward"])
                env_eval = GridWorld(seed=4321 + seed, option_reward=option_reward,
                                str_in=env_str_in, time_limit=time_limit,
                                teps=teps, reps=1., dim_obs=dim_obs,
                                reward=10., eval=False, option_reps=option_reps,
                                observation_type=obs_type,
                                use_q_for_reward=hyper["use_q_for_reward"])


                option_interests = []
                option_goals = []
                for option in [option_0, option_1, option_2, option_3]:
                    option_interest, option_goal = load_option(env, fstr=option)
                    option_interests.append(option_interest)
                    option_goals.append(option_goal)
                    options_stats.append(defaultdict(list))
                interest, _ = load_option(env, fstr=nonhallways)
                if hyper["use_only_primitive_actions"]:
                    interest = np.ones_like(interest)
                num_options = len(option_interests)

                # q_star = env.q_iteration(num_itrs=20, discount=0.8)
                # v_star = np.max(q_star, axis=1)
                if hyper["use_only_primitive_actions"]:
                    options_q_star = [None for _ in range(4)]

                else:
                    options_q_star = env.options_q_iteration(option_interests=option_interests,
                                                             num_itrs=20,
                                                             option_goals=option_goals,
                                                             discount=hyper["intra_option_discount"])

                agent_config = get_config(agent_name,
                                          interest=interest,
                                          option_interests=option_interests,
                                          option_reward=option_reward,
                                          options_q_star=options_q_star,
                                          hyper=hyper)

                agent, agent_state = agent_config["init"](
                    rng_key=jax.random.PRNGKey(4321 + seed),
                    network_spec=(network or env_id),
                    action_spec=env.action_spec(),
                    option_spec=env.action_spec(num_options),
                    # action_and_option_spec=env.option_spec(num_options=num_options),
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

                    timestep = env.step(action)
                    next_s = env._curr_idx
                    prev_s = s
                    prev_obs = agent_state.obs
                    action, agent_state, log_dict, options_log_dict = agent.step(
                        timestep=timestep, s=s,
                        next_s=next_s, agent_state=agent_state, eval=False)

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

                        if ep % eval_every == 0:
                            stats = run_eval(ep, env_eval, agent, agent_state, stats)

                    if t % log_every_steps == 0 and verbose:
                        print(f'ep {ep}',
                              f'steps: {t:10}',
                              f', episode_timesteps {episode_timesteps} reward: {timestep.reward}'
                              f', eval episode_timesteps {stats["eval_ep_timesteps"]} eval returns: {stats["eval_returns"]}'
                              f', episode_end: {timestep.last()}',
                              f', option: {agent_state.option}',
                              f', option_reward {agent_state.option_reward}',
                              f', action: {action}',
                              f', discount: {timestep.discount}',
                              f', state {prev_s}',
                              f', obs {prev_obs}',
                              f', next_state {next_s}',
                              f', next_obs {timestep.observation}')

                        # for option in range(len(option_interests)):
                        #     print(f'option {option} td error {options_log_dict[option]["td_error"]}'
                        #         f', reward {options_log_dict[option]["reward"]}'
                        #         f' discount {options_log_dict[option]["discount"]}'
                        #         f' trace_decay {options_log_dict[option]["trace_decay"]}'
                        #         f' rho_post {options_log_dict[option]["rho_post"]}'
                        #         f' rho_prior {options_log_dict[option]["rho_prior"]}'
                        #         f' q {options_log_dict[option]["q"]}')

                    if (timestep.last() and (ep == 1 or ep % log_every_episodes == 0)) or \
                            (number_of_steps > -1 and t % log_every_steps == 0):
                        # Log instantaneous reward

                        stats["rewards"].append(timestep.reward)
                        # Log td errors
                        stats["td_error"].append(log_dict["td_error"])

                        # mu_high_level = agent.get_mu(all_states, agent_state)
                        # pio = agent.get_pi(all_states, agent_state)
                        # if not hyper['use_only_primitive_actions']:
                        #     mu = env.lift_policy(mu_high_level, options_pi)
                        #     pi = env.lift_policy(pi_high_level, options_pi)
                        #     weights_high = np.tile(interest[..., None], (1, len(option_interests)))
                        # else:
                        #     weights_high = np.tile(interest[..., None], (1, env._num_actions))

                        q_pio = agent.get_q(all_states, agent_state)
                        v_pio = np.max(q_pio, axis=-1)
                        if hyper["use_true_options"]:
                            options_true_q_pi = options_q_star
                            options_q_pi = options_q_star
                            options_pi = options_q_star
                        elif not hyper["use_only_primitive_actions"]:
                            options_pi = agent.get_options_pi(all_states,
                                                              agent_state,
                                                              option_interests)

                            options_q_pi = agent.get_options_q(all_states, agent_state,
                                                               option_interests=option_interests)
                            options_true_q_pi = env.get_options_q_pi(options_pi,
                                                                     option_interests,
                                                                     v_pio,
                                                                     option_discount=agent_config["cfg"][
                                                                         "intra_option_discount"])
                            # options_f_pi = agent.get_options_f(all_states, agent_state,
                            #                                    option_interests=option_interests)
                            # true_options_f_pi = env.get_options_f_pi(agent.get_options_pi(all_states,
                            #                                                               agent_state,
                            #                                                               option_interests),
                            #                                              mu,
                            #                                              option_interests,
                            #                                              discount=agent_config["cfg"]["discount"])
                            # d_sa_mu = env.get_sa_stationary(mu,
                            #              discount=agent_config["cfg"]["discount"])

                            for option, option_interest in enumerate(option_interests):
                                weights = np.tile(option_interest[..., None], (1, env._num_actions))
                                ve = np.average((options_true_q_pi[option] - options_q_pi[option]) ** 2,
                                                weights=weights)
                                options_stats[option]["value_errors"].append(ve)
                                de = np.average((options_true_q_pi[option] - options_q_star[option]) ** 2,
                                                weights=weights)
                                options_stats[option]["decision_errors"].append(de)
                                options_stats[option]["td_error"].append(options_log_dict[option]["td_error"])
                                options_stats[option]["zloss"].append(options_log_dict[option]["zloss"])
                                options_stats[option]["values"].append(options_log_dict[option]["q"])
                                # plot_sa_values(env, options_q_star[option], invert_y=True,
                                #               invert_x=False,
                                #                title=f'option {option} v*')
                                options_stats[option]["f_td_or_mc_errors"].append(
                                    options_log_dict[option]["f_td_or_mc_error"])

                                # option_fe = np.average((true_options_f_pi[option] - options_f_pi[option])**2,
                                #                        weights=option_interest)
                                # options_stats[option]["f_exp_errors"].append(option_fe)
                                # Log the instantaneous variance
                                # option_f_var_e = (options_log_dict[option]["f"] - options_f_pi[option][s])**2
                                # options_stats[option]["f_var_errors"].append(option_f_var_e)

                                # if ep % log_every_episodes == 0:
                                #     if plot:
                                #         # plot_sa_values(env, options_true_q_pi[option], text_values=False,
                                #         #         title=f'option {option} true v_\pi')
                                #         weights = np.tile(option_interests[option][..., None], (1, env._num_actions))
                                #         plot_sa_values_4(env,
                                #                         options_pi[option],
                                #                         option_interests[option],
                                #                         options_q_pi[option] * weights,
                                #                         options_true_q_pi[option] * weights,
                                #                         options_q_star[option] * weights,
                                #                         text_values=False, title=r'$o_{}$'.format(option),
                                #                             subtitles=[r"$q^*$", r"$q$", r"$q_\pi$", r"$v,\pi$"])
                        else:
                            options_pi = [None for _ in range(4)]
                            options_q_pi = [None for _ in range(4)]
                            options_true_q_pi = [None for _ in range(4)]
                        if ep % log_every_episodes == 0 and verbose:
                            print(f'ep {ep} of {number_of_episodes}')
                        # plot_sa_values(env, options_q_pi[option], text_values=False,
                        #                title=f'option {option} v_\pi')
                        # option_v_star = np.max( options_q_star[option], axis=1)
                        # option_true_v_pi = np.max(options_true_q_pi[option], axis=1)
                        # option_v_pi = np.max(options_q_pi[option], axis=1)
                        # plot_s_values(env, option_v_star,
                        #                title=f'option {option} v*')
                        # plot_s_values(env, option_true_v_pi,
                        #                title=f'option {option} true v_pi')
                        # plot_s_values(env, option_v_pi,
                        #                title=f'option {option} v_pi')
                        # options_stats["rhos_prior"][option].append(options_log_dict[option]["rho_prior"])
                        # options_stats["rhos_post"][option].append(options_log_dict[option]["rho_post"])
                        # Log the values/action-values
                        # stats["values"].append(log_dict["q"])
                        # Log the td or the mc error for the follow-on trace
                        # stats["f_td_or_mc_errors"].append(log_dict["f_td_or_mc_error"])
                        # stats["zloss"].append(log_dict["zloss"])
                        # Log value error

                        # weights = np.tile(interest[..., None], (1, env._num_actions))
                        # plot_sa_values_pio(env,
                        #                 q_pio,
                        #                 options_q_star,
                        #                 q_star,
                        #                 interest,
                        #                 option_interests,
                        #                 text_values=False, title=r'$\pi_O$',
                        #                 subtitles=[r"$q^*$", r"$v,\pi_O$"])

                        # true_q_pi = env.get_q_pi(pi,
                        #              discount=agent_config["cfg"]["discount"])
                        # ve = np.average((true_q_pi_high_level - q_pi_high_level)**2, weights=d_sa_mu_high_level)
                        # stats["value_errors"].append(ve)
                        # plot_sa_values(env, q_pi,
                        #                    title=f'High-level q pi')
                        # v_pi_high_level = np.max(q_pi_high_level, axis=1)
                        # plot_s_values(env, v_pi,
                        #                    title=f'High-level v pi')
                        # f_pi = agent.get_f(all_states, agent_state)
                        # true_f_pi = env.get_f_pi(pi_high_level,
                        #                     mu_high_level,
                        #                     discount=agent_config["cfg"]["discount"])
                        # d_mu_high_level = env.get_s_stationary(mu_high_level, interest,
                        #         discount=agent_config["cfg"]["discount"])
                        # fe = np.average((true_f_pi - f_pi)**2, weights=d_mu_high_level)
                        # stats["f_exp_errors"].append(fe)
                        # Log the instantaneous variance
                        # f_var_e = (log_dict["f"] - f_pi[s])**2
                        # stats["f_var_errors"].append(f_var_e)
                        # w = np.asarray(agent_state.w["linear"]["w"].reshape((-1)))
                        # value_weights.append(w)
                        # stats["rhos_prior"].append(log_dict["rho_prior"])
                        # stats["rhos_post"].append(log_dict["rho_post"])
                        # de = np.average((true_q_pi - q_star)**2, weights=d_sa_mu)
                        # stats["decision_errors"].append(de)

                    if (number_of_steps > -1 and t >= number_of_steps) or \
                            (number_of_episodes > -1 and ep >= number_of_episodes):
                        break

                seeds_final_interest.append(interest)
                seeds_final_option_interests.append(option_interests)
                seeds_final_option_qstar.append(options_q_star)
                seeds_q_pi.append(q_pio)
                # seeds_true_q_pi.append(true_q_pi)
                seeds_final_option_pi.append(options_pi)
                seeds_final_option_q_pi.append(options_q_pi)
                seeds_final_true_option_q_pi.append(options_true_q_pi)
                for k in stats_keys:
                    seeds_stats[k].append(stats[k])
                    for option in range(len(option_interests)):
                        seeds_options_stats[option][k].append(options_stats[option][k])
            agents_final_q_pi.append(seeds_q_pi)
            agents_final_q_star.append(seeds_final_q_star)
            agents_final_option_interests.append(seeds_final_option_interests)
            agents_final_interest.append(seeds_final_interest)
            agents_final_option_qstar.append(seeds_final_option_qstar)
            agents_final_option_pi.append(seeds_final_option_pi)
            agents_final_option_q_pi.append(seeds_final_option_q_pi)
            agents_final_true_option_q_pi.append(seeds_final_true_option_q_pi)
            for k in stats_keys:
                agents_stats[k].append(seeds_stats[k])
                for option in range(len(option_interests)):
                    agents_options_stats[option][k].append(seeds_options_stats[option][k])

    return (agents_stats, agents_final_interest, agents_final_option_interests,
            agents_final_q_star, agents_final_q_pi,
            agents_options_stats, agents_final_option_qstar,
            agents_final_option_pi, agents_final_option_q_pi,
            agents_final_true_option_q_pi, env)

def run_eval(ep,eval_env, agent, agent_state, stats, max_episode_timesteps=1000,
             number_of_eval_episodes=10):

    eval_ep_timesteps = []
    eval_returns = []
    for state in eval_env._start_states:
        timestep = None
        episode_timesteps = 0
        t = 0
        ep_return = 0
        while(True):
            # Reset environment and pick first action
            if not timestep or timestep.last():
                timestep = eval_env.reset(state=state)
                s = eval_env._curr_idx
                action, agent_state = agent.first_step(timestep, s, agent_state, eval=True)

            prev_s = s
            prev_obs = agent_state.obs
            timestep = eval_env.step(action)

            next_s = eval_env._curr_idx
            action, agent_state, log_dict, options_log_dict = agent.step(
                timestep=timestep, s=s, eval=True,
                next_s=next_s, agent_state=agent_state)

            episode_timesteps += 1
            ep_return += timestep.reward
            t += 1
            s = next_s

            # print(f'ep {ep}',
            #       f'steps: {t:10}',
            #       f', eval episode_timesteps {stats["eval_ep_timesteps"]} eval returns: {stats["eval_returns"]}'
            #       f', episode_end: {timestep.last()}',
            #       f', option: {agent_state.option}',
            #       f', option_reward {agent_state.option_reward}',
            #       f', action: {action}',
            #       f', discount: {timestep.discount}',
            #       f', state {prev_s}',
            #       # f', obs {prev_obs}',
            #       f', next_state {next_s}',
            #       # f', next_obs {timestep.observation}'
            #       )

            if timestep.last() or episode_timesteps >= max_episode_timesteps:
                eval_ep_timesteps.append(episode_timesteps)
                eval_returns.append(ep_return)
                # print(f"start state {state} "
                #       f"start x {eval_env.get_state_xy(state)[0]}, y {eval_env.get_state_xy(state)[1]} "
                #       f"timesteps {episode_timesteps}")
                break
    # print(f"{ep}_{eval_ep_timesteps}")
    stats["eval_ep_timesteps"].append(np.mean(eval_ep_timesteps))
    stats["eval_returns"].append(np.mean(eval_returns))
    return stats