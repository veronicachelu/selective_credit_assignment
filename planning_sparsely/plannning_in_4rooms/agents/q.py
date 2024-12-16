# @title Q(lambda) agent
from imports import *
from utils import *

class Agent(NamedTuple):
    step: Callable[..., Any]
    first_step: Callable[..., Any]
    get_q: Callable[..., Any]
    get_options_q: Callable[..., Any]
    get_options_f: Callable[..., Any]
    get_mu: Callable[..., Any]
    get_pi: Callable[..., Any]
    get_options_pi: Callable[..., Any]
    get_f: Callable[..., Any]


class AgentState(NamedTuple):
    """Agent state."""
    rng_key: jnp.ndarray
    w_q: Any
    w_z: Any
    w_x: Any
    e_q: Any
    e_x: Any
    e_z: Any
    q_opt_state: Any
    x_opt_state: Any
    z_opt_state: Any
    f_opt_state: Any
    trace_decay: float
    obs: Any
    action: Any
    f: Any
    w_f: Any
    e_f: Any
    option: Any
    behaviour_action_prob: Any
    option_reward: Any
    followon_decay: Any
    option_states: Any


class OptionState(NamedTuple):
    """Agent state."""
    interest: Any
    w_q: Any
    w_z: Any
    e_q: Any
    e_z: Any
    q_opt_state: Any
    z_opt_state: Any
    f_opt_state: Any
    f: Any
    w_f: Any
    e_f: Any
    option_q_star: Any
    followon_decay: Any
    trace_decay: float


def get_config(agent_name, hyper,
               option_interests=None,
               options_q_star=None,
               option_reward=1.,
               interest=None):
    """Hyper-parameters for this agent."""
    config = {}
    config["init"] = init
    config["cfg"] = {}

    config["cfg"]["epsilon"] = hyper["epsilon"]
    config["cfg"]["option_epsilon"] = hyper["option_epsilon"]
    config["cfg"]["discount"] = hyper["discount"]
    config["cfg"]["intra_option_discount"] = hyper["intra_option_discount"]
    config["cfg"]["trace_parameter"] = hyper["lambda"]
    config["cfg"]["use_only_primitive_actions"] = hyper["use_only_primitive_actions"]

    config["cfg"]["option_interests"] = option_interests
    config["cfg"]["interest"] = interest

    config["cfg"]["use_discount_magic_rule"] = hyper["use_discount_magic_rule"]
    config["cfg"]["option_reward"] = option_reward
    config["cfg"]["q_optimiser_kwargs"] = {}
    config["cfg"]["q_optimiser_kwargs"]["learning_rate"] = hyper["alpha_q"]
    config["cfg"]["q_optimiser_kwargs"]["b1"] = 0.9
    config["cfg"]["q_optimiser_kwargs"]["b2"] = 0.999
    config["cfg"]["q_optimiser_kwargs"]["eps"] = 1e-4
    config["cfg"]["q_optimiser_kwargs"]["momentum"] = 0.9
    config["cfg"]["q_optimiser_kwargs"]["d"] = hyper["d"]
    config["cfg"]["learn_trace_by_magic"] = hyper["learn_trace_by_magic"]

    config["cfg"]["option_q_optimiser_kwargs"] = {}
    config["cfg"]["option_q_optimiser_kwargs"]["learning_rate"] = hyper["option_alpha_q"]
    config["cfg"]["option_q_optimiser_kwargs"]["b1"] = 0.9
    config["cfg"]["option_q_optimiser_kwargs"]["b2"] = 0.999
    config["cfg"]["option_q_optimiser_kwargs"]["eps"] = 1e-4
    config["cfg"]["option_q_optimiser_kwargs"]["momentum"] = 0.9
    config["cfg"]["option_q_optimiser_kwargs"]["d"] = hyper["d"]

    config["cfg"]["f_optimiser_kwargs"] = {}
    config["cfg"]["f_optimiser_kwargs"]["learning_rate"] = hyper["alpha_f"]
    config["cfg"]["f_optimiser_kwargs"]["b1"] = 0.9
    config["cfg"]["f_optimiser_kwargs"]["b2"] = 0.999
    config["cfg"]["f_optimiser_kwargs"]["eps"] = 1e-4
    config["cfg"]["f_optimiser_kwargs"]["momentum"] = 0.9
    config["cfg"]["f_optimiser_kwargs"]["d"] = hyper["d"]

    config["cfg"]["option_f_optimiser_kwargs"] = {}
    config["cfg"]["option_f_optimiser_kwargs"]["learning_rate"] = hyper["option_alpha_f"]
    config["cfg"]["option_f_optimiser_kwargs"]["b1"] = 0.9
    config["cfg"]["option_f_optimiser_kwargs"]["b2"] = 0.999
    config["cfg"]["option_f_optimiser_kwargs"]["eps"] = 1e-4
    config["cfg"]["option_f_optimiser_kwargs"]["momentum"] = 0.9
    config["cfg"]["option_f_optimiser_kwargs"]["d"] = hyper["d"]

    config["cfg"]["z_optimiser_kwargs"] = {}
    config["cfg"]["z_optimiser_kwargs"]["learning_rate"] = hyper["alpha_z"]
    config["cfg"]["z_optimiser_kwargs"]["b1"] = 0.9
    config["cfg"]["z_optimiser_kwargs"]["b2"] = 0.999
    config["cfg"]["z_optimiser_kwargs"]["eps"] = 1e-4
    config["cfg"]["z_optimiser_kwargs"]["momentum"] = 0.9
    config["cfg"]["z_optimiser_kwargs"]["d"] = hyper["dz"]

    config["cfg"]["use_magic_rule"] = hyper["use_magic_rule"]
    config["cfg"]["option_z_optimiser_kwargs"] = {}
    config["cfg"]["option_z_optimiser_kwargs"]["learning_rate"] = hyper["option_alpha_z"]
    config["cfg"]["option_z_optimiser_kwargs"]["b1"] = 0.9
    config["cfg"]["option_z_optimiser_kwargs"]["b2"] = 0.999
    config["cfg"]["option_z_optimiser_kwargs"]["eps"] = 1e-4
    config["cfg"]["option_z_optimiser_kwargs"]["momentum"] = 0.9
    config["cfg"]["option_z_optimiser_kwargs"]["d"] = hyper["dz"]

    # Default is the peng and williams q lambda
    config["cfg"]["use_true_options"] = hyper["use_true_options"]
    config["cfg"]["options_q_star"] = options_q_star
    config["cfg"]["use_post_corr"] = hyper["use_post_corr"]
    config["cfg"]["option_use_post_corr"] = hyper["option_use_post_corr"]
    config["cfg"]["use_prior_corr"] = hyper["use_prior_corr"]
    config["cfg"]["option_use_prior_corr"] = hyper["option_use_prior_corr"]
    config["cfg"]["eta_f"] = hyper["eta_f"]
    config["cfg"]["eta_x_f"] = hyper["eta_x_f"]
    config["cfg"]["option_eta_f"] = hyper["option_eta_f"]
    config["cfg"]["option_eta_x_f"] = hyper["option_eta_x_f"]
    config["cfg"]["eta_q"] = hyper["eta_q"]
    config["cfg"]["eta_z"] = hyper["eta_z"]
    config["cfg"]["option_eta_q"] = hyper["option_eta_q"]
    config["cfg"]["option_eta_z"] = hyper["option_eta_z"]
    config["cfg"]["beta"] = hyper["beta"]
    config["cfg"]["option_beta"] = hyper["option_beta"]
    config["cfg"]["use_q_for_reward"] = hyper["use_q_for_reward"]

    return config


# @title Default title text
def first_step(
        timestep: dm_env.TimeStep,
        s: Any,
        agent_state: AgentState,
        interest_fn: Callable[..., Any],
        policy: Callable[..., Any],
        option_policies: Callable[..., Any],
        option_interest_fns: Callable[..., Any],
        option_q_fns: Any,
        eval=False,
) -> Tuple[Any, AgentState]:
    rng_key, policy_key = jax.random.split(agent_state.rng_key)

    interest = interest_fn(s)
    option, option_prob = policy(policy_key, timestep.observation,
                                 s, agent_state, option_interest_fns, eval=eval)
    rng_key, policy_key = jax.random.split(rng_key)
    action, behaviour_prob = option_policies(policy_key, timestep.observation,
                                             s,
                                             agent_state,
                                             option, option_prob,
                                             option_q_fns)
    agent_state = agent_state._replace(
        rng_key=rng_key, obs=timestep.observation,
        action=action, option=option, behaviour_action_prob=behaviour_prob)
    return action, agent_state


def step(
        interest_fn: Callable[..., Any],
        # stateful things
        timestep: dm_env.TimeStep,
        s: Any,
        next_s: Any,
        eval: Any,
        agent_state: AgentState,
        # functions
        use_only_primitive_actions: bool,
        x_scale_fn: Callable[..., Any],
        q_scale_fn: Callable[..., Any],
        z_scale_fn: Callable[..., Any],
        x_fn: Callable[..., Any],
        q_fn: Callable[..., Any],
        dq_fn: Callable[..., Any],
        z_fn: Callable[..., Any],
        dzloss_fn: Callable[..., Any],
        policy: Callable[..., Any],
        option_policies: Callable[..., Any],
        f_scale_fn: Callable[..., Any],
        f_fn: Callable[..., Any],
        df_fn: Callable[..., Any],
        rho_fn: Callable[..., Any],
        # hyper-parameters
        discount: float,
        use_discount_magic_rule: bool,
        learn_trace_by_magic: bool,
        intra_option_discount: float,
        trace_parameter: float,
        eta_q: float,
        eta_z: float,
        eta_f: float,
        eta_x_f: float,
        beta: float,
        use_prior_corr: bool,
        epsilon: float,
        option_q_fns: Any,
        option_interest_fns: Any,
        use_magic_rule: Any,
        use_true_options: Any,
        **option_kwargs,
) -> Tuple[Any, AgentState, Mapping[str, Any]]:  # hyper-parameters
    """One step of the agent."""

    # Map action structure to an int
    # action = jax.tree_leaves(agent_state.action)[0]
    reward = timestep.reward
    discount = timestep.discount * discount
    option_reward = agent_state.option_reward
    option_reward += reward
    intra_option_discount = timestep.discount * intra_option_discount
    next_obs = timestep.observation
    episode_end = timestep.last()
    (f, e_f, e_q,
     e_x, e_z) = (agent_state.f, agent_state.e_f, agent_state.e_q,
                  agent_state.e_x, agent_state.e_z)
    x_f = 0
    del timestep
    trace_decay = agent_state.trace_decay
    (w_q, w_x,
     w_z, w_f) = (agent_state.w_q, agent_state.w_x,
                  agent_state.w_z, agent_state.w_f)
    obs = agent_state.obs
    option = agent_state.option
    option = jax.tree_leaves(option)[0]
    f_opt_state = agent_state.f_opt_state
    # rng_key, policy_key = jax.random.split(agent_state.rng_key)

    ### Compute values
    # Compute the features x(s)
    x = x_fn(w_x, obs)
    next_x = x_fn(w_x, next_obs)
    # Compute value v(s)
    q = q_fn(w_q, x)
    qa = q[option]

    # Compute gradient dv(s)/dw, d log pi(s, a)/dw
    dq_head, dq_torso = dq_fn(w_q, w_x, obs, option)

    # Compute v(s) at the next time step
    q_next = q_fn(w_q, next_x)
    v_next = jnp.max(q_next)

    interest = interest_fn(s)

    rho_post, rho_prior = rho_fn(agent_state, obs, option)
    # Pick new action
    # We do this before updating the weights to avoid forwarding twice
    rng_key, policy_key = jax.random.split(agent_state.rng_key)

    option_next, next_option_prob = policy(policy_key, next_obs, next_s,
                                           agent_state, option_interest_fns, eval=eval)
    rng_key, policy_key = jax.random.split(rng_key)
    a_next, behaviour_action_prob_next = option_policies(policy_key,
                                                         next_obs, next_s,
                                                         agent_state,
                                                         option_next, next_option_prob,
                                                         option_q_fns)

    ### Expected trace
    # Compute the expected trace: z(s) ~= E[ γ λ e_{t-1} | S_t=s ]
    z = z_fn(w_z, x)
    # Decay all traces
    e_q, e_z, e_x = tmap(lambda e_i: trace_decay * e_i, (e_q, e_z, e_x))

    # Compute update for the expected trace
    zloss, dw_z = dzloss_fn(w_z, x, e_z, interest)

    # Compute mixture traces η γ λ e_{t-1} + (1 - η) z(S_t)
    # Note: z(S_t) ~= E[ γ λ e_{t-1} | S_t ]
    if learn_trace_by_magic:
        eta_q = eta_q * interest + 1 - interest
        eta_z = eta_z * interest + 1 - interest
    e_q = tmap(lambda e_i, z_i: eta_q * e_i + (1 - eta_q) * z_i, e_q, z)
    e_z = tmap(lambda e_i, z_i: eta_z * e_i + (1 - eta_z) * z_i, e_z, z)

    f_error = 0

    if use_prior_corr:
        # Decay all traces
        e_f = agent_state.followon_decay * e_f
        f = agent_state.followon_decay * f
        # Compute expected follow-on
        x_f = f_fn(w_f, x)

        # Learn expected follow-on
        df = df_fn(w_f, w_x, obs)
        f_error = e_f - x_f
        dw_f = tmap(lambda df_i: f_error * df_i, df)
        dw_f, f_opt_state = f_scale_fn(dw_f, state=f_opt_state)
        w_f = tmap(lambda x, y: x + y, w_f, dw_f)

        # Compute the traces as eta interpolations between TD and MC
        f = eta_f * f + (1 - eta_f) * x_f + interest
        e_f = eta_x_f * e_f + (1 - eta_x_f) * x_f + interest
        m = trace_parameter * interest + (1 - trace_parameter) * f
        next_trace_parameter = trace_parameter
    elif use_magic_rule:
        m = interest
        next_interest = interest_fn(next_s)
        # discount = discount * next_interest + (1 - next_interest)
        # custom_trace_parameter = trace_parameter * interest + 1 - interest
        next_trace_parameter = trace_parameter * next_interest + 1 - next_interest
    else:
        m = f = interest
        next_trace_parameter = trace_parameter
        option_reward = reward

    if use_discount_magic_rule:
        discount = discount * next_interest + (1 - next_interest)
    # Compute the actual emphasis from the follow-on

    ### Add gradient to traces
    tree_add_weighted = lambda x, y, z: tmap(lambda x, y: x + y * z, x, y)
    e_q = tree_add_weighted(e_q, dq_head, m)
    e_z = tree_add_weighted(e_z, dq_head, m)
    e_x = tree_add_weighted(e_x, dq_torso, m)

    ### Update values
    # Compute term that composes the multi-step λ return
    # discount = discount * interest + (1 - interest)
    r_plus_next_v = option_reward + (1 - next_trace_parameter) * discount * v_next
    compute_dw = lambda e_i, dq_i: r_plus_next_v * e_i - qa * dq_i * m
    # Compute update for the values
    dw_q, dw_x = tmap(compute_dw, (e_q, e_x), (dq_head, dq_torso))
    dw_q, q_opt_state = q_scale_fn(dw_q, state=agent_state.q_opt_state)
    dw_x, x_opt_state = x_scale_fn(dw_x, state=agent_state.x_opt_state)
    dw_z, z_opt_state = z_scale_fn(dw_z, state=agent_state.z_opt_state)

    # Update weights
    w_q = tmap(lambda x, y: x + y, w_q, dw_q)
    w_x = tmap(lambda x, y: x + y, w_x, dw_x)
    w_z = tree_sub(w_z, dw_z)  # subtract, for gradient descent

    # Compute trace decay to apply at the next time step
    trace_decay = rho_post * discount * next_trace_parameter
    trace_decay *= 1 - jnp.float32(episode_end)
    followon_decay = rho_prior * jnp.minimum(discount, beta) * (1 - jnp.float32(episode_end))
    if use_true_options or use_only_primitive_actions:
        option_log_dicts = None
        option_states = agent_state.option_states
    else:
        option_states, option_log_dicts = update_options(
            s=s, next_s=next_s,
            obs=obs, next_obs=next_obs,
            x=x, next_x=next_x,
            w_x=w_x, option_q_fns=option_q_fns,
            episode_end=episode_end,
            trace_parameter=trace_parameter,
            intra_option_discount=intra_option_discount,
            agent_state=agent_state,
            option_states=agent_state.option_states,
            option_interest_fns=option_interest_fns,
            **option_kwargs)

    # Add some things to log
    log_dict = dict(q=qa, f=f, e_f=e_f, x_f=x_f,
                    zloss=zloss,
                    rho_prior=rho_prior,
                    rho_post=rho_post,
                    discount=discount, f_td_or_mc_error=f_error,
                    td_error=option_reward + discount * v_next - qa)

    # Assemble new agent state
    agent_state = AgentState(
        rng_key=rng_key,
        w_q=w_q,
        w_z=w_z,
        w_x=w_x,
        w_f=w_f,
        e_q=e_q,
        e_z=e_z,
        e_x=e_x,
        e_f=e_f,
        f=f,
        followon_decay=followon_decay,
        trace_decay=trace_decay,
        q_opt_state=q_opt_state,
        f_opt_state=f_opt_state,
        x_opt_state=x_opt_state,
        z_opt_state=z_opt_state,
        obs=next_obs,
        action=a_next,
        option_reward=option_reward * (1 - interest),
        option=option_next,
        behaviour_action_prob=behaviour_action_prob_next,
        option_states=option_states
    )

    return a_next, agent_state, log_dict, option_log_dicts


def update_options(obs, next_obs,
                   s, next_s,
                   x, next_x,
                   episode_end,
                   trace_parameter, option_beta,
                   w_x, intra_option_discount,
                   option_eta_z, option_eta_q,
                   option_eta_f, option_eta_x_f,
                   agent_state,
                   option_states,
                   option_q_fns,
                   option_dq_fns,
                   option_z_fns,
                   option_f_fns,
                   option_df_fns,
                   option_dzloss_fns,
                   option_q_scale_fns,
                   option_z_scale_fns,
                   option_f_scale_fns,
                   option_interest_fns,
                   option_reward_fns,
                   option_discount_fns,
                   option_use_prior_corr,
                   option_rho_fn,
                   option_policy):
    new_option_states = []
    option_log_dicts = []

    for option, (option_state, option_q_fn, option_f_fn,
                 option_dq_fn, option_z_fn, option_df_fn,
                 option_dzloss_fn, option_q_scale_fn,
                 option_z_scale_fn, option_f_scale_fn, option_reward_fn,
                 option_interest_fn, option_discount_fn) in enumerate(zip(option_states,
                                                                          option_q_fns, option_f_fns, option_dq_fns,
                                                                          option_z_fns,
                                                                          option_df_fns, option_dzloss_fns,
                                                                          option_q_scale_fns,
                                                                          option_z_scale_fns, option_f_scale_fns,
                                                                          option_reward_fns,
                                                                          option_interest_fns, option_discount_fns)):
        option_state, option_log_dict = update_option(s=s, next_s=next_s,
                                                      obs=obs, next_obs=next_obs,
                                                      x=x, next_x=next_x, intra_option_discount=intra_option_discount,
                                                      w_x=w_x, option_beta=option_beta,
                                                      trace_parameter=trace_parameter,
                                                      episode_end=episode_end,
                                                      option_eta_q=option_eta_q,
                                                      option_eta_z=option_eta_z,
                                                      option_eta_f=option_eta_f,
                                                      option_use_prior_corr=option_use_prior_corr,
                                                      option_eta_x_f=option_eta_x_f,
                                                      option_interest_fn=option_interest_fn,
                                                      option_reward_fn=option_reward_fn,
                                                      option_discount_fn=option_discount_fn,
                                                      agent_state=agent_state,
                                                      option_state=option_state,
                                                      option_q_fn=option_q_fn,
                                                      option_f_fn=option_f_fn,
                                                      option_dq_fn=option_dq_fn,
                                                      option_df_fn=option_df_fn,
                                                      option_z_fn=option_z_fn,
                                                      option_dzloss_fn=option_dzloss_fn,
                                                      option_q_scale_fn=option_q_scale_fn,
                                                      option_z_scale_fn=option_z_scale_fn,
                                                      option_f_scale_fn=option_f_scale_fn,
                                                      option_rho_fn=option_rho_fn,
                                                      option_policy=option_policy)
        new_option_states.append(option_state)
        option_log_dicts.append(option_log_dict)
    return new_option_states, option_log_dicts


def update_option(obs, s, next_s, x, w_x, next_obs, next_x,
                  trace_parameter, option_beta, intra_option_discount, episode_end,
                  option_eta_q, option_eta_z, option_use_prior_corr,
                  agent_state, option_eta_f, option_eta_x_f,
                  option_state, option_q_fn, option_dq_fn,
                  option_z_fn, option_dzloss_fn,
                  option_interest_fn, option_f_fn, option_df_fn,
                  option_reward_fn, option_discount_fn,
                  option_q_scale_fn, option_z_scale_fn, option_f_scale_fn,
                  option_rho_fn, option_policy):
    behaviour_prob = agent_state.behaviour_action_prob
    action = jax.tree_leaves(agent_state.action)[0]
    option_w_q = option_state.w_q
    option_w_z = option_state.w_z
    option_w_f = option_state.w_f
    option_trace_decay = option_state.trace_decay
    option_followon_decay = option_state.followon_decay
    # option_rng_key = option_state.rng_key
    option_q = option_q_fn(option_w_q, x)
    option_qa = option_q[action]
    option_e_q, option_e_z = option_state.e_q, option_state.e_z
    option_e_f, option_f = option_state.e_f, option_state.f
    option_f_opt_state = option_state.f_opt_state
    # Compute gradient dv(s)/dw, d log pi(s, a)/dw
    option_dq_head, _ = option_dq_fn(option_w_q, w_x, obs, action)

    # Compute v(s) at the next time step
    option_q_next = option_q_fn(option_w_q, next_x)
    option_v_next = jnp.max(option_q_next)

    option_rho_post, option_rho_prior = option_rho_fn(agent_state,
                                                      behaviour_prob,
                                                      option_state,
                                                      obs, action,
                                                      option_q_fn,
                                                      )

    ### Expected trace
    # Compute the expected trace: z(s) ~= E[ γ λ e_{t-1} | S_t=s ]
    option_z = option_z_fn(option_w_z, x)
    # Decay all traces
    option_e_q, option_e_z = tmap(lambda e_i: option_trace_decay * e_i,
                                  (option_e_q, option_e_z))

    # Compute update for the expected trace
    option_zloss, option_dw_z = option_dzloss_fn(option_w_z, x, option_e_z)

    # Compute mixture traces η γ λ e_{t-1} + (1 - η) z(S_t)
    # Note: z(S_t) ~= E[ γ λ e_{t-1} | S_t ]
    option_e_q = tmap(lambda e_i, z_i: option_eta_q * e_i + (1 - option_eta_q) * z_i, option_e_q, option_z)
    option_e_z = tmap(lambda e_i, z_i: option_eta_z * e_i + (1 - option_eta_z) * z_i, option_e_z, option_z)

    interest = option_interest_fn(s)
    next_interest = option_interest_fn(next_s)
    option_f_error = 0
    if option_use_prior_corr:
        # Decay all traces
        option_e_f = option_followon_decay * option_e_f
        option_f = option_followon_decay * option_f
        # Compute expected follow-on
        option_x_f = option_f_fn(option_w_f, x)

        # Learn expected follow-on
        option_df = option_df_fn(option_w_f, w_x, obs)
        option_f_error = option_e_f - option_x_f
        option_dw_f = tmap(lambda df_i: option_f_error * df_i, option_df)
        option_dw_f, option_f_opt_state = option_f_scale_fn(option_dw_f, state=option_f_opt_state)
        option_w_f = tmap(lambda x, y: x + y, option_w_f, option_dw_f)

        # Compute the traces as eta interpolations between TD and MC
        option_f = option_eta_f * option_f + (1 - option_eta_f) * option_x_f + interest
        option_e_f = option_eta_x_f * option_e_f + (1 - option_eta_x_f) * option_x_f + interest
        option_m = trace_parameter * interest + (1 - trace_parameter) * option_f
        next_trace_parameter = trace_parameter
    else:
        option_m = option_f = interest
        next_trace_parameter = trace_parameter * next_interest + (1 - next_interest)

    ### Add gradient to traces
    tree_add_weighted = lambda x, y, z: tmap(lambda x, y: x + y * z, x, y)
    option_e_q = tree_add_weighted(option_e_q, option_dq_head, option_m)
    option_e_z = tree_add_weighted(option_e_z, option_dq_head, option_m)

    ### Update values
    # Compute term that composes the multi-step λ return
    reward = option_reward_fn(s, action, next_s, option_v_next)
    discount = intra_option_discount * option_discount_fn(next_s)
    option_r_plus_next_v = reward + (1 - next_trace_parameter) * discount * option_v_next
    option_compute_dw = lambda e_i, option_dq_i: option_r_plus_next_v * e_i - option_qa * option_dq_i * option_m
    # Compute update for the values
    option_dw_q = tmap(option_compute_dw, option_e_q, option_dq_head)
    option_dw_q, option_q_opt_state = option_q_scale_fn(option_dw_q, state=option_state.q_opt_state)
    option_dw_z, option_z_opt_state = option_z_scale_fn(option_dw_z, state=option_state.z_opt_state)

    # Update weights
    option_w_q = tmap(lambda x, y: x + y, option_w_q, option_dw_q)
    option_w_z = tree_sub(option_w_z, option_dw_z)  # subtract, for gradient descent

    # Compute trace decay to apply at the next time step
    option_trace_decay = option_rho_post * discount * next_trace_parameter
    option_trace_decay *= 1 - jnp.float32(episode_end)
    option_followon_decay = option_rho_prior * jnp.minimum(discount, option_beta) * (1 - jnp.float32(episode_end))

    option_state = OptionState(
        # rng_key=option_rng_key,
        interest=next_interest,
        w_q=option_w_q,
        w_f=option_w_f,
        w_z=option_w_z,
        e_q=option_e_q,
        e_z=option_e_z,
        e_f=option_e_f,
        f=option_f,
        option_q_star=option_state.option_q_star,
        trace_decay=option_trace_decay,
        followon_decay=option_followon_decay,
        q_opt_state=option_q_opt_state,
        z_opt_state=option_z_opt_state,
        f_opt_state=option_f_opt_state,
    )

    option_log_dict = dict(q=option_qa,
                           interest=interest,
                           option_m=option_m,
                           f=option_f,
                           e_f=option_e_f,
                           reward=reward,
                           f_td_or_mc_error=option_f_error,
                           trace_decay=option_trace_decay,
                           zloss=option_zloss,
                           rho_prior=option_rho_prior,
                           rho_post=option_rho_post,
                           discount=discount,
                           td_error=reward + discount * option_v_next - option_qa)

    return option_state, option_log_dict


def init(
        rng_key: jnp.ndarray,
        network_spec: str,
        action_spec: Any,
        # action_and_option_spec: Any,
        option_spec: Any,
        observation_spec: Any,
        discount: float,
        use_discount_magic_rule: bool,
        trace_parameter: float,
        q_optimiser_kwargs: Mapping[str, Any],
        option_q_optimiser_kwargs: Mapping[str, Any],
        f_optimiser_kwargs: Mapping[str, Any],
        option_f_optimiser_kwargs: Mapping[str, Any],
        option_z_optimiser_kwargs: Mapping[str, Any],
        z_optimiser_kwargs: Mapping[str, Any],
        use_prior_corr: bool,
        option_use_prior_corr: bool,
        option_use_post_corr: bool,
        use_post_corr: bool,
        option_eta_f: float,
        option_eta_x_f: float,
        eta_f: float,
        eta_x_f: float,
        eta_q: float,
        eta_z: float,
        option_eta_q: float,
        option_eta_z: float,
        beta: float,
        option_beta: float,
        epsilon: float,
        option_epsilon: float,
        use_only_primitive_actions: bool,
        option_interests: Any,
        interest: Any,
        option_reward: Any,
        use_q_for_reward: Any,
        intra_option_discount: Any,
        use_magic_rule: Any,
        use_true_options: Any,
        options_q_star: Any,
        learn_trace_by_magic: Any,
) -> Tuple[Agent, AgentState]:
    """Initialise TD(λ) agent."""
    # Check the action spec
    action_structure = check_simple_action_spec(action_spec)
    # num_actions = _get_num_actions(action_spec)

    option_structure = check_simple_action_spec(option_spec)
    # num_options = _get_num_actions(option_spec)

    # action_and_option_structure = check_simple_action_spec(action_and_option_spec)
    # num_actions_and_options = _get_num_actions(action_and_option_spec)

    # Create random keys
    (net_key_x, net_key_q,
     net_key_z, net_key_f,
     net_key_options, agent_key) = jax.random.split(rng_key, 6)

    # Create Haiku network, and initialise its weights
    x_fn, w_x = create_torso(net_key_x, network_spec, observation_spec)
    # Create haiku action-value network, and initialise its weights
    dummy_observation = dummy(observation_spec)
    dummy_x = x_fn(w_x, dummy_observation)
    if use_only_primitive_actions:
        q_fn, w_q = create_q_head(net_key_q, dummy_x, action_spec)
    else:
        q_fn, w_q = create_q_head(net_key_q, dummy_x, option_spec)

    f_fn_, w_f = create_v_head(net_key_f, dummy_x)
    f_fn = lambda w, x: jnp.squeeze(f_fn_(w, x), -1)

    def rho_fn(agent_state, obs, action):
        x = x_fn(agent_state.w_x, obs)
        q = q_fn(agent_state.w_q, x)
        pi = (q == q.max(axis=-1, keepdims=True)).astype(jnp.float32)
        pi /= jnp.sum(pi, axis=-1, keepdims=True)
        mu = (1 - epsilon) * pi + epsilon / q.shape[-1]
        rho_prior = pi[action] / mu[action]
        rho_post = rho_prior
        if not use_post_corr:
            rho_post = 1
        return rho_post, rho_prior

    def option_rho_fn(agent_state, behaviour_prob, option_state, obs, action,
                      option_q_fn):
        x = x_fn(agent_state.w_x, obs)
        q_option = option_q_fn(option_state.w_q, x)

        pi_option = (q_option == q_option.max(axis=-1, keepdims=True)).astype(jnp.float32)
        pi_option /= jnp.sum(pi_option, axis=-1, keepdims=True)

        rho_prior = pi_option[action] / behaviour_prob
        rho_post = rho_prior
        if not option_use_post_corr:
            rho_post = 1
        return rho_post, rho_prior

    # Create function for the gradients of the action values
    dq_fn = jax.grad(
        lambda w_q, w_x, o, a: q_fn(w_q, x_fn(w_x, o))[a], argnums=(0, 1))

    # Create function for the gradients of the v function
    df_fn = jax.grad(lambda w_f, w_x, o: f_fn(w_f, x_fn(w_x, o)))

    # Initialise eligibility traces
    e_q, e_z, e_x = tmap(jnp.zeros_like, (w_q, w_q, w_x))

    (dqdw, _) = dq_fn(w_q, w_x, dummy_observation, 0)
    z_fn, w_z = create_z_head(net_key_z, dummy_x, dqdw)

    # Create loss function for the expected trace
    def zloss(w_z, x, e, interest):
        total_size = sum([e_i.size for e_i in jax.tree_leaves(e)])
        z = z_fn(w_z, x)
        sum_of_squares = tmap(lambda z_i, e_i: jnp.sum((z_i - e_i) ** 2), z, e)
        loss = 0.5 * sum(jax.tree_leaves(sum_of_squares)) / total_size
        if learn_trace_by_magic:
            loss *= interest
        return loss

    # Gradient of the expected trace loss (also outputs the loss, for logging)
    dzloss_fn = jax.value_and_grad(zloss)

    # Create softmax policy
    def policy(rng_key, obs, s, agent_state, option_interest_fns, eval=False):
        x = x_fn(agent_state.w_x, obs)
        q = q_fn(agent_state.w_q, x)
        # if not use_only_primitive_actions:
        #     option_interests = []
        #     for option, option_interest_fn in enumerate(option_interest_fns):
        #         option_interest = option_interest_fn(s)
        #         option_interests.append(option_interest)
        #     q = jnp.where(jnp.asarray(option_interests) == 0, -jnp.inf, q)
        local_epsilon = jnp.where(eval, 0., epsilon)
        next_op, next_op_prob = eps_greedy(rng_key=rng_key, q=q, epsilon=local_epsilon)

        if not use_only_primitive_actions:
            # behaviour_probs = []
            i = interest_fn(s)
            next_op = jnp.where(i, next_op, agent_state.option)
            next_op_prob = jnp.where(i, next_op_prob, agent_state.behaviour_action_prob)
        # next_op = jax.tree_unflatten(action_structure, [next_op])
        return next_op, next_op_prob

    def option_policies(rng_key, obs, s, agent_state,
                        option, option_prob, option_q_fns):
        if use_only_primitive_actions:
            a = option
            behaviour_prob = option_prob
        else:
            actions = []
            behaviour_probs = []
            for op, option_state in enumerate(agent_state.option_states):
                if use_true_options:
                    q = option_state.option_q_star[s]
                    a, behaviour_prob = eps_greedy(rng_key=rng_key, q=q, epsilon=option_epsilon)
                    a = jax.tree_unflatten(action_structure, [a])
                else:
                    a, behaviour_prob = option_policy(rng_key, obs, agent_state, option_state, option_q_fns[op])
                actions.append(a)
                behaviour_probs.append(behaviour_prob)
            a = jnp.array(actions)[option]
            # import pdb; pdb.set_trace()
            behaviour_prob = jnp.array(behaviour_probs)[option]

        return a, behaviour_prob

    def option_policy(rng_key, obs, agent_state, option_state, option_q_fn):
        x = x_fn(agent_state.w_x, obs)
        q = option_q_fn(option_state.w_q, x)
        a, prob = eps_greedy_policy(rng_key=rng_key, q=q, epsilon=option_epsilon)
        a = jax.tree_unflatten(action_structure, [a])
        return a, prob

    def get_q(all_states, agent_state):
        q_all_states = jax.vmap(lambda obs: q_fn(agent_state.w_q,
                                                 x_fn(agent_state.w_x, obs)))(jnp.asarray(all_states))
        return q_all_states

    def get_f(all_states, agent_state):
        f_all_states = jax.vmap(lambda obs: f_fn(agent_state.w_f,
                                                 x_fn(agent_state.w_x, obs)))(jnp.asarray(all_states))
        return f_all_states

    def interest_fn(s):
        return jnp.asarray(interest)[s]

    def get_mu(all_states, agent_state):
        def mu_fn(obs):
            x = x_fn(agent_state.w_x, obs)
            q = q_fn(agent_state.w_q, x)
            pi = (q == q.max(axis=-1, keepdims=True)).astype(jnp.float32)
            pi /= jnp.sum(pi, axis=-1, keepdims=True)
            mu = (1 - epsilon) * pi + epsilon / q.shape[-1]
            return mu

        mu_all_states = jax.vmap(mu_fn)(jnp.asarray(all_states))
        return mu_all_states

    def get_pi(all_states, agent_state):
        def pi_fn(obs):
            x = x_fn(agent_state.w_x, obs)
            q = q_fn(agent_state.w_q, x)
            pi = (q == q.max(axis=-1, keepdims=True)).astype(jnp.float32)
            pi /= jnp.sum(pi, axis=-1, keepdims=True)
            return pi

        pi_all_states = jax.vmap(pi_fn)(jnp.asarray(all_states))
        return pi_all_states

    # Create optimiser (used to transform gradients into updates)
    init_opt, x_adamify = create_optimiser(**q_optimiser_kwargs)
    x_opt_state = init_opt(w_x)
    init_opt, q_adamify = create_optimiser(**q_optimiser_kwargs)
    q_opt_state = init_opt(w_q)

    init_opt, f_adamify = create_optimiser(**f_optimiser_kwargs)
    f_opt_state = init_opt(w_f)

    # Create expected trace optimiser (used to transform gradients into updates)
    init_opt, z_adamify = create_optimiser(**z_optimiser_kwargs)
    z_opt_state = init_opt(w_z)

    (option_states, option_q_fns,
     option_interest_fns, option_reward_fns,
     option_discount_fns, option_f_fns, option_df_fns,
     option_dq_fns, option_z_fns,
     option_dzloss_fns, option_q_adamifys,
     option_z_adamifys, option_f_adamifys) = init_options(rng_key=rng_key,
                                                          action_spec=action_spec,
                                                          use_q_for_reward=use_q_for_reward,
                                                          x_fn=x_fn, w_x=w_x, dummy_x=dummy_x,
                                                          dummy_observation=dummy_observation,
                                                          options_q_star=options_q_star,
                                                          option_interests=option_interests,
                                                          option_reward=option_reward,
                                                          option_q_optimiser_kwargs=option_q_optimiser_kwargs,
                                                          option_z_optimiser_kwargs=option_z_optimiser_kwargs,
                                                          option_f_optimiser_kwargs=option_f_optimiser_kwargs)

    def get_options_q(all_states, agent_state, option_interests):
        option_states = agent_state.option_states
        options_q_all_states = []
        for option, option_interest in enumerate(option_interests):
            q_all_states = jax.vmap(lambda obs: option_q_fns[option](option_states[option].w_q,
                                                                     x_fn(agent_state.w_x, obs)))(
                jnp.asarray(all_states))
            options_q_all_states.append(q_all_states)
        return options_q_all_states

    def get_options_f(all_states, agent_state, option_interests):
        option_states = agent_state.option_states
        options_f_all_states = []
        for option, option_interest in enumerate(option_interests):
            f_all_states = jax.vmap(lambda obs: option_f_fns[option](option_states[option].w_f,
                                                                     x_fn(agent_state.w_x, obs)))(
                jnp.asarray(all_states))
            options_f_all_states.append(f_all_states)
        return options_f_all_states

    def get_options_pi(all_states, agent_state, option_interests):
        def pi_fn(obs, option_state, option_q_fn):
            x = x_fn(agent_state.w_x, obs)
            q = option_q_fn(option_state.w_q, x)
            pi = (q == q.max(axis=-1, keepdims=True)).astype(jnp.float32)
            pi /= jnp.sum(pi, axis=-1, keepdims=True)
            return pi

        options_pi_all_states = []
        option_states = agent_state.option_states
        for option, option_interest in enumerate(option_interests):
            pi_all_states = jax.vmap(functools.partial(pi_fn,
                                                       option_state=option_states[option],
                                                       option_q_fn=option_q_fns[option]))(
                jnp.asarray(all_states))
            options_pi_all_states.append(pi_all_states)
        return options_pi_all_states

    # Create initial agent state
    agent_state = AgentState(
        rng_key=agent_key,
        w_q=w_q,
        w_z=w_z,
        w_x=w_x,
        w_f=w_f,
        e_q=e_q,
        e_z=e_z,
        e_x=e_x,
        e_f=1.,
        f=1.,
        option=0,
        option_reward=0.,
        behaviour_action_prob=1.,
        trace_decay=0.,
        followon_decay=0.,
        x_opt_state=x_opt_state,
        q_opt_state=q_opt_state,
        z_opt_state=z_opt_state,
        f_opt_state=f_opt_state,
        option_states=option_states,
        obs=None,
        action=None)

    # Create the step function, pre-inputting fixed things (e.g., functions,
    # hyper-parameters)
    step_fn = functools.partial(
        step,
        x_scale_fn=x_adamify,
        use_only_primitive_actions=use_only_primitive_actions,
        x_fn=x_fn,
        q_scale_fn=q_adamify,
        option_q_scale_fns=option_q_adamifys,
        q_fn=q_fn,
        option_q_fns=option_q_fns,
        dq_fn=dq_fn,
        option_dq_fns=option_dq_fns,
        z_scale_fn=z_adamify,
        option_z_scale_fns=option_z_adamifys,
        z_fn=z_fn,
        option_z_fns=option_z_fns,
        dzloss_fn=dzloss_fn,
        option_dzloss_fns=option_dzloss_fns,
        f_scale_fn=f_adamify,
        option_f_scale_fns=option_f_adamifys,
        f_fn=f_fn,
        option_f_fns=option_f_fns,
        df_fn=df_fn,
        option_df_fns=option_df_fns,
        rho_fn=rho_fn,
        option_rho_fn=option_rho_fn,
        policy=policy,
        option_policies=option_policies,
        option_policy=option_policy,
        interest_fn=interest_fn,
        option_interest_fns=option_interest_fns,
        option_discount_fns=option_discount_fns,
        option_reward_fns=option_reward_fns,
        discount=discount,
        learn_trace_by_magic=learn_trace_by_magic,
        intra_option_discount=intra_option_discount,
        use_prior_corr=use_prior_corr,
        option_use_prior_corr=option_use_prior_corr,
        trace_parameter=trace_parameter,
        eta_q=eta_q,
        eta_z=eta_z,
        option_eta_q=option_eta_q,
        option_eta_z=option_eta_z,
        eta_f=eta_f,
        eta_x_f=eta_x_f,
        option_eta_f=option_eta_f,
        option_eta_x_f=option_eta_x_f,
        beta=beta,
        option_beta=option_beta,
        epsilon=epsilon,
        use_magic_rule=use_magic_rule,
        use_discount_magic_rule=use_discount_magic_rule,
        use_true_options=use_true_options
    )

    # Create the reset function, for use at the beginning of episodes.
    first_step_fn = functools.partial(first_step, policy=policy,
                                      option_policies=option_policies,
                                      interest_fn=interest_fn,
                                      option_interest_fns=option_interest_fns,
                                      option_q_fns=option_q_fns)

    #   agent = Agent(step=step_fn, first_step=first_step_fn,
    #                 get_v=get_v, get_mu=get_mu,
    #                 get_pi=get_pi, get_f=get_f)
    # agent njax.jit(get_f))
    agent = Agent(step=jax.jit(step_fn),
                  first_step=jax.jit(first_step_fn),
                  get_q=jax.jit(get_q),
                  get_options_q=jax.jit(get_options_q),
                  get_options_f=jax.jit(get_options_f),
                  get_options_pi=jax.jit(get_options_pi),
                  get_pi=jax.jit(get_pi),
                  get_mu=jax.jit(get_mu),
                  get_f=jax.jit(get_f))
    # agent = Agent(step=step_fn,
    #               first_step=first_step_fn,
    #               get_q=get_q,
    #               get_options_q=get_options_q,
    #               get_options_f=get_options_f,
    #               get_options_pi=get_options_pi,
    #               get_mu=get_mu,
    #               get_pi=get_pi,
    #               get_f=get_f)
    return agent, agent_state


def init_options(rng_key, action_spec,
                 x_fn, w_x, option_interests,
                 dummy_observation, dummy_x,
                 option_q_optimiser_kwargs,
                 option_z_optimiser_kwargs,
                 option_f_optimiser_kwargs,
                 options_q_star,
                 use_q_for_reward, option_reward):
    num_options = len(option_interests)
    option_q_fns = []
    option_interest_fns = []
    option_reward_fns = []
    option_discount_fns = []
    option_dq_fns = []
    option_z_fns = []
    option_f_fns = []
    option_df_fns = []
    option_dzloss_fns = []
    option_q_adamifys = []
    option_z_adamifys = []
    option_f_adamifys = []
    option_states = []
    net_key_options = jax.random.split(rng_key, num_options)
    for option in range(num_options):
        option_rng_key = net_key_options[option]
        option_interest = option_interests[option]
        option_q_star = options_q_star[option]
        (option_state, option_q_fn,
         option_interest_fn, option_reward_fn, option_discount_fn,
         option_dq_fn, option_z_fn, option_f_fn, option_df_fn,
         option_dzloss_fn, option_q_adamify,
         option_z_adamify, option_f_adamify) = init_option(option_rng_key=option_rng_key,
                                                           dummy_x=dummy_x,
                                                           action_spec=action_spec,
                                                           x_fn=x_fn, w_x=w_x, reward=option_reward,
                                                           use_q_for_reward=use_q_for_reward,
                                                           dummy_observation=dummy_observation,
                                                           option_q_star=option_q_star,
                                                           option_interest=option_interest,
                                                           option_q_optimiser_kwargs=option_q_optimiser_kwargs,
                                                           option_z_optimiser_kwargs=option_z_optimiser_kwargs,
                                                           option_f_optimiser_kwargs=option_f_optimiser_kwargs)
        option_states.append(option_state)
        option_q_fns.append(option_q_fn)
        option_interest_fns.append(option_interest_fn)
        option_reward_fns.append(option_reward_fn)
        option_discount_fns.append(option_discount_fn)
        option_dq_fns.append(option_dq_fn)
        option_df_fns.append(option_df_fn)
        option_z_fns.append(option_z_fn)
        option_f_fns.append(option_f_fn)
        option_dzloss_fns.append(option_dzloss_fn)
        option_q_adamifys.append(option_q_adamify)
        option_z_adamifys.append(option_z_adamify)
        option_f_adamifys.append(option_f_adamify)

    return (option_states, option_q_fns,
            option_interest_fns, option_reward_fns,
            option_discount_fns, option_f_fns, option_df_fns,
            option_dq_fns, option_z_fns,
            option_dzloss_fns, option_q_adamifys,
            option_z_adamifys, option_f_adamifys)


def init_option(option_rng_key,
                dummy_x, action_spec, x_fn, w_x, dummy_observation,
                option_interest,
                option_q_optimiser_kwargs,
                option_z_optimiser_kwargs,
                option_f_optimiser_kwargs,
                use_q_for_reward,
                option_q_star,
                reward):
    (net_key_q_option, net_key_z_option,
     net_key_f_option) = jax.random.split(option_rng_key, 3)

    def option_interest_fn(s):
        return jnp.asarray(option_interest)[s]

    def option_discount_fn(s):
        return jnp.asarray(option_interest)[s]

    def option_reward_fn(s, action, next_s, option_v_next):
        interest = jnp.asarray(option_interest)[s]
        next_interest = jnp.asarray(option_interest)[next_s]
        option_reward = reward * (interest - next_interest)
        if use_q_for_reward:
            option_reward *= option_v_next #TODO add discount
        return option_reward

    option_q_fn, option_w_q = create_q_head(net_key_q_option, dummy_x, action_spec)

    option_dq_fn = jax.grad(
        lambda w_q, w_x, o, a: option_q_fn(w_q, x_fn(w_x, o))[a], argnums=(0, 1))

    option_f_fn_, option_w_f = create_v_head(net_key_f_option, dummy_x)
    option_f_fn = lambda w, x: jnp.squeeze(option_f_fn_(w, x), -1)

    # Create function for the gradients of the v function
    option_df_fn = jax.grad(lambda w_f, w_x, o: option_f_fn(w_f, x_fn(w_x, o)))

    (option_dqdw, _) = option_dq_fn(option_w_q, w_x, dummy_observation, 0)
    option_z_fn, option_w_z = create_z_head(net_key_z_option, dummy_x, option_dqdw)

    # Create loss function for the expected trace
    def option_zloss(w_z, x, e):
        total_size = sum([e_i.size for e_i in jax.tree_leaves(e)])
        z = option_z_fn(w_z, x)
        sum_of_squares = tmap(lambda z_i, e_i: jnp.sum((z_i - e_i) ** 2), z, e)
        return 0.5 * sum(jax.tree_leaves(sum_of_squares)) / total_size

    # Gradient of the expected trace loss (also outputs the loss, for logging)
    option_dzloss_fn = jax.value_and_grad(option_zloss)

    # Initialise eligibility traces
    option_e_q, option_e_z = tmap(jnp.zeros_like, (option_w_q, option_w_q))

    option_init_opt, option_q_adamify = create_optimiser(**option_q_optimiser_kwargs)
    option_q_opt_state = option_init_opt(option_w_q)

    option_init_opt, option_z_adamify = create_optimiser(**option_z_optimiser_kwargs)
    option_z_opt_state = option_init_opt(option_w_z)

    option_init_opt, option_f_adamify = create_optimiser(**option_f_optimiser_kwargs)
    option_f_opt_state = option_init_opt(option_w_f)

    option_state = OptionState(
        # rng_key=option_rng_key,
        w_q=option_w_q,
        w_z=option_w_z,
        w_f=option_w_f,
        e_q=option_e_q,
        e_z=option_e_z,
        e_f=1.,
        f=1.,
        option_q_star=option_q_star,
        trace_decay=0.,
        followon_decay=0.,
        interest=option_interest,
        q_opt_state=option_q_opt_state,
        z_opt_state=option_z_opt_state,
        f_opt_state=option_f_opt_state,
    )
    return (option_state, option_q_fn,
            option_interest_fn,
            option_reward_fn,
            option_discount_fn,
            option_dq_fn, option_z_fn, option_f_fn,
            option_df_fn,
            option_dzloss_fn, option_q_adamify,
            option_z_adamify, option_f_adamify)