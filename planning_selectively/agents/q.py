# @title Q(lambda) agent
from imports import *
from utils import *

class Agent(NamedTuple):
    step: Callable[..., Any]
    first_step: Callable[..., Any]
    get_q: Callable[..., Any]
    get_options_v: Callable[..., Any]
    get_options_f: Callable[..., Any]
    get_mu: Callable[..., Any]
    get_pi: Callable[..., Any]
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
    counter: Any
    behaviour_action_prob: Any
    followon_decay: Any
    option_states: Any


class OptionState(NamedTuple):
    """Agent state."""
    interest: Any
    w_v: Any
    w_z: Any
    e_v: Any
    e_z: Any
    v_opt_state: Any
    z_opt_state: Any
    f_opt_state: Any
    f: Any
    w_f: Any
    e_f: Any
    # option_true_q_pi: Any
    option_pi: Any
    followon_decay: Any
    trace_decay: float


def get_config(agent_name, hyper,
               option_interests=None,
               option_pis=None,
               # option_true_q_pis=None,
               interest=None):
    """Hyper-parameters for this agent."""
    config = {}
    config["init"] = init
    config["cfg"] = {}
    config["cfg"]["c"] = hyper["c"]

    config["cfg"]["epsilon"] = hyper["epsilon"]
    config["cfg"]["option_epsilon"] = hyper["option_epsilon"]
    config["cfg"]["discount"] = hyper["discount"]
    config["cfg"]["trace_parameter"] = hyper["lambda"]
    config["cfg"]["action_repeats"] = hyper["action_repeats"]

    config["cfg"]["option_interests"] = option_interests
    config["cfg"]["option_pis"] = option_pis
    config["cfg"]["interest"] = interest
    config["cfg"]["q_optimiser_kwargs"] = {}
    config["cfg"]["q_optimiser_kwargs"]["learning_rate"] = hyper["alpha_q"]
    config["cfg"]["q_optimiser_kwargs"]["b1"] = 0.9
    config["cfg"]["q_optimiser_kwargs"]["b2"] = 0.999
    config["cfg"]["q_optimiser_kwargs"]["eps"] = 1e-4
    config["cfg"]["q_optimiser_kwargs"]["momentum"] = 0.9
    config["cfg"]["q_optimiser_kwargs"]["d"] = hyper["d"]

    config["cfg"]["option_v_optimiser_kwargs"] = {}
    config["cfg"]["option_v_optimiser_kwargs"]["learning_rate"] = hyper["option_alpha_v"]
    config["cfg"]["option_v_optimiser_kwargs"]["b1"] = 0.9
    config["cfg"]["option_v_optimiser_kwargs"]["b2"] = 0.999
    config["cfg"]["option_v_optimiser_kwargs"]["eps"] = 1e-4
    config["cfg"]["option_v_optimiser_kwargs"]["momentum"] = 0.9
    config["cfg"]["option_v_optimiser_kwargs"]["d"] = hyper["d"]

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
    config["cfg"]["z_optimiser_kwargs"]["d"] = hyper["d"]

    config["cfg"]["option_z_optimiser_kwargs"] = {}
    config["cfg"]["option_z_optimiser_kwargs"]["learning_rate"] = hyper["option_alpha_z"]
    config["cfg"]["option_z_optimiser_kwargs"]["b1"] = 0.9
    config["cfg"]["option_z_optimiser_kwargs"]["b2"] = 0.999
    config["cfg"]["option_z_optimiser_kwargs"]["eps"] = 1e-4
    config["cfg"]["option_z_optimiser_kwargs"]["momentum"] = 0.9
    config["cfg"]["option_z_optimiser_kwargs"]["d"] = hyper["d"]

    # Default is the peng and williams q lambda
    # config["cfg"]["option_true_q_pis"] = option_true_q_pis
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
    config["cfg"]["option_eta_v"] = hyper["option_eta_v"]
    config["cfg"]["option_eta_z"] = hyper["option_eta_z"]
    config["cfg"]["beta"] = hyper["beta"]

    return config


# @title Default title text
def first_step(
        timestep: dm_env.TimeStep,
        s: Any,
        agent_state: AgentState,
        interest_fn: Callable[..., Any],
        policy: Callable[..., Any],
) -> Tuple[Any, AgentState]:
    rng_key, policy_key = jax.random.split(agent_state.rng_key)

    action, action_prob = policy(policy_key, timestep.observation,
                                 s, agent_state)
    agent_state = agent_state._replace(
        rng_key=rng_key, obs=timestep.observation, counter=1,
        action=action, behaviour_action_prob=action_prob)
    return action, agent_state


def step(
        interest_fn: Callable[..., Any],
        # stateful things
        timestep: dm_env.TimeStep,
        s: Any,
        next_s: Any,
        agent_state: AgentState,
        # functions
        x_scale_fn: Callable[..., Any],
        q_scale_fn: Callable[..., Any],
        z_scale_fn: Callable[..., Any],
        x_fn: Callable[..., Any],
        q_fn: Callable[..., Any],
        dq_fn: Callable[..., Any],
        z_fn: Callable[..., Any],
        dzloss_fn: Callable[..., Any],
        policy: Callable[..., Any],
        f_scale_fn: Callable[..., Any],
        f_fn: Callable[..., Any],
        df_fn: Callable[..., Any],
        rho_fn: Callable[..., Any],
        # hyper-parameters
        discount: float,
        trace_parameter: float,
        eta_q: float,
        eta_z: float,
        eta_f: float,
        eta_x_f: float,
        beta: float,
        use_prior_corr: bool,
        epsilon: float,
        option_v_fns: Any,
        option_interest_fns: Any,
        **option_kwargs,
) -> Tuple[Any, AgentState, Mapping[str, Any]]:  # hyper-parameters
    """One step of the agent."""

    # Map action structure to an int
    action = jax.tree_leaves(agent_state.action)[0]
    reward = timestep.reward
    discount = timestep.discount * discount
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
    f_opt_state = agent_state.f_opt_state
    # rng_key, policy_key = jax.random.split(agent_state.rng_key)

    ### Compute values
    # Compute the features x(s)
    x = x_fn(w_x, obs)
    next_x = x_fn(w_x, next_obs)
    # Compute value v(s)
    q = q_fn(w_q, x)
    qa = q[action]

    # Compute gradient dv(s)/dw, d log pi(s, a)/dw
    dq_head, dq_torso = dq_fn(w_q, w_x, obs, action)

    # Compute v(s) at the next time step
    q_next = q_fn(w_q, next_x)
    v_next = jnp.max(q_next)

    interest = interest_fn(s)
    rho_post, rho_prior = rho_fn(agent_state, obs, action)
    # Pick new action
    # We do this before updating the weights to avoid forwarding twice
    rng_key, policy_key = jax.random.split(agent_state.rng_key)

    a_next, next_action_prob = policy(policy_key, next_obs, next_s,
                                      agent_state)

    ### Expected trace
    # Compute the expected trace: z(s) ~= E[ γ λ e_{t-1} | S_t=s ]
    z = z_fn(w_z, x)
    # Decay all traces
    e_q, e_z, e_x = tmap(lambda e_i: trace_decay * e_i, (e_q, e_z, e_x))

    # Compute update for the expected trace
    zloss, dw_z = dzloss_fn(w_z, x, e_z)

    # Compute mixture traces η γ λ e_{t-1} + (1 - η) z(S_t)
    # Note: z(S_t) ~= E[ γ λ e_{t-1} | S_t ]
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
    else:
        m = f = interest

    ### Add gradient to traces
    tree_add_weighted = lambda x, y, z: tmap(lambda x, y: x + y * z, x, y)
    e_q = tree_add_weighted(e_q, dq_head, m)
    e_z = tree_add_weighted(e_z, dq_head, m)
    e_x = tree_add_weighted(e_x, dq_torso, m)

    ### Update values
    # Compute term that composes the multi-step λ return
    # discount = discount * interest + (1 - interest)
    r_plus_next_v = reward + (1 - trace_parameter) * discount * v_next
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
    trace_decay = rho_post * discount * trace_parameter
    trace_decay *= 1 - jnp.float32(episode_end)
    followon_decay = rho_prior * jnp.minimum(discount, beta) * (1 - jnp.float32(episode_end))
    option_states, option_log_dicts = update_options(
        s=s, next_s=next_s,
        obs=obs, next_obs=next_obs,
        reward=reward, action=action,
        x=x, next_x=next_x,
        w_x=w_x, option_v_fns=option_v_fns,
        episode_end=episode_end,
        trace_parameter=trace_parameter,
        discount=discount,
        agent_state=agent_state,
        option_interest_fns=option_interest_fns,
        **option_kwargs)

    # Add some things to log
    log_dict = dict(q=qa, f=f, e_f=e_f, x_f=x_f,
                    zloss=zloss,
                    rho_prior=rho_prior,
                    rho_post=rho_post,
                    discount=discount, f_td_or_mc_error=f_error,
                    td_error=reward + discount * v_next - qa)

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
        counter=agent_state.counter + 1,
        f=f,
        followon_decay=followon_decay,
        trace_decay=trace_decay,
        q_opt_state=q_opt_state,
        f_opt_state=f_opt_state,
        x_opt_state=x_opt_state,
        z_opt_state=z_opt_state,
        obs=next_obs,
        action=a_next,
        behaviour_action_prob=next_action_prob,
        option_states=option_states
    )

    return a_next, agent_state, log_dict, option_log_dicts


def update_options(obs, next_obs,
                   s, next_s,
                   x, next_x, reward,
                   episode_end,
                   trace_parameter,
                   w_x, discount,
                   option_eta_z, option_eta_v,
                   option_eta_f, option_eta_x_f,
                   agent_state,
                   option_v_fns,
                   option_dv_fns,
                   option_z_fns,
                   option_f_fns,
                   option_df_fns,
                   option_dzloss_fns,
                   option_v_scale_fns,
                   option_z_scale_fns,
                   option_f_scale_fns,
                   option_interest_fns,
                   option_use_prior_corr,
                   option_rho_fn,
                   action,
                   option_policy):
    new_option_states = []
    option_log_dicts = []

    for option, (option_state, option_v_fn, option_f_fn,
                 option_dv_fn, option_z_fn, option_df_fn,
                 option_dzloss_fn, option_v_scale_fn,
                 option_z_scale_fn, option_f_scale_fn,
                 option_interest_fn) in enumerate(zip(agent_state.option_states,
                                                      option_v_fns, option_f_fns, option_dv_fns, option_z_fns,
                                                      option_df_fns, option_dzloss_fns, option_v_scale_fns,
                                                      option_z_scale_fns, option_f_scale_fns,
                                                      option_interest_fns)):
        option_state, option_log_dict = update_option(s=s, next_s=next_s,
                                                      obs=obs, next_obs=next_obs,
                                                      x=x, next_x=next_x, discount=discount,
                                                      w_x=w_x,
                                                      trace_parameter=trace_parameter,
                                                      reward=reward,
                                                      action=action,
                                                      episode_end=episode_end,
                                                      option_eta_v=option_eta_v,
                                                      option_eta_z=option_eta_z,
                                                      option_eta_f=option_eta_f,
                                                      option_use_prior_corr=option_use_prior_corr,
                                                      option_eta_x_f=option_eta_x_f,
                                                      option_interest_fn=option_interest_fn,
                                                      agent_state=agent_state,
                                                      option_state=option_state,
                                                      option_v_fn=option_v_fn,
                                                      option_f_fn=option_f_fn,
                                                      option_dv_fn=option_dv_fn,
                                                      option_df_fn=option_df_fn,
                                                      option_z_fn=option_z_fn,
                                                      option_dzloss_fn=option_dzloss_fn,
                                                      option_v_scale_fn=option_v_scale_fn,
                                                      option_z_scale_fn=option_z_scale_fn,
                                                      option_f_scale_fn=option_f_scale_fn,
                                                      option_rho_fn=option_rho_fn,
                                                      option_policy=option_policy)
        new_option_states.append(option_state)
        option_log_dicts.append(option_log_dict)
    return new_option_states, option_log_dicts


def update_option(obs, s, next_s, x, w_x, next_obs, reward, next_x,
                  trace_parameter, discount, episode_end, action,
                  option_eta_v, option_eta_z, option_use_prior_corr,
                  agent_state, option_eta_f, option_eta_x_f,
                  option_state, option_v_fn, option_dv_fn,
                  option_z_fn, option_dzloss_fn,
                  option_interest_fn, option_f_fn, option_df_fn,
                  option_v_scale_fn, option_z_scale_fn, option_f_scale_fn,
                  option_rho_fn, option_policy):
    option_w_v = option_state.w_v
    option_w_z = option_state.w_z
    option_w_f = option_state.w_f
    option_trace_decay = option_state.trace_decay
    option_followon_decay = option_state.followon_decay
    # option_rng_key = option_state.rng_key
    option_v = option_v_fn(option_w_v, x)
    option_e_v, option_e_z = option_state.e_v, option_state.e_z
    option_e_f, option_f = option_state.e_f, option_state.f
    option_f_opt_state = option_state.f_opt_state
    # Compute gradient dv(s)/dw, d log pi(s, a)/dw
    option_dv_head = option_dv_fn(option_w_v, w_x, obs)

    # Compute v(s) at the next time step
    option_v_next = option_v_fn(option_w_v, next_x)
    option_rho_post, option_rho_prior = option_rho_fn(agent_state,
                                                      agent_state.behaviour_action_prob,
                                                      option_state,
                                                      obs, s, action,
                                                      )

    ### Expected trace
    # Compute the expected trace: z(s) ~= E[ γ λ e_{t-1} | S_t=s ]
    option_z = option_z_fn(option_w_z, x)
    # Decay all traces
    option_trace_decay *= option_rho_post
    option_e_v, option_e_z = tmap(lambda e_i: option_trace_decay * e_i,
                                  (option_e_v, option_e_z))

    # Compute update for the expected trace
    option_zloss, option_dw_z = option_dzloss_fn(option_w_z, x, option_e_z)

    # Compute mixture traces η γ λ e_{t-1} + (1 - η) z(S_t)
    # Note: z(S_t) ~= E[ γ λ e_{t-1} | S_t ]
    option_e_v = tmap(lambda e_i, z_i: option_eta_v * e_i + (1 - option_eta_v) * z_i, option_e_v, option_z)
    option_e_z = tmap(lambda e_i, z_i: option_eta_z * e_i + (1 - option_eta_z) * z_i, option_e_z, option_z)

    interest = option_interest_fn(s)
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
    else:
        option_m = option_f = interest

    # Compute the actual emphasis from the follow-on
    option_m *= option_rho_post

    ### Add gradient to traces
    tree_add_weighted = lambda x, y, z: tmap(lambda x, y: x + y * z, x, y)
    option_e_v = tree_add_weighted(option_e_v, option_dv_head, option_m)
    option_e_z = tree_add_weighted(option_e_z, option_dv_head, option_m)

    ### Update values
    # Compute term that composes the multi-step λ return
    option_r_plus_next_v = reward + (1 - trace_parameter) * discount * option_v_next
    option_compute_dw = lambda e_i, option_dv_i: option_r_plus_next_v * e_i - option_v * option_dv_i * option_m
    # Compute update for the values
    option_dw_v = tmap(option_compute_dw, option_e_v, option_dv_head)
    option_dw_v, option_v_opt_state = option_v_scale_fn(option_dw_v, state=option_state.v_opt_state)
    option_dw_z, option_z_opt_state = option_z_scale_fn(option_dw_z, state=option_state.z_opt_state)

    # Update weights
    option_w_v = tmap(lambda x, y: x + y, option_w_v, option_dw_v)
    option_w_z = tree_sub(option_w_z, option_dw_z)  # subtract, for gradient descent

    # Compute trace decay to apply at the next time step
    option_trace_decay = discount * trace_parameter
    option_trace_decay *= 1 - jnp.float32(episode_end)
    option_followon_decay = option_rho_prior * discount * (1 - jnp.float32(episode_end))

    option_state = OptionState(
        interest=option_state.interest,
        w_v=option_w_v,
        w_f=option_w_f,
        w_z=option_w_z,
        e_v=option_e_v,
        e_z=option_e_z,
        e_f=option_e_f,
        f=option_f,
        # option_true_q_pi=option_state.option_true_q_pi,
        option_pi=option_state.option_pi,
        trace_decay=option_trace_decay,
        followon_decay=option_followon_decay,
        v_opt_state=option_v_opt_state,
        z_opt_state=option_z_opt_state,
        f_opt_state=option_f_opt_state,
    )

    option_log_dict = dict(v=option_v,
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
                           td_error=reward + discount * option_v_next - option_v)

    return option_state, option_log_dict


def init(
        rng_key: jnp.ndarray,
        network_spec: str,
        action_spec: Any,
        # action_and_option_spec: Any,
        option_spec: Any,
        observation_spec: Any,
        discount: float,
        trace_parameter: float,
        q_optimiser_kwargs: Mapping[str, Any],
        option_v_optimiser_kwargs: Mapping[str, Any],
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
        option_eta_v: float,
        option_eta_z: float,
        beta: float,
        epsilon: float,
        option_epsilon: float,
        option_interests: Any,
        interest: Any,
        c: Any,
        # option_true_q_pis: Any,
        option_pis: Any,
        action_repeats: Any,
) -> Tuple[Agent, AgentState]:
    """Initialise TD(λ) agent."""
    # Check the action spec
    action_structure = check_simple_action_spec(action_spec)
    from utils.auxilliaries import _get_num_actions
    # num_actions = _get_num_actions(action_spec)

    # option_structure = check_simple_action_spec(option_spec)
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

    def option_rho_fn(agent_state, behaviour_prob, option_state, obs, s, action):
        # x = x_fn(agent_state.w_x, obs)
        pi_option = option_state.option_pi[s]
        # greedy = (pi == pi.max(axis=-1, keepdims=True)).astype(jnp.float32)
        # greedy /= jnp.sum(pi, axis=-1, keepdims=True)
        pi_option = (1 - option_epsilon) * pi_option + option_epsilon / pi_option.shape[-1]
        # q_option = option_state.option_pi[s]
        # pi_option = (q_option == q_option.max(axis=-1, keepdims=True)).astype(jnp.float32)
        # pi_option /= jnp.sum(pi_option, axis=-1, keepdims=True)
        # pi_option = (1 - option_epsilon)*pi_option + option_epsilon/q_option.shape[-1]
        rho_prior = pi_option[action] / behaviour_prob
        rho_prior = jnp.minimum(rho_prior, c)
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
    def zloss(w_z, x, e):
        total_size = sum([e_i.size for e_i in jax.tree_leaves(e)])
        z = z_fn(w_z, x)
        sum_of_squares = tmap(lambda z_i, e_i: jnp.sum((z_i - e_i) ** 2), z, e)
        return 0.5 * sum(jax.tree_leaves(sum_of_squares)) / total_size

    # Gradient of the expected trace loss (also outputs the loss, for logging)
    dzloss_fn = jax.value_and_grad(zloss)

    # Create softmax policy
    def policy(rng_key, obs, s, agent_state):
        x = x_fn(agent_state.w_x, obs)
        q = q_fn(agent_state.w_q, x)
        greedy = (q == q.max(axis=-1, keepdims=True)).astype(jnp.float32)
        greedy /= jnp.sum(greedy, axis=-1, keepdims=True)
        pi = (1 - epsilon) * greedy + epsilon / q.shape[-1]
        cpi = jnp.cumsum(pi, axis=-1)
        rnd = jax.random.uniform(rng_key)
        next_a = jnp.argmax(cpi > rnd, axis=-1)
        prev_a = agent_state.action
        prev_a_prob = agent_state.behaviour_action_prob
        # import pdb; pdb.set_Trace)_
        next_a = jnp.where(agent_state.counter % action_repeats == 0,
                           next_a, prev_a)
        next_a_prob = jnp.where(agent_state.counter % action_repeats == 0,
                           pi[next_a], prev_a_prob)
        next_a = jax.tree_unflatten(action_structure, [next_a])
        return next_a, next_a_prob

        # eps_greedy_policy = functools.partial(eps_greedy, epsilon=epsilon)
        # next_a, next_a_prob = eps_greedy_policy(rng_key=rng_key, q=q)
        # next_a = jax.tree_unflatten(action_structure, [next_a])
        # return next_a, next_a_prob

    def option_policy(rng_key, obs, s, agent_state, option_state):
        # x = x_fn(agent_state.w_x, obs)
        pi = option_state.option_pi[s]
        # greedy = (pi == pi.max(axis=-1, keepdims=True)).astype(jnp.float32)
        # greedy /= jnp.sum(pi, axis=-1, keepdims=True)
        pi = (1 - option_epsilon) * pi + option_epsilon / pi.shape[-1]
        cpi = jnp.cumsum(pi, axis=-1)
        rnd = jax.random.uniform(rng_key)
        a = jnp.argmax(cpi > rnd, axis=-1)
        behaviour_prob = pi[a]
        a = jax.tree_unflatten(action_structure, [a])
        return a, behaviour_prob

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

    (option_states, option_v_fns,
     option_interest_fns, option_f_fns, option_df_fns,
     option_dv_fns, option_z_fns,
     option_dzloss_fns, option_v_adamifys,
     option_z_adamifys, option_f_adamifys) = init_options(rng_key=rng_key,
                                                          action_spec=action_spec,
                                                          x_fn=x_fn, w_x=w_x, dummy_x=dummy_x,
                                                          dummy_observation=dummy_observation,
                                                          # option_true_q_pis=option_true_q_pis,
                                                          option_pis=option_pis,
                                                          option_interests=option_interests,
                                                          option_v_optimiser_kwargs=option_v_optimiser_kwargs,
                                                          option_z_optimiser_kwargs=option_z_optimiser_kwargs,
                                                          option_f_optimiser_kwargs=option_f_optimiser_kwargs)

    def get_options_v(all_states, agent_state, option_interests):
        option_states = agent_state.option_states
        options_v_all_states = []
        for option, option_interest in enumerate(option_interests):
            v_all_states = jax.vmap(lambda obs: option_v_fns[option](option_states[option].w_v,
                                                                     x_fn(agent_state.w_x, obs)))(
                jnp.asarray(all_states))
            options_v_all_states.append(v_all_states)
        return options_v_all_states

    def get_options_f(all_states, agent_state, option_interests):
        option_states = agent_state.option_states
        options_f_all_states = []
        for option, option_interest in enumerate(option_interests):
            f_all_states = jax.vmap(lambda obs: option_f_fns[option](option_states[option].w_f,
                                                                     x_fn(agent_state.w_x, obs)))(
                jnp.asarray(all_states))
            options_f_all_states.append(f_all_states)
        return options_f_all_states

    # def get_options_pi(all_states, agent_state, option_interests):
    # def pi_fn(obs, option_state, option_q_fn):
    #     x = x_fn(agent_state.w_x, obs)
    #     q = option_q_fn(option_state.w_q, x)
    #     pi = (q == q.max(axis=-1, keepdims=True)).astype(jnp.float32)
    #     pi /= jnp.sum(pi, axis=-1, keepdims=True)
    #     return pi
    # options_pi_all_states = []
    # option_states = agent_state.option_states
    # for option, option_interest in enumerate(option_interests):
    #     pi_all_states = jax.vmap(functools.partial(pi_fn,
    #                      option_state=option_states[option],
    #                      option_q_fn=option_v_fns[option]))(
    #                          jnp.asarray(all_states))
    #     options_pi_all_states.append(pi_all_states)
    # return options_pi_all_states

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
        action=0,
        counter=0.,
        behaviour_action_prob=1.,
        trace_decay=0.,
        followon_decay=0.,
        x_opt_state=x_opt_state,
        q_opt_state=q_opt_state,
        z_opt_state=z_opt_state,
        f_opt_state=f_opt_state,
        option_states=option_states,
        obs=None)

    # Create the step function, pre-inputting fixed things (e.g., functions,
    # hyper-parameters)
    step_fn = functools.partial(
        step,
        x_scale_fn=x_adamify,
        x_fn=x_fn,
        q_scale_fn=q_adamify,
        option_v_scale_fns=option_v_adamifys,
        q_fn=q_fn,
        option_v_fns=option_v_fns,
        dq_fn=dq_fn,
        option_dv_fns=option_dv_fns,
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
        option_policy=option_policy,
        interest_fn=interest_fn,
        option_interest_fns=option_interest_fns,
        discount=discount,
        use_prior_corr=use_prior_corr,
        option_use_prior_corr=option_use_prior_corr,
        trace_parameter=trace_parameter,
        eta_q=eta_q,
        eta_z=eta_z,
        option_eta_v=option_eta_v,
        option_eta_z=option_eta_z,
        eta_f=eta_f,
        eta_x_f=eta_x_f,
        option_eta_f=option_eta_f,
        option_eta_x_f=option_eta_x_f,
        beta=beta,
        epsilon=epsilon,
    )
    first_step_fn = functools.partial(first_step, policy=policy,
                                      interest_fn=interest_fn)
    # Create the reset function, for use at the beginning of episodes.
    agent = Agent(step=jax.jit(step_fn),
                  first_step=jax.jit(first_step_fn),
                  get_q=jax.jit(get_q),
                  get_options_v=jax.jit(get_options_v),
                  get_options_f=jax.jit(get_options_f),
                  get_pi=jax.jit(get_pi),
                  get_mu=jax.jit(get_mu),
                  get_f=jax.jit(get_f))

    return agent, agent_state


def init_options(rng_key, action_spec,
                 x_fn, w_x, option_interests,
                 option_pis,
                 dummy_observation, dummy_x,
                 option_v_optimiser_kwargs,
                 option_z_optimiser_kwargs,
                 option_f_optimiser_kwargs,
                 # option_true_q_pis,
                 ):
    num_options = len(option_interests)
    option_v_fns = []
    option_interest_fns = []
    option_dv_fns = []
    option_z_fns = []
    option_f_fns = []
    option_df_fns = []
    option_dzloss_fns = []
    option_v_adamifys = []
    option_z_adamifys = []
    option_f_adamifys = []
    option_states = []
    net_key_options = jax.random.split(rng_key, num_options)
    for option in range(num_options):
        option_rng_key = net_key_options[option]
        option_interest = option_interests[option]
        # option_true_q_pi = option_true_q_pis[option]
        option_pi = option_pis[option]
        (option_state, option_v_fn,
         option_interest_fn,
         option_dv_fn, option_z_fn, option_f_fn, option_df_fn,
         option_dzloss_fn, option_v_adamify,
         option_z_adamify, option_f_adamify) = init_option(option_rng_key=option_rng_key,
                                                           dummy_x=dummy_x,
                                                           action_spec=action_spec,
                                                           x_fn=x_fn, w_x=w_x,
                                                           dummy_observation=dummy_observation,
                                                           # option_true_q_pi=option_true_q_pi,
                                                           option_interest=option_interest,
                                                           option_pi=option_pi,
                                                           option_v_optimiser_kwargs=option_v_optimiser_kwargs,
                                                           option_z_optimiser_kwargs=option_z_optimiser_kwargs,
                                                           option_f_optimiser_kwargs=option_f_optimiser_kwargs)
        option_states.append(option_state)
        option_v_fns.append(option_v_fn)
        option_interest_fns.append(option_interest_fn)
        option_dv_fns.append(option_dv_fn)
        option_df_fns.append(option_df_fn)
        option_z_fns.append(option_z_fn)
        option_f_fns.append(option_f_fn)
        option_dzloss_fns.append(option_dzloss_fn)
        option_v_adamifys.append(option_v_adamify)
        option_z_adamifys.append(option_z_adamify)
        option_f_adamifys.append(option_f_adamify)

    return (option_states, option_v_fns,
            option_interest_fns, option_f_fns, option_df_fns,
            option_dv_fns, option_z_fns,
            option_dzloss_fns, option_v_adamifys,
            option_z_adamifys, option_f_adamifys)


def init_option(option_rng_key,
                dummy_x, action_spec, x_fn, w_x, dummy_observation,
                option_interest,
                option_v_optimiser_kwargs,
                option_z_optimiser_kwargs,
                option_f_optimiser_kwargs,
                # option_true_q_pi,
                option_pi):
    (net_key_v_option, net_key_z_option,
     net_key_f_option) = jax.random.split(option_rng_key, 3)

    def option_interest_fn(s):
        return jnp.asarray(option_interest)[s]

    option_v_fn_, option_w_v = create_v_head(net_key_v_option, dummy_x)
    option_v_fn = lambda w, x: jnp.squeeze(option_v_fn_(w, x), -1)
    option_dv_fn = jax.grad(
        lambda w_v, w_x, o: option_v_fn(w_v, x_fn(w_x, o)))

    option_f_fn_, option_w_f = create_v_head(net_key_f_option, dummy_x)
    option_f_fn = lambda w, x: jnp.squeeze(option_f_fn_(w, x), -1)

    # Create function for the gradients of the v function
    option_df_fn = jax.grad(lambda w_f, w_x, o: option_f_fn(w_f, x_fn(w_x, o)))

    option_dvdw = option_dv_fn(option_w_v, w_x, dummy_observation)
    option_z_fn, option_w_z = create_z_head(net_key_z_option, dummy_x, option_dvdw)

    # Create loss function for the expected trace
    def option_zloss(w_z, x, e):
        total_size = sum([e_i.size for e_i in jax.tree_leaves(e)])
        z = option_z_fn(w_z, x)
        sum_of_squares = tmap(lambda z_i, e_i: jnp.sum((z_i - e_i) ** 2), z, e)
        return 0.5 * sum(jax.tree_leaves(sum_of_squares)) / total_size

    # Gradient of the expected trace loss (also outputs the loss, for logging)
    option_dzloss_fn = jax.value_and_grad(option_zloss)

    # Initialise eligibility traces
    option_e_v, option_e_z = tmap(jnp.zeros_like, (option_w_v, option_w_v))

    option_init_opt, option_v_adamify = create_optimiser(**option_v_optimiser_kwargs)
    option_v_opt_state = option_init_opt(option_w_v)

    option_init_opt, option_z_adamify = create_optimiser(**option_z_optimiser_kwargs)
    option_z_opt_state = option_init_opt(option_w_z)

    option_init_opt, option_f_adamify = create_optimiser(**option_f_optimiser_kwargs)
    option_f_opt_state = option_init_opt(option_w_f)

    option_state = OptionState(
        w_v=option_w_v,
        w_z=option_w_z,
        w_f=option_w_f,
        e_v=option_e_v,
        e_z=option_e_z,
        e_f=1.,
        f=1.,
        option_pi=option_pi,
        # option_true_q_pi=option_true_q_pi,
        trace_decay=0.,
        followon_decay=0.,
        interest=option_interest,
        v_opt_state=option_v_opt_state,
        z_opt_state=option_z_opt_state,
        f_opt_state=option_f_opt_state,
    )
    return (option_state, option_v_fn,
            option_interest_fn,
            option_dv_fn, option_z_fn, option_f_fn,
            option_df_fn,
            option_dzloss_fn, option_v_adamify,
            option_z_adamify, option_f_adamify)