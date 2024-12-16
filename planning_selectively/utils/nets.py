#@title nets
from imports import *
from utils import *
from utils.auxilliaries import _get_num_actions
def dummy(observation_spec):
    return jax.tree_map(
        lambda x: jnp.zeros(x.shape, dtype=x.dtype), observation_spec)

def linear_net(observation_spec, with_bias=True, w_init=None):
    check_simple_observation_spec(observation_spec)

    def fun(obs, num_actions):
        if w_init is not None:
            q = hk.Linear(num_actions, with_bias=with_bias, w_init=w_init)
        else:
            q = hk.Linear(num_actions, with_bias=with_bias)
        return q(obs)
    return fun


def mlp_net(observation_spec):
  check_simple_observation_spec(observation_spec)

  def fun(obs, num_actions):
    net = hk.Sequential([
        hk.Linear(20),
        jax.nn.relu,
        # hk.Linear(20),
        # jax.nn.relu,
    ])
    x = net(obs)
    q = hk.Linear(num_actions)
    return q(x)
  return fun

def identity_net(observation_spec):
  check_simple_observation_spec(observation_spec)

  def fun(obs):
    return obs
  return fun

def mlp_torso_net(observation_spec):
  check_simple_observation_spec(observation_spec)

  def fun(obs):
    net = hk.Sequential([
        hk.Linear(20),
        jax.nn.relu,
        # hk.Linear(20),
        # jax.nn.relu,
    ])
    return net(obs)
  return fun

def create_torso(
    rng_key: jnp.ndarray,
    network_spec: str,
    observation_spec: Any,
):
    if network_spec == "linear":
        net = identity_net(observation_spec)
    elif network_spec == "mlp":
        net = mlp_torso_net(observation_spec)
    else:
        print("This net type does not exists")
        exit(0)
    init, apply_fn = hk.without_apply_rng(hk.transform(net))
    dummy_observation = jax.tree_map(lambda x: jnp.zeros(x.shape, dtype=x.dtype),
                                    observation_spec)
    w = init(rng_key, obs=dummy_observation)
    return apply_fn, w

def create_q_head(
    rng_key: jnp.ndarray,
    dummy_x: Any,
    action_spec: Any,
):
    """Pick and initialise a haiku network."""
    num_actions = _get_num_actions(action_spec)

    init, q_fn = hk.without_apply_rng(
        hk.transform(functools.partial(q_head, num_actions=num_actions)))

    w = init(rng_key, x=dummy_x)
    return q_fn, w

def create_v_head(
    rng_key: jnp.ndarray,
    dummy_x: Any,
    with_bias: bool = True,
    w_init: Any = None
):
    """Pick and initialise a haiku network."""
    init, v_fn = hk.without_apply_rng(
        hk.transform(functools.partial(v_head,
                                        with_bias=with_bias,
                                        w_init=w_init)))

    w = init(rng_key, x=dummy_x)
    return v_fn, w

def create_nonlinear_z_head(
    rng_key: jnp.ndarray,
    dummy_x: Any,
    trace: Any,
):
    """Pick and initialise a haiku network."""
    init_fn, z_fn = hk.without_apply_rng(
        hk.transform(functools.partial(_non_linear_z_head, trace=trace)))

    w_z = init_fn(rng_key, x=dummy_x)
    return z_fn, w_z

def create_z_head(
    rng_key: jnp.ndarray,
    dummy_x: Any,
    trace: Any,
):
    """Pick and initialise a haiku network."""
    init_fn, z_fn = hk.without_apply_rng(
        hk.transform(functools.partial(_z_head, trace=trace)))

    w_z = init_fn(rng_key, x=dummy_x)
    return z_fn, w_z

def _non_linear_z_head(x, trace):
    """Creates a network head for the expected eligibility trace.

    The small network learns to estimate the gradients of the q-head network.

    Args:
        x: input of the expected trace net,
        trace: the trace,

    Returns:
        The expected trace z.
    """
    trace_flat, trace_tree = jax.tree_flatten(trace)

    expected_traces = []
    for trace_i in trace_flat:
        net = hk.Sequential([
            hk.Linear(20),
            jax.nn.relu,
            # hk.Linear(20),
            # jax.nn.relu,
        ])
        x = net(x)
        x = hk.Linear(np.prod(trace_i.shape))(x)
        x = jnp.reshape(x, trace_i.shape)
        expected_traces.append(x)

    z = jax.tree_unflatten(trace_tree, expected_traces)
    return z

def _z_head(x, trace):
    """Creates a network head for the expected eligibility trace.

    The small network learns to estimate the gradients of the q-head network.

    Args:
        x: input of the expected trace net,
        trace: the trace,

    Returns:
        The expected trace z.
    """
    trace_flat, trace_tree = jax.tree_flatten(trace)

    expected_traces = []
    for trace_i in trace_flat:
        x = hk.Linear(np.prod(trace_i.shape))(x)
        x = jnp.reshape(x, trace_i.shape)
        expected_traces.append(x)

    z = jax.tree_unflatten(trace_tree, expected_traces)
    return z

def q_head(x, num_actions):
    q = hk.Linear(num_actions)
    return q(x)

def v_head(x, with_bias=None, w_init=None):
    if w_init is not None:
        v = hk.Linear(1, with_bias=with_bias, w_init=w_init)
    else:
        v = hk.Linear(1, with_bias=with_bias)
    return v(x)


def create_v_net(
    rng_key: jnp.ndarray,
    network_spec: str,
    observation_spec: Any,
    with_bias: bool = True,
    w_init: Any = None
):
    """Pick and initialise a haiku network."""

    # Usually this is a genrl environment id, so we extract domain name and use
    # that to select an agent network.
    if network_spec == "linear":
        net = linear_net(observation_spec, with_bias, w_init)
    else:
        net = mlp_net(observation_spec, with_bias, w_init)

    init, v_fn = hk.without_apply_rng(
        hk.transform(functools.partial(net, num_actions=1)))
    dummy_observation = jax.tree_map(lambda x: jnp.zeros(x.shape, dtype=x.dtype),
                                    observation_spec)
    w = init(rng_key, obs=dummy_observation)
    return v_fn, w

def create_net(
    rng_key: jnp.ndarray,
    network_spec: str,
    observation_spec: Any,
    action_spec: Any,
    with_bias: bool = True,
):
    """Pick and initialise a Haiku network."""

    # Usually this is a genrl environment id, so we extract domain name and use
    # that to select an agent network.
    if network_spec == "linear":
        net = linear_net(observation_spec, with_bias)
    else:
        t = mlp_net(observation_spec, with_bias)
    num_actions = _get_num_actions(action_spec)

    init, q_fn = hk.without_apply_rng(
        hk.transform(functools.partial(net, num_actions=num_actions)))
    dummy_observation = jax.tree_map(lambda x: jnp.zeros(x.shape, dtype=x.dtype),
                                    observation_spec)
    w = init(rng_key, obs=dummy_observation)
    return q_fn, w
