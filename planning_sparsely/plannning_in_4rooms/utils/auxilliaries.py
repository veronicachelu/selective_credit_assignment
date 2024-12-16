#@title Aux functions: smooth, create_optimiser...
from imports import *

def smooth(xs, ys, window=5):
  if window <= 1:
    return xs, ys
  nx = max([1, len(xs)//window])
  ny = max([1, len(ys) // window])
  return (np.convolve(xs, np.ones(nx)/nx, 'valid'),
    np.convolve(ys, np.ones(ny)/ny, 'valid'))
#
# def smooth(x, m):
#   if m <= 1:
#     return x
#   n = max([1, len(x)//m])
#   return np.convolve(x, np.ones(n)/n, 'valid')

def check_simple_observation_spec(observation_spec):
  specs = jax.tree_leaves(observation_spec)
  if len(specs) > 1:
    raise ValueError('Complex observations not supported. Found observation '
                     f'spec: {observation_spec}')


def check_simple_action_spec(action_spec):
  specs, structure = jax.tree_flatten(action_spec)
  all_disc = np.all(list(spec.dtype in [np.int32, np.int64] for spec in specs))
  if len(specs) > 1 or not all_disc:
    raise ValueError('Learning algorithms can currently only support '
                     'environments with a single discrete action. Action spec: '
                     f'{action_spec}')
  return structure


def create_optimiser(b1, b2, eps, learning_rate, momentum, d):
  def schedule_fn(t):
    return (1 / (t + 1)) ** d

  if learning_rate is None:
    return optax.chain(
      # optax.trace(momentum, nesterov=False),
      optax.scale_by_schedule(schedule_fn),
      # optax.scale(learning_rate),
    )
  else:
    return optax.chain(
      # optax.trace(momentum, nesterov=False),
      # optax.scale_by_adam(b1=b1, b2=b2, eps=eps),
      optax.scale(learning_rate),
    )

def dummy(observation_spec):
  return jax.tree_map(
      lambda x: jnp.zeros(x.shape, dtype=x.dtype), observation_spec)

def _get_num_actions(action_spec):
  check_simple_action_spec(action_spec)
  return jax.tree_leaves(action_spec)[0].num_values

def eps_greedy(rng_key, q, epsilon):
  greedy = (q == q.max(axis=-1, keepdims=True)).astype(jnp.float32)
  greedy /= jnp.sum(greedy, axis=-1, keepdims=True)
  pi = (1 - epsilon)*greedy + epsilon/q.shape[-1]
  cpi = jnp.cumsum(pi, axis=-1)
  rnd = jax.random.uniform(rng_key)
  a = jnp.argmax(cpi > rnd, axis=-1)
  return a, pi[a]
