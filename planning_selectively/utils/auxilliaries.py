#@title Aux functions: smooth, create_optimiser...
from imports import *

# def smooth(x, m):
#   if m <= 1:
#     return x
#   n = max([1, len(x)//m])
#   return np.convolve(x, np.ones(n)/n, 'valid')

def smoothed(xs, ys, window=5):
  if window <= 1:
    return xs, ys
  nx = max([1, len(xs)//window])
  ny = max([1, len(ys) // window])
  return (np.convolve(xs, np.ones(nx)/nx, 'valid'),
    np.convolve(ys, np.ones(ny)/ny, 'valid'))

# def smoothed(xs, ys, window=5):
#   L = len(xs)
#   x = []
#   y = []
#   for i in range(L):
#     neighbors = min(i, L-1-i, window//2)
#     x.append(np.mean(xs[i-neighbors:i+neighbors+1]))
#     y.append(np.mean(ys[i-neighbors:i+neighbors+1]))
#   return np.array(x), np.array(y)

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
  if learning_rate is None:
    return optax.chain(
      # optax.trace(momentum, nesterov=False),
      optax.scale_by_schedule(lambda t: (1/t)**d),
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


def text_rendering(g_name, g_val, index=None):
  if index is not None:
    alg_name = "QET" if index == 1 else "Q"
  else:
    alg_name = ""
  if g_name == 'noisy_prob':
    return r'baselines, no noise' if g_val == 0 else r"noisy observations"
  if g_name == 'eta_q':
    return r'' #r'$\eta_q: {}$'.format(g_val)
  if g_name == 'use_i':
    # return r'$\omega_t \!=\! i^\varepsilon_t$' if g_val else r'$\omega_t\! =\! 1$'
    return r'{}$\omega_t$'.format(alg_name) if g_val else r'{} uniform $\omega$'.format(alg_name)
  if g_name == 'i_dep_trace_param':
    # return r'$\lambda_t \!=\! \beta \omega_t \!+\! (1 \!-\! \omega_t)$' if g_val else r'$\lambda_t \!=\! \lambda$'
    return r'$\lambda^{\omega^i}$' if g_val else r''  # $\lambda$'
  if g_name == 'i_dep_eta':
    # return r'$\eta_t \!=\! \beta \omega_t \!+\! (1 \!-\! \omega_t)$' if g_val else r'$\eta_t \!=\! \eta$'
    return r'$\eta^{\omega^i}$' if g_val else r''  # $\eta$'
  if g_name == 'i_dep_et_loss':
    # return r'$\Delta_{\mathbf{z}} \!*\!=\! \omega_t$' if g_val else r''
    return r'$\Delta^{\omega^i}_{\mathbf{z}}$' if g_val else r''
  else:
    return g_val


def label_fn(group, index, *g):
  # labels = []
  label_suffix = ""
  if ("noisy_prob", 0) in zip(group, *g):
    label_suffix = "(baselines, no noise)"
  if ("noisy_prob", 0.5) in zip(group, *g):
    label_suffix = "(noisy observations)"
  if ("use_i", 1) in zip(group, *g) and ("i_dep_trace_param", 1) in zip(group, *g):
    label = r"Q($\lambda_t, \omega_t$)"
  if ("use_i", 1) in zip(group, *g) and ("i_dep_trace_param", 0) in zip(group, *g):
    label = r"Q($\lambda, \omega_t$)"
  if ("use_i", 0) in zip(group, *g) :
    label = r"Q($\lambda$)-baseline"
  # for g_name, g_val in zip(group, *g):
  #   if g_name == "use_i" and g_val == 0 and g[0][group.index("noisy_prob")] == 0:
  #     labels.append(r'QET')
  #   else:
  #   labels.append(text_rendering(g_name, g_val, index=index))
  # labels = [l for l in labels if l != '']
  # labels1 = labels[:-1]
  # labels2 = labels[-1:]
  # a = [','.join(labels1)]
  a = [label]
  a.append(label_suffix)
  return ' '.join(a)

def label_fn2(group, index, *g):
  labels = []
  for g_name, g_val in zip(group, *g):
    if g_name == "use_i" and g_val == 0 and g[0][group.index("noisy_prob")] == 0:
      labels.append(r'QET')
    else:
      labels.append(text_rendering(g_name, g_val, index=index))
  labels = [l for l in labels if l != '']
  labels1 = labels[:-1]
  labels2 = labels[-1:]
  a = [','.join(labels1)]
  a.extend(labels2)
  return '\n'.join(a)


def facet_fn(group, *g):
  labels = []
  for g_name, g_val in zip(group, *g):
    labels.append(text_rendering(g_name, g_val))
  labels = [l for l in labels if l != '']
  return ','.join(labels)


def axis_fn(name):
  if name == 'agent_steps':
    return "Steps"
  if name == "mean_return":
    return "Mean return"


def ls_fn(group):
  return '-'


def alpha_fn(group):
  return 0.8