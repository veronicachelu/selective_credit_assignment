#@title imports
# from google.colab import files
import itertools
from functools import reduce
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FormatStrFormatter
import colabtools
import matplotlib.pyplot as plt
import numpy as np
import itertools
import matplotlib.cm as cm
from matplotlib.patches import Rectangle, Polygon
# from IPython.display import clear_output
# %matplotlib inline
# from colabtools import adhoc_import
import sys
from typing import Any, Dict
from bsuite.environments import base
import dm_env
import functools
import collections
from collections import defaultdict
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax

Environment = collections.namedtuple('Environment', 'step reset')

tmap = jax.tree_multimap
treduce = jax.tree_util.tree_reduce
OBS_ONEHOT = 'onehot'
OBS_RANDOM = 'random'
OBS_XY = 'xy'
import functools

from typing import Any, Callable, Mapping, NamedTuple, Tuple

import dm_env
import jax
import jax.numpy as jnp
# import ml_collections
import rlax

import sys
from typing import Any, Dict

from bsuite.environments import base
import sys
import numpy as np
import dm_env
from dm_env import specs
import numpy as np
tmap = jax.tree_multimap
add = lambda x, y: x + y
tree_add = lambda x, y: tmap(add, x, y)
tree_sub = lambda x, y: tmap(lambda x, y: x - y, x, y)