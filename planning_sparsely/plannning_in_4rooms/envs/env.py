# @title env
from imports import *
class GridWorld(base.Environment):
    _str_mdp = ''
    _num_rows = -1
    _num_cols = -1
    _num_states = -1
    _matrix_mdp = None
    _adj_matrix = None
    _reward_function = None
    _use_negative_rewards = False

    _curr_x = 0
    _curr_y = 0
    _goal_x = 0
    _goal_y = 0

    def __init__(self, seed, str_in=None,
                 teps=0.0, path=None, option_reward=1.,
                 reward=1., reps=1., option_reps=1., eval=False,
                 observation_type=OBS_ONEHOT, dim_obs=-1,
                 time_limit=-1, use_q_for_reward=False):
        super(GridWorld, self).__init__()

        if path is not None:
            self._read_file(path)
        elif str_in is not None:
            self._str_mdp = str_in
        else:
            print("You are supposed to provide an MDP specification as input!")
            sys.exit()

        self._start_states = []
        self._option_reward = option_reward
        self._eval = eval
        self._num_actions = len(self.get_action_set())
        self._parse_string()
        self._num_states = self._num_rows * self._num_cols
        self._time_limit = time_limit
        self._rval = reward
        self._q_star = None
        self._teps = teps
        self._reps = reps
        self._option_reps = option_reps
        self._nrng = np.random.RandomState(seed)
        self._transition_matrix = self.get_transition_matrix()
        self._reward_matrix = self.get_reward_matrix()
        self._discount_matrix = self.get_discount_matrix()
        self._obs_type = observation_type
        self._options_q_star = None
        self._use_q_for_reward = use_q_for_reward

        if self._obs_type == OBS_RANDOM:
            self._dim_obs = dim_obs
            self._obs_matrix = self._nrng.randn(self._num_states, self._dim_obs)
        elif self._obs_type == OBS_XY:
            self._dim_obs = self._num_rows + self._num_cols
        else:
            self._dim_obs = self._num_states

        self._t = 0
        self._time_limit = time_limit

    def get_all_states(self):
        dS = self._num_states
        all_observations = np.zeros((dS, self._dim_obs))
        for s in range(dS):
            all_observations[s] = self._observation(s)
        return all_observations

    def _read_file(self, path):
        """
        We just read the file and put its contents in strMDP.

        :param path: path to the file containing the MDP description
        """
        file_name = open(path, 'r')
        for line in file_name:
            self._str_mdp += line

    def _parse_string(self):
        """
        I now parse the received string. I'll store everything in a matrix (matrixMDP) such that -1 means wall and 0
        means available square. The letter 'S' is converted to the initial (x,y) position.

        :return:
        """
        data = self._str_mdp.split('\n')
        self._num_rows = int(data[0].split(',')[0])
        self._num_cols = int(data[0].split(',')[1])
        self._matrix_mdp = np.zeros((self._num_rows, self._num_cols))

        for i in range(len(data) - 1):
            for j in range(len(data[i + 1])):
                if data[i + 1][j] == 'X':
                    self._matrix_mdp[i][j] = -1
                elif data[i + 1][j] == '.':
                    self._matrix_mdp[i][j] = 0
                elif data[i + 1][j] == 'S':
                    self._matrix_mdp[i][j] = 0
                    self._start_states.append(self._get_state_idx(i, j))
                elif data[i + 1][j] == 'G':
                    self._matrix_mdp[i][j] = 0
                    self._goal_x = i
                    self._goal_y = j

    def lift_policy(self, policy, option_policies):
        new_policy = np.zeros_like(policy)
        for s in range(self._num_states):
            option = np.argmax(policy[s])
            option_policy = option_policies[option]
            a = np.argmax(option_policy[s])
            new_policy[s][a] = 1
        return new_policy

    def _fill_transition_matrix(self):
        self._transition_matrix = np.zeros((self._num_states, self._num_actions, self._num_states), dtype=np.float)
        for s in range(self._num_states):
            _curr_x, _curr_y = self.get_state_xy(s)
            if self._matrix_mdp[_curr_x][_curr_y] != -1:
                legal_moves = []
                for a, action in enumerate(self.get_action_set()):
                    if action == 'up' and _curr_x > 0:
                        next_x = _curr_x - 1
                        next_y = _curr_y
                    elif action == 'right' and _curr_y < self._num_cols - 2:
                        next_x = _curr_x
                        next_y = _curr_y + 1
                    elif action == 'down' and _curr_x < self._num_rows - 2:
                        next_x = _curr_x + 1
                        next_y = _curr_y
                    elif action == 'left' and _curr_y > 0:
                        next_x = _curr_x
                        next_y = _curr_y - 1
                    else:
                        next_x = _curr_x
                        next_y = _curr_y
                    if self._matrix_mdp[next_x][next_y] == -1:
                        next_x = _curr_x
                        next_y = _curr_y
                    next_s = self._get_state_idx(next_x, next_y)
                    legal_moves.append((a, next_s))
                for a, next_s in legal_moves:
                    self._transition_matrix[s][a][next_s] += 1 - self._teps
                    for other_a, other_s in legal_moves:
                        self._transition_matrix[s][a][other_s] += self._teps / len(legal_moves)

    def _fill_reward_matrix(self):
        self._reward_matrix = np.zeros((self._num_states,), dtype=np.float)
        for next_s in range(self._num_states):
            next_x, next_y = self.get_state_xy(next_s)
            if next_x == self._goal_x and next_y == self._goal_y:
                self._reward_matrix[next_s] = self._rval

    def _fill_discount_matrix(self):
        self._discount_matrix = np.ones((self._num_states,), dtype=np.float)
        for s in range(self._num_states):
            x, y = self.get_state_xy(s)
            self._discount_matrix[s] = not self.is_terminal(x, y)

    def _get_state_idx(self, x, y):
        """
        Given a state coordinate (x,y) this method returns the index that uniquely identifies this state.

        :param x: value of the coordinate x
        :param y: value of the coordinate y
        :return : unique index identifying a position in the grid
        """
        idx = y + x * self._num_cols
        return idx

    @staticmethod
    def get_action_set():
        """
        I'm only supporting the four directional actions for now.

        :return: action set
        """
        return ['up', 'right', 'down', 'left']

    def get_state_xy(self, idx):
        """
        Given the index that uniquely identifies each state this method returns its equivalent coordinate (x,y).

        :param idx: index uniquely identifying a state
        :return: values x, y describing the state's location in the grid
        """
        y = int(idx % self._num_cols)
        x = int((idx - y) / self._num_cols)
        return x, y

    def _observation(self, s):
        if self._obs_type == OBS_ONEHOT:
            return np.eye(self._dim_obs)[s]
        elif self._obs_type == OBS_RANDOM:
            return self._obs_matrix[s]
        elif self._obs_type == OBS_XY:
            xy_vec = np.zeros(self._dim_obs)
            x, y = self.get_state_xy(s)
            xy_vec[x] = 1.0
            xy_vec[y + self._num_cols] = 1.0
            return xy_vec
        else:
            raise ValueError("Invalid obs type %s" % self._obs_type)

    def reset(self, state=None) -> dm_env.TimeStep:
        """Resets the environment, calling the underlying _reset() method."""
        self._reset_next_step = False
        return self._reset(state)

    def _reset(self, state=None) -> dm_env.TimeStep:
        self._curr_idx = state if state is not None else \
            self._nrng.choice(self._start_states,
            p=[1 / len(self._start_states) for _ in self._start_states])
        self._curr_x, self._curr_ = self.get_state_xy(self._curr_idx)
        self._timestep = 0
        self._t = 0
        observation = self._observation(self._curr_idx)
        return dm_env.restart(observation)

    def _step(self, action: int) -> dm_env.TimeStep:
        # s_crt = self._get_state_idx(self._curr_x, self._curr_y)
        s_next = self._nrng.choice(range(self._num_states),
                                   p=self._transition_matrix[self._curr_idx][action])

        next_x, next_y = self.get_state_xy(s_next)
        self._curr_x = next_x
        self._curr_y = next_y
        self._curr_idx = s_next

        reward = self._reward_matrix[s_next]
        if self._eval:
            stochastic_reward = reward
        else:
            stochastic_reward = self._nrng.choice([reward, 0],
                                                  p=[self._reps, 1 - self._reps])
        discount = self._discount_matrix[s_next]
        next_observation = self._observation(s_next)
        self._t += 1

        if self._time_limit > -1 and self._t >= self._time_limit:
            discount = 0

        if discount == 0:
            return dm_env.termination(reward=stochastic_reward,
                                      observation=next_observation)
        else:
            return dm_env.transition(reward=stochastic_reward,
                                     observation=next_observation)

    def observation_spec(self):
        return specs.Array(shape=(self._dim_obs,), dtype=np.float32)

    def action_spec(self, num_actions=None):
        if num_actions is not None:
            return specs.DiscreteArray(num_actions, name='action')
        return specs.DiscreteArray(self._num_actions, name='action')

    def option_spec(self, num_options):
        return specs.DiscreteArray(num_options, name='action')

    def _save(self, observation):
        self._raw_observation = (observation * 255).astype(np.uint8)

    @property
    def optimal_return(self):
        # Returns the maximum total reward achievable in an episode.
        return 10

    def bsuite_info(self) -> Dict[str, Any]:
        return {}

    def is_terminal(self, _curr_x, _curr_y):
        """
        Checks whether the agent is in a terminal state (or goal).

        :return: true if the agent's current state is a goal state, otherwise return false
        """
        if _curr_x == self._goal_x and _curr_y == self._goal_y:
            return True
        else:
            return False

    def get_transition_matrix(self):
        """
        If I never did it before, I will fill the transition matrix. Otherwise I'll just return the one I already have.

        :return: transition matrix representing the loaded MDP
        """
        if np.all(self._transition_matrix is None):
            self._fill_transition_matrix()
        return self._transition_matrix

    def get_discount_matrix(self):
        """
        If I never did it before, I will fill the reward matrix. Otherwise I'll just return the one I already have.

        :return: reward matrix representing the loaded MDP
        """
        if np.all(self._discount_matrix is None):
            self._fill_discount_matrix()
        return self._discount_matrix

    def get_reward_matrix(self):
        """
        If I never did it before, I will fill the reward matrix. Otherwise I'll just return the one I already have.

        :return: reward matrix representing the loaded MDP
        """
        if np.all(self._reward_matrix is None):
            self._fill_reward_matrix()
        return self._reward_matrix

    def _q_backup_sparse(self, q_values, discount=0.99):
        dS = self._num_states
        dA = self._num_actions
        discounts = discount * self._discount_matrix
        new_q_values = np.zeros_like(q_values)
        value = np.max(q_values, axis=1)
        for s in range(dS):
            for a in range(dA):
                new_q_value = 0
                for ns, prob in enumerate(self._transition_matrix[s][a]):
                    new_q_value += prob * (self._reward_matrix[ns] * self._reps +
                                           discounts[ns] * value[ns])
                new_q_values[s, a] = new_q_value
        return new_q_values

    def _option_q_backup_sparse(self, q_values, option_interest, option_goal, discount=0.99):
        dS = self._num_states
        dA = self._num_actions
        #   discounts = discount * self._discount_matrix
        new_q_values = np.zeros_like(q_values)
        value = np.max(q_values, axis=1)
        for s in range(dS):
            for a in range(dA):
                new_q_value = 0
                for ns, prob in enumerate(self._transition_matrix[s][a]):
                    option_reward = self._option_reward * (option_interest[s] - option_interest[ns])
                    if self._use_q_for_reward:
                        option_reward *= option_goal[ns]
                    option_discount = option_interest[ns] * discount
                    new_q_value += prob * (option_reward * self._option_reps +
                                           option_discount * value[ns])
                new_q_values[s, a] = new_q_value * option_interest[s]
        return new_q_values

    def q_iteration(self, num_itrs=100, **kwargs):
        if self._q_star is None:
            q_values = np.zeros((self._num_states, self._num_actions))
            for i in range(num_itrs):
                q_values = self._q_backup_sparse(q_values, **kwargs)
            self._q_star = q_values
        return self._q_star

    def options_q_iteration(self, option_interests=None, num_itrs=100, option_goals=None, **kwargs):
        if self._options_q_star is None:
            self._options_q_star = []
            for option_id, option_interest in enumerate(option_interests):
                option_q_values = np.zeros((self._num_states, self._num_actions))
                for i in range(num_itrs):
                    option_q_values = self._option_q_backup_sparse(option_q_values,
                                                                   option_interest,
                                                                   option_goal=option_goals[option_id],
                                                                   **kwargs)
                self._options_q_star.append(option_q_values)
        return self._options_q_star

    # def get_options_sa_stationary(self, options_policy, option_interests, discount=1.0, T=100):
    #     dS = self._num_states
    #     dA = self._num_actions
    #     for option, option_interest in enumerate(option_interests):
    #         option_policy = options_policy[option]
    #         d_option = self.get_sa_stationary(option_policy, discount=1.0, T=100)
    def get_d_pi(self, pi, discount=None):
        discount = 1. if discount is None else discount
        discounts = discount * self._discount_matrix
        t_matrix = self._transition_matrix
        # disc_ppi = np.einsum('ikj, j, ik->ij', t_matrix, discounts, pi)
        # d_pi = np.linalg.inv(np.eye(self._num_states) - disc_ppi).dot(self._discount_matrix)
        # d_pi /= np.sum(d_pi)

        ppi = np.einsum('ikj, ik->ij', t_matrix, pi)
        A = np.eye(self._num_states) - ppi
        A = np.vstack((A.T, np.ones(self._num_states)))
        b = np.matrix([0] * self._num_states + [1]).T
        d_pi = np.linalg.lstsq(A, b)[0]
        d_pi = discounts * np.array(d_pi.T)[0]
        d_pi /= np.sum(d_pi)
        return d_pi

    def get_sa_stationary(self, policy, discount=1.0, T=100):
        dS = self._num_states
        dA = self._num_actions
        state_visitation = np.zeros((dS, 1))
        #   start_s = self._get_state_idx(self._start_x, self._start_y)
        for start_s in self._start_states:
            state_visitation[start_s] = 1 / len(self._start_states)
        t_matrix = self._transition_matrix  # S x A x S
        sa_visit_t = np.zeros((dS, dA, T))

        norm_factor = 0.0
        for i in range(T):
            sa_visit = state_visitation * policy
            cur_discount = (discount ** i)
            sa_visit_t[:, :, i] = cur_discount * sa_visit
            norm_factor += cur_discount
            # sum-out (SA)S
            new_state_visitation = np.einsum('ij,ijk->k', sa_visit, t_matrix)
            state_visitation = np.expand_dims(new_state_visitation, axis=1)
        return np.sum(sa_visit_t, axis=2) / norm_factor

    def get_s_stationary(self, policy, discount=1.0, T=100):
        dS = self._num_states
        dA = self._num_actions
        state_visitation = np.zeros((dS, 1))
        #   start_s = self._get_state_idx(self._start_x, self._start_y)
        for start_s in self._start_states:
            state_visitation[start_s] = 1 / len(self._start_states)
        t_matrix = self._transition_matrix  # S x A x S
        sa_visit_t = np.zeros((dS, dA, T))

        norm_factor = 0.0
        for i in range(T):
            sa_visit = state_visitation * policy
            cur_discount = (discount ** i)
            sa_visit_t[:, :, i] = cur_discount * sa_visit
            norm_factor += cur_discount
            # sum-out (SA)S
            new_state_visitation = np.einsum('ij,ijk->k', sa_visit, t_matrix)
            state_visitation = np.expand_dims(new_state_visitation, axis=1)
        d_sa = np.sum(sa_visit_t, axis=2) / norm_factor
        d_s = np.sum(d_sa * policy, axis=-1)
        return d_s

    def get_option_s_stationary(self, policy, discounts=None, T=100):
        dS = self._num_states
        dA = self._num_actions
        state_visitation = np.zeros((dS, 1))
        for start_s in self._start_states:
            state_visitation[start_s] = 1 / len(self._start_states)
        t_matrix = self._transition_matrix  # S x A x S
        sa_visit_t = np.zeros((dS, dA, T))

        norm_factor = 0.0
        for i in range(T):
            sa_visit = state_visitation * policy
            # cur_discount = np.matrix_power(discounts, i)
            # sa_visit_t[:, :, i] = cur_discount * sa_visit
            norm_factor += cur_discount
            # sum-out (SA)S
            new_state_visitation = np.einsum('ij,ijk, k->k', sa_visit, t_matrix, discounts)
            state_visitation = np.expand_dims(new_state_visitation, axis=1)
        d_sa = np.sum(sa_visit_t, axis=2) / norm_factor
        d_s = np.sum(d_sa * policy, axis=-1)
        return d_s

    def get_q_pi(self, pi, discount=None):
        dS = self._num_states
        dA = self._num_actions
        discount = 1. if discount is None else discount
        discounts = discount * self._discount_matrix
        t_matrix = self._transition_matrix
        disc_p = np.einsum('ikj, i->ikj', t_matrix, discounts)
        disc_ppi = np.einsum('ikj, jm->ikjm', disc_p, pi)
        disc_ppi_flat = np.reshape(disc_ppi, (dS * dA, dS * dA))
        r = self._reward_matrix * self._reps
        rpi = np.einsum("j, ikj->ik", r, t_matrix)
        rpi_flat = np.reshape(rpi, (-1))
        q = np.linalg.solve(np.eye(dS * dA) - disc_ppi_flat, rpi_flat)
        q = np.reshape(q, (dS, dA))
        return q

    def get_option_reward_matrix(self, option_interest, v_star, discount):
        dS = self._num_states
        dA = self._num_actions
        option_reward_matrix = np.zeros((dS, dA, dS), dtype=np.float)
        for s in range(dS):
            for a in range(dA):
                for next_s in range(dS):
                    option_reward_matrix[s][a][next_s] = (self._option_reward *
                                                          (option_interest[s] - option_interest[next_s]))
                    if self._use_q_for_reward:
                        option_reward_matrix[s][a][next_s] *= v_star[s]
        return option_reward_matrix

    def get_options_q_pi(self, options_pi, option_interests, v_star,
                         option_discount=None):  # , discount=None):
        dS = self._num_states
        dA = self._num_actions
        t_matrix = self._transition_matrix
        options_q = []
        #   discount = 1. if discount is None else discount
        for option, option_interest in enumerate(option_interests):
            option_discount = 1. if option_discount is None else option_discount
            discounts = option_discount * option_interest
            pi = options_pi[option]
            disc_p = np.einsum('ikj, i->ikj', t_matrix, discounts)
            disc_ppi = np.einsum('ikj, jm->ikjm', disc_p, pi)
            disc_ppi_flat = np.reshape(disc_ppi, (dS * dA, dS * dA))
            r = self.get_option_reward_matrix(option_interest, v_star, option_discount)
            rpi = np.einsum("ikj, ikj->ik", r, t_matrix)
            rpi_flat = np.reshape(rpi, (-1))
            q = np.linalg.solve(np.eye(dS * dA) - disc_ppi_flat, rpi_flat)
            q = np.reshape(q, (dS, dA))
            options_q.append(q)
        return options_q

    def get_options_f_pi(self, options_pi, mu, option_interests, discount=None):
        dS = self._num_states
        t_matrix = self._transition_matrix
        options_f = []
        for option, option_interest in enumerate(option_interests):
            discount = 1. if discount is None else discount
            discounts = discount * option_interest
            pi = options_pi[option]
            disc_p = np.einsum('ikj, j->ikj', t_matrix, discounts)
            disc_ppi = np.einsum('ikj, ik->ij', disc_p, pi)
            d_mu = self.get_s_stationary(mu, discount=discount)
            d_mu *= option_interest
            # r = self.get_option_reward_matrix(option_interest)
            # rpi = np.einsum("ikj, ikj->i", r, t_matrix, pi)
            f = d_mu.dot(np.linalg.inv(np.eye(self._num_states) - disc_ppi))
            options_f.append(f)
        return options_f

    def get_f_pi(self, pi, mu, interest=None, discount=None):
        discount = 1. if discount is None else discount
        discounts = discount * self._discount_matrix
        t_matrix = self._transition_matrix
        disc_p = np.einsum('ikj, j->ikj', t_matrix, discounts)
        disc_ppi = np.einsum('ikj, ik->ij', disc_p, pi)
        d_mu = self.get_s_stationary(mu, discount)
        if interest is not None:
            d_mu *= interest
        r = self._reward_matrix
        rpi = np.einsum("j, ikj, ik->i", r, t_matrix, pi)
        f = d_mu.dot(np.linalg.inv(np.eye(self._num_states) - disc_ppi))
        return f

    def compute_policy_deterministic(self, q_values, eps_greedy=0.0):
        policy_probs = np.zeros_like(q_values)
        policy_probs[np.arange(policy_probs.shape[0]), np.argmax(q_values, axis=1)] = 1.0 - eps_greedy
        policy_probs += eps_greedy / (policy_probs.shape[1])
        return policy_probs
