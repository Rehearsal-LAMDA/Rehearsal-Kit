import itertools

import networkx as nx
import numpy as np
import scipy
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import PolynomialFeatures


class Environment:
    def __init__(self, x_dim, z_dim, y_dim, children_parents_pairs, alterable_vars='all', alter_ranges=None,
                 mode='linear', free_var_scale=1, beta_scale=1, noise=0.05, seed=2024):
        '''
        :param child_parent_pairs: list of tuples where each tuple is of the form (children, parents), children and parents are lists of integers starting from zero that represents children and parents.
        '''
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.mode = mode
        if mode == 'poly':
            self.poly = PolynomialFeatures(degree=2, include_bias=False)

        self.children_parents_pairs = children_parents_pairs
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.z_dim = z_dim
        self.n_vars = x_dim + y_dim + z_dim
        self.x_vars = list(range(x_dim))
        self.z_vars = list(range(x_dim, x_dim + z_dim))
        self.y_vars = list(range(x_dim + z_dim, self.n_vars))
        self.alterable_vars = self.z_vars if alterable_vars == 'all' else sorted(alterable_vars)
        self.alter_ranges = [-np.inf, np.inf] if alter_ranges is None else alter_ranges
        self.free_var_scale = free_var_scale
        self.beta_scale = beta_scale
        self.noise = noise

        self.reordered = False
        self.desired_region = None
        self.generated_vars = sorted(set(itertools.chain(*[children for children, _ in self.children_parents_pairs])))
        self.free_vars = sorted(set(range(self.n_vars)) - set(self.generated_vars))

        self._check_var_groups()
        self._check_var_indices()
        self._check_unique_parents()
        self._check_rehearsal_graph()

        self._reorder()

        self.srm_params = {children: dict() for children, _ in self.children_parents_pairs}

        self._generate_srm_params()

    def observe_x(self):
        obs_x = self.get_observations(2)[0, self.x_vars].reshape(-1)
        return obs_x

    def alter_z(self, observed_x, alter_vars, values):
        assert len(set(alter_vars) - set(self.alterable_vars)) == 0
        alter_obs = self._get_alteration_samples(observed_x, alter_vars, values, n_samples=1).squeeze()
        return alter_obs

    def get_observations(self, n_observations):
        data = np.zeros((n_observations, self.n_vars))
        data[:, self.free_vars] = self._generate_free_var_samples(len(self.free_vars), n_observations)
        for children, parents in self.children_parents_pairs:
            data[:, children] = self._call_srm(data[:, parents], self.srm_params[children][parents])
        return data

    def get_success_prob(self, observed_x, alter_vars, values, n_samples=1000):
        assert self.desired_region is not None
        y = self._get_alteration_samples(observed_x, alter_vars, values, n_samples)[:, self.y_vars]
        M, d = self.desired_region['M'], self.desired_region['d']
        n_satisfied = ((np.dot(y, M.T) - d <= 0).all(axis=1)).sum()
        prob = n_satisfied / n_samples
        return prob

    def get_y_observations(self, n_observations):
        return self.get_observations(n_observations)[:, self.y_vars]

    def set_desired_region(self, M, d):  # My <= d
        self.desired_region = {'M': M, 'd': d}

    def get_ordered_children_parents(self):
        return self.children_parents_pairs.copy()

    def get_free_vars(self):
        return self.free_vars.copy()

    def get_group(self, var_idx):
        assert 0 <= var_idx < self.n_vars
        if var_idx < self.x_dim:
            return 'x'
        elif var_idx < self.x_dim + self.z_dim:
            return 'z'
        else:
            return 'y'

    def set_srm_params(self, params):
        assert self.reordered
        for children, parents in self.children_parents_pairs:
            self.srm_params[children][parents] = params[children][parents].copy()

    def _generate_srm_params(self):
        assert self.reordered
        for children, parents in self.children_parents_pairs:
            if self.mode == 'linear':
                coef = self.rng.uniform(-self.beta_scale, self.beta_scale, (len(parents), len(children)))
                bias = self.rng.uniform(-self.beta_scale, self.beta_scale, len(children))
                self.srm_params[children][parents] = {'coef': coef, 'bias': bias, 'noise_cov': np.eye(len(children)) * self.noise}
            elif self.mode == 'poly':
                aug_dim = len(parents) + len(parents) * (len(parents) - 1) // 2 + len(parents)
                c = (self.free_var_scale ** 2) * len(parents) * self.beta_scale * 10
                coef = self.rng.uniform(-self.beta_scale, self.beta_scale, (aug_dim, len(children))) / c
                bias = self.rng.uniform(-self.beta_scale, self.beta_scale, len(children)) / c
                self.srm_params[children][parents] = {'coef': coef, 'bias': bias,
                                                      'noise_cov': np.eye(len(children)) * self.noise}
            elif self.mode == 'mlp':
                hidden = 100
                W1 = self.rng.uniform(low=0.5, high=2.0, size=(len(parents), hidden))
                W1[self.rng.uniform(size=W1.shape) < 0.5] *= -1
                W2 = self.rng.uniform(low=0.5, high=2.0, size=(hidden, len(children)))
                W2[self.rng.uniform(size=W2.shape) < 0.5] *= -1
                W3 = self.rng.uniform(low=0.5, high=2.0, size=(hidden, len(children)))
                W3[self.rng.uniform(size=W3.shape) < 0.5] *= -1

                self.srm_params[children][parents] = {'w1': W1, 'w2': W2, 'w3': W3,
                                                      'noise_cov': np.eye(len(children)) * self.noise}
            else:
                raise NotImplementedError(f"Mode {self.mode} not implemented")
        if self.mode == 'mlp':
            data = np.zeros((1000, self.n_vars))
            data[:, self.free_vars] = self._generate_free_var_samples(len(self.free_vars), 1000)
            for children, parents in self.children_parents_pairs:
                data[:, children] = self._call_srm(data[:, parents], self.srm_params[children][parents])

    def _call_srm(self, parent_data, params):
        assert len(parent_data.shape) == 2
        children_dim = params['noise_cov'].shape[0]
        if self.mode == 'linear':
            return np.dot(parent_data, params['coef']) + params['bias'] + \
                self.rng.multivariate_normal([0] * children_dim, params['noise_cov'], parent_data.shape[0])
        elif self.mode == 'poly':
            parent_data = self.poly.fit_transform(parent_data)
            return np.dot(parent_data, params['coef']) + params['bias'] + \
                self.rng.multivariate_normal([0] * children_dim, params['noise_cov'], parent_data.shape[0])
        elif self.mode == 'mlp':
            data = scipy.special.expit(parent_data @ params['w1']) @ params['w2']
            if 'std' not in params:
                params['std'] = np.std(data, axis=0)
            data += self.rng.multivariate_normal([0] * children_dim, params['noise_cov'] * params['std'], parent_data.shape[0])
            return data

    def _get_alteration_samples(self, observed_x, alter_vars, values, n_samples):
        # special treatment for alterations on variables that belong to a bidirectional clique
        data = np.zeros((n_samples, self.n_vars))
        data[:, self.free_vars] = self._generate_free_var_samples(len(self.free_vars), n_samples)
        data[:, self.x_vars] = observed_x
        data[:, alter_vars] = np.array(values)

        for children, parents in self.children_parents_pairs:
            if children[0] in self.x_vars:
                continue
            intersect_children = set(children) & set(alter_vars)
            if len(intersect_children) == 0:
                data[:, children] = self._call_srm(data[:, parents], self.srm_params[children][parents])
            else:
                if len(children) == len(intersect_children):  # all children are altered, already processed at the beginning
                    continue
                else:  # part of the children are altered, the parents of the remaining children should be updated
                    new_parents = tuple(sorted(set(parents) | (set(children) - intersect_children)))
                    if new_parents not in self.srm_params[children]:
                        # print('generate new srm')
                        self.srm_params[children][new_parents] = self._get_alter_srm_params(children, new_parents)
                    data[:, children] = self._call_srm(data[:, new_parents], self.srm_params[children][new_parents])
        return data

    def _get_alter_srm_params(self, children, parents):
        data = self.get_observations(1000)
        if self.mode == 'linear':
            x = data[:, parents]
            y = data[:, children]
            model = MultiOutputRegressor(LinearRegression()).fit(x, y)
            coefs = []
            bias = []
            for lr in model.estimators_:
                coefs.append(lr.coef_)
                bias.append(lr.intercept_)
            residual = y - model.predict(x)
            noise_cov = np.cov(residual.T)
            return {'coef': np.vstack(coefs).T, 'bias': np.array(bias), 'noise_cov': noise_cov}
        elif self.mode == 'poly':
            x = self.poly.fit_transform(data[:, parents])
            y = data[:, children]
            model = MultiOutputRegressor(LinearRegression()).fit(x, y)
            coefs = []
            bias = []
            for lr in model.estimators_:
                coefs.append(lr.coef_)
                bias.append(lr.intercept_)
            residual = y - model.predict(x)
            noise_cov = np.cov(residual.T)
            return {'coef': np.vstack(coefs).T, 'bias': np.array(bias), 'noise_cov': noise_cov}

    def _generate_free_var_samples(self, dim, n_samples):
        if not isinstance(self.free_var_scale, (int, float)):
            assert dim == len(self.free_var_scale)
            arrs = []
            for low, high in self.free_var_scale:
                arrs.append(self.rng.uniform(low, high, n_samples))
            return np.vstack(arrs).T
        return self.rng.normal(scale=self.free_var_scale, size=(n_samples, dim)) + \
            self.rng.normal(scale=self.noise, size=(n_samples, dim))

    def _reorder(self):
        new_children_parents_pairs = []
        visited = [False] * len(self.children_parents_pairs)
        generated = [False] * self.n_vars
        for var in self.free_vars:
            generated[var] = True
        while len(new_children_parents_pairs) < len(self.children_parents_pairs):
            for i, (children, parents) in enumerate(self.children_parents_pairs):
                if visited[i] == False and all([generated[parent] for parent in parents]):
                    new_children_parents_pairs.append((tuple(sorted(children)), tuple(sorted(parents))))
                    visited[i] = True
                    for child in children:
                        generated[child] = True
        self.children_parents_pairs = new_children_parents_pairs
        self.reordered = True

    def _check_rehearsal_graph(self):
        G = nx.DiGraph()
        for children, parents in self.children_parents_pairs:
            for child in children:
                for parent in parents:
                    G.add_nodes_from([child, parent])
                    G.add_edge(parent, child)
        for children, parents in self.children_parents_pairs:
            if len(children) > 1:
                u = children[0]
                for v in children[1:]:
                    nx.contracted_nodes(G, u, v, self_loops=False, copy=False)
        assert nx.is_directed_acyclic_graph(G)

    def _check_var_indices(self):
        assert len([1 for c, p in self.children_parents_pairs if min(c) < 0 or min(p) < 0 or max(c) >= self.n_vars or max(p) >= self.n_vars]) == 0

    def _check_var_groups(self):
        group2order = {'x': 0, 'z': 1, 'y': 2}
        for children, parents in self.children_parents_pairs:
            children_group = None
            for child in children:
                if children_group is None:
                    children_group = self.get_group(child)
                else:
                    assert children_group == self.get_group(child)
            for parent in parents:
                assert group2order[self.get_group(parent)] <= group2order[children_group]

    def _check_unique_parents(self):
        visited_children = set()
        for children, parents in self.children_parents_pairs:
            for child in children:
                assert child not in visited_children
                visited_children.add(child)
