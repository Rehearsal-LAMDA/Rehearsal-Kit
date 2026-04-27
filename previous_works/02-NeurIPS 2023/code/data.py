import networkx as nx
import numpy as np


class DataGenBias(object):
    def __init__(self, W, method='linear',
                 sem_type='gauss', noise_scale=1.0, bias=0, input_range=None,
                 cliques=None):
        self.method = method
        self.sem_type = sem_type
        self.noise_scale = noise_scale
        self.bias = bias
        self.input_range = input_range
        self.W = W.copy()
        for i in range(self.W.shape[0]):
            for j in range(i):
                if self.W[j, i] != 0 and self.W[i, j] != 0:
                    self.W[i, j] = self.W[j, i] = 0
        self.B = (self.W != 0).astype(int)
        self.cliques = cliques

    def get_graph(self):
        assert self.cliques is None
        return self.B.copy()

    def sample_iid_obs(self, n=1000):
        if self.method == 'linear':
            return self._simulate_linear_sem(n)

    def sample_iid_interv(self, interv_targets, interv_values, n=1000, x=None):
        if self.method == 'linear':
            intervs = dict(zip(interv_targets, interv_values))
            if x is not None:
                for i, v in enumerate(x):
                    intervs[i] = v
            return self._simulate_linear_interv_sem(n, intervs)

    def _simulate_linear_interv_sem(self, n, intervs):
        sem_type = self.sem_type
        W, noise_scale = self.W, self.noise_scale
        bias = self.bias
        if np.isscalar(bias):
            bias = np.ones(W.shape[0]) * bias
        input_range = self.input_range

        def _simulate_single_equation(X, w, scale):
            """X: [n, num of parents], w: [num of parents], x: [n]"""
            if sem_type == 'gauss':
                z = np.random.normal(scale=scale, size=n)
                x = X @ w + z
            elif sem_type == 'exp':
                z = np.random.exponential(scale=scale, size=n)
                x = X @ w + z
            elif sem_type == 'gumbel':
                z = np.random.gumbel(scale=scale, size=n)
                x = X @ w + z
            elif sem_type == 'uniform':
                z = np.random.uniform(low=-scale, high=scale, size=n)
                x = X @ w + z
            else:
                raise ValueError('Unknown sem type. In a linear model, \
                                     the options are as follows: gauss, exp, \
                                     gumbel, uniform, logistic.')
            return x

        d = W.shape[0]
        if noise_scale is None:
            scale_vec = np.ones(d)
        elif np.isscalar(noise_scale):
            scale_vec = noise_scale * np.ones(d)
        else:
            if len(noise_scale) != d:
                raise ValueError('noise scale must be a scalar or has length d')
            scale_vec = noise_scale
        G_nx = nx.from_numpy_matrix(W, create_using=nx.DiGraph)

        if np.isinf(n):  # population risk for linear gauss SEM
            if sem_type == 'gauss':
                # make 1/d X'X = true cov
                X = np.sqrt(d) * np.diag(scale_vec) @ np.linalg.inv(np.eye(d) - W)
                return X
            else:
                raise ValueError('population risk not available')
        # empirical risk
        ordered_vertices = list(nx.topological_sort(G_nx))
        assert len(ordered_vertices) == d
        X = np.zeros([n, d])
        for j in ordered_vertices:
            if j in intervs.keys():
                X[:, j] = intervs[j]
            else:
                parents = list(G_nx.predecessors(j))
                if len(parents) == 0 and input_range is not None and j in input_range.keys():
                    X[:, j] = np.random.uniform(input_range[j][0], input_range[j][1], X.shape[0]) + bias[j]
                else:
                    X[:, j] = _simulate_single_equation(X[:, parents], W[parents, j], scale_vec[j]) + bias[j]

        if self.cliques is not None:
            left_cliques = []
            for c in self.cliques:
                left_c = []
                for v in c:
                    if v not in intervs.keys():
                        left_c.append(v)
                left_cliques.append(left_c)
            for c in left_cliques:
                X[:, c] += np.random.multivariate_normal(mean=np.zeros(len(c)),
                                                         cov=np.full((len(c), len(c)), noise_scale / 10))
        return X

    def _simulate_linear_sem(self, n):
        sem_type = self.sem_type
        W, noise_scale = self.W, self.noise_scale
        bias = self.bias
        if np.isscalar(bias):
            bias = np.ones(W.shape[0]) * bias
        input_range = self.input_range

        def _simulate_single_equation(X, w, scale):
            """X: [n, num of parents], w: [num of parents], x: [n]"""
            if sem_type == 'gauss':
                z = np.random.normal(scale=scale, size=n)
                x = X @ w + z
            elif sem_type == 'exp':
                z = np.random.exponential(scale=scale, size=n)
                x = X @ w + z
            elif sem_type == 'gumbel':
                z = np.random.gumbel(scale=scale, size=n)
                x = X @ w + z
            elif sem_type == 'uniform':
                z = np.random.uniform(low=-scale, high=scale, size=n)
                x = X @ w + z
            else:
                raise ValueError('Unknown sem type. In a linear model, \
                                     the options are as follows: gauss, exp, \
                                     gumbel, uniform, logistic.')
            return x

        d = W.shape[0]
        if noise_scale is None:
            scale_vec = np.ones(d)
        elif np.isscalar(noise_scale):
            scale_vec = noise_scale * np.ones(d)
        else:
            if len(noise_scale) != d:
                raise ValueError('noise scale must be a scalar or has length d')
            scale_vec = noise_scale
        G_nx = nx.from_numpy_matrix(W, create_using=nx.DiGraph)

        if np.isinf(n):  # population risk for linear gauss SEM
            if sem_type == 'gauss':
                # make 1/d X'X = true cov
                X = np.sqrt(d) * np.diag(scale_vec) @ np.linalg.inv(np.eye(d) - W)
                return X
            else:
                raise ValueError('population risk not available')
        # empirical risk
        ordered_vertices = list(nx.topological_sort(G_nx))
        assert len(ordered_vertices) == d
        X = np.zeros([n, d])
        for j in ordered_vertices:
            parents = list(G_nx.predecessors(j))
            if len(parents) == 0 and input_range is not None and j in input_range.keys():
                X[:, j] = np.random.uniform(input_range[j][0], input_range[j][1], X.shape[0]) + bias[j]
            else:
                X[:, j] = _simulate_single_equation(X[:, parents], W[parents, j], scale_vec[j]) + bias[j]

        if self.cliques is not None:
            for c in self.cliques:
                X[:, c] += np.random.multivariate_normal(mean=np.zeros(len(c)),
                                                         cov=np.full((len(c), len(c)), noise_scale / 10))
        return X


def get_bermuda_params():
    import pandas as pd
    from scipy.io import loadmat
    from sklearn.linear_model import LinearRegression

    a = loadmat('SEM_data.mat')
    for key in ['__header__', '__version__', '__globals__', 'Site', 'Lat', 'Lon', 'Year', 'Month', 'Day']:
        a.pop(key)
    data = pd.DataFrame({key: value.reshape(-1) for key, value in a.items()})
    data = (data - data.mean()) / data.std()
    parents = {'Light': [], 'Chla': ['Nutrients_PC1', 'Light', 'Temp'], 'Temp': ['Light'], 'Sal': ['Temp'],
               'Omega': ['Sal', 'DIC', 'Temp', 'TA'], 'pHsw': ['Sal', 'DIC', 'Temp', 'TA'], 'DIC': ['Sal'],
               'TA': ['Sal'], 'CO2': ['Sal', 'TA', 'DIC', 'Temp'], 'Nutrients_PC1': [],
               'NEC': ['Nutrients_PC1', 'Light', 'pHsw', 'Omega', 'Chla', 'CO2', 'Temp']}
    vars_stage_1 = ['Light', 'Temp', 'Sal']
    vars_stage_2 = ['DIC', 'TA', 'Omega', 'Chla', 'Nutrients_PC1', 'pHsw', 'CO2']
    vars_stage_3 = ['NEC']
    vars = vars_stage_1 + vars_stage_2 + vars_stage_3
    n_vars_in_stages = [len(vars_stage_1), len(vars_stage_2), len(vars_stage_3)]
    n_nodes = len(vars)
    intervables = {var: (-1, 1) for var in range(3, 3 + 5)}  # {var: (data.iloc[:,var].min(), data.iloc[:,var].max()) for var in range(3, 3+5)}
    bias = pd.Series([0] * n_nodes, index=vars)
    input_range = {idx: (data[var].min(), data[var].max()) for idx, var in enumerate(vars)}
    weighted_matrix = pd.DataFrame(np.zeros((n_nodes, n_nodes)), index=vars, columns=vars)
    for var, pas in parents.items():
        if len(pas) > 0:
            lr = LinearRegression()
            rows = data[pas + [var]].notna().all(axis=1)
            X, y = data[pas][rows], data[var][rows]
            lr.fit(X, y)
            weighted_matrix.loc[pas, var] = lr.coef_
            bias[var] = lr.intercept_
    weighted_matrix = weighted_matrix.values
    bias = bias.values
    return n_vars_in_stages, intervables, weighted_matrix, bias, input_range


def get_traffic_params():
    import pandas as pd

    parents = {'weather': [],
               'n_user': [],
               'recommend': ['weather', 'n_user'],
               'congestion': ['weather', 'n_user', 'recommend'],
               'time_spent': ['weather', 'congestion'],
               'discount': ['weather', 'n_user', 'congestion'],
               'satisfaction': ['weather', 'congestion', 'time_spent', 'discount']}
    vars_stage_1 = ['weather', 'n_user']
    vars_stage_2 = ['recommend', 'congestion', 'time_spent', 'discount']
    vars_stage_3 = ['satisfaction']
    vars = vars_stage_1 + vars_stage_2 + vars_stage_3
    n_vars_in_stages = [len(vars_stage_1), len(vars_stage_2), len(vars_stage_3)]
    n_nodes = len(vars)
    intervables = {var: (-2, 2) for var in [2, 5]}
    bias = pd.Series([0, 0, 0, 0, 0, 0, 0.5], index=vars)
    input_range = {idx: (-1, 1) for idx, var in enumerate(vars)}
    weighted_matrix = pd.DataFrame(np.zeros((n_nodes, n_nodes)), index=vars, columns=vars)

    v = 'recommend'
    for pa, w in zip(parents[v], [1, -1]):
        weighted_matrix.loc[pa, v] = w
    v = 'congestion'
    for pa, w in zip(parents[v], [-1, 2, 3]):
        weighted_matrix.loc[pa, v] = w
    v = 'time_spent'
    for pa, w in zip(parents[v], [-1, 4]):
        weighted_matrix.loc[pa, v] = w
    v = 'discount'
    for pa, w in zip(parents[v], [-1, -0.5, 0.3]):
        weighted_matrix.loc[pa, v] = w
    v = 'satisfaction'
    c = 60
    for pa, w in zip(parents[v], [0.2 / c, -3 / c, -1 / c, 0.2 / c]):
        weighted_matrix.loc[pa, v] = w
    weighted_matrix = weighted_matrix.values
    bias = bias.values
    cliques = [(2, 3)]  # ('recommend','congestion')

    return n_vars_in_stages, intervables, weighted_matrix, bias, input_range, cliques
