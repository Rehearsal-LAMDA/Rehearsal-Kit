import numpy as np
import pandas as pd


class Traffic:
    def __init__(self, noise_scale):

        parents = {
            'recommend': ['weather', 'n_user'],
            'congestion': ['weather', 'n_user', 'recommend'],
            'time_spent': ['weather', 'congestion'],
            'discount': ['weather', 'n_user', 'congestion'],
            'satisfaction': ['weather', 'congestion', 'time_spent', 'discount']}
        vars_stage_1 = ['weather', 'n_user']
        vars_stage_2 = ['recommend', 'congestion', 'time_spent', 'discount']
        vars_stage_3 = ['satisfaction']
        vars = vars_stage_1 + vars_stage_2 + vars_stage_3
        intervables = {var: (-2, 2) for var in [2, 5]}
        bias = pd.Series([0, 0, 0, 0, 0, 0, 0.5], index=vars)
        self.input_range = {idx: (-1, 1) for idx, var in enumerate(vars)}
        self.alterable_vars = sorted(intervables.keys())
        self.x_dim = len(vars_stage_1)
        self.z_dim = len(vars_stage_2)
        self.y_dim = len(vars_stage_3)
        self.n_vars = len(vars)
        var2idx = {var: i for i, var in enumerate(vars)}
        self.pairs = [((var2idx[var],), [var2idx[p] for p in parents[var]]) for var in parents.keys()]

        weighted_matrix = pd.DataFrame(np.zeros((self.n_vars, self.n_vars)), index=vars, columns=vars)

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
        for pa, w in zip(parents[v], [0.2 / c, -5 / c, -1 / c, 5 / c]):
            weighted_matrix.loc[pa, v] = w
        weighted_matrix = weighted_matrix.values
        bias = bias.values

        self.params = {}
        for children, parents in self.pairs:
            var = children[0]
            self.params[tuple(children)] = {tuple(parents): {'coef': weighted_matrix[weighted_matrix[:, var] != 0, var].reshape(-1, 1),
                                                             'bias': bias[var],
                                                             'noise_cov': np.array([[noise_scale]])}}

    def get_input_range(self):
        return self.input_range.copy()

    def get_alterable_vars(self):
        return self.alterable_vars

    def get_alter_ranges(self):
        return [-1, 1]

    def get_children_parents_pairs(self):
        return self.pairs

    def get_srm_params(self):
        return self.params

    def get_info(self):
        return self.x_dim, self.z_dim, self.y_dim, self.n_vars

    def get_desired_region(self):
        M = np.array([[1], [-1]])
        d = np.array([1.5, -0.9])
        return M, d
