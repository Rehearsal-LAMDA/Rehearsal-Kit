import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.linear_model import LinearRegression


class Bermuda:
    def __init__(self, path, noise_scale):
        a = loadmat(path)
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
        var2idx = {}
        for i, var in enumerate(vars):
            var2idx[var] = i

        self.alterable_vars = [3, 4, 5, 6, 7]
        self.params = {}
        self.x_dim = len(vars_stage_1)
        self.z_dim = len(vars_stage_2)
        self.y_dim = len(vars_stage_3)
        self.n_vars = len(vars)
        self.pairs = []
        self.input_range = {idx: (data[var].min(), data[var].max()) for idx, var in enumerate(vars)}
        self.data = data
        for var, pas in parents.items():
            if len(pas) > 0:
                p = tuple(sorted([var2idx[p] for p in pas]))
                self.pairs.append(((var2idx[var],), p))
                lr = LinearRegression()
                rows = data[pas + [var]].notna().all(axis=1)
                X, y = data[[vars[i] for i in p]][rows], data[var][rows]
                lr.fit(X, y)
                self.params[(var2idx[var],)] = {p: {'coef': lr.coef_.reshape(-1, 1), 'bias': np.zeros(1),
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
        d = np.array([2, -0.5])
        return M, d
