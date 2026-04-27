import numpy as np

from utils import generate_noise


class Baseline:
    def __init__(self, n_vars, x_dim, y_dim, ordered_children_parents_pairs, free_vars, models, noise_covs):
        self.n_vars = n_vars
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.free_vars = free_vars
        self.children_parents_pairs = ordered_children_parents_pairs
        self.models = models
        self.noise_covs = noise_covs

    def find_best_alter_range(self, observed_x, alter_var, alter_range, M, d, n_samples, seed=2024):
        noises = generate_noise(self.children_parents_pairs, self.free_vars, self.noise_covs, n_samples, seed, 'cpu')
        coef = np.zeros((n_samples, self.n_vars))
        bias = np.zeros((n_samples, self.n_vars))

        coef[:, alter_var] = 1
        bias[:, :self.x_dim] = observed_x
        for var in self.free_vars:
            if var >= self.x_dim and var != alter_var:
                bias[:, var] = noises[(var,)][()].numpy().squeeze()

        for children, parents in self.children_parents_pairs:
            if alter_var in children:
                assert len(children) == 1
            elif children[0] >= self.x_dim:
                coef[:, children] = np.dot(coef[:, parents],
                                           self.models[children][parents].linear.weight.detach().cpu().numpy().T)
                bias[:, children] = np.dot(bias[:, parents],
                                           self.models[children][parents].linear.weight.detach().cpu().numpy().T) + \
                                    self.models[children][parents].linear.bias.detach().cpu().numpy() + \
                                    noises[children][parents].numpy()

        x_coef = np.dot(coef[:, -self.y_dim:], M.T)
        b_coef = np.dot(bias[:, -self.y_dim:], M.T)
        rhs = d - b_coef
        sgn = np.sign(x_coef)
        upper = np.where(sgn > 0, rhs / (x_coef + 1e-6), np.inf)
        upper = np.where((sgn == 0) & (rhs < 0), -np.inf, upper)
        upper = upper.min(axis=1)
        lower = np.where(sgn < 0, rhs / (x_coef - 1e-6), -np.inf)
        lower = np.where((sgn == 0) & (rhs < 0), np.inf, lower)
        lower = lower.max(axis=1)

        if alter_range is not None:
            lower = np.maximum(lower, alter_range[0])
            upper = np.minimum(upper, alter_range[1])

        low, high, n_success = self._find_most_frequent_interval(lower, upper)
        return low, high, n_success/n_samples

    def _find_most_frequent_interval(self, lower, upper):
        valid = lower < upper
        lower, upper = lower[valid], upper[valid]
        if len(lower) + len(upper) == 0:
            return 0, 0, 0
        arr = sorted([(v, True) for v in lower] + [(v, False) for v in upper])
        max_cnt = 0
        start_v, end_v = None, None
        cur_cnt = 1
        cur_start_v = arr[0][0]
        for v, flag in arr[1:]:
            if flag:
                cur_start_v = v
                cur_cnt += 1
            else:
                if cur_cnt > max_cnt:
                    max_cnt = cur_cnt
                    start_v = cur_start_v
                    end_v = v
                cur_cnt -= 1
        return start_v, end_v, max_cnt
