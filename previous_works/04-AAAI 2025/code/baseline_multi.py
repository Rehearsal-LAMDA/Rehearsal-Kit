import numpy as np
import scipy

from utils import generate_noise


class MultivariateBaseline:
    def __init__(self, n_vars, x_dim, y_dim, ordered_children_parents_pairs, free_vars, models, noise_covs):
        self.n_vars = n_vars
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.free_vars = free_vars
        self.children_parents_pairs = ordered_children_parents_pairs
        self.models = models
        self.noise_covs = noise_covs

    def find_best_alter_values(self, observed_x, alter_vars, alter_ranges, M, d, n_samples, seed=2024, time_limit=600):
        if alter_ranges is not None and isinstance(alter_ranges[0], (int, float)):
            alter_ranges = [alter_ranges] * len(alter_vars)
        assert alter_ranges is None or len(alter_ranges) == len(alter_vars)

        noises = generate_noise(self.children_parents_pairs, self.free_vars, self.noise_covs, n_samples, seed, 'cpu')
        coef = np.diag([1.0 if var in alter_vars else 0 for var in range(self.n_vars)])
        bias = np.zeros((n_samples, self.n_vars))

        bias[:, :self.x_dim] = observed_x

        for var in self.free_vars:
            if var >= self.x_dim and var not in alter_vars:
                bias[:, var] = noises[(var,)][()].numpy().squeeze()

        for children, parents in self.children_parents_pairs:
            if children[0] in alter_vars:
                assert all([c in alter_vars for c in children])
            elif children[0] >= self.x_dim:
                coef[:, children] = np.dot(coef[:, parents],
                                           self.models[children][parents].linear.weight.detach().cpu().numpy().T)
                bias[:, children] = np.dot(bias[:, parents],
                                           self.models[children][parents].linear.weight.detach().cpu().numpy().T) + \
                                    self.models[children][parents].linear.bias.detach().cpu().numpy() + \
                                    noises[children][parents].numpy()

        alpha = 1e9
        z_coef = coef[alter_vars, -self.y_dim:]
        MB = np.dot(M, z_coef.T)
        A = np.hstack((np.zeros((d.shape[0] * n_samples, n_samples)), np.tile(MB, (n_samples, 1))))
        r = 0
        for i in range(n_samples):
            A[r: r + d.shape[0], i] = alpha
            r += d.shape[0]

        b = bias[:, -self.y_dim:]
        ub = np.hstack(-np.dot(b, M.T) + d + alpha)

        constraints = scipy.optimize.LinearConstraint(A, ub=ub)
        integrality = np.ones(n_samples + len(alter_vars))
        integrality[n_samples:] = 0
        bounds_lb = np.zeros_like(integrality)
        bounds_ub = np.ones_like(integrality)
        if alter_ranges is None:
            bounds_lb[n_samples:] = -np.inf
            bounds_ub[n_samples:] = np.inf
        else:
            bounds_lb[n_samples:] = [l for l, u in alter_ranges]
            bounds_ub[n_samples:] = [u for l, u in alter_ranges]
        bounds = scipy.optimize.Bounds(bounds_lb, bounds_ub)
        c = -np.ones_like(integrality)
        c[n_samples:] = 0

        res = scipy.optimize.milp(c, integrality=integrality, bounds=bounds, constraints=constraints, options={'time_limit': time_limit})
        prob = 0
        alter_values = np.zeros(len(alter_vars))
        if res.success:
            prob = -res.fun / n_samples
            alter_values = res.x[n_samples:]

        return alter_values, prob
