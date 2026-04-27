import os
from collections import defaultdict
from itertools import chain

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LinearRegression

from env import Environment


class LinearNet(nn.Module):
    def __init__(self, in_dim, out_dim, device='cuda'):
        super(LinearNet, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.device = device
        if in_dim > 0:
            self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        if self.in_dim > 0:
            out = self.linear(x)
        else:
            batch_size = x.shape[0]
            out = torch.zeros(batch_size, self.out_dim, dtype=torch.float32, device=self.device)
        return out


class LinearModel:
    def __init__(self, n_vars, ordered_children_parents_pairs, device='cuda', seed=2024):
        torch.manual_seed(seed)
        self.n_vars = n_vars
        self.device = device
        self.children_parents_pairs = [(tuple(children), tuple(parents)) for children, parents in
                                       ordered_children_parents_pairs]
        self.free_vars = sorted(
            set(range(n_vars)) - set(chain(*[children for children, _ in ordered_children_parents_pairs])))
        self.lrs = defaultdict(dict)

    def fit(self, data):
        for var in self.free_vars:
            self.lrs[(var,)][()] = self.linear(None, data[:, (var,)])
        for children, parents in self.children_parents_pairs:
            self.lrs[children][parents] = self.linear(data[:, parents], data[:, children])

    def get_models(self):
        ret = {children: {parents: self.lrs[children][parents]['model']} for children, parents in
               self.children_parents_pairs + [((var,), ()) for var in self.free_vars]}
        return ret

    def get_noise_cov(self):
        ret = {children: {parents: self.lrs[children][parents]['noise_cov']} for children, parents in
               self.children_parents_pairs + [((var,), ()) for var in self.free_vars]}
        return ret

    def linear(self, x, y):
        in_dim = 0 if x is None else x.shape[1]
        out_dim = y.shape[1]
        lr = LinearNet(in_dim, out_dim, device=self.device).to(self.device)
        if in_dim == 0:
            noise_cov = torch.cov(torch.tensor(y, dtype=torch.float32).T)
            if out_dim == 1:
                noise_cov = noise_cov.reshape(1, 1)
            return {'model': lr, 'noise_cov': noise_cov}

        coefs = []
        biases = []
        for i in range(out_dim):
            l = LinearRegression()
            l.fit(x, y[:, i])
            coefs.append(l.coef_)
            biases.append(l.intercept_)
        weight = torch.tensor(np.vstack(coefs), dtype=torch.float32, device=self.device)
        bias = torch.tensor(np.hstack(biases), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            lr.linear.weight.copy_(weight)
            lr.linear.bias.copy_(bias)
        y_pred = lr(torch.tensor(x, dtype=torch.float32, device=self.device)).cpu()
        noise_cov = torch.cov((torch.tensor(y, dtype=torch.float32) - y_pred).T)
        if out_dim == 1:
            noise_cov = noise_cov.reshape(1, 1)
        return {'model': lr, 'noise_cov': noise_cov}

    @torch.no_grad()
    def _sample_from_lr(self, lr, x, noise_cov, n_samples):
        lr.eval()
        y = lr(torch.tensor(x, dtype=torch.float32, device=self.device))
        noise = (
            torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(y.shape[1], dtype=torch.float32),
                                                                       noise_cov).
            sample((n_samples,)).to(self.device))
        samples = y + noise
        return samples.cpu().numpy()

    def simulate_alteration_samples(self, alter_vars, values, n_samples):
        data = np.zeros((n_samples, self.n_vars))
        for var in self.free_vars:
            data[:, var] = self._sample_from_lr(self.lrs[(var,)][()]['model'], np.zeros((data.shape[0], 1)),
                                                self.lrs[(var,)][()]['noise_cov'],
                                                n_samples).squeeze()
        data[:, alter_vars] = np.array(values)

        for children, parents in self.children_parents_pairs:
            if children[0] in alter_vars:
                assert all([child in alter_vars for child in children])
            else:
                data[:, children] = self._sample_from_lr(self.lrs[children][parents]['model'], data[:, parents],
                                                         self.lrs[children][parents]['noise_cov'], n_samples)
        return data

