import os
from copy import deepcopy
from functools import partial

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from env import Environment
from linear import LinearModel
from utils import lr_predict, flow_predict, generate_noise, chebyshev_center, center_mae, center_mse, ball_huber, \
    ball_insensitive


class RehearsalGraph:
    def __init__(self, x_dim, ordered_children_parents_pairs, free_vars, models, device, predict_func):
        self.x_dim = x_dim
        self.children_parents_pairs = ordered_children_parents_pairs
        self.models = models
        self.alter_vars = None
        self.device = device
        self.predict_func = predict_func
        self.free_vars = free_vars

        for children, parents in ordered_children_parents_pairs + [((var,), ()) for var in self.free_vars]:
            for param in models[children][parents].parameters():
                param.requires_grad = False

    def set_alterable_vars(self, alter_vars):
        assert len(alter_vars) > 0
        for children, parents in self.children_parents_pairs:
            if children[0] in alter_vars:
                assert all([child in alter_vars for child in children])
        self.alter_vars = alter_vars

    def forward(self, data, noises):
        assert self.alter_vars is not None
        for children, parents in self.children_parents_pairs + [((var,), ()) for var in self.free_vars]:
            if children[0] not in self.alter_vars and children[0] >= self.x_dim:
                model = self.models[children][parents]
                noise = noises[children][parents]
                model.eval()
                data[:, children] = self.predict_func(model, data[:, parents], noise, len(children))
        return data


class Rehearsal:
    def __init__(self, n_vars, x_dim, y_dim, ordered_children_parents_pairs, free_vars, models, noise_covs,
                 srm_type='linear', device='cuda'):
        self.n_vars = n_vars
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.children_parents_pairs = ordered_children_parents_pairs
        self.models = models
        self.noise_covs = noise_covs
        self.device = device
        self.free_vars = free_vars
        if srm_type == 'linear':
            predict_func = lr_predict
        elif srm_type == 'flow':
            predict_func = flow_predict
        else:
            raise ValueError(f'srm_type {srm_type} not supported')
        self.graph = RehearsalGraph(x_dim, ordered_children_parents_pairs, free_vars, models, device,
                                    predict_func=predict_func)

    def find_alter_values(self, observed_x, alter_vars, alter_ranges, n_samples, M, d, loss='center_mse', learning_rate=0.01,
                          seed=2024, epochs=1000, patience=20, verbose=True, scaler=None):
        assert all([0 <= var < self.n_vars for var in alter_vars])

        if alter_ranges is None:  # (low, high) or [(low0, high0), (low1, high1)]
            alter_ranges = [[1, 1]] * len(alter_vars)
        elif isinstance(alter_ranges[0], (int, float)):
            alter_ranges = [alter_ranges] * len(alter_vars)
        else:
            assert len(alter_ranges) == len(alter_vars)
        alter_ranges = deepcopy(alter_ranges)
        if scaler is not None:
            tmp = np.zeros((1, self.n_vars))
            tmp[0, :len(observed_x)] = observed_x
            tmp = scaler.transform(tmp)
            observed_x = tmp[0, :len(observed_x)]
            y_dim = M.shape[1]
            u = scaler.mean_[-y_dim:]
            s = scaler.scale_[-y_dim:]
            new_d = d - np.dot(M, u)
            new_M = np.dot(M, np.diag(s))
            M, d = new_M, new_d
            for i, var in enumerate(alter_vars):
                u = scaler.mean_[var]
                s = scaler.scale_[var]
                alter_ranges[i][0] = (alter_ranges[i][0] - u) / s
                alter_ranges[i][1] = (alter_ranges[i][1] - u) / s

        loss_func = self._get_loss_func(loss, M, d)

        self.graph.set_alterable_vars(alter_vars)

        rng = np.random.default_rng(seed)

        observed_x = torch.tensor(observed_x)
        alter_data = torch.tensor(rng.normal(0, 0.1, len(alter_vars)), dtype=torch.float32,
                                  requires_grad=True, device=self.device)

        def scaling(v, low, high):
            if low == high == 1:
                return v
            return F.tanh(v) / 2 * (high - low) + (high + low) / 2

        noises = generate_noise(self.children_parents_pairs, self.free_vars, self.noise_covs, n_samples, seed,
                                self.device)
        optimizer = torch.optim.Adam([alter_data], lr=learning_rate)
        min_loss = np.inf
        patience_counter = patience
        opt_alterations = None
        opt_est_prob = -1
        pbar = tqdm(range(epochs)) if verbose else range(epochs)
        for epoch in pbar:
            optimizer.zero_grad()

            data = torch.zeros(n_samples, self.n_vars, device=self.device)
            alterations = torch.zeros(len(alter_vars), device=self.device)
            for i in range(len(alter_vars)):
                alterations[i] = scaling(alter_data[i], alter_ranges[i][0], alter_ranges[i][1])

            data[:, :self.x_dim] = observed_x
            data[:, alter_vars] = alterations
            data = self.graph.forward(data, noises)
            loss = loss_func(data[:, -self.y_dim:])
            loss.backward()
            optimizer.step()

            if min_loss > loss.item():
                min_loss = loss.item()
                est_prob = np.mean((np.dot(data.detach().cpu().numpy()[:, -self.y_dim:], M.T) <= d).all(axis=1))
                if est_prob > opt_est_prob:
                    opt_alterations = alterations.detach().cpu().numpy()
                    opt_est_prob = est_prob
                patience_counter = patience
            else:
                patience_counter -= 1
            if patience_counter == 0:
                break
            if verbose:
                pbar.set_description(f'Epoch {epoch}, loss {loss.item():.5f}')
        if scaler is not None:
            u = scaler.mean_[alter_vars]
            s = scaler.scale_[alter_vars]
            opt_alterations = opt_alterations * s + u
        return opt_alterations, opt_est_prob

    def find_alter_values_iter(self, observed_x, alter_vars, alter_ranges, n_samples, M, d, loss='center_mse', learning_rate=0.01,
                               seed=2024, epochs=1000, patience=20, verbose=True, scaler=None):
        assert all([0 <= var < self.n_vars for var in alter_vars])

        if alter_ranges is None:
            alter_ranges = [[1, 1]] * len(alter_vars)
        elif isinstance(alter_ranges[0], (int, float)):
            alter_ranges = [alter_ranges] * len(alter_vars)
        else:
            assert len(alter_ranges) == len(alter_vars)
        alter_ranges = deepcopy(alter_ranges)
        if scaler is not None:
            tmp = np.zeros((1, self.n_vars))
            tmp[0, :len(observed_x)] = observed_x
            tmp = scaler.transform(tmp)
            observed_x = tmp[0, :len(observed_x)]
            y_dim = M.shape[1]
            u = scaler.mean_[-y_dim:]
            s = scaler.scale_[-y_dim:]
            new_d = d - np.dot(M, u)
            new_M = np.dot(M, np.diag(s))
            M, d = new_M, new_d
            for i, var in enumerate(alter_vars):
                u = scaler.mean_[var]
                s = scaler.scale_[var]
                alter_ranges[i][0] = (alter_ranges[i][0] - u) / s
                alter_ranges[i][1] = (alter_ranges[i][1] - u) / s

        loss_func = self._get_loss_func(loss, M, d)

        self.graph.set_alterable_vars(alter_vars)

        rng = np.random.default_rng(seed)

        observed_x = torch.tensor(observed_x)
        alter_data = torch.tensor(rng.normal(0, 0.1, len(alter_vars)), dtype=torch.float32,
                                  requires_grad=True, device=self.device)

        def scaling(v, low, high):
            if low == high == 1:
                return v
            return F.tanh(v) / 2 * (high - low) + (high + low) / 2

        noises = generate_noise(self.children_parents_pairs, self.free_vars, self.noise_covs, n_samples, seed,
                                self.device)
        optimizer = torch.optim.Adam([alter_data], lr=learning_rate)
        min_loss = np.inf
        patience_counter = patience
        opt_alterations = None
        opt_est_prob = -1
        pbar = tqdm(range(epochs)) if verbose else range(epochs)
        for epoch in pbar:
            optimizer.zero_grad()

            data = torch.zeros(n_samples, self.n_vars, device=self.device)
            alterations = torch.zeros(len(alter_vars), device=self.device)
            for i in range(len(alter_vars)):
                alterations[i] = scaling(alter_data[i], alter_ranges[i][0], alter_ranges[i][1])

            data[:, :self.x_dim] = observed_x
            data[:, alter_vars] = alterations
            data = self.graph.forward(data, noises)
            loss = loss_func(data[:, -self.y_dim:])
            loss.backward()
            optimizer.step()

            yield alterations.detach().cpu().numpy(), loss.item()

            if min_loss > loss.item():
                min_loss = loss.item()
                est_prob = np.mean((np.dot(data.detach().cpu().numpy()[:, -self.y_dim:], M.T) <= d).all(axis=1))
                if est_prob > opt_est_prob:
                    opt_alterations = alterations.detach().cpu().numpy()
                    opt_est_prob = est_prob
                patience_counter = patience
            else:
                patience_counter -= 1
            if patience_counter == 0:
                break
            if verbose:
                pbar.set_description(f'Epoch {epoch}, loss {loss.item():.5f}')
        if scaler is not None:
            u = scaler.mean_[alter_vars]
            s = scaler.scale_[alter_vars]
            opt_alterations = opt_alterations * s + u
        return opt_alterations, min_loss

    def _get_loss_func(self, loss, M, d):
        center, radius = chebyshev_center(M, d)
        center, radius = torch.tensor(center, dtype=torch.float32, device=self.device), torch.tensor(radius,
                                                                                                     dtype=torch.float32,
                                                                                                     device=self.device)
        M, d = torch.tensor(M, dtype=torch.float32, device=self.device), torch.tensor(d, dtype=torch.float32,
                                                                                      device=self.device)
        if loss == 'center_mse':
            loss_func = partial(center_mse, center=center)
        elif loss == 'center_mae':
            loss_func = partial(center_mae, center=center)
        elif loss == 'ball_huber':
            loss_func = partial(ball_huber, center=center, radius=radius)
        elif loss == 'ball_insensitive':
            loss_func = partial(ball_insensitive, center=center, radius=radius)
        else:
            raise ValueError(f'loss {loss} not supported')
        return loss_func


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '7'

    x_dim = 2
    z_dim = 4
    y_dim = 1
    n_vars = x_dim + z_dim + y_dim
    children_parents_pairs = [([2], [0]),
                              ([1], [0]),
                              ([4, 5], [2, 3]),
                              ([3], [1, 2]),
                              ([6], [2, 5])]

    env = Environment(x_dim, z_dim, y_dim, children_parents_pairs, seed=2024)
    obs_data = env.get_observations(1000)

    rf = LinearModel(n_vars, [2, 3, 4, 5], env.get_ordered_children_parents(), device='cuda')
    rf.fit(obs_data)

    x = env.observe_x()

    device = 'cuda'
    rh = Rehearsal(n_vars, x_dim, y_dim, env.get_ordered_children_parents(), rf.get_models(), rf.get_noise_cov(),
                   'linear', device)
    rh.find_alter_values(x, [2, 3, 4, 5], 10)
