import argparse
import random

import numpy as np
import scipy
import torch
from torch.utils.data import TensorDataset, DataLoader


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str)
    parser.add_argument('--cuda', type=str)
    parser.add_argument('--lr', type=float, default=0.1)
    args = parser.parse_args()
    return args


def get_dataloader(x, y, batch_size=512, shuffle=True, device='cuda'):
    dataset = TensorDataset(torch.tensor(x, dtype=torch.float32, device=device),
                            torch.tensor(y, dtype=torch.float32, device=device))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader


def flow_predict(inn, x, noise, out_dim):
    if x is None or x.shape[1] == 0:
        samples, _ = inn(noise, rev=True)
    else:
        samples, _ = inn(noise, c=[x], rev=True)
    if out_dim != noise.shape[1]:
        samples = samples[:, :out_dim]
    return samples


def lr_predict(lr, x, noise, out_dim):
    y = lr(x)
    samples = y + noise
    return samples


def generate_noise(children_parents_pairs, free_vars, noise_covs, n_samples, seed=2024, device='cuda'):
    rng = np.random.default_rng(seed)
    noises = {}
    for children, parents in children_parents_pairs + [((var,), ()) for var in free_vars]:
        noise_cov = noise_covs[children][parents]
        noise = rng.multivariate_normal(np.zeros(noise_cov.shape[0]), noise_cov.detach().cpu().numpy(),
                                        n_samples).reshape(n_samples, -1)
        noise = torch.tensor(noise, dtype=torch.float32, device=device)
        noises[children] = {parents: noise}
    return noises


def chebyshev_center(M, d):
    row_norm = np.sqrt(np.sum(M ** 2, axis=1))
    matA = np.vstack((row_norm, M.T)).T
    vecB = d
    vecC = np.zeros(M.shape[1] + 1)
    vecC[0] = -1
    res = scipy.optimize.linprog(vecC, A_ub=matA, b_ub=vecB, bounds=(None, None), method='highs')
    vecZ = res.x
    r, c = vecZ[0], vecZ[1:]
    return c, r


def center_mse(y_hat, center):
    return ((y_hat - center) ** 2).sum()


def center_mae(y_hat, center):
    return (torch.abs(y_hat - center)).sum()


def ball_huber(y_hat, center, radius):
    dis = torch.sqrt(((y_hat - center) ** 2).sum(dim=-1))
    return torch.where(dis <= radius, dis ** 2, 2 * dis * radius - radius ** 2).sum()


def ball_insensitive(y_hat, center, radius):
    dis = torch.sqrt(((y_hat - center) ** 2).sum(dim=-1))
    return torch.where(dis <= radius, 0, dis - radius).sum()


def transform_standard_range(y, standard_range):
    low, high = standard_range
    mu = np.mean(y, axis=0)
    std = np.std(y, axis=0)
    return [[l, h] for l, h in zip(low * std + mu, high * std + mu)]


def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.random.manual_seed(seed)
