import os
from collections import defaultdict
from itertools import chain
from time import time

import FrEIA.framework as Ff
import FrEIA.modules as Fm
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from env import Environment
from utils import get_dataloader


class Flow:
    def __init__(self, n_vars, ordered_children_parents_pairs, device='cuda', seed=2024):
        torch.manual_seed(seed)
        self.n_vars = n_vars
        self.device = device
        self.children_parents_pairs = [(tuple(children), tuple(parents)) for children, parents in
                                       ordered_children_parents_pairs]
        self.free_vars = sorted(
            set(range(n_vars)) - set(chain(*[children for children, _ in ordered_children_parents_pairs])))
        self.flows = defaultdict(dict)
        self.scaler = StandardScaler()

    def fit(self, data, learning_rate=0.001, epochs=100, batch_size=512, patience=20):
        data = data.copy()
        data = self.scaler.fit_transform(data)

        for children, parents in self.children_parents_pairs[::-1]:
            print(children, parents)
            self.flows[children][parents] = self._fit_flow(data[:, parents], data[:, children],
                                                           learning_rate, epochs, batch_size, patience)
        for var in self.free_vars:
            self.flows[(var,)][()] = self._fit_flow(None, data[:, (var,)], learning_rate, epochs, batch_size, patience)

    def get_scaler(self):
        return self.scaler

    def get_models(self):
        ret = {children: {parents: self.flows[children][parents]} for children, parents in
               self.children_parents_pairs + [((var,), ()) for var in self.free_vars]}
        return ret

    def get_noise_cov(self):
        ret = {children: {parents: torch.eye((2 if len(children) == 1 else len(children)), dtype=torch.float32)}
               for children, parents in self.children_parents_pairs + [((var,), ()) for var in self.free_vars]}
        return ret

    def _fit_flow(self, x, y, learning_rate, epochs, batch_size, patience=20):
        tmstamp = time()

        def subnet_fc(dims_in, dims_out):
            return nn.Sequential(nn.Linear(dims_in, 256), nn.ReLU(),
                                 nn.Linear(256, dims_out))

        def get_inn(in_dim, cond_dim):
            inn = Ff.SequenceINN(in_dim)
            for k in range(16):
                if cond_dim > 0:
                    inn.append(Fm.AllInOneBlock, cond=0, cond_shape=(cond_dim,), subnet_constructor=subnet_fc)
                else:
                    inn.append(Fm.AllInOneBlock, subnet_constructor=subnet_fc)
            return inn

        if y.shape[1] == 1:
            y = np.hstack([y, y])
        in_dim = y.shape[1]
        if x is None:
            cond_dim = 0
            x = np.ones_like(y)
        else:
            cond_dim = x.shape[1]

        tr_x, val_x, tr_y, val_y = train_test_split(x, y, test_size=0.3)
        tr_loader, val_loader = get_dataloader(tr_x, tr_y, batch_size), get_dataloader(val_x, val_y, batch_size,
                                                                                       shuffle=False)
        inn = get_inn(in_dim, cond_dim).to(self.device)
        optimizer = torch.optim.Adam(inn.parameters(), lr=learning_rate)

        pbar = tqdm(range(epochs))
        min_loss = float('inf')

        for i in pbar:
            losses = []
            for batch_x, batch_y in tr_loader:
                inn.train()
                optimizer.zero_grad()
                z, log_jac_det = inn(batch_y, c=[batch_x]) if cond_dim else inn(batch_y)
                loss = 0.5 * torch.sum(z ** 2, 1) - log_jac_det
                loss = loss.mean() / in_dim
                losses.append(loss.item())
                loss.backward()
                optimizer.step()
            loss = sum(losses) / len(losses)

            with torch.no_grad():
                inn.eval()
                val_losses = []
                for batch_x, batch_y in val_loader:
                    z, log_jac_det = inn(batch_y, c=[batch_x]) if cond_dim else inn(batch_y)
                    val_loss = 0.5 * torch.sum(z ** 2, 1) - log_jac_det
                    val_loss = val_loss.mean() / in_dim
                    val_losses.append(val_loss.item())
                val_loss = sum(val_losses) / len(val_losses)
                if val_loss < min_loss:
                    min_loss = val_loss
                    pati = patience
                    torch.save(inn.state_dict(), f'model_{tmstamp}.pth')
                else:
                    pati -= 1
                if pati == 0:
                    break

            pbar.set_description(f'tr_loss: {loss:.5f}, val_loss: {val_loss:.5f}, best_loss: {min_loss:.5f}')
        inn.load_state_dict(torch.load(f'model_{tmstamp}.pth'))
        os.remove(f'model_{tmstamp}.pth')
        return inn

    @torch.no_grad()
    def _sample_from_flow(self, inn, x, out_dim, n_samples):
        inn.eval()
        in_dim = inn.dims_in[0][0]
        z = torch.randn((n_samples, in_dim), dtype=torch.float32, device=self.device)
        if x is None:
            samples, _ = inn(z, rev=True)
        else:
            x = torch.tensor(x, dtype=torch.float32, device=self.device)
            samples, _ = inn(z, c=[x], rev=True)
        if out_dim != in_dim:
            samples = samples[:, :out_dim]
        return samples.cpu().numpy()

    def simulate_alteration_samples(self, alter_vars, values, n_samples):
        data = np.zeros((n_samples, self.n_vars))
        for var in self.free_vars:
            data[:, var] = self._sample_from_flow(self.flows[(var,)][()], None, 1, n_samples).squeeze()
        data[:, alter_vars] = np.array(values)

        for children, parents in self.children_parents_pairs:
            if children[0] in alter_vars:
                assert all([child in alter_vars for child in children])
            else:
                data[:, children] = self._sample_from_flow(self.flows[children][parents], data[:, parents],
                                                           len(children), n_samples)
        data = self.scaler.inverse_transform(data)
        return data
