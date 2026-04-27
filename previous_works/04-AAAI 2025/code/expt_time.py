import os
from itertools import chain
from time import time

import numpy as np
import torch
from tqdm import tqdm

from baseline_multi import MultivariateBaseline
from env import Environment
from flow import Flow
from linear import LinearModel
from rehearsal import Rehearsal
from synthetic import MLP1, MLP2, MLP3, MLP4
from utils import setup_seed, get_args, transform_standard_range

if __name__ == '__main__':
    setup_seed(2024)

    args = get_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    n_repeats = 10
    n_observations = 1000
    n_env_est_samples = 10000
    learning_rate = 0.01
    epochs = 200
    time_limit = 1200

    dataset = args.data
    dsmap = {'mlp1': MLP1(),
             'mlp2': MLP2(),
             'mlp3': MLP3(),
             'mlp4': MLP4(),
             }
    ds = dsmap[dataset]
    print(dataset)

    x_dim, z_dim, y_dim, n_vars = ds.get_info()
    children_parents_pairs = ds.get_children_parents_pairs()
    alterable_vars = ds.get_alterable_vars()
    free_vars = [var for var in range(n_vars) if var not in chain(*[children for children, _ in children_parents_pairs])]
    standard_alter_ranges = ds.get_standard_alter_ranges()
    free_var_scale = ds.get_free_var_scale()
    env_seed = ds.get_env_seed()

    noise_scale = ds.get_noise()
    env = Environment(x_dim, z_dim, y_dim, children_parents_pairs,
                      alterable_vars=ds.get_alterable_vars(),
                      noise=noise_scale,
                      free_var_scale=free_var_scale,
                      mode='mlp',
                      seed=env_seed)
    M, d = ds.get_desired_region()
    env.set_desired_region(M, d)
    obs_data = env.get_observations(n_observations)
    model = Flow(n_vars, env.get_ordered_children_parents(), device=device)
    model.fit(obs_data, learning_rate=1e-3, epochs=1000, batch_size=128)
    rh = Rehearsal(n_vars, x_dim, y_dim, env.get_ordered_children_parents(), env.get_free_vars(),
                   model.get_models(),
                   model.get_noise_cov(),
                   'flow',
                   device)
    linear_model = LinearModel(n_vars, env.get_ordered_children_parents(), device=device)
    linear_model.fit(obs_data)
    ba = MultivariateBaseline(n_vars, x_dim, y_dim, env.get_ordered_children_parents(), env.get_free_vars(),
                              linear_model.get_models(),
                              linear_model.get_noise_cov())

    full_records = {}
    for n_method_est_samples in [10, 50, 100, 200, 400, 600, 800, 1000, 1500, 2000, 3000, 4000, 5000]:
        records = {}
        for loss_name in ['baseline', 'center_mae']:
            records[f'{loss_name}_prob'] = []
            records[f'{loss_name}_time'] = []
            records[f'{loss_name}_est_prob'] = []
        full_records[n_method_est_samples] = records

        noise_scale = ds.get_noise()
        env = Environment(x_dim, z_dim, y_dim, children_parents_pairs,
                          alterable_vars=ds.get_alterable_vars(),
                          noise=noise_scale,
                          free_var_scale=free_var_scale,
                          mode='mlp',
                          seed=env_seed)
        M, d = ds.get_desired_region()
        env.set_desired_region(M, d)

        obs_data = env.get_observations(n_observations)
        print(obs_data.mean(), obs_data.std())
        alter_ranges = transform_standard_range(obs_data[:, alterable_vars], standard_alter_ranges)

        pbar = tqdm(range(n_repeats))
        for seed in pbar:
            x = env.observe_x()
            for loss_name in ['center_mae']:
                start_time = time()
                alter_values, est_prob = rh.find_alter_values(x, alterable_vars, alter_ranges, n_method_est_samples,
                                                              M, d,
                                                              loss=loss_name,
                                                              learning_rate=learning_rate,
                                                              epochs=epochs,
                                                              seed=seed,
                                                              verbose=False,
                                                              scaler=model.get_scaler())
                records[f'{loss_name}_time'].append(time() - start_time)
                rehearsal_prob = env.get_success_prob(x, alterable_vars, alter_values, n_env_est_samples)
                records[f'{loss_name}_prob'].append(rehearsal_prob)
                records[f'{loss_name}_est_prob'].append(est_prob)

            # baseline
            start_time = time()
            alter_values, est_prob = ba.find_best_alter_values(x, alterable_vars, alter_ranges, M, d, n_method_est_samples, seed=seed,
                                                               time_limit=time_limit)
            records['baseline_est_prob'].append(est_prob)
            records['baseline_time'].append(time() - start_time)
            baseline_prob = env.get_success_prob(x, alterable_vars, alter_values, n_env_est_samples)
            records['baseline_prob'].append(baseline_prob)
            pbar.set_description(
                f'baseline: {baseline_prob:.3f}, gradient: {rehearsal_prob:.3f},' +
                f'base_time: {records["baseline_time"][-1]:.3f}, gradient_time: {records["center_mae_time"][-1]:.3f},' +
                f'base_avg: {np.mean(records["baseline_prob"]):.3f}, gradient_avg: {np.mean(records["center_mae_prob"]):.3f}')

        torch.save(full_records, f'save/{dataset}_time.pkl')
