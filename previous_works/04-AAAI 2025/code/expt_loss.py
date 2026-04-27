import os
from itertools import chain

import torch
from tqdm import tqdm

from baseline_multi import MultivariateBaseline
from bermuda import Bermuda
from env import Environment
from linear import LinearModel
from rehearsal import Rehearsal
from synthetic import Syn1, Syn2
from traffic import Traffic
from utils import get_args

if __name__ == '__main__':

    args = get_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    n_repeats = 100
    n_observations = 1000
    n_method_est_samples = 1000
    n_env_est_samples = 10000
    noise_scale = 0.05
    epochs = 200
    learning_rate = 0.01

    dataset = args.data
    dsmap = {'traffic': Traffic(noise_scale),
             'bermuda': Bermuda('../data/SEM_data.mat', noise_scale),
             'syn1': Syn1(),
             'syn2': Syn2()}
    ds = dsmap[dataset]
    print(dataset)

    x_dim, z_dim, y_dim, n_vars = ds.get_info()
    children_parents_pairs = ds.get_children_parents_pairs()
    alterable_vars = ds.get_alterable_vars()
    free_vars = [var for var in range(n_vars) if var not in chain(*[children for children, _ in children_parents_pairs])]
    alter_ranges = ds.get_alter_ranges()
    free_var_scale = 1 if 'syn' in dataset else [ds.get_input_range()[var] for var in free_vars]
    env_seed = ds.get_env_seed() if 'syn' in dataset else 0

    records = {}
    for loss_name in ['center_mse', 'center_mae', 'ball_insensitive', 'ball_huber']:
        records[f'{loss_name}_prob'] = []
        records[f'{loss_name}_loss'] = []

    env = Environment(x_dim, z_dim, y_dim, children_parents_pairs,
                      alterable_vars=ds.get_alterable_vars(),
                      free_var_scale=free_var_scale,
                      mode='linear',
                      seed=env_seed)
    if 'syn' not in dataset:
        env.set_srm_params(ds.get_srm_params())
    M, d = ds.get_desired_region()
    env.set_desired_region(M, d)

    obs_data = env.get_observations(n_observations)
    model = LinearModel(n_vars, env.get_ordered_children_parents(), device=device)
    model.fit(obs_data)
    rh = Rehearsal(n_vars, x_dim, y_dim, env.get_ordered_children_parents(), env.get_free_vars(),
                   model.get_models(),
                   model.get_noise_cov(),
                   'linear',
                   device)
    ba = MultivariateBaseline(n_vars, x_dim, y_dim, env.get_ordered_children_parents(), env.get_free_vars(),
                              model.get_models(),
                              model.get_noise_cov())

    pbar = tqdm(range(n_repeats))
    for seed in pbar:
        x = env.observe_x()
        for loss_name in ['ball_insensitive', 'center_mse', 'center_mae', 'ball_huber']:
            for alter_values, loss in rh.find_alter_values_iter(x, alterable_vars, alter_ranges, n_method_est_samples,
                                                                M, d,
                                                                loss=loss_name,
                                                                learning_rate=learning_rate,
                                                                epochs=epochs,
                                                                seed=seed,
                                                                verbose=False):
                rehearsal_prob = env.get_success_prob(x, alterable_vars, alter_values, n_env_est_samples)
                records[f'{loss_name}_prob'].append(rehearsal_prob)
                records[f'{loss_name}_loss'].append(loss)

        pbar.set_description(f'gradient: {rehearsal_prob:.3f}')

    torch.save(records, f'save/{dataset}_loss.pkl')
