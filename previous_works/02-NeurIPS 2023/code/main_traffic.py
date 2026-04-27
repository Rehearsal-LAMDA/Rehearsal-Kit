from data import *
from utils import *

if __name__ == '__main__':
    n_vars_in_stages, intervables, weighted_random_graph, bias, input_range, cliques = get_traffic_params()

    n_nodes = sum(n_vars_in_stages)
    noise_scale = 0.08
    T = 100
    n_obs = 10

    datagen = DataGenBias(W=weighted_random_graph, method='linear', sem_type='gauss', noise_scale=noise_scale,
                          bias=bias, input_range=input_range, cliques=cliques)

    re_matrix, X = (weighted_random_graph != 0).astype(int), datagen.sample_iid_obs(n=n_obs)
    equiv_graph1 = re_matrix.copy()
    equiv_graph1[2, 3] = 1
    equiv_graph1[3, 2] = 0
    equiv_graph2 = re_matrix.copy()
    equiv_graph2[2, 3] = 0
    equiv_graph2[3, 2] = 1

    #
    graphs = [(equiv_graph1, 1 / 3), (equiv_graph2, 1 / 3)]
    re_matrix_weight = 1 / 3
    alt_graphs = {2: equiv_graph1, 5: re_matrix}

    observed_x = datagen.sample_iid_obs(T)[:, :n_vars_in_stages[0]]
    taus = [0.7] * T
    print('*' * 30)
    print(taus[0])
    Ms = [np.array([[1], [-1]])] * T
    ds = [np.array([1, -0.8])] * T
    interv_data = {}
    success_cnt = 0
    sems = learn_sems(X, graphs, n_vars_in_stages[0], n_nodes, return_weights=False, interv_data=None)
    alt_sems = {interv_target: sem for interv_target, sem in zip([2, 5], learn_sems(X, [(alt_graphs[2], 1), (alt_graphs[5], 1)],
                                                                                    n_vars_in_stages[0], n_nodes, return_weights=False,
                                                                                    interv_data=None))}
    ######
    graph_weights_hist = []
    interv_hist = []
    suc_hist = []
    success_prob_hist = []
    bound_hist = []
    info_gain_hist = []

    for round, (x, tau, M, d) in enumerate(zip(observed_x, taus, Ms, ds)):
        best_interv = (-1, -1)
        max_info_gain = -np.inf
        found_valid = False
        for interv_target, value_interval in intervables.items():

            intervals_train = find_intervals(x, sems + [alt_sems[interv_target]],
                                             graphs + [(alt_graphs[interv_target], re_matrix_weight)],
                                             interv_target, n_vars_in_stages,
                                             M, d, tau, n_samples=10000, eps=1e-2)
            intervals_val = find_intervals(x, sems + [alt_sems[interv_target]],
                                           graphs + [(alt_graphs[interv_target], re_matrix_weight)],
                                           interv_target, n_vars_in_stages,
                                           M, d, tau, n_samples=10000, eps=1e-2)
            intervals = intersect_intervals([intervals_train, intervals_val, [value_interval]])

            if len(intervals) > 0:
                found_valid = True
                interv_value, info_gain = optimize_mutual_info(interv_target, intervals, x, n_vars_in_stages,
                                                               graphs + [(alt_graphs[interv_target], re_matrix_weight)],
                                                               sems + [alt_sems[interv_target]],
                                                               n_iter=20, n_samples_out=1000, n_samples_in=1000)
                if info_gain > max_info_gain:
                    max_info_gain = info_gain
                    best_interv = (interv_target, interv_value)

        if not found_valid:
            print('constraints are not satisfiable, optimizing without constriants ...')
            for interv_target, interval in intervables.items():
                intervals = [interval]
                interv_value, info_gain = optimize_mutual_info(interv_target, intervals, x, n_vars_in_stages,
                                                               graphs + [(alt_graphs[interv_target], re_matrix_weight)],
                                                               sems + [alt_sems[interv_target]],
                                                               n_iter=10, n_samples_out=500, n_samples_in=500)
                if info_gain > max_info_gain:
                    max_info_gain = info_gain
                    best_interv = (interv_target, interv_value)

        interv_target, interv_value = best_interv
        n_in, n = simulate_interv_in_desired_region(sems + [alt_sems[interv_target]],
                                                    graphs + [(alt_graphs[interv_target], re_matrix_weight)],
                                                    interv_target, interv_value, x, M, d, 1000)
        p_hat, p_low, p_high = success_prob_bound(n_in, n, delta=0.05)
        new_interv_data = datagen.sample_iid_interv([interv_target], [interv_value], n=1 + 1000, x=x)
        new_data = new_interv_data[[-1], :]
        if (interv_target,) in interv_data.keys():
            interv_data[(interv_target,)] = np.vstack([interv_data[(interv_target,)], new_data])
        else:
            interv_data[(interv_target,)] = new_data

        sems = learn_sems(X, graphs, n_vars_in_stages[0], n_nodes, return_weights=False, interv_data=interv_data)
        alt_sems = {interv_target: sem for interv_target, sem in zip([2, 5], learn_sems(X, [(alt_graphs[2], 1), (alt_graphs[5], 1)],
                                                                                        n_vars_in_stages[0], n_nodes, return_weights=False,
                                                                                        interv_data=interv_data))}
        likelihoods = []
        for sem, (graph, _) in zip(sems, graphs):
            likelihoods.append(np.exp(log_likelihood(sem, graph, new_data, [interv_target])))
        graphs = [(graph, weight * likelihood) for (graph, weight), likelihood in zip(graphs, likelihoods)]
        re_likelihood = np.exp(log_likelihood(alt_sems[interv_target], alt_graphs[interv_target],
                                              new_data, [interv_target]))
        re_matrix_weight *= re_likelihood
        total_weights = sum([w for _, w in graphs] + [re_matrix_weight])
        graphs = [(graph, weight / total_weights) for graph, weight in graphs]
        re_matrix_weight /= total_weights

        success = False
        if (M @ new_data[0, -n_vars_in_stages[2]:].T - d <= 0).all():
            success_cnt += 1
            success = True
        success_prob = ((new_interv_data[:-1, -n_vars_in_stages[2]:] @ M.T - d <= 0).all(axis=1).sum()) / (new_interv_data.shape[0] - 1)

        print(f'input x: {x}', f'tau: {tau}')
        print(f'round {round} intervene on {interv_target}:={interv_value}, '
              f'{"success" if success else "fail"}, prob {success_prob}, rate {success_cnt / (round + 1)}')

        graph_weights_hist.append([w for _, w in graphs])
        interv_hist.append(best_interv)
        suc_hist.append(success)
        success_prob_hist.append(success_prob)
        bound_hist.append((n_in, n, p_hat, p_low, p_high))
        info_gain_hist.append(max_info_gain)
