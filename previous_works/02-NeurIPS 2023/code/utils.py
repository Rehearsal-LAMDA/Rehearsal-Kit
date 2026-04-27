import networkx as nx
import numpy as np
from bayes_opt import BayesianOptimization
from multiset import Multiset
from rpy2 import robjects
from rpy2.robjects import pandas2ri, numpy2ri
from rpy2.robjects.packages import importr
from scipy.stats import beta, norm, differential_entropy
from sklearn.linear_model import BayesianRidge

numpy2ri.activate()
pandas2ri.activate()
pcalg = importr('pcalg')
r_as = robjects.r['as']


def add_background_knowledge(graph, cpgraph, n_vars_in_stages):
    n_vars1, n_vars2, n_vars3 = tuple(n_vars_in_stages)
    n_vars = sum(n_vars_in_stages)

    cpgraph = np.array(cpgraph, dtype='int')
    xs = []
    ys = []
    for u in range(n_vars):
        for v in range(n_vars):
            u_stage = 0 if u < n_vars1 else (1 if u < n_vars1 + n_vars2 else 2)
            v_stage = 0 if v < n_vars1 else (1 if v < n_vars1 + n_vars2 else 2)
            if cpgraph[u, v] == 1 and cpgraph[v, u] == 1 and u_stage < v_stage:
                xs.append(u)
                ys.append(v)

    print(xs, ys)
    print((graph != cpgraph).sum())

    numpy2ri.deactivate()
    pandas2ri.deactivate()

    cpgraph = robjects.r.matrix(cpgraph, n_vars, n_vars)
    cpgraph.rownames = robjects.StrVector([str(i) for i in range(n_vars)])
    cpgraph.colnames = robjects.StrVector([str(i) for i in range(n_vars)])
    cpgraph = r_as(cpgraph, 'graphNEL')

    numpy2ri.activate()
    pandas2ri.activate()

    addBgKnowledge = robjects.r['addBgKnowledge']
    mpgraph = r_as(addBgKnowledge(cpgraph, x=[str(x) for x in xs], y=[str(y) for y in ys]), 'matrix')
    mpgraph = np.array(mpgraph, dtype='int')

    print((graph != mpgraph).sum())

    mpgraph[:n_vars1, :n_vars1] = graph[:n_vars1, :n_vars1]
    print((graph != mpgraph).sum())

    return mpgraph


def enumerate_graphs(mpgraph, n_max=100):
    pgraph2allgraphs = robjects.r['pgraph2allgraphs']
    nrow = robjects.r['nrow']
    matrix = robjects.r['matrix']
    n_vars = mpgraph.shape[0]
    res = pgraph2allgraphs(mpgraph.T)
    n_graphs = nrow(res.rx2('graphs')).item()
    graphs = []
    for i in range(min(n_max, n_graphs)):
        graphs.append(res.rx2('graphs')[i,].reshape(n_vars, n_vars).T)
    return graphs


def sample_graphs(mpgraph, n_graphs=10, equal_weights=False):
    graphs = []
    if nx.is_directed_acyclic_graph(nx.DiGraph(mpgraph)):
        graphs.append((mpgraph.copy(), n_graphs))
    else:
        n_vars = mpgraph.shape[0]
        addBgKnowledge = robjects.r['addBgKnowledge']
        for _ in range(n_graphs):
            graph = mpgraph.copy()
            undirected_u, undirected_v = np.nonzero(np.triu(graph == graph.T) & (graph == 1))
            while len(undirected_u) > 0:
                selected_edge_idx = np.random.randint(0, len(undirected_u))
                u, v = undirected_u[selected_edge_idx], undirected_v[selected_edge_idx]
                if np.random.rand() < 0.5:
                    u, v = v, u

                numpy2ri.deactivate()
                pandas2ri.deactivate()

                cpgraph = robjects.r.matrix(graph, n_vars, n_vars)
                cpgraph.rownames = robjects.StrVector([str(i) for i in range(n_vars)])
                cpgraph.colnames = robjects.StrVector([str(i) for i in range(n_vars)])
                cpgraph = r_as(cpgraph, 'graphNEL')

                numpy2ri.activate()
                pandas2ri.activate()

                graph = r_as(addBgKnowledge(cpgraph, x=[str(u)], y=[str(v)]), 'matrix').astype(int)

                undirected_u, undirected_v = np.nonzero(np.triu(graph == graph.T) & (graph == 1))

            found = False
            for idx, (comp_graph, weight) in enumerate(graphs):
                if (comp_graph == graph).all():
                    graphs[idx] = (graph, weight + 1)
                    found = True
                    break
            if not found:
                graphs.append((graph, 1))
    if equal_weights:
        graphs = [(graph, 1 / len(graphs)) for graph, _ in graphs]
    else:
        graphs = [(graph, w / n_graphs) for graph, w in graphs]
    return graphs


def get_mpgraph(graph, n_vars_in_stages, add_bk=True):
    graph2cpgraph = robjects.r['dag2cpdag']
    g1 = r_as(graph, 'graphNEL')
    cpgraph1 = graph2cpgraph(g1)
    cpgraph = r_as(cpgraph1, 'matrix')
    if add_bk:
        mpgraph = add_background_knowledge(graph, cpgraph, n_vars_in_stages)
    else:
        mpgraph = cpgraph
    return mpgraph


def weight_graph(graph, graphs):
    for idx, (d, w) in enumerate(graphs):
        if (d == graph).all():
            print(f'True graph index {idx}, weights {w}, total {len(graphs)}')
            return idx
    print('True graph not sampled')
    return -1


def learn_sems(data, graphs, start_var_idx, end_var_idx, return_weights=False, interv_data=None):  # interv_data: {(v1,v2,...): X}
    sems = []
    weights = []
    if interv_data is None or len(interv_data) == 0:
        X = {var: data for var in range(start_var_idx, end_var_idx)}
    else:
        X = {}
        for var in range(start_var_idx, end_var_idx):
            useful_data = [data]
            for interv_targets, d in interv_data.items():
                if var not in interv_targets:
                    useful_data.append(d)
            X[var] = np.vstack(useful_data)

    for graph, _ in graphs:
        graph = graph.copy()
        regs = {}
        flag = False
        for i in range(graph.shape[0]):
            for j in range(i):
                if graph[j, i] != 0 and graph[i, j] != 0:
                    flag = True
                    break
        if not flag:
            estimated_weights = np.zeros((graph.shape[0], graph.shape[0]))
            for var in range(start_var_idx, end_var_idx):
                parents = np.nonzero(graph[:, var])[0]
                reg = BayesianRidge()
                if len(parents) == 0:
                    reg.fit(np.zeros((X[var].shape[0], 1)), X[var][:, var])
                else:
                    reg.fit(X[var][:, parents], X[var][:, var])
                    estimated_weights[parents, var] = reg.coef_
                regs[var] = reg
            sems.append(regs)
            weights.append(estimated_weights)
        else:
            u, v = None, None
            for i in range(graph.shape[0]):
                for j in range(i):
                    if graph[j, i] != 0 and graph[i, j] != 0:
                        u, v = i, j
                        break

            estimated_weights = np.zeros((graph.shape[0], graph.shape[0]))
            for var in range(start_var_idx, end_var_idx):
                parents = np.nonzero(graph[:, var])[0]
                reg = BayesianRidge()
                if len(parents) == 0:
                    reg.fit(np.zeros((X[var].shape[0], 1)), X[var][:, var])
                else:
                    reg.fit(X[var][:, parents], X[var][:, var])
                    estimated_weights[parents, var] = reg.coef_
                regs[var] = reg
            sems.append(regs)
            weights.append(estimated_weights)

    if return_weights:
        return sems, weights
    else:
        return sems


def sample_sem_params(sem, graph, n_samples):
    n_vars = graph.shape[0]
    coefs = [0] * n_vars
    noises = [0] * n_vars
    biases = [0] * n_vars
    variances = [0] * n_vars
    for var, reg in sem.items():
        parents = np.nonzero(graph[:, var])[0]
        coefs[var] = np.zeros((n_samples, n_vars))
        coefs[var][:, parents] = np.random.multivariate_normal(reg.coef_, reg.sigma_, n_samples)
        noises[var] = np.random.normal(0, np.sqrt(1 / reg.alpha_), n_samples)
        biases[var] = reg.intercept_
        variances[var] = 1 / reg.alpha_

    params = {'coefs': coefs, 'noises': noises,
              'biases': biases, 'variances': variances}  # coef: [(n_samples, d)] * d, noise: [(n_samples,)] * d, bias: [float]*d, variance: [float]*d
    return params


def sample_intervs_single_linear(x, params, order, interv_target, interv_value):
    # for a fixed graph structure
    n_samples = params['coefs'][-1].shape[0]
    n_vars = len(order)
    n_vars_observed = len(x)
    record = np.zeros((n_samples, n_vars))
    record[:, :n_vars_observed] = x
    record[:, interv_target] = interv_value
    for var in order:
        if var < n_vars_observed or var == interv_target:
            continue
        record[:, var] = (record * params['coefs'][var]).sum(axis=1) + params['biases'][var] + params['noises'][var]
    return record


def sample_symbolic_intervs_single_linear(x, params, order, interv_target):
    # for a fixed graph
    n_samples = params['coefs'][-1].shape[0]
    n_vars = len(order)
    n_vars_observed = len(x)
    record_real, record_symbol = np.zeros((n_samples, n_vars)), np.zeros((n_samples, n_vars))
    record_real[:, :n_vars_observed] = x
    record_symbol[:, interv_target] = np.ones(n_samples)
    for var in order:
        if var < n_vars_observed or var == interv_target:
            continue
        record_real[:, var] = (record_real * params['coefs'][var]).sum(axis=1) + params['biases'][var] + params['noises'][var]
        record_symbol[:, var] = (record_symbol * params['coefs'][var]).sum(axis=1)
    return record_real, record_symbol


def find_candidate_interv_values_single_linear(symbolic_record, M, d, tau, eps=0):
    n_vars_last_stage = M.shape[1]
    record_real, record_symbol = symbolic_record
    n_samples = record_real.shape[0]
    y_record_real, y_record_symbol = record_real[:, -n_vars_last_stage:], record_symbol[:, -n_vars_last_stage:]

    U, V = -np.ones(n_samples) * np.inf, np.ones(n_samples) * np.inf
    for row, e in zip(M, d):
        rhs = e - y_record_real @ row
        lhs = y_record_symbol @ row

        r = np.where(lhs > 0, rhs / (lhs + 1e-6), np.inf)
        l = np.where(lhs < 0, rhs / (lhs - 1e-6), -np.inf)
        r = np.where((lhs == 0) & (rhs < 0), -np.inf, r)
        l = np.where((lhs == 0) & (rhs < 0), np.inf, l)
        U = np.maximum(U, l)
        V = np.minimum(V, r)
    valid_idx = (V >= U)
    U, V = U[valid_idx], V[valid_idx]

    W = list(set(np.hstack([U, V])))
    W.sort()
    U, V = Multiset(U), Multiset(V)
    cnt = 0
    intervals = []
    left = None
    for w in W:
        if w in U:
            cnt += U[w]
            if cnt >= n_samples * tau and left is None:
                left = w
        else:
            cnt -= V[w]
            if cnt < n_samples * tau and left is not None:
                if w - left > eps:
                    intervals.append((left, w))
                left = None
    if left is not None and W[-1] - left > eps:
        intervals.append((left, W[-1]))
    return intervals


def simulate_interv_in_desired_region(sems, graphs, interv_target, interv_value, x, M, d, n_samples):
    records = []
    n_vars_last_stage = M.shape[1]
    for (graph, weight), sem in zip(graphs, sems):
        n = int(np.ceil(n_samples * weight))
        order = list(nx.topological_sort(nx.DiGraph(graph)))
        params = sample_sem_params(sem, graph, n)
        record = sample_intervs_single_linear(x, params, order, interv_target, interv_value)[:, -n_vars_last_stage:]
        records.append(record)
    records = np.vstack(records)
    n_in = np.sum((records @ M.T - d <= 0).all(axis=1))
    return n_in, n_samples


def success_prob_bound(n_in, n, delta):
    n_out = n - n_in
    p_hat = n_in / n
    p1 = 1 - beta.ppf(1 - delta / 2 / n, n_out + 1, n_in)
    p2 = 1 - beta.ppf(delta / 2 / n, n_out, n_in + 1)
    eps = np.sqrt(np.log(2 / delta) / 2 / n)
    low = max(p1, p_hat - eps)
    high = min(p2, p_hat + eps)
    return p_hat, low, high


def find_intervals(x, sems, graphs, interv_target, n_vars_in_stages, M, d, tau, n_samples=1000, eps=1e-2):
    n_nodes = sum(n_vars_in_stages)
    symbolic_records = []
    for sem, (graph, weight) in zip(sems, graphs):
        g = nx.DiGraph(graph)
        if any([nx.has_path(g, interv_target, y_var) for y_var in range(n_nodes - n_vars_in_stages[2], n_nodes)]):
            order = list(nx.topological_sort(g))
            params = sample_sem_params(sem, graph, n_samples=int(n_samples * weight) + 1)
            symbolic_records.append(sample_symbolic_intervs_single_linear(x, params, order, interv_target))
    if len(symbolic_records) > 0:
        symbolic_records = (np.vstack([real_record for real_record, _ in symbolic_records]),
                            np.vstack([sym_record for _, sym_record in symbolic_records]))
        intervals = find_candidate_interv_values_single_linear(symbolic_records, M, d, tau, eps)
    else:
        intervals = []
    return intervals


def intersect_intervals(intervals):
    def intersect(interval1, interval2):
        if len(interval1) == 0 or len(interval2) == 0:
            return []
        ans = []
        for l, r in interval1:
            for u, v in interval2:
                a = max(l, u)
                b = min(r, v)
                if a <= b:
                    ans.append((a, b))
        return ans

    ints = intersect(intervals[0], intervals[1])
    idx = 2
    while idx < len(intervals):
        ints = intersect(ints, intervals[idx])
        idx += 1
    return ints


def log_likelihood(sem, graph, data, interv_targets=None):
    ll = 0
    for var, reg in sem.items():
        if interv_targets is None or var not in interv_targets:
            parents = np.nonzero(graph[:, var])[0]
            if len(parents) == 0:
                predictors = np.zeros((data.shape[0], 1))
            else:
                predictors = data[:, parents]
            mean, std = reg.predict(predictors, True)
            ll += (norm.logpdf(data[:, var], mean, std)).sum()
    return ll


def log_likelihood_intervs_single_linear(y_vars, y_values, x, params, order, interv_target, interv_value):
    # for a fixed graph structure
    n_samples = len(params['coefs'][0].shape[0])
    n_vars = len(order)
    n_vars_observed = len(x)
    means = np.zeros((n_samples, n_vars))
    means[:, :n_vars_observed] = x
    means[:, interv_target] = interv_value
    variances = np.zeros((n_samples, n_vars))
    variances[:, :n_vars_observed] = 0
    variances[:, interv_target] = 0
    for var in order:
        if var < n_vars_observed or var == interv_target:
            continue
        if var in params[0].keys():
            means[:, var] = (params['coefs'][var] * means).sum(axis=1) + params['biases'][var]
            variances[:, var] = ((params['coefs'][var] ** 2) * variances).sum(axis=1) + params['variances'][var]

    loglike = sum(
        norm.logpdf(y_value * np.ones(n_samples), means[:, y_var], np.sqrt(variances[:, y_var])) for y_var, y_value in zip(y_vars, y_values))

    return loglike


def optimize_mutual_info(interv_target, intervals, x, n_vars_in_stages, graphs, sems, n_iter, n_samples_out, n_samples_in):
    left_vars = list(range(n_vars_in_stages[0], interv_target)) + list(range(interv_target + 1, sum(n_vars_in_stages)))
    orders = []
    paramses = []
    for (graph, weight), sem in zip(graphs, sems):
        n = int(np.ceil(n_samples_out * weight))
        order = list(nx.topological_sort(nx.DiGraph(graph)))
        params = sample_sem_params(sem, graph, n)
        orders.append(order)
        paramses.append(params)

    orders2 = []
    paramses2 = []
    for (graph, weight), sem in zip(graphs, sems):
        order = list(nx.topological_sort(nx.DiGraph(graph)))
        params = sample_sem_params(sem, graph, n_samples_in)
        orders2.append(order)
        paramses2.append(params)

    def f(value):
        records = []
        for order, params in zip(orders, paramses):
            record = sample_intervs_single_linear(x, params, order, interv_target, value)
            records.append(record)
        records = np.vstack(records)

        H_y_mid_D_xi = differential_entropy(records[:, left_vars])
        H_y_mid_G_theta_xi = 0
        for order, params in zip(orders2, paramses2):
            record = sample_intervs_single_linear(x, params, order, interv_target, value)
            H_y_mid_G_theta_xi += differential_entropy(record[:, left_vars]) * weight
        gain = (H_y_mid_D_xi - H_y_mid_G_theta_xi).sum()
        return gain

    optimizer = BayesianOptimization(
        f=f,
        pbounds={'value': (-1, 1)},
        verbose=0,
        random_state=1,
        allow_duplicate_points=True
    )
    measure = sum([b - a for a, b in intervals])
    for l, r in intervals:
        max_iter = int(n_iter * (r - l) / measure) + 1
        optimizer.set_bounds(new_bounds={'value': (l, r)})
        for v in [l, r, (l + r) / 2]:
            optimizer.probe(
                params=[v],
                lazy=True,
            )
        optimizer.maximize(
            init_points=0,
            n_iter=max_iter,
        )
    max_info_gain = optimizer.max['target']
    interv_value = optimizer.max['params']['value']
    return interv_value, max_info_gain
