# -*- coding: utf-8 -*-

import numpy as np

def find_all_paths_with_costs(graph, start, end, path=[], cost=1.0):
    path = path + [start]
    if start == end:
        return [(path, cost)]
    if start not in graph:
        return []
    paths = []
    for node in graph[start]:
        if node not in path:
            # print(graph[start][node])
            new_paths = find_all_paths_with_costs(graph, node, end, path, cost * graph[start][node])
            for new_path, new_cost in new_paths:
                paths.append((new_path, new_cost))
    return paths

def find_total_costs_to_node(graph, end_node):
    total_costs = {}
    for node in graph:
        if node != end_node:
            costs = find_all_paths_with_costs(graph, node, end_node)
            total_cost = sum(cost for path, cost in costs)
            total_costs[node] = total_cost
    return total_costs

# import numpy as np

def sample_uniform_on_unit_sphere(p, m):
    # Generate m points in p-dimensional space with Gaussian distribution
    points = np.random.randn(m, p)
    
    # Normalize each point to lie on the unit sphere
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    
    return list(points)