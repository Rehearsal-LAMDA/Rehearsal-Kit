import matplotlib
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({'font.family':'sans-serif'})
plt.rcParams.update({'font.sans-serif':'Helvetica'})
matplotlib.rcParams['mathtext.fontset'] = 'stix'


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
            new_paths = find_all_paths_with_costs(graph, node, end, path, cost * graph[start][node][-1])
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


if __name__ == "__main__":
    # Example usage:
    graph = {
        'A': {'B': (0.1, 0.1)},
        'B': {'C': (0.2, 0.2)},
        'C': {'D': (0.3, 0.3), 'E': (2.0, 2.0)},
        'D': {'E': (0.4, 0.4)},
        'E': {},
    }

    end_node = 'E'
    paths_to_node = find_total_costs_to_node(graph, end_node)
    print(paths_to_node)