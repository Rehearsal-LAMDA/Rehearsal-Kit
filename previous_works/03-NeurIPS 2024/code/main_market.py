# -*- coding: utf-8 -*-

from Rh_Solver import RhSolver
from collections import defaultdict as ddict
import numpy as np
import time

if __name__ == "__main__":
    start_time = time.time()
    np.random.seed(20240101)
    nodes = ["competitor_feature", "economic_index", "competitor_raw_cost", "competitor_pricing", "self_pricing", "raw_cost", "total_profit", "custom_number"]
    theta_true = ddict(lambda: ddict(lambda: lambda t:0.0))

    theta_true["competitor_feature"]["competitor_raw_cost"] = lambda t:10.0
    theta_true["economic_index"]["raw_cost"] = lambda t:10.0
    theta_true["self_pricing"]["total_profit"] = lambda t:0.9
    theta_true["self_pricing"]["custom_number"] = lambda t:-0.9 - 0.05 * np.sin(2 * np.pi/80 * (t+10))
    theta_true["raw_cost"]["competitor_pricing"] = lambda t:0.5
    theta_true["raw_cost"]["self_pricing"] = lambda t:2.0 + 0.5 * np.sin(2 * np.pi/12 * t)
    theta_true["competitor_raw_cost"]["competitor_pricing"] = lambda t:1.3
    theta_true["competitor_raw_cost"]["self_pricing"] = lambda t:0.4
    theta_true["raw_cost"]["total_profit"] = lambda t:-1.0
    theta_true["raw_cost"]["custom_number"] = lambda t:1.6 + 0.1 * np.sin(2 * np.pi/80 * (t+10))


    C_true = np.array([ [0.04, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.04, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.06, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.03, 0.016, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.016, 0.06, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.06, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.01, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.04]
                        ])

    nodes_stage = [["competitor_feature", "economic_index"], ["competitor_raw_cost", "competitor_pricing", "self_pricing", "raw_cost"], ["total_profit", "custom_number"]]
    Para_true = [theta_true, C_true]
    Cost_dict = dict()
    z0_dict = dict()
    Cost_dict["competitor_pricing"] = float("inf")
    Cost_dict["competitor_raw_cost"] = float("inf")
    Cost_dict["self_pricing"] = 1.0
    Cost_dict["raw_cost"] = 2.0
    
    z0_dict["self_pricing"] = 0.0
    z0_dict["raw_cost"] = 0.75

    Cost_para = [Cost_dict, z0_dict]
    candidate_Rh_list = [("raw_cost",), ("self_pricing",), ("raw_cost", "self_pricing")]
    Limit_dict = dict()
    Limit_dict["self_pricing"] = [-3.0, 3.0]
    Limit_dict["raw_cost"] = [-3.0, 3.0]

    M= np.array([[-1, 0], 
                 [0, -1],
                 [-1, -1]])
    
    d = np.array([0.3, 0.3, -0.05]).reshape((-1, 1))
    target_domin = [M, d]

    binary_edge = [["competitor_pricing", "self_pricing"]]

    init_n = 10

    
    sH = ddict(lambda: [])
    
    
    
    sH["competitor_feature"] = [1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2]
    sH["competitor_raw_cost"] = [1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2]
    sH["economic_index"] = [1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2]
    sH["raw_cost"] = [1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2]
    sH["competitor_pricing"] = [1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2]
    sH["self_pricing"] = [1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2]
    sH["total_profit"] = [1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2]
    sH["custom_number"] = [1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2]

    cl = 0.7
    rnd = 100
    lr = np.sqrt(np.log(len(sH["economic_index"]))/rnd)
    
    times = 20
    seed_list = np.random.choice(np.arange(times*50), times, replace=False).tolist()
    print(seed_list)
    vic_rate_list = []
    costs = []
    
    for rnd_seed in seed_list:
        np.random.seed(rnd_seed)
        toy_eg = RhSolver(
            nodes = nodes, 
            Para_true = Para_true, 
            Cost_para = Cost_para, 
            Limit_dict = Limit_dict,
            binary_edge = binary_edge,
            target_domin = target_domin,
            init_n = init_n,
            experts_dict = sH,
            learning_rate=lr,
            confidence_level = cl,
            candidate_Rh_list=candidate_Rh_list,
            nodes_stage=nodes_stage
        )

        
        
        victory_count_list, res_list, alterations, cost_denote = toy_eg.pipeline(rounds = rnd)
        vic_rate_list.append(sum(victory_count_list)/len(victory_count_list))
        costs.append(sum(cost_denote)/len(cost_denote))

    end_time = time.time()
    execution_time = end_time - start_time
    print("Avg. Running time (s):\t", execution_time/times)
    print("Avg. AUF prob.:\t", sum(vic_rate_list)/len(vic_rate_list))
    print("Var. AUF prob.:\t", np.var(np.array(vic_rate_list)))
    print("Mid. AUF prob.:\t", np.percentile(np.array(vic_rate_list), 50))
    print("25%. AUF prob.:\t", np.percentile(np.array(vic_rate_list), 25))
    print("75%. AUF prob.:\t", np.percentile(np.array(vic_rate_list), 75))
    print("Avg. Cost:\t", sum(costs)/len(costs))
    print("Var. Cost:\t", np.var(np.array(costs)))
    print("Mid. Cost:\t", np.percentile(np.array(costs), 50))
    print("25%. Cost:\t", np.percentile(np.array(costs), 25))
    print("75%. Cost:\t", np.percentile(np.array(costs), 75))

