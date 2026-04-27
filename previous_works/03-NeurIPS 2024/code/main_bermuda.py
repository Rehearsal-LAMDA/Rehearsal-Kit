# -*- coding: utf-8 -*-

from Rh_Solver import RhSolver
from collections import defaultdict as ddict
import time
import pandas as pd
import copy
from scipy.io import loadmat
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.preprocessing import StandardScaler


if __name__ == "__main__": 
    execution_time = []
    np.random.seed(20240101)
    data_mat = loadmat('../data/SEM_data.mat')
    for key in ['__header__', '__version__', '__globals__', 'Site', 'Lat', 'Lon', 'Year', 'Month', 'Day']:
        data_mat.pop(key)
    data = pd.DataFrame({key: value.reshape(-1) for key, value in data_mat.items()})
    scaler = StandardScaler()
    cols = list(data.columns)
    data = scaler.fit_transform(data)
    data = pd.DataFrame(data, columns=cols)
    vars_stage_1 = ['Light', 'Temp', 'Sal']
    vars_stage_2 = ['DIC', 'TA', 'Omega', 'Nutrients_PC1', 'Chla', 'pHsw', 'CO2']
    vars_stage_3 = ['NEC']

    nodes = vars_stage_1 + vars_stage_2 + vars_stage_3
    nodes_stage = [vars_stage_1, vars_stage_2, vars_stage_3]
    theta_true = ddict(lambda: dict())
    C_true = np.eye(len(nodes))

    parents = {'Light': [], 'Chla': ['Nutrients_PC1', 'Light', 'Temp'], 'Temp': ['Light'], 'Sal': ['Temp'],
               'Omega': ['Sal', 'DIC', 'Temp', 'TA'], 'pHsw': ['Sal', 'DIC', 'Temp', 'TA'], 'DIC': ['Sal'],
               'TA': ['Sal'], 'CO2': ['Sal', 'TA', 'DIC', 'Temp'], 'Nutrients_PC1': [],
               'NEC': ['Nutrients_PC1', 'Light', 'pHsw', 'Omega', 'Chla', 'CO2', 'Temp']}

    ### parameters \Sigma is manually set
    C_true = np.array([ [1.2e-2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 1.6e-2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 1.6e-2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0e-2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 2.0e-2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 1.6e-2, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.6e-2, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.8e-2, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.001, 0.0, 0.0],  
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.6e-3, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.5e-2],
                        ])
    
    ### parameters learned by USE of the original data (learning step is omitted in the code)
    y_bias = -0.5402197079153831
    theta_true_time = ddict(lambda: ddict(lambda: lambda t:0.0))
    theta_true_time["Light"]["Temp"] = lambda t:0.08336954980497932
    theta_true_time["Temp"]["Sal"] = lambda t:-0.4809837373167684
    theta_true_time["Sal"]["DIC"] = lambda t:0.4777168829735183
    theta_true_time["Sal"]["TA"] = lambda t:0.5457734531124397
    theta_true_time["Temp"]["Omega"] = lambda t:0.5182253589055726
    theta_true_time["Sal"]["Omega"] = lambda t:0.03507218718555735 
    theta_true_time["DIC"]["Omega"] = lambda t:-1.1056652215053286 - 0.05 * np.sin(2 * np.pi/80 * (t+10))
    theta_true_time["TA"]["Omega"] = lambda t:1.6104231541803835
    theta_true_time["Light"]["Chla"] = lambda t:-0.15106684218500258
    theta_true_time["Temp"]["Chla"] = lambda t:-0.04451583134247557
    theta_true_time["Nutrients_PC1"]["Chla"] = lambda t:-0.07690378415962325
    theta_true_time["Temp"]["pHsw"] = lambda t:-0.7482789216296077 - 0.06 * np.sin(2 * np.pi/80 * (t-10))
    theta_true_time["Sal"]["pHsw"] = lambda t:0.013001179873522933
    theta_true_time["TA"]["pHsw"] = lambda t:0.7676261914081877
    theta_true_time["DIC"]["pHsw"] = lambda t:-0.5879618774787132
    theta_true_time["Temp"]["CO2"] = lambda t:0.8613318110706953 - 0.09 * np.e ** (-t)
    theta_true_time["Sal"]["CO2"] = lambda t:0.04051812201172802
    theta_true_time["DIC"]["CO2"] = lambda t:0.5700488513842487
    theta_true_time["TA"]["CO2"] = lambda t:-0.596251974686561
    theta_true_time["Light"]["NEC"] = lambda t:0.0322460348829162
    theta_true_time["Temp"]["NEC"] = lambda t:5.227658403563992
    theta_true_time["Omega"]["NEC"] = lambda t:-2.343629162533968 - 0.1 * np.e ** (-t)
    theta_true_time["Chla"]["NEC"] = lambda t:0.13182892043084415
    theta_true_time["Nutrients_PC1"]["NEC"] = lambda t: 0.09881771775808317
    theta_true_time["pHsw"]["NEC"] = lambda t:2.0492558654639
    theta_true_time["CO2"]["NEC"] = lambda t:-2.5146414696724295 - 0.1 * np.sin(2 * np.pi/80 * (t+10))
             
    Para_true = [theta_true_time, C_true]
    
    Cost_dict = dict()
    z0_dict = dict()
    Cost_dict['DIC'] = 10.0
    Cost_dict['TA'] = 8.0
    Cost_dict['Omega'] = 3.0
    Cost_dict['Chla'] = 5.0
    Cost_dict['Nutrients_PC1'] = 10.0
    Cost_dict['pHsw'] = float("inf")
    Cost_dict['CO2'] = float("inf")
    z0_dict["DIC"] = 0.0
    z0_dict["TA"] = 0.0
    z0_dict["Omega"] = 0.0
    z0_dict["Chla"] = 0.0
    z0_dict["Nutrients_PC1"] = 0.0
    Cost_para = [Cost_dict, z0_dict]


    candidate_Rh_list = [("DIC",), ("TA",), ("Omega",), ("Chla",), ("Nutrients_PC1",),]

    Limit_dict = dict()

    Limit_dict['DIC'] = [-1.0, 1.0]
    Limit_dict['TA'] = [-1.0, 1.0]
    Limit_dict['Omega'] = [-1.0, 1.0]
    Limit_dict['Chla'] = [-1.0, 1.0]
    Limit_dict['Nutrients_PC1'] = [-1.0, 1.0]
    M = np.array([[1], 
                  [-1]])
    d = np.array([2, -0.5]).reshape((-1, 1)) - M.dot(y_bias)
 
    target_domin = [M, d]

    binary_edge = []

    init_n = 10

    
    sH = ddict(lambda: [])
    
    for node in nodes:
        sH[node] = [1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1, 1, 5, 7, 9, 10]

    cl = 0.7
    rnd = 100
    lr = np.sqrt(np.log(len(sH["DIC"]))/rnd)
    

    times = 20
    seed_list = np.random.choice(np.arange(times*50), times, replace=False).tolist()
    print(seed_list)
    vic_rate_list = []
    costs = []

    for rnd_seed in seed_list:
        start_time = time.time()
        bermuda = RhSolver(
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
            nodes_stage = nodes_stage,
        )
        np.random.seed(rnd_seed)    
        victory_count_list, res_list, alterations, cost_denote = bermuda.pipeline(rounds = rnd)
        vic_rate_list.append(sum(victory_count_list)/len(victory_count_list))
        costs.append(sum(cost_denote)/len(cost_denote))
        end_time = time.time()
        execution_time.append(end_time - start_time)
       
    print("Avg. Running time (s):\t", sum(execution_time)/len(execution_time))
    print("Std. Running time (s):\t", np.sqrt(np.var(np.array(execution_time))))
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