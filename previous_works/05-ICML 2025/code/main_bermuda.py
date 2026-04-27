# -*- coding: utf-8 -*-

from Rh_Solver import *
import time
import pandas as pd
import matplotlib.pyplot as plt
from brokenaxes import brokenaxes
import copy
from tqdm import tqdm
from scipy.io import loadmat
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.preprocessing import StandardScaler

def Plot_Bermuda_Y_and_S_after(Y,):
    plt.figure(figsize=(10, 5.8))
    bax = brokenaxes(xlims=((0.0, 3.25),), ylims=((0.0, 1.15), ), hspace=.0, despine=False, wspace=.0, d=0.015,)
    bax.hist(Y, color="green", bins=40, density=True, label=r"$\mathbf{Y}$ Distribution after altering")
    
    bax.axvline(x=2.54021971, color='red')
    bax.axvline(x=1.04021971, color='red')
    bax.fill_between([1.04021971, 2.54021971], 0, plt.ylim(0, 1.15)[1], color='red', alpha=0.3, label=r"Desired interval $\mathcal{S}$")
    
    for ax in bax.fig.get_axes():
        ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)  
        ax.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)   
    
    bax.set_xlabel(r'$\mathbf{Y}$ Value after Altering', fontsize=22, labelpad=3)
    bax.set_ylabel(r'Empirical PDF for $\mathbf{Y}$', fontsize=22, labelpad=3)
    bax.legend(fontsize=18, loc='upper left', bbox_to_anchor=(-0.0, 1))
    plt.savefig('region2_after_alter.pdf', bbox_inches='tight', pad_inches=0.01)



if __name__ == "__main__": 
    execution_time = []
    np.random.seed(2024)
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

    parents = {'Light': [], 'Chla': ['Nutrients_PC1', 'Light', 'Temp'], 'Temp': ['Light'], 'Sal': ['Temp'],
               'Omega': ['Sal', 'DIC', 'Temp', 'TA'], 'pHsw': ['Sal', 'DIC', 'Temp', 'TA'], 'DIC': ['Sal'],
               'TA': ['Sal'], 'CO2': ['Sal', 'TA', 'DIC', 'Temp'], 'Nutrients_PC1': [],
               'NEC': ['Nutrients_PC1', 'Light', 'pHsw', 'Omega', 'Chla', 'CO2', 'Temp']}

    for var in parents:
        idx = nodes.index(var)
        pas = parents[var]
        if pas:
            lr_model = LinearRegression()
            rows = data[pas + [var]].notna().all(axis=1)
            X, y = data[pas][rows], data[var][rows]
            lr_model.fit(X, y)
            # mu_true[idx] = lr_model.intercept_
            for par in pas:
                pass
                # theta_true[par][var] = lr_model.coef_[pas.index(par)]
                # print(float(lr_model.intercept_))
            if var == "NEC":
                y_bias = float(lr_model.intercept_)
            # C_true[idx, idx] = np.var(y - lr_model.predict(X))
        else:
            pass
            # print(pas + [var])
            # rows = data[pas + [var]].notna().all(axis=1)
            # mu_true[idx] = np.mean(data[var][rows])
            # C_true[idx, idx] = np.var(data[var][rows])
    
    
    # C_true = np.array([ [1.2e-2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    #                     [0.0, 1.6e-2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    #                     [0.0, 0.0, 1.6e-2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    #                     [0.0, 0.0, 0.0, 1.0e-2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    #                     [0.0, 0.0, 0.0, 0.0, 2.0e-2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    #                     [0.0, 0.0, 0.0, 0.0, 0.0, 1.6e-2, 0.0, 0.0, 0.0, 0.0, 0.0],
    #                     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.6e-2, 0.0, 0.0, 0.0, 0.0],
    #                     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.8e-2, 0.0, 0.0, 0.0],
    #                     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.001, 0.0, 0.0],
    #                     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.6e-3, 0.0],
    #                     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.5e-2],
    #                     ])
    C_true = 2e-2 * np.eye(len(nodes))

    theta_true = ddict(lambda: ddict(lambda: 0.0))

    theta_true["Light"]["Temp"] = 0.08336954980497932
    theta_true["Temp"]["Sal"] = -0.4809837373167684
    theta_true["Sal"]["DIC"] = 0.4777168829735183
    theta_true["Sal"]["TA"] = 0.5457734531124397
    theta_true["Temp"]["Omega"] = 0.5182253589055726
    theta_true["Sal"]["Omega"] = 0.03507218718555735 
    theta_true["DIC"]["Omega"] = -1.1056652215053286
    theta_true["TA"]["Omega"] = 1.6104231541803835
    theta_true["Light"]["Chla"] = -0.15106684218500258
    theta_true["Temp"]["Chla"] = -0.04451583134247557
    theta_true["Nutrients_PC1"]["Chla"] = -0.07690378415962325
    theta_true["Temp"]["pHsw"] = -0.7482789216296077 
    theta_true["Sal"]["pHsw"] = 0.013001179873522933
    theta_true["TA"]["pHsw"] = 0.7676261914081877
    theta_true["DIC"]["pHsw"] = -0.5879618774787132
    theta_true["Temp"]["CO2"] = 0.8613318110706953  
    theta_true["Sal"]["CO2"] = 0.04051812201172802
    theta_true["DIC"]["CO2"] = 0.5700488513842487
    theta_true["TA"]["CO2"] = -0.596251974686561
    theta_true["Light"]["NEC"] = 0.0322460348829162
    theta_true["Temp"]["NEC"] = 5.227658403563992
    theta_true["Omega"]["NEC"] = -2.343629162533968 
    theta_true["Chla"]["NEC"] = 0.13182892043084415
    theta_true["Nutrients_PC1"]["NEC"] =  0.09881771775808317
    theta_true["pHsw"]["NEC"] = 2.0492558654639
    theta_true["CO2"]["NEC"] = -2.5146414696724295 
             

    Para_true = [theta_true, C_true]

    candidate_Rh_list = [("DIC",), ("TA",), ("Omega",), ("Chla",), ("Nutrients_PC1",), ]

    Limit_dict = ddict(lambda: [-1, 1])
    M = np.array([[1.0], 
                  [-1.0]])
    d = np.array([2.0, -0.5]).reshape((-1, 1))- M.dot(y_bias)
    target_domin = [M, d]
    binary_edge = []

    N_data = 1000

    val_times = 1000

    times = 10
    
    seed_list = np.random.choice(np.arange(times*50), times, replace=False).tolist()
    
    succ_prob_list = []
    succ_cnt_list = []
    est_err_list = []
    time_list = []
    Y_values = []
    def evaluate_Y(Y_value):
        sign = list((M.dot(np.array(Y_value)).reshape(-1,1) < d).ravel())
        for s in sign:
            if not s:
                return 0
        return 1

    

    for rnd_seed in tqdm(seed_list):
        np.random.seed(rnd_seed) 
        bermuda = RhSolver(
                nodes = nodes, 
                Para_true = Para_true, 
                Limit_dict = Limit_dict,
                binary_edge = binary_edge,
                target_domin = target_domin,
                N_data = N_data,
                val_times = val_times,
                candidate_Rh_list=candidate_Rh_list,
                nodes_stage=nodes_stage,
                evaluate_func=evaluate_Y,
            )
        
        succ_prob, ttime = bermuda.Rh_learning()
        
        succ_freq = bermuda.R100_Freq(100)
        succ_cnt_list.append(succ_freq)

        Y_values.extend(bermuda.node_values["NEC"][-100:])

        succ_prob_list.append(succ_prob)
        time_list.append(ttime)
        est_err = 0.0
        for n1 in bermuda.nodes:
            tmp = 0.0
            for n2 in bermuda.parents[n1]:
                tmp += (bermuda.theta_true[n2][n1] - bermuda.theta_estimator[n2][n1]) ** 2
            est_err += np.sqrt(tmp) 
        est_err_list.append(est_err)


    print(succ_prob_list)
    print("Success probability:\t", np.mean(succ_prob_list), '+-', np.std(succ_prob_list))
    # print("Nature Success probability:\t", np.mean(nature_succ_prob_list), '+-', np.std(nature_succ_prob_list))
    print("Estimation error:\t", np.mean(est_err_list), '+-', np.std(est_err_list))
    print("100 Rounds success count:\t", np.mean(succ_cnt_list), '+-', np.std(succ_cnt_list))
    print("Avg. running time:\t", np.mean(time_list), '+-', np.std(time_list))


    # Plot_Bermuda_Y_and_S_after(Y_values)











