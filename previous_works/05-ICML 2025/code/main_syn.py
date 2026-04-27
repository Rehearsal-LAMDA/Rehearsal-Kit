# -*- coding: utf-8 -*-

from Rh_Solver import *
import time
import pandas as pd
import copy
from tqdm import tqdm
from scipy.io import loadmat
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.preprocessing import StandardScaler
from matplotlib.lines import Line2D


def Plot_Manage_Y_and_S_after(Y1_values, Y2_values, ):
    plt.figure(figsize=(10, 5.8))
    plt.scatter(Y1_values, Y2_values, color="green", s=16, marker="^", zorder=1)
    radius = 1.0
    theta = np.linspace(0, 2 * np.pi, 100)
    x_circle = radius * np.cos(theta)+1.2
    y_circle = radius * np.sin(theta)+1.5
    
    square_vertices = np.array([
        [-radius/np.sqrt(2), -radius/np.sqrt(2)],
        [radius/np.sqrt(2) , -radius/np.sqrt(2)],
        [radius/np.sqrt(2) , radius/np.sqrt(2) ],
        [-radius/np.sqrt(2), radius/np.sqrt(2)]
    ])
    
    def rotation_matrix_2d(angle):
        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)
        return np.array([
            [cos_angle, -sin_angle],
            [sin_angle, cos_angle]
        ])

    angle = np.deg2rad(0)
    
    R = rotation_matrix_2d(angle)
    square_vertices = square_vertices @ R.T + np.array([[1.2, 1.5 ]])
    
    plt.fill(x_circle, y_circle, color='red', alpha=0.4, zorder=2)  
    
    square = plt.Polygon(square_vertices, facecolor='#8B0000', edgecolor='k', alpha=0.8, zorder=3)  # 深红色
    plt.gca().add_patch(square)
    legend_1 = Line2D([0], [1], marker='^', linestyle='', color='green', markerfacecolor='green', alpha=1, markersize=6,label=r"Samples from $\mathbf{Y}$ distribution after altering")
    legend_2 = Line2D([0], [1], marker='o', linestyle='', color='k', markerfacecolor='red', alpha=0.4, markersize=14,label=r"Original desired region $\mathcal{S}$")
    legend_3 = Line2D([0], [0], marker='s', linestyle='', color='k', markerfacecolor='#8B0000', alpha=0.8, markersize=10, label=r"Inner c.r. embedding of $\mathcal{S}$")
    vitrual_legend = [legend_2, legend_3, legend_1] 
    plt.legend(handles=vitrual_legend, fontsize=18, loc='lower left', bbox_to_anchor=(0.01, 0.01))
    plt.gca().xaxis.set_label_position('top')
    plt.gca().xaxis.tick_top()
    plt.xticks([])
    plt.yticks([])
    plt.tick_params(axis="y", which="both", labelleft=False)
    plt.tick_params(axis="x", which="both", labeltop=False)
    plt.xlabel(r'($\mathbf{Y_1}$) TPF Value after Altering', fontsize=22)
    plt.ylabel(r'($\mathbf{Y_2}$) NCT Value after Altering', fontsize=22)
    plt.xlim(-3.5/2, 9.5/2)
    plt.ylim(-0.835, 5.87/2)
    # plt.show()
    plt.savefig('region1_after_alter.pdf', bbox_inches='tight', pad_inches=0.01)





if __name__ == "__main__": 
    np.random.seed(20000603)
    nodes = ['competitor_feature', 'economic_index', 'competitor_raw_cost', 'raw_cost', 'self_pricing',  'competitor_pricing', 'total_profit', 'custom_number']
    theta_true = ddict(lambda: ddict(lambda: 0.0))

    theta_true["competitor_feature"]["competitor_raw_cost"] = 10.0
    theta_true["economic_index"]["raw_cost"] = 10.0
    theta_true["self_pricing"]["total_profit"] = 0.9
    theta_true["self_pricing"]["custom_number"] = -0.9 
    theta_true["raw_cost"]["competitor_pricing"] = 0.5
    theta_true["raw_cost"]["self_pricing"] = 2.0  
    theta_true["competitor_raw_cost"]["competitor_pricing"] = 1.3
    theta_true["competitor_raw_cost"]["self_pricing"] = 0.4
    theta_true["raw_cost"]["total_profit"] = -1.0
    theta_true["raw_cost"]["custom_number"] = 1.6 

    C_true =  np.array([ [0.04, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.04, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.06, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.03, 0.016, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.016, 0.06, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.06, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.04, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.12]
                        ])
    
    nodes_stage = [["competitor_feature", "economic_index"], ["competitor_raw_cost", "competitor_pricing", "self_pricing", "raw_cost"], ["total_profit", "custom_number"]]
    Para_true = [theta_true, C_true]
    

    candidate_Rh_list = [("raw_cost", "self_pricing"),]
    Limit_dict = dict()
    Limit_dict["self_pricing"] = [-6.0, 6.0]
    Limit_dict["raw_cost"] = [-6.0, 6.0]

    M = None
    d = None
    cir_center = np.array([1.2, 1.5])
    cir_r = 1.0
    ## circular region, M and d are determined by o and r when in the main body of the algorithm

    target_domin = [M, d]

    binary_edge = [["competitor_pricing", "self_pricing"]]

    N_data = 100

    val_times = 1000

    times = 100
    seed_list = np.random.choice(np.arange(times*50), times, replace=False).tolist()
 
    succ_prob_list = []
    succ_cnt_list = []
    est_err_list = []
    time_list = []
    
    Y1_values = []
    Y2_values = []

    
    def evaluate_Y(Y_value):
        sign = np.linalg.norm(np.array(Y_value).ravel() - cir_center) <= cir_r
        return 1 if sign else 0

    for rnd_seed in tqdm(seed_list):
        np.random.seed(rnd_seed) 
        toy_eg = RhSolver(
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
                cir_center=cir_center,
                cir_r=cir_r,
            )
        succ_prob, time = toy_eg.Rh_learning()
        
        succ_freq = toy_eg.R100_Freq(100)
        succ_cnt_list.append(succ_freq)
        Y1_values.extend(toy_eg.node_values["total_profit"][-100:])
        Y2_values.extend(toy_eg.node_values["custom_number"][-100:])

        succ_prob_list.append(succ_prob)
        time_list.append(time)
        est_err = 0.0
        for n1 in toy_eg.nodes:
            tmp = 0.0
            for n2 in toy_eg.parents[n1]:
                tmp += (toy_eg.theta_true[n2][n1] - toy_eg.theta_estimator[n2][n1]) ** 2
            est_err += np.sqrt(tmp) 
        est_err_list.append(est_err)


    print("Success probability:\t", np.mean(succ_prob_list), '+-', np.std(succ_prob_list))
    print("Estimation error:\t", np.mean(est_err_list), '+-', np.std(est_err_list))
    print("100 Rounds success count:\t", np.mean(succ_cnt_list), '+-', np.std(succ_cnt_list))
    print("Avg. running time:\t", np.mean(time_list), '+-', np.std(time_list))
    # data = np.array([succ_prob_list, nature_succ_prob_list]).T
    # np.savetxt('bermuda.csv', data, delimiter=',', fmt='%.4f')
    # Plot_Manage_Y_and_S_after(Y1_values, Y2_values,)












