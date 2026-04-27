from FarSight_Slover import FixEnd
# from test import FixEnd
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
from matplotlib.lines import Line2D
from brokenaxes import brokenaxes
import matplotlib.pyplot as plt

from brokenaxes import brokenaxes
import matplotlib.pyplot as plt

def Plot_Bermuda_Y_and_S_after(Y, region_center, r, succ_prob):
    plt.figure(figsize=(10, 6.8))

    bax = brokenaxes(
        xlims=((-12, 12),),
        ylims=((0.0, .3),),
        hspace=0.0,
        despine=False,
        wspace=0.0,
        d=0.015,
    )

    for ax in bax.axs:
        # 设置背景颜色与格点
        ax.set_facecolor((0.8, 0.8, 0.8, 0.15))
        ax.grid(True, alpha=0.5, zorder=1)

        # 清除所有主副刻度的线和标签（关键是 length=0 真的干掉 tickline）
        for axis in ['x', 'y']:
            ax.tick_params(axis=axis, which='both',
                           bottom=False, top=False, left=False, right=False,
                           labelbottom=False, labelleft=False,
                           length=0)

        # 隐藏四个边框线
        # for spine in ax.spines.values():
            # spine.set_visible(False)

    # 正式绘图
    bax.hist(Y, bins=23, density=True, label=r"$\mathbf{Y}$ Distribution after altering")
    bax.axvline(x=region_center - r, color='red')
    bax.axvline(x=region_center + r, color='red')
    bax.fill_between([region_center - r, region_center + r], 0, 0.55, color='red', alpha=0.3,
                     label=r"Desired interval $\mathcal{S}$")
    # legend_1 = Line2D([0], [1], marker='s', linestyle='', alpha=1, markersize=12,label=fr"$Y$ distribution with" + "\n"+"selected alteration")
    # legend_1 = Line2D([0], [1], marker='s', linestyle='', alpha=1, markersize=12,label=fr"$\frac{{1}}{{2}}(Y+Y_+)$ with" + "\n" + "short-sight alteration")
    legend_1 = Line2D([0], [1], marker='s', linestyle='', alpha=1, markersize=12, label=r"$\frac{{1}}{{2}}(Y+Y_+)$ with" + "\n" + "far-sight alteration")
    # legend_1 = Line2D([0], [1], marker='s', linestyle='', alpha=1, markersize=12,label=r"original $Y$ distribution")
    legend_2 = Line2D([0], [1], marker='s', linestyle='', color='k', markerfacecolor='red', alpha=.3, markersize=6,label=r"desired region")
    # legend_3 = Line2D([0], [1], marker='', linestyle='', label=fr"Variance $\operatorname{{Var}}(Y)$ = {np.var(Y):.4f}")
    vitrual_legend = [legend_1, legend_2,]
    # bax.text(-10, 0.18, fr"Variance $\operatorname{{Var}}(Y)$ = {np.var(Y):.4f}", fontsize=14, ) 
    bax.axs[0].text(
        -10, 0.15,
        fr"Variance: {np.var(Y):.2f}" + "\n" + r"$\mathbb{\text{Pr}}_{\text{AUF}}$:" + f"{100*succ_prob:.1f}%",
        fontsize=24,
        bbox=dict(facecolor='lightyellow', edgecolor='black', alpha=0.6, boxstyle='round,pad=0.2')
    )
    # plt.legend(handles=vitrual_legend, fontsize=16, loc='upper right', bbox_to_anchor=(0.99, 0.99))
    plt.legend(handles=vitrual_legend, fontsize=24, loc='upper left', bbox_to_anchor=(-0., 0.99))

    plt.savefig('fe_2.pdf', bbox_inches='tight', pad_inches=0.01)
    plt.show()



if __name__ == "__main__": 
    execution_time = []
    np.random.seed(15)


    nodes = ["X", "Z", "Y"]
    nodes_stage = [["X"], ["Z"], ["Y"]]

   
    # 初始化 A_true 为全零矩阵
    A_true = np.zeros((len(nodes), len(nodes)))
    B_true = np.zeros((len(nodes), len(nodes)))

    # 变量索引映射
    idx = {var: i for i, var in enumerate(nodes)}

    instantaneous_theta_true = {
        "X": {"Z": 0.5, "Y": 0.9},
        "Z": {"Y": 0.6},
    }

    lagged_theta_true = {
        "X": {"X": 0.5, "Y": 0.8},
        "Z": {"Y": -0.8},
        # "Y": {"Y":0.8}
    }


        # 填充 A_true
    for parent, children in instantaneous_theta_true.items():
        for child, value in children.items():
            A_true[idx[child], idx[parent]] = value  # A_true[y, x] 表示 x->y

    # 填充 B_true
    for parent, children in lagged_theta_true.items():
        for child, value in children.items():
            B_true[idx[child], idx[parent]] = value  # B_true[y, x] 表示 x->y

    # print(A_true)
    # print(B_true)

    C_true = np.array([[0., 0.0, 0.0],
                       [0.0, 0., 0.0],
                       [0.0, 0.0, 9],
                      ])

    noise_type = 'gaussian'
    # noise_type = 'laplace'
    Para_true = [A_true, B_true, C_true, noise_type]

    binary_edge = []

    val_times = 10000

    task_para = [100, 3]
    
    times = 1
    
    seed_list = np.random.choice(np.arange(times*50), times, replace=False).tolist()
    # print(seed_list)
    
    succ_prob_list = []

    region_center = np.array([4.0]).reshape(-1, 1)
    

    r = 2.
    def evaluate_Y(Y_value, N):
        sign = (Y_value-region_center)**2 <= r ** 2
        return 1 if sign else 0

    for rnd_seed in tqdm(seed_list):
        np.random.seed(rnd_seed) 
        
        # fe = Base_Solver(
        #         nodes = nodes, 
        #         Para_true = Para_true, 
        #         task_para = task_para,
        #         binary_edge = binary_edge,
        #         val_times = val_times,
        #         nodes_stage=nodes_stage,
        #         evaluate_func=evaluate_Y
        #     )
        fe = FixEnd(
                nodes = nodes, 
                Para_true = Para_true, 
                task_para = task_para,
                binary_edge = binary_edge,
                val_times = val_times,
                nodes_stage=nodes_stage,
                region_center=region_center,
                evaluate_func=evaluate_Y
            )

        succ_prob, sum_y_list = fe.AUF_prob()
        succ_prob_list.append(succ_prob)


    # print(sum_y_list)
    sum_y_list = [float(ele) for ele in sum_y_list]
    # print(sum_y_list)
    print("Success probability:\t", np.mean(succ_prob_list), '+-', np.std(succ_prob_list))
    # Plot_Bermuda_Y_and_S_after(sum_y_list, float(region_center), r, succ_prob)








