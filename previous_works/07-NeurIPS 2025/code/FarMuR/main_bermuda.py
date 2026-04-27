from FarSight_Slover import FixEnd
import pandas as pd
from tqdm import tqdm
from scipy.io import loadmat
import numpy as np
from sklearn.preprocessing import StandardScaler
 


if __name__ == "__main__": 
    execution_time = []
    np.random.seed(19991115)
    data_mat = loadmat('../data_bermuda/SEM_data.mat')
    for key in ['__header__', '__version__', '__globals__', 'Site', 'Lat', 'Lon', 'Year', 'Month', 'Day']:
        data_mat.pop(key)
    data = pd.DataFrame({key: value.reshape(-1) for key, value in data_mat.items()})
    scaler = StandardScaler()
    # data = (data - data.mean()) / data.std()
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
   
    # 初始化 A_true 为全零矩阵
    A_true = np.zeros((len(nodes), len(nodes)))
    B_true = np.zeros((len(nodes), len(nodes)))

    # 变量索引映射
    idx = {var: i for i, var in enumerate(nodes)}

    instantaneous_theta_true = {
        "Light": {"Temp": 0.08336954980497932, "Chla": -0.15106684218500258, "NEC": 0.0322460348829162},
        "Temp": {"Sal": -0.4809837373167684, "Omega": 0.5182253589055726, "Chla": -0.04451583134247557, 
                "pHsw": -0.7482789216296077, "CO2": 0.8613318110706953, "NEC": 5.227658403563992},
        "Sal": {"DIC": 0.4777168829735183, "TA": 0.5457734531124397, "Omega": 0.03507218718555735, 
                "pHsw": 0.013001179873522933, "CO2": 0.04051812201172802},
        "DIC": {"Omega": -1.1056652215053286, "pHsw": -0.5879618774787132, "CO2": 0.5700488513842487},
        "TA": {"Omega": 1.6104231541803835, "pHsw": 0.7676261914081877, "CO2": -0.596251974686561},
        "Nutrients_PC1": {"Chla": -0.07690378415962325, "NEC": 0.09881771775808317},
        "Omega": {"NEC": -2.343629162533968},
        "Chla": {"NEC": 0.13182892043084415},
        "pHsw": {"NEC": 2.0492558654639},
        "CO2": {"NEC": -2.5146414696724295}
    }

    lagged_theta_true = {
        "Light": {"Light":0.6},
        "Temp": {"Temp":0.6, "DIC":-0.1,},
        "Sal": {"Sal":0.6, "DIC":0.23, "TA":0.25},
        "DIC": {},
        "TA": {"Chla":-0.1},
        "Nutrients_PC1": {},
        "Omega": {},
        "Chla": {},
        "pHsw": {},
        "CO2": {"NEC": -1.1},
        "NEC": {"NEC": 0.6}
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
    print(B_true)

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
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.5e-1],
                        ])

    # noise_type = 'gaussian'
    noise_type = 'laplace'
    Para_true = [A_true, B_true, C_true, noise_type]

    # # print(d)
    binary_edge = []



    val_times = 100
    obs_sam_nums = 1000
    task_para = [100, 9]
    
    times = 1
    
    seed_list = np.random.choice(np.arange(times*50), times, replace=False).tolist()
    # print(seed_list)
    
    succ_prob_list = []

    region_center = np.array([1.95, ]).reshape(-1, 1)
    def evaluate_Y(Y_value):
        # sign = np.linalg.norm(np.array(Y_value).ravel() - np.array([]) ) <= .8
        sign = 1.9 <= float(Y_value) <= 2.0
        return 1 if sign else 0
    
    for rnd_seed in tqdm(seed_list):
        np.random.seed(rnd_seed) 
        # fe = FixEnd(
        #         nodes = nodes, 
        #         Para_true = Para_true, 
        #         task_para = task_para,
        #         binary_edge = binary_edge,
        #         val_times = val_times,
        #         nodes_stage=nodes_stage
        #     )
        fe = FixEnd(
                nodes = nodes, 
                Para_true = Para_true, 
                task_para = task_para,
                binary_edge = binary_edge,
                val_times = val_times,
                nodes_stage=nodes_stage,
                region_center=region_center,
                obs_sam_nums=obs_sam_nums,
                evaluate_func=evaluate_Y
            )

        succ_prob, sum_y_list = fe.AUF_prob()
        succ_prob_list.append(succ_prob)


    print("Data: Bermuda", "\tNoise type:", noise_type, "\tWindow length:", task_para[1], "\tApproach: FarSight Alg. 2")
    print("Success probability:\t", np.mean(succ_prob_list), '+-', np.std(succ_prob_list))








