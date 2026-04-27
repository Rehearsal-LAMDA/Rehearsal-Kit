from Greedy_Slover import FixEnd
from tqdm import tqdm
import numpy as np



if __name__ == "__main__": 
    execution_time = []
    np.random.seed(114514)


    nodes = ["competitor_feature", "economic_index", "raw_cost", "competitor_raw_cost", "competitor_pricing", "self_pricing", "total_profit", "custom_number"]
    nodes_stage = [["competitor_feature", "economic_index"], ["raw_cost", "competitor_raw_cost", "competitor_pricing", "self_pricing"], ["total_profit", "custom_number"]]

   
    # 初始化 A_true 为全零矩阵
    A_true = np.zeros((len(nodes), len(nodes)))
    B_true = np.zeros((len(nodes), len(nodes)))

    # 变量索引映射
    idx = {var: i for i, var in enumerate(nodes)}

    instantaneous_theta_true = {
        "competitor_feature": {"competitor_raw_cost": 1.0},
        "economic_index": {"raw_cost": 1.0},
        "self_pricing": {"total_profit": 0.9, "custom_number": -0.5},
        "raw_cost": {"competitor_pricing": 0.5, "self_pricing": 2.0, "total_profit": -1.0, "custom_number": 1.6},
        "competitor_raw_cost": {"competitor_pricing": 1.3, "self_pricing": 0.4},
    }

    lagged_theta_true = {
        "economic_index": {"raw_cost": 0.6},
        "competitor_feature": {"competitor_raw_cost": 0.6},
        "competitor_raw_cost": {"competitor_pricing": 0.7, "self_pricing": 0.2},
        "self_pricing": {"total_profit": 0.3,},
        "total_profit": {"economic_index": -0.6},
        "custom_number": {"competitor_feature": -0.6},
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

    C_true =  1e2*np.array([[0.04, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.04, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.06, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.06, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.03, 0.016, 0.0,0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.016, 0.06, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.04, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.12]
                        ])

    noise_type = 'gaussian'
    # noise_type = 'laplace'
    Para_true = [A_true, B_true, C_true, noise_type]


    # # print(d)
    binary_edge = [["competitor_pricing", "self_pricing"]]



    val_times = 1000
    obs_sam_nums = 1000
    task_para = [100, 1]
    
    times = 5
    
    seed_list = np.random.choice(np.arange(times*50), times, replace=False).tolist()
    # print(seed_list)
    
    succ_prob_list = []

    region_center = np.array([1.2, 1.5]).reshape(-1, 1)
    
    def evaluate_Y(Y_value):
        sign = np.linalg.norm(np.array(Y_value).ravel() - np.array([1.2, 1.5]) ) <= .8
        return 1 if sign else 0

    for rnd_seed in tqdm(seed_list):
        np.random.seed(rnd_seed) 
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
        # fe = Base_Solver(
        #         nodes = nodes, 
        #         Para_true = Para_true, 
        #         task_para = task_para,
        #         binary_edge = binary_edge,
        #         val_times = val_times,
        #         nodes_stage=nodes_stage,
        #         evaluate_func=evaluate_Y
        #     )

        succ_prob, sum_y_list = fe.AUF_prob()
        succ_prob_list.append(succ_prob)

    print("Data: Syn", "\tNoise type:", noise_type, "\tWindow length:", task_para[1], "\tApproach: Greedy Alg. 1")
    print("Success probability:\t", np.mean(succ_prob_list), '+-', np.std(succ_prob_list))








