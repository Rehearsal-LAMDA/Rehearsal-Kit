import matplotlib.pyplot as plt
import time
import pandas as pd
import copy
from tqdm import tqdm
from scipy.io import loadmat
# from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.preprocessing import StandardScaler
from collections import defaultdict as ddict
import cvxopt as cvx
from sklearn.feature_selection import mutual_info_regression



class FixEnd():
    def __init__(self, nodes, Para_true, task_para, binary_edge, val_times, nodes_stage, region_center, obs_sam_nums, evaluate_func):
        self.A_true, self.B_true, self.C_true, self.noise_type = Para_true
        assert self.noise_type in ["gaussian", "uniform", "laplace"]
        self.A_mask = (self.A_true != 0).astype(float)  # float型 mask，可用于乘法
        self.B_mask = (self.B_true != 0).astype(float)
        self.ins_mat = np.linalg.inv(np.eye(*self.A_true.shape) - self.A_true)
        self.t0, self.T = task_para
        self.nodes = nodes
        self.bi_edge = binary_edge
        self.nodes_X, self.nodes_Z, self.nodes_Y = nodes_stage
        self.B_true_alter = self.B_true.copy()
        self.B_true_alter[:len(self.nodes_X)+len(self.nodes_Z), :] = 0.0
        self.A_true_alter = self.A_true.copy()
        self.A_true_alter[:len(self.nodes_X)+len(self.nodes_Z), :] = 0.0
        self.ins_mat_alter = np.linalg.inv(np.eye(*self.A_true.shape) - self.A_true_alter)
        # print(np.abs(np.linalg.eigvals(self.ins_mat.dot(self.B_true))))
        # print(np.abs(np.linalg.eigvals(self.ins_mat_alter.dot(self.B_true_alter))))
        assert max(np.abs(np.linalg.eigvals(self.ins_mat.dot(self.B_true)))) < 1, "Non-stationary original time series"
        assert max(np.abs(np.linalg.eigvals(self.ins_mat_alter.dot(self.B_true_alter)))) < 1, "Non-stationary altered time series"
        self.val_times = val_times
        self.Ex = np.hstack([np.eye(len(self.nodes_X)), np.zeros((len(self.nodes_X), len(self.nodes_Z)+len(self.nodes_Y)))]).T
        self.Ez = np.hstack([np.zeros((len(self.nodes_Z), len(self.nodes_X))), np.eye(len(self.nodes_Z)), np.zeros((len(self.nodes_Z), len(self.nodes_Y)))]).T
        self.Ey = np.hstack([np.zeros((len(self.nodes_Y), len(self.nodes_X)+len(self.nodes_Z))), np.eye(len(self.nodes_Y))]).T
        self.U = np.linalg.inv(np.eye(len(nodes)) - (self.Ex.dot(self.Ex.T) + self.Ey.dot(self.Ey.T)).dot(self.A_true)).dot(self.Ez)
        self.C = np.linalg.inv(np.eye(len(nodes)) - (self.Ex.dot(self.Ex.T) + self.Ey.dot(self.Ey.T)).dot(self.A_true)).dot(self.Ex.dot(self.Ex.T) + self.Ey.dot(self.Ey.T))
        self.Gamma = self.C.dot(self.B_true)
        self.U_tilde = np.linalg.inv(np.eye(len(nodes)) - (self.Ey.dot(self.Ey.T)).dot(self.A_true)).dot(self.Ez)
        self.C_tilde = np.linalg.inv(np.eye(len(nodes)) - (self.Ey.dot(self.Ey.T)).dot(self.A_true)).dot(self.Ey.dot(self.Ey.T))
        self.Gamma_tilde = self.C_tilde.dot(self.B_true)
        self.X_mat = np.linalg.inv(np.eye(len(nodes)) - (self.Ey.dot(self.Ey.T)).dot(self.A_true)).dot(self.Ex)
        self.o = region_center
        self.obs_sam_nums = obs_sam_nums
        self.evaluate_Y = evaluate_func
        node_values = self.Generate_historical_data(self.obs_sam_nums)
        self.Init_Est(node_values)
        # print(self.B_true)
        # print(self.B_est)
        # print(np.linalg.norm(self.A_true-self.A_est, ord='f'))
        # print(np.linalg.norm(self.B_true-self.B_est, ord='f'))
        self.ins_mat_hat = np.linalg.inv(np.eye(*self.A_est.shape) - self.A_est)
        self.B_true_alter_hat = self.B_est.copy()
        self.B_true_alter_hat[:len(self.nodes_X)+len(self.nodes_Z), :] = 0.0
        self.A_true_alter_hat = self.A_true.copy()
        self.A_true_alter_hat[:len(self.nodes_X)+len(self.nodes_Z), :] = 0.0
        self.ins_mat_alter_hat = np.linalg.inv(np.eye(*self.A_est.shape) - self.A_est)
        self.U_hat = np.linalg.inv(np.eye(len(nodes)) - (self.Ex.dot(self.Ex.T) + self.Ey.dot(self.Ey.T)).dot(self.A_est)).dot(self.Ez)
        self.C_hat = np.linalg.inv(np.eye(len(nodes)) - (self.Ex.dot(self.Ex.T) + self.Ey.dot(self.Ey.T)).dot(self.A_est)).dot(self.Ex.dot(self.Ex.T) + self.Ey.dot(self.Ey.T))
        self.Gamma_hat = self.C.dot(self.B_est)
        self.U_tilde_hat = np.linalg.inv(np.eye(len(nodes)) - (self.Ey.dot(self.Ey.T)).dot(self.A_est)).dot(self.Ez)
        self.C_tilde_hat = np.linalg.inv(np.eye(len(nodes)) - (self.Ey.dot(self.Ey.T)).dot(self.A_est)).dot(self.Ey.dot(self.Ey.T))
        self.Gamma_tilde_hat = self.C_tilde.dot(self.B_est)
        self.X_mat_hat = np.linalg.inv(np.eye(len(nodes)) - (self.Ey.dot(self.Ey.T)).dot(self.A_est)).dot(self.Ex)
        # print(self.X_mat_hat-self.X_mat)
        # print(self.Gamma_tilde_hat-self.Gamma_tilde)
        # print(self.U_tilde_hat-self.U_tilde)

    def Noise_sampler(self, dim, num, noise_type = "gaussian"):
        assert noise_type in ["gaussian", "laplace", "uniform"], "Noise type not support"
        if noise_type == "gaussian":
            return np.random.multivariate_normal(np.zeros((dim, )), self.C_true, num) 
    
        elif noise_type == "laplace":
            marginal_var = np.diag(self.C_true)
            scales = np.sqrt(marginal_var / 2)
            return np.random.laplace(loc=0.0, scale=scales, size=(num, dim))
        
        elif noise_type == "uniform":
            z = np.random.uniform(low=-1.0, high=1.0, size=(num, dim))
            L = np.linalg.cholesky(self.C_true)
            x = z @ L.T 

            return x

    def Generate_historical_data(self, obs_sam_nums):
        v0 = self.Noise_sampler(len(self.nodes), 1, self.noise_type)[0].reshape(-1, 1)
        nodes_values = [v0, ]
        for _ in range(obs_sam_nums):
            v_past = nodes_values[-1] 
            v_current = self.ins_mat.dot(self.B_true.dot(v_past) + self.Noise_sampler(len(self.nodes), 1, self.noise_type)[0].reshape(-1, 1))
            nodes_values.append(v_current)
        return nodes_values

    def Init_Est(self, nodes_values):
        n = nodes_values[0].shape[0]  
        T = len(nodes_values) - 1     
        
        A = np.zeros((n, n))
        B = np.zeros((n, n))
        
        for i in range(n): 
            S_Ai = np.where(self.A_mask[i])[0].tolist()
            S_Bi = np.where(self.B_mask[i])[0].tolist()
            num_A = len(S_Ai)
            num_B = len(S_Bi)
            
            y = np.zeros(T)
            X = np.zeros((T, num_A + num_B))
            
            for t in range(1, T + 1): 
                current = nodes_values[t]    # v_t
                lagged = nodes_values[t-1]   # v_{t-1}

                y[t-1] = current[i, 0]
                X_a = [current[j, 0] for j in S_Ai]
                X_b = [lagged[k, 0] for k in S_Bi]
                X[t-1, :] = X_a + X_b
            if X.size == 0:
                continue
            XtX = X.T @ X
            if np.linalg.matrix_rank(XtX) < XtX.shape[0]:
                theta = np.linalg.pinv(XtX) @ X.T @ y
            else:
                theta = np.linalg.inv(XtX) @ X.T @ y
            if num_A > 0:
                A[i, S_Ai] = theta[:num_A]
            if num_B > 0:
                B[i, S_Bi] = theta[num_A:]
        # self.A_est = np.round(A, 2)
        # self.B_est = np.round(B, 2)
        self.A_est = A
        self.B_est = B

    def AUF_prob(self, ):
        y_dict = ddict(lambda: [])
        sum_y_list = []
        v0 = self.Noise_sampler(len(self.nodes), 1, self.noise_type)[0].reshape(-1, 1)
        for j in tqdm(range(self.val_times)):
            nodes_values = [v0, ]
            for t in range(self.t0):
                v_past = nodes_values[-1] 
                v_current = self.ins_mat.dot(self.B_true.dot(v_past) + self.Noise_sampler(len(self.nodes), 1, self.noise_type)[0].reshape(-1, 1))
                nodes_values.append(v_current)
            
            T = self.T
            M_whole = [self.Ey.T.dot(np.eye(len(self.nodes))).dot(self.X_mat_hat)] + [self.Ey.T.dot(np.linalg.matrix_power(self.Gamma_hat, i)).dot(self.X_mat_hat) for i in range(1, T)]
            N_whole = [self.Ey.T.dot(np.eye(len(self.nodes))).dot(self.Gamma_tilde_hat)] + [self.Ey.T.dot(np.linalg.matrix_power(self.Gamma_hat, i)).dot(self.Gamma_tilde_hat) for i in range(1, T)]
            tmp_mat = [np.eye(len(self.nodes))] + [np.linalg.matrix_power(self.Gamma_hat, i) for i in range(1, T)] 
            core_mat = [self.Ey.T.dot(np.sum(tmp_mat[:i+1], axis=0)) for i in range(T+1)]
            o = self.o.copy()
            for t in range(self.t0, self.t0+self.T):
                v_past = nodes_values[-1] 
                epsilon_t = self.Noise_sampler(len(self.nodes), 1, self.noise_type)[0].reshape(-1, 1)
                x_t = self.Ex.T.dot(self.ins_mat.dot(self.B_true.dot(v_past) + epsilon_t))
                M = np.sum(M_whole[:T], axis=0)/self.T
                N = np.sum(N_whole[:T], axis=0)/self.T
                H = np.hstack([core_mat[i].dot(self.U_hat) for i in range(T-1)] + [core_mat[T-1].dot(self.U_tilde_hat)])/self.T
                b = o-(M.dot(x_t) + N.dot(v_past))
                # z_t_rh = np.linalg.pinv(H.T.dot(H)).dot(H.T.dot(b))[-len(self.nodes_Z):, :]
                z_t_rh = H.T.dot(np.linalg.inv(H.dot(H.T))).dot(b)[-len(self.nodes_Z):, :]
                epsilon_t[:len(self.nodes_X), :] = x_t
                epsilon_t[len(self.nodes_X):len(self.nodes_X)+len(self.nodes_Z), :] = np.array(z_t_rh).reshape(-1, 1)
                y_t = self.Ey.T.dot(self.ins_mat_alter.dot(self.B_true_alter.dot(v_past) + epsilon_t))
                y_dict[t].append(y_t)
                epsilon_t[-len(self.nodes_Y):, :] = y_t
                nodes_values.append(epsilon_t)
                T -= 1
                o -= y_t/self.T
            cache = nodes_values[-self.T:]
            sum_y = np.sum([v_value[-len(self.nodes_Y):, 0].reshape(-1, 1) for v_value in cache], axis=0)
            sum_y_list.append(sum_y/self.T)

        success_count = 0
        for sum_y in sum_y_list:
            success_count += self.evaluate_Y(sum_y)
        return success_count/self.val_times, sum_y_list
 




