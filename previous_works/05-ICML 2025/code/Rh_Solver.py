# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import random
import numpy as np
from scipy.optimize import minimize
from scipy.linalg import solve
import itertools
from collections import defaultdict as ddict
import copy
from utils import find_total_costs_to_node
from tqdm import tqdm
from scipy.stats import norm
import warnings
import time
import mpmath
mpmath.mp.dps = 50

DATA_ACCURACY = 2

class RhSolver():
    def __init__(self, nodes, Para_true, Limit_dict, binary_edge, target_domin, N_data, val_times, candidate_Rh_list, nodes_stage, evaluate_func, cir_center = None, cir_r = None, max_iters = 200, theta_hat_init=None):
        self.theta_true, self.C_true = Para_true
        self.nodes = nodes # should in topological order
        self.bi_edge = binary_edge
        self.parents = ddict(lambda: [])
        self.sons = ddict(lambda: [])
        for parent, childern in self.theta_true.items():
            childern_list = list(childern)
            self.sons[parent] = childern_list
            for node in childern_list:
                self.parents[node].append(parent)
        self.Limit_dict = Limit_dict
        self.nodes_X, self.nodes_Z, self.nodes_Y = nodes_stage
        index_map = {node: index for index, node in enumerate(self.nodes)}
        self.nodes_Z = sorted(self.nodes_Z, key=lambda x: index_map[x])
        self.iters = max_iters
        self.candidate_Rh_list = candidate_Rh_list
        self.M, self.d  = target_domin
        self.N_data = N_data
        self.val_times = val_times
        self.node_values = self.Generate_historical_data()
        self.theta_estimator = ddict(lambda: ddict(lambda: 0.0))
        self.C_estimator = np.diag([0.0] * len(nodes))
        self.evaluate_Y = evaluate_func
        self.cir_center = cir_center
        self.cir_r = cir_r
    
    def Generate_historical_data(self, ):
        init_values = ddict(lambda x:[])
        Noise_mat = np.random.multivariate_normal(np.zeros((len(self.nodes), )), self.C_true, self.N_data)
        for node in self.nodes:
            idx = self.nodes.index(node)
            noise = Noise_mat[:, idx]
            init_values[node] = noise
            parents = self.parents[node]
            for parent in parents:
                init_values[node] += self.theta_true[parent][node] * init_values[parent]
        for node in self.nodes:
            init_values[node] = list(init_values[node])
        return init_values
    
    def LSE_Estimator(self, X, y, reg = 0.0):
        M = X.shape[0]
        if reg < 0:
            raise KeyError("Regularization term must be non-negative")
        feature_dim = X.shape[1]
        beta_estimator = np.linalg.pinv(X.T.dot(X) + reg*np.eye(feature_dim)).dot(X.T.dot(y))
        sigma2_estimator = (1/M) * (y-X.dot(beta_estimator)).T.dot(y-X.dot(beta_estimator))
        return list(np.round(beta_estimator, DATA_ACCURACY).ravel()), float(sigma2_estimator)

    def Rh_learning(self, ):
        # first, using historical data to estimate SRM parameters
        for node in self.nodes:
            parents = self.parents[node]
            node_id = self.nodes.index(node)
            X = np.zeros((self.N_data, len(parents)))
            for idx, parent in enumerate(parents):
                X[:, idx] = self.node_values[parent]

            y = np.array(self.node_values[node]).reshape(-1, 1)

            if not parents:
                sigma2 = float((1/y.shape[0]) * y.T.dot(y))
                self.C_estimator[node_id, node_id] = np.array(sigma2, ndmin=0)
            else:
                thetas, sigma2 = self.LSE_Estimator(X, y)
                self.C_estimator[node_id, node_id] = np.array(sigma2, ndmin=0)
                for idx, parent in enumerate(parents):
                    self.theta_estimator[parent][node] = thetas[idx]

        for node_pair in self.bi_edge:
            node1, node2 = node_pair
            id1 = self.nodes.index(node1)
            id2 = self.nodes.index(node2)
            node1_value = np.array(self.node_values[node1]) 
            node2_value = np.array(self.node_values[node2])
            res1 = node1_value
            res2 = node2_value
            for parent in self.parents[node1]:
                res1 -= self.theta_estimator[parent][node1] * np.array(self.node_values[parent])
            for parent in self.parents[node1]:
                res2 -= self.theta_estimator[parent][node2] * np.array(self.node_values[parent])
                
            self.C_estimator[id1, id2] = np.cov(res1, res2)[0, 1]
            self.C_estimator[id2, id1] = np.cov(res1, res2)[0, 1]
        
        # then, use the learned influence to make decisions        
        Noise_evaluate_mat = np.random.multivariate_normal(np.zeros((len(self.nodes), )), self.C_true, self.val_times+1)
        x_t = np.zeros((len(self.nodes_X), ))
        for node in self.nodes_X:
            idx = self.nodes.index(node)
            self.node_values[node].append(Noise_evaluate_mat[0, idx])
            x_t[self.nodes_X.index(node)] = self.node_values[node][-1]
        x_t = x_t.reshape(-1, 1)
        # Rh_Z_dict, tot_time = self.choose_Rh(x_t, self.theta_true, self.C_true)
        Rh_Z_dict, tot_time = self.choose_Rh(x_t, self.theta_estimator, self.C_estimator)
        un_Rh_set = set(self.nodes_Z) - set(Rh_Z_dict)
        un_Rh_Z = [node for node in self.nodes_Z if node in un_Rh_set]

        for node in list(Rh_Z_dict):
            self.node_values[node].append(Rh_Z_dict[node])
        success_prob = self.evaluate_xi(un_Rh_Z, Noise_evaluate_mat)
        return success_prob, tot_time

    def choose_Rh(self, x_t, theta, Sigma):
        result_dict = dict()
        sol_list = []
        time_s = time.time()
        for idx, Rh_list in enumerate(self.candidate_Rh_list):
            y_len = len(self.nodes_Y)
            z_len = len(Rh_list)
            theta_hat_copy = copy.deepcopy(theta)
            for parent in theta:
                for son in theta[parent]:
                    if son in Rh_list:
                        theta_hat_copy[parent].pop(son, [])
            mat_C = np.zeros((len(self.nodes_Y), len(self.nodes)))
            mat_B = np.zeros((len(self.nodes_Y), len(Rh_list)))
            mat_A = np.zeros((len(self.nodes_Y), len(self.nodes_X)))
            for Y in self.nodes_Y:
                idx2 = self.nodes_Y.index(Y)
                mat_C[idx2][self.nodes.index(Y)] = 1.0
                muti_para_dict = find_total_costs_to_node(theta_hat_copy, Y)
                for node in muti_para_dict:
                    if node in Rh_list:
                        mat_B[idx2][Rh_list.index(node)] = muti_para_dict[node]
                    elif node in self.nodes_X:
                        mat_A[idx2][self.nodes_X.index(node)] = muti_para_dict[node]
                    else:
                        mat_C[idx2][self.nodes.index(node)] = muti_para_dict[node]
            z_l = []
            z_r = []
            for idy, node in enumerate(Rh_list):
                z_l.append(self.Limit_dict[node][0])
                z_r.append(self.Limit_dict[node][1])
            z_l = np.array(z_l).reshape(-1, 1)
            z_r = np.array(z_r).reshape(-1, 1)
            if y_len == 1:
                lamd = mat_C.dot(Sigma.dot(mat_C.T))
                M_mat = self.M/np.sqrt(lamd)
                d_vec = self.d/np.sqrt(lamd)
                K_mat = M_mat.dot(mat_B)
                b_vec = M_mat.dot(mat_A.dot(x_t)) - d_vec
                Rh_Z_dict = dict()
                b = (list(b_vec.ravel())[0] - list(b_vec.ravel())[1])/2
                k = K_mat[1, :].reshape(-1, 1)
                m = k.T.dot((k>=0)*z_l + (k<0)*z_r)
                M = k.T.dot((k<0)*z_l + (k>=0)*z_r)
                if b <= m:
                    z_star = (k>=0) * (z_l) + (k<0) * (z_r)
                elif b>= M:
                    z_star = (k>=0) * (z_r) + (k<0) * (z_l)
                else:
                    z_star = np.zeros((len(Rh_list), 1))
                    for z_id in range(z_len):
                        l_j = z_l[z_id, 0]
                        r_j = z_r[z_id, 0]
                        if not k[z_id, 0]:
                            continue
                        elif l_j <= b/k[z_id, 0] and b/k[z_id, 0] <= r_j:
                            z_star[z_id, 0] = b/k[z_id, 0]
                            break
                        else:
                            z_star[z_id, 0] = l_j * (b/k[z_id, 0] < l_j) + r_j * (b/k[z_id, 0] >= r_j) 
                            b = b - z_star[z_id, 0]
                for idy, node in enumerate(Rh_list):
                    ans = list(z_star.ravel())
                    Rh_Z_dict[node] = ans[idy]
                result_dict[idx] = Rh_Z_dict
                
                x1 = np.float64(k.T.dot(z_star)-list(b_vec.ravel())[0])
                x2 = np.float64(k.T.dot(z_star)+list(b_vec.ravel())[1])
                result_fun = norm.cdf(x1) - norm.cdf(x2)
                sol_list.append(result_fun)
                tot_time = time.time() - time_s
            else:
                ## codes below just for circular regions
                eign_val, Q  = np.linalg.eig(mat_C.dot(Sigma.dot(mat_C.T)))
                Q = Q.T
                Q = np.linalg.inv(Q)
                Lam = np.diag(1/np.sqrt(eign_val))
                M_mat = np.vstack((np.eye(Q.shape[0]), -np.eye(Q.shape[0]))).dot(Lam).dot(Q)
                d_vec = np.vstack((np.eye(Q.shape[0]), np.eye(Q.shape[0]))).dot(Lam).dot((self.cir_r/np.sqrt(Q.shape[0]))*np.ones((Q.shape[0], 1)) )+ np.vstack((np.eye(Q.shape[0]), -np.eye(Q.shape[0]))).dot(Lam).dot(Q.dot(self.cir_center.reshape(-1, 1)))
                ## codes above just for circular regions

                K_mat = M_mat.dot(mat_B)
                b_vec = M_mat.dot(mat_A.dot(x_t)) - d_vec       
                Rh_Z_dict = dict()
                x0 = np.zeros((len(Rh_list), 1)).reshape(-1, 1)
                x_iter = x0
                tot_time = 0.0
                time3 = time.time()
                for _ in range(self.iters):
                    x_old = x_iter
                    grad = self.compute_grad(x_old, K_mat, b_vec)
                    hessian = self.compute_hessian(x_old, K_mat, b_vec)
                    indices = np.where((np.isclose(x_old, z_l, atol=1e-8) & (grad > 0)) | (np.isclose(x_old, z_r, atol=1e-8) & (grad < 0)))[0]

                    for i in range(z_len):
                        for j in range(z_len):
                            if i == j:
                                continue 
                            if i in indices or grad[j, 0] > 0: 
                                hessian[i, j] = 0.0 
                    
                    time1 = time.time()
                    D = np.linalg.inv(hessian)
                    time2 = time.time()
                    tot_time += time2-time1
                    beta=0.9
                    sig = 0.001
                    p = D.dot(grad)
                    curr_loss = self.loss_function(x_old, K_mat, b_vec)
                    for m in range(1000):
                        alpha = beta ** (m+1)
                        x_app = self.Project(x_old-alpha*p, z_l, z_r)
                        app_loss = self.loss_function(x_app, K_mat, b_vec)
                        tmp = 0.0
                        for i in indices:
                            tmp += grad[i, 0] * (alpha*p[i, 0] + x_old[i, 0] - x_app[i, 0])
                        if curr_loss - app_loss >= sig * tmp:
                            break
                    time1 = time.time()
                    x_iter = self.Project(x_old - alpha*p, z_l, z_r)
                    time2 = time.time()
                    tot_time += time2-time1
                    
                    if np.linalg.norm(x_iter - x_old)<1e-2:
                        break
                for idy, node in enumerate(Rh_list):
                    ans = list(x_iter.ravel())
                    Rh_Z_dict[node] = ans[idy]
                result_dict[idx] = Rh_Z_dict
                
                result_fun = 1.0
                for j in range(y_len):
                    kj = K_mat[j, :].reshape(-1, 1)
                    kjd = -kj 
                    bj = b_vec[j, 0]
                    bjd = b_vec[j + y_len, 0]
                    xj = np.float64(- kj.T.dot(x_iter)-bj)
                    xjd = np.float64(kjd.T.dot(x_iter)+bjd)
                    result_fun *= norm.cdf(xj) - norm.cdf(xjd)
                sol_list.append(result_fun)

        index = sol_list.index(max(sol_list))
        return result_dict[index], tot_time

    def evaluate_xi(self, un_Rh_Z, Noise_evaluate_mat):
        
        success_count = 0
        for j in range(1, self.val_times+1):
            for node in un_Rh_Z:
                idx = self.nodes.index(node)
                tmp = Noise_evaluate_mat[j, idx]  
                parents = self.parents[node]
                for parent in parents:
                    tmp += self.theta_true[parent][node] * self.node_values[parent][-1]
                self.node_values[node].append(tmp)

            for node in self.nodes_Y:
                idx = self.nodes.index(node)
                tmp = Noise_evaluate_mat[j, idx]  
                parents = self.parents[node]
                for parent in parents:
                    tmp += self.theta_true[parent][node] * self.node_values[parent][-1]
                self.node_values[node].append(tmp)
            
            success_count += self.evaluate_Y([[self.node_values[node][-1]] for node in self.nodes_Y])
        return success_count/self.val_times

    def R100_Freq(self, rounds):
        Noise_evaluate_mat = np.random.multivariate_normal(np.zeros((len(self.nodes), )), self.C_true, rounds)
        success_count = 0
        for j in range(rounds):
            x_t = np.zeros((len(self.nodes_X), ))
            for node in self.nodes_X:
                idx = self.nodes.index(node)
                self.node_values[node].append(Noise_evaluate_mat[j, idx])
                x_t[self.nodes_X.index(node)] = self.node_values[node][-1]
            x_t = x_t.reshape(-1, 1)

            # Rh_Z_dict, _ = self.choose_Rh(x_t, self.theta_true, self.C_true)
            Rh_Z_dict, _ = self.choose_Rh(x_t, self.theta_estimator, self.C_estimator)
            un_Rh_set = set(self.nodes_Z) - set(Rh_Z_dict)
            un_Rh_Z = [node for node in self.nodes_Z if node in un_Rh_set]

            for node in list(Rh_Z_dict):
                self.node_values[node].append(Rh_Z_dict[node])

            for node in un_Rh_Z:
                idx = self.nodes.index(node)
                tmp = Noise_evaluate_mat[j, idx]  
                parents = self.parents[node]
                for parent in parents:
                    tmp += self.theta_true[parent][node] * self.node_values[parent][-1]
                self.node_values[node].append(tmp)

            for node in self.nodes_Y:
                idx = self.nodes.index(node)
                tmp = Noise_evaluate_mat[j, idx]  
                parents = self.parents[node]
                for parent in parents:
                    tmp += self.theta_true[parent][node] * self.node_values[parent][-1]
                self.node_values[node].append(tmp)
            
            success_count += self.evaluate_Y([[self.node_values[node][-1]] for node in self.nodes_Y])
        
        return success_count

    def compute_grad(self, z, K, b):
        y_len = len(self.nodes_Y)
        Kz = K @ z  
        b_flat = b.flatten()
        Kz_j = Kz[:y_len, 0]
        Kz_jd = Kz[y_len:, 0]
        b_j = b_flat[:y_len]
        b_jd = b_flat[y_len:]
        xj = - (Kz_j + b_j) 
        xjd = Kz_jd + b_jd

        cdf_xj = norm.cdf(xj)
        cdf_xjd = norm.cdf(xjd)
        pdf_xj = norm.pdf(xj)
        pdf_xjd = norm.pdf(xjd)
        cdf_diff = cdf_xj - cdf_xjd

        factor_grad = np.where(cdf_diff != 0, (pdf_xj - pdf_xjd) / cdf_diff, 0.0)

        kj = K[:y_len, :]
        grad = (factor_grad[:, None] * kj).sum(axis=0, keepdims=True).reshape(-1, 1)

        return grad

    def compute_hessian(self, z, K, b):
        y_len = len(self.nodes_Y)

        Kz = K @ z 
        b_flat = b.flatten()
        Kz_j = Kz[:y_len, 0]
        Kz_jd = Kz[y_len:, 0]
        b_j = b_flat[:y_len]
        b_jd = b_flat[y_len:]

        xj = - (Kz_j + b_j)  # (y_len,)
        xjd = Kz_jd + b_jd   # (y_len,)

        cdf_xj = norm.cdf(xj)
        cdf_xjd = norm.cdf(xjd)
        pdf_xj = norm.pdf(xj)
        pdf_xjd = norm.pdf(xjd)
        cdf_diff = cdf_xj - cdf_xjd

        numerator = (cdf_diff * (xj * pdf_xj - xjd * pdf_xjd) + (pdf_xj - pdf_xjd)**2)
        denominator = cdf_diff**2
        factor_hess = np.where(denominator > 0, numerator / denominator, -1.0) 

        kj = K[:y_len, :] 
        hessian = (factor_hess[:, None, None] * (kj[:, :, None] @ kj[:, None, :])).sum(axis=0)

        return hessian
    
    def loss_function(self, z, K, b):
        y_len = len(self.nodes_Y)
        
        Kz = K @ z 
        b_flat = b.flatten()
        
        Kz_j = Kz[:y_len, 0]
        Kz_jd = Kz[y_len:, 0]
        b_j = b_flat[:y_len]
        b_jd = b_flat[y_len:]
        
        xj = - (Kz_j + b_j)
        xjd = Kz_jd + b_jd 
        
        cdf_diff = norm.cdf(xj) - norm.cdf(xjd)  # (y_len, )
        res = -np.sum(np.log(cdf_diff))
        return res

    def Project(self, z, z_l, z_r):
        return np.clip(z, z_l, z_r)
