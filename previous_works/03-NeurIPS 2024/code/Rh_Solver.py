# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import random
import numpy as np
import itertools
from collections import defaultdict as ddict
import copy
import cvxopt as cvx
from utils import find_total_costs_to_node
from scipy.stats import chi2
from tqdm import tqdm
import warnings
from utils import *

DATA_ACCURACY = 2

class RhSolver():
    def __init__(self, nodes, Para_true, Cost_para, Limit_dict, binary_edge, target_domin, init_n, experts_dict, learning_rate, confidence_level, candidate_Rh_list, nodes_stage, theta_hat_init=None):
        self.theta_true, self.C_true = Para_true
        self.cost_dict = Cost_para[0]
        self.z0_dict = Cost_para[1]
        self.nodes = nodes
        self.bi_edge = binary_edge
        self.init_n = init_n
        self.parents = ddict(lambda: [])
        self.sons = ddict(lambda: [])
        for parent, childern in self.theta_true.items():
            childern_list = list(childern)
            self.sons[parent] = childern_list
            for node in childern_list:
                self.parents[node].append(parent)
        self.topo_nodes = self.topological_sort(self.parents)
        self.node_values = ddict(lambda: [])
        self.Limit_dict = Limit_dict
        
        self.sH = experts_dict
        self.alpha = learning_rate
        self.nodes_X, self.nodes_Z, self.nodes_Y= nodes_stage
        self.candidate_Rh_list = candidate_Rh_list

        self.time_count = 0
        if not theta_hat_init:
            self.theta_hat = self.init_theta(self.init_n)
        else:
            self.theta_hat = theta_hat_init
 
        self.experts = ddict(lambda: ddict(lambda: []))
        
        
        for node in self.topo_nodes:
            the_0 = np.zeros((len(self.parents[node]), ))
            for (idx, parent) in enumerate(self.parents[node]): 
                the_0[idx] = self.theta_hat[parent][node][-1]
            for eta in self.sH[node]:
                self.experts[node][eta].append([copy.deepcopy(the_0), 1/len(self.sH[node])])

        self.M, self.d  = target_domin
        self.cl = confidence_level
        
        

    def init_theta(self, init_n, reg=0.0):
        theta_hat_init = ddict(lambda: ddict(lambda: []))
        init_values = ddict(lambda: np.zeros(init_n, ))
        Noise_init = np.random.multivariate_normal(np.zeros((len(self.nodes), )), self.C_true, init_n)
        time_idx = np.array(list(range(init_n)))
        for node in self.topo_nodes:
            idx = self.nodes.index(node)
            noise = Noise_init[:, idx]
            init_values[node] = noise  
            parents = self.parents[node]
            for parent in parents:
                init_values[node] += self.theta_true[parent][node](time_idx) * init_values[parent]
            self.node_values[node].extend(init_values[node])
            y = init_values[node] 
            est_node_value_init = np.zeros(init_n, )
            if parents:
                X = np.zeros((init_n, len(parents)))
                for idx, parent in enumerate(parents):
                    X[:, idx] = init_values[parent]
                thetas = self.LSE_Estimator(X, y, reg)
                for idx, parent in enumerate(parents):
                    theta_hat_init[parent][node].extend([thetas[idx]]*init_n)
                for parent in parents:
                    est_node_value_init += np.array(theta_hat_init[parent][node]) * init_values[parent]
        self.time_count += init_n
        return theta_hat_init

    def topological_sort(self, parents):
        def visit(node):
            if node not in visited:
                visited.add(node)
                for parent in parents.get(node, []):
                    visit(parent)
                stack.append(node)

        visited = set()
        stack = []

        for node in parents:
            visit(node)

        return stack
            
    def LSE_Estimator(self, X, y, reg = 0.0):
        if reg < 0:
            raise KeyError("Regularization term must be non-negative")
        feature_dim = X.shape[1]
        return list(np.round(np.linalg.pinv(X.T.dot(X) + reg*np.eye(feature_dim)).dot(X.T.dot(y)), DATA_ACCURACY).ravel())
    
    def pipeline(self, rounds):
        victory_count = []
        res_denote = ddict(lambda: [])
        alterations = []
        cost_denote = []
        for _ in tqdm(range(rounds)):
            noise_one_step = np.random.multivariate_normal(np.zeros((len(self.nodes)), ), self.C_true)
            for node in self.nodes_X:
                idx = self.nodes.index(node)
                self.node_values[node].append(noise_one_step[idx])
              
            Rh_Z_dict, cost = self.choose_Rh()
            alterations.append(Rh_Z_dict)
            cost_denote.append(cost)

            un_Rh_set = set(self.nodes_Z) - set(Rh_Z_dict)
            un_Rh_Z = [x for x in self.nodes_Z if x in un_Rh_set]

            for node in list(Rh_Z_dict):
                self.node_values[node].append(Rh_Z_dict[node])
            
            for node in un_Rh_Z:
                idx = self.nodes.index(node)
                tmp = noise_one_step[idx]  
                parents = self.parents[node]
                for parent in parents:
                    tmp += self.theta_true[parent][node](self.time_count) * self.node_values[parent][-1]
                self.node_values[node].append(tmp)

            for node in self.nodes_Y:
                idx = self.nodes.index(node)
                tmp = noise_one_step[idx]  
                parents = self.parents[node]
                for parent in parents:
                    tmp += self.theta_true[parent][node](self.time_count) * self.node_values[parent][self.time_count]
                self.node_values[node].append(tmp)

            victory_count.append(self.evaluate_Y([[self.node_values[node][-1]] for node in self.nodes_Y]))
            self.one_step_OnlineEnsemble(Rh_Z_dict)
            for node in self.topo_nodes:
                res_denote[node].append(sum([(self.theta_true[p][node](self.time_count) - self.theta_hat[p][node][self.time_count]) ** 2 for p in self.parents[node]]))
            self.time_count += 1
        return victory_count, res_denote, alterations, cost_denote


    def choose_Rh(self):
        result_dict = dict()
        cost_list = []
        sol_list = []
        for idx, Rh_list in enumerate(self.candidate_Rh_list):
            theta_hat_copy = copy.deepcopy(self.theta_hat)
            for parent in self.theta_hat:
                for son in self.theta_hat[parent]:
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
            mat_P = self.estimate_probability_ball(mat_C)  
            
            W = np.diag([self.cost_dict[node] for node in Rh_list])
            z0 = np.array([self.z0_dict[node] for node in Rh_list])
            
             
            x_t = np.zeros((len(self.nodes_X), ))
            for node in self.nodes_X:
                x_t[self.nodes_X.index(node)] = self.node_values[node][-1]
            cvx.solvers.options['show_progress'] = False

            Rh_Z_dict = dict()
            val_sign = 0

            try:
                solution = cvx.solvers.qp(P=cvx.matrix(2*W), 
                                        q=cvx.matrix(-2 * z0.T.dot(W)),
                                        G=cvx.matrix(self.M.dot(mat_B)), 
                                        h=cvx.matrix(
                                            self.d - self.M.dot(mat_A).dot(x_t).reshape(-1, 1) - np.linalg.norm(self.M.dot(mat_P), axis=1).reshape(-1, 1)
                                            )
                                        ) 
                
                for idy, node in enumerate(Rh_list):
                    ans = solution['x'][idy]
                    if  ans < self.Limit_dict[node][0]:
                        Rh_Z_dict[node] = self.Limit_dict[node][0]
                    elif ans > self.Limit_dict[node][1]:
                        Rh_Z_dict[node] = self.Limit_dict[node][1]
                    else:
                        Rh_Z_dict[node] = solution['x'][idy]
                
    
                z_xi = np.array(list(Rh_Z_dict.values()))
                cost_list.append((z_xi-z0).T.dot(W).dot((z_xi-z0)))
                sol_list.append(solution['primal objective'])
                result_dict[idx] = Rh_Z_dict
                val_sign = 1
            except:
                solution = cvx.solvers.qp(P=cvx.matrix(2*W), 
                                        q=cvx.matrix(-2 * z0.T.dot(W))
                )
                for idy, node in enumerate(Rh_list):
                    ans = solution['x'][idy]    
                    if  ans < self.Limit_dict[node][0]:
                        Rh_Z_dict[node] = self.Limit_dict[node][0]
                    elif ans > self.Limit_dict[node][1]:
                        Rh_Z_dict[node] = self.Limit_dict[node][1]
                    else:
                        Rh_Z_dict[node] = solution['x'][idy]
                sol_list.append(float('inf'))
                z_xi = np.array(list(Rh_Z_dict.values()))
                cost_list.append((z_xi-z0).T.dot(W).dot((z_xi-z0)))
                result_dict[idx] = Rh_Z_dict
        if not val_sign:
            print("constraints are not satisfiable, optimizing without constriants ...")
            

        index = sol_list.index(min(sol_list))
        return result_dict[index], cost_list[index]



    def estimate_probability_ball(self, mat_C):
        Q = self.C_true
        P = np.linalg.cholesky(mat_C.dot(Q).dot(mat_C.T)) 
        P *= np.sqrt(chi2.ppf(self.cl, len(self.nodes_Y)))
        return P

            
            


    def evaluate_Y(self, Y_value):
        sign = list((self.M.dot(np.array(Y_value)).reshape(-1, 1) < self.d).ravel())
        for s in sign:
            if not s:
                return 0
        return 1


    def one_step_OnlineEnsemble(self, Rh_Z_dict):
        not_update_list = list(Rh_Z_dict)
        upd_list = list(set(self.nodes) - set(not_update_list))
        for node in upd_list:
            if not self.parents[node]:
                continue
            pa_value = np.zeros((len(self.parents[node]), ))
            for (idx, parent) in enumerate(self.parents[node]): 
                pa_value[idx] = self.node_values[parent][-1]
            fringe = dict()
            w_old = []
            loss = []
            for eta in self.sH[node]:
                the_old, the_w = self.experts[node][eta][-1]
                w_old.append(the_w)
                the_new = the_old - eta * (-2 * (self.node_values[node][-1] - the_old.T.dot(pa_value))) * pa_value
                fringe[eta] = [the_new, ]
                eta_loss = the_w * np.exp(-self.alpha * (self.node_values[node][-1] - the_new.T.dot(pa_value)) ** 2)
                loss.append(eta_loss)
                tmp = np.zeros(the_old.shape)
            weight = list(np.array(loss) / (1e-10 + sum(loss)))
            
            
            for eta in self.sH[node]:
                fringe[eta].append(weight[self.sH[node].index(eta)])
                tmp += fringe[eta][0] * fringe[eta][1]
                self.experts[node][eta].append(fringe[eta])

            for (idx, parent) in enumerate(self.parents[node]): 
                self.theta_hat[parent][node].append(tmp[idx])
            
        for node in not_update_list:
            for parent in self.parents[node]:
                old_value = self.theta_hat[parent][node][-1]
                self.theta_hat[parent][node].append(old_value)
