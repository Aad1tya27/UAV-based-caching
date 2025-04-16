import numpy as np
import random
import copy
from scipy.special import gamma

M_U = 40
M_V = 120

class Vulture:
    def __init__(self, dim, total_contents, seed):
        np.random.seed(seed)
        # Initialize cache probabilities for UAVs and users
        q_V_initial = np.random.uniform(0, 1, total_contents)
        q_U_initial = np.random.uniform(0, 1, total_contents)
        self.position = np.zeros(dim)
        self.position[:total_contents] = q_V_initial
        self.position[total_contents:] = q_U_initial
        
        # FIFO queues for constraint enforcement
        self.fifo_V = [i for i in range(total_contents)]
        random.shuffle(self.fifo_V)

        self.fifo_U = [i for i in range(total_contents, 2*total_contents)]
        random.shuffle(self.fifo_U)
        
        self._enforce_cache_constraints(total_contents)
        
    def _enforce_cache_constraints(self, total_contents):
        # Enforce UAV cache capacity
        sum_qV = np.sum(self.position[:total_contents])
        while sum_qV > M_V:
            # c = np.random.randint(0, total_contents)

            c = self.fifo_V.pop(0)
            self.fifo_V.append(c)
            self.position[c] = 0
            sum_qV = np.sum(self.position[:total_contents])
            
        # Enforce user cache capacity
        sum_qU = np.sum(self.position[total_contents:])
        while sum_qU > M_U:
            c = self.fifo_U.pop(0)
            self.fifo_U.append(c)
            # c = np.random.randint(total_contents, 2*total_contents)

            self.position[c] = 0
            sum_qU = np.sum(self.position[total_contents:])

def avoa_optimizer(fitness_func, user_requests, user_pos, uav_pos, P_u_v_k, 
                   B_u_v_k, cluster_labels, K, num_users, num_UAVs, 
                   uav_density, tau_U, total_contents, max_iter=30, n_vultures=30):
    print("Initializing Vulture Optimization")
    # AVOA parameters
    p1 = 0.6    # Exploration parameter
    p2 = 0.4    # Exploitation stage 1 parameter
    p3 = 0.6    # Exploitation stage 2 parameter
    alpha = 0.8 # Mobility randomness parameter
    w = 2.5     # Exploration constant
    
    # Calculate dimension based on total_contents
    dim = 2 * total_contents
    
    # Initialize population
    population = [Vulture(dim, total_contents, i) for i in range(n_vultures)]
    fitness = [fitness_func(v.position, user_requests, user_pos, uav_pos, 
                           P_u_v_k, B_u_v_k, cluster_labels, K, num_users, 
                           num_UAVs, uav_density, tau_U, total_contents) for v in population]
    print("Initial Vultures Computed ...")
    iterations_values = []
    # Sort and select best vultures
    sorted_idx = np.argsort(fitness)[::-1]
    BestV1 = copy.deepcopy(population[sorted_idx[0]])
    BestV2 = copy.deepcopy(population[sorted_idx[1]])
    prev_iter_profit = 0
    repeats = 0
    for iter in range(max_iter):
        # Calculate starvation rate F
        z = np.random.uniform(-1, 1)
        h = np.random.uniform(-2, 2)
        t = h * (np.sin(np.pi/2 * iter/max_iter)**w + 
                 np.cos(np.pi/2 * iter/max_iter - 1))
        F = (2 * np.random.random() + 1) * z * (1 - iter/max_iter) + t
        
        for i, vulture in enumerate(population):
            # Roulette wheel selection for R(i)
            if np.random.random() < alpha:
                R = BestV1
            else:
                R = BestV2
                
            # Exploration phase (|F| >= 1)
            if abs(F) >= 1:
                D = abs(np.random.random() * R.position - vulture.position)
                if np.random.random() < p1:
                    new_pos = R.position - D * F
                else:
                    new_pos = R.position - F + np.random.random() * (
                        np.random.random() * (1 - 0) + 0)  # ub=1, lb=0
                        
            # Exploitation phase
            else:
                d = R.position - vulture.position
                if 0.5 <= abs(F) < 1:  # Stage 1
                    if np.random.random() < p2:
                        new_pos = d * (F + np.random.random()) - d
                    else:
                        theta = np.random.random() * 2 * np.pi
                        S1 = R.position * (np.random.random() * p2/(2*np.pi)) * np.cos(theta)
                        S2 = R.position * (np.random.random() * p2/(2*np.pi)) * np.sin(theta)
                        new_pos = R.position - (S1 + S2)
                        
                else:  # Stage 2 (|F| < 0.5)
                    if np.random.random() < p3:
                        A1 = BestV1.position - (BestV1.position*vulture.position)/(BestV1.position-vulture.position+1e-10)*F
                        A2 = BestV2.position - (BestV2.position*vulture.position)/(BestV2.position-vulture.position+1e-10)*F
                        new_pos = (A1 + A2)/2
                    else:
                        # Levy flight
                        beta = 1.5
                        sigma = (gamma(1+beta)*np.sin(np.pi*beta/2)/(gamma((1+beta)/2)*beta*2**((beta-1)/2)))**(1/beta)
                        u = np.random.normal(0, sigma)
                        v = np.random.normal(0, 1)
                        LF = 0.01 * u / (abs(v)**(1/beta))
                        new_pos = R.position - abs(d) * F * LF
                        
            # Update position with constraints
            vulture.position = np.clip(new_pos, 0, 1)
            vulture._enforce_cache_constraints(total_contents)
            
            # Update fitness
            new_fitness = fitness_func(vulture.position, user_requests, user_pos, 
                                      uav_pos, P_u_v_k, B_u_v_k, cluster_labels, 
                                      K, num_users, num_UAVs, uav_density, tau_U, total_contents)
            
            # Update best solutions
            if new_fitness > fitness[sorted_idx[0]]:
                BestV2 = copy.deepcopy(BestV1)
                BestV1 = copy.deepcopy(vulture)
                fitness[i] = new_fitness
            elif new_fitness > fitness[sorted_idx[1]]:
                BestV2 = copy.deepcopy(vulture)
                fitness[i] = new_fitness
    
            sorted_idx = np.argsort(fitness)[::-1]
        if prev_iter_profit == fitness[sorted_idx[0]]:
            repeats+=1
        else:
            repeats = 0
            prev_iter_profit = fitness[sorted_idx[0]]
      
        print(f"Iteration {iter+1}: Best Profit = {fitness[sorted_idx[0]]:.2f}")
        iterations_values.append(fitness[sorted_idx[0]])
    # Return best solution in same format as WOA
    optimal_qV = BestV1.position[:total_contents]
    optimal_qU = BestV1.position[total_contents:]
    return np.concatenate([optimal_qV, optimal_qU]), iterations_values