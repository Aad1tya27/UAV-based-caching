import numpy as np
import random
import copy
from sklearn.cluster import KMeans
from profit_calc import main as main_simulation  # Assume your dry-run code is encapsulated in this function
from profit_calc import generate_user_requests


# Content and Cache Parameters
M = 300           # Total content items
M_V = 120         # UAV cache capacity
M_U = 40          # User cache capacity
dim = 2 * M       # qV (M) + qU (M)
num_UAVs = 4
num_users = 50
area_size = 500
K = 1
UAV_altitude_range = (60,80)
system_bandwidth_UAV = 1e9  # 1 GHz


class Whale:
    def __init__(self, dim, seed):
        q_V_inital = np.random.uniform(0, 1, M)
        q_U_inital = np.random.uniform(0, 1, M)
        self.position = np.zeros(dim)
        
        # Initialize qV and qU randomly in [0, 1]
        self.position[:M] = q_V_inital
        self.position[M:] = q_U_inital
        
        # Apply cache capacity constraints
        self._enforce_cache_constraints()

    def _enforce_cache_constraints(self):
        # Normalize qV to sum <= M_V
        sum_qV = np.sum(self.position[:M])
        if sum_qV > M_V:
            self.position[:M] *= M_V / sum_qV
            # self.position[:M] = np.clip(self.position[:M], 0, 1)  # Ensure no value >1
            
        # Normalize qU to sum <= M_U
        sum_qU = np.sum(self.position[M:])
        if sum_qU > M_U:
            self.position[M:] *= M_U / sum_qU
            # self.position[M:] = np.clip(self.position[M:], 0, 1)

user_requests = generate_user_requests(K, num_users) # 1st param

    
user_pos = np.random.uniform(0, area_size, (num_users, 2)) # 2nd param
        
kmeans = KMeans(n_clusters=num_UAVs).fit(user_pos)
cluster_labels = kmeans.labels_
cluster_user_counts = np.bincount(cluster_labels, minlength=num_UAVs) 
uav_pos = np.hstack([kmeans.cluster_centers_, np.random.uniform(*UAV_altitude_range, (num_UAVs, 1))]) # 5th param

P_u_v = 0.1* np.ones((num_users, num_UAVs)) # 3rd param
B_u_v = np.zeros((num_users, num_UAVs))
for u in range(num_users):
        v = cluster_labels[u]
        B_u_v[u, v] = system_bandwidth_UAV / cluster_user_counts[v]

def fitness_func(position):
    """Calculate profit for given caching probabilities"""
    qV = position[:M]
    qU = position[M:]
    
    # Run simulation with these caching probabilities
    profit = main_simulation(qV, qU, user_requests, user_pos, uav_pos, P_u_v, B_u_v, cluster_labels, K)  # Modify your main() to accept qV/qU and return profit
    return profit

def woa_optimizer(fitness_func, max_iter=50, n_whales=30):
    print("Initializing Whale Optimization...")
    
    # Initialize population
    population = [Whale(dim, i) for i in range(n_whales)]
    best_whale = max(population, key=lambda x: fitness_func(x.position))
    
    iter = 0
    while iter <= max_iter: 
    # for iter in range(max_iter):
        a = 2 * (1 - iter/max_iter)  # Decreases from 2 to 0
        b = 1
        prev_iter_profit = 0
        repeats = 0
        for whale in population:
            # WOA position update logic (continuous version)
            y = np.random.rand()
            A = 2*a*y - a
            C = 2*y
            p = np.random.rand()
            
            if p < 0.5:
                if abs(A) < 1:
                    # Exploitation: Update towards best solution (bubble net feeding)
                    new_pos = best_whale.position - A*np.abs(C*best_whale.position - whale.position)
                else:
                    # Exploration: Update using random whale
                    rand_whale = population[np.random.randint(0, n_whales)]
                    while rand_whale == whale:
                        rand_whale = population[np.random.randint(0, n_whales)]
                    new_pos = rand_whale.position - A*np.abs(C*rand_whale.position - whale.position)
            else:
                # Spiral update
                l = np.random.uniform(-1, 1)
                new_pos = np.abs(best_whale.position - whale.position)*np.exp(l*b)*np.cos(2*np.pi*l) + best_whale.position
            
            # Update position with constraints
            whale.position = np.clip(new_pos, 0, 1)
            whale._enforce_cache_constraints()
            
            # Update best solution
            if fitness_func(whale.position) > fitness_func(best_whale.position):
                best_whale = copy.deepcopy(whale)
        if prev_iter_profit == (fitness_func(best_whale.position)):
            repeats+=1
        else:
            repeats = 0
            prev_iter_profit = fitness_func(best_whale.position)
        
        print(f"Iteration {iter+1}: Best Profit = {fitness_func(best_whale.position):.2f}")
        if repeats > 5:
            break
        # print(f"Best Position: \n {best_whale.position}")
        iter+=1
    return best_whale.position

if __name__ == "__main__":
    # Run optimization
    optimal_solution = woa_optimizer(fitness_func)
    
    print("\nOptimal Caching Probabilities:")
    print(f"UAV Caching (qV): {optimal_solution[:M]}, sum of qV :{np.sum(optimal_solution[:M])}")
    print(f"User Caching (qU): {optimal_solution[M:]}, sum of qU :{np.sum(optimal_solution[M:])}")
    print(f"Maximized Profit: {fitness_func(optimal_solution):.2f}")