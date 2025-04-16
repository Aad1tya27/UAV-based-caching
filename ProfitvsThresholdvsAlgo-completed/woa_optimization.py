import numpy as np
import random
import copy
from sklearn.cluster import KMeans
# from profit_calc import main as main_simulation  # Assume your dry-run code is encapsulated in this function
# from profit_calc import generate_user_requests


# Content and Cache Parameters
M = 300           # Total content items
M_V = 120         # UAV cache capacity
M_U = 40          # User cache capacity
dim = 2 * M       # qV (M) + qU (M)
num_UAVs = 20
num_users = 200
area_size = 500
K = 10
UAV_altitude_range = (60,80)
system_bandwidth_UAV = 1e9  # 1 GHz
uav_density = num_UAVs / area_size**2
tau_U = num_users / area_size**2

# user_requests = generate_user_requests(K, num_users) # 1st param
    
user_pos = np.random.uniform(0, area_size, (num_users, 2)) # 2nd param
        
kmeans = KMeans(n_clusters=num_UAVs).fit(user_pos)
cluster_labels = kmeans.labels_
cluster_user_counts = np.bincount(cluster_labels, minlength=num_UAVs) 
uav_pos = np.hstack([kmeans.cluster_centers_, np.random.uniform(*UAV_altitude_range, (num_UAVs, 1))]) # 5th param

P_u_v = 3* np.ones((K, num_users, num_UAVs)) # 3rd param
B_u_v = np.zeros((num_users, num_UAVs))
for u in range(num_users):
        v = cluster_labels[u]
        B_u_v[u, v] = system_bandwidth_UAV / cluster_user_counts[v]


class Whale:
    def __init__(self, dim, seed):
        q_V_inital = np.random.uniform(0, 1, M)
        q_U_inital = np.random.uniform(0, 1, M)

        self.position = np.zeros(dim)
        
        self.position[:M] = q_V_inital
        self.position[M:] = q_U_inital
        
        self.fifo_V = [i for i in range(M)]
        random.shuffle(self.fifo_V)

        self.fifo_U = [i for i in range(M, 2*M)]
        random.shuffle(self.fifo_U)
        
        self._enforce_cache_constraints()

    def _enforce_cache_constraints(self):
        # Normalize qV to sum <= M_V
        sum_qV = np.sum(self.position[:M])

        # while sum_qV > M_V :

        
        while sum_qV > M_V:
            # self.position[:M] *= M_V / sum_qV
            # c = np.random.randint(0, M)
            c =  self.fifo_V.pop(0)
            self.fifo_V.append(c)
            self.position[c] = 0
            sum_qV = np.sum(self.position[:M])
            # print(self.position[c], sum_qV)

        # Normalize qU to sum <= M_U
        sum_qU = np.sum(self.position[M:])
        while sum_qU > M_U:
            # c = np.random.randint(M, 2*M)
            c = self.fifo_U.pop(0)
            self.fifo_U.append(c)
            self.position[c] = 0
            sum_qU = np.sum(self.position[M:])
            # print(self.position[c], sum_qU)
        # if sum_qU > M_U:
        #     self.position[M:] *= M_U / sum_qU


def woa_optimizer(fitness_func, user_requests, user_pos, uav_pos, P_u_v_k, B_u_v_k, cluster_labels, K, num_users, num_UAVs, uav_density, tau_U, max_iter=30, n_whales=30, omega=0.5):
    print("Initializing Whale Optimization...")
    
    
    # Initialize population
    population = [Whale(dim, i) for i in range(n_whales)]
    print("Initial Whales Computed ...")
    best_whale = max(population, key=lambda x: fitness_func(x.position, user_requests, user_pos, uav_pos, P_u_v_k, B_u_v_k, cluster_labels, K, num_users, num_UAVs, uav_density, tau_U, omega))
    best_whale_fitness = fitness_func(best_whale.position, user_requests, user_pos, uav_pos, P_u_v_k, B_u_v_k, cluster_labels, K, num_users, num_UAVs, uav_density, tau_U, omega)
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
            curr_fitness = fitness_func(whale.position, user_requests, user_pos, uav_pos, P_u_v_k, B_u_v_k, cluster_labels, K, num_users, num_UAVs, uav_density, tau_U, omega)
            if  curr_fitness > best_whale_fitness:
                best_whale = copy.deepcopy(whale)
                best_whale_fitness = curr_fitness
        if prev_iter_profit == best_whale_fitness:
            repeats+=1
        else:
            repeats = 0
            prev_iter_profit = best_whale_fitness
        
        print(f"Iteration {iter+1}: Best Profit = {best_whale_fitness:.2f}")

        iter+=1
    return best_whale.position


# def optimize_caching(q_V, q_U, fitness_func, num_users, num_UAVs, )

if __name__ == "__main__":
    # Run optimization

    
    # optimal_solution = woa_optimizer(fitness_func)
    
    print("\nOptimal Caching Probabilities:")
    # print(f"UAV Caching (qV): {optimal_solution[:M]}, sum of qV :{np.sum(optimal_solution[:M])}")
    # print(f"User Caching (qU): {optimal_solution[M:]}, sum of qU :{np.sum(optimal_solution[M:])}")
    # print(f"Maximized Profit: {fitness_func(optimal_solution):.2f}")