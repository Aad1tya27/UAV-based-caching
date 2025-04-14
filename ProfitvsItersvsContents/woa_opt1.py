import numpy as np
import random
import copy


M_U = 40          # User cache capacity
M_V = 120

class Whale:
    def __init__(self, dim, total_contents, seed):
        q_V_inital = np.random.uniform(0, 1, total_contents)
        q_U_inital = np.random.uniform(0, 1, total_contents)

        self.position = np.zeros(dim)
        
        self.position[:total_contents] = q_V_inital
        self.position[total_contents:] = q_U_inital
        
        self.fifo_V = [i for i in range(total_contents)]
        random.shuffle(self.fifo_V)

        self.fifo_U = [i for i in range(total_contents, 2*total_contents)]
        random.shuffle(self.fifo_U)
        
        self._enforce_cache_constraints(total_contents)

    def _enforce_cache_constraints(self, total_contents):
        # Normalize qV to sum <= M_V
        sum_qV = np.sum(self.position[:total_contents])
    
        # while sum_qV > M_V :

        
        while sum_qV > M_V:
            # self.position[:total_contents] *= M_V / sum_qV
            # c = np.random.randint(0, total_contents)
            c =  self.fifo_V.pop(0)
            self.fifo_V.append(c)
            self.position[c] = 0
            sum_qV = np.sum(self.position[:total_contents])
            # print(self.position[c], sum_qV)

        # Normalize qU to sum <= M_U
        sum_qU = np.sum(self.position[total_contents:])
        while sum_qU > M_U:
            # c = np.random.randint(total_contents, 2*total_contents)
            c = self.fifo_U.pop(0)
            self.fifo_U.append(c)
            self.position[c] = 0
            sum_qU = np.sum(self.position[total_contents:])
            # print(self.position[c], sum_qU)
        # if sum_qU > M_U:
        #     self.position[total_contents:] *= M_U / sum_qU


def woa_optimizer(fitness_func, user_requests, user_pos, uav_pos, P_u_v_k, B_u_v_k, cluster_labels, K, num_users, num_UAVs, uav_density, tau_U, total_contents, max_iter=30, n_whales=30):
    print("Initializing Whale Optimization...")
    iterations_values = []
    
    # Calculate dimension based on total_contents
    dim = 2 * total_contents
    
    # Initialize population
    population = [Whale(dim, total_contents, i) for i in range(n_whales)]
    print("Initial Whales Computed ...")
    best_whale = max(population, key=lambda x: fitness_func(x.position, user_requests, user_pos, uav_pos, P_u_v_k, B_u_v_k, cluster_labels, K, num_users, num_UAVs, uav_density, tau_U, total_contents))
    best_whale_fitness = fitness_func(best_whale.position, user_requests, user_pos, uav_pos, P_u_v_k, B_u_v_k, cluster_labels, K, num_users, num_UAVs, uav_density, tau_U, total_contents)
    iter = 0
    while iter < max_iter: 
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
            whale._enforce_cache_constraints(total_contents)
            
            # Update best solution
            curr_fitness = fitness_func(whale.position, user_requests, user_pos, uav_pos, P_u_v_k, B_u_v_k, cluster_labels, K, num_users, num_UAVs, uav_density, tau_U, total_contents)
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

        iterations_values.append(best_whale_fitness)
    return best_whale.position, iterations_values


# def optimize_caching(q_V, q_U, fitness_func, num_users, num_UAVs, )

if __name__ == "__main__":
    # Run optimization

    
    # optimal_solution = woa_optimizer(fitness_func)
    
    print("\nOptimal Caching Probabilities:")
    # print(f"UAV Caching (qV): {optimal_solution[:total_contents]}, sum of qV :{np.sum(optimal_solution[:total_contents])}")
    # print(f"User Caching (qU): {optimal_solution[total_contents:]}, sum of qU :{np.sum(optimal_solution[total_contents:])}")
    # print(f"Maximized Profit: {fitness_func(optimal_solution):.2f}")