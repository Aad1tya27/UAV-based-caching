import numpy as np
from scipy.integrate import quad
from scipy.special import gamma, betainc, expn, gammaincc, beta, betaincc
from sklearn.cluster import KMeans


# CONSTANTS
total_contents = 300
num_UAVs = 4
num_users = 50
omega = 0.5
M_U = 40
M_V = 120
range_v = 100
range_u = 30
area_size = 500
uav_density = num_UAVs / area_size**2 
tau_U = num_users / area_size ** 2


total_area = area_size ** 2
UAV_altitude_range = (60, 80)  # meters
sigma2_dBm = -174  # dBm/Hz
# bandwidth = 1e6  # 1MHz
sigma2 = 10**((sigma2_dBm - 30)/10)
rho_UAV = -1

sigma_los = 2.0  # dB (standard deviation for LoS shadowing)
sigma_nlos = 5.0
mu_los = 2.0  # Path loss exponents
mu_nlos = 3.0
X_env = 11.9  
Y_env = 0.13  

beta_pl = 2.5  # Path loss exponent for D2D
P_U_dbm = 23
P_U = 10**((P_U_dbm - 30)/10)
rho_d2d_db = -2
rho_D2D = 10**(rho_d2d_db/10)
rho_H2D_db = 2
rho_H2D = 10**(rho_H2D_db / 10)
P_H_dbm = 40
P_H = 10**((P_H_dbm -30) / 10)


z_h = 200000.0
HAPS_pos = [area_size / 2, area_size / 2, z_h]
bandwidth_HAPS = 10e6  # 10 MHz
bandwidth_users = 10e6  # 10 MHz
system_bandwidth_UAV = 1e9  # 1 GHz

P_V_max= 3

m0 = 2
omega0 = 1

# Revenue and Cost params

eta_U2D = 5e-7
eta_H2D = 3e-7
eta_D2D = 1e-6

vartheta_link = 0.4
vartheta_comp = 0.6

mu_v = 1000  
nu_v = 1e-27  # Capacitance coefficient (from paper [2])
f_max_v = 2e9  # Maximum CPU frequency of UAV (Hz)
tau_comp = 1  # Time slot duration (seconds)
eta_comp = 1.5e-6  # Revenue rate for computation offloading ($/Hz)
P_v_comp = 1


a1 = 0.6       
b1 = 0.11



def generate_user_requests(K, num_users, content_prob=1):
    
    requests = np.zeros((K, num_users, 2))
    for k in range(K):
        for u in range(num_users):
            # Random choice with 70% probability for content delivery
            if np.random.rand() < content_prob:
                requests[k,u] = [1, np.floor(np.random.rand()*total_contents) + 1]  # Content request
            else:
                requests[k,u] = [0, np.random.uniform(1e6, 1e7)]  # required cycles
    return requests

# -------------------- Caching Probabilities ----------------------

def content_request_probability():
    denom = np.sum([j**-omega for j in range(1, total_contents+1)])
    p_C = np.array([i**-omega / denom for i in range(1, total_contents+1)])
    return p_C 

def content_cached_prob(M):
    q = np.zeros((total_contents,))
    while np.sum(q) < M - 1 :
        random_num = np.random.randint(0, total_contents)
        if q[random_num] == 0:
            q[random_num] = np.random.rand()

    return q

def U2D_caching_hit(pc, q, num, r_v = range_v):
    pc_V = np.zeros((num))
    for i in range(num):
        for j in range(total_contents):
            pc_V[i] += pc[j] * (1 - q[j,i]) * ( 1 - np.exp(-num * q[j,i] * np.pi * r_v**2 ))
    return pc_V

def D2D_caching_hit(pc , q, num, rho, r_d = range_u):
    pc_U = np.zeros((num))
    for i in range(num):
        for j in range(total_contents):
            pc_U[i] += pc[j] * (1 - q[j,i]) * ( 1 - np.exp(-rho * q[j,i] * np.pi * r_d**2 ))
    return pc_U

# -------------------- Path Loss ----------------------

def path_loss_los(distance, f_c=28e9, d0=1):
    """Exact LoS path loss with shadowing (Equation 10)"""
    pl_fs = 20*np.log10((4*np.pi*d0*f_c)/3e8)
    path_loss = pl_fs + 10*mu_los*np.log10(distance/d0)
    # shadowing = np.random.normal(0, sigma_los)
    return path_loss 

def path_loss_nlos(distance, f_c=28e9, d0=1):
    """Exact NLoS path loss with shadowing (Equation 11)"""
    pl_fs = 20*np.log10((4*np.pi*d0*f_c)/3e8)
    path_loss = pl_fs + 10*mu_nlos*np.log10(distance/d0)
    # shadowing = np.random.normal(0, sigma_nlos)
    return path_loss 

def los_probability(phi, X=X_env, Y=Y_env):
    """Exact LoS probability (Equation 14)"""
    return 1 / (1 + X * np.exp(-Y * (np.degrees(phi) - X)))

def satellite_path_loss(d_horizontal, f_c= 2e9, l_los = 0 , l_nlos = 20):
    c = 3e8  # speed of light
    d_3d = np.sqrt(z_h**2 + d_horizontal**2)  # 3D distance
    # Free-space path loss term:
    pl_fs = 20.0 * np.log10( (4.0 * np.pi * f_c * d_3d) / c )
    phi = np.arcsin(z_h/d_3d)
    p_los = los_probability(phi)
    PL = pl_fs + p_los * l_los + (1.0 - p_los) * l_nlos

    return PL

# -------------------- Success Probabilities ------------------

def U2D_succes_prob(q, u, uav_pos, user_pos, c, P,B, max_radius=500):
    
    distances = [np.linalg.norm(uav - np.append(user_pos[u], 0  )) for uav in uav_pos]
    v = np.argmin(distances)
    # print(uav_pos[v,2])
    q_i_V = q[c]

    sigma = sigma2 * B[u,v]
    if q_i_V == 0 or B[u,v] <= 1e-10: return 0

    def integrand(r):
        if r <= 0: return 0
        altitude = uav_pos[v, 2]
        distance = np.sqrt(r**2 + altitude**2)
        phi = np.arcsin(altitude / distance)
        
        los_prob = los_probability(phi)
        pl_los = path_loss_los(distance)
        pl_nlos = path_loss_nlos(distance)
        PL_avg = (pl_los*los_prob + pl_nlos * (1 - los_prob))

        z = (10**(PL_avg/10) * sigma * 10**(rho_UAV/10)) / P[u,v]
        
        exponent = -np.pi * (r ** 2) * (q_i_V) * uav_density - z
        
        return r * np.exp(exponent) * (z + 1)

    result, _ = quad(integrand, 0, max_radius)
    return min(2 * np.pi * q_i_V * uav_density * result, 1)


def D2D_succes_prob(q_U, u, user_pos, c, P_U, tau_U=tau_U):
    sigma = sigma2 *bandwidth_users
    
    distances = [np.linalg.norm(user_pos[u] - user_pos[m]) for m in range(num_users) if m != u]
    m = np.argmin(distances)
    q_u_U = q_U[c]

    if q_u_U == 0 : return 0
    # d = np.min(distances)
    
    def integrand(r):
        
        betaIncomp = q_u_U * betaincc(2 / beta_pl , 1 - 2/beta_pl, 1 / (rho_D2D + 1))
        betaComp = (1 - q_u_U) * beta(2 / beta_pl , 1 - 2/ beta_pl)
        # gamma_sample = np.random.gamma(shape=m0, scale=omega0/m0)
        g_h_u = np.sqrt(omega0)
        # g_h_u = 1

        PL_u = satellite_path_loss(np.linalg.norm(user_pos[u] - HAPS_pos[:2]))
        interference_noise_term = (rho_D2D * (r ** beta_pl) * sigma)/P_U + (rho_D2D * P_H * (g_h_u**2) * (10**( -PL_u / 10)) * (r ** beta_pl))/P_U
        return r * np.exp( - 2 * (np.pi / beta_pl) * (rho_D2D**(2/beta_pl)) *  tau_U * ( betaComp + betaIncomp) 
                          - interference_noise_term - np.pi * q_u_U * tau_U * r**2)
    
    result, _ = quad(integrand, 0,  range_u)
    return min(2* np.pi * tau_U * q_u_U *result, 1)

def H2D_succes_prob(u, user_pos):
    sigma = sigma2*bandwidth_HAPS
    # PL_u = satellite_path_loss(np.linalg.norm(user_pos[u] - HAPS_pos[:2]))
    # z0 = (10**(PL_u/10) * sigma * rho_H2D) / P_H

    # prob = (z0 + 1)*np.exp(-z0)
    # # print(z0)
    # return min(prob, 1)

    def integrand(r):
        # if r <= 0: return 0
        PL_u = satellite_path_loss(np.linalg.norm(user_pos[u] - HAPS_pos[:2]))
        # PL_u = satellite_path_loss(np.linalg.norm(r))
        z = (10**(PL_u/10) * sigma * rho_H2D) / P_H
        
        exponent = -np.pi * (r ** 2) - z
        
        return r * np.exp(exponent) * (z + 1)

    result, _ = quad(integrand, 0, np.inf)
    return min(2 * np.pi * result, 1)

# -------------------- SINR Calculations ------------------

def U2D_sinr(u ,v ,P, uav_pos, user_pos, B, omega = 1 ):
    sigma = sigma2 * B
    # gamma_sample = np.random.gamma(shape=m0, scale=omega/m0)
    # g_h_u = np.sqrt(gamma_sample)
    # Square it => |g|^2
    g_sq = omega
    r = np.linalg.norm(uav_pos[v, 0:2] - user_pos[u])
    altitude = uav_pos[v, 2]
    distance = np.sqrt(r**2 + altitude**2)
    phi = np.arcsin(altitude / distance)
        
    los_prob = los_probability(phi)
    pl_los = path_loss_los(distance)
    pl_nlos = path_loss_nlos(distance)
    PL_avg = (pl_los*los_prob + pl_nlos * (1 - los_prob))

    # Compute the SINR
    sinr = (P * g_sq) / (10 **(PL_avg/10) * sigma)
    return sinr

def D2D_sinr(u, m, user_pos, P_U, bandwidth_users, omega = 1):

    d_desired = np.linalg.norm(user_pos[u] - user_pos[m])
    d_desired = max(d_desired, 1e-6)  # Ensures no division by zero
    # gamma_sample = np.random.gamma(shape=m0, scale=omega/m0)
    # g_h_u = np.sqrt(gamma_sample)
    # g_sq = g_h_u**2
    g_sq = omega
    # g_sq = 1
    signal = P_U * g_sq * (d_desired ** (-beta_pl))
    
    # Noise power over the allocated bandwidth
    noise = sigma2 * bandwidth_users

    I_d2d = 0
    for i in range(num_users):
        if i == u or i == m:
            continue
        d_int = np.linalg.norm(user_pos[i] - user_pos[m])
        if d_int > range_u:
            continue
        g_int = omega
        I_d2d += P_U * g_int * (d_int ** (-beta_pl))
    
    # HAPS interference: apply Rayleigh fading and convert HAPS link loss from dB to linear scale.
    # g_HAPS = np.sqrt(np.random.gamma(shape=m0, scale=omega/m0))**2
    g_HAPS = omega
    L_u_m = satellite_path_loss(np.linalg.norm(user_pos[u] - HAPS_pos[:2]))
    I_HAPS = P_H * g_HAPS * 10**(-L_u_m/10)
    
    # Total interference plus noise
    denom = noise + I_d2d + I_HAPS
    
    sinr = signal / denom
    return sinr

def H2D_sinr(u, user_pos, omega = 1):
    distance_horizontal = np.linalg.norm(user_pos[u] - HAPS_pos[:2])
    PL = satellite_path_loss(distance_horizontal)
    # gamma_sample = np.random.gamma(shape=m0, scale=omega/m0)
    # g_h_u = np.sqrt(gamma_sample)
    # g_sq = g_h_u**2
    g_sq = omega


    received_power = P_H * g_sq * 10**(-PL / 10)
    noise = sigma2 * bandwidth_HAPS
    return received_power / noise

# ------------------ THROUGHPUT CALCULATION ---------------------
def calculate_U2D_throughput(pc_i, q_U,q_V, range_v, pc_V_success, B_u_v, cluster_labels, P, user_pos, uav_pos ):
    ans = 0
    for i in range(total_contents):
        temp  = 0
        for u in range(num_users):
            v = cluster_labels[u]
            temp += pc_V_success[i, u]*  B_u_v[u,v] * np.log2(1 + U2D_sinr(u ,v,P[u,v], uav_pos, user_pos, B_u_v[u,v]))
        ans += temp * pc_i[i] * ( 1 - q_U[i] ) * (1 - np.exp( - uav_density * q_V[i] * np.pi * range_v**2 ))
    return ans

def calculate_D2D_throughput(pc_i, q_U, pr_V, pr_U, pc_U_success, B_D2D, N_D, P_U, user_pos):
    if N_D == 0:
        return 0.0
    throughput = 0.0
    bandwidth_per_user = B_D2D / N_D
    for i in range(total_contents):
        temp = 0
        for u in range(num_users):
            if pc_U_success[i, u] == 0: continue
            distances = [np.linalg.norm(user_pos[u] - user_pos[m]) for m in range(num_users) if m != u]
            m = np.argmin(distances)
            sinr = D2D_sinr(u, m, user_pos, P_U, bandwidth_per_user)
            sinr = min(sinr, 1e3)  # Maximum realistic SINR of 30 dB
            temp += pc_U_success[i, u] * bandwidth_per_user * np.log2(1 + sinr)
            # print(pc_U_success[i,u], np.log2(1 + sinr_linear))
        throughput += temp * pc_i[i] * (1 - q_U[i]) * (1 - pr_V[i]) * pr_U[i]
        # if temp: print(temp * pc_i[i] * (1 - q_U[i]) * (1 - pr_V[i]) * pr_U[i],pc_i[i], bandwidth_per_user, temp, (1 - q_U[i]), (1 - pr_V[i]), pr_U[i] )
    return throughput

def calculate_H2D_throughput(pc_i, pr_V, pr_U, pc_H_success, B_H2D, N_H, user_pos):
    if N_H == 0:
        return 0.0
    throughput = 0.0
    bandwidth_per_user = B_H2D / N_H
    # print(bandwidth_per_user)
    for i in range(total_contents):
        temp = 0
        for u in range(num_users):
            if pc_H_success[i,u] == 0: continue
            # print(pc_H_success[i,u])
            snr = H2D_sinr(u, user_pos)
            temp += pc_H_success[i, u] * bandwidth_per_user * np.log2(1 + snr)
            # if temp: print(temp * pc_i[i] * (1 - pr_V[i]) * (1 - pr_U[i]), temp)
        throughput += temp * pc_i[i] * (1 - pr_V[i]) * (1 - pr_U[i])
    return throughput

# -------------------- Energy Calculation ----------------------------

def calculate_U2D_energy(pc_i, q_U,q_V, range_v, pc_V_success, cluster_labels, P):
    energy = 0
    for i in range(total_contents):
        temp  = 0
        for u in range(num_users):
            if pc_V_success[i,u] == 0: continue
            v = cluster_labels[u]
            temp += pc_V_success[i,u] * P[u,v]
        energy += temp * pc_i[i] * ( 1- q_U[i]) * (1 - np.exp( - uav_density * q_V[i] * np.pi * range_v**2 ))
    return energy

# def calculate_D2D_energy(pc_i, q_U, pr_V, pr_U, pc_U_success, P_U):
#     energy = 0.0
#     for i in range(total_contents):
#         temp = 0
#         for u in range(num_users):
#             if pc_U_success[i, u] == 0: continue
            
#             temp += pc_U_success[i, u]  * P_U
#         energy += temp * pc_i[i] * (1 - q_U[i]) * (1 - pr_V[i]) * pr_U[i]
#         # if pr_U[i]: print(temp, pc_i[i], (1 - q_U[i]) ,(1 - pr_V[i]),  pr_U[i] )
#     return energy

def calculate_D2D_energy(pc_i, current_requests, q_U, pr_V, pr_U, pc_U_success, P_U):
    energy = 0.0
    for u in range(num_users):
        req_type, req_content = current_requests[u]
        if req_type != 1:  # Skip computation requests
            continue
        c = int(req_content - 1)  # Content index (0-based)
        if pc_U_success[c, u] == 0:
            continue
        # Terms: (1 - q_U[c]) * (1 - pr_V[c]) * pr_U[c]
        energy_term = pc_i[c] * (1 - q_U[c]) * (1 - pr_V[c]) * pr_U[c]
        energy += pc_U_success[c, u] * P_U * energy_term
    return energy

def calculate_H2D_energy(pc_i, pr_V, pr_U, pc_H_success): 
    energy = 0.0
    # print(bandwidth_per_user)
    for i in range(total_contents):
        temp = 0
        for u in range(num_users):
            if pc_H_success[i,u] == 0: continue
            # print(pc_H_success[i,u])
            temp += pc_H_success[i, u] * P_H
            # if temp: print(temp * pc_i[i] * (1 - pr_V[i]) * (1 - pr_U[i]), temp)
        energy += temp * pc_i[i] * (1 - pr_V[i]) * (1 - pr_U[i])
    return energy

    
# _____________________ COMPUTATION FUNCTIONS _______________________________

def compute_p_los(u, v, user_pos, uav_pos):
    z_v = uav_pos[v, 2]
    dist = np.sqrt((np.linalg.norm(user_pos[u] - uav_pos[v, 0:2]))**2 + z_v**2)
    phi_u_v = np.arcsin(z_v/dist)
    return a1 * ((180/np.pi)*phi_u_v - 15)**b1

def compute_path_loss(p_los, d_u_v_k, f_c = 2e9, d0 = 1, eta_los = 3, eta_nlos = 5):
    c=3e8
    term = ((4*np.pi*f_c*d0)/c) * (d_u_v_k/d0)**2
    return (p_los*(eta_los - eta_nlos) + eta_nlos) * term


def computation_sinr(P_U, u ,v ,user_pos, uav_pos, beta0 = 1e-4, delta_dbm = -90):
    delta = 10**((delta_dbm-30)/10)
    z_v = uav_pos[v, 2]
    dist = np.sqrt((np.linalg.norm(user_pos[u] - uav_pos[v, 0:2]))**2 + z_v**2)
    return (P_U * beta0)/(dist**2 * delta)

def main(q_V, q_U ,user_requests, user_pos, uav_pos, P_u_v, B_u_v, cluster_labels, K):
    pc_content_prob = content_request_probability()

    # q_V = content_cached_prob(M_V)
    # q_U = content_cached_prob(M_U)
    total_profit = 0
    
    for k in range(K):
        RV_comp = 0.0
        E_comp = 0.0
        current_requests = user_requests[k]
        
        pc_V_success = np.zeros((total_contents, num_users))
        pc_U_success = np.zeros((total_contents, num_users))
        pc_H_success = np.zeros((total_contents, num_users))
        
        
        
        N_U = 0
        N_H = 0

        pr_V = (1 - np.exp(-uav_density * q_V * np.pi * range_v**2))
        pr_U = (1 - np.exp(-tau_U * q_U * np.pi * range_u**2))
        
        a_comp = np.empty(num_UAVs, dtype=object) 
        for i in range(num_UAVs):
            a_comp[i] = []
        
        # uav_occupied = np.zeros((num_UAVs, ))
        for u in range(num_users):
            user_req = current_requests[u]
            if user_req[0] == 1:
                res_U2D = U2D_succes_prob(q_V, u, uav_pos, user_pos, int(user_req[1]-1), P_u_v, B_u_v)
                if res_U2D > 0:
                    # print("U2D: ", res_U2D)
                    pc_V_success[int(user_req[1]-1), u] = res_U2D
                    continue
                res_D2D = D2D_succes_prob(q_U, u, user_pos, int(user_req[1]-1), P_U)
                if res_D2D > 0:
                    # print("D2D:",res_D2D)
                    N_U += 1
                    pc_U_success[int(user_req[1]-1), u] = res_D2D
                    continue
                N_H += 1   
                res_H2D = H2D_succes_prob(u , user_pos)
                pc_H_success[int(user_req[1]-1), u] = res_H2D
            else:
                v = cluster_labels[u]
                a_comp[v].append(u)
        

        # for v in range(num_UAVs):
        #     if len(a_comp[v]) == 0 : continue
        #     u = np.random.choice(a_comp[v])
        #     user_req = current_requests[u]
        #     task_size = user_req[1]
        #     required_cycles = mu_v * task_size
        #     if required_cycles <= tau_comp * f_max_v:
        #         print("if condition")
        #             # sinr = U2D_sinr(u, v, P_u_v[u,v], uav_pos, user_pos, B_u_v[u,v])
        #             # rate = B_u_v[u,v] * np.log2(1 + sinr)
        #         sinr = computation_sinr(P_U, u,v, user_pos, uav_pos)
        #         rate = B_u_v[u,v]  * np.log2(1 + sinr)
        #         RV_comp += eta_comp * rate
        #         E_comp += (nu_v * (required_cycles**3)) / (tau_comp**2)
        #     else:    
        #         print("else condition")
        #         snr = H2D_sinr(u, user_pos)
        #         rate = bandwidth_HAPS * np.log2(1 + snr)
        #         RV_comp += eta_comp * rate

        RV_V = eta_U2D * calculate_U2D_throughput(pc_content_prob, q_U,q_V, range_v, pc_V_success, B_u_v, cluster_labels, P_u_v, user_pos, uav_pos )
        RV_U =  eta_D2D* calculate_D2D_throughput(pc_content_prob, q_U, pr_V, pr_U, pc_U_success, bandwidth_users, N_U, P_U, user_pos)
        RV_H =  eta_H2D * calculate_H2D_throughput(pc_content_prob, pr_V, pr_U, pc_H_success, bandwidth_HAPS, N_H, user_pos)
        # print(f"Requests handled by U2D link: {num_users - N_H - N_U}, D2D link: {N_U}, H2D link: {N_H}")
        # print("Revenues as follows: ")
        # print(RV_V)
        # print(RV_U)
        # print(RV_H)
        total_rev = RV_V + RV_U + RV_H

        Power_U2D = vartheta_link * calculate_U2D_energy(pc_content_prob, q_U,q_V, range_v, pc_V_success, cluster_labels, P_u_v)
        
        Power_D2D = vartheta_link * calculate_D2D_energy(pc_content_prob,current_requests, q_U, pr_V, pr_U, pc_U_success, P_U)
        Power_H2D = vartheta_link * calculate_H2D_energy(pc_content_prob, pr_V,pr_U, pc_H_success)
        # print("Power as follows: ")

        # print(Power_U2D)
        # print(Power_D2D)
        # print(Power_H2D)
        total_energy = (Power_U2D + Power_D2D + Power_H2D)

        # print(total_rev ,total_energy)

        total_profit += total_rev - total_energy
        # print(f"D2D users: {N_U} and H2D users: {N_H}")
        # print(f"Total profit for time slot {k+1} is: {total_profit}")
        # print(f"RV comp is:  {RV_comp} and Energy Comp is: {E_comp}")
    # print(f"Total profit over k is: {total_profit}")

    return total_profit
if __name__ == "__main__":
    q_V = content_cached_prob(M_V)
    q_U = content_cached_prob(M_U)
    K = 1
    user_requests = generate_user_requests(K, num_users) # 1st param

    
    user_pos = np.random.uniform(0, area_size, (num_users, 2)) # 2nd param
        
    kmeans = KMeans(n_clusters=num_UAVs).fit(user_pos)
    cluster_labels = kmeans.labels_
    cluster_user_counts = np.bincount(cluster_labels, minlength=num_UAVs) 
    uav_pos = np.hstack([kmeans.cluster_centers_, np.random.uniform(*UAV_altitude_range, (num_UAVs, 1))]) # 5th param

    P_u_v = 0.1* np.ones((num_users, num_UAVs)) # 3rd param
    B_u_v = np.zeros((num_users, num_UAVs)) # 4th param

    for u in range(num_users):
        v = cluster_labels[u]
        B_u_v[u, v] = system_bandwidth_UAV / cluster_user_counts[v]


    totalprofit = main(q_V, q_U, user_requests, user_pos, uav_pos, P_u_v, B_u_v, cluster_labels, K )