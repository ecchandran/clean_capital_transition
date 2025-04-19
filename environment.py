import numpy as np
import os

# Economic parameters: see overleaf for discussion
sim_eta_0 = 0.2      # Initial green capital share
sim_H_0 = 1.5 #1.5 #1.0        # Initial climate damage
sim_K_0 = 1.0        # Initial capital


rho = 0.04 #0.05 # 0.03 doesn't converge for some reason 0.04 worked tho
beta = 1 / rho
psi_1 = 0.125 # changed to 2 * 0.125 
psi_2 = psi_1
A_1 = 0.09 * np.exp(psi_1 * sim_H_0) #0.12                # not rigorously cal
A_2 = 0.10 * np.exp(psi_1 * sim_H_0)#0.125
phi = 2  
delta = 0.03 #0.03
mu_2 = 0.05 # 1 #0.025
eps = 0.5 * np.log(2) / 50 #np.log(2) / 50  
sigma_H = 0.05 #0.1 # 1 yr stddev of 0.1, 100 yr stddev of 1 - not unreasonable! # 1e-4         # not rigorously calibrated

# FD max iter*.s
max_iter = 500 #1_000

# Default tax parameters (can be overridden via command line
tau_0 = 0.71 # optimal 4.17 9pm trial
#tau_0 = (A_2 - A_1) / A_2 # make agents indfferent between capital types. ** Perfect since they internalize no impact on future climate
sim_Tau_0 = tau_0    # initial tax
alpha = 0.25 #0 # for volatile tau: #0.25 # assume reverts on order of every 4 years...
sigma_tau = (tau_0 / 2) * np.sqrt(2 * alpha) #0 # for part a - set to 0


tau_star = 1 - A_1/A_2  # Optimal tax rate
eta_high = 0.7  # Higher green capital share 
H_high = 6.0    # Higher climate damage
TAU_MAX = 1
ERGODIC_SAMPLE_TAU = False # if true, sample from O-U ergodic dist. if false, sample on unif[0, TAU_MAX]


#sigma_tau = tau_0 * np.sqrt(alpha / 2)  # want stddev of ergodic distribution to be tau_0 / 2

# def get_alpha(t0, st): NVM, DON"T DO THIS - want alpha constant if we vary sigma_tau!!!
#     if sigma_tau == 0:
#         return 1 # for safety elsewhere
#     return 2 * st ** 2 / t0 ** 2
# alpha = get_alpha(tau_0, sigma_tau)

# Grid parameters (for FD solution comparison)
eta_min = 0.01 #0.01
eta_max = 0.99#0.99
N_eta = 99#100
H_min = 0
H_max = 8 #6 #8
N_H = 101

# Deep learning parameters for original model
num_epochs = 30000
MINI_BATCH_PER_EPOCH = 1  # only for main training. recall that this just multiplies num epochs
batch_size = 500  # try substantially increasing
pretrain_epochs = 2000

# Deep learning parameters for model_3
model3_num_epochs = 20000 #30000
model3_pretrain_epochs = 10000
model3_hjbe_epochs = 10000 #10000
model3_batch_size = 1000
model3_mini_batch_per_epoch = 1 # Multiplier for Equilibrium and HJBE epochs.

# Display parameters
print_every = 50 # note: also frequency at which losses are saved
save_every = 5000

# Simulation parameters
sim_N_T = 10000      # Number of time steps (10000 * 0.01 = 100 years)
sim_dt = 0.01        # Small time step for stability


sim_num_sims = 10    # Number of simulations to run

# Convergence parameters
FD_LOG_ARG_CLIP = 0.1
DL_LOG_ARG_CLIP = 1e-4
conv_thresh = 1e-8

# Numerical stability parameters
DIVISION_EPSILON = 1e-9  # For safe division # 4/17: decreasing to 10^-9 to enable this to work with the range of our s values.  
CLIP_MIN = -10.0  # For clipping extreme values
CLIP_MAX = 10.0   # For clipping extreme values
S_MIN = DIVISION_EPSILON

# Boundary condition parameters
ETA_BOUNDARY_THRESHOLD = 0.01  # For eta=0 boundary
ETA_ONE_BOUNDARY_THRESHOLD = 0.99  # For eta=1 boundary
H_BOUNDARY_THRESHOLD = 0.01  # For H=0 boundary
BOUNDARY_SAMPLE_PROB = 0.01  # 1% probability of sampling at each boundary

# Sampling parameters
lambda_param_H = np.log(2) / 2  # sampling H=2 half as likely as H=0, etc.
epsilon_param_eta = 0.05  # Small value to prevent division by zero in eta sampling



# Random seed
seed = 1

# Percentiles for fan charts
sim_percentiles = [5, 15, 25, 35, 45, 50, 55, 65, 75, 85, 95]

# NN architecture parameters
hidden_layers = [64, 64, 64, 64, 64]# 4.18[128, 128, 128, 64]  # Increased network capacity
learning_rate = 1e-3
weight_decay = 1e-5  # L2 regularization to prevent overfitting

# Loss function weights
HJBE_WEIGHT = 1.0
HJBE_SHAPE_WEIGHT = 0.01 #0.01 # already scaling down hjbe s ** 0.4, roughly 10^-2 # higher this time bc in past, dominated hjbe loss
HJBE_ZERO_AV_LOSS_WEIGHT = 0

CONSISTENCY_WEIGHT = 5
MARKET_CLEARING_WEIGHT = 5 #10.0 #
BOUNDARY_WEIGHT = 1 #10.0 # both of these are fixed equations essentially. 

# Adaptive loss weighting parameters
#ADAPTIVE_WEIGHT_ALPHA = 1 # 4/16: experiment with effectively removing $S$ #0.5  # Parameter for inverse weighting in loss function. 
# clearing conditions have loss equal to non-normalized equation multiplied thorugh by s ** (1-adaptive_alpha)
#WEIGHT_NORMALIZATION = True  # Whether to normalize weights

# HJBE training flags - set these before running the model again
RETRAIN_HJBE = False  # Set to True to force retraining HJBE even if model exists
ASSUME_TRAINED_EQUILIBRIUM = True # set to True to assume equilibrium solution has already been trained and avoid retraining.
RECOMPUTE_VISUALIZATIONS = True  # Set to True to force recomputation of visualization data
RERUN_SIMULATIONS = True  # Set to True to force new simulations
# DISABLE_S_MULTIPLIER = False  # Set to True to disable multiplying HJBE by s^2

# Regularization parameter for HJBE weighting
HJBE_ADAPTIVE_ALPHA = 0.5 # 0.8 didn't work before  # Set to 1.0 for uniform weighting # 1 didn't work either

# Define utility functions commonly used
def Phi(iota, phi_val=phi):
    """Adjustment cost function"""
    return (1 / phi_val) * np.log(np.clip(1 + phi_val * iota, a_min=DL_LOG_ARG_CLIP, a_max=None))

def A_k1(H):
    return A_1 * np.exp(-psi_1 * H)

def A_k2(H):
    return A_2 * np.exp(-psi_2 * H)

def W_m1H0(A_eff = A_1):
    """Return steady solution of W(eta, H) for eta=1, H=0"""
    iota_1 = (A_eff * beta - 1) / (phi + beta) # PREV VERSION HAD bug - missed -1!!!
    return (1 / rho) * (np.log(A_eff - iota_1) + (Phi(iota_1) - delta) / rho)

def initial_guess(eta, H, tau=None):
    """Initial guess for W(eta, H, tau) based on the steady state solution"""
    H_effective = H * (1 - eta/2)
    A_k1_val = A_k1(H_effective)
    A_k2_val = A_k2(H_effective)
    
    if tau is not None:
        # For decentralized model with tax
        A_effective = eta * A_k1_val+ (1-eta) * A_k2_val * (1-tau)
    else:
        # For central planner model
        # PREVIOUSLY USED THIS. OK TO NOT BE CONSISTENT BC MODELS FUNDAMENTALLY DIFFERENT (planner accounts for impact on futre)
        A_effective = A_k1(H_effective)
        #A_effective = eta * A_k1_val + (1-eta) * A_k2_val
        
    return W_m1H0(A_effective)
