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
tau_0 = 0.70 # optimal 4.24 10pm  #0.71 # optimal 4.17 9pm trial  #tau_0 = (A_2 - A_1) / A_2 # make agents indfferent between capital types. ** Perfect since they internalize no impact on future climate
sim_Tau_0 = tau_0    # initial tax
alpha = 0.25 #0 # for volatile tau: #0.25 # assume reverts on order of every 4 years...
sigma_tau = (tau_0 / 2) * np.sqrt(2 * alpha) 
TAU_MAX = 1
ERGODIC_SAMPLE_TAU = False # 4.24 been much more successful keeping this false # if true, sample from O-U ergodic dist. if false, sample on unif[0, TAU_MAX]
tau_star = 1 - A_1/A_2  # tax rate at which capital productivity equal
eta_high = 0.7  # Higher green capital share 
H_high = 6.0    # Higher climate damage

# Grid parameters (for FD solution comparison)
eta_min = 0.01 #0.01
eta_max = 0.99#0.99
N_eta = 99#100
H_min = 0
H_max = 8 #6 #8
N_H = 101

ETA_VALIDATATION = np.linspace(eta_min, eta_max, 99)
H_VALIDATION = np.linspace(H_min, H_max, 9)
TAU_VALIDATION = np.linspace(0, TAU_MAX, 11)

# Active sampling parameters
ACTIVE_SAMPLING_ENABLED = True  # Enable active sampling
ACTIVE_SAMPLING_START_EPOCH = 5  # Start active sampling after this epoch
VALIDATION_POINTS_PER_BIN = 5  # Number of points to sample for validation in each bin

NORMALIZE_GRAD_WEIGHTS = False
NORMALIZE_GRAD_DENOM_MIN = 1e-2

EXPLORATION_DECAY_RATE = 1  # Rate at which exploration weight decays

# Bin boundaries for active sampling
#ETA_BIN_BOUNDARIES = [0.01, 0.02, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.98, 0.99]
#ETA_BIN_BOUNDARIES = [0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.98, 0.99]
# More granular bin boundaries for eta
ETA_BIN_BOUNDARIES = [0.01, 0.02, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.30, 
                     0.40, 0.50, 0.60, 0.70, 0.80, 0.85, 0.88, 0.90, 0.92, 0.95, 0.98, 0.99]
H_BIN_BOUNDARIES = [0, 1, 2, 3, 4, 5, 6, 7, 8]
TAU_BIN_BOUNDARIES = [0, 0.1, 0.2, 0.4, 0.6, 0.8, 0.9, 1]



# Deep learning parameters for model_3
model3_num_epochs = 100#
model3_pretrain_epochs = 500 # ok to train exactly! #300
model3_hjbe_epochs = 50 
model3_batch_size = 1000
model3_mini_batch_per_epoch = 100 # Multiplier for Equilibrium and HJBE epochs.

# Display parameters
print_every = 1 # LEAVE AT 1 if 100 mini batches per epoch. note: also frequency at which losses are saved. 
save_every = 20 #50


# Deep learning parameters for original model CP_DL
num_epochs = 30000
MINI_BATCH_PER_EPOCH = 1  # only for main training. recall that this just multiplies num epochs
batch_size = 500  # try substantially increasing
pretrain_epochs = 2000



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
ETA_BOUNDARY_THRESHOLD = 0.005#0.01  # For eta=0 boundary # we don't need these to ever occur
ETA_ONE_BOUNDARY_THRESHOLD = 0.995 #0.99  # For eta=1 boundary
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
hidden_layers = [128, 128, 128, 128, 64] # adding extra layer! 

# Learning rate and Decay
learning_rate = 1e-3
LR_DECAY = True
min_learning_rate = 1e-4 #1e-4 - if LR_DECAY is true, used
decay_by_epoch = 50

# L2 regularization to prevent overfitting
weight_decay = 1e-5  

# Loss function weights
HJBE_WEIGHT = 1.0
HJBE_SHAPE_WEIGHT = 0.01 #0.01 # already scaling down hjbe s ** 0.4, roughly 10^-2 # higher this time bc in past, dominated hjbe loss
HJBE_ZERO_AV_LOSS_WEIGHT = 0

CONSISTENCY_WEIGHT = 5
MARKET_CLEARING_WEIGHT = 5 #10.0 #
BOUNDARY_WEIGHT = 1 #10.0 # both of these are fixed equations essentially. 
CONSTRAINT_WEIGHT = 1 # new 4.22

# HJBE training flags - set these before running the model again
RETRAIN_HJBE = False  # Set to True to force retraining HJBE even if model exists
ASSUME_TRAINED_EQUILIBRIUM = True # set to True to assume equilibrium solution has already been trained and avoid retraining.
RECOMPUTE_VISUALIZATIONS = True  # Set to True to force recomputation of visualization data
RERUN_SIMULATIONS = True  # Set to True to force new simulations

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
        ################ New Guess ########################################
        #ADJUSTED 4/22 based on empirical results
        A_effective = eta * A_k1_val + (1-eta) * A_k2_val # MODIFIED 4/22
        scale = 15 - 5 * (H - sim_H_0) / (H_high - sim_H_0)
        tau_mod = tau * (1-2 * (eta - sim_eta_0) / (eta_high - sim_eta_0)) * scale
        return W_m1H0(A_effective) + tau_mod
    
        ################# Old guess ########################################
        # A_effective = eta * A_k1_val + (1-eta) * A_k2_val * (1-tau) #original 4/22
        # return W_m1H0(A_effective)
    else:
        # For central planner model
        # PREVIOUSLY USED THIS. OK TO NOT BE CONSISTENT BC MODELS FUNDAMENTALLY DIFFERENT (planner accounts for impact on futre)
        A_effective = A_k1(H_effective)
        #A_effective = eta * A_k1_val + (1-eta) * A_k2_val
        return W_m1H0(A_effective)
