import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import time
import random
import pandas as pd
import matplotlib.pyplot as plt
import environment as env
from scipy.stats import norm
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description='Train and simulate decentralized model')
parser.add_argument('--regime', type=int, choices=[1, 2], required=True,
                    help='Regime (1 or 2)')
parser.add_argument('--tag', type=str, required=True,
                    help='Custom tag for saved files')
parser.add_argument('--num_epochs', type=int, default=env.model3_num_epochs,
                    help=f'Number of training epochs (default: {env.model3_num_epochs})')
parser.add_argument('--tau_0', type=float, default=env.tau_0,
                    help=f'Base carbon tax rate (default: {env.tau_0})')
parser.add_argument('--sigma_tau', type=float, default=env.sigma_tau,
                    help=f'Carbon tax volatility (default: {env.sigma_tau})')

args = parser.parse_args()

# Set global parameters
REGIME = args.regime
MAIN_NUM_EPOCHS = args.num_epochs
TAU_0 = args.tau_0
SIGMA_TAU = args.sigma_tau
TAG = args.tag

# Create output directory
OUTPUT_DIR = f"{TAG}_M3DL"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Update environment variables
env.tau_0 = TAU_0
env.sigma_tau = SIGMA_TAU
env.sim_Tau_0 = TAU_0

# For reproducibility
def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

# Use device variable consistently throughout
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Safe division function to prevent division by zero
def safe_divide(numerator, denominator, epsilon=env.DIVISION_EPSILON):
    """Safely divide tensor values with a minimum denominator threshold"""
    return numerator / torch.clamp(denominator, min=epsilon)

# # Weighted loss function to account for scaling by s
# def weighted_equation_loss(eq_residuals, s_values, alpha=env.ADAPTIVE_WEIGHT_ALPHA):
#     """
#     Compute weighted loss with higher weights for small s values
    
#     Parameters:
#     - eq_residuals: equation residuals
#     - s_values: denominator values that can be small
#     - alpha: control parameter for weight strength
    
#     Returns:
#     - Weighted MSE loss
#     """
#     # Compute weights that increase for small s values
#     weights = 1.0 / (s_values**alpha + env.DIVISION_EPSILON)
    
#     # Normalize weights if enabled
#     if env.WEIGHT_NORMALIZATION:
#         weights = weights * (len(weights) / weights.sum())
    
#     # Compute weighted squared error
#     return torch.mean(weights * eq_residuals**2)

# Custom neural network for equilibrium variables
class EquilibriumModel(nn.Module):
    def __init__(self, input_dim=3, hidden_dims=env.hidden_layers):
        super(EquilibriumModel, self).__init__()
        
        # Common feature extractor
        self.feature_layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.SiLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.SiLU(),
        )
        
        # Individual heads for the 8 equilibrium objects
        self.q1_head = nn.Sequential(
            nn.Linear(hidden_dims[1], hidden_dims[2]),
            nn.SiLU(),
            nn.Linear(hidden_dims[2], 1),
            nn.Softplus()  # Ensure positive q1
        )
        
        self.mu_q1_head = nn.Sequential(
            nn.Linear(hidden_dims[1], hidden_dims[2]),
            nn.SiLU(), 
            nn.Linear(hidden_dims[2], 1)
        )
        
        self.sigma_q1_a_head = nn.Sequential(
            nn.Linear(hidden_dims[1], hidden_dims[2]),
            nn.SiLU(),
            nn.Linear(hidden_dims[2], 1)
        )
        
        self.sigma_q1_b_head = nn.Sequential(
            nn.Linear(hidden_dims[1], hidden_dims[2]),
            nn.SiLU(),
            nn.Linear(hidden_dims[2], 1)
        )
        
        self.q2_head = nn.Sequential(
            nn.Linear(hidden_dims[1], hidden_dims[2]),
            nn.SiLU(),
            nn.Linear(hidden_dims[2], 1),
            nn.Softplus()  # Ensure positive q2
        )
        
        self.mu_q2_head = nn.Sequential(
            nn.Linear(hidden_dims[1], hidden_dims[2]),
            nn.SiLU(),
            nn.Linear(hidden_dims[2], 1)
        )
        
        self.sigma_q2_a_head = nn.Sequential(
            nn.Linear(hidden_dims[1], hidden_dims[2]),
            nn.SiLU(),
            nn.Linear(hidden_dims[2], 1)
        )
        
        self.sigma_q2_b_head = nn.Sequential(
            nn.Linear(hidden_dims[1], hidden_dims[2]),
            nn.SiLU(),
            nn.Linear(hidden_dims[2], 1)
        )
        
        # Initialize weights for better convergence
        self._initialize_weights()
    
    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        features = self.feature_layers(x)
        
        q1 = self.q1_head(features)
        mu_q1 = self.mu_q1_head(features)
        sigma_q1_a = self.sigma_q1_a_head(features)
        sigma_q1_b = self.sigma_q1_b_head(features)
        
        q2 = self.q2_head(features)
        mu_q2 = self.mu_q2_head(features)
        sigma_q2_a = self.sigma_q2_a_head(features)
        sigma_q2_b = self.sigma_q2_b_head(features)
        
        return q1, mu_q1, sigma_q1_a, sigma_q1_b, q2, mu_q2, sigma_q2_a, sigma_q2_b
    
    def boundary_forward(self, eta, H, tau, regime):
        """Apply boundary conditions directly"""
        batch_size = eta.shape[0]
        
        # Create empty tensors for outputs
        q1 = torch.zeros((batch_size, 1), device=device)
        mu_q1 = torch.zeros((batch_size, 1), device=device)
        sigma_q1_a = torch.zeros((batch_size, 1), device=device)
        sigma_q1_b = torch.zeros((batch_size, 1), device=device)
        q2 = torch.zeros((batch_size, 1), device=device)
        mu_q2 = torch.zeros((batch_size, 1), device=device)
        sigma_q2_a = torch.zeros((batch_size, 1), device=device)
        sigma_q2_b = torch.zeros((batch_size, 1), device=device)
        
        # Boundary conditions for eta = 1
        eta_1_mask = (eta > env.ETA_ONE_BOUNDARY_THRESHOLD)
        if eta_1_mask.any():
            A_k1_val = env.A_1 * torch.exp(-env.psi_1 * H[eta_1_mask])
            q1[eta_1_mask] = (1 + env.phi * A_k1_val) / (1 + env.phi * env.rho)
            mu_q1[eta_1_mask] = (-env.phi * env.psi_1 * A_k1_val / (1 + env.phi * env.rho)) * (-env.eps * H[eta_1_mask] + env.mu_2 * (1 - eta[eta_1_mask])) + \
                                (0.5 * env.phi * env.psi_1**2 * A_k1_val / (1 + env.phi * env.rho)) * (H[eta_1_mask] * env.sigma_H)**2
            sigma_q1_a[eta_1_mask] = (-env.phi * env.psi_1 * A_k1_val / (1 + env.phi * env.rho)) * H[eta_1_mask] * env.sigma_H
            sigma_q1_b[eta_1_mask] = 0
        
        # Boundary conditions for eta = 0
        eta_0_mask = (eta < env.ETA_BOUNDARY_THRESHOLD)
        if eta_0_mask.any():
            A_k2_val = env.A_2 * torch.exp(-env.psi_2 * H[eta_0_mask])
            
            if regime == 2:  # Apply tau for regime 2
                A_k2_effective = A_k2_val * (1 - tau[eta_0_mask])
                q2[eta_0_mask] = (1 + env.phi * A_k2_effective) / (1 + env.phi * env.rho)
                mu_q2[eta_0_mask] = (-env.phi * env.psi_2 * A_k2_effective / (1 + env.phi * env.rho)) * (-env.eps * H[eta_0_mask] + env.mu_2) + \
                                    (0.5 * env.phi * env.psi_2**2 * A_k2_effective / (1 + env.phi * env.rho)) * (H[eta_0_mask] * env.sigma_H)**2
                sigma_q2_a[eta_0_mask] = (-env.phi * env.psi_2 * A_k2_effective / (1 + env.phi * env.rho)) * H[eta_0_mask] * env.sigma_H
                sigma_q2_b[eta_0_mask] = (-env.phi * A_k2_val / (1 + env.phi * env.rho))
            else:  # Regime 1
                q2[eta_0_mask] = (1 + env.phi * A_k2_val) / (1 + env.phi * env.rho)
                mu_q2[eta_0_mask] = (-env.phi * env.psi_2 * A_k2_val / (1 + env.phi * env.rho)) * (-env.eps * H[eta_0_mask] + env.mu_2) + \
                                    (0.5 * env.phi * env.psi_2**2 * A_k2_val / (1 + env.phi * env.rho)) * (H[eta_0_mask] * env.sigma_H)**2
                sigma_q2_a[eta_0_mask] = (-env.phi * env.psi_2 * A_k2_val / (1 + env.phi * env.rho)) * H[eta_0_mask] * env.sigma_H
                sigma_q2_b[eta_0_mask] = 0
        
        # Boundary conditions for H = 0
        H_0_mask = (H < env.H_BOUNDARY_THRESHOLD)
        if H_0_mask.any():
            sigma_q1_a[H_0_mask] = 0
            sigma_q2_a[H_0_mask] = 0
        
        return q1, mu_q1, sigma_q1_a, sigma_q1_b, q2, mu_q2, sigma_q2_a, sigma_q2_b

# Value function network
class ValueFunctionModel(nn.Module):
    def __init__(self, input_dim=3, hidden_dims=env.hidden_layers):
        super(ValueFunctionModel, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.SiLU())
            prev_dim = hidden_dim
            
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.5)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        return self.network(x)

# Modified sampling functions with boundary point inclusion
def sample_H_exponential(n_samples, H_min, H_max):
    """
    Sample from truncated exponential distribution with 1% probability at H=0 boundary
    """
    # Determine how many samples to place at the boundary
    n_boundary = int(n_samples * env.BOUNDARY_SAMPLE_PROB)
    n_regular = n_samples - n_boundary
    
    # Generate boundary samples
    boundary_samples = np.zeros(n_boundary).reshape(-1, 1)
    
    # Generate regular samples
    # Calculate normalization constants
    a = np.exp(-env.lambda_param_H * H_min)
    b = np.exp(-env.lambda_param_H * H_max)
    
    # Generate uniform samples
    u = np.random.uniform(0, 1, n_regular)
    
    # Transform to truncated exponential
    regular_samples = -np.log(a - u * (a - b)) / env.lambda_param_H
    
    # Combine samples
    all_samples = np.vstack([boundary_samples, regular_samples.reshape(-1, 1)])
    
    # Shuffle to mix boundary and regular samples
    np.random.shuffle(all_samples)
    
    return all_samples

def sample_eta_proportional(n_samples, eta_min, eta_max):
    """
    Sample from distribution proportional to 1/((eta+epsilon)*(1-eta+epsilon))
    with 1% probability at each boundary (eta=0 and eta=1)
    """
    # Determine how many samples to place at each boundary
    n_boundary_0 = int(n_samples * env.BOUNDARY_SAMPLE_PROB)
    n_boundary_1 = int(n_samples * env.BOUNDARY_SAMPLE_PROB)
    n_regular = n_samples - n_boundary_0 - n_boundary_1
    
    # Generate boundary samples
    boundary_samples_0 = np.zeros(n_boundary_0).reshape(-1, 1)
    boundary_samples_1 = np.ones(n_boundary_1).reshape(-1, 1)
    
    # Generate regular samples
    # Create a grid for numerical approximation
    grid_size = 10000
    grid_step = (eta_max - eta_min) / grid_size
    
    eta_grid = np.linspace(eta_min + 0.5 * grid_step, eta_max - 0.5 * grid_step, grid_size) + np.random.uniform(-0.5, 0.5) * grid_step
    
    # Calculate PDF values (unnormalized)
    pdf_values = 1 / ((eta_grid) * (1 - eta_grid) + env.epsilon_param_eta)
    
    # Normalize to create a proper PDF
    pdf_values = pdf_values / np.sum(pdf_values)
    
    # Sample using the PDF
    samples_idx = np.random.choice(grid_size, size=n_regular, p=pdf_values)
    regular_samples = eta_grid[samples_idx]
    
    # Combine samples
    all_samples = np.vstack([boundary_samples_0, boundary_samples_1, regular_samples.reshape(-1, 1)])
    
    # Shuffle to mix boundary and regular samples
    np.random.shuffle(all_samples)
    
    return all_samples

def sample_tau(n_samples, tau_0, sigma_tau, alpha):
    """Sample tau according to parameters specified in env"""
    if env.ERGODIC_SAMPLE_TAU:
        if sigma_tau == 0:
            return tau_0 * np.ones(n_samples).reshape(-1, 1)
        else:
            std_dev = np.sqrt(sigma_tau**2 / (2*alpha))
            samples = np.random.normal(tau_0, std_dev, n_samples)
            return samples.reshape(-1, 1)
    else:
        return np.random.uniform(0, env.TAU_MAX, n_samples).reshape(-1, 1)
    
# Function to compute economic variables
def compute_economic_variables(eta, H, tau, q1, mu_q1, sigma_q1_a, sigma_q1_b, q2, mu_q2, sigma_q2_a, sigma_q2_b, regime):
    """Compute all economic variables based on state variables and equilibrium objects"""
    # Basic economic functions
    A_k1_val = env.A_1 * torch.exp(-env.psi_1 * H)
    A_k2_val = env.A_2 * torch.exp(-env.psi_2 * H)
    
    # Investment rates
    iota_1 = (q1 - 1) / env.phi
    iota_2 = (q2 - 1) / env.phi
    
    # Calculate Phi terms
    Phi_1 = (1/env.phi) * torch.log(torch.clamp(1 + env.phi * iota_1, min=env.DL_LOG_ARG_CLIP))
    Phi_2 = (1/env.phi) * torch.log(torch.clamp(1 + env.phi * iota_2, min=env.DL_LOG_ARG_CLIP))
    
    # Regime-specific calculations
    if regime == 2:  # Regime 2
        # Government investment in green capital
        iota_g = tau * safe_divide(1-eta, eta) * A_k2_val
        #iota_g = tau * ((1-eta)/eta) * A_k2_val 
        Phi_g = (1/env.phi) * torch.log(torch.clamp(1 + env.phi * iota_g, min=env.DL_LOG_ARG_CLIP))
        
        # Capital returns
        r_k1 = Phi_1 + Phi_g - env.delta + mu_q1 + (A_k1_val - iota_1) / q1
        r_k2 = Phi_2 - env.delta + mu_q2 + (A_k2_val * (1-tau) - iota_2) / q2
    else:  # Regime 1
        # Capital returns
        r_k1 = Phi_1 - env.delta + mu_q1 + (A_k1_val - iota_1) / q1
        r_k2 = Phi_2 - env.delta + mu_q2 + (A_k2_val * (1-tau) - iota_2) / q2
    
    # Calculate S (denominator in theta calculation)
    s = (sigma_q1_a - sigma_q2_a)**2 + (sigma_q1_b - sigma_q2_b)**2
    
    # Portfolio weight theta using safe division
    theta_numerator = -sigma_q2_a * (sigma_q1_a - sigma_q2_a) - sigma_q2_b * (sigma_q1_b - sigma_q2_b) + (r_k1 - r_k2)
    theta = safe_divide(theta_numerator, s)
    
    # Clip theta to [0, 1] range
    theta = torch.clamp(theta, 0, 1)
    
    # Agent wealth drift
    if regime == 2:
        mu_n = -env.rho + theta * r_k1 + (1-theta) * r_k2
    else:  # Regime 1
        mu_n = -env.rho + theta * r_k1 + (1-theta) * r_k2 + tau * (1-eta) * A_k2_val
    
    # Agent wealth volatility
    sigma_n_a = theta * sigma_q1_a + (1-theta) * sigma_q2_a
    sigma_n_b = theta * sigma_q1_b + (1-theta) * sigma_q2_b
    
    # State variable drifts
    if regime == 2: 
        mu_eta = eta * (1-eta) * (Phi_1 + Phi_g - Phi_2)
    else: # regime 1
        mu_eta = eta * (1-eta) * (Phi_1 - Phi_2)
    mu_H = env.mu_2 * (1-eta) - env.eps * H
    mu_tau = env.alpha * (env.tau_0 - tau)
    
    return {
        'A_k1': A_k1_val,
        'A_k2': A_k2_val,
        'iota_1': iota_1,
        'iota_2': iota_2,
        'Phi_1': Phi_1,
        'Phi_2': Phi_2,
        'r_k1': r_k1,
        'r_k2': r_k2,
        's': s,
        'theta': theta,
        'mu_n': mu_n,
        'sigma_n_a': sigma_n_a,
        'sigma_n_b': sigma_n_b,
        'mu_eta': mu_eta,
        'mu_H': mu_H,
        'mu_tau': mu_tau
    }

# Compute losses for the equilibrium model
def compute_equilibrium_losses(eta, H, tau, model, regime):
    """
    Compute all losses for the equilibrium model based on consistency conditions,
    market clearing, and boundary conditions
    """
    # Get model predictions
    q1, mu_q1, sigma_q1_a, sigma_q1_b, q2, mu_q2, sigma_q2_a, sigma_q2_b = model(torch.cat([eta, H, tau], dim=1))
    
    # Compute economic variables
    econ_vars = compute_economic_variables(
        eta, H, tau, q1, mu_q1, sigma_q1_a, sigma_q1_b, q2, mu_q2, sigma_q2_a, sigma_q2_b, regime
    )
    
    # Extract variables from econ_vars
    A_k1 = econ_vars['A_k1']
    A_k2 = econ_vars['A_k2']
    iota_1 = econ_vars['iota_1']
    iota_2 = econ_vars['iota_2']
    s = econ_vars['s']
    theta = econ_vars['theta']
    mu_eta = econ_vars['mu_eta']
    mu_H = econ_vars['mu_H']
    mu_tau = econ_vars['mu_tau']
    
    # Compute derivatives needed for consistency conditions
    # For q1
    q1.requires_grad_(True)
    eta.requires_grad_(True)
    H.requires_grad_(True)
    tau.requires_grad_(True)
    SV = torch.cat([eta, H, tau], dim=1)
    SV.requires_grad_(True)
    
    # Get fresh predictions with gradients enabled
    q1, mu_q1, sigma_q1_a, sigma_q1_b, q2, mu_q2, sigma_q2_a, sigma_q2_b = model(SV)
    
    # Compute gradients for q1
    dq1_dSV = torch.autograd.grad(
        q1, SV, grad_outputs=torch.ones_like(q1), create_graph=True, retain_graph=True
    )[0]
    
    dq1_deta = dq1_dSV[:, 0:1]
    dq1_dH = dq1_dSV[:, 1:2]
    dq1_dtau = dq1_dSV[:, 2:3]
    
    # Second derivatives for q1
    H_only = SV[:, 1:2].detach().clone().requires_grad_(True)
    eta_fixed = SV[:, 0:1].detach()
    tau_fixed = SV[:, 2:3].detach()
    SV_H_only = torch.cat([eta_fixed, H_only, tau_fixed], dim=1)
    
    # Recompute q1 with fixed eta and new H tensor
    q1_recomp, _, _, _, _, _, _, _ = model(SV_H_only)
    
    # Get first derivative with respect to H only
    dq1_dH_recomp = torch.autograd.grad(
        q1_recomp, H_only, grad_outputs=torch.ones_like(q1_recomp), 
        create_graph=True, retain_graph=True
    )[0]
    
    # Get second derivative with respect to H only
    d2q1_dH2 = torch.autograd.grad(
        dq1_dH_recomp, H_only, grad_outputs=torch.ones_like(dq1_dH_recomp), 
        create_graph=True, retain_graph=True
    )[0]
    
    # Compute second derivatives for tau
    tau_only = SV[:, 2:3].detach().clone().requires_grad_(True)
    eta_fixed = SV[:, 0:1].detach()
    H_fixed = SV[:, 1:2].detach()
    SV_tau_only = torch.cat([eta_fixed, H_fixed, tau_only], dim=1)
    
    # Recompute q1 with fixed eta, H and new tau tensor
    q1_recomp_tau, _, _, _, _, _, _, _ = model(SV_tau_only)
    
    # Get first derivative with respect to tau only
    dq1_dtau_recomp = torch.autograd.grad(
        q1_recomp_tau, tau_only, grad_outputs=torch.ones_like(q1_recomp_tau), 
        create_graph=True, retain_graph=True
    )[0]
    
    # Get second derivative with respect to tau only
    d2q1_dtau2 = torch.autograd.grad(
        dq1_dtau_recomp, tau_only, grad_outputs=torch.ones_like(dq1_dtau_recomp), 
        create_graph=True
    )[0]
    
    # Repeat for q2
    # Compute gradients for q2
    dq2_dSV = torch.autograd.grad(
        q2, SV, grad_outputs=torch.ones_like(q2), create_graph=True, retain_graph=True
    )[0]
    
    dq2_deta = dq2_dSV[:, 0:1]
    dq2_dH = dq2_dSV[:, 1:2]
    dq2_dtau = dq2_dSV[:, 2:3]
    
    # Second derivatives for q2
    H_only = SV[:, 1:2].detach().clone().requires_grad_(True)
    eta_fixed = SV[:, 0:1].detach()
    tau_fixed = SV[:, 2:3].detach()
    SV_H_only = torch.cat([eta_fixed, H_only, tau_fixed], dim=1)
    
    # Recompute q2 with fixed eta and new H tensor
    _, _, _, _, q2_recomp, _, _, _ = model(SV_H_only)
    
    # Get first derivative with respect to H only
    dq2_dH_recomp = torch.autograd.grad(
        q2_recomp, H_only, grad_outputs=torch.ones_like(q2_recomp), 
        create_graph=True, retain_graph=True
    )[0]
    
    # Get second derivative with respect to H only
    d2q2_dH2 = torch.autograd.grad(
        dq2_dH_recomp, H_only, grad_outputs=torch.ones_like(dq2_dH_recomp), 
        create_graph=True, retain_graph=True
    )[0]
    
    # Compute second derivatives for tau
    tau_only = SV[:, 2:3].detach().clone().requires_grad_(True)
    eta_fixed = SV[:, 0:1].detach()
    H_fixed = SV[:, 1:2].detach()
    SV_tau_only = torch.cat([eta_fixed, H_fixed, tau_only], dim=1)
    
    # Recompute q2 with fixed eta, H and new tau tensor
    _, _, _, _, q2_recomp_tau, _, _, _ = model(SV_tau_only)
    
    # Get first derivative with respect to tau only
    dq2_dtau_recomp = torch.autograd.grad(
        q2_recomp_tau, tau_only, grad_outputs=torch.ones_like(q2_recomp_tau), 
        create_graph=True, retain_graph=True
    )[0]
    
    # Get second derivative with respect to tau only
    d2q2_dtau2 = torch.autograd.grad(
        dq2_dtau_recomp, tau_only, grad_outputs=torch.ones_like(dq2_dtau_recomp), 
        create_graph=True
    )[0]
    
    # Compute residuals for consistency conditions
    
    # First consistency condition for q1
    consistency_q1 = q1 * mu_q1 - (dq1_deta * mu_eta + dq1_dH * mu_H + dq1_dtau * mu_tau + 
                                    0.5 * ((H * env.sigma_H)**2 * d2q1_dH2 + env.sigma_tau**2 * d2q1_dtau2))
    
    # Second consistency condition for q1
    consistency_q1_a = q1 * sigma_q1_a - H * env.sigma_H * dq1_dH
    
    # Third consistency condition for q1
    consistency_q1_b = q1 * sigma_q1_b - env.sigma_tau * dq1_dtau
    
    # First consistency condition for q2
    consistency_q2 = q2 * mu_q2 - (dq2_deta * mu_eta + dq2_dH * mu_H + dq2_dtau * mu_tau + 
                                    0.5 * ((H * env.sigma_H)**2 * d2q2_dH2 + env.sigma_tau**2 * d2q2_dtau2))
    
    # Second consistency condition for q2
    consistency_q2_a = q2 * sigma_q2_a - H * env.sigma_H * dq2_dH
    
    # Third consistency condition for q2
    consistency_q2_b = q2 * sigma_q2_b - env.sigma_tau * dq2_dtau
    
    # Portfolio market clearing conditions

    # s_theta = s * theta  # Avoid division by zero by multiplying both sides
    # s_one_minus_theta = s * (1 - theta)
    # clearing_q = q2 * (1-eta) * s_theta - q1 * eta * s_one_minus_theta
    # clearing_mu_q = mu_q2 * (1-eta) * s_theta - mu_q1 * eta * s_one_minus_theta
    # clearing_sigma_q_a = sigma_q2_a * (1-eta) * s_theta - sigma_q1_a * eta * s_one_minus_theta
    # clearing_sigma_q_b = sigma_q2_b * (1-eta) * s_theta - sigma_q1_b * eta * s_one_minus_theta

    clearing_q = q2 * (1-eta) * theta - q1 * eta * (1-theta)
    clearing_mu_q = mu_q2 * (1-eta) * theta - mu_q1 * eta * (1-theta)
    clearing_sigma_q_a = sigma_q2_a * (1-eta) * theta - sigma_q1_a * eta * (1-theta)
    clearing_sigma_q_b = sigma_q2_b * (1-eta) * theta - sigma_q1_b * eta * (1-theta)

    
    
    # Goods market clearing
    if regime == 2:  # Regime 2
        goods_market_clearing = A_k1 * eta + A_k2 * (1-tau) * (1-eta) - iota_1 * eta - iota_2 * (1-eta) - env.rho * (eta * q1 + (1-eta) * q2)
    else:  # Regime 1
        goods_market_clearing = A_k1 * eta + A_k2 * (1-eta) - iota_1 * eta - iota_2 * (1-eta) - env.rho * (eta * q1 + (1-eta) * q2)
    
    # Calculate boundary condition residuals
    # Get boundary predictions from model
    q1_bound, mu_q1_bound, sigma_q1_a_bound, sigma_q1_b_bound, q2_bound, mu_q2_bound, sigma_q2_a_bound, sigma_q2_b_bound = model.boundary_forward(eta, H, tau, regime)
    
    # Calculate boundary condition losses
    # For eta = 1
    eta_1_mask = (eta > env.ETA_ONE_BOUNDARY_THRESHOLD).float()
    boundary_eta_1_loss = torch.mean(eta_1_mask * ((q1 - q1_bound)**2 + 
                                                   (mu_q1 - mu_q1_bound)**2 + 
                                                   (sigma_q1_a - sigma_q1_a_bound)**2 + 
                                                   (sigma_q1_b - sigma_q1_b_bound)**2))
    
    # For eta = 0
    eta_0_mask = (eta < env.ETA_BOUNDARY_THRESHOLD).float()
    boundary_eta_0_loss = torch.mean(eta_0_mask * ((q2 - q2_bound)**2 + 
                                                   (mu_q2 - mu_q2_bound)**2 + 
                                                   (sigma_q2_a - sigma_q2_a_bound)**2 + 
                                                   (sigma_q2_b - sigma_q2_b_bound)**2))
    
    # For H = 0
    H_0_mask = (H < env.H_BOUNDARY_THRESHOLD).float()
    boundary_H_0_loss = torch.mean(H_0_mask * ((sigma_q1_a)**2 + (sigma_q2_a)**2))
    
    # Calculate loss for each consistency condition with adaptive weighting
    consistency_loss_q1 = torch.mean(consistency_q1 ** 2) #weighted_equation_loss(consistency_q1, s)
    consistency_loss_q1_a = torch.mean(consistency_q1_a ** 2)#weighted_equation_loss(consistency_q1_a, s)
    consistency_loss_q1_b = torch.mean(consistency_q1_b ** 2)#weighted_equation_loss(consistency_q1_b, s)
    consistency_loss_q2 = torch.mean(consistency_q2 ** 2)#weighted_equation_loss(consistency_q2, s)
    consistency_loss_q2_a = torch.mean(consistency_q2_a ** 2) #weighted_equation_loss(consistency_q2_a, s)
    consistency_loss_q2_b = torch.mean(consistency_q2_b ** 2)#weighted_equation_loss(consistency_q2_b, s)
    
    clearing_loss_q = torch.mean(clearing_q ** 2)#weighted_equation_loss(clearing_q, s)
    clearing_loss_mu_q = torch.mean(clearing_mu_q ** 2)#weighted_equation_loss(clearing_mu_q, s)
    clearing_loss_sigma_q_a = torch.mean(clearing_sigma_q_a ** 2) #weighted_equation_loss(clearing_sigma_q_a, s)
    clearing_loss_sigma_q_b = torch.mean(clearing_sigma_q_b ** 2) #weighted_equation_loss(clearing_sigma_q_b, s)
    
    goods_market_clearing_loss = torch.mean(goods_market_clearing**2)
    
    # Combine losses with appropriate weights
    total_consistency_loss = (
        consistency_loss_q1 + consistency_loss_q1_a + consistency_loss_q1_b +
        consistency_loss_q2 + consistency_loss_q2_a + consistency_loss_q2_b
    ) * env.CONSISTENCY_WEIGHT
    
    total_clearing_loss = (
        clearing_loss_q + clearing_loss_mu_q + 
        clearing_loss_sigma_q_a + clearing_loss_sigma_q_b
    ) * env.MARKET_CLEARING_WEIGHT
    
    total_boundary_loss = (
        boundary_eta_1_loss + boundary_eta_0_loss + boundary_H_0_loss
    ) * env.BOUNDARY_WEIGHT
    
    goods_market_loss = goods_market_clearing_loss * env.MARKET_CLEARING_WEIGHT
    
    total_loss = total_consistency_loss + total_clearing_loss + total_boundary_loss + goods_market_loss
    
    # Return individual losses for monitoring
    loss_dict = {
        'total_loss': total_loss,
        'consistency_q1': consistency_loss_q1,
        'consistency_q1_a': consistency_loss_q1_a,
        'consistency_q1_b': consistency_loss_q1_b,
        'consistency_q2': consistency_loss_q2,
        'consistency_q2_a': consistency_loss_q2_a,
        'consistency_q2_b': consistency_loss_q2_b,
        'clearing_q': clearing_loss_q,
        'clearing_mu_q': clearing_loss_mu_q,
        'clearing_sigma_q_a': clearing_loss_sigma_q_a,
        'clearing_sigma_q_b': clearing_loss_sigma_q_b,
        'goods_market': goods_market_clearing_loss,
        'boundary_eta_1': boundary_eta_1_loss,
        'boundary_eta_0': boundary_eta_0_loss,
        'boundary_H_0': boundary_H_0_loss,
    }
    
    return loss_dict, econ_vars

# Compute HJBE for value function
def compute_hjbe_loss(eta, H, tau, model, value_model, regime):
    """Compute HJB equation loss for the value function"""
    # Forward pass for equilibrium variables
    q1, mu_q1, sigma_q1_a, sigma_q1_b, q2, mu_q2, sigma_q2_a, sigma_q2_b = model(torch.cat([eta, H, tau], dim=1))
    
    # Compute economic variables
    econ_vars = compute_economic_variables(
        eta, H, tau, q1, mu_q1, sigma_q1_a, sigma_q1_b, q2, mu_q2, sigma_q2_a, sigma_q2_b, regime
    )
    
    # Extract needed variables
    mu_n = econ_vars['mu_n']
    sigma_n_a = econ_vars['sigma_n_a']
    sigma_n_b = econ_vars['sigma_n_b']
    mu_eta = econ_vars['mu_eta']
    mu_H = econ_vars['mu_H']
    mu_tau = econ_vars['mu_tau']
    s = econ_vars['s']
    
    # Get value function and its derivatives
    SV = torch.cat([eta, H, tau], dim=1)
    SV.requires_grad_(True)
    W = value_model(SV)
    
    # Compute first derivatives
    W_grad = torch.autograd.grad(
        W, SV, grad_outputs=torch.ones_like(W), create_graph=True, retain_graph=True
    )[0]
    
    W_eta = W_grad[:, 0:1]
    W_H = W_grad[:, 1:2]
    W_tau = W_grad[:, 2:3]
    
    # Compute second derivatives
    # For H
    H_only = SV[:, 1:2].detach().clone().requires_grad_(True)
    eta_fixed = SV[:, 0:1].detach()
    tau_fixed = SV[:, 2:3].detach()
    SV_H_only = torch.cat([eta_fixed, H_only, tau_fixed], dim=1)
    
    W_recomputed = value_model(SV_H_only)
    W_H_recomputed = torch.autograd.grad(
        W_recomputed, H_only, grad_outputs=torch.ones_like(W_recomputed), 
        create_graph=True, retain_graph=True
    )[0]
    
    W_H_H = torch.autograd.grad(
        W_H_recomputed, H_only, grad_outputs=torch.ones_like(W_H_recomputed), 
        create_graph=True, retain_graph=True
    )[0]
    
    # For tau
    tau_only = SV[:, 2:3].detach().clone().requires_grad_(True)
    eta_fixed = SV[:, 0:1].detach()
    H_fixed = SV[:, 1:2].detach()
    SV_tau_only = torch.cat([eta_fixed, H_fixed, tau_only], dim=1)
    
    W_recomputed_tau = value_model(SV_tau_only)
    W_tau_recomputed = torch.autograd.grad(
        W_recomputed_tau, tau_only, grad_outputs=torch.ones_like(W_recomputed_tau), 
        create_graph=True, retain_graph=True
    )[0]
    
    W_tau_tau = torch.autograd.grad(
        W_tau_recomputed, tau_only, grad_outputs=torch.ones_like(W_tau_recomputed), 
        create_graph=True
    )[0]
    
    # HJBE components
    log_rho = torch.log(torch.tensor(env.rho, device=device))
    mu_n_term = mu_n / env.rho
    drift_eta = mu_eta * W_eta
    drift_H = mu_H * W_H
    drift_tau = mu_tau * W_tau
    diffusion_n = -((sigma_n_a**2 + sigma_n_b**2) / env.rho)
    diffusion_H = 0.5 * (H * env.sigma_H)**2 * W_H_H
    diffusion_tau = 0.5 * env.sigma_tau**2 * W_tau_tau
    discount = env.rho * W
    
    # HJB residual
    hjb_residual = log_rho + mu_n_term + drift_eta + drift_H + drift_tau + diffusion_n + diffusion_H + diffusion_tau - discount
    
    # Shape constraint: W_H should be negative
    shape_constraint = torch.relu(W_H)
    
    # Compute loss
    # multiply through by s & renormalize to avoid explosion near zero. ISSUE: currently feeding in say s=10^-10, but clipped to 10^-6. redo
    # s_multiplier = torch.clamp(s, min=env.S_MIN)  # ** 2 # try single power instead given clamp in weight div
    # hjb_resid_mult = hjb_residual * s_multiplier

    # use adaptive weight much closer to 1 (weights by s ** 0.4)
    #hjb_loss = weighted_equation_loss(hjb_resid_mult, s_multiplier, alpha=env.HJBE_ADAPTIVE_ALPHA) # normalize by one power of s
    hjb_loss = torch.mean(hjb_residual ** 2)
    # IN FUTURE: CAN ALSO TRY W/O S AT ALL! 
    #
    shape_loss = torch.mean(shape_constraint**2) # don't worry about renormalizing, s doesn't affect this one.

    zero_avoidance_loss =  torch.mean(1 / (torch.abs(W) + 0.1))
    
    total_loss = hjb_loss * env.HJBE_WEIGHT + env.HJBE_SHAPE_WEIGHT * shape_loss + env.HJBE_ZERO_AV_LOSS_WEIGHT * zero_avoidance_loss
    
    return total_loss, hjb_loss, shape_loss, zero_avoidance_loss

# Pre-training function for value function
def pretrain_value_function(value_model, equilibrium_model, regime):
    """Pre-train the value function model to match the initial guess"""
    print("Starting value function pre-training...")
    optimizer = torch.optim.Adam(value_model.parameters(), lr=env.learning_rate)
    
    losses = []
    
    # Create a larger batch size for pre-training
    batch_size = env.model3_batch_size * 2
    
    # Use uniform sampling for better coverage during pre-training
    for epoch in tqdm(range(env.model3_pretrain_epochs)):
        # Sample state variables
        eta_np = np.random.uniform(env.eta_min + 0.01, env.eta_max - 0.01, (batch_size, 1))
        H_np = np.random.uniform(env.H_min, env.H_max, (batch_size, 1))
        tau_np = np.random.uniform(0, env.TAU_MAX, (batch_size, 1))
        
        eta = torch.tensor(eta_np, dtype=torch.float32, device=device)
        H = torch.tensor(H_np, dtype=torch.float32, device=device)
        tau = torch.tensor(tau_np, dtype=torch.float32, device=device)
        
        # Create state variable tensor
        SV = torch.cat([eta, H, tau], dim=1)
        
        # Forward pass
        W_predicted = value_model(SV)
        
        # Compute target values based on the initial guess
        W_target_np = np.array([env.initial_guess(e[0], h[0], t[0]) for e, h, t in zip(eta_np, H_np, tau_np)])
        W_target = torch.tensor(W_target_np, dtype=torch.float32, device=device).reshape(-1, 1)
        
        # Compute MSE loss
        loss = nn.MSELoss()(W_predicted, W_target)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        
        # Clip gradients to prevent instability
        torch.nn.utils.clip_grad_norm_(value_model.parameters(), max_norm=10.0)
        
        optimizer.step()
        
        if epoch % env.print_every == 0:
            losses.append(loss.item())
            print(f"Epoch {epoch}, Loss: {loss.item():.6f}")
    
    # Save pre-training loss
    np.save(f"{OUTPUT_DIR}/value_pretrain_loss.npy", np.array(losses))
    
    print(f"Value function pre-training completed with final loss: {loss.item():.6f}")
    return losses

# Compute value function slices for visualization
# In compute_value_function_slices function, modify to calculate W_effective
def compute_value_function_slices(value_model, equilibrium_model, regime):
    """Precompute value function slices for visualization with proper tensor handling"""
    print("Computing value function slices for visualization...")
    print(f"tau_0 check: {env.tau_0}")

    # Helper function to safely convert tensors to CPU numpy arrays
    def to_numpy(tensor):
        if tensor is None:
            return None
        if hasattr(tensor, 'cpu'):
            return tensor.cpu().detach().numpy()
        if hasattr(tensor, 'detach'):
            return tensor.detach().numpy()
        if hasattr(tensor, 'numpy'):
            return tensor.numpy()
        return tensor

    with torch.no_grad():
        # Create grids for evaluation
        eta_grid = np.linspace(env.eta_min, env.eta_max, 100)
        H_grid = np.linspace(env.H_min, env.H_max, 100)
        tau_grid = np.linspace(0, env.TAU_MAX, 100)
        
        # 1. Compute W(eta, H, tau=tau_0) and W_effective
        eta_mesh, H_mesh = np.meshgrid(eta_grid, H_grid, indexing='ij')
        W_eta_H = np.zeros_like(eta_mesh)
        W_effective_eta_H = np.zeros_like(eta_mesh)
        
        # Process in batches
        batch_size = 1000
        points = []
        
        for i in range(len(eta_grid)):
            for j in range(len(H_grid)):
                points.append((eta_grid[i], H_grid[j], env.tau_0)) # all of these take tau_0. need to separately run over tau
                
        for i in range(0, len(points), batch_size):
            batch_points = points[i:i+batch_size]
            batch_tensor = torch.tensor(batch_points, dtype=torch.float32, device=device)
            
            # Get equilibrium objects for W_effective
            q1, _, _, _, q2, _, _, _ = equilibrium_model(batch_tensor)
            
            # Get value function
            batch_W = value_model(batch_tensor)
            
            # Convert to numpy arrays immediately
            q1_np = to_numpy(q1)
            q2_np = to_numpy(q2)
            batch_W_np = to_numpy(batch_W)
            
            # Compute W_effective = log(η q1 + (1-η)q2)/ρ + W
            eta_values = np.array([p[0] for p in batch_points])
            weighted_q = (eta_values * q1_np.flatten() + 
                         (1 - eta_values) * q2_np.flatten())
            batch_W_effective = np.log(weighted_q) / env.rho + batch_W_np.flatten()
            
            for k, (batch_i, batch_j, _) in enumerate(batch_points):
                idx_i = np.where(eta_grid == batch_i)[0][0]
                idx_j = np.where(H_grid == batch_j)[0][0]
                W_eta_H[idx_i, idx_j] = batch_W_np.flatten()[k]
                W_effective_eta_H[idx_i, idx_j] = batch_W_effective[k]
                
        # Continue with similar updates for W_eta_tau and W_tau_lines
        # 2. Compute W(eta, H=H_0, tau) and W_effective(eta, H=H_0, tau)
        eta_mesh_tau, tau_mesh = np.meshgrid(eta_grid, tau_grid, indexing='ij')
        W_eta_tau = np.zeros_like(eta_mesh_tau)
        W_effective_eta_tau = np.zeros_like(eta_mesh_tau)

        # Process in batches
        points = []

        for i in range(len(eta_grid)):
            for j in range(len(tau_grid)):
                points.append((eta_grid[i], env.sim_H_0, tau_grid[j]))
                
        for i in range(0, len(points), batch_size):
            batch_points = points[i:i+batch_size]
            batch_tensor = torch.tensor(batch_points, dtype=torch.float32, device=device)
            
            # Get equilibrium objects for W_effective
            q1, _, _, _, q2, _, _, _ = equilibrium_model(batch_tensor)
            
            # Get value function
            batch_W = value_model(batch_tensor)
            
            # Convert to numpy immediately
            q1_np = to_numpy(q1)
            q2_np = to_numpy(q2)
            batch_W_np = to_numpy(batch_W)
            
            # Compute W_effective
            eta_values = np.array([p[0] for p in batch_points])
            weighted_q = (eta_values * q1_np.flatten() + 
                         (1 - eta_values) * q2_np.flatten())
            batch_W_effective = np.log(weighted_q) / env.rho + batch_W_np.flatten()
            
            for k, (batch_i, _, batch_j) in enumerate(batch_points):
                idx_i = np.where(eta_grid == batch_i)[0][0]
                idx_j = np.where(tau_grid == batch_j)[0][0]
                W_eta_tau[idx_i, idx_j] = batch_W_np.flatten()[k]
                W_effective_eta_tau[idx_i, idx_j] = batch_W_effective[k]

        # 3. Compute W(tau), W_effective(tau), q1(tau), q2(tau) for various (eta, H) combinations
        combinations = [
            (env.sim_eta_0, env.sim_H_0, "η=η_0, H=H_0"),
            (env.sim_eta_0, env.H_high, "η=η_0, H=H_high"),
            (env.eta_high, env.sim_H_0, "η=η_high, H=H_0"),
            (env.eta_high, env.H_high, "η=η_high, H=H_high")
        ]

        W_tau_lines = {}
        W_effective_tau_lines = {}
        q1_vals = {}
        q2_vals = {}

        for eta_val, H_val, label in combinations:
            # Create points for this combination
            points = [(eta_val, H_val, tau) for tau in tau_grid]
            points_tensor = torch.tensor(points, dtype=torch.float32, device=device)
            
            # Get equilibrium objects for W_effective
            q1, _, _, _, q2, _, _, _ = equilibrium_model(points_tensor)

            # Convert to numpy immediately
            q1_np = to_numpy(q1).flatten()
            q2_np = to_numpy(q2).flatten()
            
            # Get value function
            W_values = to_numpy(value_model(points_tensor)).flatten()
            
            # Compute W_effective
            eta_array = np.full(len(points), eta_val)
            weighted_q = (eta_array * q1_np + (1 - eta_array) * q2_np)
            W_effective_values = np.log(weighted_q) / env.rho + W_values
            
            # Store numpy arrays directly
            W_tau_lines[label] = W_values
            W_effective_tau_lines[label] = W_effective_values
            q1_vals[label] = q1_np
            q2_vals[label] = q2_np

        # Calculate optimal tau values for each combination
        W_effective_max_tau = {}
        
        for eta_val, H_val, label in combinations:
            # Find tau that maximizes W_effective
            W_eff_values = W_effective_tau_lines[label]
            max_idx = np.argmax(W_eff_values)
            max_tau = tau_grid[max_idx]
            W_effective_max_tau[label] = {
                'tau': max_tau,
                'value': W_eff_values[max_idx]
            }
            
        # Save all results as numpy arrays
        np.savez(
            f"{OUTPUT_DIR}/value_function_slices_{TAG}.npz",
            eta_grid=eta_grid,
            H_grid=H_grid,
            tau_grid=tau_grid,
            W_eta_H=W_eta_H,
            W_eta_tau=W_eta_tau,
            W_tau_lines=W_tau_lines,
            q1_vals=q1_vals,
            q2_vals=q2_vals,
            W_effective_eta_H=W_effective_eta_H,
            W_effective_eta_tau=W_effective_eta_tau,
            W_effective_tau_lines=W_effective_tau_lines,
            W_effective_max_tau=W_effective_max_tau,
            combinations=combinations,
            tau_star=env.tau_star
        )
# Forward simulation of the model
def simulate_forward(equilibrium_model, value_model, regime):
    """Simulate the economy forward from an initial state"""
    print("Starting forward simulation...")
    
    # Initial state
    eta_0 = env.sim_eta_0
    H_0 = env.sim_H_0
    tau_0 = env.sim_Tau_0
    K_0 = env.sim_K_0
    
    # Time steps
    N_T = env.sim_N_T
    dt = env.sim_dt
    
    # Number of simulations
    n_sims = env.sim_num_sims
    
    # Initialize arrays for results
    eta_results = np.zeros((n_sims, N_T + 1))
    H_results = np.zeros((n_sims, N_T + 1))
    tau_results = np.zeros((n_sims, N_T + 1))
    K_results = np.zeros((n_sims, N_T + 1))
    iota_1_results = np.zeros((n_sims, N_T + 1))
    iota_2_results = np.zeros((n_sims, N_T + 1))
    value_results = np.zeros((n_sims, N_T + 1))
    
    # Set models to evaluation mode
    equilibrium_model.eval()
    value_model.eval()
    
    with torch.no_grad():
        for sim in tqdm(range(n_sims)):
            # Initial values
            eta_t = eta_0
            H_t = H_0
            tau_t = tau_0
            K_t = K_0
            
            # Store initial values
            eta_results[sim, 0] = eta_t
            H_results[sim, 0] = H_t
            tau_results[sim, 0] = tau_t
            K_results[sim, 0] = K_t
            
            # Initial control values
            eta_tensor = torch.tensor([[eta_t]], dtype=torch.float32, device=device)
            H_tensor = torch.tensor([[H_t]], dtype=torch.float32, device=device)
            tau_tensor = torch.tensor([[tau_t]], dtype=torch.float32, device=device)
            
            q1, mu_q1, sigma_q1_a, sigma_q1_b, q2, mu_q2, sigma_q2_a, sigma_q2_b = equilibrium_model(
                torch.cat([eta_tensor, H_tensor, tau_tensor], dim=1)
            )
            
            econ_vars = compute_economic_variables(
                eta_tensor, H_tensor, tau_tensor, q1, mu_q1, sigma_q1_a, sigma_q1_b, 
                q2, mu_q2, sigma_q2_a, sigma_q2_b, regime
            )
            
            iota_1_t = econ_vars['iota_1'].item()
            iota_2_t = econ_vars['iota_2'].item()
            
            # Calculate initial value function
            W_t = value_model(torch.cat([eta_tensor, H_tensor, tau_tensor], dim=1)).item()
            value_t = W_t + np.log(K_t) / env.rho
            iota_1_results[sim, 0] = iota_1_t
            iota_2_results[sim, 0] = iota_2_t
            value_results[sim, 0] = value_t
            
            # Generate all random shocks at once
            dW_H_values = np.random.normal(loc=0, scale=np.sqrt(dt), size=N_T)
            dW_tau_values = np.random.normal(loc=0, scale=np.sqrt(dt), size=N_T)
            
            # Main simulation loop
            for t in range(N_T):
                # Current Brownian increments
                dW_H = dW_H_values[t]
                dW_tau = dW_tau_values[t]
                
                # Get economic variables at current state
                eta_tensor = torch.tensor([[eta_t]], dtype=torch.float32, device=device)
                H_tensor = torch.tensor([[H_t]], dtype=torch.float32, device=device)
                tau_tensor = torch.tensor([[tau_t]], dtype=torch.float32, device=device)
                
                q1, mu_q1, sigma_q1_a, sigma_q1_b, q2, mu_q2, sigma_q2_a, sigma_q2_b = equilibrium_model(
                    torch.cat([eta_tensor, H_tensor, tau_tensor], dim=1)
                )
                
                econ_vars = compute_economic_variables(
                    eta_tensor, H_tensor, tau_tensor, q1, mu_q1, sigma_q1_a, sigma_q1_b, 
                    q2, mu_q2, sigma_q2_a, sigma_q2_b, regime
                )
                
                # Extract needed variables
                
                iota_1_t = econ_vars['iota_1'].item()
                iota_2_t = econ_vars['iota_2'].item()
                Phi_1_t = econ_vars['Phi_1'].item()
                Phi_2_t = econ_vars['Phi_2'].item()
                mu_eta_t = econ_vars['mu_eta'].item()
                mu_H_t = econ_vars['mu_H'].item()
                mu_tau_t = econ_vars['mu_tau'].item()
                
                # Update eta using semi-implicit Euler with Newton's method
                def F(x):
                    return x * (1 - x) * (Phi_1_t - Phi_2_t) * dt - x + eta_t
                
                def F_prime(x):
                    return (1 - 2 * x) * (Phi_1_t - Phi_2_t) * dt - 1
                
                # Single Newton step (can be iterated for better accuracy)
                eta_next = eta_t - F(eta_t) / F_prime(eta_t)
                eta_next = np.clip(eta_next, 0, 1)  # Ensure bounds
                
                # Update H using semi-implicit Euler
                H_next = (H_t + env.mu_2 * (1 - eta_next) * dt) / (1 + env.eps * dt - env.sigma_H * dW_H)
                H_next = max(H_next, 0)  # Ensure non-negative
                
                # Update tau using Euler-Maruyama for Ornstein-Uhlenbeck
                tau_next = tau_t + env.alpha * (env.tau_0 - tau_t) * dt + env.sigma_tau * dW_tau
                tau_next = max(tau_next, 0)  # Ensure non-negative
                
                # Update K
                K_next = K_t * (1 + (eta_t * Phi_1_t + (1 - eta_t) * Phi_2_t - env.delta) * dt)
                
                # Store results
                eta_t = eta_next
                H_t = H_next
                tau_t = tau_next
                K_t = K_next
                
                eta_results[sim, t+1] = eta_t
                H_results[sim, t+1] = H_t
                tau_results[sim, t+1] = tau_t
                K_results[sim, t+1] = K_t
                
                # Calculate value function at new state
                eta_tensor = torch.tensor([[eta_t]], dtype=torch.float32, device=device)
                H_tensor = torch.tensor([[H_t]], dtype=torch.float32, device=device)
                tau_tensor = torch.tensor([[tau_t]], dtype=torch.float32, device=device)
                
                q1, mu_q1, sigma_q1_a, sigma_q1_b, q2, mu_q2, sigma_q2_a, sigma_q2_b = equilibrium_model(
                    torch.cat([eta_tensor, H_tensor, tau_tensor], dim=1)
                )
                
                econ_vars = compute_economic_variables(
                    eta_tensor, H_tensor, tau_tensor, q1, mu_q1, sigma_q1_a, sigma_q1_b, 
                    q2, mu_q2, sigma_q2_a, sigma_q2_b, regime
                )
                
                iota_1_t = econ_vars['iota_1'].item()
                iota_2_t = econ_vars['iota_2'].item()
                
                W_t = value_model(torch.cat([eta_tensor, H_tensor, tau_tensor], dim=1)).item()
                value_t = W_t + np.log(K_t) / env.rho

             
                iota_1_results[sim, t+1] = iota_1_t
                iota_2_results[sim, t+1] = iota_2_t
                value_results[sim, t+1] = value_t
    
    # Save simulation results
    results = {
        'eta': eta_results,
        'H': H_results,
        'tau': tau_results,
        'K': K_results,
        'iota_1': iota_1_results,
        'iota_2': iota_2_results,
        'value': value_results,
        'dt': dt,
        'N_T': N_T,
        'regime': regime
    }
    
    np.savez(f"{OUTPUT_DIR}/simulation_results_{TAG}.npz", **results)
    
    print(f"Simulation completed. Results saved to {OUTPUT_DIR}/simulation_results_{TAG}.npz")
    return results

# Main training function for equilibrium model
def train_equilibrium_model(model, regime, num_epochs):
    """Train the equilibrium model with multiple mini-batches per epoch"""
    print(f"Starting main training for tag {TAG}, regime {regime}...")
    
    # Set random seeds for reproducibility
    set_seeds(env.seed)
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=env.learning_rate, weight_decay=env.weight_decay)
    
    # For loss tracking
    epoch_loss_dict = {
        'epoch': [],
        'total_loss': [],
        'consistency_loss': [],
        'clearing_loss': [],
        'boundary_loss': [],
        'goods_market_loss': []
    }
    
    # For time tracking
    total_training_time = 0.0
    num_epochs_timed = 0
    
    # Training loop - epochs
    for epoch in tqdm(range(num_epochs)):
        # Start timing this epoch
        epoch_start_time = time.time()
        
        epoch_total_loss = 0.0
        epoch_consistency_loss = 0.0
        epoch_clearing_loss = 0.0
        epoch_boundary_loss = 0.0
        epoch_goods_market_loss = 0.0
        
        # Inner loop - mini-batches
        for mini_batch in range(env.model3_mini_batch_per_epoch):
            # Sample state variables for this mini-batch
            eta_np = sample_eta_proportional(env.model3_batch_size, env.eta_min + 0.01, env.eta_max - 0.01)
            H_np = sample_H_exponential(env.model3_batch_size, env.H_min, env.H_max)
            tau_np = sample_tau(env.model3_batch_size, env.tau_0, env.sigma_tau, env.alpha) # changed from normal 4/17
            
            eta = torch.tensor(eta_np, dtype=torch.float32, device=device)
            H = torch.tensor(H_np, dtype=torch.float32, device=device)
            tau = torch.tensor(tau_np, dtype=torch.float32, device=device)
            
            # Forward pass and loss computation
            loss_dict, econ_vars = compute_equilibrium_losses(eta, H, tau, model, regime)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss_dict['total_loss'].backward()
            
            # Clip gradients to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Accumulate losses for this epoch
            epoch_total_loss += loss_dict['total_loss'].item()
            epoch_consistency_loss += (
                loss_dict['consistency_q1'].item() + loss_dict['consistency_q1_a'].item() + 
                loss_dict['consistency_q1_b'].item() + loss_dict['consistency_q2'].item() + 
                loss_dict['consistency_q2_a'].item() + loss_dict['consistency_q2_b'].item()
            )
            epoch_clearing_loss += (
                loss_dict['clearing_q'].item() + loss_dict['clearing_mu_q'].item() + 
                loss_dict['clearing_sigma_q_a'].item() + loss_dict['clearing_sigma_q_b'].item()
            )
            epoch_boundary_loss += (
                loss_dict['boundary_eta_1'].item() + loss_dict['boundary_eta_0'].item() + 
                loss_dict['boundary_H_0'].item()
            )
            epoch_goods_market_loss += loss_dict['goods_market'].item()
        
        # Average losses across mini-batches
        epoch_total_loss /= env.model3_mini_batch_per_epoch
        epoch_consistency_loss /= env.model3_mini_batch_per_epoch
        epoch_clearing_loss /= env.model3_mini_batch_per_epoch
        epoch_boundary_loss /= env.model3_mini_batch_per_epoch
        epoch_goods_market_loss /= env.model3_mini_batch_per_epoch
        
        # Calculate epoch time and update running average
        epoch_time = time.time() - epoch_start_time
        total_training_time += epoch_time
        num_epochs_timed += 1
        avg_epoch_time = total_training_time / num_epochs_timed
        
        # Log results
        if epoch % env.print_every == 0:
            print(f"Epoch {epoch}: Loss = {epoch_total_loss:.6f}, Time = {epoch_time:.2f}s, Avg Time = {avg_epoch_time:.2f}s/epoch")
            
            epoch_loss_dict['epoch'].append(epoch)
            epoch_loss_dict['total_loss'].append(epoch_total_loss)
            epoch_loss_dict['consistency_loss'].append(epoch_consistency_loss)
            epoch_loss_dict['clearing_loss'].append(epoch_clearing_loss)
            epoch_loss_dict['boundary_loss'].append(epoch_boundary_loss)
            epoch_loss_dict['goods_market_loss'].append(epoch_goods_market_loss)
        
        # Save intermediate model
        if (epoch + 1) % env.save_every == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_total_loss,
            }, f"{OUTPUT_DIR}/equilibrium_model_{TAG}_epoch_{epoch+1}.pt")
    
    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'loss': epoch_total_loss,
    }, f"{OUTPUT_DIR}/equilibrium_model_{TAG}_final.pt")
    
    # Save loss history
    pd.DataFrame(epoch_loss_dict).to_csv(f"{OUTPUT_DIR}/equilibrium_model_{TAG}_loss.csv", index=False)
    
    # Print final timing information
    final_avg_time = total_training_time / num_epochs_timed
    print(f"Training completed for tag {TAG}!")
    print(f"Average training time: {final_avg_time:.2f} seconds per epoch")
    print(f"Total training time: {total_training_time:.2f} seconds ({total_training_time/3600:.2f} hours)")
    
    return model, epoch_loss_dict

# Train value function model
def train_value_function_model(value_model, equilibrium_model, regime):
    """Train the value function model with multiple mini-batches per epoch"""
    print(f"Starting value function training for tag {TAG}, regime {regime}...")
    
    # Set random seeds for reproducibility
    set_seeds(env.seed)
    
    # Optimizer
    optimizer = torch.optim.Adam(value_model.parameters(), lr=env.learning_rate, weight_decay=env.weight_decay)
    
    # For loss tracking
    epoch_loss_dict = {
        'epoch': [],
        'total_loss': [],
        'hjb_loss': [],
        'shape_loss': [],
        'zero_avoidance_loss': []
    }
    
    # For time tracking
    total_training_time = 0.0
    num_epochs_timed = 0
    
    # Training loop - epochs
    for epoch in tqdm(range(env.model3_hjbe_epochs)):
        # Start timing this epoch
        epoch_start_time = time.time()
        
        epoch_total_loss = 0.0
        epoch_hjb_loss = 0.0
        epoch_shape_loss = 0.0
        epoch_zero_avoidance_loss = 0.0
        
        # Inner loop - mini-batches
        for mini_batch in range(env.model3_mini_batch_per_epoch):
            # Sample state variables for this mini-batch
            assert env.eta_min > 0 and env.eta_max < 1
            eta_np = sample_eta_proportional(env.model3_batch_size, env.eta_min, env.eta_max)
            H_np = sample_H_exponential(env.model3_batch_size, env.H_min, env.H_max)
            tau_np = sample_tau(env.model3_batch_size, env.tau_0, env.sigma_tau, env.alpha)
            
            eta = torch.tensor(eta_np, dtype=torch.float32, device=device)
            H = torch.tensor(H_np, dtype=torch.float32, device=device)
            tau = torch.tensor(tau_np, dtype=torch.float32, device=device)
            
            # Forward pass and loss computation
            total_loss, hjb_loss, shape_loss, zero_avoidance_loss = compute_hjbe_loss(eta, H, tau, equilibrium_model, value_model, regime)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            total_loss.backward()
            
            # Clip gradients to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(value_model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Accumulate losses for this epoch
            epoch_total_loss += total_loss.item()
            epoch_hjb_loss += hjb_loss.item()
            epoch_shape_loss += shape_loss.item()
            epoch_zero_avoidance_loss += zero_avoidance_loss.item()
        
        # Average losses across mini-batches
        epoch_total_loss /= env.model3_mini_batch_per_epoch
        epoch_hjb_loss /= env.model3_mini_batch_per_epoch
        epoch_shape_loss /= env.model3_mini_batch_per_epoch
        epoch_zero_avoidance_loss /= env.model3_mini_batch_per_epoch
        
        # Calculate epoch time and update running average
        epoch_time = time.time() - epoch_start_time
        total_training_time += epoch_time
        num_epochs_timed += 1
        avg_epoch_time = total_training_time / num_epochs_timed
        
        # Log results
        if epoch % env.print_every == 0:
            print(f"Epoch {epoch}: Loss = {epoch_total_loss:.6f}, Time = {epoch_time:.2f}s, Avg Time = {avg_epoch_time:.2f}s/epoch")
            
            epoch_loss_dict['epoch'].append(epoch)
            epoch_loss_dict['total_loss'].append(epoch_total_loss)
            epoch_loss_dict['hjb_loss'].append(epoch_hjb_loss)
            epoch_loss_dict['shape_loss'].append(epoch_shape_loss)
            epoch_loss_dict['zero_avoidance_loss'].append(epoch_zero_avoidance_loss)
        
        # Save intermediate model
        if (epoch + 1) % env.save_every == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': value_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_total_loss,
            }, f"{OUTPUT_DIR}/value_model_{TAG}_epoch_{epoch+1}.pt")
    
    # Save final model
    torch.save({
        'model_state_dict': value_model.state_dict(),
        'loss': epoch_total_loss,
    }, f"{OUTPUT_DIR}/value_model_{TAG}_final.pt")
    
    # Save loss history
    pd.DataFrame(epoch_loss_dict).to_csv(f"{OUTPUT_DIR}/value_model_{TAG}_loss.csv", index=False)
    
    # Print final timing information
    final_avg_time = total_training_time / num_epochs_timed
    print(f"Value function training completed for tag {TAG}!")
    print(f"Average training time: {final_avg_time:.2f} seconds per epoch")
    print(f"Total training time: {total_training_time:.2f} seconds ({total_training_time/3600:.2f} hours)")
    
    return value_model, epoch_loss_dict

# Check probability of tau > 1
def check_tau_probability():
    """Check the probability of tau being GREATER THAN 1 under the stationary distribution"""
    print("bypassing this check")
    return 0
    # if env.sigma_tau == 0:
    #     assert 0 <= env.tau_0 and env.tau_0 <= 1
    #     prob_g1 = 0
    # else:
    #     std_dev = np.sqrt(env.sigma_tau**2 / (2*env.alpha))
    #     prob_g1 = 1 - norm.cdf(1, loc=env.tau_0, scale=std_dev)
    # print(f"Probability of tau > 1: {prob_g1:.8f}, tau_0: {env.tau_0}, sigma_tau: {env.sigma_tau}, alpha: {env.alpha}")
    # assert prob_g1 < 1e-4, f"Probability of tau > 1 ({prob_g1:.8f}) exceeds 1e-4"
    # return prob_g1

def compute_s_values(equilibrium_model, regime):
    """Compute s values (denominator in theta calculation) for analysis"""
    print("Computing s values for analysis...")
    
    # Create grids
    eta_grid = np.linspace(env.eta_min, env.eta_max, 100)
    H_grid = np.linspace(env.H_min, min(env.H_max, 10), 100)
    
    # Create meshgrid for later plotting
    eta_mesh, H_mesh = np.meshgrid(eta_grid, H_grid, indexing='ij')
    s_values = np.zeros_like(eta_mesh)
    
    with torch.no_grad():
        # Process in batches
        batch_size = 1000
        points = []
        
        # Create all grid points with tau=tau_0
        for i in range(len(eta_grid)):
            for j in range(len(H_grid)):
                points.append((eta_grid[i], H_grid[j], env.tau_0))
        
        # Process points in batches
        for i in tqdm(range(0, len(points), batch_size), desc="Computing s values"):
            batch_points = points[i:i+batch_size]
            batch_tensor = torch.tensor(batch_points, dtype=torch.float32, device=device)
            
            # Get equilibrium objects
            q1, mu_q1, sigma_q1_a, sigma_q1_b, q2, mu_q2, sigma_q2_a, sigma_q2_b = equilibrium_model(batch_tensor)
            
            # Calculate s values
            s = (sigma_q1_a - sigma_q2_a)**2 + (sigma_q1_b - sigma_q2_b)**2
            s_val = s.cpu().numpy()
            
            # Store in grid
            for k, (batch_i, batch_j, _) in enumerate(batch_points):
                idx_i = np.where(eta_grid == batch_i)[0][0]
                idx_j = np.where(H_grid == batch_j)[0][0]
                s_values[idx_i, idx_j] = s_val[k]
    
    # Save s_values
    np.savez(
        f"{OUTPUT_DIR}/s_values_{TAG}.npz",
        eta_grid=eta_grid,
        H_grid=H_grid,
        s_values=s_values
    )
    
    # Analyze s-values statistics
    min_s = np.min(s_values)
    max_s = np.max(s_values)
    mean_s = np.mean(s_values)
    median_s = np.median(s_values)
    
    print(f"s-values statistics:")
    print(f"Min: {min_s:.8e}")
    print(f"Max: {max_s:.8e}")
    print(f"Mean: {mean_s:.8e}")
    print(f"Median: {median_s:.8e}")
    
    # Save statistics to text file
    with open(f"{OUTPUT_DIR}/s_values_stats_{TAG}.txt", 'w') as f:
        f.write(f"s-values statistics for {TAG}:\n")
        f.write(f"Min: {min_s:.8e}\n")
        f.write(f"Max: {max_s:.8e}\n")
        f.write(f"Mean: {mean_s:.8e}\n")
        f.write(f"Median: {median_s:.8e}\n")
    
    print(f"s-values saved to {OUTPUT_DIR}/s_values_{TAG}.npz")
    print(f"s-values statistics saved to {OUTPUT_DIR}/s_values_stats_{TAG}.txt")
    
    return s_values

def main():
    # Set random seeds for reproducibility
    set_seeds(env.seed)
    
    # Check tau probability
    # prob_g1 = check_tau_probability()
    # print(f"Probability of tau > 1: {prob_g1:.8e}")

    # Initialize models
    print("Initializing models...")
    equilibrium_model = EquilibriumModel().to(device)
    value_model = ValueFunctionModel().to(device)
    
    # Step 1: Train or load equilibrium model
    eq_model_path = f"{OUTPUT_DIR}/equilibrium_model_{TAG}_final.pt"
    if env.ASSUME_TRAINED_EQUILIBRIUM and os.path.exists(eq_model_path):
        print(f"Loading existing equilibrium model from {eq_model_path}")
        eq_checkpoint = torch.load(eq_model_path, map_location=device)
        equilibrium_model.load_state_dict(eq_checkpoint['model_state_dict'])
        print(f"Equilibrium model loaded successfully (loss: {eq_checkpoint.get('loss', 'unknown')})")
    else:
        print(f"Step 1: Training equilibrium model for tag {TAG}, regime {REGIME}...")
        equilibrium_model, eq_loss_dict = train_equilibrium_model(equilibrium_model, REGIME, MAIN_NUM_EPOCHS)
        # Save is already done in train_equilibrium_model

    compute_s_values(equilibrium_model, REGIME)
    
    # Step 2: Pre-train or load pre-trained value function
    pretrain_checkpoint_path = f"{OUTPUT_DIR}/value_model_{TAG}_pretrained.pt"
    if os.path.exists(pretrain_checkpoint_path) and not env.RETRAIN_HJBE:
        print(f"Loading pre-trained value function from {pretrain_checkpoint_path}")
        pretrain_checkpoint = torch.load(pretrain_checkpoint_path, map_location=device)
        value_model.load_state_dict(pretrain_checkpoint['model_state_dict'])
        print(f"Pre-trained value function loaded successfully (loss: {pretrain_checkpoint.get('loss', 'unknown')})")
    else:
        print(f"Step 2: Pre-training value function model for tag {TAG}, regime {REGIME}...")
        pretrain_losses = pretrain_value_function(value_model, equilibrium_model, REGIME)
        
        # Save pre-trained model
        torch.save({
            'model_state_dict': value_model.state_dict(),
            'loss': pretrain_losses[-1] if pretrain_losses else None,
        }, pretrain_checkpoint_path)
        print(f"Pre-trained value function saved to {pretrain_checkpoint_path}")
    
    # Step 3: Train value function with HJBE or load final value function
    value_model_path = f"{OUTPUT_DIR}/value_model_{TAG}_final.pt"
    if os.path.exists(value_model_path) and not env.RETRAIN_HJBE:
        print(f"Loading final value function from {value_model_path}")
        value_checkpoint = torch.load(value_model_path, map_location=device)
        value_model.load_state_dict(value_checkpoint['model_state_dict'])
        print(f"Final value function loaded successfully (loss: {value_checkpoint.get('loss', 'unknown')})")
    else:
        print(f"Step 3: Training value function model for tag {TAG}, regime {REGIME}...")
        value_model, val_loss_dict = train_value_function_model(value_model, equilibrium_model, REGIME)
        # Save is already done in train_value_function_model
    
    # Step 4: Generate visualization data
    slices_file = f"{OUTPUT_DIR}/value_function_slices_{TAG}.npz"
    if os.path.exists(slices_file) and not env.RECOMPUTE_VISUALIZATIONS:
        print(f"Value function slices file already exists at {slices_file}")
    else:
        print(f"Step 4: Computing value function slices for tag {TAG}, regime {REGIME}...")
        compute_value_function_slices(value_model, equilibrium_model, REGIME)
    
    # Step 5: Run simulations
    sim_file = f"{OUTPUT_DIR}/simulation_results_{TAG}.npz"
    if os.path.exists(sim_file) and not env.RERUN_SIMULATIONS:
        print(f"Simulation results file already exists at {sim_file}")
    else:
        print(f"Step 5: Simulating economy forward for tag {TAG}, regime {REGIME}...")
        simulation_results = simulate_forward(equilibrium_model, value_model, REGIME)
    
    print("All tasks completed successfully!")

if __name__ == "__main__":
    main()
