import os
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import pandas as pd
import time
import random
from scipy import stats
import environment as env
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description='Run central planner deep learning model')
parser.add_argument('--tag', type=str, required=True,
                    help='Custom tag for saved files')
args = parser.parse_args()

TAG = args.tag
OUTPUT_DIR = f"{TAG}_CPDL"

# Create directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Use device variable consistently throughout
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Set random seeds for reproducibility
def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

# Set seeds
set_seeds(env.seed)

# Custom MLP Network for value function W
class ValueFunction(nn.Module):
    def __init__(self, input_dim=2, hidden_dims=[64, 64, 64, 32], output_dim=1):
        super(ValueFunction, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.Tanh())
            prev_dim = hidden_dim
            
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights with small values for better stability
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.5)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        return self.network(x)

# Custom sampling functions
def sample_H_exponential(n_samples, H_min, H_max):
    """Sample from truncated exponential distribution"""
    # Calculate normalization constants
    a = np.exp(-env.lambda_param_H * H_min)
    b = np.exp(-env.lambda_param_H * H_max)
    
    # Generate uniform samples
    u = np.random.uniform(0, 1, n_samples)
    
    # Transform to truncated exponential
    samples = -np.log(a - u * (a - b)) / env.lambda_param_H
    
    return samples.reshape(-1, 1)

def sample_eta_proportional(n_samples, eta_min, eta_max):
    """Sample from distribution proportional to 1/((eta+epsilon)*(1-eta+epsilon))"""
    # Create a grid for numerical approximation
    grid_size = 10000
    grid_step = (eta_max - eta_min) / grid_size
    assert 1.5/grid_size < eta_min and 1-1.5/grid_size > eta_max
    
    eta_grid = np.linspace(eta_min + 0.5 * grid_step, eta_max - 0.5 * grid_step, grid_size) + np.random.uniform(-0.5, 0.5) * grid_step
    
    # Calculate PDF values (unnormalized)
    pdf_values = 1 / ((eta_grid) * (1 - eta_grid) + env.epsilon_param_eta)
    
    # Normalize to create a proper PDF
    pdf_values = pdf_values / np.sum(pdf_values)
    
    # Sample using the PDF
    samples_idx = np.random.choice(grid_size, size=n_samples, p=pdf_values)
    samples = eta_grid[samples_idx]
    
    return samples.reshape(-1, 1)

# Pre-training function to match the initial guess
def pretrain_model(model):
    print("Starting pre-training to match initial guess...")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    batch_size = env.batch_size * 2  # Larger batch for pre-training
    eta_min = env.eta_min + 0.01  # Avoid boundary issues
    eta_max = env.eta_max - 0.01
    H_min = env.H_min
    H_max = env.H_max
    
    loss_history = {'epoch': [], 'loss': []}
    
    # Pre-training loop
    for epoch in tqdm(range(env.pretrain_epochs), desc="Pre-training"):
        # Sample state variables uniformly for better coverage during pre-training
        # Create tensors on CPU then move to device
        eta_np = np.random.uniform(eta_min, eta_max, (batch_size, 1))
        H_np = np.random.uniform(H_min, H_max, (batch_size, 1))
        
        eta = torch.tensor(eta_np, dtype=torch.float32, device=device)
        H = torch.tensor(H_np, dtype=torch.float32, device=device)
        
        # Create state variable tensor
        SV = torch.cat([eta, H], dim=1)
        
        # Forward pass
        W_predicted = model(SV)
        
        # Compute target values based on the initial guess
        # Use numpy-based environment function since we're working with numpy arrays here
        W_target_np = env.initial_guess(eta_np, H_np)
        W_target = torch.tensor(W_target_np, dtype=torch.float32, device=device).reshape(-1, 1)
        
        # Compute MSE loss
        loss = nn.MSELoss()(W_predicted, W_target)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        
        # Clip gradients to prevent instability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        
        optimizer.step()
        
        # Record loss every 10 epochs
        if epoch % 10 == 0:
            loss_history['epoch'].append(epoch)
            loss_history['loss'].append(loss.item())
    
    # Save pre-trained model
    torch.save({
        'model_state_dict': model.state_dict(),
        'loss': loss.item(),
    }, f"{OUTPUT_DIR}/pretrained_model_{TAG}.pt")
    
    # Save loss history
    pd.DataFrame(loss_history).to_csv(f"{OUTPUT_DIR}/pretrain_loss_{TAG}.csv", index=False)
    
    print(f"Pre-training completed! Final loss: {loss.item():.6f}")
    
    return loss_history

# Training function with Adam optimizer
def train_model(model):
    print("Starting training with Adam optimizer...")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Extract parameters from environment for cleaner code
    rho = env.rho
    beta = env.beta
    delta = env.delta
    phi = env.phi
    eps = env.eps
    mu_2 = env.mu_2
    sigma_H = env.sigma_H
    
    batch_size = env.batch_size
    eta_min = env.eta_min + 0.01  # Avoid boundary issues
    eta_max = env.eta_max - 0.01
    H_min = env.H_min
    H_max = env.H_max
    
    loss_history = {'epoch': [], 'loss': [], 'iteration': []}
    
    iteration = 0
    total_iterations = env.num_epochs * env.MINI_BATCH_PER_EPOCH  
    
    for epoch in range(env.num_epochs):
        print(f"Epoch {epoch+1}/{env.num_epochs}")
        
        # Run mini-batches per epoch for more granular updates
        for mini_batch in range(env.MINI_BATCH_PER_EPOCH):
            try:
                # Sample state variables using custom distributions
                eta_numpy = sample_eta_proportional(batch_size, eta_min, eta_max)
                H_numpy = sample_H_exponential(batch_size, H_min, H_max)
                
                # Create tensors with proper device
                eta = torch.tensor(eta_numpy, dtype=torch.float32, device=device)
                H = torch.tensor(H_numpy, dtype=torch.float32, device=device)
                
                # Create state variable tensor
                SV = torch.cat([eta, H], dim=1)
                SV.requires_grad_(True)
                
                # Forward pass - clear gradients
                optimizer.zero_grad()
                
                # First get W and its first derivatives
                W = model(SV)
                W_grad = torch.autograd.grad(
                    W, SV, grad_outputs=torch.ones_like(W), create_graph=True, retain_graph=True
                )[0]
                W_eta = W_grad[:, 0:1]  # First derivative w.r.t. eta
                W_H = W_grad[:, 1:2]  # First derivative w.r.t. H
    
                # Safer second derivative calculation
                try:
                    # Isolate H inputs for second derivative
                    H_only = SV[:, 1:2].detach().clone().requires_grad_(True)
                    eta_fixed = SV[:, 0:1].detach()
                    SV_H_only = torch.cat([eta_fixed, H_only], dim=1)
                    
                    # Recompute W with fixed eta and new H tensor
                    W_recomputed = model(SV_H_only)
                    
                    # Get first derivative with respect to H only, with proper shapes
                    ones_for_grad = torch.ones_like(W_recomputed)
                    W_H_recomputed = torch.autograd.grad(
                        W_recomputed, H_only, grad_outputs=ones_for_grad, 
                        create_graph=True, retain_graph=True
                    )[0]
                    
                    # Get second derivative with respect to H only
                    ones_for_second_grad = torch.ones_like(W_H_recomputed)
                    W_H_H = torch.autograd.grad(
                        W_H_recomputed, H_only, grad_outputs=ones_for_second_grad, 
                        create_graph=True
                    )[0]
                    
                    # Handle NaN values in second derivative
                    if torch.isnan(W_H_H).any():
                        print("WARNING: NaN values in W_H_H, replacing with zeros")
                        W_H_H = torch.where(torch.isnan(W_H_H), torch.zeros_like(W_H_H), W_H_H)
                    
                except Exception as e:
                    print(f"Error in second derivative calculation: {e}")
                    # Fallback to zeros if gradient calculation fails
                    W_H_H = torch.zeros_like(SV[:, 1:2], device=device)
    
                # Handle NaN values in first derivatives
                if torch.isnan(W_eta).any():
                    print("WARNING: NaN values in W_eta, replacing with zeros")
                    W_eta = torch.where(torch.isnan(W_eta), torch.zeros_like(W_eta), W_eta)
                
                if torch.isnan(W_H).any():
                    print("WARNING: NaN values in W_H, replacing with zeros")
                    W_H = torch.where(torch.isnan(W_H), torch.zeros_like(W_H), W_H)
                
                # Calculate all equations as per model specification
                # Define PyTorch-compatible versions of environment functions
                def A_k1_torch(H_tensor):
                    return env.A_1 * torch.exp(-env.psi_1 * H_tensor)
                
                def A_k2_torch(H_tensor):
                    return env.A_2 * torch.exp(-env.psi_2 * H_tensor)
                
                # Basic equations using environment-compatible functions
                A_k1 = A_k1_torch(H)
                A_k2 = A_k2_torch(H)
                
                # Calculate iota_1 with better numerical stability
                numerator = (-(phi + beta) + 
                            (W_eta * (1-eta) + beta) * 
                            (1 + phi * (eta * A_k1 + (1-eta) * A_k2)))
                denominator = (phi * (phi + beta))
                iota_1 = numerator / denominator
                
                # Calculate iota_2
                term1 = iota_1 * (-eta/(1-eta))
                term2 = (-1 + beta * eta * A_k1 + beta * (1-eta) * A_k2) / ((1-eta) * (phi + beta))
                iota_2 = term1 + term2
                
                # Define PyTorch-compatible version of environment Phi function
                def Phi_torch(iota_tensor, phi_val=env.phi):
                    return (1/phi_val) * torch.log(torch.clamp(1 + phi_val * iota_tensor, min=env.DL_LOG_ARG_CLIP))
                
                # Calculate Phi terms using environment-compatible function
                Phi_1 = Phi_torch(iota_1)
                Phi_2 = Phi_torch(iota_2)
                
                # Calculate mu_H
                mu_H = mu_2 * (1-eta) - eps * H
                
                # Calculate consumption with better clamping
                consumption = (A_k1 - iota_1) * eta + (A_k2 - iota_2) * (1-eta)
                consumption = torch.clamp(consumption, min=1e-6)
                
                # Calculate HJB residual components
                utility = torch.log(consumption)
                adjustment = (eta * Phi_1 + (1-eta) * Phi_2 - delta) / rho
                drift_eta = eta * (1-eta) * (Phi_1 - Phi_2) * W_eta
                drift_H = mu_H * W_H
                diffusion_H = (H * sigma_H) ** 2 * W_H_H / 2
                discount = rho * W
                
                # Full HJB residual
                residual = utility + adjustment + drift_eta + drift_H + diffusion_H - discount
                
                # Shape constraint
                shape_residual = torch.where(W_H > 0, W_H, torch.zeros_like(W_H))
                
                # Compute loss as mean squared error of residual
                loss = torch.mean(residual**2) + env.HJBE_SHAPE_WEIGHT * torch.mean(shape_residual ** 2)
                
                # Handle NaN in loss
                if torch.isnan(loss).any() or torch.isinf(loss).any():
                    print("WARNING: NaN or Inf in loss, skipping this iteration")
                    continue
                    
                # Backward pass
                loss.backward()
                
                # Clip gradients to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                
                # Take optimization step
                optimizer.step()
                
                # Record loss
                iteration += 1
                loss_history['epoch'].append(epoch)
                loss_history['iteration'].append(iteration)
                loss_history['loss'].append(loss.item())
                
                # Print progress for every 5th mini-batch
                if (mini_batch + 1) % 5 == 0:
                    print(f"  Mini-batch {mini_batch+1}/{env.MINI_BATCH_PER_EPOCH}, Loss: {loss.item():.6f}")
                
            except Exception as e:
                print(f"Error in training iteration: {e}")
                import traceback
                traceback.print_exc()
                continue  # Skip to next mini-batch on error
            
        # Save intermediate model after each epoch
        if (epoch+1) % env.save_every == 0:
            try:
                torch.save({
                    'epoch': epoch+1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss.item(),
                }, f"{OUTPUT_DIR}/model_{TAG}_epoch_{epoch+1}.pt")
            except Exception as e:
                print(f"Error saving model: {e}")
        
    # Save final model
    try:
        torch.save({
            'model_state_dict': model.state_dict(),
            'loss': loss.item(),
        }, f"{OUTPUT_DIR}/model_{TAG}_final.pt")
        
        # Save loss history
        pd.DataFrame(loss_history).to_csv(f"{OUTPUT_DIR}/model_{TAG}_loss.csv", index=False)
    except Exception as e:
        print(f"Error saving final outputs: {e}")
    
    print("Training completed!")
    return loss_history

def evaluate_model_on_grid(model):
    """Evaluate the trained model on the FD grid for direct comparison"""
    try:
        print("Evaluating trained model on finite difference grid...")
        
        # Create grid matching FD grid
        eta_grid = np.linspace(env.eta_min, env.eta_max, env.N_eta)
        H_grid = np.linspace(env.H_min, env.H_max, env.N_H)
        eta_mesh, H_mesh = np.meshgrid(eta_grid, H_grid, indexing="ij")
        
        # Flatten for batch processing
        eta_flat = eta_mesh.flatten()
        H_flat = H_mesh.flatten()
        grid_points = np.column_stack([eta_flat, H_flat])
        
        # Process in batches to prevent memory issues
        batch_size = 1000
        total_points = grid_points.shape[0]
        W_values = np.zeros(total_points)
        W_eta_values = np.zeros(total_points)
        W_H_values = np.zeros(total_points)
        
        for i in range(0, total_points, batch_size):
            end_idx = min(i + batch_size, total_points)
            batch_points = grid_points[i:end_idx]
            
            # Convert to torch tensors
            batch_tensor = torch.tensor(batch_points, dtype=torch.float32, device=device)
            batch_tensor.requires_grad_(True)
            
            # Get value function
            with torch.enable_grad():
                W_batch = model(batch_tensor)
                
                # Get gradients
                W_grad = torch.autograd.grad(
                    W_batch, batch_tensor, 
                    grad_outputs=torch.ones_like(W_batch),
                    create_graph=False
                )[0]
                
            # Store values
            W_values[i:end_idx] = W_batch.detach().cpu().numpy().flatten()
            W_eta_values[i:end_idx] = W_grad[:, 0].detach().cpu().numpy()
            W_H_values[i:end_idx] = W_grad[:, 1].detach().cpu().numpy()
            
        # Reshape to match grid
        W_values_grid = W_values.reshape(env.N_eta, env.N_H)
        W_eta_grid = W_eta_values.reshape(env.N_eta, env.N_H)
        W_H_grid = W_H_values.reshape(env.N_eta, env.N_H)
        
        # Save results for later comparison
        np.savez(
            f"{OUTPUT_DIR}/grid_evaluation_{TAG}.npz",
            eta_mesh=eta_mesh,
            H_mesh=H_mesh,
            W_values=W_values_grid,
            W_eta=W_eta_grid,
            W_H=W_H_grid
        )
        
        print(f"Grid evaluation saved to {OUTPUT_DIR}/grid_evaluation_{TAG}.npz")
        
    except Exception as e:
        print(f"Error in evaluate_model_on_grid: {e}")
        import traceback
        traceback.print_exc()

def generate_distribution_samples():
    """Generate and save distribution samples for visualization"""
    # Generate distribution samples for plotting
    n_samples = 10000
    eta_samples = sample_eta_proportional(n_samples, 
                                          env.eta_min + 0.01, 
                                          env.eta_max - 0.01)
    H_samples = sample_H_exponential(n_samples, env.H_min, env.H_max)
    
    # Save distribution samples
    np.savez(
        f"{OUTPUT_DIR}/distribution_samples_{TAG}.npz",
        eta_samples=eta_samples,
        H_samples=H_samples,
        lambda_param_H=env.lambda_param_H,
        epsilon_param_eta=env.epsilon_param_eta,
        eta_min=env.eta_min,
        eta_max=env.eta_max,
        H_min=env.H_min,
        H_max=env.H_max
    )
    
    print(f"Distribution samples saved to {OUTPUT_DIR}/distribution_samples_{TAG}.npz")

# Helper functions for simulations

def F(x, eta_n, iota_1n, iota_2n, dt):
    """Helper function for eta update using Newton's method"""
    return x * (1 - x) * (env.Phi(iota_1n) - env.Phi(iota_2n)) * dt - x + eta_n

def F_x(x, iota_1n, iota_2n, dt):
    """Derivative of F with respect to x"""
    return (1 - 2 * x) * (env.Phi(iota_1n) - env.Phi(iota_2n)) * dt - 1

def eta_next(eta_n, iota_1, iota_2, dt):
    """Calculate next eta value using Newton's method"""
    return eta_n - F(eta_n, eta_n, iota_1, iota_2, dt) / F_x(eta_n, iota_1, iota_2, dt)

def H_next(H_n, eta_n, dt, dW, eps, mu_2, sigma_H):
    """Calculate next H value"""
    return (H_n + mu_2 * (1 - eta_n) * dt) / (1 + eps * dt + sigma_H * dW)

def K_next(K, eta, iota_1, iota_2, dt, delta):
    """Calculate next K value"""
    dK = dt * K * (eta * env.Phi(iota_1) + (1 - eta) * env.Phi(iota_2) - delta)
    return K + dK

def propagate_economy_dl(model, eta_0, H_0, K_0, dt, N_T, model_params=None):
    """
    Propagate economy using deep learning model solution
    
    Parameters:
    -----------
    model : nn.Module
        Trained neural network model for value function
    eta_0, H_0, K_0 : float
        Initial state variables
    dt : float
        Time step
    N_T : int
        Number of time steps
    model_params : dict, optional
        Model parameters (defaults provided if None)
        
    Returns:
    --------
    tuple
        Time series for state variables and controls
    """
    # Default parameters if not provided
    if model_params is None:
        model_params = {
            'eps': env.eps,
            'mu_2': env.mu_2,
            'sigma_H': env.sigma_H,
            'delta': env.delta,
            'phi': env.phi,
            'beta': env.beta
        }
    
    # Unpack parameters
    eps = model_params.get('eps', env.eps)
    mu_2 = model_params.get('mu_2', env.mu_2)
    sigma_H = model_params.get('sigma_H', env.sigma_H)
    delta = model_params.get('delta', env.delta)
    phi = model_params.get('phi', env.phi)
    beta = model_params.get('beta', env.beta)
    
    # Initialize arrays for simulation results
    eta_sim = np.zeros(N_T + 1)
    iota_1_sim = np.zeros(N_T + 1)
    iota_2_sim = np.zeros(N_T + 1)
    H_sim = np.zeros(N_T + 1)
    K_sim = np.zeros(N_T + 1)
    c_sim = np.zeros(N_T + 1)
    
    # Set initial conditions
    eta_sim[0] = eta_0
    H_sim[0] = H_0
    K_sim[0] = K_0
    
    # Generate all random shocks at once
    dW_values = np.random.normal(loc=0, scale=np.sqrt(dt), size=N_T)
    
    # Main simulation loop
    for t in range(N_T):
        # Current state
        eta_t = eta_sim[t]
        H_t = H_sim[t]
        K_t = K_sim[t]
        
        # Get state as tensor for model
        SV = torch.tensor([[eta_t, H_t]], dtype=torch.float32, device=device)
        SV.requires_grad_(True)
        
        # Get value function and derivative
        W = model(SV)
        W_grad = torch.autograd.grad(
            W, SV, grad_outputs=torch.ones_like(W), create_graph=False
        )[0]
        W_eta = W_grad[:, 0].item()
        
        # Calculate optimal controls (iota_1, iota_2)
        if np.isclose(eta_t, 0, atol=1e-6):
            iota_1 = (beta * env.A_k1(H_t) - 1) / (phi + beta)
            iota_2 = (beta * env.A_k2(H_t) - 1) / (phi + beta)
        elif np.isclose(eta_t, 1, atol=1e-6):
            iota_1 = (beta * env.A_k1(H_t) - 1) / (phi + beta)
            iota_2 = 0  # Arbitrary as it's not used
        else:
            # Standard case
            numerator = -(phi + beta) + (W_eta * (1-eta_t) + beta) * (1 + phi * (eta_t * env.A_k1(H_t) + (1-eta_t) * env.A_k2(H_t)))
            denominator = phi * (phi + beta)
            iota_1 = numerator / denominator
            
            C_1 = -eta_t / (1 - eta_t)
            C_0 = (-1 + beta * (eta_t * env.A_k1(H_t) + (1-eta_t) * env.A_k2(H_t))) / ((1-eta_t) * (phi + beta))
            iota_2 = iota_1 * C_1 + C_0
        
        # Store controls
        iota_1_sim[t] = iota_1
        iota_2_sim[t] = iota_2
        
        # Calculate consumption
        c_sim[t] = K_t * ((env.A_k1(H_t) - iota_1) * eta_t + (env.A_k2(H_t) - iota_2) * (1 - eta_t))
        
        # Update state variables using the helper functions
        eta_sim[t+1] = eta_next(eta_t, iota_1, iota_2, dt)
        eta_sim[t+1] = np.clip(eta_sim[t+1], 0, 1)  # Ensure bounds are respected
        
        H_sim[t+1] = H_next(H_t, eta_t, dt, dW_values[t], eps, mu_2, sigma_H)
        H_sim[t+1] = max(H_sim[t+1], 0)  # Ensure H remains non-negative
        
        K_sim[t+1] = K_next(K_t, eta_t, iota_1, iota_2, dt, delta)
    
    # Calculate final step controls/consumption
    eta_t = eta_sim[-1]
    H_t = H_sim[-1]
    K_t = K_sim[-1]
    
    # Get state as tensor for model
    SV = torch.tensor([[eta_t, H_t]], dtype=torch.float32, device=device)
    SV.requires_grad_(True)
    
    # Get value function and derivative
    W = model(SV)
    W_grad = torch.autograd.grad(
        W, SV, grad_outputs=torch.ones_like(W), create_graph=False
    )[0]
    W_eta = W_grad[:, 0].item()
    
    # Calculate optimal controls (iota_1, iota_2)
    if np.isclose(eta_t, 0, atol=1e-6):
        iota_1 = (beta * env.A_k1(H_t) - 1) / (phi + beta)
        iota_2 = (beta * env.A_k2(H_t) - 1) / (phi + beta)
    elif np.isclose(eta_t, 1, atol=1e-6):
        iota_1 = (beta * env.A_k1(H_t) - 1) / (phi + beta)
        iota_2 = 0  # Arbitrary as it's not used
    else:
        # Standard case
        numerator = -(phi + beta) + (W_eta * (1-eta_t) + beta) * (1 + phi * (eta_t * env.A_k1(H_t) + (1-eta_t) * env.A_k2(H_t)))
        denominator = phi * (phi + beta)
        iota_1 = numerator / denominator
        
        C_1 = -eta_t / (1 - eta_t)
        C_0 = (-1 + beta * (eta_t * env.A_k1(H_t) + (1-eta_t) * env.A_k2(H_t))) / ((1-eta_t) * (phi + beta))
        iota_2 = iota_1 * C_1 + C_0
    
    iota_1_sim[-1] = iota_1
    iota_2_sim[-1] = iota_2
    c_sim[-1] = K_t * ((env.A_k1(H_t) - iota_1) * eta_t + (env.A_k2(H_t) - iota_2) * (1 - eta_t))
    
    return (eta_sim, H_sim, K_sim, c_sim, iota_1_sim, iota_2_sim)

def run_monte_carlo_simulations_dl(model, eta_0, H_0, K_0, dt, N_T, num_sims=10, 
                                 model_params=None, show_progress=True):
    """
    Run multiple simulations with deep learning model
    
    Parameters:
    -----------
    model : nn.Module
        Trained neural network model for value function
    eta_0, H_0, K_0 : float
        Initial state variables
    dt : float
        Time step
    N_T : int
        Number of time steps
    num_sims : int
        Number of simulations to run
    model_params : dict, optional
        Model parameters
    show_progress : bool
        Whether to show progress updates
        
    Returns:
    --------
    dict
        Dictionary containing all simulation results
    """
    # Arrays to store results from all simulations
    eta_results = np.zeros((num_sims, N_T + 1))
    H_results = np.zeros((num_sims, N_T + 1))
    K_results = np.zeros((num_sims, N_T + 1))
    c_results = np.zeros((num_sims, N_T + 1))
    iota_1_results = np.zeros((num_sims, N_T + 1))
    iota_2_results = np.zeros((num_sims, N_T + 1))
    
    # Run multiple simulations
    start_time = time.time()
    update_interval = max(1, num_sims // 10)  # Update progress at most 10 times
    
    for i in range(num_sims):
        # Run a single simulation
        eta_sim, H_sim, K_sim, c_sim, iota_1_sim, iota_2_sim = propagate_economy_dl(
            model, eta_0, H_0, K_0, dt, N_T, model_params
        )
        
        # Store results
        eta_results[i, :] = eta_sim
        H_results[i, :] = H_sim
        K_results[i, :] = K_sim
        c_results[i, :] = c_sim
        iota_1_results[i, :] = iota_1_sim
        iota_2_results[i, :] = iota_2_sim
        
        # Show progress
        if show_progress and (i + 1) % update_interval == 0:
            elapsed = time.time() - start_time
            progress = (i + 1) / num_sims
            eta_remaining = elapsed / progress - elapsed
            print(f"Progress: {progress:.1%} ({i+1}/{num_sims}), " 
                  f"Elapsed: {elapsed:.1f}s, ETA: {eta_remaining:.1f}s")
    
    # Calculate additional statistics
    total_time = time.time() - start_time
    print(f"Completed {num_sims} simulations in {total_time:.2f} seconds")
    print(f"Average time per simulation: {total_time/num_sims:.4f} seconds")
    
    # Return all results in a dictionary
    return {
        'eta': eta_results,
        'H': H_results,
        'K': K_results,
        'c': c_results,
        'iota_1': iota_1_results,
        'iota_2': iota_2_results,
        'time_taken': total_time,
        'dt': dt,
        'N_T': N_T
    }

def compute_value_function_slices(model):
    """Precompute value function slices for visualization"""
    print("Computing value function slices for visualization...")
    
    with torch.no_grad():
        # Create grids for evaluation
        eta_grid = np.linspace(env.eta_min, env.eta_max, 100)
        H_grid = np.linspace(env.H_min, min(env.H_max, 10), 100)  # Cap at 10 for better visualization
        
        # Compute W(eta, H)
        eta_mesh, H_mesh = np.meshgrid(eta_grid, H_grid, indexing='ij')
        W_eta_H = np.zeros_like(eta_mesh)
        
        # Process in batches
        batch_size = 1000
        points = []
        
        for i in range(len(eta_grid)):
            for j in range(len(H_grid)):
                points.append((eta_grid[i], H_grid[j]))
                
        for i in range(0, len(points), batch_size):
            batch_points = points[i:i+batch_size]
            batch_tensor = torch.tensor(batch_points, dtype=torch.float32, device=device)
            batch_W = model(batch_tensor).cpu().numpy().flatten()
            
            for k, (batch_i, batch_j) in enumerate(batch_points):
                idx_i = np.where(eta_grid == batch_i)[0][0]
                idx_j = np.where(H_grid == batch_j)[0][0]
                W_eta_H[idx_i, idx_j] = batch_W[k]
    
    # Save results for visualization
    np.savez(
        f"{OUTPUT_DIR}/value_function_slices_{TAG}.npz",
        eta_grid=eta_grid,
        H_grid=H_grid,
        W_eta_H=W_eta_H
    )
    
    print(f"Value function slices saved to {OUTPUT_DIR}/value_function_slices_{TAG}.npz")

def simulate_economy(model):
    """Run Monte Carlo simulations of the economy and save results"""
    try:
        print("Starting economy simulations...")
        
        # Set model to evaluation mode
        model.eval()  # Set model to evaluation mode
        
        print("Model loaded successfully. Running simulations...")
        
        # Run simulations with parameters from environment
        results = run_monte_carlo_simulations_dl(
            model=model,
            eta_0=env.sim_eta_0,
            H_0=env.sim_H_0,
            K_0=env.sim_K_0,
            dt=env.sim_dt,
            N_T=env.sim_N_T,
            num_sims=env.sim_num_sims
        )
        
        # Save simulation results
        np.savez(
            f"{OUTPUT_DIR}/simulation_results_{TAG}.npz",
            eta=results['eta'],
            H=results['H'],
            K=results['K'],
            c=results['c'],
            iota_1=results['iota_1'],
            iota_2=results['iota_2'],
            dt=results['dt'],
            N_T=results['N_T']
        )
        
        print(f"Simulations saved to {OUTPUT_DIR}/simulation_results_{TAG}.npz")
        
    except Exception as e:
        print(f"Error in simulate_economy: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    try:
        # Initialize model and move to device
        model = ValueFunction().to(device)
        print(f"Model architecture:\n{model}")
    
        # Start training workflow
        print(f"Starting DL solution workflow for tag {TAG}...")
        
        # Pre-train to match initial guess
        pretrain_model(model)
        
        # Main training
        train_model(model)
        
        # Evaluate model on FD grid for comparison
        evaluate_model_on_grid(model)
        
        # Generate distribution samples for visualization
        generate_distribution_samples()
        
        # Compute value function slices for visualization
        compute_value_function_slices(model)
        
        # Run economy simulations
        simulate_economy(model)
    
        print("Training, evaluation, and simulation completed!")
    except Exception as e:
        print(f"Critical error in main execution: {e}")
        import traceback
        traceback.print_exc()
