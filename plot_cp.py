import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
import environment as env
from cp_fd import build_grid
import argparse
import seaborn as sns

# Set up command line argument parsing
parser = argparse.ArgumentParser(description='Plot central planner model results')
parser.add_argument('--tag', type=str, required=True,
                    help='Custom tag for files to plot')
args = parser.parse_args()

TAG = args.tag

# Determine model type based on looking for directories
if os.path.exists(f"{TAG}_CPDL"):
    MODEL_TYPE = "CPDL"
    OUTPUT_DIR = f"{TAG}_CPDL"
elif os.path.exists(f"{TAG}_CPFD"):
    MODEL_TYPE = "CPFD"
    OUTPUT_DIR = f"{TAG}_CPFD"
else:
    raise ValueError(f"No model directory found for tag {TAG}")

# Create plot directory
PLOT_DIR = f"{TAG}_plots"
os.makedirs(PLOT_DIR, exist_ok=True)

# Set Seaborn style
sns.set_theme(style="whitegrid")
sns.set_context("paper", font_scale=1.2)

def plot_initial_guess():
    eta, H, d_eta, dH, dt, eta_grid, H_grid = build_grid(env.eta_min, env.eta_max, env.N_eta, env.H_min, env.H_max, env.N_H)
    contour = plt.contourf(eta, H, env.initial_guess(eta, H), 50, cmap='viridis')
    plt.colorbar(contour, label='Value Function W')
    plt.xlabel('η')
    plt.ylabel('H')
    plt.title('Initial Guess of Value Function W(η, H)')
    plt.savefig(f"{PLOT_DIR}/initial_guess_{TAG}.png")
    plt.close()
    
def plot_pretrain_loss():
    """Plot the pretraining loss history"""
    try:
        loss_file = f"{OUTPUT_DIR}/pretrain_loss_{TAG}.csv"
        if not os.path.exists(loss_file):
            print(f"Error: File {loss_file} not found")
            return
            
        loss_history = pd.read_csv(loss_file)
        plt.figure(figsize=(10, 6))
        plt.plot(loss_history['epoch'], loss_history['loss'], 'o-')
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        plt.title('Pre-training Loss')
        plt.grid(True)
        plt.yscale('log')
        plt.savefig(f"{PLOT_DIR}/pretrain_loss_{TAG}.png")
        plt.close()
        print(f"Pre-training loss plot saved to {PLOT_DIR}/pretrain_loss_{TAG}.png")
    except Exception as e:
        print(f"Error in plot_pretrain_loss: {e}")

def plot_training_loss():
    """Plot the main training loss history"""
    try:
        loss_file = f"{OUTPUT_DIR}/model_{TAG}_loss.csv"
        
        if not os.path.exists(loss_file):
            print(f"Error: File {loss_file} not found")
            return
            
        loss_history = pd.read_csv(loss_file)
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.plot(loss_history['iteration'], loss_history['loss'], 'o-')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.yscale('log')
        plt.title('Training Loss by Iteration')
        plt.grid(True)
    
        plt.subplot(1, 2, 2)
        # Calculate epoch average loss
        epoch_df = pd.DataFrame(loss_history)
        epoch_means = epoch_df.groupby('epoch')['loss'].mean().reset_index()
        plt.plot(epoch_means['epoch'], epoch_means['loss'], 'o-', color='red')
        plt.xlabel('Epoch')
        plt.ylabel('Average Loss')
        plt.yscale('log')
        plt.title('Average Loss by Epoch')
        plt.grid(True)
    
        plt.tight_layout()
        plt.savefig(f"{PLOT_DIR}/training_loss_{TAG}.png")
        plt.close()
        print(f"Training loss plot saved to {PLOT_DIR}/training_loss_{TAG}.png")
    except Exception as e:
        print(f"Error in plot_training_loss: {e}")

def plot_sampling_distributions():
    """Plot the sampling distributions used in training"""
    try:
        data_file = f"{OUTPUT_DIR}/distribution_samples_{TAG}.npz"
        if not os.path.exists(data_file):
            print(f"Error: File {data_file} not found")
            return
            
        # Load precomputed data
        data = np.load(data_file)
        eta_samples = data['eta_samples']
        H_samples = data['H_samples']
        lambda_param_H = data['lambda_param_H']
        epsilon_param_eta = data['epsilon_param_eta']
        eta_min = data['eta_min']
        eta_max = data['eta_max']
        H_min = data['H_min']
        H_max = data['H_max']
        
        # Plot sample distributions
        plt.figure(figsize=(15, 6))
        
        # Plot eta distribution
        plt.subplot(1, 2, 1)
        plt.hist(eta_samples, bins=50, density=True, alpha=0.7)
        
        # Plot theoretical pdf
        x = np.linspace(eta_min, eta_max, 1000)
        y = 1 / ((x) * (1 - x) + epsilon_param_eta)
        y = y / np.trapz(y, x)  # Normalize
        plt.plot(x, y, 'r-', linewidth=2)
        
        plt.xlabel('η')
        plt.ylabel('Density')
        plt.title('η Sampling Distribution')
        
        # Plot H distribution
        plt.subplot(1, 2, 2)
        plt.hist(H_samples, bins=50, density=True, alpha=0.7)
        
        # Plot theoretical pdf
        x = np.linspace(H_min, H_max, 1000)
        a = np.exp(-lambda_param_H * H_min)
        b = np.exp(-lambda_param_H * H_max)
        y = lambda_param_H * np.exp(-lambda_param_H * x) / (a - b)
        plt.plot(x, y, 'r-', linewidth=2)
        
        plt.xlabel('H')
        plt.ylabel('Density')
        plt.title('H Sampling Distribution (Truncated Exponential)')
        
        plt.tight_layout()
        plt.savefig(f"{PLOT_DIR}/sampling_distributions_{TAG}.png")
        plt.close()
        print(f"Sampling distribution plots saved to {PLOT_DIR}/sampling_distributions_{TAG}.png")
        
    except Exception as e:
        print(f"Error in plot_sampling_distributions: {e}")

def plot_fd_solution():
    """Plot the finite difference solution"""
    try:
        if MODEL_TYPE != "CPFD":
            return
            
        data_file = f"{OUTPUT_DIR}/CP_FD_results_{TAG}.npz"
        if not os.path.exists(data_file):
            print(f"Error: File {data_file} not found")
            return
            
        # Load precomputed data
        data = np.load(data_file)
        eta_mesh = data['eta_mesh']
        H_mesh = data['H_mesh']
        W_sol = data['W_sol']
        W_eta_sol = data['W_eta_sol']
        iota_1 = data['iota_1']
        iota_2 = data['iota_2']
        
        # Plot value function
        plt.figure(figsize=(15, 12))
        
        # Value function
        plt.subplot(2, 2, 1)
        contour = plt.contourf(eta_mesh, H_mesh, W_sol, 50, cmap='viridis')
        plt.colorbar(contour, label='Value Function W')
        plt.xlabel('η')
        plt.ylabel('H')
        plt.title('FD: Value Function W(η, H)')
        
        # Gradient w.r.t. eta
        plt.subplot(2, 2, 2)
        contour = plt.contourf(eta_mesh, H_mesh, W_eta_sol, 50, cmap='coolwarm')
        plt.colorbar(contour, label='∂W/∂η')
        plt.xlabel('η')
        plt.ylabel('H')
        plt.title('FD: Partial Derivative ∂W/∂η')
        
        # Investment rates
        plt.subplot(2, 2, 3)
        vmax = max(abs(iota_1.min()), abs(iota_1.max()))
        vmin = -vmax
        contour = plt.contourf(eta_mesh, H_mesh, iota_1, 50, vmin=vmin, vmax=vmax, cmap='bwr')
        plt.colorbar(contour, label='iota_1')
        plt.xlabel('η')
        plt.ylabel('H')
        plt.title('FD: Green Investment Rate (iota_1)')
        
        plt.subplot(2, 2, 4)
        vmax = max(abs(iota_2.min()), abs(iota_2.max()))
        vmin = -vmax
        contour = plt.contourf(eta_mesh, H_mesh, iota_2, 50, vmin=vmin, vmax=vmax, cmap='bwr')
        plt.colorbar(contour, label='iota_2')
        plt.xlabel('η')
        plt.ylabel('H')
        plt.title('FD: Brown Investment Rate (iota_2)')
        
        plt.tight_layout()
        plt.savefig(f"{PLOT_DIR}/fd_solution_{TAG}.png")
        plt.close()
        print(f"FD solution plot saved to {PLOT_DIR}/fd_solution_{TAG}.png")
        
    except Exception as e:
        print(f"Error in plot_fd_solution: {e}")

def plot_dl_solution():
    """Plot the deep learning solution"""
    try:
        if MODEL_TYPE != "CPDL":
            return
            
        data_file = f"{OUTPUT_DIR}/grid_evaluation_{TAG}.npz"
        if not os.path.exists(data_file):
            print(f"Error: File {data_file} not found")
            return
            
        # Load precomputed data
        data = np.load(data_file)
        eta_mesh = data['eta_mesh']
        H_mesh = data['H_mesh']
        W_values = data['W_values']
        W_eta = data['W_eta']
        W_H = data['W_H']
        
        # Compute investment rates
        iota_1, iota_2 = compute_investment_rates(W_eta, eta_mesh, H_mesh)
        
        # Plot value function
        plt.figure(figsize=(15, 12))
        
        # Value function
        plt.subplot(2, 2, 1)
        contour = plt.contourf(eta_mesh, H_mesh, W_values, 50, cmap='viridis')
        plt.colorbar(contour, label='Value Function W')
        plt.xlabel('η')
        plt.ylabel('H')
        plt.title('DL: Value Function W(η, H)')
        
        # Gradient w.r.t. eta
        plt.subplot(2, 2, 2)
        contour = plt.contourf(eta_mesh, H_mesh, W_eta, 50, cmap='coolwarm')
        plt.colorbar(contour, label='∂W/∂η')
        plt.xlabel('η')
        plt.ylabel('H')
        plt.title('DL: Partial Derivative ∂W/∂η')
        
        # Investment rates
        plt.subplot(2, 2, 3)
        contour = plt.contourf(eta_mesh, H_mesh, iota_1, 50, cmap='viridis')
        plt.colorbar(contour, label='iota_1')
        plt.xlabel('η')
        plt.ylabel('H')
        plt.title('DL: Green Investment Rate (iota_1)')
        
        plt.subplot(2, 2, 4)
        contour = plt.contourf(eta_mesh, H_mesh, iota_2, 50, cmap='viridis')
        plt.colorbar(contour, label='iota_2')
        plt.xlabel('η')
        plt.ylabel('H')
        plt.title('DL: Brown Investment Rate (iota_2)')
        
        plt.tight_layout()
        plt.savefig(f"{PLOT_DIR}/dl_solution_{TAG}.png")
        plt.close()
        print(f"DL solution plot saved to {PLOT_DIR}/dl_solution_{TAG}.png")
        
    except Exception as e:
        print(f"Error in plot_dl_solution: {e}")

def compute_investment_rates(W_eta, eta, H):
    """Compute investment rates from value function derivative"""
    # Initialize arrays
    iota_1 = np.zeros_like(eta)
    iota_2 = np.zeros_like(eta)
    
    # Compute iota_1 for most cells (excluding boundaries)
    mask = (eta > 0.01) & (eta < 0.99)
    
    # Using the formulas from the FD solution
    A_k1 = env.A_1 * np.exp(-env.psi_1 * H)
    A_k2 = env.A_2 * np.exp(-env.psi_2 * H)
    
    # Compute iota_1
    num = -(env.phi + env.beta) + (W_eta * (1 - eta) + env.beta) * (1 + env.phi * (eta * A_k1 + (1 - eta) * A_k2))
    denom = env.phi * (env.phi + env.beta)
    iota_1 = np.where(mask, num / denom, 0)
    
    # Compute iota_2
    C_1 = -eta / (1 - eta + 1e-10)  # Avoid division by zero
    C_0 = (-1 + env.beta * (eta * A_k1 + (1 - eta) * A_k2)) / ((1 - eta + 1e-10) * (env.phi + env.beta))
    iota_2 = np.where(mask, iota_1 * C_1 + C_0, 0)
    
    # Clean up potential anomalies
    iota_1 = np.clip(iota_1, -10, 10)
    iota_2 = np.clip(iota_2, -10, 10)
    
    # Set boundary values
    # eta = 0 boundary
    iota_1[eta <= 0.01] = 0
    iota_2[eta <= 0.01] = (env.beta * A_k2[eta <= 0.01] - 1) / (env.phi + env.beta)
    
    # eta = 1 boundary
    iota_1[eta >= 0.99] = (env.beta * A_k1[eta >= 0.99] - 1) / (env.phi + env.beta)
    iota_2[eta >= 0.99] = 0
    
    return iota_1, iota_2

def create_fan_charts(eta_results, H_results, K_results, c_results, iota_1_results, iota_2_results, dt, N_T,
                    percentiles=None, figsize=(15, 10)):
    """
    Create fan charts for all state variables and controls
    
    Parameters:
    -----------
    *_results : 2D numpy arrays
        Arrays containing simulation results
    dt : float
        Time step
    N_T : int
        Number of time steps
    percentiles : list, optional
        Percentiles to show in fan charts
    figsize : tuple
        Figure size
    """
    # Default percentiles if not provided
    if percentiles is None:
        percentiles = env.sim_percentiles
    
    # Create time array
    time = np.arange(0, (N_T + 1) * dt, dt)
    
    # Setup plot
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    fig.suptitle(f'Central Planner Model Simulation Results - Tag {TAG} (T={N_T*dt:.1f}, dt={dt:.3f})', fontsize=16)
    
    # Flatten axes for easier indexing
    axes = axes.flatten()
    
    # Data to plot
    all_data = [
        (eta_results, 'Green Capital Share η', 0),
        (H_results, 'Climate Damage H', 1),
        (K_results, 'Aggregate Capital K', 2),
        (c_results, 'Consumption c', 3),
        (iota_1_results, 'Green Investment Rate ι₁', 4),
        (iota_2_results, 'Brown Investment Rate ι₂', 5)
    ]
    
    # Create fan charts for each variable
    for data, title, idx in all_data:
        ax = axes[idx]
        
        # Calculate percentiles for fan chart
        p_data = np.percentile(data, percentiles, axis=0)
        
        # Create colormap for fan chart
        n_bands = len(percentiles) // 2
        colors = plt.cm.Blues(np.linspace(0.4, 0.8, n_bands))
        
        # Plot median
        ax.plot(time, p_data[len(percentiles)//2], 'r-', linewidth=2, label='Median')
        
        # Plot fan chart bands
        for i in range(n_bands):
            upper_idx = len(percentiles) - i - 1
            lower_idx = i
            ax.fill_between(time, p_data[lower_idx], p_data[upper_idx], color=colors[i], alpha=0.7)
        
        # Add mean across simulations
        mean_data = np.mean(data, axis=0)
        ax.plot(time, mean_data, 'k--', linewidth=1.5, label='Mean')
        
        # Set labels and title
        ax.set_xlabel('Time')
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Add legend to the first plot only
        if idx == 0:
            ax.legend()
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    
    # Save
    plt.savefig(f"{PLOT_DIR}/simulation_fan_charts_{TAG}.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Simulation fan charts saved to {PLOT_DIR}/simulation_fan_charts_{TAG}.png")

def plot_simulations():
    """Plot simulation results with fan charts"""
    try:
        # Look for simulation results file in either CPDL or CPFD directory
        sim_file = f"{OUTPUT_DIR}/simulation_results_{TAG}.npz"
        if not os.path.exists(sim_file):
            print(f"Error: Simulation file {sim_file} not found")
            return
            
        # Load simulation results
        results = np.load(sim_file)
        
        eta_results = results['eta']
        H_results = results['H']
        K_results = results['K']
        c_results = results['c']
        iota_1_results = results['iota_1']
        iota_2_results = results['iota_2']
        dt = float(results['dt'])
        N_T = int(results['N_T'])
        
        # Create fan charts using the function from plot_solo.py
        create_fan_charts(
            eta_results, H_results, K_results, c_results, 
            iota_1_results, iota_2_results, dt, N_T
        )
        
    except Exception as e:
        print(f"Error in plot_simulations: {e}")
        import traceback
        traceback.print_exc()

def plot_value_function_slices():
    """Plot value function slices - similar to what's in plot_solo.py"""
    try:
        if MODEL_TYPE == "CPDL":
            # For CPDL model, check for value function slices
            slice_file = f"{OUTPUT_DIR}/value_function_slices_{TAG}.npz"
            
            if os.path.exists(slice_file):
                # Load data
                data = np.load(slice_file)
                eta_grid = data['eta_grid']
                H_grid = data['H_grid']
                W_eta_H = data['W_eta_H']
                
                # Plot W(eta, H)
                plt.figure(figsize=(10, 8))
                eta_mesh, H_mesh = np.meshgrid(eta_grid, H_grid, indexing='ij')
                
                contour = plt.contourf(eta_mesh, H_mesh, W_eta_H, 50, cmap='viridis')
                plt.colorbar(contour, label='Value Function W')
                plt.xlabel('Green Capital Share η')
                plt.ylabel('Climate Damage H')
                plt.title(f'Value Function W(η, H)')
                plt.grid(True, linestyle='--', alpha=0.3)
                
                plt.savefig(f"{PLOT_DIR}/W_eta_H_slice_{TAG}.png", dpi=300)
                plt.close()
                
                print(f"Value function slice plot saved to {PLOT_DIR}/W_eta_H_slice_{TAG}.png")
                
        elif MODEL_TYPE == "CPFD":
            # For CPFD model, use the main solution file
            results_file = f"{OUTPUT_DIR}/CP_FD_results_{TAG}.npz"
            
            if os.path.exists(results_file):
                # Load data
                data = np.load(results_file)
                eta_mesh = data['eta_mesh']
                H_mesh = data['H_mesh']
                W_sol = data['W_sol']
                
                # Plot W(eta, H)
                plt.figure(figsize=(10, 8))
                contour = plt.contourf(eta_mesh, H_mesh, W_sol, 50, cmap='viridis')
                plt.colorbar(contour, label='Value Function W')
                plt.xlabel('Green Capital Share η')
                plt.ylabel('Climate Damage H')
                plt.title(f'Value Function W(η, H)')
                plt.grid(True, linestyle='--', alpha=0.3)
                
                plt.savefig(f"{PLOT_DIR}/W_eta_H_slice_{TAG}.png", dpi=300)
                plt.close()
                
                print(f"Value function slice plot saved to {PLOT_DIR}/W_eta_H_slice_{TAG}.png")
                
    except Exception as e:
        print(f"Error in plot_value_function_slices: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    try:
        print(f"Starting plotting process for tag {TAG}...")
        
        # Plot initial guess
        plot_initial_guess()
        
        # Plot training diagnostics (applicable to CPDL only)
        if MODEL_TYPE == "CPDL":
            plot_pretrain_loss()
            plot_training_loss()
            plot_sampling_distributions()
        
        # Plot individual solutions
        plot_fd_solution()  # For CPFD
        plot_dl_solution()  # For CPDL
        
        # Plot value function slices (similar to plot_solo.py)
        plot_value_function_slices()
        
        # Plot simulation results (for both CPDL and CPFD)
        plot_simulations()
        
        print("All plots generated successfully!")
    except Exception as e:
        print(f"Critical error in main execution: {e}")
        import traceback
        traceback.print_exc()