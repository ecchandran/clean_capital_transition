import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
import seaborn as sns
import environment as env
import argparse
import sys
import torch

# Set Seaborn style
sns.set_theme(style="whitegrid")
sns.set_context("paper", font_scale=1.2)

# Set up command line argument parsing
parser = argparse.ArgumentParser(description='Plot model results')
parser.add_argument('--tag', type=str, required=True,
                    help='Custom tag for files to plot')
args = parser.parse_args()

TAG = args.tag

# Determine model type based on looking for directories
if os.path.exists(f"{TAG}_M3DL"):
    MODEL_TYPE = "M3DL"
    OUTPUT_DIR = f"{TAG}_M3DL"
elif os.path.exists(f"{TAG}_CPDL"):
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

print(f"Plotting model with tag {TAG} (type: {MODEL_TYPE})")

# Helper function to fix subscript labels
def fix_subscript_labels(ax):
    """Replace subscript numerals with regular numbers to avoid font issues"""
    ylabel = ax.get_ylabel()
    ylabel = ylabel.replace('ι₁', 'ι1')
    ylabel = ylabel.replace('ι₂', 'ι2')
    ax.set_ylabel(ylabel)
    
    title = ax.get_title()
    title = title.replace('ι₁', 'ι1')
    title = title.replace('ι₂', 'ι2')
    ax.set_title(title)

def plot_training_loss():
    """Plot model training loss"""
    try:
        if MODEL_TYPE == "M3DL":
            # For M3 model, need to plot equilibrium and value function losses
            # First, check equilibrium loss
            loss_file = f"{OUTPUT_DIR}/equilibrium_model_{TAG}_loss.csv"
            if os.path.exists(loss_file):
                loss_history = pd.read_csv(loss_file)
                
                plt.figure(figsize=(12, 4))
                
                # Total loss (log scale)
                plt.subplot(1, 2, 1)
                plt.semilogy(loss_history['epoch'], loss_history['total_loss'], 'b-')
                plt.xlabel('Epoch')
                plt.ylabel('Loss (log scale)')
                plt.title('Total Loss')
                plt.grid(True)
                
                # Component losses (log scale)
                plt.subplot(1, 2, 2)
                plt.semilogy(loss_history['epoch'], loss_history['consistency_loss'], 'r-', label='Consistency')
                plt.semilogy(loss_history['epoch'], loss_history['clearing_loss'], 'g-', label='Clearing')
                plt.semilogy(loss_history['epoch'], loss_history['boundary_loss'], 'm-', label='Boundary')
                plt.semilogy(loss_history['epoch'], loss_history['goods_market_loss'], 'y-', label='Goods Market')
                plt.xlabel('Epoch')
                plt.ylabel('Loss (log scale)')
                plt.title('Component Losses')
                plt.legend()
                plt.grid(True)
                
                # # Total loss (linear scale)
                # plt.subplot(2, 2, 3)
                # plt.plot(loss_history['epoch'], loss_history['total_loss'], 'b-')
                # plt.xlabel('Epoch')
                # plt.ylabel('Loss')
                # plt.title('Total Loss (linear scale)')
                # plt.grid(True)
                
                # # Component losses (linear scale)
                # plt.subplot(2, 2, 4)
                # plt.plot(loss_history['epoch'], loss_history['consistency_loss'], 'r-', label='Consistency')
                # plt.plot(loss_history['epoch'], loss_history['clearing_loss'], 'g-', label='Clearing')
                # plt.plot(loss_history['epoch'], loss_history['boundary_loss'], 'm-', label='Boundary')
                # plt.plot(loss_history['epoch'], loss_history['goods_market_loss'], 'y-', label='Goods Market')
                # plt.xlabel('Epoch')
                # plt.ylabel('Loss')
                # plt.title('Component Losses (linear scale)')
                # plt.legend()
                #plt.grid(True)
                
                plt.tight_layout()
                plt.savefig(f"{PLOT_DIR}/equilibrium_loss_{TAG}.png")
                plt.close()
                
                print(f"Equilibrium training loss plot saved to {PLOT_DIR}/equilibrium_loss_{TAG}.png")
                
            # Then, check value function loss
            loss_file = f"{OUTPUT_DIR}/value_model_{TAG}_loss.csv"
            if os.path.exists(loss_file):
                loss_history = pd.read_csv(loss_file)
                
                plt.figure(figsize=(10, 8))
                
                # Total loss (log scale)
                plt.subplot(2, 2, 1)
                plt.semilogy(loss_history['epoch'], loss_history['total_loss'], 'b-')
                plt.xlabel('Epoch')
                plt.ylabel('Loss (log scale)')
                plt.title('Total HJB Loss')
                plt.grid(True)
                
                # Component losses (log scale)
                plt.subplot(2, 2, 2)
                plt.semilogy(loss_history['epoch'], loss_history['hjb_loss'], 'r-', label='HJB Equation')
                plt.semilogy(loss_history['epoch'], loss_history['shape_loss'], 'g-', label='Shape Constraint')
                plt.semilogy(loss_history['epoch'], loss_history['zero_avoidance_loss'], 'b-', label='Zero Avoidance Constraint')
                plt.xlabel('Epoch')
                plt.ylabel('Loss (log scale)')
                plt.title('Component Losses')
                plt.legend()
                plt.grid(True)
                
                # Total loss (linear scale)
                plt.subplot(2, 2, 3)
                plt.plot(loss_history['epoch'], loss_history['total_loss'], 'b-')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.title('Total HJB Loss (linear scale)')
                plt.grid(True)
                
                # Component losses (linear scale)
                plt.subplot(2, 2, 4)
                plt.plot(loss_history['epoch'], loss_history['hjb_loss'], 'r-', label='HJB Equation')
                plt.plot(loss_history['epoch'], loss_history['shape_loss'], 'g-', label='Shape Constraint')
                plt.plot(loss_history['epoch'], loss_history['zero_avoidance_loss'], 'g-', label='Zero Avoidance Constraint')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.title('Component Losses (linear scale)')
                plt.legend()
                plt.grid(True)
                
                plt.tight_layout()
                plt.savefig(f"{PLOT_DIR}/value_function_loss_{TAG}.png")
                plt.close()
                
                print(f"Value function training loss plot saved to {PLOT_DIR}/value_function_loss_{TAG}.png")
                
        elif MODEL_TYPE == "CPDL":
            # For CPDL model, we have a single loss file
            loss_file = f"{OUTPUT_DIR}/model_{TAG}_loss.csv"
            if os.path.exists(loss_file):
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
        print(f"Error plotting training loss: {e}")
        import traceback
        traceback.print_exc()

def plot_pretrain_loss_m3dl():
    """Plot the pretraining loss for M3DL models"""
    try:
        if MODEL_TYPE != "M3DL":
            return
            
        loss_file = f"{OUTPUT_DIR}/value_pretrain_loss.npy"
        if not os.path.exists(loss_file):
            print(f"Error: Pretraining loss file {loss_file} not found")
            return
            
        loss_data = np.load(loss_file)
        
        # Create plot
        plt.figure(figsize=(10, 6))
        plt.semilogy(np.arange(0, len(loss_data)*env.print_every, env.print_every), loss_data)
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        plt.title('Value Function Pre-training Loss')
        plt.grid(True)
        plt.savefig(f"{PLOT_DIR}/value_pretrain_loss_{TAG}.png", dpi=300)
        plt.close()
        
        print(f"Value function pre-training loss plot saved to {PLOT_DIR}/value_pretrain_loss_{TAG}.png")
        
    except Exception as e:
        print(f"Error plotting pretraining loss: {e}")
        import traceback
        traceback.print_exc()

# def plot_value_function_slices():
#     """Plot value function slices"""
#     try:
#         if MODEL_TYPE == "M3DL":
#             # For M3 model, plot from value_function_slices
#             slice_file = f"{OUTPUT_DIR}/value_function_slices_{TAG}.npz"
#             if not os.path.exists(slice_file):
#                 print(f"Error: Value function slices file {slice_file} not found")
#                 return
                
#             # Load data
#             data = np.load(slice_file, allow_pickle=True) 
#             eta_grid = data['eta_grid']
#             H_grid = data['H_grid']
#             tau_grid = data['tau_grid']
#             W_eta_H = data['W_eta_H']
#             W_eta_tau = data['W_eta_tau']
#             W_effective_tau_lines = data['W_effective_tau_lines'].item()
#             W_effective_max_tau = data['W_effective_max_tau'].item()
#             q1_vals = data['q1_vals'].item()
#             q2_vals = data['q2_vals'].item()

#             combinations = data['combinations']
            
#             # 1. Plot W(eta, H, tau=tau_0)
#             plt.figure(figsize=(10, 8))
#             eta_mesh, H_mesh = np.meshgrid(eta_grid, H_grid, indexing='ij')
            
#             contour = plt.contourf(eta_mesh, H_mesh, W_eta_H, 50, cmap='viridis')
#             plt.colorbar(contour, label='Value Function W')
#             plt.xlabel('Green Capital Share η')
#             plt.ylabel('Climate Damage H')
#             plt.title(f'Value Function W(η, H, τ_0)')
#             plt.grid(True, linestyle='--', alpha=0.3)
            
#             plt.savefig(f"{PLOT_DIR}/W_eta_H_slice_{TAG}.png", dpi=300)
#             plt.close()
            
#             print(f"W(eta, H) slice plot saved to {PLOT_DIR}/W_eta_H_slice_{TAG}.png")
            
#             # 2. Plot W(eta, H=1, tau)
#             plt.figure(figsize=(10, 8))
#             eta_mesh_tau, tau_mesh = np.meshgrid(eta_grid, tau_grid, indexing='ij')
            
#             contour = plt.contourf(eta_mesh_tau, tau_mesh, W_eta_tau, 50, cmap='viridis')
#             plt.colorbar(contour, label='Value Function W')
#             plt.xlabel('Green Capital Share η')
#             plt.ylabel('Carbon Tax τ')
#             plt.title(f'Value Function W(η, H=1.0, τ)')
#             plt.grid(True, linestyle='--', alpha=0.3)
            
#             plt.savefig(f"{PLOT_DIR}/W_eta_tau_slice_{TAG}.png", dpi=300)
#             plt.close()
            
#             print(f"W(eta, tau) slice plot saved to {PLOT_DIR}/W_eta_tau_slice_{TAG}.png")
            
#             # ...
#             # 3. Plot W_effective(tau) lines for various combinations with optimal tau marked
#             plt.figure(figsize=(12, 8))
#             # Get the first combination (eta_0, H_0) to make bold
#             eta_0_H_0_label = combinations[0][2]

#             # Plot each line
#             for label, values in W_effective_tau_lines.items():
#                 if label == eta_0_H_0_label:
#                     # Plot (eta_0, H_0) in bold
#                     plt.plot(tau_grid, values, linewidth=3, label=label, color='blue')
#                 else:
#                     plt.plot(tau_grid, values, linewidth=1.5, label=label)
                    
#                 # Plot vertical line at optimal tau for each curve
#                 optimal_tau = W_effective_max_tau[label]['tau']
#                 plt.axvline(x=optimal_tau, color='gray', linestyle='--', alpha=0.7)
                
#                 # Add text annotation for optimal tau
#                 max_idx = np.argmax(values)
#                 plt.annotate(f'τ_opt = {optimal_tau:.4f}',
#                             xy=(optimal_tau, values[max_idx]),
#                             xytext=(optimal_tau + 0.01, values[max_idx] + 0.1),
#                             arrowprops=dict(facecolor='black', shrink=0.05),
#                             fontsize=8)

#             # Add vertical line at tau_star
#             plt.axvline(x=env.tau_star, color='red', linestyle=':', label=f'τ* = {env.tau_star:.4f}')

#             plt.xlabel('Carbon Tax τ')
#             plt.ylabel('Effective Value Function W_eff')
#             plt.title(f'Effective Value Function W_eff(τ) for Different (η, H) Combinations')
#             plt.grid(True)
#             plt.legend()

#             plt.savefig(f"{PLOT_DIR}/W_eff_tau_lines_{TAG}.png", dpi=300)
                        
#             print(f"W(tau) line plots saved to {PLOT_DIR}/W_tau_lines_{TAG}.png")


#             # 4. plot q lines for various combinations
#             plt.figure(figsize=(12, 8))
#             # Get the first combination (eta_0, H_0) to make bold
#             eta_0_H_0_label = combinations[0][2]

#             # Plot each line
#             for label, values in q1_vals.items():
#                 if label == eta_0_H_0_label:
#                     # Plot (eta_0, H_0) in bold
#                     plt.plot(tau_grid, values, linewidth=3, label=label)
#                 else:
#                     plt.plot(tau_grid, values, linewidth=1.5, label=label)

#             for label, values in q2_vals.items():
#                 if label == eta_0_H_0_label:
#                     # Plot (eta_0, H_0) in bold
#                     plt.plot(tau_grid, values, linewidth=3, label=label)
#                 else:
#                     plt.plot(tau_grid, values, linewidth=1.5, label=label)

#             plt.xlabel('Carbon Tax τ')
#             plt.ylabel('Capital prices q_1, q_2')
#             plt.title(f'q_1(tau), q_2(tau) for Different (η, H) Combinations')
#             plt.grid(True)
#             plt.legend()

#             plt.savefig(f"{PLOT_DIR}/q12_tau_lines_{TAG}.png", dpi=300)
                        
#             print(f"q12(tau) line plots saved to {PLOT_DIR}/q12_tau_lines_{TAG}.png")
            
#         elif MODEL_TYPE in ["CPDL", "CPFD"]:
#             # For CP models, we need to look at grid evaluation or results file
#             if MODEL_TYPE == "CPDL":
#                 grid_file = f"{OUTPUT_DIR}/grid_evaluation_{TAG}.npz"
#                 value_slice_file = f"{OUTPUT_DIR}/value_function_slices_{TAG}.npz"
                
#                 if os.path.exists(value_slice_file):
#                     # Use precomputed slices if available
#                     data = np.load(value_slice_file, allow_pickle=True)
#                     eta_grid = data['eta_grid']
#                     H_grid = data['H_grid']
#                     W_eta_H = data['W_eta_H']
                    
#                     plt.figure(figsize=(10, 8))
#                     eta_mesh, H_mesh = np.meshgrid(eta_grid, H_grid, indexing='ij')
                    
#                     contour = plt.contourf(eta_mesh, H_mesh, W_eta_H, 50, cmap='viridis')
#                     plt.colorbar(contour, label='Value Function W')
#                     plt.xlabel('Green Capital Share η')
#                     plt.ylabel('Climate Damage H')
#                     plt.title(f'Value Function W(η, H)')
#                     plt.grid(True, linestyle='--', alpha=0.3)
                    
#                     plt.savefig(f"{PLOT_DIR}/W_eta_H_{TAG}.png", dpi=300)
#                     plt.close()
                    
#                     print(f"W(eta, H) plot saved to {PLOT_DIR}/W_eta_H_{TAG}.png")
                    
#                 elif os.path.exists(grid_file):
#                     # Use grid evaluation
#                     data = np.load(grid_file, allow_pickle=True)
#                     eta_mesh = data['eta_mesh']
#                     H_mesh = data['H_mesh']
#                     W_values = data['W_values']
                    
#                     plt.figure(figsize=(10, 8))
#                     contour = plt.contourf(eta_mesh, H_mesh, W_values, 50, cmap='viridis')
#                     plt.colorbar(contour, label='Value Function W')
#                     plt.xlabel('Green Capital Share η')
#                     plt.ylabel('Climate Damage H')
#                     plt.title(f'Value Function W(η, H)')
#                     plt.grid(True, linestyle='--', alpha=0.3)
                    
#                     plt.savefig(f"{PLOT_DIR}/W_eta_H_{TAG}.png", dpi=300)
#                     plt.close()
                    
#                     print(f"W(eta, H) plot saved to {PLOT_DIR}/W_eta_H_{TAG}.png")
                
#             elif MODEL_TYPE == "CPFD":
#                 results_file = f"{OUTPUT_DIR}/CP_FD_results_{TAG}.npz"
                
#                 if os.path.exists(results_file):
#                     data = np.load(results_file, allow_pickle=True)
#                     eta_mesh = data['eta_mesh']
#                     H_mesh = data['H_mesh']
#                     W_sol = data['W_sol']
                    
#                     plt.figure(figsize=(10, 8))
#                     contour = plt.contourf(eta_mesh, H_mesh, W_sol, 50, cmap='viridis')
#                     plt.colorbar(contour, label='Value Function W')
#                     plt.xlabel('Green Capital Share η')
#                     plt.ylabel('Climate Damage H')
#                     plt.title(f'Value Function W(η, H)')
#                     plt.grid(True, linestyle='--', alpha=0.3)
                    
#                     plt.savefig(f"{PLOT_DIR}/W_eta_H_{TAG}.png", dpi=300)
#                     plt.close()
                    
#                     print(f"W(eta, H) plot saved to {PLOT_DIR}/W_eta_H_{TAG}.png")
                
#     except Exception as e:
#         print(f"Error in plot_value_function_slices: {e}")
#         import traceback
#         traceback.print_exc()

# Add plot_w_vs_w_effective function
def plot_w_vs_w_effective():
    """Plot W, W_effective, and their difference"""
    try:
        if MODEL_TYPE != "M3DL":
            return
            
        # Load value function slices
        slice_file = f"{OUTPUT_DIR}/value_function_slices_{TAG}.npz"
        if not os.path.exists(slice_file):
            print(f"Error: Value function slices file {slice_file} not found")
            return
            
        # Load data
        data = np.load(slice_file, allow_pickle=True)
        eta_grid = data['eta_grid']
        H_grid = data['H_grid']
        W_eta_H = data['W_eta_H']
        W_effective_eta_H = data['W_effective_eta_H']
        
        # Create proper meshgrids
        eta_mesh, H_mesh = np.meshgrid(eta_grid, H_grid, indexing='ij')
        
        # Calculate difference
        W_diff = W_effective_eta_H - W_eta_H
        
        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Plot W
        contour = axes[0].contourf(eta_mesh, H_mesh, W_eta_H, 50, cmap='viridis')
        plt.colorbar(contour, ax=axes[0], label='Value')
        axes[0].set_xlabel('Green Capital Share η')
        axes[0].set_ylabel('Climate Damage H')
        axes[0].set_title('W(η, H)')
        
        # Plot W_effective
        contour = axes[1].contourf(eta_mesh, H_mesh, W_effective_eta_H, 50, cmap='viridis')
        plt.colorbar(contour, ax=axes[1], label='Value')
        axes[1].set_xlabel('Green Capital Share η')
        axes[1].set_ylabel('Climate Damage H')
        axes[1].set_title('W_effective(η, H)')
        
        # Plot difference with diverging colormap centered at zero
        max_abs_diff = max(abs(W_diff.min()), abs(W_diff.max()))
        norm = colors.TwoSlopeNorm(vmin=-max_abs_diff, vcenter=0, vmax=max_abs_diff)
        contour = axes[2].contourf(eta_mesh, H_mesh, W_diff, 50, cmap='bwr', norm=norm)
        plt.colorbar(contour, ax=axes[2], label='Difference')
        axes[2].set_xlabel('Green Capital Share η')
        axes[2].set_ylabel('Climate Damage H')
        axes[2].set_title('Difference (W_effective - W)')
        
        plt.savefig(f"{PLOT_DIR}/W_vs_W_effective_{TAG}.png", dpi=300)
        plt.close()
        
    except Exception as e:
        print(f"Error plotting W_vs_W_effective: {e}")
        import traceback
        traceback.print_exc()


def plot_w_vs_guess():
    """Plot W, W_guess, and their difference"""
    try:
        if MODEL_TYPE != "M3DL":
            return
        # Load value function slices
        slice_file = f"{OUTPUT_DIR}/value_function_slices_{TAG}.npz"
        if not os.path.exists(slice_file):
            print(f"Error: Value function slices file {slice_file} not found")
            return
            
        # Load data
        data = np.load(slice_file, allow_pickle=True)
        eta_grid = data['eta_grid']
        H_grid = data['H_grid']
        W_eta_H = data['W_eta_H']

        # Create proper meshgrids
        eta_mesh, H_mesh = np.meshgrid(eta_grid, H_grid, indexing='ij')

        # Calculate initial guess
        W_guess = env.initial_guess(eta_mesh, H_mesh, tau=env.tau_0)

        # Calculate difference
        W_diff = W_eta_H - W_guess
        
        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Plot W
        contour = axes[0].contourf(eta_mesh, H_mesh, W_eta_H, 50, cmap='viridis')
        plt.colorbar(contour, ax=axes[0], label='Value')
        axes[0].set_xlabel('Green Capital Share η')
        axes[0].set_ylabel('Climate Damage H')
        axes[0].set_title('Solution for W(η, H)')
        
        # Plot W_guess
        contour = axes[1].contourf(eta_mesh, H_mesh, W_guess, 50, cmap='viridis')
        plt.colorbar(contour, ax=axes[1], label='Value')
        axes[1].set_xlabel('Green Capital Share η')
        axes[1].set_ylabel('Climate Damage H')
        axes[1].set_title('Initial Guess of W(η, H)')
        
        # Plot difference with diverging colormap centered at zero
        max_abs_diff = max(abs(W_diff.min()), abs(W_diff.max()))
        norm = colors.TwoSlopeNorm(vmin=-max_abs_diff, vcenter=0, vmax=max_abs_diff)
        contour = axes[2].contourf(eta_mesh, H_mesh, W_diff, 50, cmap='bwr', norm=norm)
        plt.colorbar(contour, ax=axes[2], label='Difference')
        axes[2].set_xlabel('Green Capital Share η')
        axes[2].set_ylabel('Climate Damage H')
        axes[2].set_title('Difference (W_solution - W_guess)')
        
        plt.savefig(f"{PLOT_DIR}/W_vs_initial_guess_{TAG}.png", dpi=300)
        plt.close()
        print(f"w vs initial guess plots saved to {PLOT_DIR}/W_vs_initial_guess_{TAG}.png")
        
    except Exception as e:
        print(f"Error plotting W_vs_W_effective: {e}")
        import traceback
        traceback.print_exc()



def compute_q_from_iota(sim_data):
    if 'iota_1' in sim_data and 'iota_2' in sim_data:
        q1 = 1 + env.phi * sim_data['iota_1']
        q2 = 1 + env.phi * sim_data['iota_2']
        return q1, q2
    return None, None

def create_fan_charts():
    """Create fan charts from simulation results"""
    try:
        # Check for simulation results
        if MODEL_TYPE in ["M3DL", "CPDL"]:
            sim_file = f"{OUTPUT_DIR}/simulation_results_{TAG}.npz"
            if not os.path.exists(sim_file):
                print(f"Error: Simulation results file {sim_file} not found")
                return
                
            # Load simulation results
            results = np.load(sim_file, allow_pickle=True)
            
            if MODEL_TYPE == "M3DL":
                # M3 model has specific variables
                eta_results = results['eta']
                H_results = results['H']
                tau_results = results['tau']
                K_results = results['K']
                iota_1_results = results['iota_1']
                iota_2_results = results['iota_2']
                value_results = results['value']
                dt = float(results['dt'])
                N_T = int(results['N_T'])
                
                # Create time array
                time = np.arange(0, (N_T + 1) * dt, dt)
                
                # Set percentiles for fan charts
                percentiles = env.sim_percentiles
                
                # Create figure
                fig, axes = plt.subplots(3, 3, figsize=(15, 12))
                fig.suptitle(f'Simulation Results - Tag: {TAG} (T={N_T*dt:.1f}, dt={dt:.3f})', fontsize=16)
                
                # Flatten axes for easier indexing
                axes = axes.flatten()
                
                # Data to plot
                all_data = [
                    (eta_results, 'Green Capital Share η', 0),
                    (H_results, 'Climate Damage H', 1),
                    (tau_results, 'Carbon Tax τ', 2),
                    (K_results, 'Aggregate Capital K', 3),
                    (iota_1_results, 'Green Investment Rate ι₁', 4),
                    (iota_2_results, 'Brown Investment Rate ι₂', 5),
                    (value_results, 'Value Function V', 6),
                ]
                # In simulation plotting for M3DL models:
                q1_results, q2_results = compute_q_from_iota(results)
                all_data.append((q1_results, 'Green Capital Price q1', 7))
                all_data.append((q2_results, 'Brown Capital Price q2', 8))
                
                
            elif MODEL_TYPE == "CPDL":
                # Central planner has different variables
                eta_results = results['eta']
                H_results = results['H']
                K_results = results['K']
                c_results = results['c']
                iota_1_results = results['iota_1']
                iota_2_results = results['iota_2']
                dt = float(results['dt'])
                N_T = int(results['N_T'])
                
                # Create time array
                time = np.arange(0, (N_T + 1) * dt, dt)
                
                # Set percentiles for fan charts
                percentiles = env.sim_percentiles
                
                # Create figure
                fig, axes = plt.subplots(2, 3, figsize=(15, 10))
                fig.suptitle(f'Central Planner Simulation Results - Tag: {TAG} (T={N_T*dt:.1f}, dt={dt:.3f})', fontsize=16)
                
                # Flatten axes for easier indexing
                axes = axes.flatten()
                
                # Data to plot
                all_data = [
                    (eta_results, 'Green Capital Share η', 0),
                    (H_results, 'Climate Damage H', 1),
                    (K_results, 'Aggregate Capital K', 2),
                    (c_results, 'Consumption c', 3),
                    (iota_1_results, 'Green Investment Rate ι₁', 4),
                    (iota_2_results, 'Brown Investment Rate ι₂', 5),
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
                
                # Apply fix for subscript labels
                fix_subscript_labels(ax)
                
                # Add legend to the first plot only
                if idx == 0:
                    ax.legend()
            
            # Remove unused subplots if any
            if MODEL_TYPE == "M3DL" and len(axes) > 7:
                if len(all_data) < len(axes):
                    for i in range(len(all_data), len(axes)):
                        fig.delaxes(axes[i])
            
            # Adjust layout
            plt.tight_layout()
            plt.subplots_adjust(top=0.92)
            
            # Save
            plt.savefig(f"{PLOT_DIR}/simulation_fan_charts_{TAG}.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Simulation fan charts saved to {PLOT_DIR}/simulation_fan_charts_{TAG}.png")
            
    except Exception as e:
        print(f"Error creating fan charts: {e}")
        import traceback
        traceback.print_exc()

def plot_s_values():
    """Plot the s values (volatility denominator) on the (eta, H) slice at tau=tau_0"""
    try:
        if MODEL_TYPE != "M3DL":
            print("Skipping s-value plot - only applicable to M3DL models")
            return
            
        # Check if we have precomputed values or need to compute them
        s_values_file = f"{OUTPUT_DIR}/s_values_{TAG}.npz"
        
        if os.path.exists(s_values_file):
            # Load precomputed s values
            data = np.load(s_values_file, allow_pickle=True)
            eta_grid = data['eta_grid']
            H_grid = data['H_grid']
            s_values = data['s_values']
            
            # Create meshgrid for plotting
            eta_mesh, H_mesh = np.meshgrid(eta_grid, H_grid, indexing='ij')
        else:
            print(f"Error: s-values file {s_values_file} not found")
            print("Run gen.py to generate the required s-values file")
            return
        
        # Create the plot
        plt.figure(figsize=(10, 8))
        
        # Use log scale for colormap since s can vary by orders of magnitude
        log_s = np.log10(np.clip(s_values, 1e-10, None))  # Clip to avoid log(0)
        contour = plt.contourf(eta_mesh, H_mesh, log_s, 50, cmap='viridis')
        
        # Create colorbar with exponent labels
        cbar = plt.colorbar(contour)
        cbar.set_label('log10(s)')
        
        plt.xlabel('Green Capital Share η')
        plt.ylabel('Climate Damage H')
        plt.title(f'Log10 of Volatility Term s(η, H, τ={env.tau_0:.4f})')
        plt.grid(True, linestyle='--', alpha=0.3)
        
        # Add contour lines for easier reading
        contour_lines = plt.contour(eta_mesh, H_mesh, log_s, 
                                  levels=np.linspace(log_s.min(), log_s.max(), 10),
                                  colors='k', alpha=0.3, linewidths=0.5)
        plt.clabel(contour_lines, inline=True, fontsize=8, fmt='%.1f')
        
        plt.savefig(f"{PLOT_DIR}/s_values_{TAG}.png", dpi=300)
        plt.close()
        
        # Also plot the original s (not log scale) with a different colormap
        plt.figure(figsize=(10, 8))
        contour = plt.pcolormesh(eta_mesh, H_mesh, s_values, 
                               cmap='magma', norm=colors.LogNorm())
        cbar = plt.colorbar(contour)
        cbar.set_label('s')
        
        plt.xlabel('Green Capital Share η')
        plt.ylabel('Climate Damage H')
        plt.title(f'Volatility Term s(η, H, τ={env.tau_0:.4f})')
        plt.grid(True, linestyle='--', alpha=0.3)
        
        plt.savefig(f"{PLOT_DIR}/s_values_raw_{TAG}.png", dpi=300)
        plt.close()
        
        print(f"s-value plots saved to {PLOT_DIR}/s_values_{TAG}.png and {PLOT_DIR}/s_values_raw_{TAG}.png")
        
    except Exception as e:
        print(f"Error plotting s values: {e}")
        import traceback
        traceback.print_exc()

def plot_value_function_slices():
    """Plot value function slices with proper tensor handling"""
    try:
        if MODEL_TYPE == "M3DL":
            # For M3 model, plot from value_function_slices
            slice_file = f"{OUTPUT_DIR}/value_function_slices_{TAG}.npz"
            if not os.path.exists(slice_file):
                print(f"Error: Value function slices file {slice_file} not found")
                return
                
            # Load data
            data = np.load(slice_file, allow_pickle=True) 
            eta_grid = data['eta_grid']
            H_grid = data['H_grid']
            tau_grid = data['tau_grid']
            W_eta_H = data['W_eta_H']
            W_eta_tau = data['W_eta_tau']
            
            # Safe conversion function for tensors
            def safe_convert(data):
                if hasattr(data, 'device'):
                    return data.cpu().numpy()
                elif hasattr(data, 'numpy'):
                    return data.numpy()
                return data
            
            # Safely load dictionary data
            W_tau_lines = {}
            W_effective_tau_lines = {}
            q1_vals = {}
            q2_vals = {}
            W_effective_max_tau = {}
            
            try:
                # Load dictionaries with proper tensor handling
                if 'W_tau_lines' in data:
                    W_tau_lines_data = data['W_tau_lines'].item()
                    for key, val in W_tau_lines_data.items():
                        W_tau_lines[key] = safe_convert(val)
                
                if 'W_effective_tau_lines' in data:
                    W_effective_tau_lines_data = data['W_effective_tau_lines'].item()
                    for key, val in W_effective_tau_lines_data.items():
                        W_effective_tau_lines[key] = safe_convert(val)
                
                if 'q1_vals' in data:
                    q1_vals_data = data['q1_vals'].item()
                    for key, val in q1_vals_data.items():
                        q1_vals[key] = safe_convert(val)
                
                if 'q2_vals' in data:
                    q2_vals_data = data['q2_vals'].item()
                    for key, val in q2_vals_data.items():
                        q2_vals[key] = safe_convert(val)
                
                # Handle nested dictionary carefully
                if 'W_effective_max_tau' in data:
                    W_effective_max_tau_data = data['W_effective_max_tau'].item()
                    for key, val in W_effective_max_tau_data.items():
                        if isinstance(val, dict):
                            new_inner_dict = {}
                            for inner_key, inner_val in val.items():
                                new_inner_dict[inner_key] = safe_convert(inner_val)
                            W_effective_max_tau[key] = new_inner_dict
                        else:
                            W_effective_max_tau[key] = safe_convert(val)
                
                # Load combinations
                if 'combinations' in data:
                    combinations = data['combinations']
                else:
                    print("Warning: 'combinations' not found in data file")
                    combinations = []
                    
            except Exception as e:
                print(f"Error loading dictionary data: {e}")
                print("Attempting alternative data loading approach...")
                
                # If the above fails, try a more direct approach
                try:
                    import torch
                    
                    # Map any tensors to CPU explicitly
                    def cpu_tensor(tensor_bytes):
                        try:
                            tensor = torch.load(BytesIO(tensor_bytes), map_location='cpu')
                            return tensor.numpy()
                        except:
                            return tensor_bytes
                            
                    # Reload with CPU mapping
                    data = np.load(slice_file, allow_pickle=True)
                    
                    # Try to extract basic combinations 
                    if 'combinations' in data:
                        combinations = data['combinations']
                    else:
                        combinations = [
                            (env.sim_eta_0, env.sim_H_0, f"η={env.sim_eta_0}, H={env.sim_H_0}"),
                            (env.sim_eta_0, env.H_high, f"η={env.sim_eta_0}, H={env.H_high}"),
                            (env.eta_high, env.sim_H_0, f"η={env.eta_high}, H={env.sim_H_0}"),
                            (env.eta_high, env.H_high, f"η={env.eta_high}, H={env.H_high}")
                        ]
                except Exception as e2:
                    print(f"Alternative loading also failed: {e2}")
                    print("Proceeding with basic plots only")
                    W_tau_lines = {}
                    W_effective_tau_lines = {}
                    q1_vals = {}
                    q2_vals = {}
                    W_effective_max_tau = {}
                    combinations = []
            
            # 1. Plot W(eta, H, tau=tau_0)
            plt.figure(figsize=(10, 8))
            eta_mesh, H_mesh = np.meshgrid(eta_grid, H_grid, indexing='ij')
            
            contour = plt.contourf(eta_mesh, H_mesh, W_eta_H, 50, cmap='viridis')
            plt.colorbar(contour, label='Value Function W')
            plt.xlabel('Green Capital Share η')
            plt.ylabel('Climate Damage H')
            plt.title(f'Value Function W(η, H, τ_0)')
            plt.grid(True, linestyle='--', alpha=0.3)
            
            plt.savefig(f"{PLOT_DIR}/W_eta_H_slice_{TAG}.png", dpi=300)
            plt.close()
            
            print(f"W(eta, H) slice plot saved to {PLOT_DIR}/W_eta_H_slice_{TAG}.png")
            
            # 2. Plot W(eta, H=1, tau)
            plt.figure(figsize=(10, 8))
            eta_mesh_tau, tau_mesh = np.meshgrid(eta_grid, tau_grid, indexing='ij')
            
            contour = plt.contourf(eta_mesh_tau, tau_mesh, W_eta_tau, 50, cmap='viridis')
            plt.colorbar(contour, label='Value Function W')
            plt.xlabel('Green Capital Share η')
            plt.ylabel('Carbon Tax τ')
            plt.title(f'Value Function W(η, H=1.0, τ)')
            plt.grid(True, linestyle='--', alpha=0.3)
            
            plt.savefig(f"{PLOT_DIR}/W_eta_tau_slice_{TAG}.png", dpi=300)
            plt.close()
            
            print(f"W(eta, tau) slice plot saved to {PLOT_DIR}/W_eta_tau_slice_{TAG}.png")
            
            # 3. Plot W_effective(tau) lines only if we have data
            if (W_effective_tau_lines is not None and len(W_effective_tau_lines) > 0 and 
                W_effective_max_tau is not None and len(W_effective_max_tau) > 0 and 
                combinations is not None and len(combinations) > 0):
                plt.figure(figsize=(12, 8))
                # Get the first combination to make bold
                eta_0_H_0_label = combinations[0][2] if len(combinations) > 0 else None

                # Plot each line
                for label, values in W_effective_tau_lines.items():
                    if eta_0_H_0_label and label == eta_0_H_0_label:
                        # Plot in bold
                        plt.plot(tau_grid, values, linewidth=3, label=label, color='blue')
                    else:
                        plt.plot(tau_grid, values, linewidth=1.5, label=label)
                        
                    # Plot vertical line at optimal tau for each curve
                    if label in W_effective_max_tau:
                        optimal_tau = W_effective_max_tau[label]['tau']
                        plt.axvline(x=optimal_tau, color='gray', linestyle='--', alpha=0.7)
                        
                        # Add text annotation for optimal tau
                        max_idx = np.argmax(values)
                        plt.annotate(f'τ_opt = {optimal_tau:.4f}',
                                    xy=(optimal_tau, values[max_idx]),
                                    xytext=(optimal_tau + 0.01, values[max_idx] + 0.1),
                                    arrowprops=dict(facecolor='black', shrink=0.05),
                                    fontsize=8)

                # Add vertical line at tau_star
                plt.axvline(x=env.tau_star, color='red', linestyle=':', label=f'τ* = {env.tau_star:.4f}')

                plt.xlabel('Carbon Tax τ')
                plt.ylabel('Effective Value Function W_eff')
                plt.title(f'Effective Value Function W_eff(τ) for Different (η, H) Combinations')
                plt.grid(True)
                plt.legend()

                plt.savefig(f"{PLOT_DIR}/W_eff_tau_lines_{TAG}.png", dpi=300)
                print(f"W_effective(tau) line plots saved to {PLOT_DIR}/W_eff_tau_lines_{TAG}.png")
            else:
                print("Skipping W_effective(tau) plots: insufficient data")

            # 4. Plot q1/q2 lines only if we have data
            if (q1_vals is not None and len(q1_vals) > 0 and 
                q2_vals is not None and len(q2_vals) > 0 and 
                combinations is not None and len(combinations) > 0):
                plt.figure(figsize=(12, 8))
                # Get the first combination to make bold
                eta_0_H_0_label = combinations[0][2] if len(combinations) > 0 else None

                # Plot each line
                for label, values in q1_vals.items():
                    if eta_0_H_0_label and label == eta_0_H_0_label:
                        plt.plot(tau_grid, values, linewidth=3, label=f"{label} - q1")
                    else:
                        plt.plot(tau_grid, values, linewidth=1.5, label=f"{label} - q1")

                for label, values in q2_vals.items():
                    if eta_0_H_0_label and label == eta_0_H_0_label:
                        plt.plot(tau_grid, values, linewidth=3, label=f"{label} - q2", linestyle='--')
                    else:
                        plt.plot(tau_grid, values, linewidth=1.5, label=f"{label} - q2", linestyle='--')

                plt.xlabel('Carbon Tax τ')
                plt.ylabel('Capital prices q_1, q_2')
                plt.title(f'q_1(tau), q_2(tau) for Different (η, H) Combinations')
                plt.grid(True)
                plt.legend()

                plt.savefig(f"{PLOT_DIR}/q12_tau_lines_{TAG}.png", dpi=300)
                print(f"q12(tau) line plots saved to {PLOT_DIR}/q12_tau_lines_{TAG}.png")
            else:
                print("Skipping q1/q2 plots: insufficient data")
            
            # Rest of the function remains the same...
        elif MODEL_TYPE in ["CPDL", "CPFD"]:
            # For CP models, we need to look at grid evaluation or results file
            if MODEL_TYPE == "CPDL":
                grid_file = f"{OUTPUT_DIR}/grid_evaluation_{TAG}.npz"
                value_slice_file = f"{OUTPUT_DIR}/value_function_slices_{TAG}.npz"
                
                if os.path.exists(value_slice_file):
                    # Use precomputed slices if available
                    data = np.load(value_slice_file, allow_pickle=True)
                    eta_grid = data['eta_grid']
                    H_grid = data['H_grid']
                    W_eta_H = data['W_eta_H']
                    
                    plt.figure(figsize=(10, 8))
                    eta_mesh, H_mesh = np.meshgrid(eta_grid, H_grid, indexing='ij')
                    
                    contour = plt.contourf(eta_mesh, H_mesh, W_eta_H, 50, cmap='viridis')
                    plt.colorbar(contour, label='Value Function W')
                    plt.xlabel('Green Capital Share η')
                    plt.ylabel('Climate Damage H')
                    plt.title(f'Value Function W(η, H)')
                    plt.grid(True, linestyle='--', alpha=0.3)
                    
                    plt.savefig(f"{PLOT_DIR}/W_eta_H_{TAG}.png", dpi=300)
                    plt.close()
                    
                    print(f"W(eta, H) plot saved to {PLOT_DIR}/W_eta_H_{TAG}.png")
                    
                elif os.path.exists(grid_file):
                    # Use grid evaluation
                    data = np.load(grid_file, allow_pickle=True)
                    eta_mesh = data['eta_mesh']
                    H_mesh = data['H_mesh']
                    W_values = data['W_values']
                    
                    plt.figure(figsize=(10, 8))
                    contour = plt.contourf(eta_mesh, H_mesh, W_values, 50, cmap='viridis')
                    plt.colorbar(contour, label='Value Function W')
                    plt.xlabel('Green Capital Share η')
                    plt.ylabel('Climate Damage H')
                    plt.title(f'Value Function W(η, H)')
                    plt.grid(True, linestyle='--', alpha=0.3)
                    
                    plt.savefig(f"{PLOT_DIR}/W_eta_H_{TAG}.png", dpi=300)
                    plt.close()
                    
                    print(f"W(eta, H) plot saved to {PLOT_DIR}/W_eta_H_{TAG}.png")
                
            elif MODEL_TYPE == "CPFD":
                results_file = f"{OUTPUT_DIR}/CP_FD_results_{TAG}.npz"
                
                if os.path.exists(results_file):
                    data = np.load(results_file, allow_pickle=True)
                    eta_mesh = data['eta_mesh']
                    H_mesh = data['H_mesh']
                    W_sol = data['W_sol']
                    
                    plt.figure(figsize=(10, 8))
                    contour = plt.contourf(eta_mesh, H_mesh, W_sol, 50, cmap='viridis')
                    plt.colorbar(contour, label='Value Function W')
                    plt.xlabel('Green Capital Share η')
                    plt.ylabel('Climate Damage H')
                    plt.title(f'Value Function W(η, H)')
                    plt.grid(True, linestyle='--', alpha=0.3)
                    
                    plt.savefig(f"{PLOT_DIR}/W_eta_H_{TAG}.png", dpi=300)
                    plt.close()
                    
                    print(f"W(eta, H) plot saved to {PLOT_DIR}/W_eta_H_{TAG}.png")         
    except Exception as e:
        print(f"Error in plot_value_function_slices: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    try:
        print(f"Starting plotting process for tag {TAG}...")
        
        # Plot training loss
        plot_training_loss()
        
        # Plot pretraining loss for M3DL models
        plot_pretrain_loss_m3dl()
        
        # Plot value function slices
        plot_value_function_slices()

        # Plot w vs guess
        plot_w_vs_guess()
        
        # Plot W vs W_effective comparison
        plot_w_vs_w_effective()
        
        # Plot s values (for M3DL models)
        plot_s_values()
        
        # Plot simulation results
        create_fan_charts()
        
        print("All plots generated successfully!")
    except Exception as e:
        print(f"Critical error in main execution: {e}")
        import traceback
        traceback.print_exc()