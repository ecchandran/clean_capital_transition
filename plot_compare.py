import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import environment as env
import argparse
import sys
import torch
from matplotlib.colors import TwoSlopeNorm

# Set Seaborn style
sns.set_theme(style="whitegrid")
sns.set_context("paper", font_scale=1.2)

# Set up command line argument parsing
parser = argparse.ArgumentParser(description='Compare results from two models')
parser.add_argument('--tag_a', type=str, required=True,
                    help='Custom tag for model A')
parser.add_argument('--tag_b', type=str, required=True,
                    help='Custom tag for model B')
args = parser.parse_args()

TAG_A = args.tag_a
TAG_B = args.tag_b

# Determine model types
def get_model_type(tag):
    if os.path.exists(f"{tag}_M3DL"):
        return "M3DL", f"{tag}_M3DL"
    elif os.path.exists(f"{tag}_CPDL"):
        return "CPDL", f"{tag}_CPDL"
    elif os.path.exists(f"{tag}_CPFD"):
        return "CPFD", f"{tag}_CPFD"
    else:
        raise ValueError(f"No model directory found for tag {tag}")

MODEL_TYPE_A, OUTPUT_DIR_A = get_model_type(TAG_A)
MODEL_TYPE_B, OUTPUT_DIR_B = get_model_type(TAG_B)

# Create plot directory
COMPARE_DIR = f"compare_{TAG_A}_vs_{TAG_B}"
os.makedirs(COMPARE_DIR, exist_ok=True)

print(f"Comparing models: {TAG_A} ({MODEL_TYPE_A}) vs {TAG_B} ({MODEL_TYPE_B})")

def compare_value_functions():
    """Compare value functions between two models"""
    
    # Determine how to load value functions based on model types
    def load_value_function(model_type, output_dir, tag):
        """Safely load value function data with tensor handling"""
        # Convert tensors to numpy if needed
        def convert_tensor(data):
            if hasattr(data, 'device'):
                return data.cpu().numpy()
            elif hasattr(data, 'numpy'):
                return data.numpy()
            return data
        
        if model_type == "M3DL":
            # For M3 model, load W_effective from value_function_slices
            slice_file = f"{output_dir}/value_function_slices_{tag}.npz"
            if os.path.exists(slice_file):
                data = np.load(slice_file, allow_pickle=True)
                eta_grid = convert_tensor(data['eta_grid'])
                H_grid = convert_tensor(data['H_grid'])
                W_effective = convert_tensor(data['W_effective_eta_H'])
                return eta_grid, H_grid, W_effective
            else:
                raise FileNotFoundError(f"Value function slice file {slice_file} not found")
        elif model_type == "CPDL":
            # For CPDL model, check for slice file first, then grid evaluation
            slice_file = f"{output_dir}/value_function_slices_{tag}.npz"
            if os.path.exists(slice_file):
                data = np.load(slice_file, allow_pickle=True)
                eta_grid = convert_tensor(data['eta_grid'])
                H_grid = convert_tensor(data['H_grid'])
                W = convert_tensor(data['W_eta_H'])
                return eta_grid, H_grid, W
            
            grid_file = f"{output_dir}/grid_evaluation_{tag}.npz"
            if os.path.exists(grid_file):
                data = np.load(grid_file, allow_pickle=True)
                eta_mesh = convert_tensor(data['eta_mesh'])
                H_mesh = convert_tensor(data['H_mesh'])
                W_values = convert_tensor(data['W_values'])
                return eta_mesh[:,0], H_mesh[0,:], W_values
            
            raise FileNotFoundError(f"No value function data found for {tag}")
            
        elif model_type == "CPFD":
            # For CPFD model, load from results file
            results_file = f"{output_dir}/CP_FD_results_{tag}.npz"
            if os.path.exists(results_file):
                data = np.load(results_file, allow_pickle=True)
                eta_mesh = convert_tensor(data['eta_mesh'])
                H_mesh = convert_tensor(data['H_mesh'])
                W_sol = convert_tensor(data['W_sol'])
                return eta_mesh[:,0], H_mesh[0,:], W_sol
            else:
                raise FileNotFoundError(f"Results file {results_file} not found")
    try:
        # Load value functions
        eta_grid_a, H_grid_a, W_a = load_value_function(MODEL_TYPE_A, OUTPUT_DIR_A, TAG_A)
        eta_grid_b, H_grid_b, W_b = load_value_function(MODEL_TYPE_B, OUTPUT_DIR_B, TAG_B)
        
        # Check if grids are compatible, interpolate if needed
        grid_compatible = (len(eta_grid_a) == len(eta_grid_b) and 
                           len(H_grid_a) == len(H_grid_b) and
                           np.allclose(eta_grid_a, eta_grid_b) and
                           np.allclose(H_grid_a, H_grid_b))
        
        if not grid_compatible:
            # Need to interpolate to make comparable
            from scipy.interpolate import RectBivariateSpline
            
            # Determine common grid
            eta_common = np.linspace(max(eta_grid_a.min(), eta_grid_b.min()),
                                     min(eta_grid_a.max(), eta_grid_b.max()), 100)
            H_common = np.linspace(max(H_grid_a.min(), H_grid_b.min()),
                                  min(H_grid_a.max(), H_grid_b.max()), 100)
            
            # Create interpolation functions
            interp_a = RectBivariateSpline(eta_grid_a, H_grid_a, W_a)
            interp_b = RectBivariateSpline(eta_grid_b, H_grid_b, W_b)
            
            # Create meshgrid for evaluation
            eta_mesh, H_mesh = np.meshgrid(eta_common, H_common, indexing='ij')
            
            # Evaluate on common grid
            W_a_common = interp_a(eta_common, H_common)
            W_b_common = interp_b(eta_common, H_common)
            
            # Use common grid for comparison
            eta_grid = eta_common
            H_grid = H_common
            W_a = W_a_common
            W_b = W_b_common
        else:
            # Grids are already compatible
            eta_grid = eta_grid_a
            H_grid = H_grid_a
        
        # Create difference
        W_diff = W_b - W_a
        
        # Calculate difference statistics
        abs_diff = np.abs(W_diff)
        mean_abs_diff = np.mean(abs_diff)
        median_abs_diff = np.median(abs_diff)
        max_abs_diff = np.max(abs_diff)
        
        # Calculate relative differences
        W_a_range = np.max(W_a) - np.min(W_a)
        rel_diff = abs_diff / W_a_range
        mean_rel_diff = np.mean(rel_diff)
        median_rel_diff = np.median(rel_diff)
        
        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Plot W_a
        eta_mesh, H_mesh = np.meshgrid(eta_grid, H_grid, indexing='ij')
        contour = axes[0].contourf(eta_mesh, H_mesh, W_a, 50, cmap='viridis')
        plt.colorbar(contour, ax=axes[0], label=f'W_{TAG_A}')
        axes[0].set_xlabel('Green Capital Share η')
        axes[0].set_ylabel('Climate Damage H')
        axes[0].set_title(f'Value Function - Model {TAG_A}')
        axes[0].grid(True, linestyle='--', alpha=0.3)
        
        # Plot W_b
        contour = axes[1].contourf(eta_mesh, H_mesh, W_b, 50, cmap='viridis')
        plt.colorbar(contour, ax=axes[1], label=f'W_{TAG_B}')
        axes[1].set_xlabel('Green Capital Share η')
        axes[1].set_ylabel('Climate Damage H')
        axes[1].set_title(f'Value Function - Model {TAG_B}')
        axes[1].grid(True, linestyle='--', alpha=0.3)
        
        # Create symmetric diverging normalization around 0
        max_val = max(abs(W_diff.min()), abs(W_diff.max()))
        norm = TwoSlopeNorm(vmin=-max_val, vcenter=0, vmax=max_val)
        
        # Plot difference
        contour = axes[2].contourf(eta_mesh, H_mesh, W_diff, 50, cmap='bwr', norm=norm)
        plt.colorbar(contour, ax=axes[2], label=f'W_{TAG_B} - W_{TAG_A}')
        axes[2].set_xlabel('Green Capital Share η')
        axes[2].set_ylabel('Climate Damage H')
        axes[2].set_title(f'Difference: W_{TAG_B} - W_{TAG_A}')
        axes[2].grid(True, linestyle='--', alpha=0.3)
        
        # Add statistics to the figure
        stats_text = (
            f"Mean Absolute Diff: {mean_abs_diff:.4f}\n"
            f"Median Absolute Diff: {median_abs_diff:.4f}\n"
            f"Max Absolute Diff: {max_abs_diff:.4f}\n"
            f"Mean Relative Diff: {mean_rel_diff:.4f}\n"
            f"Median Relative Diff: {median_rel_diff:.4f}"
        )
        
        plt.figtext(0.5, 0.01, stats_text, ha="center", fontsize=12, 
                   bbox={"facecolor":"white", "alpha":0.8, "pad":5})
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15)
        plt.savefig(f"{COMPARE_DIR}/value_function_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Value function comparison saved to {COMPARE_DIR}/value_function_comparison.png")
        
        # Save statistics to file
        with open(f"{COMPARE_DIR}/value_function_comparison_stats.txt", 'w') as f:
            f.write(f"Value Function Comparison: {TAG_A} vs {TAG_B}\n")
            f.write(f"Mean Absolute Difference: {mean_abs_diff:.6f}\n")
            f.write(f"Median Absolute Difference: {median_abs_diff:.6f}\n")
            f.write(f"Maximum Absolute Difference: {max_abs_diff:.6f}\n")
            f.write(f"Value Function A Range: {W_a_range:.6f}\n")
            f.write(f"Mean Relative Difference: {mean_rel_diff:.6f}\n")
            f.write(f"Median Relative Difference: {median_rel_diff:.6f}\n")
        
    except Exception as e:
        print(f"Error comparing value functions: {e}")
        import traceback
        traceback.print_exc()

def compare_simulations():
    """Compare simulation results between two models with improved error handling"""
    # Define variable ordering and labels
    var_order = {
        'eta': ('Green Capital Share η', 0),
        'H': ('Climate Damage H', 1), 
        'K': ('Aggregate Capital K', 2),
        'c': ('Consumption c', 3),
        'iota_1': ('Green Investment Rate ι₁', 4),
        'iota_2': ('Brown Investment Rate ι₂', 5),
        'tau': ('Carbon Tax τ', 6),
        'value': ('Value Function V', 7)
    }
    
    # Load simulation results safely
    def load_simulation(model_type, output_dir, tag):
        sim_file = f"{output_dir}/simulation_results_{tag}.npz"
        if os.path.exists(sim_file):
            # Load the npz file and convert to a mutable dictionary
            npz_file = np.load(sim_file, allow_pickle=True)
            # Convert to dictionary so we can add new keys
            sim_dict = {key: npz_file[key] for key in npz_file.files}
            return sim_dict
        else:
            print(f"Simulation file {sim_file} not found")
            return None
    
    try:
        # Load simulation results
        sim_a = load_simulation(MODEL_TYPE_A, OUTPUT_DIR_A, TAG_A)
        sim_b = load_simulation(MODEL_TYPE_B, OUTPUT_DIR_B, TAG_B)
        
        if sim_a is None or sim_b is None:
            print("Cannot compare simulations: one or both simulation files missing")
            return
        
        # Compute q1 and q2 for M3DL models if not already present
        def compute_q_from_iota(sim_data):
            """Calculate capital prices from investment rates"""
            if 'iota_1' in sim_data and 'iota_2' in sim_data:
                q1 = 1 + env.phi * sim_data['iota_1']
                q2 = 1 + env.phi * sim_data['iota_2']
                return q1, q2
            return None, None

        # Add q1, q2 for Model A if M3DL
        if MODEL_TYPE_A == "M3DL" and 'q1' not in sim_a and 'q2' not in sim_a:
            q1_results_a, q2_results_a = compute_q_from_iota(sim_a)
            if q1_results_a is not None and q2_results_a is not None:
                sim_a['q1'] = q1_results_a
                sim_a['q2'] = q2_results_a

        # Add q1, q2 for Model B if M3DL
        if MODEL_TYPE_B == "M3DL" and 'q1' not in sim_b and 'q2' not in sim_b:
            q1_results_b, q2_results_b = compute_q_from_iota(sim_b)
            if q1_results_b is not None and q2_results_b is not None:
                sim_b['q1'] = q1_results_b
                sim_b['q2'] = q2_results_b
        
        # Find common variables between the two simulations AFTER computing derived variables
        common_vars = []
        for var_name in var_order:
            # Skip tau if one model is a central planner
            if var_name == 'tau' and (MODEL_TYPE_A == 'CPDL' or MODEL_TYPE_B == 'CPDL'):
                continue
                
            if var_name in sim_a and var_name in sim_b:
                common_vars.append((var_name, var_order[var_name][0], var_order[var_name][1]))
        
        # Now add q1 and q2 to common_vars ONLY if both models have them
        if 'q1' in sim_a and 'q1' in sim_b:
            common_vars.append(('q1', 'Green Capital Price q1', 8))
        
        if 'q2' in sim_a and 'q2' in sim_b:
            common_vars.append(('q2', 'Brown Capital Price q2', 9))
                
        # Use the smaller number of time steps
        dt = min(float(sim_a['dt']), float(sim_b['dt']))
        N_T = min(int(sim_a['N_T']), int(sim_b['N_T']))
        
        # Create time array
        time = np.arange(0, (N_T + 1) * dt, dt)
        
        # Sort variables by their order
        common_vars.sort(key=lambda x: x[2])
        
        # Create a figure with the appropriate number of subplots
        num_plots = len(common_vars)
        num_rows = (num_plots + 2) // 3  # At most 3 columns
        fig, axes = plt.subplots(num_rows, 3, figsize=(18, 6*num_rows))
        fig.suptitle(f'Comparison of Simulation Results: {TAG_A} (blue) vs {TAG_B} (green)', fontsize=16)
        
        # Flatten axes for easier indexing
        axes = axes.flatten()
        
        # Plot each variable
        for i, (var_name, var_label, _) in enumerate(common_vars):
            if i >= len(axes):
                print(f"Warning: Not enough subplots for all variables. Skipping {var_name}.")
                continue
                
            # Get the axis for this subplot
            ax = axes[i]
            
            # Get data for both models
            data_a = sim_a[var_name]
            data_b = sim_b[var_name]
            
            # Calculate percentiles for both models
            p_data_a = np.percentile(data_a, env.sim_percentiles, axis=0)
            p_data_b = np.percentile(data_b, env.sim_percentiles, axis=0)
            
            # Create colormap for fan charts
            n_bands = len(env.sim_percentiles) // 2
            colors_a = plt.cm.Blues(np.linspace(0.4, 0.8, n_bands))
            colors_b = plt.cm.Greens(np.linspace(0.4, 0.8, n_bands))
            
            # Plot fan chart for model A
            for j in range(n_bands):
                upper_idx = len(env.sim_percentiles) - j - 1
                lower_idx = j
                ax.fill_between(time, p_data_a[lower_idx], p_data_a[upper_idx], 
                               color=colors_a[j], alpha=0.4)
            
            # Plot fan chart for model B
            for j in range(n_bands):
                upper_idx = len(env.sim_percentiles) - j - 1
                lower_idx = j
                ax.fill_between(time, p_data_b[lower_idx], p_data_b[upper_idx], 
                               color=colors_b[j], alpha=0.4)
            
            # Plot medians
            ax.plot(time, p_data_a[len(env.sim_percentiles)//2], 'b-', linewidth=2.5, 
                   label=f'Median {TAG_A}')
            ax.plot(time, p_data_b[len(env.sim_percentiles)//2], 'g-', linewidth=2.5, 
                   label=f'Median {TAG_B}')
            
            # Plot means
            ax.plot(time, np.mean(data_a, axis=0), 'b--', linewidth=1.5, 
                   label=f'Mean {TAG_A}')
            ax.plot(time, np.mean(data_b, axis=0), 'g--', linewidth=1.5, 
                   label=f'Mean {TAG_B}')
            
            # Set labels and title
            ax.set_xlabel('Time')
            ax.set_ylabel(var_label)
            ax.set_title(var_label)
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # Fix subscript labels
            fix_subscript_labels(ax)
            
            # Add legend only to the first subplot
            if i == 0:
                ax.legend()
        
        # Remove empty subplots
        for i in range(len(common_vars), len(axes)):
            fig.delaxes(axes[i])
        
        # Adjust layout
        plt.tight_layout()
        plt.subplots_adjust(top=0.93)
        
        # Save the combined figure
        plt.savefig(f"{COMPARE_DIR}/all_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Combined comparison plot saved to {COMPARE_DIR}/all_comparison.png")
                
    except Exception as e:
        print(f"Error comparing simulations: {e}")
        import traceback
        traceback.print_exc()



# In both plot files, add this helper function:
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

if __name__ == "__main__":
    try:
        print(f"Starting comparison of {TAG_A} vs {TAG_B}...")
        
        # Compare value functions
        compare_value_functions()
        
        # Compare simulations
        compare_simulations()
        
        print("Comparison completed successfully!")
    except Exception as e:
        print(f"Critical error in comparison: {e}")
        import traceback
        traceback.print_exc()