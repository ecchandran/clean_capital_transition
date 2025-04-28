import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.stats import norm
import os
import environment as env
import argparse
import time
from scipy.interpolate import RegularGridInterpolator

# Parse command line arguments
parser = argparse.ArgumentParser(description='Run central planner finite difference model')
parser.add_argument('--tag', type=str, required=True,
                    help='Custom tag for saved files')
args = parser.parse_args()

TAG = args.tag
OUTPUT_DIR = f"{TAG}_CPFD"

# Create directory structure if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

def build_grid(eta_min, eta_max, N_eta, H_min, H_max, N_H):
    eta_grid = np.linspace(eta_min, eta_max, N_eta)
    d_eta = eta_grid[1] - eta_grid[0]
    H_grid = np.linspace(H_min, H_max, N_H)
    dH = H_grid[1] - H_grid[0]
    eta, H = np.meshgrid(eta_grid, H_grid, indexing="ij") # eta changes on axis 0, H changes on axis 1
    dt = 0.1 * min(d_eta**2, dH ** 2)  # Showed in HW that dt must scale with square of spatial step size.
    return eta, H, d_eta, dH, dt, eta_grid, H_grid

def get_W_eta(W, d_eta):
    """Return forward step approximation of partial_eta W for computing iota_1, iota_2"""
    W_eta = (np.roll(W, -1, axis=0) - W) / d_eta
    W_eta[-1, :] = W_eta[-2, :]
    return W_eta

def get_i1_mid(W_eta_mid, eta_mid, H_mid):
    """Input shapes must match, eta cannot be zero or 1 anywhere"""
    num = -(env.phi + env.beta) + (W_eta_mid * (1 - eta_mid) +
                       env.beta) * (1 + env.phi * (eta_mid * env.A_k1(H_mid) +
                                           (1 - eta_mid) * env.A_k2(H_mid)))
    denom = env.phi * (env.phi + env.beta)
    return num / denom

def get_i2_mid(i1_mid, eta_mid, H_mid):
    """Input shapes must match, eta cannot be 1 anywhere"""
    C_1 = -eta_mid / (1 - eta_mid)
    C_0 = (-1 + env.beta * (eta_mid * env.A_k1(H_mid) +
                    (1 - eta_mid) * env.A_k2(H_mid))) / ((1 - eta_mid) *
                                                     (env.phi + env.beta))
    return i1_mid * C_1 + C_0

def get_c_per_k(iota_1, iota_2, eta, H):
    """Return consumption per unit capital using goods market clearing
    CLIP TO POSITIVE VALUE"""
    result = (env.A_k1(H) - iota_1) * eta + (env.A_k2(H) - iota_2) * (1 - eta)
    return np.clip(result, 0, None)  # Ensure positive consumption

def get_iotas(W_eta, eta, H):
    """ASSUME eta 0, 1 at boundaries, return iota_1, iota_2 given W_eta'"""
    iota_1 = np.ones(eta.shape)
    iota_2 = np.ones(eta.shape)
    if np.min(eta) == 0:
        iota_1[-1, :] = (env.beta * env.A_k1(H[-1, :]) - 1) / (
            env.phi + env.beta)  # boundary 1's irrelevant as never used
        iota_1[1:-1, :] = get_i1_mid(W_eta[1:-1, :], eta[1:-1, :], H[1:-1, :])
        # however, set boundary to nearest neighbor for plotting 
        iota_1[0, :] = iota_1[1, :]
    else:
        iota_1 = get_i1_mid(W_eta, eta, H)
    if np.max(eta) == 1:
        iota_2[0, :] = (env.beta * env.A_k2(H[0, :]) - 1) / (env.phi + env.beta)
        iota_2[1:-1, :] = get_i2_mid(iota_1[1:-1, :], eta[1:-1, :], H[1:-1, :])
        iota_2[-1, :] = iota_2[-2, :]
    else:
        iota_2 = get_i2_mid(iota_1, eta, H)
    return iota_1, iota_2

def get_mu_eta(iota_1, iota_2, eta):
    return eta * (1 - eta) * (env.Phi(iota_1) - env.Phi(iota_2))

def get_mu_H(eta, H):
    return -env.eps * H + env.mu_2 * (1 - eta)

def build_U(iota_1, iota_2, eta, H):
    return np.log(
        np.clip(get_c_per_k(iota_1, iota_2, eta, H),
                a_min=env.FD_LOG_ARG_CLIP,
                a_max=None)) + env.beta * (eta * env.Phi(iota_1) +
                                   (1 - eta) * env.Phi(iota_2) - env.delta)

def policy_iteration_improved(W_guess, eta, H, d_eta, dH, alpha=0.5, policy_damping=0.5):
    """
    Solve using policy iteration with relaxation parameters to improve stability
    
    Parameters:
    - W_guess: Initial guess for the value function
    - eta, H: State variable grids
    - d_eta, dH: Grid spacings
    - alpha: Relaxation parameter for value function updates (0 < alpha <= 1)
    - policy_damping: Damping parameter for policy updates (0 < policy_damping <= 1)
    """
    W_curr = W_guess.copy()
    old_iota_1 = None
    old_iota_2 = None
    err = 1
    err_history = []
    policy_err_history = []
    
    for i in range(env.max_iter):
        # 1. Get optimal policy using current value function
        W_eta = get_W_eta(W_curr, d_eta)
        iota_1, iota_2 = get_iotas(W_eta, eta, H)
        
        # Apply policy damping if not first iteration
        if old_iota_1 is not None and old_iota_2 is not None:
            iota_1 = policy_damping * iota_1 + (1 - policy_damping) * old_iota_1
            iota_2 = policy_damping * iota_2 + (1 - policy_damping) * old_iota_2
        
        old_iota_1, old_iota_2 = iota_1.copy(), iota_2.copy()
        
        # 2. Solve for value function given fixed policy with boundary adjustments
        W_next = solve_hjb_fixed_policy_improved(W_curr, iota_1, iota_2, eta, H, d_eta, dH)
        
        # 3. Apply relaxation to value function update
        W_next = alpha * W_next + (1 - alpha) * W_curr
        
        # 4. Check convergence
        err = np.max(np.abs(W_next - W_curr))
        err_history.append(err)
        
        # Calculate policy error for diagnostics
        if i > 0:
            policy_err = np.max(np.abs(iota_1 - old_iota_1)) + np.max(np.abs(iota_2 - old_iota_2))
            policy_err_history.append(policy_err)
        
        if err < env.conv_thresh:
            print(f"Policy iteration converged at iter {i+1}!")
            break
        elif i % env.print_every == 0:
            print(f"Error after iter {i+1}: {err}")
            if i > 0:
                print(f"Policy error: {policy_err}")
        
        W_curr = W_next.copy()  # Ensure we're making a copy
    
    if i == env.max_iter - 1:
        print(f"Warning: Policy iteration did not converge after {env.max_iter} iterations. Final error: {err}")
    
    return W_curr, err_history, policy_err_history
    
def build_A(W_curr, iota_1, iota_2, eta, H, d_eta, d_H):
    """Solve HJB equation with fixed policy (iota_1, iota_2) with improved boundary handling"""
    # Compute utility with fixed policy
    c = get_c_per_k(iota_1, iota_2, eta, H)
    utility = np.log(np.clip(c, env.FD_LOG_ARG_CLIP, None))
    
    # Compute capital growth
    k_growth = eta * env.Phi(iota_1) + (1 - eta) * env.Phi(iota_2) - env.delta
    
    # Compute drifts with fixed policy
    mu_eta = get_mu_eta(iota_1, iota_2, eta)
    mu_H = get_mu_H(eta, H)
    
    # Set up linear system A x = b
    N_eta, N_H = eta.shape
    N = N_eta * N_H
    
    # Setup indicators for upwinding
    s_plus = np.where(mu_eta >= 0, 1, 0)
    s_plus[0, :] = 1
    s_plus[-1, :] = 0
    s_minus = 1 - s_plus
    
    q_plus = np.where(mu_H >= 0, 1, 0)
    q_plus[:, 0] = 1
    q_plus[:, -1] = 0
    q_minus = 1 - q_plus
    
    # Improved handling of diffusion terms
    p_0 = np.ones(eta.shape)
    p_0[:, 0] = 0   # Boundary condition at H_min
    p_0[:, -1] = 0  # Boundary condition at H_max
    p_plus = np.zeros(eta.shape)
    p_plus[:, 0] = 1
    p_minus = np.zeros(eta.shape)
    p_minus[:, -1] = 1
    
    # Flatten arrays
    mu_eta_f = mu_eta.flatten()
    mu_H_f = mu_H.flatten()
    H_f = H.flatten()
    utility_f = utility.flatten()
    k_growth_f = k_growth.flatten()
    
    s_plus_f = s_plus.flatten()
    s_minus_f = s_minus.flatten()
    q_plus_f = q_plus.flatten()
    q_minus_f = q_minus.flatten()
    p_0_f = p_0.flatten()
    p_plus_f = p_plus.flatten()
    p_minus_f = p_minus.flatten()
    
    # Build sparse matrix diagonals with improved stability
    diffusion_factor = (H_f * env.sigma_H)**2 / (2 * d_H**2)

    
    diag_00 = env.rho - (mu_eta_f * (-s_plus_f + s_minus_f) / d_eta + 
                     mu_H_f * (-q_plus_f + q_minus_f) / d_H - 
                     diffusion_factor * (p_plus_f - 2 * p_0_f + p_minus_f))
    
    diag_u0 = -mu_eta_f * s_plus_f / d_eta
    diag_d0 = mu_eta_f * s_minus_f / d_eta
    
    diag_0uu = -diffusion_factor * p_plus_f
    diag_0u = -(mu_H_f * q_plus_f / d_H + 
               diffusion_factor * (-2 * p_plus_f + p_0_f))
    diag_0d = (mu_H_f * q_minus_f / d_H - 
              diffusion_factor * (p_0_f - 2 * p_minus_f))
    diag_0dd = -diffusion_factor * p_minus_f

    
    # Roll diagonals to proper positions
    diag_u0 = np.roll(diag_u0, N_H)
    diag_d0 = np.roll(diag_d0, -N_H)
    diag_0uu = np.roll(diag_0uu, 2)
    diag_0u = np.roll(diag_0u, 1)
    diag_0d = np.roll(diag_0d, -1)
    diag_0dd = np.roll(diag_0dd, -2)

    A = sparse.spdiags(
        [diag_u0, diag_0uu, diag_0u, diag_00, diag_0d, diag_0dd, diag_d0],
        [N_H, 2, 1, 0, -1, -2, -N_H], N, N, format="csr" 
    )
    return A
    
def solve_hjb_fixed_policy_improved(W_curr, iota_1, iota_2, eta, H, d_eta, d_H):
    """Solve HJB equation with fixed policy (iota_1, iota_2) with improved boundary handling"""
    # Compute utility with fixed policy
    c = get_c_per_k(iota_1, iota_2, eta, H)
    utility = np.log(np.clip(c, env.FD_LOG_ARG_CLIP, None))
    
    # Compute capital growth
    k_growth = eta * env.Phi(iota_1) + (1 - eta) * env.Phi(iota_2) - env.delta
    
    # Set up linear system A x = b
    N_eta, N_H = eta.shape
    N = N_eta * N_H
  
    # Flatten arrays
    utility_f = utility.flatten()
    k_growth_f = k_growth.flatten()

    A = build_A(W_curr, iota_1, iota_2, eta, H, d_eta, d_H)
    
    # Right-hand side: flow utility + capital growth utility
    b = utility_f + env.beta * k_growth_f

    # Solve linear system with error checking
    try:
        W_next_f = sparse.linalg.spsolve(A, b)
        # Check for NaN or inf values
        if np.any(np.isnan(W_next_f)) or np.any(np.isinf(W_next_f)):
            print("Warning: NaN or inf values in solution, reverting to previous iteration")
            return W_curr
    except Exception as e:
        print(f"Error in sparse solver: {e}")
        return W_curr
    
    W_next = np.reshape(W_next_f, (N_eta, N_H))
    
    return W_next

def analyze_solution(W, eta, H, d_eta, d_H):
    """Analyze the solution for potential issues"""
    # Check gradients
    W_eta = get_W_eta(W, d_eta)
    
    # Get investments and check if they're reasonable
    iota_1, iota_2 = get_iotas(W_eta, eta, H)
    
    if np.any(np.isnan(iota_1)) or np.any(np.isnan(iota_2)):
        print("Warning: NaN investment rates detected!")
    
    # Check drifts
    mu_eta = get_mu_eta(iota_1, iota_2, eta)
    mu_H = get_mu_H(eta, H)
    
    print(f"Drift ranges: mu_eta: [{np.min(mu_eta)}, {np.max(mu_eta)}], mu_H: [{np.min(mu_H)}, {np.max(mu_H)}]")
    
    # Visualize value function and its derivatives
    plt.figure(figsize=(15, 10))
    
    # Value function plot
    plt.subplot(221)
    W_min, W_max = np.min(W), np.max(W)
    plt.pcolormesh(H[0], eta[:, 0], W, shading='auto', cmap='viridis')
    plt.colorbar(label=f'Value Function [{W_min:.2f}, {W_max:.2f}]')
    plt.xlabel('H')
    plt.ylabel('eta')
    plt.title('Value Function W(eta, H)')
    
    # Value function gradient plot
    plt.subplot(222)
    W_eta_min, W_eta_max = np.min(W_eta), np.max(W_eta)
    plt.pcolormesh(H[0], eta[:, 0], W_eta, shading='auto', cmap='bwr')
    plt.colorbar(label=f'dW/deta [{W_eta_min:.2f}, {W_eta_max:.2f}]')
    plt.xlabel('H')
    plt.ylabel('eta')
    plt.title('Value Function Gradient dW/deta')
    
    # Green investment rate plot
    plt.subplot(223)
    iota1_min, iota1_max = np.min(iota_1), np.max(iota_1)
    plt.pcolormesh(H[0], eta[:, 0], iota_1, shading='auto', cmap='viridis')
    plt.colorbar(label=f'iota_1 [{iota1_min:.2f}, {iota1_max:.2f}]')
    plt.xlabel('H')
    plt.ylabel('eta')
    plt.title('Green Investment Rate (iota_1)')
    
    # Brown investment rate plot
    plt.subplot(224)
    iota2_min, iota2_max = np.min(iota_2), np.max(iota_2)
    plt.pcolormesh(H[0], eta[:, 0], iota_2, shading='auto', cmap='viridis')
    plt.colorbar(label=f'iota_2 [{iota2_min:.2f}, {iota2_max:.2f}]')
    plt.xlabel('H')
    plt.ylabel('eta')
    plt.title('Brown Investment Rate (iota_2)')
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/value_function_analysis_{TAG}.png")
    plt.close()
    
    # Plot drifts
    plt.figure(figsize=(15, 5))
    
    # Drift in eta plot
    plt.subplot(121)
    mu_eta_min, mu_eta_max = np.min(mu_eta), np.max(mu_eta)
    plt.pcolormesh(H[0], eta[:, 0], mu_eta, shading='auto', cmap='bwr')
    plt.colorbar(label=f'mu_eta [{mu_eta_min:.2f}, {mu_eta_max:.2f}]')
    plt.xlabel('H')
    plt.ylabel('eta')
    plt.title('Drift in eta')
    
    # Drift in H plot
    plt.subplot(122)
    mu_H_min, mu_H_max = np.min(mu_H), np.max(mu_H)
    plt.pcolormesh(H[0], eta[:, 0], mu_H, shading='auto', cmap='bwr')
    plt.colorbar(label=f'mu_H [{mu_H_min:.2f}, {mu_H_max:.2f}]')
    plt.xlabel('H')
    plt.ylabel('eta')
    plt.title('Drift in H')
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/drift_analysis_{TAG}.png")
    plt.close()
    
    return iota_1, iota_2, mu_eta, mu_H

# ------------ New simulation functions (similar to cp_dl.py) ------------

def F(x, eta_n, iota_1n, iota_2n, dt):
    """Helper function for eta update using Newton's method"""
    return x * (1 - x) * (env.Phi(iota_1n) - env.Phi(iota_2n)) * dt - x + eta_n

def F_x(x, iota_1n, iota_2n, dt):
    """Derivative of F with respect to x"""
    return (1 - 2 * x) * (env.Phi(iota_1n) - env.Phi(iota_2n)) * dt - 1

def eta_next(eta_n, iota_1, iota_2, dt):
    """Calculate next eta value using Newton's method"""
    try:
        return eta_n - F(eta_n, eta_n, iota_1, iota_2, dt) / F_x(eta_n, iota_1, iota_2, dt)
    except:
        # Fallback if Newton's method fails
        return eta_n + eta_n * (1 - eta_n) * (env.Phi(iota_1) - env.Phi(iota_2)) * dt

def H_next(H_n, eta_n, dt, dW, eps, mu_2, sigma_H):
    """Calculate next H value"""
    return (H_n + mu_2 * (1 - eta_n) * dt) / (1 + eps * dt + sigma_H * dW)

def K_next(K, eta, iota_1, iota_2, dt, delta):
    """Calculate next K value"""
    dK = dt * K * (eta * env.Phi(iota_1) + (1 - eta) * env.Phi(iota_2) - delta)
    return K + dK

def create_interpolators(eta_mesh, H_mesh, W_sol, W_eta_sol):
    """Create interpolators for value function and its derivative"""
    eta_grid = eta_mesh[:, 0]
    H_grid = H_mesh[0, :]
    
    # Create interpolators
    W_interp = RegularGridInterpolator((eta_grid, H_grid), W_sol, 
                                      bounds_error=False, fill_value=None)
    W_eta_interp = RegularGridInterpolator((eta_grid, H_grid), W_eta_sol, 
                                          bounds_error=False, fill_value=None)
    
    return W_interp, W_eta_interp

def get_optimal_controls(eta_t, H_t, W_eta_interp):
    """Get optimal controls for a given state using the interpolated W_eta"""
    # Handle boundary cases
    if np.isclose(eta_t, 0, atol=1e-6):
        iota_1 = (env.beta * env.A_k1(H_t) - 1) / (env.phi + env.beta)
        iota_2 = (env.beta * env.A_k2(H_t) - 1) / (env.phi + env.beta)
        return iota_1, iota_2
    
    elif np.isclose(eta_t, 1, atol=1e-6):
        iota_1 = (env.beta * env.A_k1(H_t) - 1) / (env.phi + env.beta)
        iota_2 = 0  # Arbitrary as it's not used
        return iota_1, iota_2
    
    # Get W_eta at the current state
    W_eta = W_eta_interp([eta_t, H_t])[0]
    
    # Calculate iota_1
    numerator = -(env.phi + env.beta) + (W_eta * (1-eta_t) + env.beta) * \
                (1 + env.phi * (eta_t * env.A_k1(H_t) + (1-eta_t) * env.A_k2(H_t)))
    denominator = env.phi * (env.phi + env.beta)
    iota_1 = numerator / denominator
    
    # Calculate iota_2
    C_1 = -eta_t / (1 - eta_t)
    C_0 = (-1 + env.beta * (eta_t * env.A_k1(H_t) + (1-eta_t) * env.A_k2(H_t))) / \
          ((1-eta_t) * (env.phi + env.beta))
    iota_2 = iota_1 * C_1 + C_0
    
    return iota_1, iota_2

def propagate_economy_fd(eta_0, H_0, K_0, dt, N_T, W_eta_interp, 
                       eps=env.eps, mu_2=env.mu_2, sigma_H=env.sigma_H, delta=env.delta):
    """
    Propagate economy forward using finite difference solution
    
    Parameters:
    -----------
    eta_0, H_0, K_0 : float
        Initial state variables
    dt : float
        Time step
    N_T : int
        Number of time steps
    W_eta_interp : callable
        Interpolator for the gradient of value function
    eps, mu_2, sigma_H, delta : float
        Model parameters
    
    Returns:
    --------
    tuple
        Time series for state variables and controls
    """
    # Initialize arrays
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
        
        # Get optimal controls through interpolation
        iota_1, iota_2 = get_optimal_controls(eta_t, H_t, W_eta_interp)
        
        # Store controls
        iota_1_sim[t] = iota_1
        iota_2_sim[t] = iota_2
        
        # Calculate consumption
        c_sim[t] = K_t * ((env.A_k1(H_t) - iota_1) * eta_t + (env.A_k2(H_t) - iota_2) * (1 - eta_t))
        
        # Update state variables
        eta_sim[t+1] = eta_next(eta_t, iota_1, iota_2, dt)
        eta_sim[t+1] = np.clip(eta_sim[t+1], 0, 1)  # Ensure bounds are respected
        
        H_sim[t+1] = H_next(H_t, eta_t, dt, dW_values[t], eps, mu_2, sigma_H)
        H_sim[t+1] = max(H_sim[t+1], 0)  # Ensure H remains non-negative
        
        K_sim[t+1] = K_next(K_t, eta_t, iota_1, iota_2, dt, delta)
    
    # Calculate final step controls/consumption
    eta_t = eta_sim[-1]
    H_t = H_sim[-1]
    K_t = K_sim[-1]
    
    # Get optimal controls
    iota_1, iota_2 = get_optimal_controls(eta_t, H_t, W_eta_interp)
    
    iota_1_sim[-1] = iota_1
    iota_2_sim[-1] = iota_2
    c_sim[-1] = K_t * ((env.A_k1(H_t) - iota_1) * eta_t + (env.A_k2(H_t) - iota_2) * (1 - eta_t))
    
    return (eta_sim, H_sim, K_sim, c_sim, iota_1_sim, iota_2_sim)

def run_monte_carlo_simulations_fd(eta_0, H_0, K_0, dt, N_T, W_eta_interp, num_sims=10, 
                                 eps=env.eps, mu_2=env.mu_2, sigma_H=env.sigma_H, 
                                 delta=env.delta, show_progress=True):
    """
    Run multiple simulations with finite difference solution
    
    Parameters:
    -----------
    eta_0, H_0, K_0 : float
        Initial state variables
    dt : float
        Time step
    N_T : int
        Number of time steps
    W_eta_interp : callable
        Interpolator for the gradient of value function
    num_sims : int
        Number of simulations to run
    eps, mu_2, sigma_H, delta : float
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
        eta_sim, H_sim, K_sim, c_sim, iota_1_sim, iota_2_sim = propagate_economy_fd(
            eta_0, H_0, K_0, dt, N_T, W_eta_interp, eps, mu_2, sigma_H, delta
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

def simulate_economy_fd(W_sol, W_eta_sol, eta_mesh, H_mesh):
    """Run Monte Carlo simulations of the economy using the FD solution"""
    try:
        print("Starting economy simulations...")
        
        # Create interpolators for the value function and its derivative
        W_interp, W_eta_interp = create_interpolators(eta_mesh, H_mesh, W_sol, W_eta_sol)
        
        print("Running simulations...")
        
        # Run simulations with parameters from environment
        results = run_monte_carlo_simulations_fd(
            eta_0=env.sim_eta_0,
            H_0=env.sim_H_0,
            K_0=env.sim_K_0,
            dt=env.sim_dt,
            N_T=env.sim_N_T,
            W_eta_interp=W_eta_interp,
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
        print(f"Error in simulate_economy_fd: {e}")
        import traceback
        traceback.print_exc()

def main():
    # Create grid
    eta, H, d_eta, dH, dt, eta_grid, H_grid = build_grid(
        env.eta_min, env.eta_max, env.N_eta, 
        env.H_min, env.H_max, env.N_H
    )
    
    # Initial guess
    W_guess = env.initial_guess(eta, H)
    
    # Solve
    print("Starting policy iteration...")
    W_sol, err_hist, pol_err_hist = policy_iteration_improved(
        W_guess, eta, H, d_eta, dH, alpha=0.5, policy_damping=0.5
    )
    
    # Analyze solution
    print("Analyzing solution...")
    iota_1_sol, iota_2_sol, mu_eta, mu_H = analyze_solution(W_sol, eta, H, d_eta, dH)
    
    # Compute gradient
    W_eta_sol = get_W_eta(W_sol, d_eta)
    
    # Save results to files for later plotting
    np.savez(
        f"{OUTPUT_DIR}/CP_FD_results_{TAG}.npz",
        eta_mesh=eta,
        H_mesh=H,
        W_sol=W_sol,
        W_eta_sol=W_eta_sol,
        iota_1=iota_1_sol,
        iota_2=iota_2_sol,
        mu_eta=mu_eta,
        mu_H=mu_H
    )
    
    print(f"Solution saved to {OUTPUT_DIR}/CP_FD_results_{TAG}.npz")
    print(f"Analytic steady state value at eta=1, H=0: {env.W_m1H0()}")
    print(f"Finite difference solution value at eta=1, H=0: {W_sol[-1, 0]}")
    
    # Run simulations with the computed solution
    print("Running simulations...")
    simulate_economy_fd(W_sol, W_eta_sol, eta, H)

if __name__ == "__main__":
    main()
    print("done")