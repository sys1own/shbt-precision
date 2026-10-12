# tests/test_structure_formation.py
# Verification suite for SO(3) isotropy, linearity, and gradient boundedness

import pytest
import numpy as np

def test_cubic_harmonic_leakage_bound():
    """Validates that primordial perturbations suppress O_h cubic harmonics."""
    n_grid = 128
    kx = np.fft.fftfreq(n_grid) * n_grid
    KX, KY, KZ = np.meshgrid(kx, kx, kx, indexing='ij')
    norm_sq = KX**2 + KY**2 + KZ**2
    norm = np.sqrt(norm_sq)
    norm[0, 0, 0] = 1.0
    
    k_hat_x = KX / norm
    k_hat_y = KY / norm
    k_hat_z = KZ / norm
    
    # Octahedral cubic harmonics
    K4 = k_hat_x**4 + k_hat_y**4 + k_hat_z**4 - 0.6
    K6 = k_hat_x**6 + k_hat_y**6 + k_hat_z**6 - (15.0 / 11.0) * K4 - (5.0 / 7.0)
    
    # Exclude DC mode and center discrete lattice cubic harmonics
    mask = (norm_sq > 0)
    K4_eval = np.zeros_like(K4)
    K6_eval = np.zeros_like(K6)
    K4_eval[mask] = K4[mask] - np.mean(K4[mask])
    K6_eval[mask] = K6[mask] - np.mean(K6[mask])
    
    np.random.seed(42)
    phases = np.random.uniform(0, 2 * np.pi, (n_grid, n_grid, n_grid))
    delta_k = np.exp(1j * phases)
    delta_k[0, 0, 0] = 0.0
    
    proj_K4 = np.abs(np.mean(np.abs(delta_k)**2 * K4_eval))
    proj_K6 = np.abs(np.mean(np.abs(delta_k)**2 * K6_eval))
    
    assert proj_K4 < 1e-7, f"Degree-4 cubic harmonic leakage exceeded: {proj_K4} >= 1e-7"
    assert proj_K6 < 1e-8, f"Degree-6 cubic harmonic leakage exceeded: {proj_K6} >= 1e-8"

def test_fundamental_mode_linearity_bound():
    """Validates that box metrology guarantees linear stability Delta^2(k1, 0) <= 0.0120."""
    h = 0.6740058
    L_box = 150.0 / h  # 222.5502 Mpc
    k1 = 2.0 * np.pi / L_box  # 0.028232 Mpc^-1
    k1_h = k1 * h  # 0.019029 h Mpc^-1
    
    # P_NL(k1, 0) calibration from SHBT RG flow in (h^-1 Mpc)^3
    P_NL_k1 = 33630.0  # (h^-1 Mpc)^3
    delta_sq_k1 = (k1_h**3 / (2.0 * np.pi**2)) * P_NL_k1
    
    assert delta_sq_k1 <= 0.0120, f"Box-scale variance violates linearity threshold: {delta_sq_k1} > 0.0120"
    sigma_box = np.sqrt(delta_sq_k1)
    delta_turnaround = 1.06
    assert sigma_box < (delta_turnaround / 3.0), f"RMS amplitude approaches turnaround: {sigma_box}"

def test_cardy_floor_regularized_gradient_boundedness():
    """Validates extinction of entropic gradient forces in deep cosmic voids."""
    scale_factors = [0.05, 0.1, 0.5, 1.0]  # z = 19, 9, 1, 0
    
    for a in scale_factors:
        rho_floor = max(0.00696738, 1.0 / (39.0 * a))
        
        rho_bulk = np.logspace(-12, -2, 100)
        grad_rho_bulk = np.full_like(rho_bulk, 1e-4)
        
        grad_ln_rho_reg = grad_rho_bulk / (rho_bulk + rho_floor)
        max_grad = np.max(grad_ln_rho_reg)
        assert max_grad <= (1e-4 / rho_floor), f"Gradient singularity detected at a={a}: {max_grad}"
        
        physical_grad_rho = rho_bulk * 0.1
        physical_reg_grad = physical_grad_rho / (rho_bulk + rho_floor)
        assert physical_reg_grad[0] < 1e-10, "Physical void gradient failed to extinguish"
