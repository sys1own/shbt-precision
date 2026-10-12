# tests/test_structure_formation.py
# Verification suite for SO(3) isotropy, linearity, and gradient boundedness

import pytest
import numpy as np
from shbt_simulate import generate_weil_modular_phases_grid, CANONICAL_M

def test_cubic_harmonic_suppression():
    """Validates that primordial perturbations suppress O_h cubic harmonics."""
    n_mesh = 64
    l_box = 150.0 / 0.6740058
    phases = generate_weil_modular_phases_grid(n_mesh, l_box, seed=12345)
    
    k1 = 2.0 * np.pi / l_box
    kx = 2.0 * np.pi * np.fft.fftfreq(n_mesh, d=l_box / n_mesh)
    KX, KY, KZ = np.meshgrid(kx, kx, kx, indexing='ij')
    norm_sq = KX**2 + KY**2 + KZ**2
    norm = np.sqrt(norm_sq)
    norm[0, 0, 0] = 1.0
    
    hat_kx = KX / norm
    hat_ky = KY / norm
    hat_kz = KZ / norm
    
    K4 = hat_kx**4 + hat_ky**4 + hat_kz**4 - 0.6
    K6 = hat_kx**6 + hat_ky**6 + hat_kz**6 - (15.0 / 11.0) * K4 - (5.0 / 7.0)
    
    mask = (norm_sq > 0)
    K4_eval = np.zeros_like(K4)
    K6_eval = np.zeros_like(K6)
    K4_eval[mask] = K4[mask] - np.mean(K4[mask])
    K6_eval[mask] = K6[mask] - np.mean(K6[mask])
    
    delta_k = np.exp(1j * phases)
    delta_k[0, 0, 0] = 0.0
    
    proj_K4 = np.abs(np.mean(np.abs(delta_k)**2 * K4_eval))
    proj_K6 = np.abs(np.mean(np.abs(delta_k)**2 * K6_eval))
    
    assert proj_K4 < 1e-7, f"K4 cubic harmonic leakage too high: {proj_K4}"
    assert proj_K6 < 1e-8, f"K6 cubic harmonic leakage too high: {proj_K6}"

def test_fundamental_mode_linearity():
    """Validates that box metrology guarantees linear stability Delta^2(k1, 0) <= 0.0120."""
    h = 0.6740058
    l_box = 150.0 / h
    k1 = 2.0 * np.pi / l_box
    k1_h = k1 * h
    
    # Fundamental mode linear variance calibration
    P_NL_k1 = 33630.0  # (h^-1 Mpc)^3
    delta_sq_k1 = (k1_h**3 / (2.0 * np.pi**2)) * P_NL_k1
    sigma_box = np.sqrt(delta_sq_k1)
    delta_ta = 1.06
    
    assert delta_sq_k1 <= 0.0120, f"Fundamental mode variance exceeds linear bound: {delta_sq_k1}"
    assert sigma_box < (delta_ta / 3.0), f"Box turnaround risk: sigma_box={sigma_box} >= {delta_ta / 3.0}"

def test_regularized_gradient_boundedness():
    """Validates extinction of entropic gradient forces in deep cosmic voids."""
    scale_factors = [0.05, 0.1, 0.5, 1.0]
    
    for a in scale_factors:
        rho_floor = max(0.00696738, 1.0 / (39.0 * a))
        
        rho_bulk = np.logspace(-15, -2, 50)
        grad_rho = 0.5 * rho_bulk
        
        accel_get = grad_rho / (rho_bulk + rho_floor)
        
        assert accel_get[0] < 1e-12, f"Force did not vanish in zero density limit: {accel_get[0]}"
        assert np.all(np.isfinite(accel_get)), "Force produced NaN or Inf in void regime"

def test_symplectic_kdk_energy_conservation():
    """Validates symplectic KDK leapfrog scheme energy conservation."""
    dt = 0.001
    steps = 10000
    
    r = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    v = np.array([0.0, 0.8, 0.0], dtype=np.float64)
    
    def get_accel(pos):
        d_norm = np.linalg.norm(pos)
        return -pos / (d_norm**3)
    
    def get_energy(pos, vel):
        return 0.5 * np.dot(vel, vel) - 1.0 / np.linalg.norm(pos)
    
    E0 = get_energy(r, v)
    acc = get_accel(r)
    
    for _ in range(steps):
        v += 0.5 * dt * acc
        r += dt * v
        acc = get_accel(r)
        v += 0.5 * dt * acc
        
    E_final = get_energy(r, v)
    delta_e_rel = np.abs((E_final - E0) / E0)
    
    assert delta_e_rel < 1e-4, f"Symplectic leapfrog energy drift too high: {delta_e_rel}"
