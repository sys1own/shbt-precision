# tests/test_grid_parity.py
# Verification suite for spatial parity restoration and cubic harmonic suppression

import struct
import numpy as np
import pytest
from shbt_simulate import generate_2lpt_particles, generate_weil_modular_phases_grid

def test_initial_mass_parity():
    """
    Validates that the fundamental mode dipole nullification prevents
    spatial mass pre-shift across the comoving hemispheres:
    |M(x > 0.5) - M(x < 0.5)| / M_tot < 0.01
    """
    n_grid = 64
    l_box_canonical = 150.0 / 0.6740058
    particles = generate_2lpt_particles(n_particles_1d=n_grid, l_box=l_box_canonical, z_init=100.0)
    
    total_particles = len(particles)
    left_mass = sum(p[3] for p in particles if p[0] < 0.5)
    right_mass = sum(p[3] for p in particles if p[0] >= 0.5)
    total_mass = left_mass + right_mass
    
    imbalance = abs(right_mass - left_mass) / total_mass
    print(f"[TEST] Left Mass: {left_mass:.6f}, Right Mass: {right_mass:.6f}, Parity Imbalance: {imbalance:.6e}")
    assert imbalance < 0.01, f"Initial mass parity violation: {imbalance:.6e} >= 0.01"


def test_cubic_harmonic_suppression():
    """
    Validates that direction-cosine Haar dispersion suppresses cubic harmonic leakage:
    |K_4| < 10^-7
    """
    n_mesh = 64
    l_box_canonical = 150.0 / 0.6740058
    phases = generate_weil_modular_phases_grid(n_mesh, l_box=l_box_canonical, seed=12345)
    
    k_freq = np.fft.fftfreq(n_mesh, d=l_box_canonical / n_mesh) * 2.0 * np.pi
    KX, KY, KZ = np.meshgrid(k_freq, k_freq, k_freq, indexing='ij')
    norm_sq = KX**2 + KY**2 + KZ**2
    norm = np.sqrt(norm_sq)
    norm[0, 0, 0] = 1.0
    
    hx = KX / norm
    hy = KY / norm
    hz = KZ / norm
    
    k4 = hx**4 + hy**4 + hz**4 - 0.6
    
    # Measure parity-centered phase projection onto K4
    mask = (norm_sq > 0)
    phi_centered = np.where(phases > np.pi, phases - 2.0 * np.pi, phases)
    mean_leakage = np.abs(np.mean(phi_centered[mask] * k4[mask]))
    
    print(f"[TEST] Mean K4 Leakage: {mean_leakage:.6e}")
    assert mean_leakage < 1e-7, f"Cubic harmonic leakage violation: {mean_leakage:.6e} >= 1e-7"
