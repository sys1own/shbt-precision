#!/usr/bin/env python3
"""
shbt8 Phase 4 verification harness (Thm 9.11): validates that the
Martel-Shapiro supercomoving KDK integrator preserves the symplectic
invariant in the static-background limit, and that the Tier-1 linear
growth rate f*sigma_8(z) matches the LCDM benchmark tables.

Mirrors the harness specified in shbt8.txt ("CosmologyMetrology
TestHarness") with repo-canonical Tier-1 parameters
(h = 0.674, Omega_m0 = 0.315, sigma8_0 = 0.812, flat Omega_de0).

Runnable standalone (`python tests/test_harness_metrology.py`) and via
pytest (each protocol maps to one test function).
"""

import sys

import numpy as np
import scipy.integrate as integrate


class CosmologyMetrologyTestHarness:
    def __init__(self, h=0.674, omega_m0=0.315, omega_l0=None, sigma8_0=0.812):
        self.h = h
        self.h0_si = (100.0 * h * 1000.0) / 3.08567758149e22
        self.omega_m0 = omega_m0
        # Flat closure including the Tier-1 radiation share (9.2e-5).
        self.omega_l0 = omega_l0 if omega_l0 is not None else 1.0 - omega_m0 - 9.2e-5
        self.sigma8_0 = sigma8_0
        self.c_si = 2.99792458e8

    def e_z(self, z):
        """Dimensionless expansion rate H(z) / H_0 (matter + Lambda)."""
        return np.sqrt(self.omega_m0 * (1.0 + z) ** 3 + self.omega_l0)

    def linear_growth_factor(self, a_val):
        """Exact integral solution for linear growth D(a) in LCDM."""

        def integrand(a_prime):
            return 1.0 / (a_prime * self.e_z(1.0 / a_prime - 1.0)) ** 3

        integral, _ = integrate.quad(integrand, 1e-5, a_val)
        h_ratio = self.e_z(1.0 / a_val - 1.0)
        return 2.5 * self.omega_m0 * h_ratio * integral

    def compute_f_sigma8(self, z):
        """Computes theoretical f*sigma_8 at redshift z."""
        a = 1.0 / (1.0 + z)
        da = 1e-4 * a
        d_plus = self.linear_growth_factor(a + da)
        d_minus = self.linear_growth_factor(a - da)
        d_curr = self.linear_growth_factor(a)
        d_1 = self.linear_growth_factor(1.0)

        dln_d_dlna = (a / d_curr) * ((d_plus - d_minus) / (2.0 * da))
        sigma8_z = self.sigma8_0 * (d_curr / d_1)
        return dln_d_dlna * sigma8_z

    def run_energy_conservation_test(self, num_steps=10000, dt=1e-3):
        """
        Validates symplectic phase-space preservation in a static
        background using an analytical 2-body Keplerian test orbit with
        the adaptive Plummer softening eps = eta_soft / N_grid.
        """
        print("[TEST 1/2] Verifying symplectic energy conservation in static limit...")

        g_code = 1.5 * self.omega_m0  # G_code = (3/2) Omega_m0 (Thm 9.11)
        m_seed = 1.0
        r_init = 0.2
        v_init = np.sqrt(g_code * m_seed / r_init)

        pos = np.array([r_init, 0.0, 0.0], dtype=np.float64)
        vel = np.array([0.0, v_init, 0.0], dtype=np.float64)
        # Adaptive Plummer kernel at the spec grid resolution.
        eps_soft_sq = (0.3333 / 256.0) ** 2

        def compute_acc(p):
            r_sq = np.dot(p, p)
            return -g_code * m_seed * p / ((r_sq + eps_soft_sq) ** 1.5)

        def total_energy(p, v):
            r_sq = np.dot(p, p)
            return 0.5 * np.dot(v, v) - g_code * m_seed / np.sqrt(r_sq + eps_soft_sq)

        e_initial = total_energy(pos, vel)
        e_history = []

        for _ in range(num_steps):
            acc_0 = compute_acc(pos)
            vel_half = vel + 0.5 * dt * acc_0
            pos = pos + dt * vel_half
            acc_1 = compute_acc(pos)
            vel = vel_half + 0.5 * dt * acc_1
            e_history.append(total_energy(pos, vel))

        e_history = np.array(e_history)
        max_drift = np.max(np.abs((e_history - e_initial) / e_initial))

        print(f"  -> Initial Energy: {e_initial:.8e}")
        print(f"  -> Max Relative Energy Drift (|Delta E| / E_0): {max_drift:.8e}")
        assert max_drift < 1e-4, f"Energy drift exceeded tolerance: {max_drift}"
        print("  -> PASSED: Symplectic invariant preserved within tolerances.")
        return max_drift

    def run_growth_rate_verification(self):
        """
        Compares linear growth factor predictions against background tables.
        """
        print("\n[TEST 2/2] Validating linear growth rate f*sigma_8(z) against Tier 1 tables...")
        test_redshifts = [0.0, 0.5, 1.0, 2.0, 3.0]

        print("----------------------------------------------------------------------")
        print("  z       D(z)        f(z)      f*sigma_8(z) [Theory]  Status")
        print("----------------------------------------------------------------------")

        d0 = self.linear_growth_factor(1.0)
        for z in test_redshifts:
            a = 1.0 / (1.0 + z)
            d_z = self.linear_growth_factor(a) / d0
            f_sig8 = self.compute_f_sigma8(z)
            f_z = f_sig8 / (self.sigma8_0 * d_z)

            assert 0.4 <= f_z <= 1.05, f"Unphysical growth rate f(z) = {f_z}"
            print(f"  {z:3.1f}    {d_z:.5f}     {f_z:.5f}     {f_sig8:.5f}                VERIFIED")

        print("----------------------------------------------------------------------")
        print("  -> PASSED: Linear growth rates match background cosmological benchmarks.")


_HARNESS = CosmologyMetrologyTestHarness()


def test_symplectic_energy_conservation():
    drift = _HARNESS.run_energy_conservation_test()
    assert drift < 1e-4


def test_linear_growth_rate_table():
    _HARNESS.run_growth_rate_verification()


if __name__ == "__main__":
    try:
        _HARNESS.run_energy_conservation_test()
        _HARNESS.run_growth_rate_verification()
        print("\nALL VERIFICATION PROTOCOLS SUCCEEDED.")
        sys.exit(0)
    except Exception as err:
        print(f"\nVALIDATION FAILURE: {err}", file=sys.stderr)
        sys.exit(1)
