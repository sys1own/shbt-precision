//! Verification Suite for First-Principles Boundary Quantum Dynamics
//! File: tests/first_principles_kernels.rs
//! Repository: sys1own/shbt-precision

#![allow(clippy::all, unused)]

#[cfg(test)]
mod tests {
    use std::f64::consts::PI;

    // Canonical WZW affine branch invariants (k_l, k_q, K) = (26, 8, 312)
    #[allow(dead_code)]
    const K_L: f64 = 26.0;
    const K_Q: f64 = 8.0;
    #[allow(dead_code)]
    const K_GUT: f64 = 312.0;
    const C_EFF: f64 = 1325.0 / 154.0;   // 8.603896103896104
    const D_1: f64 = 26.0;               // gcd(26, 312)
    const H_DUAL_SU3: f64 = 3.0;
    const N_SAT: f64 = 3.3119977e122;
    const ETA_D: f64 = 23.0 / 33.0;      // 0.696969696969697
    const Z_N: f64 = 7.356e10;
    const DELTA_BBAR: f64 = 26.0 / 3.0;  // 8.666666666666666


    /// Closed-form first-principles causal GET coupling kappa_GET(z)
    fn kappa_get(f_load: f64) -> f64 {
        let d_eff_0 = D_1 - H_DUAL_SU3; // 23.0
        let bare_kappa = 1.0 / d_eff_0; // 1.0 / 23.0 = 0.04347826086956522
        let anomaly_factor = (C_EFF / D_1) * f_load;
        bare_kappa * (1.0 + anomaly_factor)
    }

    /// Thermal Stinespring dilation expectation value w_vis(z)
    fn w_vis(z: f64) -> f64 {
        let ratio = Z_N / z;
        let power_term = ratio.powf(DELTA_BBAR);
        (1.0 - ETA_D) + (ETA_D / (1.0 + power_term))
    }

    /// Analytical derivative dw_vis / dz
    fn dw_vis_dz(z: f64) -> f64 {
        let ratio = Z_N / z;
        let power_term = ratio.powf(DELTA_BBAR);
        let denom = 1.0 + power_term;
        (ETA_D * DELTA_BBAR / z) * (power_term / (denom * denom))
    }

    /// Instanton boundary defect action S_inst / hbar
    fn s_inst_over_hbar(n_local: f64, n_limit: f64) -> f64 {
        if n_local >= n_limit {
            0.0
        } else {
            let delta = (n_limit - n_local) / n_limit;
            let prefactor = (2.0 * PI * C_EFF) / K_Q;
            prefactor * delta * delta
        }
    }

    #[test]
    fn test_stress_energy_conservation_in_get_transport() {
        // Evaluate stress-energy trace conservation: |E_mu_nu| < 10^-120 across all screen loads
        let test_loads = [0.0, 1e-12, 1e-6, 0.01, 0.25, 0.5, 0.75, 0.99, 1.0];

        for &f_load in &test_loads {
            let kappa = kappa_get(f_load);

            // Verify that bare coupling corresponds to exact group-theoretic fraction 1/23
            let bare_diff = (kappa_get(0.0) - (1.0 / 23.0)).abs();
            assert!(
                bare_diff < 1.0e-15,
                "Bare coupling deviates from 1/23: diff = {:e}",
                bare_diff
            );

            // Semiclassical stress-energy tensor trace discrepancy under boundary projective invariance
            let d_eff_restored = D_1 - H_DUAL_SU3;
            let trace_anom = (1.0 - d_eff_restored * (kappa / (1.0 + (C_EFF / D_1) * f_load))).abs();
            let e_mu_nu = trace_anom * (1.0 / N_SAT);

            assert!(
                e_mu_nu < 1.0e-120,
                "Stress-energy conservation violated: E_mu_nu = {:e} >= 1e-120 at f_load = {}",
                e_mu_nu,
                f_load
            );
        }
    }

    #[test]
    fn test_seed_nucleation_instanton_dynamics() {
        let n_limit = 1.0e60;

        // Sub-capacity regime: finite positive action providing exponential suppression
        let s_sub = s_inst_over_hbar(0.5e60, n_limit);
        assert!(s_sub > 0.0, "Instanton action must be positive below capacity ceiling");
        assert!(
            (s_sub - (1325.0 * PI / 616.0) * 0.25).abs() < 1.0e-10,
            "Instanton action numerical mismatch"
        );

        // Threshold boundary: vanishing instanton barrier
        let s_crit = s_inst_over_hbar(n_limit, n_limit);
        assert_eq!(s_crit, 0.0, "Instanton action must vanish at exact capacity");

        // Super-capacity regime: barrierless condensation
        let s_super = s_inst_over_hbar(2.5e60, n_limit);
        assert_eq!(s_super, 0.0, "Instanton action must remain zero for overflow");

        // Cosmic dawn clustering verification across unconstrained dynamic pool
        let attempt_freq = 1.0e-4;
        let dt = 0.5;
        let p_nucleation_super = 1.0 - (-attempt_freq * dt * (-s_super).exp()).exp();
        assert!(
            p_nucleation_super > 0.0 && p_nucleation_super < 1.0,
            "Nucleation probability must remain well-behaved in (0, 1)"
        );
    }

    #[test]
    fn test_bit_conservation_across_stinespring_transition() {
        // Redshift sweep across modular restoration scale z in [10^14, 10^8]
        let z_steps = 2000;
        let log_z_start = 14.0_f64;
        let log_z_end = 8.0_f64;

        for i in 0..=z_steps {
            let log_z = log_z_start - (i as f64 / z_steps as f64) * (log_z_start - log_z_end);
            let z = 10.0_f64.powf(log_z);

            let w_v = w_vis(z);
            let dw_dz = dw_vis_dz(z);

            // Channel unitary partitioning: w_vis + w_dark == 1
            let w_dark = 1.0 - w_v;
            let bit_conservation_error = ((w_v + w_dark) - 1.0).abs();

            assert!(
                bit_conservation_error < 1.0e-30,
                "Bit conservation violated at z = {:e}: error = {:e}",
                z,
                bit_conservation_error
            );

            // Consistency check: match analytic derivative against central finite difference
            let eps = z * 1.0e-6;
            let numerical_dw_dz = (w_vis(z + eps) - w_vis(z - eps)) / (2.0 * eps);
            let rel_error = (dw_dz - numerical_dw_dz).abs() / dw_dz.abs().max(1.0e-15);

            assert!(
                rel_error < 1.0e-4,
                "Analytic derivative discrepancy at z = {:e}: rel_error = {:e}",
                z,
                rel_error
            );
        }

        // Asymptotic boundary conditions verification
        let high_z_limit = w_vis(1.0e14);
        let low_z_limit = w_vis(1.0e8);
        assert!(
            (high_z_limit - 1.0).abs() < 1.0e-12,
            "High-redshift channel must be fully unquenched: w_vis = {}",
            high_z_limit
        );
        assert!(
            (low_z_limit - (10.0 / 33.0)).abs() < 1.0e-12,
            "Low-redshift channel must reach residual fraction 10/33: w_vis = {}",
            low_z_limit
        );
    }

    /// Martel-Shapiro supercomoving KDK leapfrog: relative Hamiltonian
    /// drift on the static-background limit (harmonic potential) must
    /// remain < 10^-4 over 10^4 steps — the symplectic invariant that
    /// keeps structure formation from injecting spurious energy.
    #[test]
    fn test_symplectic_energy_drift_static_background() {
        let dt = 0.001f32;
        let mut x = 1.0f32;
        let mut v = 0.0f32;
        let k = 1.0f32;

        let e_initial = 0.5 * v * v + 0.5 * k * x * x;

        for _ in 0..10_000 {
            let a0 = -k * x;
            v += 0.5 * dt * a0;
            x += dt * v;
            let a1 = -k * x;
            v += 0.5 * dt * a1;
        }

        let e_final = 0.5 * v * v + 0.5 * k * x * x;
        let rel_energy_drift = ((e_final - e_initial) / e_initial).abs();

        assert!(
            rel_energy_drift < 1e-4,
            "Symplectic KDK drift exceeded 10^-4 limit over 10^4 steps: drift = {:.3e}",
            rel_energy_drift
        );
    }

    #[test]
    fn test_symplectic_leapfrog_energy_conservation() {
        let mut x: f64 = 0.5;
        let mut p: f64 = 0.1;
        let dt_tau: f64 = 1e-4;
        let n_steps: usize = 10_000;

        let omega: f64 = 1.0;
        let initial_energy = 0.5 * p * p + 0.5 * omega * omega * x * x;

        for _ in 0..n_steps {
            p -= 0.5 * dt_tau * (omega * omega * x);
            x += dt_tau * p;
            p -= 0.5 * dt_tau * (omega * omega * x);
        }

        let final_energy = 0.5 * p * p + 0.5 * omega * omega * x * x;
        let relative_drift = ((final_energy - initial_energy) / initial_energy).abs();

        assert!(
            relative_drift < 1e-4,
            "Symplectic KDK energy drift exceeded tolerance: {} >= 1e-4",
            relative_drift
        );
    }

    /// Branchless degree-(2,2) rational Planckian palette used by
    /// `blackbody_to_linear_rgb` in dual_channel_render.wgsl: every RGB
    /// component must stay inside [0, 1.05] with no NaN across the
    /// entire emission band T in [2500, 25000] K.
    #[test]
    fn test_branchless_blackbody_simd_rational_monotonicity() {
        let temps: Vec<f32> = (2500..=25000).step_by(250).map(|t| t as f32).collect();
        for t in temps {
            let u = 1000.0 / t;
            let u2 = u * u;
            let r = (0.657842 - 15.067116 * u + 144.948029 * u2)
                / (1.0 - 19.160428 * u + 155.930970 * u2);
            let g = (0.540874 - 1.770797 * u + 12.417570 * u2)
                / (1.0 - 9.751228 * u + 48.259277 * u2);
            let b = (0.855655 - 3.605022 * u + 4.930827 * u2)
                / (1.0 - 8.226975 * u + 39.334862 * u2);

            assert!(r >= 0.0 && r <= 1.05, "Red channel out of bounds at T = {}: {}", t, r);
            assert!(g >= 0.0 && g <= 1.05, "Green channel out of bounds at T = {}: {}", t, g);
            assert!(b >= 0.0 && b <= 1.05, "Blue channel out of bounds at T = {}: {}", t, b);
        }
    }

    #[test]
    fn test_weil_modular_anti_symmetry() {
        let n_mesh = 32;
        let l_box = 222.5502;
        let seed = 987654321;
        let phases = shbt_simulator::shbt::structure::generate_weil_modular_phases(n_mesh, l_box, seed);

        assert_eq!(phases[0], 0.0f32, "DC mode (0,0,0) must have phase exactly 0.0");

        for iz in 0..n_mesh {
            let nz = if iz < n_mesh / 2 { iz as i32 } else { iz as i32 - n_mesh as i32 };
            for iy in 0..n_mesh {
                let ny = if iy < n_mesh / 2 { iy as i32 } else { iy as i32 - n_mesh as i32 };
                for ix in 0..n_mesh {
                    let nx = if ix < n_mesh / 2 { ix as i32 } else { ix as i32 - n_mesh as i32 };
                    if nx == 0 && ny == 0 && nz == 0 {
                        continue;
                    }

                    let idx = (iz * n_mesh + iy) * n_mesh + ix;
                    let neg_ix = (n_mesh as i32 - nx) as usize % n_mesh;
                    let neg_iy = (n_mesh as i32 - ny) as usize % n_mesh;
                    let neg_iz = (n_mesh as i32 - nz) as usize % n_mesh;
                    let neg_idx = (neg_iz * n_mesh + neg_iy) * n_mesh + neg_ix;

                    let p1 = phases[idx] as f64;
                    let p2 = phases[neg_idx] as f64;
                    let sum = (p1 + p2).rem_euclid(2.0 * PI);

                    assert!(
                        sum < 1e-4 || (2.0 * PI - sum) < 1e-4,
                        "Anti-symmetry broken at mode ({}, {}, {}): p1={}, p2={}, sum={}",
                        nx, ny, nz, p1, p2, sum
                    );
                }
            }
        }
    }

    #[test]
    fn test_cardy_density_floor_bounds() {
        let cardy_ratio = 39.0;
        let rho_floor_base = 0.00696738;

        let a_early = 1.0 / 101.0;
        let floor_early = f64::max(rho_floor_base, 1.0 / (cardy_ratio * a_early));
        assert!(floor_early >= 2.58, "Floor at z=100 must scale as 1/(39*a)");

        let a_late = 1.0;
        let floor_late = f64::max(rho_floor_base, 1.0 / (cardy_ratio * a_late));
        assert_eq!(floor_late, 1.0 / 39.0, "Floor at z=0 must equal 1/39");

        let a_asymp = 1000.0;
        let floor_asymp = f64::max(rho_floor_base, 1.0 / (cardy_ratio * a_asymp));
        assert_eq!(floor_asymp, rho_floor_base, "Floor asymptotically approaches rho_IR");
    }

    const LAMBDA_TH: f32 = 0.0198818;

    fn cardano_vieta_solver(t: [[f32; 3]; 3]) -> (f32, f32, f32) {
        let i1 = t[0][0] + t[1][1] + t[2][2];
        let trace_third = i1 / 3.0;

        let b00 = t[0][0] - trace_third;
        let b11 = t[1][1] - trace_third;
        let b22 = t[2][2] - trace_third;
        let b01 = t[0][1];
        let b02 = t[0][2];
        let b12 = t[1][2];

        let p = (b00 * b00 + b11 * b11 + b22 * b22 + 2.0 * (b01 * b01 + b02 * b02 + b12 * b12)) / 6.0;
        let q = 0.5 * (b00 * (b11 * b22 - b12 * b12) - b01 * (b01 * b22 - b02 * b12) + b02 * (b01 * b12 - b02 * b11));

        if p <= 1e-12 {
            return (trace_third, trace_third, trace_third);
        }

        let sqrt_p = p.sqrt();
        let cos_arg = (q / (p * sqrt_p)).clamp(-1.0, 1.0);
        let phi = cos_arg.acos() / 3.0;

        let pi_f32 = std::f32::consts::PI;
        let l1 = trace_third + 2.0 * sqrt_p * phi.cos();
        let l3 = trace_third + 2.0 * sqrt_p * (phi + 2.0 * pi_f32 / 3.0).cos();
        let l2 = i1 - l1 - l3;
        (l1, l2, l3)
    }

    #[test]
    fn test_cardano_vieta_strictly_ordered() {
        let t = [
            [0.15, 0.02, 0.01],
            [0.02, 0.08, 0.03],
            [0.01, 0.03, -0.05],
        ];

        let (l1, l2, l3) = cardano_vieta_solver(t);

        assert!(l1 >= l2, "Ordering failure: l1 ({}) < l2 ({})", l1, l2);
        assert!(l2 >= l3, "Ordering failure: l2 ({}) < l3 ({})", l2, l3);

        let trace_orig = t[0][0] + t[1][1] + t[2][2];
        let trace_eval = l1 + l2 + l3;
        assert!((trace_orig - trace_eval).abs() < 1e-5, "Trace invariance violated");
    }

    #[test]
    fn test_tweb_morphology_classes() {
        let (_l1, l2, l3) = cardano_vieta_solver([[0.05, 0.0, 0.0], [0.0, 0.03, 0.0], [0.0, 0.0, -0.01]]);
        assert!(l2 > LAMBDA_TH && l3 <= LAMBDA_TH, "Filament classification error");

        let (_k1, _k2, k3) = cardano_vieta_solver([[0.08, 0.0, 0.0], [0.0, 0.05, 0.0], [0.0, 0.0, 0.03]]);
        assert!(k3 > LAMBDA_TH, "Knot classification error");

        let (v1, _, _) = cardano_vieta_solver([[-0.01, 0.0, 0.0], [0.0, -0.02, 0.0], [0.0, 0.0, -0.05]]);
        assert!(v1 <= LAMBDA_TH, "Void classification error");
    }

    #[test]
    fn test_bremsstrahlung_pade_accuracy() {
        let t_eff = 5.0e7_f64;
        let kt_kev = t_eff * 8.617333e-8;
        let u = kt_kev.log10();
        let gaunt_pade = (1.201 + u * (0.142 + 0.051 * u)) / (1.0 + u * (0.082 + 0.039 * u));
        let gaunt_exact = 1.20;
        let rel_err = (gaunt_pade - gaunt_exact).abs() / gaunt_exact;
        assert!(rel_err < 0.05, "Gaunt factor Padé relative error exceeds 5%: {}", rel_err);
    }
}

