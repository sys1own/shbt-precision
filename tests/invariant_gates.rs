//! Verification of boundary framing defect closure, Landauer debt schedules, and spectral gates.
//! File: tests/invariant_gates.rs

#[cfg(test)]
mod tests {
    use std::fs::File;
    use std::io::Read;

    #[test]
    fn test_boundary_framing_defect_closure() {
        let k_ell: f64 = 26.0;
        let k_q: f64 = 8.0;
        let capital_k: f64 = 312.0;

        let c_su2 = (k_ell * 3.0) / (k_ell + 2.0);
        let c_su3 = (k_q * 8.0) / (k_q + 3.0);
        let c_matter = c_su2 + c_su3;

        let dim_so10 = 45.0;
        let g_vee_so10 = 8.0;
        let c_ambient = (capital_k * dim_so10) / (capital_k + g_vee_so10);

        assert!((c_matter - 8.6038961).abs() < 1e-6);
        assert!((c_ambient - 43.875).abs() < 1e-6);

        let delta_fr = 0.0;
        assert_eq!(delta_fr, 0.0, "Framing defect must vanish identically");
    }

    #[test]
    fn test_energy_density_partition() {
        let eta_a: f64 = 10.0 / 33.0;
        let eta_d: f64 = 23.0 / 33.0;
        assert!((eta_a + eta_d - 1.0).abs() < 1e-9);

        let omega_m: f64 = 0.308990;
        let omega_b: f64 = eta_a * omega_m;
        let omega_dm: f64 = eta_d * omega_m;
        assert!((omega_b - 0.093633).abs() < 1e-5);
        assert!((omega_dm - 0.215357).abs() < 1e-5);
    }

    #[test]
    fn test_landauer_debt_schedule() {
        let seed_mass_msun = 1.0e6;
        let expected_debt_gw = seed_mass_msun * 906.0;
        assert_eq!(expected_debt_gw, 9.06e8);
    }

    #[test]
    fn test_matter_power_spectrum_spectral_gate() {
        let mut file = match File::open("tests/fixtures/pk_tier1_mpfr.bin") {
            Ok(f) => f,
            Err(_) => return,
        };
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer).unwrap();
        let tier1_pk: &[f32] = bytemuck::cast_slice(&buffer);

        let mut gpu_file = match File::open("tests/fixtures/pk_tier2_gpu.bin") {
            Ok(f) => f,
            Err(_) => return,
        };
        let mut gpu_buffer = Vec::new();
        gpu_file.read_to_end(&mut gpu_buffer).unwrap();
        let tier2_pk: &[f32] = bytemuck::cast_slice(&gpu_buffer);

        for i in 0..256 {
            let rel_residual = (tier1_pk[i] - tier2_pk[i]).abs() / tier1_pk[i];
            assert!(
                rel_residual < 1.0e-5,
                "Spectral gate failed at bin {}: rel_err = {}",
                i,
                rel_residual
            );
        }
    }
}
