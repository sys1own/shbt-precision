use shbt_simulator::shbt::stability_audit::{mass_congestion_coupling, AnomalyClosureError};

#[test]
fn mass_congestion_coupling_zero_detuning_passes() {
    let alpha = 1.67e-51_f64;
    let n_local = 6.0e59_f64;
    let n_limit = 3.478e56_f64; // resolution ceiling from precision_cosmology.py
    let mass = mass_congestion_coupling(alpha, n_local, n_limit, 0.0).unwrap();
    assert!(mass > 0.0);
}

#[test]
fn mass_congestion_coupling_onee_minus_twelve_detuning_triggers_anomaly() {
    let alpha = 1.67e-51_f64;
    let n_local = 6.0e59_f64;
    let n_limit = 3.478e56_f64;
    let result = mass_congestion_coupling(alpha, n_local, n_limit, 1e-12);
    assert!(
        result.is_err(),
        "a 1e-12 detuning of the mass-congestion coupling must exceed closure tolerance"
    );
    let _ = result.unwrap_err();
}

#[test]
fn mass_congestion_coupling_sub_tolerance_detuning_passes() {
    let alpha = 1.67e-51_f64;
    let n_local = 6.0e59_f64;
    let n_limit = 3.478e56_f64;
    assert!(mass_congestion_coupling(alpha, n_local, n_limit, 1e-13).is_ok());
}

#[test]
fn test_neveu_schwarz_stationarity_residual() {
    let b = 1.0 / 19.0_f64.sqrt();
    let q_charge = 20.0 / 19.0_f64.sqrt();
    let bq = 20.0 / 19.0;
    let mut maximum_residual = 0.0_f64;
    for step in 0..=1000 {
        let tau = step as f64 / 1000.0;
        let delta_one = 0.08 * q_charge * (2.0 * std::f64::consts::PI * tau).sin();
        let delta_two = 0.08 * q_charge * (4.0 * std::f64::consts::PI * tau).cos();
        let sum_alpha = q_charge / 3.0 + delta_one
            + q_charge / 3.0 + delta_two
            + q_charge / 3.0 - delta_one - delta_two;
        maximum_residual = maximum_residual.max((sum_alpha - q_charge).abs());
        maximum_residual = maximum_residual.max((bq - b * sum_alpha).abs());
    }
    assert!(maximum_residual < 1.0e-12);
}

#[test]
fn test_dark_sector_modular_closure() {
    assert!(shbt_simulator::shbt::StaticBoundary::verify_dark_modular_closure().unwrap());
    let (numerator, denominator) = shbt_simulator::shbt::StaticBoundary::c_dark_completed_rational();
    assert_eq!((numerator, denominator), (1197103, 362670));
}
