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
