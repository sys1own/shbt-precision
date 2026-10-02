//! Conservation and admissibility invariant gates for the Tier-2
//! visualization ledger (shbt4 spec, `tests/invariant_gates.rs`).

use shbt_visualizer::{HorizonLedger, N_SAT};

#[test]
fn test_bit_conservation_across_all_epochs() {
    let mut ledger = HorizonLedger::new();
    let test_redshifts = [
        1.0e14, 1.0e11, 1.0e9, 1100.0, 100.0, 30.0, 15.0, 7.0, 2.0, 0.0, -0.5, -0.999,
    ];

    for &z in &test_redshifts {
        ledger.update(z);
        let bit_sum = ledger.active_visible_bits + ledger.dark_completion_bits;
        let expected = ledger.total_bits_loaded;
        let diff_ratio = (bit_sum - expected).abs() / N_SAT;

        assert!(
            diff_ratio < 1.0e-30,
            "Bit conservation violated at z = {z}: residual = {diff_ratio}"
        );
    }
}

#[test]
fn test_exact_modular_partition_split() {
    let mut ledger = HorizonLedger::new();
    ledger.update(0.0);

    let eta_vis = ledger.active_visible_bits / ledger.total_bits_loaded;
    let eta_dark = ledger.dark_completion_bits / ledger.total_bits_loaded;

    let expected_vis = 10.0 / 33.0;
    let expected_dark = 23.0 / 33.0;

    assert!((eta_vis - expected_vis).abs() < 1.0e-12);
    assert!((eta_dark - expected_dark).abs() < 1.0e-12);
}

#[test]
fn test_observer_admissibility_freeze_at_de_sitter() {
    let mut ledger = HorizonLedger::new();
    ledger.update(-0.999);

    assert_eq!(
        ledger.observer_admissibility_cardinality, 0,
        "Observer set must be empty at asymptotic freeze"
    );
    assert!(
        !ledger.is_observer_admissible(),
        "Admissibility flag must evaluate to false at z -> -1"
    );
}

#[test]
fn test_stress_energy_conservation_residual() {
    let mut ledger = HorizonLedger::new();
    ledger.update(1.0e10);

    assert!(
        ledger.conservation_residual < 1.0e-120,
        "Stress-energy divergence residual exceeded threshold: {}",
        ledger.conservation_residual
    );
}
