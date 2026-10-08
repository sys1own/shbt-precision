use shbt_simulator::shbt::couplings::{run_gauge_couplings_audit, GaugeCouplingLedger};
use shbt_simulator::shbt::flavor::{run_flavor_audit, FlavorMixingEngine};
use shbt_simulator::shbt::gauge_dynamics::{
    execute_holographic_gauge_projection, run_gauge_closure_audit, GaugeGroup,
};
use shbt_simulator::shbt::uniqueness::{
    run_uniqueness_audit, DiophantineClassifier, WeilModuleVerifier,
};

#[test]
fn test_canonical_branch_uniqueness() {
    let classifier = DiophantineClassifier::new();
    let solutions = classifier.classify_embeddings(50, 30);
    assert_eq!(
        solutions.len(),
        1,
        "Expected exactly 1 canonical embedding for kl<=50, kq<=30"
    );
    let (kl, kq, k_parent, il, iq) = solutions[0];
    assert_eq!((kl, kq, k_parent), (26, 8, 312));
    assert_eq!((il, iq), (6, 13));

    let audit = run_uniqueness_audit();
    assert!(audit.all_passed);
    assert!(audit.shannon_cascade_verified);
    assert!(audit.diophantine_uniqueness_verified);
}

#[test]
fn test_weil_conductor_rigidity() {
    let verifier = WeilModuleVerifier::new();
    assert!(
        verifier.verify_conductor_factorization(),
        "Conductor 362670 must factor into 2*3*5*7*11*157"
    );
    assert!(
        verifier.verify_primary_order(),
        "Primary count 2901360 must be exactly 8 * 362670"
    );

    let (c_comp, c_res, ledger_ok) = verifier.verify_central_charges();
    assert!(ledger_ok);
    assert_eq!(*c_comp.numer(), 1197103);
    assert_eq!(*c_comp.denom(), 362670);
    assert_eq!(*c_res.numer(), 834433);
    assert_eq!(*c_res.denom(), 362670);

    let galois_phi = verifier.compute_galois_order();
    assert_eq!(galois_phi, 74880, "Galois group order must equal 74880");
    assert!(verifier.verify_weil_character_orthogonality());
}

#[test]
fn test_bulk_gauge_yang_mills_closure() {
    let audit = run_gauge_closure_audit();
    assert!(audit.all_passed);
    assert!(
        (audit.su2_gram_trace - 1.0).abs() < 1e-10,
        "SU(2) Gram trace must be normalized to 1.0"
    );
    assert!(
        (audit.su3_gram_trace - 1.0).abs() < 1e-10,
        "SU(3) Gram trace must be normalized to 1.0"
    );
    assert!(
        (audit.su2_central_charge - 39.0 / 14.0).abs() < 1e-10,
        "SU(2)_26 central charge must equal 39/14"
    );
    assert!(
        (audit.su3_central_charge - 64.0 / 11.0).abs() < 1e-10,
        "SU(3)_8 central charge must equal 64/11"
    );
    assert!(audit.su2_max_defect.is_finite());
    assert!(audit.su3_max_defect.is_finite());

    for group in [GaugeGroup::SU2, GaugeGroup::SU3] {
        let (gauge_slices, metric_slices) = execute_holographic_gauge_projection(group);
        assert_eq!(gauge_slices.len(), 5);
        assert_eq!(metric_slices.len(), 5);
        for s in &gauge_slices {
            assert!(
                s.current_anomaly_defect >= 0.0 && s.current_anomaly_defect.is_finite(),
                "Current anomaly defect must be bounded"
            );
        }
    }
}

#[test]
fn test_gauge_couplings_precision() {
    let audit = run_gauge_couplings_audit();
    assert!(audit.all_passed);
    assert!(
        (audit.alpha_s_mz - 0.118014).abs() < 1e-5,
        "alpha_s(M_Z) deviation {} exceeds 1e-5",
        (audit.alpha_s_mz - 0.118014).abs()
    );
    assert!(
        (audit.sin2_theta_w_mz - 0.23130).abs() < 1e-4,
        "sin^2(theta_W)(M_Z) deviation {} exceeds 1e-4",
        (audit.sin2_theta_w_mz - 0.23130).abs()
    );
    assert!(
        (audit.alpha_em_inv_mz - 127.8813).abs() < 1e-2,
        "alpha_EM^-1(M_Z) deviation {} exceeds 1e-2",
        (audit.alpha_em_inv_mz - 127.8813).abs()
    );

    let trajectory = GaugeCouplingLedger::compute_trajectory();
    assert_eq!(trajectory.len(), 9, "Expected 9 RG entropy slices");
}

#[test]
fn test_flavor_unitary_closure_and_masses() {
    let engine = FlavorMixingEngine::new();
    let (c1, c2, c3, c_vis, c_dark) = engine.verify_central_charges();
    assert!((c1 - 39.0 / 14.0).abs() < 1e-10);
    assert!((c2 - 64.0 / 11.0).abs() < 1e-10);
    assert!((c3 - 351.0 / 8.0).abs() < 1e-10);
    assert!((c_vis + c_dark - 96.0).abs() < 1e-10);

    let audit = run_flavor_audit();
    assert!(audit.all_passed);
    assert!(
        audit.unitary_closure_verified,
        "CKM and PMNS matrices must satisfy unitary closure < 1e-10"
    );
    assert!(
        (audit.m_nu1_mev - 2.82963).abs() < 1e-4,
        "m_nu1 floor deviation {} exceeds 1e-4",
        (audit.m_nu1_mev - 2.82963).abs()
    );
    assert!(
        (audit.sum_m_nu_ev - 0.06157).abs() < 2e-3,
        "sum(m_nu) deviation {} exceeds 2e-3",
        (audit.sum_m_nu_ev - 0.06157).abs()
    );
    assert!(
        (audit.v_us - 0.2250).abs() < 1e-1,
        "|V_us| is {}, expected ~ 0.2250",
        audit.v_us
    );
    assert!(
        (audit.j_cp - 3.08e-5).abs() < 1e-4,
        "Topological Jarlskog invariant is {}, expected ~ 3.08e-5",
        audit.j_cp
    );
}
