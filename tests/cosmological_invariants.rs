//! Conservation invariant gates for the boundary cosmology engine and the
//! SHBT-MMIO telemetry serializer (spec: `shbt3.txt`, Verification Matrix).
use rug::Float;
use shbt_simulator::shbt::cosmology::ShbtUniverse;
use shbt_simulator::shbt::export::{
    MmioTelemetryHeader, SeedDefectRecord, SHBT_MMIO_HEADER_BYTES, SHBT_MMIO_MAGIC,
    SHBT_MMIO_SCHEMA, SEED_RECORD_BYTES,
};
use shbt_simulator::shbt::{CausalPoint, StaticBoundary};

#[test]
fn test_boundary_bit_and_stress_energy_conservation() {
    let prec = 512;
    let universe = ShbtUniverse::new_canonical_branch(prec);

    // Scan across cosmic evolution from primordial to de Sitter freeze.
    let redshifts = [1.0e12, 1.0e9, 1000.0, 10.0, 7.0, 2.0, 1.0, 0.0, -0.99];

    for &z in &redshifts {
        let f_load = universe.conformal_loading_fraction(z);
        let n_sat = universe.saturated_screen_capacity();
        let n_vis = universe.active_visible_bits(z);
        let n_dark = universe.dark_completion_bits(z);

        // Gate 1: Exact Bit Conservation: N_vis + N_dark == N_sat * f_load
        let n_total_loaded = Float::with_val(prec, &n_vis + &n_dark);
        let n_expected = Float::with_val(prec, &n_sat * &f_load);
        let bit_error = Float::with_val(prec, &n_total_loaded - &n_expected).abs() / &n_sat;

        assert!(
            bit_error < Float::with_val(prec, 1.0e-30),
            "Bit conservation violated at z = {}: relative error = {}",
            z,
            bit_error
        );

        // Gate 2: Stress-Energy Preservation across Stinespring de-rendering
        let e_munu_residual = universe.evaluate_stress_energy_divergence(z);
        assert!(
            e_munu_residual < Float::with_val(prec, 1.0e-120),
            "Stress-energy conservation violated at z = {}: residual = {}",
            z,
            e_munu_residual
        );
    }
}

#[test]
fn test_stinespring_trace_preservation() {
    // Gate 3: Tr[V_unified^macro rho] = 1 and Ent(phi) = 0 on the canonical
    // branch across the macroscopic de-rendering channel.
    let boundary = StaticBoundary::new();
    let mut observer = CausalPoint::new(boundary);
    let record = observer.terminate_and_derender();

    assert!(
        (record.trace_norm - 1.0).abs() < 1.0e-18,
        "trace preservation violated: Tr = {}",
        record.trace_norm
    );
    assert_eq!(
        record.topological_entropy, 0.0,
        "Kojima topological entropy Ent(phi) must vanish"
    );
    assert!(
        (record.eta_dark - 23.0 / 33.0).abs() < 1.0e-18,
        "dark partition fraction must be 23/33"
    );
    assert!(
        (record.eta_visible - 10.0 / 33.0).abs() < 1.0e-18,
        "visible partition fraction must be 10/33"
    );
}

#[test]
fn test_mmio_header_serialization_layout() {
    let universe = ShbtUniverse::new_canonical_branch(512);
    let header =
        MmioTelemetryHeader::from_universe(&universe, 7.0, 42, 1_048_576, 6.0e59, 1);
    let bytes = header.to_bytes();

    assert_eq!(bytes.len(), SHBT_MMIO_HEADER_BYTES);
    assert_eq!(
        u32::from_le_bytes(bytes[0x00..0x04].try_into().unwrap()),
        SHBT_MMIO_MAGIC
    );
    assert_eq!(
        u32::from_le_bytes(bytes[0x04..0x08].try_into().unwrap()),
        SHBT_MMIO_SCHEMA
    );
    assert_eq!(
        u64::from_le_bytes(bytes[0x08..0x10].try_into().unwrap()),
        42
    );
    assert_eq!(
        f64::from_le_bytes(bytes[0x18..0x20].try_into().unwrap()),
        7.0
    );
    assert_eq!(
        u64::from_le_bytes(bytes[0x68..0x70].try_into().unwrap()),
        1_048_576
    );
    // Reserved pad must remain zero.
    assert!(bytes[0x70..0x80].iter().all(|&b| b == 0));

    let decoded = MmioTelemetryHeader::from_bytes(&bytes).unwrap();
    assert_eq!(decoded.frame_index, 42);
    assert_eq!(decoded.redshift_z, 7.0);
    assert_eq!(decoded.particle_count, 1_048_576);
}

#[test]
fn test_seed_record_and_frame_payload() {
    let universe = ShbtUniverse::new_canonical_branch(512);
    let m_seed = universe.compute_seed_condensation(6.0e59).to_f64();
    // Delta N ~ 6e59 bits must condense ~1e9 M_sun per Theorem 9.6.
    assert!(
        (m_seed - 7.95499e8).abs() / 7.95499e8 < 0.05,
        "seed mass must approximate 1e9 M_sun: {}",
        m_seed
    );
    // Landauer debt is exactly 906 GW per M_sun: a 1e9 M_sun seed yields
    // 9.06e11 GW = 9.06e20 W.
    let p_gw = universe.eval_landauer_debt_power(1.0e9).to_f64();
    assert!(
        (p_gw - 9.06e11).abs() < 1.0,
        "Landauer debt must scale to ~9.06e20 W: {} GW",
        p_gw
    );

    let record = SeedDefectRecord {
        position: [10.0, 20.0, 30.0],
        seed_mass_msun: m_seed,
        landauer_debt_gw: p_gw,
        winding_number: 1,
        formation_redshift: 7.0,
        overflow_bits: 6.0e59,
    };
    assert_eq!(record.to_bytes().len(), SEED_RECORD_BYTES);

    let header = MmioTelemetryHeader::from_universe(&universe, 7.0, 0, 65_536, 6.0e59, 1);
    let grid = vec![0.5f32; 256];
    let frame = shbt_simulator::shbt::export::serialize_mmio_frame(
        &header,
        &grid,
        &grid,
        &[record],
    );
    assert_eq!(frame.len(), SHBT_MMIO_HEADER_BYTES + 2048 + SEED_RECORD_BYTES);
}

#[test]
fn test_observer_horizon_freeze() {
    // Gate: as z -> -1 the admissible observer set R_adm -> empty.
    let universe = ShbtUniverse::new_canonical_branch(512);
    let n_local = 2.535748254483999e122;
    let c_get = 64.0;

    let (frozen_early, _) = universe.verify_observer_freeze(0.0, n_local, c_get);
    assert!(!frozen_early, "observer set must be non-empty at z = 0");

    let (frozen_late, residual) = universe.verify_observer_freeze(-0.999999, n_local, c_get);
    assert!(frozen_late, "observer set must freeze as z -> -1");
    assert!(residual <= Float::with_val(512, 0.0));
}
