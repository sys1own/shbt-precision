//! SHBT-MMIO telemetry protocol serializer.
//!
//! Produces the 128-byte header frame plus binary payload consumed by the
//! Tier-2 WebGPU visualizer ring buffer, following the byte offsets defined
//! in `shbt3.txt`.  The header is little-endian throughout.

use pyo3::prelude::*;
use pyo3::types::PyBytes;

use crate::shbt::cosmology::{ShbtUniverse, H0_CMB};

/// Telemetry magic: ASCII 'SHBT' little-endian.
pub const SHBT_MMIO_MAGIC: u32 = 0x5442_4853;
/// Header schema version 2.0.
pub const SHBT_MMIO_SCHEMA: u32 = 0x0002_0000;
/// Header frame size in bytes.
pub const SHBT_MMIO_HEADER_BYTES: usize = 128;
/// Topological seed defect descriptor record size in bytes.
pub const SEED_RECORD_BYTES: usize = 128;
/// Transfer-function and matter-power grid side lengths (f32 values each).
pub const SPECTRUM_GRID_LEN: usize = 256;

/// One telemetry frame header (128 bytes, offsets per `shbt3.txt`).
#[derive(Debug, Clone, Copy, Default)]
pub struct MmioTelemetryHeader {
    /// 0x08: monotonic integration frame index.
    pub frame_index: u64,
    /// 0x10: emergent proper bulk time elapsed (Gyr).
    pub bulk_time_gyr: f64,
    /// 0x18: current comoving bulk slice redshift z.
    pub redshift_z: f64,
    /// 0x20: bulk conformal scale factor a = 1/(1+z).
    pub scale_factor_a: f64,
    /// 0x28: Hubble parameter H_SHBT(z) (km s^-1 Mpc^-1).
    pub hubble_param: f64,
    /// 0x30: boundary conformal loading fraction f_load(z).
    pub loading_frac: f64,
    /// 0x38: active visible register capacity N_vis(z) (bits).
    pub active_bits: f64,
    /// 0x40: dark completion ghost capacity N_dark(z) (bits).
    pub dark_bits: f64,
    /// 0x48: continuous boundary Landauer debt dissipation rate (GW).
    pub landauer_debt_gw: f64,
    /// 0x50: cumulative condensed topological seed mass (M_sun).
    pub smbh_seed_mass: f64,
    /// 0x58: boundary-suppressed perturbation growth rate f_sigma8.
    pub f_sigma8_val: f64,
    /// 0x60: logarithmic ISW residual.
    pub delta_isw_val: f64,
    /// 0x68: active particle simulation count.
    pub particle_count: u64,
}

impl MmioTelemetryHeader {
    /// Sample the header at redshift `z` for frame `frame_index` with
    /// `particle_count` live particles and `seed_count` condensed seeds.
    pub fn from_universe(
        universe: &ShbtUniverse,
        z: f64,
        frame_index: u64,
        particle_count: u64,
        seed_delta_n_bits: f64,
        seed_count: u32,
    ) -> Self {
        let a = 1.0 / (1.0 + z);
        let _ = seed_count;
        let m_seed = universe.compute_seed_condensation(seed_delta_n_bits).to_f64();
        Self {
            frame_index,
            bulk_time_gyr: cosmic_age_gyr(z),
            redshift_z: z,
            scale_factor_a: a,
            hubble_param: universe.hubble(z).to_f64(),
            loading_frac: universe.conformal_loading_fraction(z).to_f64(),
            active_bits: universe.active_visible_bits(z).to_f64(),
            dark_bits: universe.dark_completion_bits(z).to_f64(),
            landauer_debt_gw: universe.eval_landauer_debt_power(m_seed).to_f64(),
            smbh_seed_mass: m_seed,
            f_sigma8_val: universe.evaluate_growth_suppression(z).to_f64(),
            delta_isw_val: universe.delta_isw_residual(z).to_f64(),
            particle_count,
        }
    }

    /// Serialize to the canonical 128-byte little-endian frame.
    pub fn to_bytes(&self) -> [u8; SHBT_MMIO_HEADER_BYTES] {
        let mut out = [0u8; SHBT_MMIO_HEADER_BYTES];
        out[0x00..0x04].copy_from_slice(&SHBT_MMIO_MAGIC.to_le_bytes());
        out[0x04..0x08].copy_from_slice(&SHBT_MMIO_SCHEMA.to_le_bytes());
        out[0x08..0x10].copy_from_slice(&self.frame_index.to_le_bytes());
        out[0x10..0x18].copy_from_slice(&self.bulk_time_gyr.to_le_bytes());
        out[0x18..0x20].copy_from_slice(&self.redshift_z.to_le_bytes());
        out[0x20..0x28].copy_from_slice(&self.scale_factor_a.to_le_bytes());
        out[0x28..0x30].copy_from_slice(&self.hubble_param.to_le_bytes());
        out[0x30..0x38].copy_from_slice(&self.loading_frac.to_le_bytes());
        out[0x38..0x40].copy_from_slice(&self.active_bits.to_le_bytes());
        out[0x40..0x48].copy_from_slice(&self.dark_bits.to_le_bytes());
        out[0x48..0x50].copy_from_slice(&self.landauer_debt_gw.to_le_bytes());
        out[0x50..0x58].copy_from_slice(&self.smbh_seed_mass.to_le_bytes());
        out[0x58..0x60].copy_from_slice(&self.f_sigma8_val.to_le_bytes());
        out[0x60..0x68].copy_from_slice(&self.delta_isw_val.to_le_bytes());
        out[0x68..0x70].copy_from_slice(&self.particle_count.to_le_bytes());
        // 0x70..0x80 reserved_pad: zero-initialized.
        out
    }

    /// Decode a 128-byte frame produced by `to_bytes`.
    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        if bytes.len() < SHBT_MMIO_HEADER_BYTES {
            return None;
        }
        let magic = u32::from_le_bytes(bytes[0x00..0x04].try_into().ok()?);
        if magic != SHBT_MMIO_MAGIC {
            return None;
        }
        let u64_at = |o: usize| u64::from_le_bytes(bytes[o..o + 8].try_into().unwrap());
        let f64_at = |o: usize| f64::from_le_bytes(bytes[o..o + 8].try_into().unwrap());
        Some(Self {
            frame_index: u64_at(0x08),
            bulk_time_gyr: f64_at(0x10),
            redshift_z: f64_at(0x18),
            scale_factor_a: f64_at(0x20),
            hubble_param: f64_at(0x28),
            loading_frac: f64_at(0x30),
            active_bits: f64_at(0x38),
            dark_bits: f64_at(0x40),
            landauer_debt_gw: f64_at(0x48),
            smbh_seed_mass: f64_at(0x50),
            f_sigma8_val: f64_at(0x58),
            delta_isw_val: f64_at(0x60),
            particle_count: u64_at(0x68),
        })
    }
}

/// 128-byte topological seed defect descriptor (payload record).
#[derive(Debug, Clone, Copy, Default)]
pub struct SeedDefectRecord {
    /// Comoving coordinates (Mpc/h).
    pub position: [f64; 3],
    /// Condensed seed mass M_seed (M_sun).
    pub seed_mass_msun: f64,
    /// Landauer dissipation level P_debt (GW).
    pub landauer_debt_gw: f64,
    /// Topological winding number of the register defect.
    pub winding_number: i32,
    /// Formation redshift.
    pub formation_redshift: f64,
    /// Local coordinate bit overflow Delta N that triggered condensation.
    pub overflow_bits: f64,
}

impl SeedDefectRecord {
    /// Serialize to the canonical 128-byte little-endian record.
    pub fn to_bytes(&self) -> [u8; SEED_RECORD_BYTES] {
        let mut out = [0u8; SEED_RECORD_BYTES];
        for (i, coord) in self.position.iter().enumerate() {
            out[i * 8..i * 8 + 8].copy_from_slice(&coord.to_le_bytes());
        }
        out[24..32].copy_from_slice(&self.seed_mass_msun.to_le_bytes());
        out[32..40].copy_from_slice(&self.landauer_debt_gw.to_le_bytes());
        out[40..44].copy_from_slice(&self.winding_number.to_le_bytes());
        out[44..52].copy_from_slice(&self.formation_redshift.to_le_bytes());
        out[52..60].copy_from_slice(&self.overflow_bits.to_le_bytes());
        // 60..128 reserved.
        out
    }
}

/// Serialize one complete telemetry frame: 128-byte header, followed by the
/// 256 x f32 transfer grid T(k,z), the 256 x f32 matter-power grid P_m(k,z),
/// and the seed descriptor records.
pub fn serialize_mmio_frame(
    header: &MmioTelemetryHeader,
    transfer_grid: &[f32],
    power_grid: &[f32],
    seeds: &[SeedDefectRecord],
) -> Vec<u8> {
    debug_assert_eq!(transfer_grid.len(), SPECTRUM_GRID_LEN);
    debug_assert_eq!(power_grid.len(), SPECTRUM_GRID_LEN);
    let mut frame = Vec::with_capacity(
        SHBT_MMIO_HEADER_BYTES
            + (transfer_grid.len() + power_grid.len()) * 4
            + seeds.len() * SEED_RECORD_BYTES,
    );
    frame.extend_from_slice(&header.to_bytes());
    for value in transfer_grid {
        frame.extend_from_slice(&value.to_le_bytes());
    }
    for value in power_grid {
        frame.extend_from_slice(&value.to_le_bytes());
    }
    for seed in seeds {
        frame.extend_from_slice(&seed.to_bytes());
    }
    frame
}

/// PyO3 wrapper: build and serialize one telemetry frame.
///
/// `seeds` items are 8-tuples `(x, y, z, seed_mass_msun, landauer_debt_gw,
/// winding_number, formation_redshift, overflow_bits)`.
#[pyfunction]
#[pyo3(signature = (frame_index, redshift, particle_count, seed_delta_n_bits, transfer_grid, power_grid, seeds))]
pub fn serialize_mmio_frame_py(
    py: Python<'_>,
    frame_index: u64,
    redshift: f64,
    particle_count: u64,
    seed_delta_n_bits: f64,
    transfer_grid: Vec<f32>,
    power_grid: Vec<f32>,
    seeds: Vec<(f64, f64, f64, f64, f64, i32, f64, f64)>,
) -> PyResult<Py<PyBytes>> {
    if transfer_grid.len() != SPECTRUM_GRID_LEN || power_grid.len() != SPECTRUM_GRID_LEN {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "grids must hold {} f32 values",
            SPECTRUM_GRID_LEN
        )));
    }
    let universe = ShbtUniverse::new_canonical_branch(512);
    let header = MmioTelemetryHeader::from_universe(
        &universe,
        redshift,
        frame_index,
        particle_count,
        seed_delta_n_bits,
        seeds.len() as u32,
    );
    let records: Vec<SeedDefectRecord> = seeds
        .into_iter()
        .map(|(x, y, z, mass, debt, winding, form_z, overflow)| SeedDefectRecord {
            position: [x, y, z],
            seed_mass_msun: mass,
            landauer_debt_gw: debt,
            winding_number: winding,
            formation_redshift: form_z,
            overflow_bits: overflow,
        })
        .collect();
    let frame = serialize_mmio_frame(&header, &transfer_grid, &power_grid, &records);
    Ok(PyBytes::new_bound(py, &frame).unbind())
}

/// Approximate bulk proper time elapsed (Gyr) at redshift z under the loaded
/// matter-Lambda background (fast analytic look-back integral).
pub fn cosmic_age_gyr(z: f64) -> f64 {
    if z <= -1.0 {
        return f64::INFINITY;
    }
    // Look-back time in Gyr: t = (1/H0) ∫_0^z dz'/[(1+z') E(z')].
    // Integrating in u = ln(1+z') gives dz'/(1+z') = du, so t = (1/H0) ∫ du/E(u).
    let h0_per_gyr_inv = H0_CMB * 1.0227121650537077e-3; // km/s/Mpc -> 1/Gyr
    let n = 512;
    let upper = z.max(0.0);
    if upper == 0.0 {
        return 13.276616557;
    }
    let du = (1.0 + upper).ln() / n as f64;
    let mut acc = 0.0;
    for i in 0..=n {
        let u = du * i as f64;
        let one_plus_z = u.exp();
        let e = (0.315_f64 * one_plus_z.powi(3)
            + 9.2e-5_f64 * one_plus_z.powi(4)
            + (1.0 - 0.315 - 9.2e-5))
            .sqrt();
        let w = if i == 0 || i == n { 1.0 } else if i % 2 == 0 { 2.0 } else { 4.0 };
        acc += w / e;
    }
    du * acc / 3.0 / h0_per_gyr_inv
}
