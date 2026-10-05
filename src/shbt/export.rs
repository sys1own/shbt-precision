//! SHBT-MMIO telemetry protocol serializer.
//!
//! Produces the 128-byte header frame plus binary payload consumed by the
//! Tier-2 WebGPU visualizer ring buffer, following the byte offsets defined
//! in `shbt3.txt`.  The header is little-endian throughout.

use pyo3::prelude::*;
use pyo3::types::PyBytes;

use crate::shbt::cosmology::ShbtUniverse;

/// Telemetry magic: ASCII 'SHBT' in frame order (bytes 53 48 42 54).
pub const SHBT_MMIO_MAGIC: u32 = 0x5442_4853;
/// Header schema version 2 (shbt13 revised layout).
pub const SHBT_MMIO_SCHEMA: u32 = 0x0000_0002;
/// Header frame size in bytes.
pub const SHBT_MMIO_HEADER_BYTES: usize = 128;
/// Topological seed defect descriptor record size in bytes.
pub const SEED_RECORD_BYTES: usize = 128;
/// Transfer-function and matter-power grid side lengths (f32 values each).
pub const SPECTRUM_GRID_LEN: usize = 256;
/// Canonical thin-screen source-plane redshift used for the D_s/D_ds
/// telemetry columns (the frame's own slice is the deflector z_d = z).
pub const MMIO_SOURCE_Z: f64 = 3.0;
/// Speed of light (km/s) for the comoving-distance quadrature.
pub const C_KM_S: f64 = 299_792.458;

/// Status-flag bits serialized at offset 0x74.
pub const MMIO_FLAG_HORIZON_FROZEN: u32 = 1 << 0;
pub const MMIO_FLAG_HAS_SEEDS: u32 = 1 << 1;

/// One telemetry frame header (128 bytes, shbt13 revised layout).
///
/// Offset map (all little-endian):
///   0x00 magic u32 | 0x04 version u32 | 0x08 frame_index u64
///   0x10 z f64 | 0x18 a f64 | 0x20 h_z f64
///   0x28 f_load_cosmo f64 | 0x30 f_load_debt f64 | 0x38 t_lookback_gyr f64
///   0x40 d_c f64 | 0x48 d_d f64 | 0x50 d_s f64 | 0x58 d_ds f64
///   0x60 num_seeds u32 | 0x64 num_caustics u32
///   0x68 psi_nuc f32 | 0x6C delta_max f32 | 0x70 d_tau f32 | 0x74 flags u32
///   0x78 checksum_crc32 u32 (CRC32 over bytes 0x00..0x77)
///   0x7C reserved_pad u32
#[derive(Debug, Clone, Copy, Default)]
pub struct MmioTelemetryHeader {
    /// 0x08: monotonic integration frame index.
    pub frame_index: u64,
    /// 0x10: current comoving bulk slice redshift z.
    pub redshift_z: f64,
    /// 0x18: bulk conformal scale factor a = 1/(1+z).
    pub scale_factor_a: f64,
    /// 0x20: Hubble parameter H_SHBT(z) (km s^-1 Mpc^-1).
    pub hubble_param: f64,
    /// 0x28: forward cosmic capacity loading fraction f_load^cosmo(z).
    pub f_load_cosmo: f64,
    /// 0x30: backward past-light-cone lookback debt fraction f_load^debt(z).
    pub f_load_debt: f64,
    /// 0x38: cosmological lookback time t_lookback(z) = t(0) - t(z) in Gyr.
    pub t_lookback_gyr: f64,
    /// 0x40: line-of-sight comoving distance D_c(z) in Mpc.
    pub d_c: f64,
    /// 0x48: angular diameter distance to the deflector plane D_d (Mpc).
    pub d_d: f64,
    /// 0x50: angular diameter distance to the source plane D_s (Mpc).
    pub d_s: f64,
    /// 0x58: deflector-to-source angular diameter distance D_ds (Mpc).
    pub d_ds: f64,
    /// 0x60: active emergent topological defect count.
    pub num_seeds: u32,
    /// 0x64: active gravitational caustic count (1:1 with seed defects).
    pub num_caustics: u32,
    /// 0x68: quintic nucleation partition factor Psi_nuc(z).
    pub psi_nuc: f32,
    /// 0x6C: maximum comoving density contrast observed this frame.
    pub delta_max: f32,
    /// 0x70: supercomoving integration time-step d_tau.
    pub d_tau: f32,
    /// 0x74: simulation status and lifecycle bitmask.
    pub flags: u32,
    // 0x78 checksum is computed by to_bytes; 0x7C stays zero.
}

/// IEEE 802.3 CRC-32 (reflected polynomial 0xEDB88320) over `data`.
pub fn crc32_ieee(data: &[u8]) -> u32 {
    let mut crc: u32 = 0xFFFF_FFFF;
    for &byte in data {
        crc ^= byte as u32;
        for _ in 0..8 {
            let mask = (crc & 1).wrapping_neg();
            crc = (crc >> 1) ^ (0xEDB8_8320 & mask);
        }
    }
    !crc
}

/// Line-of-sight comoving distance D_c(z) = c int_0^z dz' / H_SHBT(z')
/// in Mpc (Simpson quadrature over the Tier-1 kernel; mirrors
/// `units.rs::CosmologicalContext::comoving_distance_mpc` in the
/// visualizer and `boltzmann_shbt._comoving_distance_mpc`).
pub fn comoving_distance_mpc(universe: &ShbtUniverse, z: f64) -> f64 {
    if z <= 0.0 {
        return 0.0;
    }
    let n = 1024usize;
    let dz = z / n as f64;
    let mut acc = 0.0;
    for i in 0..=n {
        let zi = dz * i as f64;
        let w = match i {
            0 => 1.0,
            _ if i == n => 1.0,
            _ if i % 2 == 1 => 4.0,
            _ => 2.0,
        };
        acc += w / universe.hubble(zi).to_f64();
    }
    C_KM_S * dz / 3.0 * acc
}

/// Angular diameter distance D_A(z) = D_c(z)/(1+z) (Etherington duality).
pub fn angular_diameter_mpc(universe: &ShbtUniverse, z: f64) -> f64 {
    comoving_distance_mpc(universe, z) / (1.0 + z)
}

/// Lens-to-source angular diameter distance
/// D_ds = (D_c(z_s) - D_c(z_d)) / (1 + z_s) in Mpc.
pub fn lens_source_distance_mpc(universe: &ShbtUniverse, z_d: f64, z_s: f64) -> f64 {
    let dchi = (comoving_distance_mpc(universe, z_s) - comoving_distance_mpc(universe, z_d))
        .max(0.0);
    dchi / (1.0 + z_s)
}

impl MmioTelemetryHeader {
    /// Sample the header at redshift `z` for frame `frame_index`.
    ///
    /// `num_seeds` is the live topological defect count; `num_caustics`
    /// mirrors it 1:1 (the caustic population is generated by the seed
    /// defects). `particle_count` remains a call-site argument for API
    /// compatibility with the legacy frame but is not serialized in the
    /// revised layout (the payload particle block carries the field data).
    pub fn from_universe(
        universe: &ShbtUniverse,
        z: f64,
        frame_index: u64,
        particle_count: u64,
        seed_delta_n_bits: f64,
        seed_count: u32,
    ) -> Self {
        let _ = particle_count;
        let _ = seed_delta_n_bits;
        let a = 1.0 / (1.0 + z);
        let f_cosmo = universe
            .evaluate_forward_cosmic_loading_fraction(z)
            .to_f64();
        let d_c = comoving_distance_mpc(universe, z);
        let d_d = angular_diameter_mpc(universe, z);
        let d_s = angular_diameter_mpc(universe, MMIO_SOURCE_Z);
        let d_ds = lens_source_distance_mpc(universe, z.min(MMIO_SOURCE_Z), MMIO_SOURCE_Z);
        let mut flags = 0u32;
        if f_cosmo >= 0.999_9 {
            flags |= MMIO_FLAG_HORIZON_FROZEN;
        }
        if seed_count > 0 {
            flags |= MMIO_FLAG_HAS_SEEDS;
        }
        Self {
            frame_index,
            redshift_z: z,
            scale_factor_a: a,
            hubble_param: universe.hubble(z).to_f64(),
            f_load_cosmo: f_cosmo,
            f_load_debt: universe.evaluate_backward_debt_fraction(z).to_f64(),
            t_lookback_gyr: lookback_gyr(z),
            d_c,
            d_d,
            d_s,
            d_ds,
            num_seeds: seed_count,
            num_caustics: seed_count,
            psi_nuc: ShbtUniverse::evaluate_psi_nuc(z) as f32,
            delta_max: 0.0,
            d_tau: 0.0,
            flags,
        }
    }

    /// Serialize to the canonical 128-byte little-endian frame.
    pub fn to_bytes(&self) -> [u8; SHBT_MMIO_HEADER_BYTES] {
        let mut out = [0u8; SHBT_MMIO_HEADER_BYTES];
        out[0x00..0x04].copy_from_slice(&SHBT_MMIO_MAGIC.to_le_bytes());
        out[0x04..0x08].copy_from_slice(&SHBT_MMIO_SCHEMA.to_le_bytes());
        out[0x08..0x10].copy_from_slice(&self.frame_index.to_le_bytes());
        out[0x10..0x18].copy_from_slice(&self.redshift_z.to_le_bytes());
        out[0x18..0x20].copy_from_slice(&self.scale_factor_a.to_le_bytes());
        out[0x20..0x28].copy_from_slice(&self.hubble_param.to_le_bytes());
        out[0x28..0x30].copy_from_slice(&self.f_load_cosmo.to_le_bytes());
        out[0x30..0x38].copy_from_slice(&self.f_load_debt.to_le_bytes());
        out[0x38..0x40].copy_from_slice(&self.t_lookback_gyr.to_le_bytes());
        out[0x40..0x48].copy_from_slice(&self.d_c.to_le_bytes());
        out[0x48..0x50].copy_from_slice(&self.d_d.to_le_bytes());
        out[0x50..0x58].copy_from_slice(&self.d_s.to_le_bytes());
        out[0x58..0x60].copy_from_slice(&self.d_ds.to_le_bytes());
        out[0x60..0x64].copy_from_slice(&self.num_seeds.to_le_bytes());
        out[0x64..0x68].copy_from_slice(&self.num_caustics.to_le_bytes());
        out[0x68..0x6C].copy_from_slice(&self.psi_nuc.to_le_bytes());
        out[0x6C..0x70].copy_from_slice(&self.delta_max.to_le_bytes());
        out[0x70..0x74].copy_from_slice(&self.d_tau.to_le_bytes());
        out[0x74..0x78].copy_from_slice(&self.flags.to_le_bytes());
        let crc = crc32_ieee(&out[0x00..0x78]);
        out[0x78..0x7C].copy_from_slice(&crc.to_le_bytes());
        // 0x7C reserved_pad: zero-initialized.
        out
    }

    /// Decode a 128-byte frame produced by `to_bytes`. The CRC32 trailer
    /// is verified: a corrupt frame decodes to None.
    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        if bytes.len() < SHBT_MMIO_HEADER_BYTES {
            return None;
        }
        let magic = u32::from_le_bytes(bytes[0x00..0x04].try_into().ok()?);
        if magic != SHBT_MMIO_MAGIC {
            return None;
        }
        let crc_stored = u32::from_le_bytes(bytes[0x78..0x7C].try_into().ok()?);
        if crc_stored != crc32_ieee(&bytes[0x00..0x78]) {
            return None;
        }
        let u64_at = |o: usize| u64::from_le_bytes(bytes[o..o + 8].try_into().unwrap());
        let f64_at = |o: usize| f64::from_le_bytes(bytes[o..o + 8].try_into().unwrap());
        let u32_at = |o: usize| u32::from_le_bytes(bytes[o..o + 4].try_into().unwrap());
        let f32_at = |o: usize| f32::from_le_bytes(bytes[o..o + 4].try_into().unwrap());
        Some(Self {
            frame_index: u64_at(0x08),
            redshift_z: f64_at(0x10),
            scale_factor_a: f64_at(0x18),
            hubble_param: f64_at(0x20),
            f_load_cosmo: f64_at(0x28),
            f_load_debt: f64_at(0x30),
            t_lookback_gyr: f64_at(0x38),
            d_c: f64_at(0x40),
            d_d: f64_at(0x48),
            d_s: f64_at(0x50),
            d_ds: f64_at(0x58),
            num_seeds: u32_at(0x60),
            num_caustics: u32_at(0x64),
            psi_nuc: f32_at(0x68),
            delta_max: f32_at(0x6C),
            d_tau: f32_at(0x70),
            flags: u32_at(0x74),
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

/// Bulk cosmic age (Gyr) at redshift z and lookback time
/// t_lookback(z) = t(0) - t(z) — canonical quadratures live in `cosmology`;
/// re-exported here (a same-named wrapper would collide under the crate's
/// glob re-exports).
pub use crate::shbt::cosmology::{cosmic_age_gyr, lookback_gyr};
