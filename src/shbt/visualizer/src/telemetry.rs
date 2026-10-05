//! f64 mirror of the SHBT-MMIO telemetry frame (`src/shbt/export.rs`) so the
//! visualizer can produce host-side frames without linking the MPFR core.
//!
//! shbt13 revision: the forward/backward loading fractions are decoupled
//! (Table 23) — `loading_fraction` is now the forward cosmic-loading
//! fraction f_load_cosmo (capacity-side, saturates at 1 toward the de
//! Sitter asymptote) and `debt_fraction` is the backward-integrated
//! register debt f_load_debt (asymptotes to ~0.10744). The 128-byte frame
//! carries both plus physical lens distances and a CRC32 trailer.

use crate::hud::offsets;

pub const N_SAT_BITS: f64 = 3.311997720142366e122;
pub const A_H: f64 = 4.797960072861;
pub const GAMMA_LOCK: f64 = 3.0 * A_H;
pub const H0_CMB: f64 = 67.4;
pub const OMEGA_M: f64 = 0.315;
pub const OMEGA_R0: f64 = 9.2e-5;
pub const OMEGA_L: f64 = 1.0 - OMEGA_M - OMEGA_R0;
pub const C_KM_S: f64 = 299_792.458;
#[allow(dead_code)]
pub const ALPHA_SEED_MSUN_PER_BIT: f64 = 1.3258316e-51;
#[allow(dead_code)]
pub const LANDAUER_GW_PER_MSUN: f64 = 906.0;
#[allow(dead_code)]
pub const ETA_DARK: f64 = 23.0 / 33.0;
pub const ETA_VISIBLE: f64 = 10.0 / 33.0;
/// MMIO lensing geometry source plane (matches `MMIO_SOURCE_Z` in
/// `src/shbt/export.rs`).
pub const MMIO_SOURCE_Z: f64 = 3.0;
/// Omega_L release flag for the horizon-freeze marker.
pub const MMIO_FLAG_HORIZON_FROZEN: u32 = 1 << 0;
/// Non-empty emergent-defect pool flag.
pub const MMIO_FLAG_HAS_SEEDS: u32 = 1 << 1;

const U_HI: f64 = 40.0;

/// Loaded SHBT Hubble rate H(z) in km s^-1 Mpc^-1.
pub fn hubble(z: f64) -> f64 {
    let one_plus_z = 1.0 + z;
    let expansion = (OMEGA_M * one_plus_z.powi(3)
        + OMEGA_R0 * one_plus_z.powi(4)
        + OMEGA_L)
        .sqrt();
    (H0_CMB + A_H / one_plus_z) * expansion
}

/// Shared conformal kernel Γ_lock e^{-u} / H(e^u - 1) in ln(1+z) space
/// (shbt13 Table 23 kernel).
fn conformal_kernel(u: f64) -> f64 {
    let one_plus_z = u.exp();
    let h0_z = H0_CMB + A_H / one_plus_z;
    let expansion = (OMEGA_M * one_plus_z.powi(3)
        + OMEGA_R0 * one_plus_z.powi(4)
        + OMEGA_L)
        .sqrt();
    GAMMA_LOCK * (-u).exp() / (h0_z * expansion)
}

fn integrate_kernel(u_lo: f64, u_hi: f64, n: usize) -> f64 {
    if u_lo >= u_hi {
        return 0.0;
    }
    let du = (u_hi - u_lo) / n as f64;
    let mut acc = 0.0;
    for i in 0..=n {
        let u = u_lo + du * i as f64;
        let w = if i == 0 || i == n { 0.5 } else { 1.0 };
        acc += w * conformal_kernel(u);
    }
    du * acc
}

/// Backward debt fraction f_load_debt(z) = ∫_0^{ln(1+z)} Γ e^{-u}/H du —
/// the register debt accumulated since z=0; 0 at z<=0, asymptotes to
/// ~0.10744 at the recombination ceiling. Mirror of
/// `ShbtUniverse::evaluate_backward_debt_fraction`.
pub fn debt_fraction(z: f64) -> f64 {
    if z <= 0.0 {
        return 0.0;
    }
    integrate_kernel(0.0, (1.0 + z).ln(), 4096).min(1.0)
}

/// Forward cosmic loading fraction f_load_cosmo(z) =
/// 1 - exp(-∫_{ln(1+z)}^∞ Γ e^{-u}/H du) — the forward capacity loading
/// of the boundary screen from redshift z through the de Sitter
/// asymptote: ~0 at z = 1e14, ~0.102 at z = 0, -> 1 for z -> -1. Mirror
/// of `ShbtUniverse::evaluate_forward_cosmic_loading_fraction`; this is
/// the f_load driving the PM drag coupling and the horizon freeze.
pub fn loading_fraction(z: f64) -> f64 {
    if z <= -1.0 {
        return 1.0;
    }
    let u_lo = (1.0 + z).ln();
    1.0 - (-integrate_kernel(u_lo, U_HI, 4096)).exp()
}

/// Bulk cosmic age t(z) in Gyr — proper time elapsed from the Big Bang to
/// redshift z (monotonic increasing as z decreases):
///     t(z) = (1/H0) * \int_{ln(1+z)}^inf du / E(u).
/// Convergent for every z > -1 (integrand ~ e^{-3u/2} for u -> +inf and
/// saturates at 1/sqrt(Omega_L) for u < 0), +inf at the conformal boundary
/// z <= -1. Mirror of `src/shbt/cosmology.rs::cosmic_age_gyr` (the MPFR core
/// is not linked into this crate).
pub fn bulk_time_gyr(z: f64) -> f64 {
    if z <= -1.0 {
        return f64::INFINITY;
    }
    let u_lo = (1.0 + z).ln();
    if u_lo >= U_HI {
        return 0.0;
    }
    let h0_gyr = H0_CMB * 1.0227121650537077e-3;
    let n = 2048;
    let du = (U_HI - u_lo) / n as f64;
    let mut acc = 0.0;
    for i in 0..=n {
        let u = u_lo + du * i as f64;
        let one_plus_z = u.exp();
        let e = (OMEGA_M * one_plus_z.powi(3)
            + OMEGA_R0 * one_plus_z.powi(4)
            + OMEGA_L)
            .sqrt();
        let w = if i == 0 || i == n { 0.5 } else { 1.0 };
        acc += w / e;
    }
    du * acc / h0_gyr
}

/// Lookback time (Gyr): t_lookback(z) = t(0) - t(z). Positive in the past,
/// negative on the asymptotic future branch (z < 0).
pub fn lookback_gyr(z: f64) -> f64 {
    bulk_time_gyr(0.0) - bulk_time_gyr(z)
}

/// Comoving distance integral D_C(z) = c * int_0^z dz'/H(z'), Mpc
/// (Simpson, mirrors `MmioTelemetryHeader::comoving_distance_mpc`).
pub fn comoving_distance_mpc(z: f64) -> f64 {
    if z <= 0.0 {
        return 0.0;
    }
    let n = 1024;
    let h = z / n as f64;
    let mut acc = 0.0;
    for i in 0..=n {
        let zp = h * i as f64;
        let w = if i == 0 || i == n {
            1.0
        } else if i % 2 == 1 {
            4.0
        } else {
            2.0
        };
        acc += w / hubble(zp);
    }
    C_KM_S * h * acc / 3.0
}

/// Angular-diameter distance D_A(z) = D_C(z)/(1+z), Mpc.
pub fn angular_diameter_mpc(z: f64) -> f64 {
    comoving_distance_mpc(z) / (1.0 + z)
}

/// Lens-source angular-diameter distance D_ds(z_d, z_s), Mpc
/// (flat-distance proxy: D_s - D_d evaluated on the same kernel).
pub fn lens_source_distance_mpc(z_d: f64, z_s: f64) -> f64 {
    (comoving_distance_mpc(z_s) - comoving_distance_mpc(z_d)) / (1.0 + z_s)
}

/// Quintic nucleation envelope Psi_nuc(z): an open floor — exactly 1 for
/// z <= 18 (no suppression below the onset), exactly 0 for z >= 30, and
/// C^2-smooth 6u^5 - 15u^4 + 10u^3 in between. Mirrors
/// `ShbtUniverse::evaluate_psi_nuc` and `compute_nucleation_weight` in
/// seed_emergence.wgsl.
pub fn psi_nuc(z: f64) -> f64 {
    let u = ((30.0 - z) / 12.0).clamp(0.0, 1.0);
    u * u * u * (10.0 + u * (-15.0 + 6.0 * u))
}

/// f_sigma8 damping multiplier [1 - gamma_SHBT f_load_cosmo(z)].
pub fn growth_suppression(z: f64) -> f64 {
    1.0 - ETA_VISIBLE * loading_fraction(z)
}

/// Logarithmic ISW residual Delta_ISW(z) = 2 A_H Gamma / [H0(z) H(z)].
pub fn delta_isw(z: f64) -> f64 {
    let one_plus_z = 1.0 + z;
    let h0_z = H0_CMB + A_H / one_plus_z;
    2.0 * A_H * GAMMA_LOCK / (h0_z * hubble(z))
}

/// IEEE 802.3 CRC-32 (reflected, poly 0xEDB88320) — mirrors
/// `shbt::export::crc32_ieee`; covers frame bytes [0x00..0x78).
pub fn crc32_ieee(bytes: &[u8]) -> u32 {
    let mut crc: u32 = 0xFFFF_FFFF;
    for &b in bytes {
        crc ^= b as u32;
        for _ in 0..8 {
            crc = if crc & 1 != 0 {
                (crc >> 1) ^ 0xEDB8_8320
            } else {
                crc >> 1
            };
        }
    }
    !crc
}

/// Build one 128-byte SHBT-MMIO telemetry frame for redshift `z`, in the
/// shbt13 revised layout (f_load_cosmo @ 0x28, f_load_debt @ 0x30, lens
/// distances 0x40-0x58, seed/caustic counts @ 0x60/0x64, f32 scalars
/// 0x68-0x70, flags @ 0x74, CRC32 trailer @ 0x78).
///
/// `particle_count` and `delta_n_bits` are accepted for API parity with
/// the Tier-1 producer but are not serialized in the v2 frame.
pub fn encode_mmio_frame(
    frame_index: u64,
    z: f64,
    particle_count: u64,
    delta_n_bits: f64,
    seed_count: u32,
) -> [u8; 128] {
    let f_cosmo = loading_fraction(z);
    let f_debt = debt_fraction(z);
    let _ = particle_count;
    let _ = delta_n_bits;
    // Physical lens geometry for the MMIO channel: the defect source plane
    // at z_s = 3.0 against the screen's current epoch.
    let z_d = z.clamp(0.05, MMIO_SOURCE_Z);
    let z_s = MMIO_SOURCE_Z;
    let d_c = comoving_distance_mpc(z.max(0.0));
    let d_d = angular_diameter_mpc(z_d);
    let d_s = angular_diameter_mpc(z_s);
    let d_ds = lens_source_distance_mpc(z_d, z_s);
    let mut flags = 0u32;
    if f_cosmo >= 0.999_999 {
        flags |= MMIO_FLAG_HORIZON_FROZEN;
    }
    if seed_count > 0 {
        flags |= MMIO_FLAG_HAS_SEEDS;
    }

    let mut out = [0u8; 128];
    out[offsets::MAGIC..offsets::MAGIC + 4].copy_from_slice(&0x5442_4853u32.to_le_bytes());
    out[offsets::VERSION..offsets::VERSION + 4].copy_from_slice(&2u32.to_le_bytes());
    out[offsets::FRAME_INDEX..offsets::FRAME_INDEX + 8]
        .copy_from_slice(&frame_index.to_le_bytes());
    out[offsets::REDSHIFT..offsets::REDSHIFT + 8].copy_from_slice(&z.to_le_bytes());
    out[offsets::SCALE_FACTOR..offsets::SCALE_FACTOR + 8]
        .copy_from_slice(&(1.0 / (1.0 + z)).to_le_bytes());
    out[offsets::HUBBLE..offsets::HUBBLE + 8].copy_from_slice(&hubble(z).to_le_bytes());
    out[offsets::F_LOAD_COSMO..offsets::F_LOAD_COSMO + 8]
        .copy_from_slice(&f_cosmo.to_le_bytes());
    out[offsets::F_LOAD_DEBT..offsets::F_LOAD_DEBT + 8]
        .copy_from_slice(&f_debt.to_le_bytes());
    out[offsets::T_LOOKBACK..offsets::T_LOOKBACK + 8]
        .copy_from_slice(&lookback_gyr(z).to_le_bytes());
    out[offsets::D_C..offsets::D_C + 8].copy_from_slice(&d_c.to_le_bytes());
    out[offsets::D_D..offsets::D_D + 8].copy_from_slice(&d_d.to_le_bytes());
    out[offsets::D_S..offsets::D_S + 8].copy_from_slice(&d_s.to_le_bytes());
    out[offsets::D_DS..offsets::D_DS + 8].copy_from_slice(&d_ds.to_le_bytes());
    out[offsets::NUM_SEEDS..offsets::NUM_SEEDS + 4]
        .copy_from_slice(&seed_count.to_le_bytes());
    // Caustic parity (shbt13 Stage 4): the analytically tracked caustic
    // count matches the live defect count — one caustic node per seed.
    out[offsets::NUM_CAUSTICS..offsets::NUM_CAUSTICS + 4]
        .copy_from_slice(&seed_count.to_le_bytes());
    out[offsets::PSI_NUC..offsets::PSI_NUC + 4]
        .copy_from_slice(&(psi_nuc(z) as f32).to_le_bytes());
    out[offsets::DELTA_MAX..offsets::DELTA_MAX + 4]
        .copy_from_slice(&0f32.to_le_bytes());
    out[offsets::D_TAU..offsets::D_TAU + 4].copy_from_slice(&0f32.to_le_bytes());
    out[offsets::FLAGS..offsets::FLAGS + 4].copy_from_slice(&flags.to_le_bytes());
    let crc = crc32_ieee(&out[..offsets::CRC32]);
    out[offsets::CRC32..offsets::CRC32 + 4].copy_from_slice(&crc.to_le_bytes());
    out
}
