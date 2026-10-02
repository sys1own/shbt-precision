//! f64 mirror of the SHBT-MMIO telemetry frame (`src/shbt/export.rs`) so the
//! visualizer can produce host-side frames without linking the MPFR core.

use crate::hud::offsets;

pub const N_SAT_BITS: f64 = 3.311997720142366e122;
pub const A_H: f64 = 4.797960072861;
pub const GAMMA_LOCK: f64 = 3.0 * A_H;
pub const H0_CMB: f64 = 67.4;
pub const OMEGA_M: f64 = 0.315;
pub const OMEGA_R0: f64 = 9.2e-5;
pub const ALPHA_SEED_MSUN_PER_BIT: f64 = 1.3258316e-51;
pub const LANDAUER_GW_PER_MSUN: f64 = 906.0;
pub const ETA_DARK: f64 = 23.0 / 33.0;
pub const ETA_VISIBLE: f64 = 10.0 / 33.0;

/// Loaded SHBT Hubble rate H(z) in km s^-1 Mpc^-1.
pub fn hubble(z: f64) -> f64 {
    let one_plus_z = 1.0 + z;
    let expansion = (OMEGA_M * one_plus_z.powi(3)
        + OMEGA_R0 * one_plus_z.powi(4)
        + (1.0 - OMEGA_M - OMEGA_R0))
        .sqrt();
    (H0_CMB + A_H / one_plus_z) * expansion
}

/// Conformal loading fraction on (-1, +inf); matches
/// `ShbtUniverse::conformal_loading_fraction` / `loading_fraction_asymptotic`.
pub fn loading_fraction(z: f64) -> f64 {
    if z <= -1.0 {
        return 1.0;
    }
    if z < 0.0 {
        return 1.0 - (1.0 + z).powi(3);
    }
    if z == 0.0 {
        return 0.0;
    }
    // df_load/du = Gamma e^{-u} / H(e^u - 1), u = ln(1+z); trapezoid.
    let upper = (1.0 + z).ln();
    let n = 4096;
    let du = upper / n as f64;
    let mut acc = 0.0;
    for i in 0..=n {
        let u = du * i as f64;
        let one_plus_z = u.exp();
        let h0_z = H0_CMB + A_H / one_plus_z;
        let expansion = (OMEGA_M * one_plus_z.powi(3)
            + OMEGA_R0 * one_plus_z.powi(4)
            + (1.0 - OMEGA_M - OMEGA_R0))
            .sqrt();
        let w = if i == 0 || i == n { 0.5 } else { 1.0 };
        acc += w * GAMMA_LOCK * (-u).exp() / (h0_z * expansion);
    }
    (du * acc).min(1.0)
}

/// Bulk proper time elapsed (Gyr) at redshift z (loaded matter-Lambda
/// look-back integral; de Sitter asymptote returns +inf).
pub fn bulk_time_gyr(z: f64) -> f64 {
    if z <= -1.0 {
        return f64::INFINITY;
    }
    if z <= 0.0 {
        return 13.276616557;
    }
    let h0_gyr = H0_CMB * 1.0227121650537077e-3;
    let upper = (1.0 + z).ln();
    let n = 1024;
    let du = upper / n as f64;
    let mut acc = 0.0;
    for i in 0..=n {
        let u = du * i as f64;
        let one_plus_z = u.exp();
        let e = (OMEGA_M * one_plus_z.powi(3)
            + OMEGA_R0 * one_plus_z.powi(4)
            + (1.0 - OMEGA_M - OMEGA_R0))
            .sqrt();
        let w = if i == 0 || i == n { 0.5 } else { 1.0 };
        acc += w / e;
    }
    du * acc / h0_gyr
}

/// f_sigma8 damping multiplier [1 - gamma_SHBT f_load(z)].
pub fn growth_suppression(z: f64) -> f64 {
    1.0 - ETA_VISIBLE * loading_fraction(z)
}

/// Logarithmic ISW residual Delta_ISW(z) = 2 A_H Gamma / [H0(z) H(z)].
pub fn delta_isw(z: f64) -> f64 {
    let one_plus_z = 1.0 + z;
    let h0_z = H0_CMB + A_H / one_plus_z;
    2.0 * A_H * GAMMA_LOCK / (h0_z * hubble(z))
}

/// Build one 128-byte SHBT-MMIO telemetry frame for redshift `z`.
pub fn encode_mmio_frame(
    frame_index: u64,
    z: f64,
    particle_count: u64,
    delta_n_bits: f64,
    seed_count: u32,
) -> [u8; 128] {
    let f_load = loading_fraction(z);
    let _ = seed_count;
    let seed_mass = ALPHA_SEED_MSUN_PER_BIT * delta_n_bits;
    let p_debt = seed_mass * LANDAUER_GW_PER_MSUN;
    let mut out = [0u8; 128];
    out[offsets::MAGIC..offsets::MAGIC + 4].copy_from_slice(&0x5442_4853u32.to_le_bytes());
    out[offsets::SCHEMA..offsets::SCHEMA + 4].copy_from_slice(&0x0002_0000u32.to_le_bytes());
    out[offsets::FRAME_INDEX..offsets::FRAME_INDEX + 8]
        .copy_from_slice(&frame_index.to_le_bytes());
    out[offsets::BULK_TIME..offsets::BULK_TIME + 8]
        .copy_from_slice(&bulk_time_gyr(z).to_le_bytes());
    out[offsets::REDSHIFT..offsets::REDSHIFT + 8].copy_from_slice(&z.to_le_bytes());
    out[offsets::SCALE_FACTOR..offsets::SCALE_FACTOR + 8]
        .copy_from_slice(&(1.0 / (1.0 + z)).to_le_bytes());
    out[offsets::HUBBLE..offsets::HUBBLE + 8].copy_from_slice(&hubble(z).to_le_bytes());
    out[offsets::LOADING..offsets::LOADING + 8].copy_from_slice(&f_load.to_le_bytes());
    out[offsets::ACTIVE_BITS..offsets::ACTIVE_BITS + 8]
        .copy_from_slice(&(ETA_VISIBLE * N_SAT_BITS * f_load).to_le_bytes());
    out[offsets::DARK_BITS..offsets::DARK_BITS + 8]
        .copy_from_slice(&(ETA_DARK * N_SAT_BITS * f_load).to_le_bytes());
    out[offsets::LANDAUER..offsets::LANDAUER + 8].copy_from_slice(&p_debt.to_le_bytes());
    out[offsets::SEED_MASS..offsets::SEED_MASS + 8]
        .copy_from_slice(&seed_mass.to_le_bytes());
    out[offsets::FSIGMA8..offsets::FSIGMA8 + 8]
        .copy_from_slice(&growth_suppression(z).to_le_bytes());
    out[offsets::ISW..offsets::ISW + 8].copy_from_slice(&delta_isw(z).to_le_bytes());
    out[offsets::PARTICLES..offsets::PARTICLES + 8]
        .copy_from_slice(&particle_count.to_le_bytes());
    out
}
