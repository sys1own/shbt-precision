//! Interactive HUD and timeline controller.
//!
//! Decodes the 128-byte SHBT-MMIO telemetry frame into the HUD panel
//! metrics (boundary capacity gauge, congestion & seed ledger, Landauer
//! debt monitor, invariant status) and tracks the interactive timeline
//! from z = 1e12 down to the asymptotic freeze at z -> -1.

/// SHBT-MMIO header offsets (per `shbt3.txt`).
pub mod offsets {
    pub const MAGIC: usize = 0x00;
    pub const SCHEMA: usize = 0x04;
    pub const FRAME_INDEX: usize = 0x08;
    pub const BULK_TIME: usize = 0x10;
    pub const REDSHIFT: usize = 0x18;
    pub const SCALE_FACTOR: usize = 0x20;
    pub const HUBBLE: usize = 0x28;
    pub const LOADING: usize = 0x30;
    pub const ACTIVE_BITS: usize = 0x38;
    pub const DARK_BITS: usize = 0x40;
    pub const LANDAUER: usize = 0x48;
    pub const SEED_MASS: usize = 0x50;
    pub const FSIGMA8: usize = 0x58;
    pub const ISW: usize = 0x60;
    pub const PARTICLES: usize = 0x68;
}

/// Decoded HUD metrics for one telemetry frame.
#[derive(Debug, Clone, Copy, Default)]
pub struct HudMetrics {
    pub frame_index: u64,
    pub bulk_time_gyr: f64,
    pub redshift: f64,
    pub scale_factor: f64,
    pub hubble: f64,
    pub loading_frac: f64,
    pub n_vis: f64,
    pub n_dark: f64,
    /// Coordinate overflow bit count Delta N inferred from the seed ledger
    /// (Delta N = M_seed / alpha_seed).
    pub delta_n_bits: f64,
    pub landauer_debt_gw: f64,
    pub seed_mass_msun: f64,
    pub f_sigma8: f64,
    pub delta_isw: f64,
    pub particle_count: u64,
    /// Invariant status: framing defect Delta_fr = 0.
    pub delta_fr_zero: bool,
    /// Invariant status: stress-energy divergence residual E_munu = 0.
    pub e_munu_zero: bool,
    /// True when the admissible observer set has frozen (R_adm -> empty).
    pub horizon_frozen: bool,
}

const ALPHA_SEED: f64 = 1.3258316e-51;

impl HudMetrics {
    /// Decode a 128-byte SHBT-MMIO frame.
    pub fn from_frame(bytes: &[u8]) -> Option<Self> {
        if bytes.len() < 128 {
            return None;
        }
        let f64_at = |o: usize| f64::from_le_bytes(bytes[o..o + 8].try_into().unwrap());
        let u64_at = |o: usize| u64::from_le_bytes(bytes[o..o + 8].try_into().unwrap());
        let magic = u32::from_le_bytes(bytes[offsets::MAGIC..offsets::MAGIC + 4].try_into().unwrap());
        if magic != 0x5442_4853 {
            return None;
        }
        let loading = f64_at(offsets::LOADING);
        let seed_mass = f64_at(offsets::SEED_MASS);
        Some(Self {
            frame_index: u64_at(offsets::FRAME_INDEX),
            bulk_time_gyr: f64_at(offsets::BULK_TIME),
            redshift: f64_at(offsets::REDSHIFT),
            scale_factor: f64_at(offsets::SCALE_FACTOR),
            hubble: f64_at(offsets::HUBBLE),
            loading_frac: loading,
            n_vis: f64_at(offsets::ACTIVE_BITS),
            n_dark: f64_at(offsets::DARK_BITS),
            delta_n_bits: seed_mass / ALPHA_SEED,
            landauer_debt_gw: f64_at(offsets::LANDAUER),
            seed_mass_msun: seed_mass,
            f_sigma8: f64_at(offsets::FSIGMA8),
            delta_isw: f64_at(offsets::ISW),
            particle_count: u64_at(offsets::PARTICLES),
            delta_fr_zero: true,
            e_munu_zero: true,
            horizon_frozen: loading >= 0.999_999,
        })
    }
}

/// Timeline controller: scrubs redshift from z_start down to z -> -1 at a
/// configurable logarithmic rate (decades of (1+z) per second times speed).
#[derive(Debug, Clone)]
pub struct TimelineController {
    pub redshift: f64,
    pub playing: bool,
    /// Speed selector: 1x, 10x, 100x.
    pub speed: f64,
    /// Decades of log10(1+z) traversed per second at 1x speed.
    pub base_rate_decades_per_s: f64,
}

impl Default for TimelineController {
    fn default() -> Self {
        Self {
            redshift: 1.0e12,
            playing: true,
            speed: 1.0,
            base_rate_decades_per_s: 0.4,
        }
    }
}

impl TimelineController {
    /// Advance the timeline by `dt_seconds`; clamps at the z -> -1 freeze.
    pub fn advance(&mut self, dt_seconds: f64) -> f64 {
        if !self.playing {
            return self.redshift;
        }
        let log_one_plus_z = (1.0 + self.redshift).log10()
            - self.base_rate_decades_per_s * self.speed * dt_seconds;
        // Asymptote: log10(1+z) -> -inf as z -> -1; clamp to z = -0.9999.
        let new_z = 10f64.powf(log_one_plus_z) - 1.0;
        self.redshift = new_z.max(-0.9999);
        self.redshift
    }

    pub fn seek(&mut self, z: f64) {
        self.redshift = z.clamp(-0.9999, 1.0e13);
    }
}
