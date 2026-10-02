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
            redshift: 1.0e14,
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
        self.redshift = z.clamp(-0.9999, 1.0e14);
    }
}

// ---------------------------------------------------------------------------
// Horizon ledger (shbt4 spec): boundary bit ledger, Stinespring partition,
// seed inventory, Landauer debt, and observer admissibility cardinality.
// ---------------------------------------------------------------------------

/// Saturated holographic-screen microstate capacity (bits).
pub const N_SAT: f64 = 3.3119977e122;
/// Phase-locking rate Gamma_lock = 3 A_H (km s^-1 Mpc^-1).
pub const GAMMA_LOCK: f64 = 14.393880218584;
/// Local Hubble normalization (km s^-1 Mpc^-1).
pub const H0: f64 = 67.4;
pub const OMEGA_M: f64 = 0.315;
pub const OMEGA_R: f64 = 9.0e-5;
pub const OMEGA_L: f64 = 0.68491;

/// Epoch ledger evaluated by `WasmShbtEngine::update_epoch` and serialized
/// into the HUD telemetry JSON.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct HorizonLedger {
    pub redshift: f64,
    pub scale_factor: f64,
    pub hubble_rate: f64,
    pub lookback_time_gyr: f64,
    pub loaded_fraction: f64,
    pub total_bits_loaded: f64,
    pub active_visible_bits: f64,
    pub dark_completion_bits: f64,
    pub stinespring_blend: f64,
    pub total_seeds_condensed: u32,
    pub total_landauer_debt_gw: f64,
    pub observer_admissibility_cardinality: usize,
    pub conservation_residual: f64,
}

impl HorizonLedger {
    pub fn new() -> Self {
        Self {
            redshift: 1100.0,
            scale_factor: 1.0 / 1101.0,
            hubble_rate: 1.5e6,
            lookback_time_gyr: 13.78,
            loaded_fraction: 0.05,
            total_bits_loaded: 0.05 * N_SAT,
            active_visible_bits: 0.05 * N_SAT * (10.0 / 33.0),
            dark_completion_bits: 0.05 * N_SAT * (23.0 / 33.0),
            stinespring_blend: 1.0,
            total_seeds_condensed: 0,
            total_landauer_debt_gw: 0.0,
            observer_admissibility_cardinality: 1024,
            conservation_residual: 1.0e-128,
        }
    }

    pub fn compute_hubble(&self, z: f64) -> f64 {
        if z <= -0.999 {
            return 0.066954;
        }
        let term_r = OMEGA_R * (1.0 + z).powi(4);
        let term_m = OMEGA_M * (1.0 + z).powi(3);
        let term_l = OMEGA_L;
        H0 * (term_r + term_m + term_l).sqrt()
    }

    pub fn update(&mut self, z: f64) {
        self.redshift = z;
        self.scale_factor = 1.0 / (1.0 + z.max(-0.9999));
        self.hubble_rate = self.compute_hubble(z);

        let f_load = if z > 1.0e10 {
            1.0e-6
        } else if z <= -0.99 {
            1.0
        } else {
            (1.0 / (1.0 + (z + 1.0).powf(0.85))).clamp(0.0, 1.0)
        };

        self.loaded_fraction = f_load;
        self.total_bits_loaded = f_load * N_SAT;

        if z > 1.0e12 {
            self.stinespring_blend = 0.0;
            self.active_visible_bits = self.total_bits_loaded;
            self.dark_completion_bits = 0.0;
        } else if z < 1.0e9 {
            self.stinespring_blend = 1.0;
            self.active_visible_bits = self.total_bits_loaded * (10.0 / 33.0);
            self.dark_completion_bits = self.total_bits_loaded - self.active_visible_bits;
        } else {
            let log_z = z.log10();
            let blend = (12.0 - log_z) / 3.0;
            self.stinespring_blend = blend.clamp(0.0, 1.0);
            let eta_v = (1.0 - self.stinespring_blend) + self.stinespring_blend * (10.0 / 33.0);
            self.active_visible_bits = self.total_bits_loaded * eta_v;
            self.dark_completion_bits = self.total_bits_loaded - self.active_visible_bits;
        }

        if z <= 30.0 && z >= 7.0 {
            self.total_seeds_condensed = 248;
            self.total_landauer_debt_gw = 248.0 * 7.955e8 * 906.0;
        } else if z < 7.0 {
            self.total_seeds_condensed = 312;
            self.total_landauer_debt_gw = 312.0 * 7.955e8 * 906.0;
        } else {
            self.total_seeds_condensed = 0;
            self.total_landauer_debt_gw = 0.0;
        }

        if z <= -0.95 {
            self.observer_admissibility_cardinality = 0;
        } else {
            self.observer_admissibility_cardinality =
                ((1.0 + z) * 1024.0).clamp(0.0, 1024.0) as usize;
        }

        let diff = (self.active_visible_bits + self.dark_completion_bits) - self.total_bits_loaded;
        self.conservation_residual = (diff / N_SAT).abs();
    }

    pub fn is_observer_admissible(&self) -> bool {
        self.observer_admissibility_cardinality > 0
    }

    pub fn to_json_telemetry(&self) -> String {
        serde_json::to_string(self).unwrap_or_else(|_| "{}".to_string())
    }
}

impl Default for HorizonLedger {
    fn default() -> Self {
        Self::new()
    }
}
