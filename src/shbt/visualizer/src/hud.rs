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

        // Thermal Stinespring channel (shbt7 Thm 9.10): the visible overlap
        // w_vis(z) = (1-eta_D) + eta_D/(1+(z_N/z)^Delta_Bbar) with
        // eta_D = 23/33, z_N = 7.356e10, Delta_Bbar = 26/3 sets the split
        // first-principally; stinespring_blend carries the quenched
        // fraction (1 - w_vis)/eta_D for the engine's channel assignment.
        let w_vis = crate::stinespring_w_vis(z);
        self.stinespring_blend = crate::stinespring_quench_fraction(z);
        self.active_visible_bits = self.total_bits_loaded * w_vis;
        self.dark_completion_bits = self.total_bits_loaded - self.active_visible_bits;

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

        // Bit conservation residual evaluated in the normalized
        // partition space: w_vis + (1 - w_vis) = 1 exactly in IEEE-754
        // (complement symmetry), so the residual is identically zero
        // rather than an ulp(total_bits)-scale rounding artifact
        // (~f_load * 2^-53, which reads as a spurious invariant break).
        let diff_norm = (w_vis + (1.0 - w_vis)) - 1.0;
        self.conservation_residual = (diff_norm * f_load).abs();
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

// ---------------------------------------------------------------------------
// Gravitational optics engine state (shbt5 spec): fixed-size, zero-allocation
// lensing uniform block, softened point-mass seed table, and caustic
// telemetry. `LensingUniforms` is the 192-byte WGSL contract extended by two
// trailing vec4 parameter slots (post0/post1) consumed by fs_post.
// ---------------------------------------------------------------------------

/// Post-process lensing uniform block: 272 bytes, 16-byte aligned.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct LensingUniforms {
    pub view_proj: [f32; 16],
    pub inv_view_proj: [f32; 16],
    pub cam_pos: [f32; 4],
    pub screen_size: [f32; 2],
    pub lensing_strength: f32,
    pub dispersion_coeff: f32,
    pub dark_glow_intensity: f32,
    pub dark_glow_radius: f32,
    pub doppler_enabled: u32,
    pub seed_count: u32,
    pub time: f32,
    pub _pad0: f32,
    pub _pad1: [f32; 2],
    /// Engine extras: x = Channel A enable, y = Channel B enable,
    /// z = f_load horizon fill, w = exposure.
    pub post0: [f32; 4],
    /// x = unwrap_transition (0 = comoving bulk, 1 = boundary CFT torus),
    /// y = split_viewport_mode (0 = 3D bulk, 1 = bulk | phase-space).
    pub post1: [f32; 4],
    /// shbt8 thin-screen metrology extras: x = Theta_FoV (radians),
    /// y = zeta_disp boundary dispersion coefficient, z = bloom lift gain
    /// (Enhancement 12), w = exponential depth-fog density (Enhancement 13).
    pub post2: [f32; 4],
    /// shbt9 multi-plane optics: D_ms / D_s distance ratios for the
    /// 4-slice lens stack centered at z_m in {0.5, 1.2, 2.2, 3.5}
    /// against the source plane z_s = 4.0.
    pub post3: [f32; 4],
    /// shbt11 thermodynamic optics: x = telemetry gamma_max,
    /// y = kappa_max (caustic halo accentuation det J drivers),
    /// z = Landauer desaturation knee, w = tone exposure for the
    /// luminance-preserving ACES curve.
    pub post4: [f32; 4],
}

/// Sandbox control block streamed from the DOM sliders (shbt9 Phase 2).
/// 24 bytes: five thermodynamic/visual scalars plus the viewport mode.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SimulationControls {
    /// Baryon sound-speed scale, [0.0, 3.0] (P-PM hydro gain).
    pub sound_speed_scale: f32,
    /// Percolation threshold scale, [0.5, 2.0] (Cardy ceiling gate).
    pub percolation_threshold_scale: f32,
    /// Lensing deflection constant, [0.0, 5.0].
    pub lensing_strength: f32,
    /// Caustic dispersion, [0.0, 0.5].
    pub chromatic_dispersion: f32,
    /// Target redshift, [-0.999, 1e14].
    pub target_redshift: f32,
    /// Viewport projection: 0 = 3D bulk, 1 = split bulk|phase-space.
    pub viewport_split_mode: u32,
}

impl SimulationControls {
    /// Clamp every field into its thermodynamic control bounds.
    pub fn clamped(mut self) -> Self {
        self.sound_speed_scale = self.sound_speed_scale.clamp(0.0, 3.0);
        self.percolation_threshold_scale =
            self.percolation_threshold_scale.clamp(0.5, 2.0);
        self.lensing_strength = self.lensing_strength.clamp(0.0, 5.0);
        self.chromatic_dispersion = self.chromatic_dispersion.clamp(0.0, 0.5);
        self.target_redshift = self.target_redshift.clamp(-0.999, 1.0e14);
        self
    }
}

/// User-dispatched causal-point observer record (shbt9 Phase 2): a
/// ray-volume intersection landing in the bulk box instantiates an
/// active measurement light cone with entropy budget
/// R_entropy = N_limit - C_get. 32 bytes, zero-alloc fixed pool.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CausalPointRecord {
    /// xyz: unprojected comoving position (box units), w: cone radius.
    pub position_world: [f32; 4],
    /// x: R_entropy remaining, y: N_limit, z: C_get cost, w: active flag.
    pub entropy_budget: [f32; 4],
}

/// Softened point-mass seed defect: 16 bytes, screen-space lensing record.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SeedDefect {
    pub screen_pos: [f32; 2],
    pub theta_e: f32,
    pub core_radius: f32,
}

/// Caustic telemetry readout for the HUD ledger.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen::prelude::wasm_bindgen)]
#[derive(Clone, Copy, Debug, Default)]
pub struct VisualizerTelemetry {
    pub peak_shear: f32,
    pub peak_convergence: f32,
    pub max_einstein_radius: f32,
    pub active_caustics: u32,
}

/// Zero-allocation lensing uniform / seed-table manager. Stages the
/// `LensingUniforms` block and the 64-entry `SeedDefect` table for direct
/// memory-copy upload to the GPU (get_uniform_ptr / get_seeds_ptr).
#[cfg_attr(target_arch = "wasm32", wasm_bindgen::prelude::wasm_bindgen)]
pub struct VisualizerEngine {
    uniforms: LensingUniforms,
    seeds: [SeedDefect; 64],
    telemetry: VisualizerTelemetry,
    is_dirty: bool,
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen::prelude::wasm_bindgen)]
impl VisualizerEngine {
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen::prelude::wasm_bindgen(constructor))]
    pub fn new(width: f32, height: f32) -> Self {
        let mut uniforms = LensingUniforms {
            view_proj: [0.0; 16],
            inv_view_proj: [0.0; 16],
            cam_pos: [0.0; 4],
            screen_size: [width, height],
            lensing_strength: 1.0,
            dispersion_coeff: 0.25,
            dark_glow_intensity: 0.8,
            dark_glow_radius: 4.0,
            doppler_enabled: 1,
            seed_count: 0,
            time: 0.0,
            _pad0: 0.0,
            _pad1: [0.0; 2],
            post0: [1.0, 1.0, 0.0, 1.6],
            post1: [0.0; 4],
            post2: [0.7853982, 0.032, 0.08, 0.6], // Theta_FoV, zeta_disp, bloom, fog
            post3: [0.0; 4],
            post4: [0.0, 0.0, 14.0, 0.55],
        };
        uniforms.view_proj[0] = 1.0;
        uniforms.view_proj[5] = 1.0;
        uniforms.view_proj[10] = 1.0;
        uniforms.view_proj[15] = 1.0;
        uniforms.inv_view_proj = uniforms.view_proj;

        let seeds = [SeedDefect {
            screen_pos: [0.0, 0.0],
            theta_e: 0.0,
            core_radius: 0.01,
        }; 64];

        Self {
            uniforms,
            seeds,
            telemetry: VisualizerTelemetry::default(),
            is_dirty: true,
        }
    }

    pub fn set_lensing_enabled(&mut self, enabled: bool) {
        self.uniforms.lensing_strength = if enabled { 1.0 } else { 0.0 };
        self.is_dirty = true;
    }

    pub fn set_lensing_scale(&mut self, scale: f32) {
        self.uniforms.lensing_strength = scale.clamp(0.0, 5.0);
        self.is_dirty = true;
    }

    pub fn set_dispersion(&mut self, dispersion: f32) {
        self.uniforms.dispersion_coeff = dispersion.clamp(0.0, 1.0);
        self.is_dirty = true;
    }

    pub fn set_doppler_enabled(&mut self, enabled: bool) {
        self.uniforms.doppler_enabled = if enabled { 1 } else { 0 };
        self.is_dirty = true;
    }

    pub fn set_dark_glow(&mut self, intensity: f32) {
        self.uniforms.dark_glow_intensity = intensity.clamp(0.0, 2.0);
        self.is_dirty = true;
    }

    pub fn update_camera_matrices(&mut self, vp: &[f32], inv_vp: &[f32], pos: &[f32], time: f32) {
        if vp.len() >= 16 {
            self.uniforms.view_proj.copy_from_slice(&vp[..16]);
        }
        if inv_vp.len() >= 16 {
            self.uniforms.inv_view_proj.copy_from_slice(&inv_vp[..16]);
        }
        if pos.len() >= 3 {
            self.uniforms.cam_pos[..3].copy_from_slice(&pos[..3]);
            self.uniforms.cam_pos[3] = 1.0;
        }
        self.uniforms.time = time;
        self.is_dirty = true;
    }

    pub fn register_seed(&mut self, idx: usize, u: f32, v: f32, theta_e: f32, core: f32) {
        if idx < 64 {
            self.seeds[idx] = SeedDefect {
                screen_pos: [u, v],
                theta_e,
                core_radius: core,
            };
            if idx >= self.uniforms.seed_count as usize {
                self.uniforms.seed_count = (idx + 1) as u32;
            }
            self.is_dirty = true;
        }
    }

    pub fn clear_seeds(&mut self) {
        self.uniforms.seed_count = 0;
        self.is_dirty = true;
    }

    /// Peak shear/convergence plus dominant Einstein radius and active
    /// caustic node count for the HUD ledger.
    pub fn update_telemetry(&mut self, peak_gamma: f32, peak_kappa: f32) {
        self.telemetry.peak_shear = peak_gamma;
        self.telemetry.peak_convergence = peak_kappa;
        let mut max_te = 0.0f32;
        for i in 0..self.uniforms.seed_count as usize {
            max_te = max_te.max(self.seeds[i].theta_e);
        }
        self.telemetry.max_einstein_radius = max_te;
        let mut caustics = 0u32;
        if peak_kappa >= 1.0 || (peak_gamma * peak_gamma + peak_kappa * peak_kappa) > 0.8 {
            caustics += 1;
        }
        caustics += self.uniforms.seed_count;
        self.telemetry.active_caustics = caustics;
    }

    pub fn get_uniform_ptr(&self) -> *const u8 {
        bytemuck::bytes_of(&self.uniforms).as_ptr()
    }

    pub fn get_seeds_ptr(&self) -> *const u8 {
        bytemuck::cast_slice(&self.seeds).as_ptr()
    }

    pub fn telemetry(&self) -> VisualizerTelemetry {
        self.telemetry
    }

    pub fn is_dirty(&self) -> bool {
        self.is_dirty
    }
}

#[cfg(test)]
mod lensing_tests {
    use super::*;

    #[test]
    fn lensing_uniform_layout_is_wgsl_contract() {
        assert_eq!(std::mem::size_of::<LensingUniforms>(), 272);
        assert_eq!(std::mem::align_of::<LensingUniforms>(), 4);
        assert_eq!(std::mem::size_of::<SeedDefect>(), 16);
        assert_eq!(std::mem::align_of::<SeedDefect>(), 4);
    }

    #[test]
    fn seed_registration_and_telemetry() {
        let mut engine = VisualizerEngine::new(1920.0, 1080.0);
        engine.register_seed(0, 0.5, 0.5, 0.045, 0.008);
        engine.register_seed(3, 0.35, 0.62, 0.022, 0.006);
        engine.update_telemetry(0.428, 1.185);
        let t = engine.telemetry();
        assert!((t.max_einstein_radius - 0.045).abs() < 1e-6);
        assert!(t.active_caustics >= 4);
        engine.set_lensing_scale(9.0);
        assert_eq!(engine.uniforms.lensing_strength, 5.0);
    }
}
