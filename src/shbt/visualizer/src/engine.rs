//! Wasm-facing SHBT engine records and `WasmShbtEngine` (shbt4 spec).
//!
//! `WasmShbtEngine` owns the zero-copy particle, causal-point, and
//! seed-defect buffers consumed by `causal_point_get.wgsl` and the
//! Fast-PM / dual-channel pipeline.  `HorizonLedger` (in `hud.rs`)
//! evaluates the boundary bit ledger and observer admissibility that
//! drive `update_epoch(z)`.

use crate::hud::HorizonLedger;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

pub use crate::particle::Particle as ParticleRecord;

/// Causal-point observer record: 64 bytes, 16-byte aligned (std430).
#[repr(C, align(16))]
#[derive(Copy, Clone, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CausalPointRecord {
    pub center: [f32; 3],        // Offset 0,  12 bytes
    pub radius: f32,             // Offset 12,  4 bytes
    pub entropy_budget: f32,     // Offset 16,  4 bytes
    pub get_cost: f32,           // Offset 20,  4 bytes
    pub collapse_phase: f32,     // Offset 24,  4 bytes
    pub active_flag: u32,        // Offset 28,  4 bytes
    pub seed_index: u32,         // Offset 32,  4 bytes
    pub pad: [u32; 3],           // Offset 36, 12 bytes
    pub projection_dir: [f32; 3],// Offset 48, 12 bytes
    pub pad2: u32,               // Offset 60,  4 bytes
} // Total: 64 bytes

/// Topological seed defect record: 128 bytes, 16-byte aligned (std430,
/// SHBT-MMIO v2.0).
#[repr(C, align(16))]
#[derive(Copy, Clone, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SeedDefectRecord {
    pub position: [f32; 3],      // Offset 0,  12 bytes
    pub mass: f32,               // Offset 12,  4 bytes (M_seed in M_sun)
    pub power_debt: f32,         // Offset 16,  4 bytes (Landauer debt in GW)
    pub winding_l: u32,          // Offset 20,  4 bytes (k_l = 26)
    pub winding_q: u32,          // Offset 24,  4 bytes (k_q = 8)
    pub winding_k: u32,          // Offset 28,  4 bytes (K = 312)
    pub overflow_bits: f32,      // Offset 32,  4 bytes (Delta N, in 1e30-bit units)
    pub condensation_z: f32,     // Offset 36,  4 bytes
    pub softening: f32,          // Offset 40,  4 bytes
    pub flags: u32,              // Offset 44,  4 bytes
    pub reserved: [f32; 20],     // Offset 48, 80 bytes (SHBT-MMIO v2.0 expansion)
} // Total: 128 bytes

/// Wasm/JS-facing engine: owns the shared simulation buffers and the
/// epoch ledger; exposes raw buffer pointers for zero-allocation upload.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
pub struct WasmShbtEngine {
    particles: Vec<ParticleRecord>,
    causal_points: Vec<CausalPointRecord>,
    seeds: Vec<SeedDefectRecord>,
    ledger: HorizonLedger,
    particle_count: usize,
    causal_point_count: usize,
    seed_count: usize,
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
impl WasmShbtEngine {
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen(constructor))]
    pub fn new(particle_count: usize, causal_point_count: usize, seed_count: usize) -> Self {
        #[cfg(target_arch = "wasm32")]
        console_error_panic_hook::set_once();

        let mut particles = Vec::with_capacity(particle_count);
        for i in 0..particle_count {
            let fi = i as f32;
            let p = ParticleRecord {
                position: [
                    (fi * 1.6180339887) % 1.0,
                    (fi * 2.7182818284) % 1.0,
                    (fi * 3.1415926535) % 1.0,
                ],
                channel: 0,
                velocity: [0.0; 3],
                charge_flags: 0,
                shear: [0.0; 4],
                landauer_debt: 0.0,
                grav_mass: 1.0 + (fi % 8.0) * 0.25,
                _pad: [0.0; 2],
            };
            particles.push(p);
        }

        let mut causal_points = Vec::with_capacity(causal_point_count);
        for i in 0..causal_point_count {
            let cp = CausalPointRecord {
                center: [
                    0.5 + (i as f32 * 0.1) % 0.4,
                    0.5 + (i as f32 * 0.15) % 0.4,
                    0.5,
                ],
                radius: 0.12,
                entropy_budget: 1000.0,
                get_cost: 12.5,
                collapse_phase: 0.0,
                active_flag: 0,
                seed_index: i as u32,
                pad: [0; 3],
                projection_dir: [0.0, 1.0, 0.0],
                pad2: 0,
            };
            causal_points.push(cp);
        }

        let mut seeds = Vec::with_capacity(seed_count);
        for _ in 0..seed_count {
            let seed = SeedDefectRecord {
                position: [0.5, 0.5, 0.5],
                mass: 7.95498e8,
                power_debt: 7.95498e8 * 906.0,
                winding_l: 26,
                winding_q: 8,
                winding_k: 312,
                overflow_bits: 6.0e29, // Delta N = 6.0e59 bits, in 1e30-bit units
                condensation_z: 15.0,
                softening: 0.02,
                flags: 1,
                reserved: [0.0; 20],
            };
            seeds.push(seed);
        }

        let ledger = HorizonLedger::new();

        Self {
            particles,
            causal_points,
            seeds,
            ledger,
            particle_count,
            causal_point_count,
            seed_count,
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
    pub fn get_particle_buffer_ptr(&self) -> *const ParticleRecord {
        self.particles.as_ptr()
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
    pub fn get_particle_buffer_byte_len(&self) -> usize {
        self.particles.len() * std::mem::size_of::<ParticleRecord>()
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
    pub fn get_causal_point_buffer_ptr(&self) -> *const CausalPointRecord {
        self.causal_points.as_ptr()
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
    pub fn get_causal_point_buffer_byte_len(&self) -> usize {
        self.causal_points.len() * std::mem::size_of::<CausalPointRecord>()
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
    pub fn get_seed_buffer_ptr(&self) -> *const SeedDefectRecord {
        self.seeds.as_ptr()
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
    pub fn get_seed_buffer_byte_len(&self) -> usize {
        self.seeds.len() * std::mem::size_of::<SeedDefectRecord>()
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
    pub fn update_epoch(&mut self, z: f64) {
        self.ledger.update(z);
        let stinespring_active = self.ledger.stinespring_blend;

        // Stinespring de-rendering: the first 23/33 * blend of particles
        // quench into Channel B (dark ghost); the remainder stay baryonic.
        for (i, p) in self.particles.iter_mut().enumerate() {
            let frac = i as f32 / self.particle_count.max(1) as f32;
            p.channel = if frac < stinespring_active as f32 * (23.0 / 33.0) as f32 {
                1
            } else {
                0
            };
        }

        let is_admissible = self.ledger.is_observer_admissible();
        for cp in self.causal_points.iter_mut() {
            if is_admissible {
                cp.entropy_budget = (self.ledger.active_visible_bits / 1.0e120) as f32;
            } else {
                cp.entropy_budget = 0.0;
                cp.active_flag = 0;
            }
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
    pub fn get_hud_telemetry_json(&self) -> String {
        self.ledger.to_json_telemetry()
    }

    pub fn particle_count(&self) -> usize {
        self.particle_count
    }

    pub fn causal_point_count(&self) -> usize {
        self.causal_point_count
    }

    pub fn seed_count(&self) -> usize {
        self.seed_count
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn record_layouts_match_std430() {
        assert_eq!(std::mem::size_of::<ParticleRecord>(), 64);
        assert_eq!(std::mem::size_of::<CausalPointRecord>(), 64);
        assert_eq!(std::mem::size_of::<SeedDefectRecord>(), 128);
        assert_eq!(std::mem::align_of::<SeedDefectRecord>(), 16);
    }

    #[test]
    fn engine_epoch_update_quenches_channel() {
        let mut engine = WasmShbtEngine::new(64, 8, 2);
        engine.update_epoch(0.0);
        // Full Stinespring blend at z = 0: ~23/33 of particles flip to
        // Channel B (dark ghost), ~10/33 remain baryonic.
        let mut dark = 0usize;
        for p in 0..64 {
            let c = unsafe { &*engine.get_particle_buffer_ptr().add(p) }.channel;
            if c == 1 {
                dark += 1;
            }
        }
        let expected = ((23.0_f64 / 33.0) * 64.0).round() as usize;
        assert!((dark as i64 - expected as i64).abs() <= 1, "dark {dark} != {expected}");
        assert!(engine.get_hud_telemetry_json().contains("redshift"));
    }
}
