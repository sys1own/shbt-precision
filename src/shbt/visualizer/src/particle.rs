//! 64-byte packed, 16-byte aligned particle state shared with the WGSL
//! compute kernels (`nbody_pm.wgsl`, `seed_emergence.wgsl`,
//! `dual_channel_render.wgsl`).
//!
//! Layout contract (shbt6 spec, `ParticleGpu`):
//!   position[3]      comoving coordinates in Mpc/h
//!   channel          0 = Channel A (baryon), 1 = Channel B (dark ghost)
//!   velocity[3]      supercomoving momentum p_tilde (shbt8; physical
//!                    peculiar velocity v = p_tilde * V_0 / a)
//!   charge_flags     bit 0: active gauge charge; bits 1-31: phase age
//!   shear[4]         gamma_1, gamma_2, convergence kappa, unused
//!   landauer_debt    accumulated thermodynamic cost Delta N (scaled bits)
//!   grav_mass        gravitational coupling mass in internal units
//!   _pad[2]          trailing alignment pad

/// Particle state: 64 bytes, aligned to 16.
#[repr(C, align(16))]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Particle {
    pub position: [f32; 3],
    pub channel: u32,
    pub velocity: [f32; 3],
    pub charge_flags: u32,
    pub shear: [f32; 4],
    pub landauer_debt: f32,
    pub grav_mass: f32,
    pub _pad: [f32; 2],
}

impl Particle {
    /// Deterministic initial particle lattice filling `box_size` Mpc/h with
    /// `n` particles, with primordial scalar density ripples.
    ///
    /// Positions are displaced by a low-frequency sinusoidal modulation plus
    /// a golden-ratio jitter so the Cloud-In-Cell density field carries
    /// coherent overdense basins (delta ~ 1) that stay sub-critical at
    /// z = 30 and condense once the linear growth factor amplifies them
    /// across the cosmic-dawn window (shbt6, Eq. N_local / Delta N).
    pub fn seed_lattice(n: usize, box_size: f32) -> Vec<Particle> {
        let side = ((n as f64).cbrt().ceil() as usize).max(1);
        let step = box_size / side as f32;
        (0..n)
            .map(|i| {
                let gx = (i % side) as f32;
                let gy = ((i / side) % side) as f32;
                let gz = (i / (side * side)) as f32;
                // Primordial density spectrum: a small set of coherent
                // scalar modes displaces the lattice along its wavevectors,
                // imprinting cell-scale overdensities delta ~ 0.5-0.6 —
                // sub-critical at z = 30 (delta_eff < gamma - 1) but driven
                // super-critical once the linear growth term 31/(1+z)
                // amplifies them through the cosmic-dawn window.
                let bx = gx * step;
                let by = gy * step;
                let bz = gz * step;
                let k = std::f32::consts::TAU / box_size;
                let wave = |n1: f32, n2: f32, n3: f32, ph: f32| {
                    (k * (n1 * bx + n2 * by + n3 * bz) + ph).sin()
                };
                // Six modes with wavelengths ~20-30 Mpc seed ~10^3 candidate
                // basins; amplitude tuned so peak CIC contrast sits between
                // the z=7 (0.30) and z=30 (0.73) overflow thresholds.
                let dx = 0.336 * wave(8.0, 3.0, 2.0, 0.4)
                    + 0.189 * wave(3.0, 9.0, 4.0, 2.1)
                    + 0.105 * wave(11.0, 1.0, 6.0, 4.4);
                let dy = 0.336 * wave(2.0, 8.0, 3.0, 1.7)
                    + 0.189 * wave(9.0, 4.0, 3.0, 3.9)
                    + 0.105 * wave(1.0, 6.0, 11.0, 0.8);
                let dz = 0.336 * wave(3.0, 2.0, 8.0, 3.0)
                    + 0.189 * wave(4.0, 3.0, 9.0, 5.5)
                    + 0.105 * wave(6.0, 11.0, 1.0, 2.6);
                let ripple = (i as f32 * 0.618_034).fract() - 0.5;
                let px = (bx + dx + ripple * step * 0.15 + box_size) % box_size;
                let py = (by + dy + (i as f32 * 0.381_966).fract() * step * 0.12 + box_size) % box_size;
                let pz = (bz + dz + (i as f32 * 0.754_878).fract() * step * 0.10 + box_size) % box_size;
                // eta_D = 23/33 of particles de-render into the dark branch.
                let branch_hash = ((i as f32 * 12.9898).sin() * 43758.5453).fract().abs();
                let dark = branch_hash < (23.0 / 33.0);
                Particle {
                    position: [px, py, pz],
                    // Dark-bound anti-baryons start in Channel A holding an
                    // unquenched gauge charge (bit 0); the Stinespring
                    // de-rendering kernel quenches the bit and flips them to
                    // Channel B, emitting a transition tether.
                    channel: 0,
                    velocity: [0.0; 3],
                    charge_flags: if dark { 1 } else { 0 },
                    shear: [0.0; 4],
                    landauer_debt: 0.0,
                    grav_mass: if dark { 1.0 } else { 0.45 },
                    _pad: [0.0; 2],
                }
            })
            .collect()
    }
}
