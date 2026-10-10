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

const N_MOD: i64 = 2_901_360;

#[inline]
fn isotropic_weil_phase(kx: i32, ky: i32, kz: i32) -> f32 {
    let kx2 = (kx as i64) * (kx as i64);
    let ky2 = (ky as i64) * (ky as i64);
    let kz2 = (kz as i64) * (kz as i64);

    let sigma1 = kx2 + ky2 + kz2;
    if sigma1 == 0 {
        return 0.0;
    }

    let sigma2 = kx2 * ky2 + ky2 * kz2 + kz2 * kx2;
    let sigma3 = kx2 * ky2 * kz2;

    let sgn_h = if kz > 0 {
        1.0
    } else if kz == 0 && ky > 0 {
        1.0
    } else if kz == 0 && ky == 0 && kx > 0 {
        1.0
    } else {
        -1.0
    };

    let w_poly = ((sigma1 % N_MOD) * 157
        + (sigma2 % N_MOD) * 11
        + (sigma3 % N_MOD) * 7
        + (((sigma1 % N_MOD) * (sigma2 % N_MOD)) % N_MOD) * 3
        + 1)
        .rem_euclid(N_MOD);

    (sgn_h * (2.0 * std::f64::consts::PI) * (w_poly as f64 / N_MOD as f64)) as f32
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
        let ph_x1 = isotropic_weil_phase(8, 3, 2);
        let ph_x2 = isotropic_weil_phase(3, 9, 4);
        let ph_x3 = isotropic_weil_phase(11, 1, 6);

        let ph_y1 = isotropic_weil_phase(2, 8, 3);
        let ph_y2 = isotropic_weil_phase(9, 4, 3);
        let ph_y3 = isotropic_weil_phase(1, 6, 11);

        let ph_z1 = isotropic_weil_phase(3, 2, 8);
        let ph_z2 = isotropic_weil_phase(4, 3, 9);
        let ph_z3 = isotropic_weil_phase(6, 11, 1);

        (0..n)
            .map(|i| {
                let gx = (i % side) as f32;
                let gy = ((i / side) % side) as f32;
                let gz = (i / (side * side)) as f32;
                let bx = gx * step;
                let by = gy * step;
                let bz = gz * step;
                let k = std::f32::consts::TAU / box_size;
                let wave = |n1: f32, n2: f32, n3: f32, ph: f32| {
                    (k * (n1 * bx + n2 * by + n3 * bz) + ph).sin()
                };
                let dx = 0.336 * wave(8.0, 3.0, 2.0, ph_x1)
                    + 0.189 * wave(3.0, 9.0, 4.0, ph_x2)
                    + 0.105 * wave(11.0, 1.0, 6.0, ph_x3);
                let dy = 0.336 * wave(2.0, 8.0, 3.0, ph_y1)
                    + 0.189 * wave(9.0, 4.0, 3.0, ph_y2)
                    + 0.105 * wave(1.0, 6.0, 11.0, ph_y3);
                let dz = 0.336 * wave(3.0, 2.0, 8.0, ph_z1)
                    + 0.189 * wave(4.0, 3.0, 9.0, ph_z2)
                    + 0.105 * wave(6.0, 11.0, 1.0, ph_z3);
                let rx = (i as f32 * 0.618_034).fract() - 0.5;
                let ry = (i as f32 * 0.381_966).fract() - 0.5;
                let rz = (i as f32 * 0.754_878).fract() - 0.5;
                let px = (bx + dx + rx * step * 0.12 + box_size) % box_size;
                let py = (by + dy + ry * step * 0.12 + box_size) % box_size;
                let pz = (bz + dz + rz * step * 0.12 + box_size) % box_size;
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
