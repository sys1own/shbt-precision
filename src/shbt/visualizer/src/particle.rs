//! 32-byte packed, 16-byte aligned particle state shared with the WGSL
//! compute kernel (`nbody_pm.wgsl`).

/// Particle state: 32 bytes, aligned to 16.
#[repr(C, align(16))]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Particle {
    pub position: [f32; 3], // 12 bytes: Spatial coordinates [x, y, z] in comoving Mpc/h
    pub vis_weight: f32,    //  4 bytes: Active gauge coupling factor (1.0 = visible, 0.0 = dark)
    pub velocity: [f32; 3], // 12 bytes: Peculiar velocity [vx, vy, vz] in km/s
    pub grav_mass: f32,     //  4 bytes: Total gravitational coupling mass in internal units
}

impl Particle {
    /// Deterministic initial particle lattice filling `box_size` Mpc/h with
    /// `n` particles, with primordial scalar density ripples.
    pub fn seed_lattice(n: usize, box_size: f32) -> Vec<Particle> {
        let side = ((n as f64).cbrt().ceil() as usize).max(1);
        let step = box_size / side as f32;
        (0..n)
            .map(|i| {
                let gx = (i % side) as f32;
                let gy = ((i / side) % side) as f32;
                let gz = (i / (side * side)) as f32;
                // Scalar-perturbation ripple displacement.
                let ripple = (i as f32 * 0.618_034).fract() - 0.5;
                let px = (gx * step + ripple * step * 0.35) % box_size;
                let py = (gy * step + (i as f32 * 0.381_966).fract() * step * 0.25) % box_size;
                let pz = (gz * step + (i as f32 * 0.754_878).fract() * step * 0.20) % box_size;
                // eta_D = 23/33 of particles de-render into the dark branch.
                let branch_hash = ((i as f32 * 12.9898).sin() * 43758.5453).fract().abs();
                let dark = branch_hash < (23.0 / 33.0);
                Particle {
                    position: [px, py, pz],
                    vis_weight: if dark { 1.0 } else { 1.0 }, // decays via shader epoch envelope
                    velocity: [0.0; 3],
                    grav_mass: if dark { 1.0 } else { 0.45 },
                }
            })
            .collect()
    }
}
