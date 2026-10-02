// Fast-PM / 2LPT particle drift compute shader (SHBT boundary cosmology).
// Advances comoving particle trajectories under the boundary-loaded
// conformal background with Stinespring anti-baryon de-rendering.

struct Particle {
    position: vec3<f32>,
    vis_weight: f32,
    velocity: vec3<f32>,
    grav_mass: f32,
};

struct CosmologicalParams {
    a: f32,
    hubble: f32,
    dt: f32,
    d1_growth: f32,
    d2_growth: f32,
    f_load: f32,
    box_size: f32,
    grid_dim: u32,
    num_particles: u32,
};

@group(0) @binding(0) var<uniform> cosmo: CosmologicalParams;
@group(0) @binding(1) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(2) var density_grid: texture_storage_3d<r32float, write>;
@group(0) @binding(3) var force_grid: texture_3d<f32>;
@group(0) @binding(4) var force_sampler: sampler;

// Stinespring isometric de-rendering decay envelope
fn evaluate_stinespring_decay(z: f32) -> f32 {
    let z_sphaleron: f32 = 1.0e12;
    let lambda_mn: f32 = 2.302585; // ln(10) decay scale across modular restoration
    if (z > z_sphaleron) {
        return 1.0;
    }
    let log_ratio = log2(max(z, 1.0e-3)) / log2(10.0) - 12.0;
    return clamp(exp(lambda_mn * log_ratio * 0.1), 0.0, 1.0);
}

@compute @workgroup_size(256, 1, 1)
fn cs_advance_particles(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= cosmo.num_particles) {
        return;
    }

    var p = particles[idx];
    let z_current = (1.0 / cosmo.a) - 1.0;

    // Apply Stinespring de-rendering transition to anti-baryonic channel
    // If the particle belongs to the dark branch (determined by lower 23/33 index hash)
    let branch_hash = fract(sin(f32(idx) * 12.9898) * 43758.5453);
    if (branch_hash < (23.0 / 33.0)) {
        p.vis_weight = evaluate_stinespring_decay(z_current);
    } else {
        p.vis_weight = 1.0;
    }

    // Comoving coordinate mapping to normalized grid texture UVW [0, 1]
    let uvw = fract(p.position / cosmo.box_size);
    let force = textureSampleLevel(force_grid, force_sampler, uvw, 0.0).xyz;

    // Boundary loaded conformal friction factor
    let friction = 1.0 + (cosmo.f_load * (10.0 / 33.0));

    // Seed attractor boost across the condensation window z in [2, 20]:
    // strengthened softened gravity funnels particles into seed wells.
    let seed_gain = select(1.0, 3.0, z_current >= 2.0 && z_current <= 20.0);

    // Ghost-seed point-mass wells in normalized comoving box units [0, 1]
    // with respective defect mass weights (M_seed ~ 10^9 M_sun)
    var seed_wells = array<vec4<f32>, 4>(
        vec4<f32>(0.25, 0.25, 0.25, 1.0),
        vec4<f32>(0.75, 0.75, 0.25, 0.8),
        vec4<f32>(0.25, 0.75, 0.75, 0.9),
        vec4<f32>(0.75, 0.25, 0.75, 0.7),
    );

    var a_seed = vec3<f32>(0.0);
    // Point-mass gravitational acceleration active during condensation and cosmic web collapse (z <= 30)
    if (z_current <= 30.0) {
        let softening = 0.025; // softening length in normalized box units
        for (var wi = 0; wi < 4; wi = wi + 1) {
            let w_pos = seed_wells[wi].xyz;
            let w_mass = seed_wells[wi].w;
            var r_vec = w_pos - uvw;
            // Minimum image convention for periodic torus
            r_vec = r_vec - round(r_vec);
            let r2 = dot(r_vec, r_vec) + softening * softening;
            let inv_r3 = 1.0 / (r2 * sqrt(r2));
            // a_seed = G * M_seed * (x_seed - x) / (|x_seed - x|^2 + eps^2)^(3/2)
            a_seed += r_vec * (w_mass * inv_r3 * 0.015);
        }
    }

    // Symplectic Kick-Drift step under loaded background with both PM grid force and point-mass a_seed
    let total_force = force * seed_gain + a_seed;
    let kick = (total_force / (cosmo.a * cosmo.a * friction)) * cosmo.dt;
    p.velocity += kick;
    p.position += (p.velocity / (cosmo.a * cosmo.hubble)) * cosmo.dt;

    // Periodic boundary wrapping across the holographic spatial patch
    p.position = (p.position + vec3<f32>(cosmo.box_size)) % vec3<f32>(cosmo.box_size);

    particles[idx] = p;

    // Scatter particle density onto the PM grid cell.
    let cell = vec3<i32>(uvw * f32(cosmo.grid_dim));
    textureStore(density_grid, cell, vec4<f32>(p.grav_mass));
}
