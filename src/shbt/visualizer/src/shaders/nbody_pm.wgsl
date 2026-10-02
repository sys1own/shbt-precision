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

    // Symplectic Kick-Drift step under loaded background
    let kick = (force / (cosmo.a * cosmo.a * friction)) * cosmo.dt;
    p.velocity += kick;
    p.position += (p.velocity / (cosmo.a * cosmo.hubble)) * cosmo.dt;

    // Periodic boundary wrapping across the holographic spatial patch
    p.position = (p.position + vec3<f32>(cosmo.box_size)) % vec3<f32>(cosmo.box_size);

    particles[idx] = p;

    // Scatter particle density onto the PM grid cell.
    let cell = vec3<i32>(uvw * f32(cosmo.grid_dim));
    textureStore(density_grid, cell, vec4<f32>(p.grav_mass));
}
