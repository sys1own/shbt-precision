// Fast-PM / 2LPT particle drift compute shader (SHBT boundary cosmology).
// Advances comoving particle trajectories under the boundary-loaded
// conformal background with Stinespring anti-baryon de-rendering.
//
// shbt6 update: the four hardcoded ghost-seed wells are gone. Seed gravity
// now comes from the emergent condensation pipeline via
// compute_seed_gravitational_acceleration() reading the `active_seeds`
// buffer written by seed_emergence.wgsl (cs_temporal_tracking):
//   a_seed(x) = -sum_k G M_seed,k (x - x_seed,k) / (|x - x_seed,k|^2 + eps^2)^(3/2)
// with eps = 50 kpc softening and the minimum-image convention on the torus.
//
// Stinespring de-rendering (Eq. 164): when a Channel-A anti-baryon's gauge
// charge quenches, the particle flips to Channel B and emits a pair of
// TetherVertex endpoints into the TetherVertexBuffer for the line-list
// tether pass (tether_render.wgsl). The quench also deposits Landauer debt
// into particle.landauer_debt for the Channel-B heat map.

struct Particle {
    position: vec3<f32>,
    channel: u32,              // 0 = Channel A baryon, 1 = Channel B dark ghost
    velocity: vec3<f32>,
    charge_flags: u32,         // bit 0: unquenched gauge charge
    shear: vec4<f32>,          // gamma_1, gamma_2, kappa, pad
    landauer_debt: f32,
    grav_mass: f32,
    pad: vec2<f32>,
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

// Emergent seed defect record (mirrors seed_emergence.wgsl output).
struct SeedDefectRecord {
    position: vec4<f32>,       // xyz: comoving centroid, w: active flag
    dynamics: vec4<f32>,       // x: mass (M_sun), y: dM/dt, z: P_debt, w: id
};

// Stinespring transition tether vertex (Enhancement 4).
struct TetherVertex {
    pos: vec3<f32>,
    alpha_decay: f32,
    color_tint: vec4<f32>,
};

struct IndirectArgs {
    vertex_count: atomic<u32>,
    instance_count: u32,
    first_vertex: u32,
    first_instance: u32,
};

@group(0) @binding(0) var<uniform> cosmo: CosmologicalParams;
@group(0) @binding(1) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(2) var density_grid: texture_storage_3d<r32float, write>;
@group(0) @binding(3) var force_grid: texture_3d<f32>;
@group(0) @binding(4) var force_sampler: sampler;
@group(0) @binding(5) var<storage, read> active_seeds: array<SeedDefectRecord, 256>;
@group(0) @binding(6) var<storage, read> seed_state: array<u32, 4>;

@group(1) @binding(0) var<storage, read_write> tethers: array<TetherVertex>;
@group(1) @binding(1) var<storage, read_write> indirect_draw: IndirectArgs;

// Thermal Stinespring channel overlap (shbt7 Section 4 / Thm 9.10):
// the visible-sector overlap follows the anti-baryon scaling dimension
// Delta_Bbar = 26/3 across the modular crossover z_N = 7.356e10,
//   w_vis(z) = (1 - eta_D) + eta_D / (1 + (z_N/z)^Delta)
// with eta_D = 23/33 = 1 - Omega_DM/Omega_B residual. Replaces the
// empirical exponential decay envelope.
const ETA_D: f32 = 0.6969696970;   // 23/33
const Z_N: f32 = 7.356e10;         // modular crossover redshift
const DELTA_BBAR: f32 = 8.6666667; // 26/3 anti-baryon scaling dimension

fn evaluate_stinespring_channel(z: f32) -> f32 {
    if (z <= 0.0) {
        return 1.0 - ETA_D;
    }
    let power = pow(Z_N / max(z, 1.0e-3), DELTA_BBAR);
    return (1.0 - ETA_D) + ETA_D / (1.0 + power);
}

// Quenched fraction of the register as seen by the Stinespring channel:
// f_quench = (1 - w_vis) / eta_D saturates at 1 for z -> 0.
fn evaluate_quench_fraction(z: f32) -> f32 {
    return clamp((1.0 - evaluate_stinespring_channel(z)) / ETA_D, 0.0, 1.0);
}

// Emit a Stinespring transition tether from the Channel-A origin to the
// Channel-B decoupled coordinate (spec: cyan -> deep-violet pair endpoints).
fn emit_stinespring_tether(pos_a: vec3<f32>, pos_b: vec3<f32>) {
    let base = atomicAdd(&indirect_draw.vertex_count, 2u);
    if (base + 1u >= 65536u) {
        return;
    }
    let tint = vec4<f32>(0.25, 0.85, 1.0, 1.0);
    tethers[base] = TetherVertex(pos_a, 1.0, tint);
    tethers[base + 1u] = TetherVertex(pos_b, 1.0, vec4<f32>(0.15, 0.05, 0.45, 0.0));
}

// Softened Newtonian acceleration from emergent topological defect seeds
// (epsilon = 50 kpc -> 0.05 Mpc; masses in 1e10 M_sun units for G_const).
fn compute_seed_gravitational_acceleration(pos: vec3<f32>, box_size: f32) -> vec3<f32> {
    var total_acc = vec3<f32>(0.0);
    let count = min(seed_state[1], 256u);
    let G_const = 4.30091e-3; // (km/s)^2 * Mpc / (10^10 M_sun)
    let softening_sq = 0.0025; // (50 kpc)^2 softening length

    for (var k = 0u; k < count; k = k + 1u) {
        let seed = active_seeds[k];
        if (seed.position.w > 0.5) {
            var r_vec = seed.position.xyz - pos;
            r_vec = r_vec - box_size * round(r_vec / box_size);

            let r_sq = dot(r_vec, r_vec);
            let inv_dist_cube = 1.0 / pow(r_sq + softening_sq, 1.5);
            let m_scaled = seed.dynamics.x * 1e-10;

            total_acc = total_acc + G_const * m_scaled * r_vec * inv_dist_cube;
        }
    }
    return total_acc;
}

@compute @workgroup_size(256, 1, 1)
fn cs_advance_particles(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= cosmo.num_particles) {
        return;
    }

    var p = particles[idx];
    let z_current = (1.0 / cosmo.a) - 1.0;

    // Stinespring de-rendering (thermal isometric channel): the quenched
    // fraction f_quench(z) = (1 - w_vis)/eta_D advances monotonically with
    // redshift descent; each unquenched anti-baryon commits to Channel B
    // when its deterministic hash falls inside the quenched volume. The
    // transition emits a tether to the Channel-B image and banks Landauer
    // debt on the particle for the heat-map emission profile.
    let f_quench = evaluate_quench_fraction(z_current);
    let draw = fract(sin(f32(idx) * 12.9898 + 78.233) * 43758.5453);
    if ((p.charge_flags & 1u) != 0u && p.channel == 0u && draw < f_quench) {
        let pos_b = p.position + vec3<f32>(0.011, -0.007, 0.008) * cosmo.box_size;
        emit_stinespring_tether(p.position, pos_b);
        p.channel = 1u;
        p.charge_flags = p.charge_flags & ~1u;
        p.landauer_debt += 1.0e6;
    }
    if (p.channel == 1u) {
        // Quenched ghosts keep accruing small debt as the register erases
        // residual coordinate bits (visual heat accumulation only).
        p.landauer_debt += 906.0 * cosmo.dt * 1.0e3;
    }

    // Comoving coordinate mapping to normalized grid texture UVW [0, 1]
    let uvw = fract(p.position / cosmo.box_size);
    var force = textureSampleLevel(force_grid, force_sampler, uvw, 0.0).xyz;

    // Boundary loaded conformal friction factor
    let friction = 1.0 + (cosmo.f_load * (10.0 / 33.0));

    // Primordial fluctuation growth boost across the condensation window
    // z in [2, 30]: strengthens the density-gradient field so that basins
    // deepen organically into capacity-overflow nucleation sites.
    let seed_gain = select(1.0, 3.0, z_current >= 2.0 && z_current <= 20.0);

    // Pre-condensation homogeneity (z > 17.5): the bulk register is
    // thermalized and structure cannot condense, so gravitational kicks
    // are suppressed until the linear growth factor opens the window.
    force = force * select(0.0, 1.0, z_current <= 17.5);

    // Emergent ghost-seed attraction, replacing the hardcoded well table.
    var a_seed = vec3<f32>(0.0);
    if (z_current <= 30.0) {
        a_seed = compute_seed_gravitational_acceleration(p.position, cosmo.box_size);
    }

    // Symplectic Kick-Drift step under loaded background with both PM grid
    // force and emergent point-mass a_seed.
    let total_force = force * seed_gain + a_seed;
    let kick = (total_force / (cosmo.a * cosmo.a * friction)) * cosmo.dt;
    p.velocity += kick;
    // Numerical bound: cap peculiar velocity so a single soft-kernel
    // fluctuation cannot send the particle to NaN (register overflow in
    // the integrator corresponds to a frame-drop in the SHBT picture).
    p.velocity = clamp(p.velocity, vec3<f32>(-5.0e2), vec3<f32>(5.0e2));
    p.position += (p.velocity / (cosmo.a * cosmo.hubble)) * cosmo.dt;

    // Periodic boundary wrapping across the holographic spatial patch
    p.position = (p.position + vec3<f32>(cosmo.box_size)) % vec3<f32>(cosmo.box_size);

    // Projected tidal shear estimate from the PM field for the Channel-B
    // billboard stretch (gamma ~ grad of the sampled force magnitude).
    let eps = 1.0 / f32(cosmo.grid_dim);
    let fx1 = textureSampleLevel(force_grid, force_sampler, uvw + vec3<f32>(eps, 0.0, 0.0), 0.0).x;
    let fx0 = textureSampleLevel(force_grid, force_sampler, uvw - vec3<f32>(eps, 0.0, 0.0), 0.0).x;
    let fy1 = textureSampleLevel(force_grid, force_sampler, uvw + vec3<f32>(0.0, eps, 0.0), 0.0).y;
    let fy0 = textureSampleLevel(force_grid, force_sampler, uvw - vec3<f32>(0.0, eps, 0.0), 0.0).y;
    p.shear = vec4<f32>(fx1 - fx0, fy1 - fy0, length(force), 0.0);

    particles[idx] = p;

    // Scatter particle density onto the PM grid cell.
    let cell = vec3<i32>(uvw * f32(cosmo.grid_dim));
    textureStore(density_grid, cell, vec4<f32>(p.grav_mass));
}
