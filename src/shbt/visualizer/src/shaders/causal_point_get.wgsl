// causal_point_get.wgsl — Active Causal-Point GET dynamics (shbt7 spec).
//
// The entropic GET acceleration constant is now derived from first
// principles on the observer boundary (shbt7 Section 5.1 / Thm 9.8):
//
//   kappa_GET(f_load) = (1 / (d_1 - h_dual_su3)) * (1 + (c_eff/d_1) * f_load)
//                     = (1/23) * (1 + (1325/4004) * f_load)
//
// The hardcoded kappa_GET = 0.0435 is gone; the TheoryInvariants uniform
// streams the canonical coset integers (k_l, k_q, K = k_l*k_q - k_l^2,
// c_eff = 1325/154, d_1 = gcd(k_l,K) = 26, h^v_SU(3) = 3, N_sat).
//
// Compute Pass 1 (cs_causal_point_get): evaluates observer admissibility
// R_entropy = N_limit - C_get for every causal-point node and applies the
// entropic clustering kick
//     a_GET(x) = -kappa_GET(f_load) * grad ln rho_proj(x)
// Compute Pass 2 (apply_get_acceleration): legacy per-particle kick path
// using the same derived constant.

struct TheoryInvariants {
    hubble_rate: f32,
    redshift: f32,
    dt: f32,
    load_fraction: f32,        // f_load(z)
    particle_count: u32,
    causal_point_count: u32,
    mesh_dim: u32,
    // Coset theory constants, streamed so compute_kappa_get is a pure
    // function of the invariant block (shbt7 Section 1):
    k_l: f32,                  // so(10) level = 26
    k_q: f32,                  // su(3) color level = 8
    k_gut: f32,                // K = k_l*k_q - k_l^2 = 312
    c_eff: f32,                // 1325/154 ~ 8.6039
    d_1: f32,                  // gcd(k_l, K) = 26
    h_dual_su3: f32,           // h^v(SU(3)) = 3
    n_sat: f32,                // N_sat (scaled units: 3.3119977e92)
    _pad: f32,
};

// Entropic GET coupling from first principles (shbt7 Eq. Thm 9.8):
// d_eff,0 = d_1 - h_dual_su3 = 23 fixes the isometric load slope.
fn compute_kappa_get(f_load: f32) -> f32 {
    let d_eff_0 = params.d_1 - params.h_dual_su3;
    return (1.0 / d_eff_0) * (1.0 + (params.c_eff / params.d_1) * f_load);
}

struct Particle {
    pos: vec3<f32>,
    vis_weight: f32,
    vel: vec3<f32>,
    grav_mass: f32,
};

struct CausalPoint {
    center: vec3<f32>,
    radius: f32,
    entropy_budget: f32,
    get_cost: f32,
    collapse_phase: f32,
    active_flag: u32,
    seed_index: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
    projection_dir: vec3<f32>,
    pad3: u32,
};

@group(0) @binding(0) var<uniform> params: TheoryInvariants;
@group(0) @binding(1) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(2) var<storage, read_write> causal_points: array<CausalPoint>;
@group(0) @binding(3) var<storage, read> density_grid: array<f32>;

fn get_grid_index(coord: vec3<u32>, dim: u32) -> u32 {
    return coord.x + coord.y * dim + coord.z * dim * dim;
}

fn sample_density(pos: vec3<f32>, dim: u32) -> f32 {
    let clamped_pos = clamp(pos, vec3<f32>(0.0), vec3<f32>(1.0)) * f32(dim - 1u);
    let base = vec3<u32>(floor(clamped_pos));
    let f = fract(clamped_pos);

    let i000 = get_grid_index(base, dim);
    let i100 = get_grid_index(base + vec3<u32>(1u, 0u, 0u), dim);
    let i010 = get_grid_index(base + vec3<u32>(0u, 1u, 0u), dim);
    let i110 = get_grid_index(base + vec3<u32>(1u, 1u, 0u), dim);
    let i001 = get_grid_index(base + vec3<u32>(0u, 0u, 1u), dim);
    let i101 = get_grid_index(base + vec3<u32>(1u, 0u, 1u), dim);
    let i011 = get_grid_index(base + vec3<u32>(0u, 1u, 1u), dim);
    let i111 = get_grid_index(base + vec3<u32>(1u, 1u, 1u), dim);

    let c00 = mix(density_grid[i000], density_grid[i100], f.x);
    let c10 = mix(density_grid[i010], density_grid[i110], f.x);
    let c01 = mix(density_grid[i001], density_grid[i101], f.x);
    let c11 = mix(density_grid[i011], density_grid[i111], f.x);

    let c0 = mix(c00, c10, f.y);
    let c1 = mix(c01, c11, f.y);

    return mix(c0, c1, f.z);
}

fn sample_density_gradient(pos: vec3<f32>, dim: u32) -> vec3<f32> {
    let delta = 1.0 / f32(dim);
    let dx = (sample_density(pos + vec3<f32>(delta, 0.0, 0.0), dim) - sample_density(pos - vec3<f32>(delta, 0.0, 0.0), dim)) / (2.0 * delta);
    let dy = (sample_density(pos + vec3<f32>(0.0, delta, 0.0), dim) - sample_density(pos - vec3<f32>(0.0, delta, 0.0), dim)) / (2.0 * delta);
    let dz = (sample_density(pos + vec3<f32>(0.0, 0.0, delta), dim) - sample_density(pos - vec3<f32>(0.0, 0.0, delta), dim)) / (2.0 * delta);
    return vec3<f32>(dx, dy, dz);
}

// Pass 1 (shbt7 5.1): observer admissibility under the entropic budget and
// the nonlocal kick d_x -= dt * kappa_GET * grad ln rho_proj on projectors.
@compute @workgroup_size(64, 1, 1)
fn cs_causal_point_get(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let cp_idx = global_id.x;
    if (cp_idx >= params.causal_point_count) {
        return;
    }

    var cp = causal_points[cp_idx];
    let r_entropy = cp.entropy_budget - cp.get_cost;
    let kappa_get = compute_kappa_get(params.load_fraction);

    if (r_entropy >= 0.0 && params.redshift > -0.99) {
        cp.active_flag = 1u;
        cp.collapse_phase = min(1.0, cp.collapse_phase + params.dt * 1.8);
        // Entropic erasure rate scales with the derived GET coupling.
        cp.entropy_budget -= cp.get_cost * kappa_get * params.dt;
        let grad_ln_rho = sample_density_gradient(cp.center, params.mesh_dim)
            / (sample_density(cp.center, params.mesh_dim) + 1e-4);
        cp.center -= params.dt * kappa_get * grad_ln_rho * 1e-3;
    } else {
        cp.active_flag = 0u;
        cp.collapse_phase = max(0.0, cp.collapse_phase - params.dt * 0.5);
    }

    causal_points[cp_idx] = cp;
}

@compute @workgroup_size(256, 1, 1)
fn apply_get_acceleration(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let p_idx = global_id.x;
    if (p_idx >= params.particle_count) {
        return;
    }

    var p = particles[p_idx];
    var total_a_get = vec3<f32>(0.0);

    for (var i = 0u; i < params.causal_point_count; i = i + 1u) {
        let cp = causal_points[i];
        if (cp.active_flag == 1u) {
            let dist_vec = p.pos - cp.center;
            let dist = length(dist_vec);

            if (dist < cp.radius && dist > 1e-5) {
                let rho = sample_density(p.pos, params.mesh_dim);
                let grad_rho = sample_density_gradient(p.pos, params.mesh_dim);
                let grad_ln_rho = grad_rho / (rho + 1e-4);

                let window = 1.0 - smoothstep(0.0, cp.radius, dist);
                let kappa_get = compute_kappa_get(params.load_fraction);
                let a_collapse = -kappa_get * cp.collapse_phase * grad_ln_rho * window;
                total_a_get += a_collapse;

                p.vis_weight = mix(p.vis_weight, 1.0, 0.02 * cp.collapse_phase);
            }
        }
    }

    p.vel += total_a_get * params.dt;
    particles[p_idx] = p;
}

// ---------------------------------------------------------------------------
// Render stage (shbt5): synthetic Fresnel ripple shells around active causal
// observer nodes (R_entropy >= 0). Instanced billboards inject shear and
// convergence rings into Channel B so quantum measurement boundaries stay
// visually delineated on the holographic manifold.
//
// Bindings live on @group(1): the compute entry points above occupy group(0).

struct CausalRenderUniforms {
    view_proj: mat4x4<f32>,
    cam_pos: vec4<f32>,
    // x: box size (comoving Mpc/h), y: time (s), z: master entropy scale,
    // w: billboard half-extent in world units.
    params: vec4<f32>,
};

@group(1) @binding(0) var<uniform> causal: CausalRenderUniforms;
@group(1) @binding(1) var<storage, read> causal_nodes: array<CausalPoint>;

struct CausalVertexOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) quad_uv: vec2<f32>,
    @location(1) world_pos: vec3<f32>,
    @location(2) entropy_level: f32,
};

@vertex
fn vs_causal(
    @builtin(vertex_index) v_idx: u32,
    @builtin(instance_index) node_idx: u32,
) -> CausalVertexOut {
    var out: CausalVertexOut;
    let node = causal_nodes[node_idx];

    var corners = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0),
        vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, 1.0), vec2<f32>(-1.0, 1.0),
    );
    let corner = corners[v_idx];
    out.quad_uv = corner;

    let center_world = (node.center - vec3<f32>(0.5)) * causal.params.x;
    let extent = causal.params.w;
    var clip_center = causal.view_proj * vec4<f32>(center_world, 1.0);
    clip_center.x += corner.x * extent * clip_center.w;
    clip_center.y += corner.y * extent * clip_center.w;

    out.clip_pos = clip_center;
    out.world_pos = center_world;
    out.entropy_level = node.entropy_budget * causal.params.z * f32(node.active_flag);
    return out;
}

struct CausalFragOut {
    @location(0) channel_a: vec4<f32>,
    @location(1) channel_b: vec4<f32>,
};

// Writes Channel B only (location 1): synthetic Fresnel shear + convergence
// rings marking the observer's measurement boundary. Channel A gets zeros
// (blend-add no-op) because the attachment set requires both targets.
@fragment
fn fs_causal(in: CausalVertexOut) -> CausalFragOut {
    var out: CausalFragOut;
    out.channel_a = vec4<f32>(0.0);
    let r2 = dot(in.quad_uv, in.quad_uv);
    if (r2 > 1.0) {
        discard;
    }

    // Fresnel-style rim weighting on the billboard: fades to zero at the
    // core, peaks near the shell radius.
    let fresnel = pow(1.0 - abs(in.quad_uv.x * in.quad_uv.y), 3.5) * (1.0 - r2);

    let phase = causal.params.y * 2.0 - length(in.quad_uv) * 4.0;
    let ripple = sin(phase) * 0.5 + 0.5;

    let boundary_alpha = fresnel * ripple * in.entropy_level;
    let synthetic_shear = vec2<f32>(fresnel * 0.5, -fresnel * 0.5);
    let synthetic_conv = boundary_alpha * 2.0;

    out.channel_b = vec4<f32>(synthetic_shear.x, synthetic_shear.y, synthetic_conv, in.entropy_level);
    return out;
}

// ---------------------------------------------------------------------------
// Compute Pass 3 (shbt9 Phase 1): sandbox observer measurement backaction.
//
// User-dispatched causal observers (CausalPointBuffer records created by
// unproject_and_dispatch_causal_point) exert the entropic clustering pull
// on particles intersecting their past light cone, balanced by the
// Fluctuation-Dissipation-Theorem kinetic friction counter-term
//   gamma_obs(x,t) = kappa_GET * <v . grad ln rho_proj> / (3 sigma_v^2)
// which bounds the total modular Hamiltonian and preserves Liouville
// phase-space volume. Each observer depletes its entropy budget
//   R_entropy = N_limit - C_get
// every step; depletion to zero terminates the measurement channel.
//
// The engine particle buffer (48-byte records, comoving Mpc positions) is
// bound at binding 6 under the nbody layout; sandbox_params at binding 5.

struct SandboxGetParams {
    particle_count: u32,
    sandbox_count: u32,
    dt: f32,
    kappa_get: f32,
    gamma_scale: f32,          // FDT 1/(3 sigma_v^2) prefactor
    box_size: f32,             // comoving Mpc/h
    pad0: f32,
    pad1: f32,
};

// Mirrors the 48-byte CausalObserver record shared with
// causal_cone_render.wgsl / causal_sphere_render.wgsl.
struct SandboxObserver {
    position: vec3<f32>,       // comoving Mpc
    cone_radius: f32,          // light-cone radius (Mpc)
    entropy_budget: f32,       // R_entropy remaining (bits)
    active_flag: u32,
    direction: vec3<f32>,      // cone axis
    cone_angle: f32,
    decay_rate: f32,           // C_get per step (bits/s-equiv)
    pad: f32,
};

struct EngineParticle {
    position: vec3<f32>,       // comoving Mpc
    channel: u32,
    velocity: vec3<f32>,       // supercomoving momentum p_tilde
    charge_flags: u32,
    shear: vec4<f32>,
    landauer_debt: f32,
    grav_mass: f32,
    pad: vec2<f32>,
};

@group(0) @binding(4) var<storage, read_write> sandbox_observers: array<SandboxObserver>;
@group(0) @binding(5) var<uniform> sandbox_params: SandboxGetParams;
@group(0) @binding(6) var<storage, read_write> sandbox_particles: array<EngineParticle>;

const SANDBOX_SHELL_T: f32 = 4.0;   // Mpc shell thickness on the cone surface

@compute @workgroup_size(256, 1, 1)
fn cs_sandbox_get(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let p_idx = global_id.x;
    if (p_idx >= sandbox_params.particle_count) {
        return;
    }

    var p = sandbox_particles[p_idx];
    let count = min(sandbox_params.sandbox_count, 1024u);
    var a_get = vec3<f32>(0.0);

    for (var i = 0u; i < count; i = i + 1u) {
        let obs = sandbox_observers[i];
        if (obs.active_flag == 0u || obs.entropy_budget <= 0.0) {
            continue;
        }
        // Minimum-image displacement on the periodic box.
        var r_vec = p.position - obs.position;
        r_vec = r_vec - sandbox_params.box_size *
            floor(r_vec / sandbox_params.box_size + vec3<f32>(0.5));
        let dist = max(length(r_vec), 1.0e-4);
        let delta_shell = abs(dist - obs.cone_radius);

        if (delta_shell < SANDBOX_SHELL_T) {
            let kw = 1.0 - delta_shell / SANDBOX_SHELL_T;
            let dir = r_vec / dist;
            // Entropic clustering pull toward the measurement cone.
            let entropic_pull = sandbox_params.kappa_get * dir *
                (kw / dist) * 0.01;
            // FDT counter-term: -gamma_obs * (v . dir) dir.
            let v_proj = dot(p.velocity, dir);
            let gamma_obs = min(
                sandbox_params.kappa_get * abs(v_proj) * sandbox_params.gamma_scale,
                8.0,
            );
            let stabilization = -gamma_obs * v_proj * dir;
            a_get = a_get + entropic_pull + stabilization;
        }
    }

    p.velocity = p.velocity + a_get * sandbox_params.dt;
    sandbox_particles[p_idx] = p;

    // Observer entropy depletion: thread 0 amortizes the per-step C_get
    // cost so the pass stays single-dispatch.
    if (p_idx == 0u) {
        for (var i = 0u; i < count; i = i + 1u) {
            var obs = sandbox_observers[i];
            obs.entropy_budget = obs.entropy_budget - obs.decay_rate * sandbox_params.dt;
            if (obs.entropy_budget <= 0.0) {
                obs.active_flag = 0u;
            }
            sandbox_observers[i] = obs;
        }
    }
}
