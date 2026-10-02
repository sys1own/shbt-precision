// causal_point_get.wgsl — Active Causal-Point GET dynamics (SHBT shbt4 spec).
//
// Compute Pass 1: evaluates observer admissibility R_entropy = N_limit - C_get
// for every causal-point node and advances/relaxes its collapse phase.
// Compute Pass 2: applies the emergent thermodynamic clustering acceleration
//     a_GET(x) = -kappa_GET * grad ln rho_proj(x)
// focusing dark-matter filaments and baryonic gas into observer-selected nodes.

struct UniformParams {
    hubble_rate: f32,
    redshift: f32,
    kappa_get: f32,
    dt: f32,
    particle_count: u32,
    causal_point_count: u32,
    mesh_dim: u32,
    load_fraction: f32,
};

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

@group(0) @binding(0) var<uniform> params: UniformParams;
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

@compute @workgroup_size(64, 1, 1)
fn evaluate_causal_points(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let cp_idx = global_id.x;
    if (cp_idx >= params.causal_point_count) {
        return;
    }

    var cp = causal_points[cp_idx];
    let r_entropy = cp.entropy_budget - cp.get_cost;

    if (r_entropy >= 0.0 && params.redshift > -0.99) {
        cp.active_flag = 1u;
        cp.collapse_phase = min(1.0, cp.collapse_phase + params.dt * 1.8);
        cp.entropy_budget -= cp.get_cost * 0.005 * params.dt;
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
                let a_collapse = -params.kappa_get * cp.collapse_phase * grad_ln_rho * window;
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

// Writes Channel B only (location 1): synthetic Fresnel shear + convergence
// rings marking the observer's measurement boundary.
@fragment
fn fs_causal(in: CausalVertexOut) -> @location(1) vec4<f32> {
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

    return vec4<f32>(synthetic_shear.x, synthetic_shear.y, synthetic_conv, in.entropy_level);
}
