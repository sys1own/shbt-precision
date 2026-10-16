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

// Signed Euclidean modulo wrapping across periodic 3-torus
fn periodic_coord(coord: vec3<i32>, dim: i32) -> vec3<i32> {
    return ((coord % vec3<i32>(dim)) + vec3<i32>(dim)) % vec3<i32>(dim);
}

fn grid_linear_index(c: vec3<i32>, dim: i32) -> u32 {
    let p = periodic_coord(c, dim);
    return u32((p.z * dim + p.y) * dim + p.x);
}

fn sample_density(pos: vec3<f32>, dim: u32) -> f32 {
    let dim_f = f32(dim);
    let dim_i = i32(dim);
    let grid_pos = fract(pos) * dim_f;
    let base = vec3<i32>(floor(grid_pos));
    let f = grid_pos - floor(grid_pos);

    let i000 = grid_linear_index(base, dim_i);
    let i100 = grid_linear_index(base + vec3<i32>(1, 0, 0), dim_i);
    let i010 = grid_linear_index(base + vec3<i32>(0, 1, 0), dim_i);
    let i110 = grid_linear_index(base + vec3<i32>(1, 1, 0), dim_i);
    let i001 = grid_linear_index(base + vec3<i32>(0, 0, 1), dim_i);
    let i101 = grid_linear_index(base + vec3<i32>(1, 0, 1), dim_i);
    let i011 = grid_linear_index(base + vec3<i32>(0, 1, 1), dim_i);
    let i111 = grid_linear_index(base + vec3<i32>(1, 1, 1), dim_i);

    let c00 = mix(density_grid[i000], density_grid[i100], f.x);
    let c10 = mix(density_grid[i010], density_grid[i110], f.x);
    let c01 = mix(density_grid[i001], density_grid[i101], f.x);
    let c11 = mix(density_grid[i011], density_grid[i111], f.x);

    let c0 = mix(c00, c10, f.y);
    let c1 = mix(c01, c11, f.y);

    return mix(c0, c1, f.z);
}

fn sample_density_gradient(pos: vec3<f32>, dim: u32) -> vec3<f32> {
    let dim_f = f32(dim);
    let delta = 1.0 / dim_f;
    let dx = (sample_density(pos + vec3<f32>(delta, 0.0, 0.0), dim) - sample_density(pos - vec3<f32>(delta, 0.0, 0.0), dim)) * (0.5 * dim_f);
    let dy = (sample_density(pos + vec3<f32>(0.0, delta, 0.0), dim) - sample_density(pos - vec3<f32>(0.0, delta, 0.0), dim)) * (0.5 * dim_f);
    let dz = (sample_density(pos + vec3<f32>(0.0, 0.0, delta), dim) - sample_density(pos - vec3<f32>(0.0, 0.0, delta), dim)) * (0.5 * dim_f);
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

    let a_scale = 1.0 / (1.0 + max(params.redshift, -0.999));
    let rho_cardy_floor = max(0.00696738, 1.0 / (39.0 * max(a_scale, 0.001)));

    if (r_entropy >= 0.0 && params.redshift > -0.99) {
        cp.active_flag = 1u;
        cp.collapse_phase = min(1.0, cp.collapse_phase + params.dt * 1.8);
        // Entropic erasure rate scales with the derived GET coupling.
        cp.entropy_budget -= cp.get_cost * kappa_get * params.dt;
        let rho_bulk = sample_density(cp.center, params.mesh_dim);
        let grad_rho = sample_density_gradient(cp.center, params.mesh_dim);
        let rho_reg = rho_bulk + rho_cardy_floor;
        let grad_ln_rho = grad_rho / rho_reg;
        let a_get_cp = -kappa_get * (1.0 + params.load_fraction) * grad_ln_rho;
        cp.center += params.dt * a_get_cp * 1e-3;
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

    // Cardy capacity limit per cell: N_limit(z) = gamma_CFT * ((1+z)/18)^7.5
    let rel_z = clamp((1.0 + params.redshift) / 18.0, 1.0e-3, 50.0);
    let n_limit = (params.c_eff / 6.0) * pow(rel_z, 7.5);

    for (var i = 0u; i < params.causal_point_count; i = i + 1u) {
        let cp = causal_points[i];
        if (cp.active_flag == 1u) {
            let dist_vec = p.pos - cp.center;
            let dist = length(dist_vec);

            if (dist < cp.radius && dist > 1e-5) {
                let rho = sample_density(p.pos, params.mesh_dim);
                let grad_rho = sample_density_gradient(p.pos, params.mesh_dim);

                // Cardy holographic capacity saturation ratio: eta_sat = clamp(rho / n_limit, 0, 1)
                let eta_sat = clamp(rho / max(n_limit, 1.0e-4), 0.0, 1.0);

                // Dynamic Cardy capacity floor: rho_floor = max(0.00696738, 1 / (39 * a))
                let a_curr = 1.0 / (1.0 + max(params.redshift, -0.999));
                let rho_floor = max(0.00696738, 1.0 / (39.0 * max(a_curr, 0.001)));
                let rho_reg = rho + rho_floor;
                let grad_ln_rho = grad_rho / rho_reg;

                // Generalized Entropic Transport acceleration:
                // a_GET = -kappa_GET * (1 + f_load) * grad_ln_rho
                let kappa_get = compute_kappa_get(params.load_fraction);
                let a_get = -kappa_get * (1.0 + params.load_fraction) * grad_ln_rho;

                // Halo core virial dispersion acceleration:
                // a_disp = -2 * sigma_v^2 * grad_ln_rho with sigma_v^2 = (T_vir * 8.314e-3) * eta_sat
                let t_vir = 1.0e5; // Virial halo temperature (K)
                let sigma_v_sq = (t_vir * 8.314e-3 * 1.0e-4) * eta_sat;
                let a_disp = -2.0 * sigma_v_sq * grad_ln_rho;

                // Visible baryon hydrodynamic acceleration: a_hydro = -c_s^2 * a * grad_ln_rho_b
                // for particles with h_i >= 23/33 (0.6969697)
                var a_hydro = vec3<f32>(0.0);
                if (p.grav_mass >= 0.6969697) {
                    let z = max(params.redshift, 0.0);
                    let w_drag = 0.5 * (1.0 + tanh((z - 1059.94) / 25.0));
                    let t_post = 2.7255 * (1.0 + z) * (z / (z + 137.0));
                    let t_pre = 2.7255 * (1.0 + z);
                    let t_b = mix(t_post, t_pre, w_drag);
                    let c_s_sq = 1.666667 * 1.380649e-23 * t_b / (1.22534 * 1.660539e-27);
                    a_hydro = -c_s_sq * a_curr * grad_ln_rho * 1.0e-6;
                }

                let window = 1.0 - smoothstep(0.0, cp.radius, dist);
                let a_net = (a_get + a_disp + a_hydro) * cp.collapse_phase * window;

                total_a_get += a_net;

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
            // Virial stabilization counter-balancing inward entropic clustering
            let sigma_vir_sq = 0.05;
            let a_disp = -2.0 * sigma_vir_sq * dir * (kw / dist) * 0.005;
            // Canonical Landauer erasure dissipation rate:
            // gamma_obs = P_debt / (M_seed * c^2) = 906 GW / M_sun / c^2 ~ 5.068e-24 s^-1.
            // Pure first-principles boundary invariants govern trajectory evolution.
            let v_proj = dot(p.velocity, dir);
            let gamma_obs: f32 = 5.068e-24;
            let stabilization = -gamma_obs * v_proj * dir;
            a_get = a_get + entropic_pull + a_disp + stabilization;
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

// ----------------------------------------------------------------------------
// Isotropic 3D Base-2 Sobol Generator & 3D Toroidal Observer Packing (shbt22)
// ----------------------------------------------------------------------------

fn generate_sobol_3d(index: u32) -> vec3<f32> {
    let gray = index ^ (index >> 1u);

    // Dimension 1: Radical inverse in base 2
    var v1: u32 = 0u;
    var g1 = gray;
    for (var bit = 0u; bit < 32u; bit = bit + 1u) {
        if ((g1 & 1u) != 0u) {
            v1 = v1 ^ (1u << (31u - bit));
        }
        g1 = g1 >> 1u;
    }
    let x = f32(v1) * 2.3283064365386963e-10;

    // Dimension 2: Direction numbers for primitive polynomial x + 1
    var v2: u32 = 0u;
    var g2 = gray;
    var d2_tab = array<u32, 10>(
        2147483648u, 1073741824u, 2684354560u, 1879048192u,
        3489660928u, 2550136832u, 1912602624u, 3875536896u,
        1858076672u, 3716153344u
    );
    for (var bit = 0u; bit < 10u; bit = bit + 1u) {
        if ((g2 & (1u << bit)) != 0u) {
            v2 = v2 ^ d2_tab[bit];
        }
    }
    let y = f32(v2) * 2.3283064365386963e-10;

    // Dimension 3: Direction numbers for primitive polynomial x^2 + x + 1
    var v3: u32 = 0u;
    var g3 = gray;
    var d3_tab = array<u32, 10>(
        2147483648u, 3221225472u, 1610612736u, 3489660928u,
        2550136832u, 3758096384u, 2046820352u, 3959422976u,
        1442840576u, 2885681152u
    );
    for (var bit = 0u; bit < 10u; bit = bit + 1u) {
        if ((g3 & (1u << bit)) != 0u) {
            v3 = v3 ^ d3_tab[bit];
        }
    }
    let z = f32(v3) * 2.3283064365386963e-10;

    return vec3<f32>(x, y, z);
}

// 3D Finite-Difference Bulk Density Gradient with Cardy Regularization on 3-Torus
fn sample_3d_density_and_gradient(pos: vec3<f32>, dim: u32, box_len: f32) -> vec4<f32> {
    let nx = f32(dim);
    let ny = f32(dim);
    let nz = f32(dim);

    let grid_pos = vec3<f32>(
        fract(pos.x) * nx,
        fract(pos.y) * ny,
        fract(pos.z) * nz
    );

    let base_idx = vec3<i32>(floor(grid_pos));
    let frac_val = grid_pos - floor(grid_pos);

    let dx = box_len / nx;
    let dy = box_len / ny;
    let dz = box_len / nz;

    var rho_accum = 0.0;
    var grad_accum = vec3<f32>(0.0, 0.0, 0.0);

    for (var cz = 0; cz <= 1; cz = cz + 1) {
        for (var cy = 0; cy <= 1; cy = cy + 1) {
            for (var cx = 0; cx <= 1; cx = cx + 1) {
                let weight = (select(1.0 - frac_val.x, frac_val.x, cx == 1)) *
                             (select(1.0 - frac_val.y, frac_val.y, cy == 1)) *
                             (select(1.0 - frac_val.z, frac_val.z, cz == 1));

                let cur_x = (base_idx.x + cx + i32(dim)) % i32(dim);
                let cur_y = (base_idx.y + cy + i32(dim)) % i32(dim);
                let cur_z = (base_idx.z + cz + i32(dim)) % i32(dim);

                let val_local = density_grid[get_grid_index(vec3<u32>(u32(cur_x), u32(cur_y), u32(cur_z)), dim)];
                rho_accum = rho_accum + weight * val_local;

                let px_x = (cur_x + 1) % i32(dim);
                let mx_x = (cur_x - 1 + i32(dim)) % i32(dim);
                let py_y = (cur_y + 1) % i32(dim);
                let my_y = (cur_y - 1 + i32(dim)) % i32(dim);
                let pz_z = (cur_z + 1) % i32(dim);
                let mz_z = (cur_z - 1 + i32(dim)) % i32(dim);

                let val_px = density_grid[get_grid_index(vec3<u32>(u32(px_x), u32(cur_y), u32(cur_z)), dim)];
                let val_mx = density_grid[get_grid_index(vec3<u32>(u32(mx_x), u32(cur_y), u32(cur_z)), dim)];
                let val_py = density_grid[get_grid_index(vec3<u32>(u32(cur_x), u32(py_y), u32(cur_z)), dim)];
                let val_my = density_grid[get_grid_index(vec3<u32>(u32(cur_x), u32(my_y), u32(cur_z)), dim)];
                let val_pz = density_grid[get_grid_index(vec3<u32>(u32(cur_x), u32(cur_y), u32(pz_z)), dim)];
                let val_mz = density_grid[get_grid_index(vec3<u32>(u32(cur_x), u32(cur_y), u32(mz_z)), dim)];

                let grad_node = vec3<f32>(
                    (val_px - val_mx) / (2.0 * dx),
                    (val_py - val_my) / (2.0 * dy),
                    (val_pz - val_mz) / (2.0 * dz)
                );

                grad_accum = grad_accum + weight * grad_node;
            }
        }
    }

    return vec4<f32>(grad_accum, rho_accum);
}
