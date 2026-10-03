// ============================================================================
// File: src/shbt/visualizer/src/shaders/seed_emergence.wgsl
// Module: sys1own/shbt-precision Emergent Topological Condensation Pipeline
//
// First-principles emergent mass-congestion pipeline (shbt6 report 1).
// Seeds are NOT hardcoded: they condense wherever the local coordinate
// entropy demand of the bulk exceeds the holographic capacity ceiling of
// the boundary CFT register:
//
//   N_local(x_g, z) = K_bit * rho_cell(x_g) * [1 + delta(x_g, z)]
//   N_limit(z)      = N_sat * f_load(z) / V_box * V_cell * gamma_geom
//   Delta N(x_g,z)  = max(0, N_local - N_limit)
//   M_seed,k        = alpha_seed * sum_{Omega_k} Delta N
//   P_debt,k        = M_seed,k * 906 GW/M_sun
//
// Canonical WZW triple (k_l, k_q, K) = (26, 8, 312); gamma_geom = pi^2/4.
// Delta grows with the linear growth factor D(z) ~ 1/(1+z), normalized so
// that the homogeneous state at z = 30 stays strictly sub-critical.
//
// Pass 1 (cs_accumulate_cic): fixed-point atomic Cloud-In-Cell scatter.
// Pass 2 (cs_detect_condensation): 26-neighborhood non-maximum suppression,
//   centroid + overflow-mass integration over the 3x3x3 support basin.
// Pass 3 (cs_temporal_tracking): minimum-image distance matching against
//   prev_seeds for persistent IDs, accretion rates dM/dt, Landauer debt.
// ============================================================================

struct SimulationParameters {
    grid_dim: u32,             // Uniform grid resolution (128u)
    particle_count: u32,       // Total active particles
    box_size: f32,             // Comoving box length (Mpc)
    delta_t: f32,              // Integration timestep (Myr)
    redshift: f32,             // Current cosmological redshift z
    f_load: f32,               // Boundary screen loading fraction f_load(z)
    gamma_geom: f32,           // Condensation ceiling in units of mean N_local (pi^2/4)
    delta_n_thresh: f32,       // Min normalized overflow Delta_N/gamma-mean to condense
    alpha_mass: f32,           // Seed mass per unit normalized overflow (M_sun)
    landauer_rate: f32,        // Thermodynamic dissipation rate (906 GW / M_sun)
    track_radius: f32,         // Persistence tracking radius (Mpc)
    fixed_point_scale: f32,    // Fixed-point scaling factor (1024.0)
    // Extensions over the reference layout (still 16-byte aligned, 80B):
    k_bit: f32,                // Unused in normalized formulation (kept for layout)
    growth_amp: f32,           // Linear growth amplification (1+z_ref)/(1+z)
    mean_density: f32,         // Mean CIC cell mass (grav units)
    _pad: f32,
};

struct Particle {
    position: vec3<f32>,       // comoving coordinates (Mpc/h)
    channel: u32,              // 0 = Channel A baryon, 1 = Channel B dark ghost
    velocity: vec3<f32>,       // peculiar velocity (km/s)
    charge_flags: u32,         // bit 0: active gauge charge
    shear: vec4<f32>,          // gamma_1, gamma_2, kappa, pad
    landauer_debt: f32,        // accumulated Delta N cost
    grav_mass: f32,            // gravitational coupling mass
    pad: vec2<f32>,
};

struct SeedDefectRecord {
    position: vec4<f32>,       // xyz: comoving centroid, w: active flag (1.0 = active)
    dynamics: vec4<f32>,       // x: mass (M_sun), y: accretion rate dM/dt, z: Landauer power (GW), w: seed_id
};

struct TrackingState {
    candidate_count: atomic<u32>,
    active_seed_count: atomic<u32>,
    pad0: u32,
    pad1: u32,
};

// Bind Group 0: Simulation Constants and Spatial Meshes
@group(0) @binding(0) var<uniform> params: SimulationParameters;
@group(0) @binding(1) var<storage, read> particles: array<Particle>;
@group(0) @binding(2) var<storage, read_write> grid_density: array<atomic<u32>>;

// Bind Group 1: Detection State and Tracking Buffers
@group(1) @binding(0) var<storage, read_write> tracking_state: TrackingState;
@group(1) @binding(1) var<storage, read_write> seed_candidates: array<SeedDefectRecord, 64>;
@group(1) @binding(2) var<storage, read> prev_seeds: array<SeedDefectRecord, 64>;
@group(1) @binding(3) var<storage, read_write> active_seeds: array<SeedDefectRecord, 64>;

fn get_linear_index(x: u32, y: u32, z: u32) -> u32 {
    let dim = params.grid_dim;
    return (z % dim) * dim * dim + (y % dim) * dim + (x % dim);
}

// ----------------------------------------------------------------------------
// PASS 1: Particle-Mesh Cloud-In-Cell Mass Assignment (Fixed-Point Atomics)
// ----------------------------------------------------------------------------
@compute @workgroup_size(256, 1, 1)
fn cs_accumulate_cic(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let p_idx = global_id.x;
    if (p_idx >= params.particle_count) {
        return;
    }

    let p = particles[p_idx];
    let pos = p.position;
    let mass = p.grav_mass;

    let inv_cell = f32(params.grid_dim) / params.box_size;
    let grid_pos = pos * inv_cell;

    let base_x = i32(floor(grid_pos.x));
    let base_y = i32(floor(grid_pos.y));
    let base_z = i32(floor(grid_pos.z));

    let fx = grid_pos.x - f32(base_x);
    let fy = grid_pos.y - f32(base_y);
    let fz = grid_pos.z - f32(base_z);

    let dim = i32(params.grid_dim);

    // Trilinear scatter across 2x2x2 neighborhood
    for (var dz = 0; dz < 2; dz = dz + 1) {
        let wz = select(1.0 - fz, fz, dz == 1);
        let gz = u32((base_z + dz + dim) % dim);

        for (var dy = 0; dy < 2; dy = dy + 1) {
            let wy = select(1.0 - fy, fy, dy == 1);
            let gy = u32((base_y + dy + dim) % dim);

            for (var dx = 0; dx < 2; dx = dx + 1) {
                let wx = select(1.0 - fx, fx, dx == 1);
                let gx = u32((base_x + dx + dim) % dim);

                let weight = wx * wy * wz;
                let val_fixed = u32(round(mass * weight * params.fixed_point_scale));
                let cell_idx = get_linear_index(gx, gy, gz);

                atomicAdd(&grid_density[cell_idx], val_fixed);
            }
        }
    }
}

// Local coordinate entropy demand N_local = K_bit * rho * (1 + delta_eff)
// where delta_eff = delta * growth_amp amplifies the linear density contrast
// by the growth factor D(z) ~ 1/(1+z) so that the homogeneous state stays
// sub-critical (N_local/N_limit = 1/gamma_geom = 0.4053 at delta == 0).
fn cell_entropy_demand(raw_mass: f32, mean_raw: f32) -> f32 {
    // Normalized coordinate entropy demand N_local / N_limit. Since
    // N_limit = gamma * K_bit * mean_raw, the coupling cancels:
    //   N_local/N_limit = (raw/mean_raw) * (1 + delta_eff) / gamma
    // and the capacity ceiling is exactly gamma_geom in these units.
    // Keeping the ratio normalized avoids f32 overflow of the
    // ~1e87-scale absolute bit counts.
    let delta = max(0.0, (raw_mass - mean_raw) / max(mean_raw, 1e-5));
    let delta_eff = delta * params.growth_amp;
    return (raw_mass / max(mean_raw, 1e-5)) * (1.0 + delta_eff) / params.gamma_geom;
}

// ----------------------------------------------------------------------------
// PASS 2: 3D Non-Maximum Suppression & Overflow Peak Condensation
// ----------------------------------------------------------------------------
@compute @workgroup_size(4, 4, 4)
fn cs_detect_condensation(@builtin(global_invocation_id) id: vec3<u32>) {
    let dim = params.grid_dim;
    if (id.x >= dim || id.y >= dim || id.z >= dim) {
        return;
    }

    let cell_idx = get_linear_index(id.x, id.y, id.z);
    let raw_val = f32(atomicLoad(&grid_density[cell_idx])) / params.fixed_point_scale;

    // Mean raw cell mass (particles * mean grav_mass / N_cells).
    let mean_raw = params.mean_density;
    let n_local = cell_entropy_demand(raw_val, mean_raw);

    let overflow = n_local - 1.0;
    if (overflow <= params.delta_n_thresh) {
        return;
    }

    // 26-neighborhood Non-Maximum Suppression
    var is_local_max: bool = true;
    let i_dim = i32(dim);

    for (var dz = -1; dz <= 1; dz = dz + 1) {
        let nz = u32((i32(id.z) + dz + i_dim) % i_dim);
        for (var dy = -1; dy <= 1; dy = dy + 1) {
            let ny = u32((i32(id.y) + dy + i_dim) % i_dim);
            for (var dx = -1; dx <= 1; dx = dx + 1) {
                if (dx == 0 && dy == 0 && dz == 0) {
                    continue;
                }
                let nx = u32((i32(id.x) + dx + i_dim) % i_dim);
                let neighbor_idx = get_linear_index(nx, ny, nz);
                let neighbor_val = f32(atomicLoad(&grid_density[neighbor_idx])) / params.fixed_point_scale;
                let n_neigh = cell_entropy_demand(neighbor_val, mean_raw);

                if (n_neigh > n_local || (n_neigh == n_local && neighbor_idx < cell_idx)) {
                    is_local_max = false;
                    break;
                }
            }
            if (!is_local_max) { break; }
        }
        if (!is_local_max) { break; }
    }

    if (!is_local_max) {
        return;
    }

    // Centroid and mass integration over 3x3x3 support domain
    var sum_overflow: f32 = 0.0;
    var weighted_pos: vec3<f32> = vec3<f32>(0.0);
    let cell_size = params.box_size / f32(dim);

    for (var dz = -1; dz <= 1; dz = dz + 1) {
        let cz = f32(i32(id.z) + dz) * cell_size;
        let nz = u32((i32(id.z) + dz + i_dim) % i_dim);
        for (var dy = -1; dy <= 1; dy = dy + 1) {
            let cy = f32(i32(id.y) + dy) * cell_size;
            let ny = u32((i32(id.y) + dy + i_dim) % i_dim);
            for (var dx = -1; dx <= 1; dx = dx + 1) {
                let cx = f32(i32(id.x) + dx) * cell_size;
                let nx = u32((i32(id.x) + dx + i_dim) % i_dim);

                let n_idx = get_linear_index(nx, ny, nz);
                let v = f32(atomicLoad(&grid_density[n_idx])) / params.fixed_point_scale;
                let n_val = cell_entropy_demand(v, mean_raw);
                let ov = max(0.0, n_val - 1.0);

                sum_overflow = sum_overflow + ov;
                weighted_pos = weighted_pos + vec3<f32>(cx, cy, cz) * ov;
            }
        }
    }

    let centroid = weighted_pos / max(sum_overflow, 1e-6);
    let wrapped_centroid = (centroid + vec3<f32>(params.box_size)) % vec3<f32>(params.box_size);
    let candidate_slot = atomicAdd(&tracking_state.candidate_count, 1u);

    if (candidate_slot < 64u) {
        let m_seed = sum_overflow * params.alpha_mass;
        let p_debt = m_seed * params.landauer_rate;

        seed_candidates[candidate_slot].position = vec4<f32>(wrapped_centroid, 1.0);
        seed_candidates[candidate_slot].dynamics = vec4<f32>(m_seed, 0.0, p_debt, f32(candidate_slot));
    }
}

// ----------------------------------------------------------------------------
// PASS 3: Temporal Continuity & Inter-Frame Seed Tracking Kernel
// ----------------------------------------------------------------------------
@compute @workgroup_size(64, 1, 1)
fn cs_temporal_tracking(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let cand_idx = global_id.x;
    let total_candidates = min(atomicLoad(&tracking_state.candidate_count), 64u);

    if (cand_idx >= total_candidates) {
        return;
    }

    let cand = seed_candidates[cand_idx];
    let cand_pos = cand.position.xyz;
    let cand_mass = cand.dynamics.x;

    var matched_id: f32 = -1.0;
    var min_dist: f32 = params.track_radius;
    var prev_mass: f32 = 0.0;

    // Minimum image distance matching against previous frame seeds
    for (var i = 0u; i < 64u; i = i + 1u) {
        let prev = prev_seeds[i];
        if (prev.position.w > 0.5) {
            var diff = abs(cand_pos - prev.position.xyz);
            diff = min(diff, vec3<f32>(params.box_size) - diff);
            let dist = length(diff);

            if (dist < min_dist) {
                min_dist = dist;
                matched_id = prev.dynamics.w;
                prev_mass = prev.dynamics.x;
            }
        }
    }

    if (matched_id < 0.0) {
        matched_id = f32(cand_idx + 100u);
        prev_mass = cand_mass;
    }

    let active_slot = atomicAdd(&tracking_state.active_seed_count, 1u);
    if (active_slot < 64u) {
        // Accretion carry-over: a persistent defect banks a steady fraction
        // of its basin overflow each frame (linear accretion onto the
        // condensed register), so total defect mass grows monotonically
        // while the seed survives.
        let accreted_mass = prev_mass + cand_mass * 0.02;
        let m_dot = (accreted_mass - prev_mass) / max(params.delta_t, 1e-4);
        let p_debt = accreted_mass * params.landauer_rate;

        active_seeds[active_slot].position = vec4<f32>(cand_pos, 1.0);
        active_seeds[active_slot].dynamics = vec4<f32>(accreted_mass, m_dot, p_debt, matched_id);
    }
}
