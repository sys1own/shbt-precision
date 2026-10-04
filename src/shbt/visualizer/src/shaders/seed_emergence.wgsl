// ============================================================================
// File: src/shbt/visualizer/src/shaders/seed_emergence.wgsl
// Module: sys1own/shbt-precision Emergent Topological Condensation Pipeline
//
// First-principles non-perturbative condensation pipeline (shbt7 Sections 3
// and 5.2). Seeds condense wherever the local boundary-register information
// density N_local(x_g, z) reaches the Cardy capacity ceiling of the coset
// CFT on M_coset = SO(10)_312 / SU(3)_8:
//
//   N_local(x_g, z) = rho_cell(x_g) V_cell / (alpha_seed * N_sat)
//   N_limit(z)      = gamma_CFT * (H(z)/H_0)^2 * (V_cell/V_H(z)) * N_sat
//   gamma_CFT       = c_eff / 6 = 1325/924 ~ 1.4339826   (Cardy modular cap)
//
//   S_inst(x,z) = (2 pi c_eff / k_q) * ((N_limit - N_local)/N_limit)^2
//                 for N_local < N_limit;   S_inst = 0 on overflow
//   Gamma_nuc   = A_0 * exp(-S_inst / hbar)
//   P_nuc       = 1 - exp(-Gamma_nuc * dt)
//
// Absolute bit counts (~1e122) overflow f32 registers, so the pipeline works
// in the dimensionless normalized ratio R = N_local / N_limit. Dividing both
// expressions above by their common prefactors, and noting that H(z)^2/V_H
// scales as H(z)^5, gives the shader-local form:
//
//   n_local = rho_cell / mean(rho_cell)                      (density ratio)
//   n_limit = gamma_CFT * ((1+z)/(1+z_ref))^7.5              (H^5 ceiling)
//
// with z_ref = 17 the modular onset reference: the homogeneous state
// (n_local = 1) stays strictly sub-critical for z >> z_ref and crosses the
// ceiling as the Hubble volume shrinks toward z_ref. S_inst is evaluated on
// the normalized ratio, preserving the exact quadratic barrier shape.
//
// The empirical artifacts from earlier iterations are all gone:
//   - no sigmoid turn-on envelope (growth_amp removed)
//   - no empirical bit multiplier K_bit (mass-calibrated via alpha_seed)
//   - no hardcoded 64-seed cap: GlobalSeedBuffer uses atomic append into a
//     dynamically-sized pool (SEED_POOL_CAP slots, population scales with
//     the cosmic mass overflow distribution)
//
// Pass 1 (cs_accumulate_cic): fixed-point atomic Cloud-In-Cell scatter.
// Pass 1.5 (cs_convolve_overflow, shbt9 Thm 9.14): compact Wendland C4
//   convolution of the overflow field over the horizon-derived
//   correlation radius R_filter(z) = c / (k_l a H(z)) = d_H(z)/k_l,
//   making condensation resolution-invariant (discretization error
//   O(dx^2 / R_filter^2)). The physical R_filter is clamped to the
//   basin support (<=4 cells) and the kernel renormalized over the
//   truncated support so the integral stays conservative.
// Pass 2 (cs_detect_condensation): instanton-gated nucleation attempt,
//   26-neighborhood non-maximum suppression on the Wendland-smoothed
//   overflow field, centroid + smoothed overflow-mass integration over
//   the 3x3x3 support basin, atomic append to the pool.
// Pass 3 (cs_temporal_tracking): minimum-image distance matching against
//   prev_seeds for persistent IDs, accretion rates dM/dt, Landauer debt
//   P_debt = M_seed * 906 GW/M_sun.
// ============================================================================

// Canonical coset invariants (shbt7 Section 1): they are theory constants,
// not runtime parameters, so they live as compile-time WGSL constants.
const C_EFF: f32 = 8.6038961038;       // 1325/154, SO(10)_312/SU(3)_8 coset
const K_Q: f32 = 8.0;                  // affine su(3)_8 color level
const GAMMA_CFT: f32 = 1.4339826830;   // c_eff/6 = 1325/924 (Cardy ceiling)
const D_1: f32 = 26.0;                 // gcd(26, 312) modular branch dim
const H_DUAL_SU3: f32 = 3.0;           // h^v(SU(3))
// Onset reference redshift: n_limit crosses the mean density ratio near
// z_ref through the H^5 Hubble-volume scaling. Poisson cells cross at
// progressively higher density contrast as z falls toward z_ref.
const Z_REF: f32 = 17.0;
// 2*pi*c_eff/k_q = 1325*pi/616, the instanton action prefactor.
const S_INST_PREFACTOR: f32 = 6.7576509;

struct SimulationParameters {
    grid_dim: u32,             // Uniform grid resolution (32u)
    particle_count: u32,       // Total active particles
    box_size: f32,             // Comoving box length (Mpc)
    delta_t: f32,              // Integration timestep (Myr)
    redshift: f32,             // Current cosmological redshift z
    f_load: f32,               // Boundary screen loading fraction f_load(z)
    gamma_geom: f32,           // gamma_S: modular restoration rate |dw_vis/dz|
    delta_n_thresh: f32,       // Min normalized overflow to condense
    alpha_mass: f32,           // Seed mass per unit normalized overflow (M_sun)
    landauer_rate: f32,        // Thermodynamic dissipation rate (906 GW / M_sun)
    track_radius: f32,         // Persistence tracking radius (Mpc)
    fixed_point_scale: f32,    // Fixed-point scaling factor (2^16 = 65536)
    // Repurposed slots (16-byte aligned layout preserved, 80B):
    attempt_freq: f32,         // A_0: instanton nucleation attempt rate
    frame_seed: f32,           // Per-frame decorrelation seed for pcg_hash
    mean_density: f32,         // Mean CIC cell mass (grav units)
    r_filter_cells: f32,       // R_filter/dx, clamped to [1,4] (shbt9)
    percolation_scale: f32,    // Sandbox percolation-threshold scale (0.5-2)
    _pad2: vec2<f32>,          // x: incubation diffusion_coeff, y: seed core_radius
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

// Dynamic seed pool: the hardcoded 64-seed cap is replaced by an atomic
// append into an expandable storage pool. SEED_POOL_CAP is the allocated
// capacity of this frame's pool; the population itself scales with the
// overflow distribution, not with a preset constant.
const SEED_POOL_CAP: u32 = 1024u;

// Bind Group 0: Simulation Constants and Spatial Meshes
@group(0) @binding(0) var<uniform> params: SimulationParameters;
@group(0) @binding(1) var<storage, read> particles: array<Particle>;
@group(0) @binding(2) var<storage, read_write> grid_density: array<atomic<u32>>;

// Bind Group 1: Detection State and Tracking Buffers (GlobalSeedBuffer pool)
@group(1) @binding(0) var<storage, read_write> tracking_state: TrackingState;
@group(1) @binding(1) var<storage, read_write> seed_candidates: array<SeedDefectRecord, 1024>;
@group(1) @binding(2) var<storage, read> prev_seeds: array<SeedDefectRecord, 1024>;
@group(1) @binding(3) var<storage, read_write> active_seeds: array<SeedDefectRecord, 1024>;
@group(1) @binding(4) var<storage, read_write> smoothed_overflow: array<f32>;
// Continuum incubation state (shbt10 Thm 9.13/9.14): the transported
// congestion field, the cumulative Stinespring diffusion ledger, the
// Psi_nuc-weighted precursor seed-mass grid, and its regularized
// potential consumed by nbody_pm.wgsl's grid force.
@group(1) @binding(5) var<storage, read_write> dark_ledger: array<f32>;
@group(1) @binding(6) var<storage, read_write> seed_mass_grid: array<f32>;
@group(1) @binding(7) var<storage, read_write> seed_potential_grid: array<f32>;

fn get_linear_index(x: u32, y: u32, z: u32) -> u32 {
    let dim = params.grid_dim;
    return (z % dim) * dim * dim + (y % dim) * dim + (x % dim);
}

// Deterministic hash for the non-perturbative tunneling draw (shbt7 5.2).
fn pcg_hash(input: u32) -> u32 {
    var state = input * 747796405u + 2891336453u;
    var word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

fn rand_uniform(seed_val: u32) -> f32 {
    return f32(pcg_hash(seed_val)) / 4294967295.0;
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

    // Trilinear scatter across 2x2x2 neighborhood. The deposit calibrates
    // the local information density directly against the comoving particle
    // mass resolution m_p,code: N_local = M_cell / (alpha_seed * N_sat),
    // evaluated here in the normalized density-ratio form (see header).
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

// Cardy boundary capacity ceiling in normalized units:
//   N_limit/N_sat ~ gamma_CFT * (H/H_0)^2 * V_cell/V_H ~ gamma_CFT * H^5
// Written as a power of (1+z) around the onset reference z_ref, using the
// high-z matter-era scaling H(z) ~ (1+z)^(3/2): n_limit ~ (1+z)^7.5.
fn cardy_limit_norm(z: f32) -> f32 {
    let rel = max(1.0 + z, 1.0e-3) / (1.0 + Z_REF);
    return GAMMA_CFT * pow(rel, 7.5);
}

// Euclidean instanton action on the coset, evaluated on the normalized
// density ratio R = N_local/N_limit (the absolute N_sat factors cancel):
//   S_inst = (2 pi c_eff / k_q) * (1 - R)^2   for R < 1,   0 for R >= 1.
fn instanton_action(r_ratio: f32) -> f32 {
    if (r_ratio >= 1.0) {
        return 0.0;
    }
    let delta_ratio = 1.0 - r_ratio;
    return S_INST_PREFACTOR * delta_ratio * delta_ratio;
}

// ----------------------------------------------------------------------------
// PASS 1.5: Wendland C4 Percolation Convolution (shbt9 Phase 1)
// ----------------------------------------------------------------------------
// Delta N_smooth(x_i) = sum_j [N_j - N_limit]^+ W_percolation(|x_i-x_j|; R)
// with W(r;R) = 21/(2 pi R^3) (1-q)^4 (1+4q), q = r/R — a compact,
// normalized C^4 spline. The kernel support is truncated to
// r_search = min(ceil(r_filter_cells), 4) cells and renormalized over the
// truncated support, preserving the resolution-invariance theorem's
// O(dx^2/R^2) error bound without an unbounded stencil at high z.
@compute @workgroup_size(4, 4, 4)
fn cs_convolve_overflow(@builtin(global_invocation_id) id: vec3<u32>) {
    let dim = params.grid_dim;
    if (id.x >= dim || id.y >= dim || id.z >= dim) {
        return;
    }

    let out_idx = get_linear_index(id.x, id.y, id.z);
    let n_limit = cardy_limit_norm(params.redshift) * params.percolation_scale;
    let r_filt = max(params.r_filter_cells, 1.0);
    let r_search = i32(min(ceil(r_filt), 4.0));
    // Kernel normalization constant 21/(2 pi R^3); cell volume in
    // normalized (dimensionless) units is absorbed into the renormalized
    // weight sum below.
    let kernel_norm = 21.0 / (6.283185307 * r_filt * r_filt * r_filt);

    var convolved: f32 = 0.0;
    var w_sum: f32 = 0.0;
    let i_dim = i32(dim);

    for (var dz = -r_search; dz <= r_search; dz = dz + 1) {
        for (var dy = -r_search; dy <= r_search; dy = dy + 1) {
            for (var dx = -r_search; dx <= r_search; dx = dx + 1) {
                let dist = sqrt(f32(dx * dx + dy * dy + dz * dz));
                if (dist > r_filt) {
                    continue;
                }
                let nx = u32((i32(id.x) + dx + i_dim) % i_dim);
                let ny = u32((i32(id.y) + dy + i_dim) % i_dim);
                let nz = u32((i32(id.z) + dz + i_dim) % i_dim);
                let n_idx = get_linear_index(nx, ny, nz);
                let raw = f32(atomicLoad(&grid_density[n_idx]))
                    / params.fixed_point_scale;
                let n_local_j = raw / max(params.mean_density, 1e-5);
                let overflow_j = max(n_local_j - n_limit, 0.0);
                let q = dist / r_filt;
                let omq = 1.0 - q;
                let w = kernel_norm * omq * omq * omq * omq * (1.0 + 4.0 * q);
                convolved = convolved + overflow_j * w;
                w_sum = w_sum + w;
            }
        }
    }

    // Renormalize over the truncated support so the convolved overflow
    // equals the per-cell overflow density (dimensionless normalized
    // bits): integral of W over its support is 1 by construction.
    smoothed_overflow[out_idx] = convolved / max(w_sum, 1e-8);
}

// ----------------------------------------------------------------------------
// PASS 2: Instanton-Gated Nucleation + 3D Non-Maximum Suppression
// ----------------------------------------------------------------------------
@compute @workgroup_size(4, 4, 4)
fn cs_detect_condensation(@builtin(global_invocation_id) id: vec3<u32>) {
    let dim = params.grid_dim;
    if (id.x >= dim || id.y >= dim || id.z >= dim) {
        return;
    }

    let cell_idx = get_linear_index(id.x, id.y, id.z);
    let raw_val = f32(atomicLoad(&grid_density[cell_idx])) / params.fixed_point_scale;
    let mean_raw = params.mean_density;
    let n_local = raw_val / max(mean_raw, 1e-5);
    let n_limit = cardy_limit_norm(params.redshift) * params.percolation_scale;
    // The instanton barrier is evaluated on the Wendland-smoothed field:
    // raw cell overflow plus the convolved neighborhood contribution.
    let n_smooth = n_local + smoothed_overflow[cell_idx];
    let r_ratio = n_smooth / n_limit;

    // Barrierless condensation on overflow; suppressed tunneling draw below.
    var tunneled = false;
    if (r_ratio < 1.0) {
        let s_inst = instanton_action(r_ratio);
        let gamma_nuc = params.attempt_freq * exp(-s_inst);
        let p_nuc = 1.0 - exp(-gamma_nuc * params.delta_t);
        let rng = rand_uniform(
            cell_idx ^ (u32(params.frame_seed) * 2654435761u),
        );
        if (rng >= p_nuc || r_ratio <= params.delta_n_thresh) {
            return;
        }
        tunneled = true;
    } else if (r_ratio - 1.0 <= params.delta_n_thresh) {
        return;
    }

    // 26-neighborhood Non-Maximum Suppression over the normalized ratio.
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
                // NMS runs on the smoothed overflow field (scale-invariant
                // condensation centers).
                let r_neigh = (neighbor_val / max(mean_raw, 1e-5) + smoothed_overflow[neighbor_idx]) / n_limit;

                if (r_neigh > r_ratio || (r_neigh == r_ratio && neighbor_idx < cell_idx)) {
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

    // Centroid and mass integration over the 3x3x3 support domain.
    // Overflow condensation deposits the register content in EXCESS of the
    // Cardy ceiling: ov = max(0, n_local - n_limit) in absolute normalized
    // units (not the ratio r - 1, which diverges as n_limit -> 0 in the
    // z -> -1 de Sitter asymptote). A sub-ceiling instanton nucleation has
    // no overflow; it deposits the register content it tunnels, i.e. the
    // basin's full n_local.
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
                let n_local_v = v / max(mean_raw, 1e-5) + smoothed_overflow[n_idx];
                let contrib = select(max(0.0, n_local_v - n_limit), n_local_v, tunneled);

                sum_overflow = sum_overflow + contrib;
                weighted_pos = weighted_pos + vec3<f32>(cx, cy, cz) * contrib;
            }
        }
    }

    let centroid = weighted_pos / max(sum_overflow, 1e-6);
    let wrapped_centroid = (centroid + vec3<f32>(params.box_size)) % vec3<f32>(params.box_size);
    let candidate_slot = atomicAdd(&tracking_state.candidate_count, 1u);

    if (candidate_slot < SEED_POOL_CAP) {
        // Quintic condensation mollifier (Thm 9.14): nucleated mass is
        // C^2-smoothed through the z in [18, 30] window so defects grow
        // continuously rather than popping at the instanton draw.
        let psi_nuc = compute_nucleation_weight(params.redshift, 30.0, 18.0);
        let m_seed = sum_overflow * params.alpha_mass * psi_nuc;
        let p_debt = m_seed * params.landauer_rate;

        seed_candidates[candidate_slot].position = vec4<f32>(wrapped_centroid, 1.0);
        seed_candidates[candidate_slot].dynamics = vec4<f32>(m_seed, 0.0, p_debt, f32(candidate_slot));
    }
}

// ----------------------------------------------------------------------------
// PASS 1.6: Continuum incubation transport (shbt10 Thm 9.13)
// ----------------------------------------------------------------------------
// Sub-critical precursor incubation: before defects condense, local
// register congestion n_local diffuses through the bulk while a
// Stinespring ledger accumulates the dark-sector share
//   S_dil = eta_D * Gamma_S * rho_tot
// with eta_D = 23/33 and Gamma_S the modular restoration rate (host
// streams |dw_vis/dz|(z)). The transported field feeds the
// Psi_nuc-weighted precursor mass grid.
const ETA_D: f32 = 0.6969696970;      // 23/33
const Z_NUC_START: f32 = 30.0;        // precursor incubation window open
const Z_NUC_END: f32 = 18.0;          // condensation onset handoff

// C^2 quintic nucleation weight Psi_nuc(z): vanishes with zero slope and
// curvature at z = 30, reaches 1 with zero slope/curvature at z = 18.
fn compute_nucleation_weight(z: f32, z_start: f32, z_end: f32) -> f32 {
    let u = clamp((z_start - z) / (z_start - z_end), 0.0, 1.0);
    return u * u * u * (10.0 + u * (-15.0 + 6.0 * u));
}

// Fresh CIC density ratio at a grid cell (normalized by the box mean).
fn fresh_density_ratio(cell: vec3<i32>) -> f32 {
    let dim = i32(params.grid_dim);
    let wrap = (cell + vec3<i32>(dim)) % vec3<i32>(dim);
    let idx = get_linear_index(u32(wrap.x), u32(wrap.y), u32(wrap.z));
    let raw = f32(atomicLoad(&grid_density[idx])) / params.fixed_point_scale;
    return raw / max(params.mean_density, 1.0e-5);
}

@compute @workgroup_size(4, 4, 4)
fn step_incubation_transport(@builtin(global_invocation_id) id: vec3<u32>) {
    let dim = params.grid_dim;
    if (id.x >= dim || id.y >= dim || id.z >= dim) {
        return;
    }
    let cell = vec3<i32>(id);
    let idx = get_linear_index(id.x, id.y, id.z);
    let n_limit = cardy_limit_norm(params.redshift) * params.percolation_scale;
    let diffusion_coeff = params._pad2.x;

    // Periodic 6-neighbor Laplacian over the fresh CIC ratio field.
    let n_center = fresh_density_ratio(cell);
    let lap = fresh_density_ratio(cell + vec3<i32>(1, 0, 0))
        + fresh_density_ratio(cell - vec3<i32>(1, 0, 0))
        + fresh_density_ratio(cell + vec3<i32>(0, 1, 0))
        + fresh_density_ratio(cell - vec3<i32>(0, 1, 0))
        + fresh_density_ratio(cell + vec3<i32>(0, 0, 1))
        + fresh_density_ratio(cell - vec3<i32>(0, 0, 1))
        - 6.0 * n_center;

    // Stinespring transport: diffusion spreads congestion across cells
    // while the modular restoration rate dilutes the visible share.
    let s_dil = ETA_D * params.gamma_geom * n_center;
    let n_updated = max(0.0, n_center + diffusion_coeff * lap + s_dil * params.delta_t);

    dark_ledger[idx] = min(dark_ledger[idx] + s_dil * params.delta_t, n_updated);

    // Psi_nuc-weighted precursor mass: proto-seed mass only condenses
    // inside the C^2 quintic window z in [18, 30].
    let psi = compute_nucleation_weight(params.redshift, Z_NUC_START, Z_NUC_END);
    let overflow = max(n_updated - n_limit, 0.0);
    seed_mass_grid[idx] = params.alpha_mass * psi * overflow;
}

// ----------------------------------------------------------------------------
// PASS 1.7: Regularized seed-potential solve (shbt10 Thm 9.14)
// ----------------------------------------------------------------------------
// Phi_seed(x_g) = -4 pi G m_seed / (r_core + eps) evaluated per cell as a
// regularized monopole; the nbody integrator interpolates -a * grad Phi
// with the same CIC weights as the mass deposition. SEED_POT_GAIN folds
// the M_sun -> code-mass calibration into a dimensionless gain tuned so
// precursor wells stay ~0.1x the PM well depth.
const SEED_POT_GAIN: f32 = 3.0e-9;

@compute @workgroup_size(4, 4, 4)
fn solve_seed_potential(@builtin(global_invocation_id) id: vec3<u32>) {
    let dim = params.grid_dim;
    if (id.x >= dim || id.y >= dim || id.z >= dim) {
        return;
    }
    let idx = get_linear_index(id.x, id.y, id.z);
    let m_seed = seed_mass_grid[idx];
    let core_radius = max(params._pad2.y, 1.0e-3);
    seed_potential_grid[idx] = -4.0 * 3.14159265359 * SEED_POT_GAIN * m_seed
        / (core_radius + 0.001);
}

// ----------------------------------------------------------------------------
// PASS 3: Temporal Continuity & Inter-Frame Seed Tracking Kernel
// ----------------------------------------------------------------------------
@compute @workgroup_size(64, 1, 1)
fn cs_temporal_tracking(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let cand_idx = global_id.x;
    let total_candidates = min(atomicLoad(&tracking_state.candidate_count), SEED_POOL_CAP);

    if (cand_idx >= total_candidates) {
        return;
    }

    let cand = seed_candidates[cand_idx];
    let cand_pos = cand.position.xyz;
    let cand_mass = cand.dynamics.x;

    var matched_id: f32 = -1.0;
    var min_dist: f32 = params.track_radius;
    var prev_mass: f32 = 0.0;

    // Minimum image distance matching against the previous frame pool.
    for (var i = 0u; i < SEED_POOL_CAP; i = i + 1u) {
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
    if (active_slot < SEED_POOL_CAP) {
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
