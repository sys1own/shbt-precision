// Symplectic supercomoving KDK compute shader (shbt8 Phase 1 / Thm 9.11).
// Advances particle trajectories in supercomoving coordinates
//   x_tilde = x / L_box,  p_tilde = a * v / V_0
// under the Martel-Shapiro conformal Hamiltonian
//   H = p_tilde^2/2 + a^2 Phi_tilde
// with the adaptive Plummer kernel
//   eps_tilde = eta_soft / N_grid   (eta_soft ~ 0.3333)
// replacing the fixed 50 kpc softening. Each step is
//   p_{n+1/2} = p_n + (dtau/2) * F(x_n, a_n)
//   x_{n+1}  = fract(x_n + dtau * p_{n+1/2})          (torus wrap)
//   p_{n+1}  = p_{n+1/2} + (dtau/2) * F(x_{n+1}, a_{n+1})
// with dtau = Da / (a_{1/2}^3 * H(a_{1/2})) and the boundary-loaded
// conformal coupling A(a) = (3/2) Omega_m0 * a * (1 - (10/33) f_load).
//
// Force decomposition at each evaluation point:
//   F_PM   = -A(a) * grad Phi_tilde(x_tilde)          (PM texture rgb)
//   F_seed = -(3 Omega_m0/2)^2 * a * (1 - (10/33) f_load)
//            * sum_k M_tilde,k * dr / (|dr|^2 + eps_tilde^2)^(3/2)
//   F_GET  = -kappa_GET * grad ln rho_proj(x_tilde)   (PM texture alpha)
//   F_hydro = -cs_sq * a * grad ln rho_b(x_tilde)     (P-PM, visible only)
// The P-PM baryon pressure term (shbt9 Eq. a_hydro = -c_s^2 grad ln rho_b,
// Thm 9.13) applies strictly to Channel-A particles (the visible gauge
// sector, h_i >= 23/33 analog); dark ghosts stay collisionless with an
// infinite Jeans wavenumber. The host streams cs_sq = (c_s(z)/V_0)^2
// folded with the sandbox sound-speed scale into cs_sq_scaled.
// Emergent seed masses arrive as M_sun and convert via inv_m_box.
//
// Velocity convention: particle.velocity stores p_tilde (dimensionless
// supercomoving momentum); render passes recover the physical velocity
// as v = p_tilde * V_0 / a for Doppler beaming.
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

// shbt8 SimulationUniforms — byte-exact mirror of
// units::GpuSimulationUniforms (80 bytes, std430 uniform layout).
struct CosmologicalParams {
    a: f32,                  // scale factor at step start
    a_next: f32,             // scale factor at step end
    dtau: f32,               // supercomoving increment Delta_tau
    half_dtau: f32,          // Delta_tau / 2
    h_h0: f32,               // H_SHBT(a_half) / H0
    omega_m0: f32,
    f_load: f32,
    eta_soft: f32,           // adaptive softening scale ~0.3333
    grid_size: f32,          // PM grid N_grid
    eps_soft_sq: f32,        // (eta_soft / N_grid)^2
    kappa_get: f32,          // GET entropic coupling kappa_GET(f_load)
    num_seeds: u32,
    num_particles: u32,
    g_code: f32,             // (3/2) Omega_m0
    a_coupling: f32,         // A(a_n) = g_code * a * (1 - (10/33) f_load)
    box_size: f32,           // L_box, comoving Mpc/h
    dt_legacy: f32,          // code-time tick for de-render accruals
    inv_m_box: f32,          // 1 / M_box in M_sun^-1
    wall_dt: f32,            // wall-clock step (s) for tether fade
    cs_sq_scaled: f32,       // (c_s/V_0)^2 x sound-speed scale (P-PM hydro)
    t_cmb_0: f32,            // T_CMB,0 = 2.7255 K (exact thermal history)
    z_dec: f32,              // thermal decoupling redshift 137.0
    pressure_norm: f32,      // k_B/(mu m_p)/V_0^2 x sound-speed scale
    mean_cell_fixed: f32,    // mean atomic CIC deposit per grid cell
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
@group(0) @binding(5) var<storage, read> active_seeds: array<SeedDefectRecord, 1024>;
@group(0) @binding(6) var<storage, read> seed_state: array<u32, 4>;
// Continuum hydrodynamics grids (Thm 9.15): fixed-point atomic CIC
// deposition feeds the thermodynamic pressure solve; the KDK force
// interpolates the central-difference pressure acceleration with the
// identical trilinear weights (telescopic self-force cancellation,
// Sigma_i F_i = 0 on the periodic mesh).
@group(0) @binding(7) var<storage, read_write> rho_tot_atomic: array<atomic<u32>>;
@group(0) @binding(8) var<storage, read_write> rho_baryon_atomic: array<atomic<u32>>;
@group(0) @binding(9) var<storage, read_write> grid_rho_baryon: array<f32>;
@group(0) @binding(10) var<storage, read_write> grid_pressure_baryon: array<f32>;
// Regularized incubation seed potential Phi_seed(x_g) written by
// solve_seed_potential in seed_emergence.wgsl (Psi_nuc-mollified).
@group(0) @binding(11) var<storage, read> grid_seed_potential: array<f32>;

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
const GAMMA_CFT: f32 = 1.4339826830; // c_eff/6 = 1325/924 Cardy ceiling
const Z_REF: f32 = 17.0;           // condensation onset reference

// Cardy boundary capacity ceiling in normalized units (mirror of the
// seed_emergence kernel): n_limit ~ gamma_CFT * ((1+z)/(1+z_ref))^7.5.
fn cardy_limit_norm(z: f32) -> f32 {
    let rel = max(1.0 + z, 1.0e-3) / (1.0 + Z_REF);
    return GAMMA_CFT * pow(rel, 7.5);
}

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

// PM channel: sample the comoving force gradient field and the
// projected register density (alpha channel) at a supercomoving point.
fn pm_grad(x_tilde: vec3<f32>) -> vec3<f32> {
    return textureSampleLevel(force_grid, force_sampler, fract(x_tilde), 0.0).xyz;
}

fn pm_rho(x_tilde: vec3<f32>) -> f32 {
    return textureSampleLevel(force_grid, force_sampler, fract(x_tilde), 0.0).w;
}

// F_PM = -A(a) * grad Phi_tilde(x_tilde), where A(a) folds the boundary
// load drag 1 - (10/33) f_load into the conformal coupling (shbt8 Eq. 5).
fn compute_pm_force(x_tilde: vec3<f32>, a_coupling: f32) -> vec3<f32> {
    return -a_coupling * pm_grad(x_tilde);
}

// Emergent seed gravity (shbt8 Eq. 6): supercomoving Plummer force
//   F_seed = -(3 Omega_m0/2)^2 * a * (1 - (10/33) f_load)
//            * sum_k M_tilde,k * dr / (|dr|^2 + eps_tilde^2)^(3/2)
// with minimum-image separation dr on the torus and eps_tilde =
// eta_soft / N_grid adaptive softening. The kernel coefficient factors
// as a_coupling * g_code.
fn compute_seed_force(x_tilde: vec3<f32>, a_coupling: f32) -> vec3<f32> {
    var acc = vec3<f32>(0.0);
    let count = min(seed_state[1], 1024u);
    for (var k = 0u; k < count; k = k + 1u) {
        let seed = active_seeds[k];
        if (seed.position.w > 0.5) {
            let x_seed = seed.position.xyz / cosmo.box_size;
            var dr = x_tilde - x_seed;
            dr = dr - round(dr); // minimum image on the torus
            let m_code = seed.dynamics.x * cosmo.inv_m_box;
            acc = acc + m_code * dr / pow(dot(dr, dr) + cosmo.eps_soft_sq, 1.5);
        }
    }
    return -a_coupling * cosmo.g_code * acc;
}

// Projected-density log gradient on the PM alpha channel, shared by the
// GET entropic pull and the P-PM baryon pressure force.
fn grad_ln_rho(x_tilde: vec3<f32>) -> vec3<f32> {
    let eps = 1.0 / cosmo.grid_size;
    let rho_c = pm_rho(x_tilde);
    let gx = pm_rho(x_tilde + vec3<f32>(eps, 0.0, 0.0))
        - pm_rho(x_tilde - vec3<f32>(eps, 0.0, 0.0));
    let gy = pm_rho(x_tilde + vec3<f32>(0.0, eps, 0.0))
        - pm_rho(x_tilde - vec3<f32>(0.0, eps, 0.0));
    let gz = pm_rho(x_tilde + vec3<f32>(0.0, 0.0, eps))
        - pm_rho(x_tilde - vec3<f32>(0.0, 0.0, eps));
    return vec3<f32>(gx, gy, gz) / (2.0 * eps * max(rho_c, 1.0e-3));
}

// GET entropic transport (shbt7 Thm 9.8 / shbt8 Eq. 7):
//   F_GET = -kappa_GET(f_load) * grad ln rho_proj(x_tilde)
fn compute_get_force(x_tilde: vec3<f32>, kappa_get: f32) -> vec3<f32> {
    return -kappa_get * grad_ln_rho(x_tilde) * 0.01;
}

// Conservative P-PM baryon pressure force (shbt9 Thm 9.13):
//   a_hydro = -(c_s^2 / rho_b) grad P_b = -c_s^2 grad ln rho_b
// Restricted to the visible sector (channel == 0, the h_i >= eta_D gauge
// sector); ghosts collapse collisionlessly. In code units the pressure
// acceleration carries the same conformal a-factor as gravity, so the
// force is -cs_sq * a * grad_ln_rho with cs_sq = (c_s/V_0)^2 x scale.
fn compute_hydro_force(x_tilde: vec3<f32>, a_eval: f32) -> vec3<f32> {
    return -cosmo.cs_sq_scaled * a_eval * grad_ln_rho(x_tilde);
}

// Continuum hydrodynamics (Thm 9.15): fixed-point atomic CIC
// deposition feeds the thermodynamic pressure solve; the KDK kick
// interpolates the central-difference pressure acceleration with the
// identical trilinear weights (telescopic self-force cancellation,
// Sigma_i F_i = 0 on the periodic mesh).
const FIXED_POINT_SCALE: f32 = 1048576.0; // 2^20 atomic mass deposits

fn grid_idx_cell(cell: vec3<i32>, gd: u32) -> u32 {
    let wrap = (cell + vec3<i32>(i32(gd))) % vec3<i32>(i32(gd));
    return u32(wrap.x + wrap.y * i32(gd) + wrap.z * i32(gd) * i32(gd));
}

/// CIC-weighted central-difference pressure acceleration: for each of
/// the 8 deposition cells the local acceleration
///   a_press(c) = -(a / rho_b(c)) * grad p_gas(c)
/// is evaluated from the periodic 6-point stencil, then interpolated
/// with the same trilinear weights used at deposit time (exact
/// telescopic cancellation: Sigma_i F_i = 0 on the periodic mesh).
fn interpolate_pressure_force(frac_uvw: vec3<f32>, grid_dim: u32, cell_size: f32, a: f32) -> vec3<f32> {
    let base = vec3<i32>(floor(frac_uvw));
    let wx = frac_uvw - floor(frac_uvw);
    var acc = vec3<f32>(0.0);
    let inv_cell = 1.0 / cell_size;
    for (var dz = 0; dz < 2; dz++) {
        for (var dy = 0; dy < 2; dy++) {
            for (var dx = 0; dx < 2; dx++) {
                let w = select(1.0 - wx.x, wx.x, dx == 1)
                    * select(1.0 - wx.y, wx.y, dy == 1)
                    * select(1.0 - wx.z, wx.z, dz == 1);
                let cell = base + vec3<i32>(dx, dy, dz);
                let i0 = grid_idx_cell(cell, grid_dim);
                let rho_c = grid_rho_baryon[i0];
                if (rho_c > 1.0e-12) {
                    let grad_p = vec3<f32>(
                        grid_pressure_baryon[grid_idx_cell(cell + vec3<i32>(1, 0, 0), grid_dim)]
                            - grid_pressure_baryon[grid_idx_cell(cell - vec3<i32>(1, 0, 0), grid_dim)],
                        grid_pressure_baryon[grid_idx_cell(cell + vec3<i32>(0, 1, 0), grid_dim)]
                            - grid_pressure_baryon[grid_idx_cell(cell - vec3<i32>(0, 1, 0), grid_dim)],
                        grid_pressure_baryon[grid_idx_cell(cell + vec3<i32>(0, 0, 1), grid_dim)]
                            - grid_pressure_baryon[grid_idx_cell(cell - vec3<i32>(0, 0, 1), grid_dim)],
                    ) * (0.5 * inv_cell);
                    acc += w * (-(a / rho_c) * grad_p);
                }
            }
        }
    }
    return acc;
}

/// CIC-weighted incubation seed-potential acceleration
/// a = -a * grad Phi_seed over the Psi_nuc-mollified precursor grid
/// (applies to both gauge sectors).
fn interpolate_seed_potential_force(frac_uvw: vec3<f32>, grid_dim: u32, cell_size: f32, a: f32) -> vec3<f32> {
    let base = vec3<i32>(floor(frac_uvw));
    let wx = frac_uvw - floor(frac_uvw);
    var acc = vec3<f32>(0.0);
    let inv_cell = 1.0 / cell_size;
    for (var dz = 0; dz < 2; dz++) {
        for (var dy = 0; dy < 2; dy++) {
            for (var dx = 0; dx < 2; dx++) {
                let w = select(1.0 - wx.x, wx.x, dx == 1)
                    * select(1.0 - wx.y, wx.y, dy == 1)
                    * select(1.0 - wx.z, wx.z, dz == 1);
                let cell = base + vec3<i32>(dx, dy, dz);
                let grad_phi = vec3<f32>(
                    grid_seed_potential[grid_idx_cell(cell + vec3<i32>(1, 0, 0), grid_dim)]
                        - grid_seed_potential[grid_idx_cell(cell - vec3<i32>(1, 0, 0), grid_dim)],
                    grid_seed_potential[grid_idx_cell(cell + vec3<i32>(0, 1, 0), grid_dim)]
                        - grid_seed_potential[grid_idx_cell(cell - vec3<i32>(0, 1, 0), grid_dim)],
                    grid_seed_potential[grid_idx_cell(cell + vec3<i32>(0, 0, 1), grid_dim)]
                        - grid_seed_potential[grid_idx_cell(cell - vec3<i32>(0, 0, 1), grid_dim)],
                ) * (0.5 * inv_cell);
                acc += w * (-a * grad_phi);
            }
        }
    }
    return acc;
}

// Total supercomoving force at (x_tilde, a): PM + seed + incubation
// potential + GET (+ thermodynamic pressure when the particle belongs
// to the visible gauge sector).
fn compute_total_force(x_tilde: vec3<f32>, a_eval: f32, is_visible: bool, hydro_gate: f32) -> vec3<f32> {
    let drag = 1.0 - (10.0 / 33.0) * cosmo.f_load;
    let a_c = cosmo.g_code * a_eval * drag;
    let cell_size = cosmo.box_size / f32(cosmo.grid_size);
    let frac_uvw = x_tilde * f32(cosmo.grid_size);
    var total = compute_pm_force(x_tilde, a_c)
        + compute_seed_force(x_tilde, a_c)
        + compute_get_force(x_tilde, cosmo.kappa_get)
        + interpolate_seed_potential_force(frac_uvw, u32(cosmo.grid_size), cell_size, a_eval);
    if (is_visible) {
        total = total + hydro_gate * interpolate_pressure_force(frac_uvw, u32(cosmo.grid_size), cell_size, a_eval);
    }
    return total;
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
        p.landauer_debt += 906.0 * cosmo.dt_legacy * 1.0e3;
    }

    // Thermodynamic hydro floor (Thm 9.15): the pressure kick is
    // attenuated once the particle's local congestion falls below the
    // eta_D = 23/33 dual-horizon fraction of the Cardy ceiling — the
    // register erases thermal support and the particle returns to
    // collisionless infall. p.pad.x == 0 (grid not yet deposited this
    // frame) defaults to full support.
    let n_limit = cardy_limit_norm(z_current);
    var hydro_gate = 1.0;
    if (p.pad.x > 0.0 && n_limit > 1.0e-6) {
        hydro_gate = smoothstep(0.35 * ETA_D, ETA_D, p.pad.x / n_limit);
    }

    // Tether fade (Enhancement 4): each thread decays one vertex pair of
    // the TetherVertexBuffer by exp(-wall_dt / 0.5 s), so Stinespring
    // lines dissolve over ~0.5 s of wall time.
    let v_base = idx * 2u;
    if (v_base + 1u < 65536u && v_base + 1u < atomicLoad(&indirect_draw.vertex_count)) {
        let tether_fade = exp(-cosmo.wall_dt / 0.5);
        tethers[v_base].alpha_decay = tethers[v_base].alpha_decay * tether_fade;
        tethers[v_base + 1u].alpha_decay = tethers[v_base + 1u].alpha_decay * tether_fade;
    }

    // Martel-Shapiro supercomoving KDK step (shbt8 Eq. 3-8):
    //   Stage 1: half-kick at (x_n, a_n)
    //   Stage 2: drift x_{n+1} = fract(x_n + dtau * p_{n+1/2})
    //   Stage 3: half-kick at (x_{n+1}, a_{n+1})
    var x_tilde = fract(p.position / cosmo.box_size);
    var p_tilde = p.velocity;

    // Kick 1 (x_n, a_n)
    let is_visible = p.channel == 0u;
    p_tilde = p_tilde + cosmo.half_dtau * compute_total_force(x_tilde, cosmo.a, is_visible, hydro_gate);
    // Drift
    x_tilde = fract(x_tilde + vec3<f32>(1.0) + cosmo.dtau * p_tilde);
    // Kick 2 (x_{n+1}, a_{n+1})
    p_tilde = p_tilde + cosmo.half_dtau * compute_total_force(x_tilde, cosmo.a_next, is_visible, hydro_gate);

    // Numerical bound: cap the supercomoving momentum so a single
    // soft-kernel fluctuation cannot send the particle to NaN (register
    // overflow in the integrator corresponds to a frame-drop in the
    // SHBT picture). |p_tilde| <= 0.5 keeps the drift sub-box per step.
    p_tilde = clamp(p_tilde, vec3<f32>(-0.5), vec3<f32>(0.5));

    p.velocity = p_tilde;
    p.position = x_tilde * cosmo.box_size;

    // Projected tidal shear estimate from the PM field for the Channel-B
    // billboard stretch (gamma ~ grad of the sampled force magnitude).
    let eps = 1.0 / cosmo.grid_size;
    let uvw = x_tilde;
    let fx1 = textureSampleLevel(force_grid, force_sampler, uvw + vec3<f32>(eps, 0.0, 0.0), 0.0).x;
    let fx0 = textureSampleLevel(force_grid, force_sampler, uvw - vec3<f32>(eps, 0.0, 0.0), 0.0).x;
    let fy1 = textureSampleLevel(force_grid, force_sampler, uvw + vec3<f32>(0.0, eps, 0.0), 0.0).y;
    let fy0 = textureSampleLevel(force_grid, force_sampler, uvw - vec3<f32>(0.0, eps, 0.0), 0.0).y;
    p.shear = vec4<f32>(fx1 - fx0, fy1 - fy0, length(pm_grad(uvw)), 0.0);

    particles[idx] = p;

    // Scatter particle density onto the PM grid cell.
    let cell = vec3<i32>(uvw * cosmo.grid_size);
    textureStore(density_grid, cell, vec4<f32>(p.grav_mass));

    // Normalized incubation density ratio n_local/mean for the Channel-A
    // incubation-weight glow (Thm 9.13); carried in particle pad.x.
    let home = vec3<i32>(floor(uvw * cosmo.grid_size));
    particles[idx].pad.x =
        f32(atomicLoad(&rho_tot_atomic[grid_idx_cell(home, u32(cosmo.grid_size))]))
        / max(cosmo.mean_cell_fixed, 1.0);
}

// ----------------------------------------------------------------------------
// shbt10 continuum hydro (Thm 9.15): fixed-point CIC deposition +
// thermodynamic pressure field
// ----------------------------------------------------------------------------

/// Pass H1: atomic fixed-point CIC mass assignment at S = 2^20. Total
/// mass goes to rho_tot_atomic (pad.x incubation ratio denominator);
/// Channel-A (visible) mass additionally accumulates into
/// rho_baryon_atomic for the baryon pressure field.
@compute @workgroup_size(256, 1, 1)
fn deposit_mass_cic(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= cosmo.num_particles) {
        return;
    }
    let p = particles[idx];
    let gd = i32(cosmo.grid_size);
    let inv_cell = cosmo.grid_size / cosmo.box_size;
    let grid_pos = p.position * inv_cell;
    let base = vec3<i32>(floor(grid_pos));
    let frac = grid_pos - floor(grid_pos);
    let is_baryon = p.channel == 0u;
    for (var dz = 0; dz < 2; dz = dz + 1) {
        let wz = select(1.0 - frac.z, frac.z, dz == 1);
        let gz = (base.z + dz + gd) % gd;
        for (var dy = 0; dy < 2; dy = dy + 1) {
            let wy = select(1.0 - frac.y, frac.y, dy == 1);
            let gy = (base.y + dy + gd) % gd;
            for (var dx = 0; dx < 2; dx = dx + 1) {
                let wx = select(1.0 - frac.x, frac.x, dx == 1);
                let gx = (base.x + dx + gd) % gd;
                let w = wx * wy * wz;
                let fixed = u32(round(p.grav_mass * w * FIXED_POINT_SCALE));
                let cell = grid_idx_cell(vec3<i32>(gx, gy, gz), u32(cosmo.grid_size));
                atomicAdd(&rho_tot_atomic[cell], fixed);
                if (is_baryon) {
                    atomicAdd(&rho_baryon_atomic[cell], fixed);
                }
            }
        }
    }
}

/// Pass H2: thermodynamic pressure assembly on the exact baryon
/// temperature history (Thm 9.15):
///   T_b(z) = T_CMB,0 (1+z) z / (z + z_dec),   T_CMB,0 = 2.7255, z_dec = 137
///   p_gas = (k_B / (mu m_p) / V_0^2) * T_b(z) * rho_b
/// evaluated cell-wise; the KDK kick interpolates its central
/// differences so mesh self-force cancels telescopically.
@compute @workgroup_size(4, 4, 4)
fn compute_thermodynamic_pressure(@builtin(global_invocation_id) gid: vec3<u32>) {
    let gd = u32(cosmo.grid_size);
    if (gid.x >= gd || gid.y >= gd || gid.z >= gd) {
        return;
    }
    let idx = grid_idx_cell(vec3<i32>(gid), gd);
    let rho_b = f32(atomicLoad(&rho_baryon_atomic[idx])) / FIXED_POINT_SCALE;
    grid_rho_baryon[idx] = rho_b;
    let z = (1.0 / max(cosmo.a, 1.0e-6)) - 1.0;
    let zz = max(z, 1.0e-4);
    let t_b = cosmo.t_cmb_0 * (1.0 + z) * (zz / (zz + cosmo.z_dec));
    grid_pressure_baryon[idx] = cosmo.pressure_norm * t_b * rho_b;
}
