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
    let count = min(seed_state[1], 256u);
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

// Total supercomoving force at (x_tilde, a): PM + seed + GET (+ hydro
// when the particle belongs to the visible gauge sector).
fn compute_total_force(x_tilde: vec3<f32>, a_eval: f32, is_visible: bool) -> vec3<f32> {
    let drag = 1.0 - (10.0 / 33.0) * cosmo.f_load;
    let a_c = cosmo.g_code * a_eval * drag;
    var total = compute_pm_force(x_tilde, a_c)
        + compute_seed_force(x_tilde, a_c)
        + compute_get_force(x_tilde, cosmo.kappa_get);
    if (is_visible) {
        total = total + compute_hydro_force(x_tilde, a_eval);
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
    p_tilde = p_tilde + cosmo.half_dtau * compute_total_force(x_tilde, cosmo.a, is_visible);
    // Drift
    x_tilde = fract(x_tilde + vec3<f32>(1.0) + cosmo.dtau * p_tilde);
    // Kick 2 (x_{n+1}, a_{n+1})
    p_tilde = p_tilde + cosmo.half_dtau * compute_total_force(x_tilde, cosmo.a_next, is_visible);

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
}
