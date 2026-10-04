// Dual-channel particle render pass (shbt6 visual enhancements).
// Channel A (visible_gauge_glow): emission & ionization weighted by the
// gauge-charge state; unquenched anti-baryons carry an oscillating 410 nm
// violet fringe (Enhancement 5), active seeds glow as starburst cores.
// Channel B (passive_metric_distortion): gravitational lensing shear and
// ghost mass density, persisting across de-rendering (E_munu = 0
// conserved); quenched particles emit a Landauer-debt blackbody heat map
// (Enhancement 2) and are billboard-stretched along the local tidal shear
// direction theta_shear = 0.5 atan2(2*gamma_2, gamma_1) (Enhancement 6).
// History crystallization flashes (Enhancement 9): expanding causal-limit
// shockwaves r(t) = c (t - t_GET) around rank-one GET acquisitions.

struct Particle {
    position: vec3<f32>,
    channel: u32,
    velocity: vec3<f32>,
    charge_flags: u32,
    shear: vec4<f32>,
    landauer_debt: f32,
    grav_mass: f32,
    pad: vec2<f32>,
};

struct Camera {
    view_proj: mat4x4<f32>,
    // x: point extent (clip units), y: box_size, z: unwrap_transition
    // (0 = comoving bulk 3D, 1 = boundary CFT torus), w: redshift.
    params: vec4<f32>,
    // xyz: camera world position, w: relativistic Doppler beaming flag.
    aux: vec4<f32>,
    // x/y: viewport size in pixels; z/w: min/max splat radius in pixels
    // (dynamic SPH splat sizing, Thm 9.16).
    params2: vec4<f32>,
};

// Emergent seed defect record (seed_emergence.wgsl output).
struct SeedDefectRecord {
    position: vec4<f32>,
    dynamics: vec4<f32>,
};

// History crystallization event written by causal_point_get.wgsl.
struct CrystallizationEvent {
    origin: vec3<f32>,
    start_time: f32,
    intensity: f32,
    _pad: f32,
};

// Velocity conventions (shbt8): particle.velocity stores the
// supercomoving momentum p_tilde; the physical peculiar velocity is
// v = p_tilde * V_0 / a with V_0 = H_0 * L_box ~ 2.0e4 km/s for the
// canonical 200 Mpc/h patch. Doppler beaming uses the physical beta.
const V0_KMS: f32 = 20000.0;
const C_SPEED: f32 = 299792.458;

// Thermal Stinespring channel overlap (shbt7 Section 4 / Thm 9.10):
// Channel-A visibility follows the anti-baryon scaling dimension
// Delta_Bbar = 26/3 across the modular crossover z_N = 7.356e10,
//   w_vis(z) = (1 - eta_D) + eta_D / (1 + (z_N/z)^Delta),
// with eta_D = 23/33. Channel-B emission is independent of w_vis.
const ETA_D: f32 = 0.6969696970;   // 23/33
const Z_N: f32 = 7.356e10;         // modular crossover redshift
const DELTA_BBAR: f32 = 8.6666667; // 26/3 anti-baryon scaling dimension
const GAMMA_CFT: f32 = 1.4339826830; // Cardy capacity ceiling (1325/924)
const Z_REF_CARDY: f32 = 17.0;     // condensation onset reference

// Cardy capacity ceiling in normalized units (mirror of the
// seed_emergence kernel): n_limit ~ gamma_CFT ((1+z)/18)^7.5.
fn cardy_limit_norm(z: f32) -> f32 {
    let rel = max(1.0 + z, 1.0e-3) / (1.0 + Z_REF_CARDY);
    return GAMMA_CFT * pow(rel, 7.5);
}

// Logarithmic redshift easing chi_z over the simulation domain
// log10(z) in [-3, 14]: splat size eases continuously with the log
// scrub so early-universe markers do not collapse or pop (Thm 9.13).
fn chi_redshift(z: f32) -> f32 {
    return clamp((log2(max(z, 1.0e-3)) * 0.30103 + 3.0) / 17.0, 0.0, 1.0);
}

// Returns (w_vis(z), d w_vis / d z) — the thermal isometric overlap and
// its cooling rate across the modular crossover (shbt8 spec signature).
fn evaluate_stinespring_channel(z: f32) -> vec2<f32> {
    if (z <= 0.0) {
        return vec2<f32>(1.0 - ETA_D, 0.0);
    }
    let zz = max(z, 1.0e-3);
    let power = pow(Z_N / zz, DELTA_BBAR);
    let denom = 1.0 + power;
    let w_vis = (1.0 - ETA_D) + ETA_D / denom;
    let dw_dz = ETA_D * power * DELTA_BBAR / (zz * denom * denom);
    return vec2<f32>(w_vis, dw_dz);
}

// Branchless blackbody color temperature mapping (shbt11): a single
// rational polynomial evaluation in the inverse-temperature coordinate
// u = 1000/T over u in [0.040, 0.400] maps T in [2500, 25000] K onto
// linear RGB with zero SIMD divergence (Thm 9.17).
fn blackbody_to_linear_rgb(t_kelvin: f32) -> vec3<f32> {
    let t = clamp(t_kelvin, 2500.0, 25000.0);
    let u_temp = 1000.0 / t;

    let r = clamp(0.443 + 3.918 * u_temp - 1.842 * u_temp * u_temp, 0.0, 1.0);

    let delta_g = u_temp - 0.154;
    let g = clamp(1.000 - 18.550 * delta_g * delta_g + 27.600 * delta_g * delta_g * delta_g, 0.280, 1.0);

    let delta_b = max(0.0, u_temp - 0.125);
    let b = clamp(1.000 - 3.250 * delta_b - 2.850 * delta_b * delta_b, 0.015, 1.0);

    return vec3<f32>(r, g, b);
}

// Multi-tier thermodynamic and boundary-capacity transfer model
// (shbt11 / Thm 9.16): maps the normalized baryon density rho_b
// (mean = 1.0) and the boundary capacity ratio sigma = N_local/N_limit
// onto a C-infinity effective temperature and surface radiance.
// Returns (t_eff, surface_brightness, debt_activation, softplus_sigma):
//   t_vir      = 2500 + 6500 * tanh(max(0, rho_b - 1) / 4)
//   debt_act   = sigma^3 / (1 + exp(-12 (sigma - 1)))
//   t_eff      = t_vir + 16000 * debt_act                [2500, 25000 K]
//   I_vis      = rho_b (1 - exp(-0.45 rho_b))
//                * (1 + 0.08 rho_b^2 + 4.5 sigma^2 softplus(sigma) (1 + 0.1 m))
// with softplus(sigma) = log(1 + exp(12 (sigma - 1))) / 12.
fn compute_thermodynamic_state(rho_b: f32, sigma: f32, mass_seed: f32) -> vec4<f32> {
    let t_min = 2500.0;
    let delta_t_vir = 6500.0;
    let rho_shock = 4.0;
    let t_corona_max = 16000.0;
    let kappa_sigma = 12.0;

    let t_vir = t_min + delta_t_vir * tanh(max(0.0, rho_b - 1.0) / rho_shock);
    let debt_activation = (sigma * sigma * sigma) / (1.0 + exp(-kappa_sigma * (sigma - 1.0)));
    let t_eff = t_vir + t_corona_max * debt_activation;

    let tau_0 = 0.45;
    let opt_depth_factor = 1.0 - exp(-tau_0 * rho_b);
    let shock_bremsstrahlung = 0.08 * rho_b * rho_b;

    // Overflow-safe softplus: log(1+e^y) = max(y,0) + log(1+e^{-|y|}).
    // The naive exp(kappa * (sigma-1)) overflows f32 for sigma >~ 8,
    // which happens once cardy_limit_norm(z) collapses at low z — inf
    // radiance then propagates as NaN through the tone map and the
    // bilateral post filter, producing black splat-shaped regions.
    let softplus_arg = kappa_sigma * (sigma - 1.0);
    let softplus_sigma = (max(softplus_arg, 0.0)
        + log(1.0 + exp(-abs(softplus_arg)))) / kappa_sigma;
    let landauer_debt_lum = 4.50 * sigma * sigma * softplus_sigma * (1.0 + 0.10 * mass_seed);

    let surface_brightness = rho_b * opt_depth_factor * (1.0 + shock_bremsstrahlung + landauer_debt_lum);

    return vec4<f32>(t_eff, surface_brightness, debt_activation, softplus_sigma);
}

// Blackbody thermal color mapping (Tanner Helland approximation):
// maps an effective Kelvin temperature onto an RGB tint.
fn kelvin_to_rgb(temp_kelvin: f32) -> vec3<f32> {
    let t = clamp(temp_kelvin, 1000.0, 40000.0) / 100.0;
    var r: f32;
    var g: f32;
    var b: f32;

    if (t <= 66.0) {
        r = 1.0;
        g = clamp((99.4708025861 * log(max(t, 1.0)) - 161.1195681661) / 255.0, 0.0, 1.0);
    } else {
        r = clamp((288.1221695283 * pow(t - 60.0, -0.0755148492)) / 255.0, 0.0, 1.0);
        g = clamp((285.6773943610 * pow(t - 60.0, -0.0507658035)) / 255.0, 0.0, 1.0);
    }

    if (t >= 66.0) {
        b = 1.0;
    } else if (t <= 19.0) {
        b = 0.0;
    } else {
        b = clamp((138.5177312231 * log(t - 10.0) - 305.0447927307) / 255.0, 0.0, 1.0);
    }

    return vec3<f32>(r, g, b);
}

// Landauer heat map (Enhancement 2): logarithmic blackbody emission profile
// over the accumulated bit-erasure debt Delta N.
fn compute_landauer_emission(debt: f32) -> vec3<f32> {
    // Normalizes debt across logarithmic operational scales up to 10^12 GW
    let norm = clamp(log(max(debt, 1.0)) / 27.63, 0.0, 1.0);
    let deep_red = vec3<f32>(0.55, 0.01, 0.0);
    let amber = vec3<f32>(1.0, 0.38, 0.05);
    let incandescent_white = vec3<f32>(0.95, 0.92, 1.0);

    var heat = mix(deep_red, amber, smoothstep(0.05, 0.55, norm));
    heat = mix(heat, incandescent_white, smoothstep(0.55, 1.0, norm));
    return heat * (norm * 3.8);
}

// Baryogenesis color map (shbt8 Phase 2): Channel-A baryons (10/33
// sector) render solar-gold (1.0, 0.85, 0.5); unquenched anti-baryons
// (23/33 sector) burn electric-violet (0.8, 0.2, 1.0) at z > 1e12 and
// pass through a thermal cooling gradient over z in [1e12, 1e9] as the
// Stinespring visibility weight w_vis -> 0. Below z = 1e9 anti-baryon
// Channel-A emission is extinguished — they persist only as Channel-B
// gravitational ghosts.
fn resolve_charge_color(p: Particle, time: f32, z: f32) -> vec4<f32> {
    let has_unquenched_charge = (p.charge_flags & 1u) != 0u;

    if (p.channel == 0u) {
        if (has_unquenched_charge) {
            let osc = sin(time * 28.0 + dot(p.position, vec3<f32>(12.0)));
            let electric_violet = vec3<f32>(0.8, 0.2, 1.0);
            // Thermal cooling gradient: effective temperature slides
            // 40000 K -> 2500 K as z sweeps 1e12 -> 1e9.
            let cool = clamp((1.0e12 - z) / (1.0e12 - 1.0e9), 0.0, 1.0);
            let t_eff = 2500.0 + (40000.0 - 2500.0) * (1.0 - cool);
            let cooling = kelvin_to_rgb(t_eff);
            let tint = mix(electric_violet, cooling, cool * 0.8);
            // Anti-baryon visibility floor: w_vis -> 0 below z = 1e9.
            let vis_gate = smoothstep(1.0e9, 3.0e9, z);
            return vec4<f32>(tint * (0.8 + 0.2 * osc), 0.95 * vis_gate);
        }
        return vec4<f32>(1.0, 0.85, 0.5, 0.9);  // Solar-gold baryon (10/33)
    }
    return vec4<f32>(0.12, 0.14, 0.22, 0.25);   // Quenched Channel B ghost
}

@group(0) @binding(0) var<storage, read> particles: array<Particle>;
@group(0) @binding(1) var<uniform> camera: Camera;

@group(1) @binding(0) var<storage, read> events: array<CrystallizationEvent>;
@group(1) @binding(1) var<uniform> event_count: u32;
@group(1) @binding(2) var<storage, read> active_seeds: array<SeedDefectRecord, 256>;
@group(1) @binding(3) var<storage, read> seed_state: array<u32, 4>;

// History crystallization flash (Enhancement 9): incandescent shockwave at
// the causal limit r(t) = c * (t - t_GET).
fn sample_crystallization_flash(pos: vec3<f32>, time: f32) -> vec3<f32> {
    var emission = vec3<f32>(0.0);
    let c = 1.95;
    let wave_thickness = 0.035;
    let count = min(event_count, 32u);

    for (var i = 0u; i < count; i = i + 1u) {
        let ev = events[i];
        let dt = time - ev.start_time;
        if (dt > 0.0 && dt < 0.55) {
            let radius = dt * c;
            let dist = distance(pos, ev.origin);
            let profile = exp(-pow((dist - radius) / wave_thickness, 2.0));
            let fade = 1.0 - (dt / 0.55);
            // 0.25: 64 concurrent condensation shells cover the whole box;
            // any higher gain fuses them into a uniform cyan-white wash.
            let cyan_flash = vec3<f32>(0.7, 0.95, 1.0) * (ev.intensity * 0.25);
            emission += cyan_flash * (profile * fade);
        }
    }
    return emission;
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) vis_factor: f32,
    @location(2) mass: f32,
    @location(3) point_coord: vec2<f32>,
    @location(4) branch_hash: f32,
    @location(5) seed_glow: f32,
    @location(6) causal_env: f32,
    @location(7) doppler_rgb: vec3<f32>,
    @location(8) linear_depth: f32,
    @location(9) landauer_debt: f32,
    @location(10) @interpolate(flat) channel: u32,
    @location(11) @interpolate(flat) charge_flags: u32,
    @location(12) seed_debt: f32,
    // Normalized incubation density ratio n_local/mean written by
    // cs_advance_particles (Thm 9.13 cyan precursor glow).
    @location(13) n_local: f32,
};

struct GBufferOutput {
    @location(0) visible_gauge_glow: vec4<f32>,        // Channel A: Emission & Ionization
    @location(1) passive_metric_distortion: vec4<f32>, // Channel B: Lensing Shear & Ghost Mass
};

@vertex
fn vs_particle_billboard(
    @builtin(vertex_index) vertex_index: u32,
    @builtin(instance_index) instance_index: u32,
) -> VertexOutput {
    var out: VertexOutput;
    let p = particles[instance_index];

    // Comoving box [0, box] -> centered world coordinates [-box/2, box/2].
    var world = p.position - vec3<f32>(camera.params.y * 0.5);
    // Boundary unwrapping: blend the depth coordinate onto the screen as
    // unwrap_transition (camera.params.z) goes 0 -> 1.
    world.z *= 1.0 - camera.params.z;

    var corners = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0),
        vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, 1.0), vec2<f32>(-1.0, 1.0),
    );
    var corner = corners[vertex_index];

    // Tidal shear billboard stretch (Enhancement 6): Channel-B ghosts are
    // deformed along theta_shear = 0.5 atan2(2*gamma_2, gamma_1) with an
    // area-preserving strain matrix (Eq. 358).
    if (p.channel == 1u) {
        let g1 = p.shear.x;
        let g2 = p.shear.y;
        let g_mag = length(vec2<f32>(g1, g2));
        let theta = 0.5 * atan2(2.0 * g2, g1);

        let ct = cos(theta);
        let st = sin(theta);
        let rot = mat2x2<f32>(ct, -st, st, ct);
        let inv_rot = mat2x2<f32>(ct, st, -st, ct);
        let strain = mat2x2<f32>(1.0 + g_mag * 2.2, 0.0, 0.0, 1.0 / (1.0 + g_mag * 2.2));

        corner = rot * strain * inv_rot * corner;
    }

    var clip = camera.view_proj * vec4<f32>(world, 1.0);

    let z_cam = camera.params.w;
    let branch_hash = fract(sin(f32(instance_index) * 12.9898) * 43758.5453);

    // Emergent seed attractor markers: distance to the closest active
    // condensation centroid in normalized box units (no hardcoded wells).
    let u_norm = fract(p.position / camera.params.y);
    var d_min = 1.0e9;
    var seed_debt = 0.0;
    let n_seeds = min(seed_state[1], 256u);
    for (var si = 0u; si < n_seeds; si = si + 1u) {
        let seed = active_seeds[si];
        if (seed.position.w > 0.5) {
            let s_norm = seed.position.xyz / camera.params.y;
            var dd = abs(u_norm - s_norm);
            dd = min(dd, vec3<f32>(1.0) - dd);
            let d = length(dd);
            // Carry the closest defect's Landauer dissipation power
            // P_debt = M_seed * 906 GW for the thermal corona (shbt7 5.3).
            seed_debt = select(seed_debt, seed.dynamics.z, d < d_min);
            d_min = min(d_min, d);
        }
    }

    // Dynamic SPH splat radius in pixels (shbt11 / Thm 9.16): the
    // hydrodynamic smoothing length shrinks with the cube root of the
    // local overdensity, contracts under boundary-capacity saturation
    // sigma = N_local/N_limit, and projects through the camera distance.
    //   R_splat = (min_px + (max_px - min_px) (1 + 0.45 max(0, rho_b - 1))^{-1/3})
    //             * (1 - 0.40 smoothstep(0.8, 1.2, sigma)) * (d_ref / d_cam)
    // rho_b is the normalized incubation density carried on pad.x.
    let cam_pos_w = camera.aux.xyz;
    let cam_dist = max(1.0e-3, length(world - cam_pos_w));
    let rho_b = max(p.pad.x, 0.0);
    // Same unity floor as the fragment-stage capacity ratio: once
    // cardy_limit_norm(z) collapses below 1 the over-capacity regime is
    // universal and rho_b is the discriminative coordinate.
    let sigma = rho_b / max(cardy_limit_norm(z_cam), 1.0);
    let min_px = max(camera.params2.z, 0.5);
    let max_px = max(camera.params2.w, min_px);
    let density_term = pow(1.0 + 0.45 * max(0.0, rho_b - 1.0), -0.3333333);
    let topo_contract = 1.0 - 0.40 * smoothstep(0.8, 1.2, sigma);
    // Nominal orbit distance (1.4 * BOX_SIZE): at the default camera
    // distance the splat range resolves to roughly min_px..max_px,
    // matching the pre-dynamic footprint; zooming in grows the splats
    // up to the 16 px cap as structure resolves.
    let ref_dist = 280.0;
    var splat_px = (min_px + (max_px - min_px) * density_term)
        * topo_contract * (ref_dist / cam_dist);

    // Seed condensation window z in [2, 30]: particles co-moving with an
    // emergent basin render as oversized pulsing attractor markers.
    var seed_glow = 0.0;
    if (z_cam <= 30.0 && z_cam >= 2.0 && d_min < 0.025) {
        seed_glow = 1.0 - d_min / 0.025;
        splat_px = splat_px * 2.5;
    }
    // Active-baryon branch renders slightly larger so the 10/33 share stays
    // legible against the 23/33 anti-baryon population.
    if (branch_hash >= (23.0 / 33.0)) {
        splat_px = splat_px * 1.4;
    }
    // Log-redshift easing (chi_z): splat extent grows 15% toward the
    // de Sitter floor so late-time structure stays resolved.
    splat_px = splat_px * (1.0 + 0.15 * (1.0 - chi_redshift(z_cam)));
    // Absolute 16 px cap (spec) once the seed/branch/chi multipliers land.
    splat_px = clamp(splat_px, min_px, 16.0);
    // Pixels -> NDC half-extent per axis (clip-space offset * w).
    let psize_x = splat_px * 2.0 / max(camera.params2.x, 1.0);
    let psize_y = splat_px * 2.0 / max(camera.params2.y, 1.0);

    // Causal-point projection envelope: faint spherical shell at
    // r = 0.10 box units around each emergent seed while GET clustering
    // (0 < z <= 7).
    var causal_env = 0.0;
    if (z_cam <= 7.0 && z_cam > 0.0) {
        causal_env = exp(-pow((d_min - 0.10) * 40.0, 2.0));
    }

    clip.x += corner.x * psize_x * clip.w;
    clip.y += corner.y * psize_y * clip.w;

    out.clip_position = clip;
    out.world_pos = world;
    // Effective visibility weight: unquenched charges glow fully; quenched
    // Channel-B ghosts are dimmed by the Stinespring decay envelope.
    let decay = clamp(1.0 - f32(p.channel), 0.0, 1.0);
    out.vis_factor = mix(0.25, 1.0, decay);
    out.mass = p.grav_mass;
    out.point_coord = corner * 0.5 + vec2<f32>(0.5);
    out.branch_hash = branch_hash;
    out.seed_glow = seed_glow;
    out.causal_env = causal_env;
    out.landauer_debt = p.landauer_debt;
    out.seed_debt = seed_debt;
    out.channel = p.channel;
    out.charge_flags = p.charge_flags;
    out.n_local = p.pad.x;

    // Relativistic Doppler beaming: line-of-sight velocity beta_los,
    // Doppler factor D = sqrt(1 - beta^2) / (1 - beta_los), cubic
    // intensity boost I_obs = I_0 * D^3 with a blackbody color shift.
    let dist_cam = cam_dist;
    let to_cam = cam_pos_w - world;
    let n_los = to_cam / max(dist_cam, 1.0e-5);
    let a_cam = 1.0 / (1.0 + max(z_cam, -0.999));
    let v_phys = p.velocity * (V0_KMS / a_cam); // p_tilde -> km/s
    let v_mag = length(v_phys);
    let beta = clamp(v_mag / C_SPEED, 0.0, 0.999);
    let beta_los = clamp(dot(v_phys, n_los) / C_SPEED, -0.999, 0.999);
    var doppler_factor = 1.0;
    if (camera.aux.w > 0.5) {
        doppler_factor = sqrt(1.0 - beta * beta) / (1.0 - beta_los);
    }
    let eff_temp = 6500.0 * doppler_factor;
    out.doppler_rgb = kelvin_to_rgb(eff_temp) * pow(doppler_factor, 3.0);
    out.linear_depth = dist_cam;

    return out;
}

@fragment
fn fs_render_dual_channel(in: VertexOutput) -> GBufferOutput {
    var output: GBufferOutput;

    // Gaussian splat profile (shbt8 Phase 2): soft radial falloff
    // exp(-d^2 * 3.5) in quad-radius units replaces the hard-edged
    // point sprite.
    let quad_r = in.point_coord * 2.0 - vec2<f32>(1.0);
    let dist_sq = dot(quad_r, quad_r);
    let dist_from_center = sqrt(dist_sq);
    // Circular splat support (Thm 9.13): discard beyond unit quad
    // radius so the Gaussian never leaves square corner artifacts.
    if (dist_sq > 1.0) {
        discard;
    }

    let z = camera.params.w;
    // Dual-tier radial splat profile (shbt11): a quartic skirt fused with
    // an exponential core that sharpens as the boundary capacity ratio
    // sigma = N_local/N_limit saturates past ~1.
    let rho_b = max(in.n_local, 0.0);
    let n_limit_cardy = cardy_limit_norm(z);
    // Capacity ratio sigma = rho_b / N_limit, floored at the unity
    // regime: cardy_limit_norm(z) collapses to ~1e-5 for z < ~10, which
    // would saturate sigma uniformly and wash out all low-z structure.
    // Once the register is over-capacity everywhere, rho_b itself is
    // the discriminative coordinate (the knee at sigma ~ 1 then tracks
    // mean-normalized density).
    let sigma = rho_b / max(n_limit_cardy, 1.0);
    let radial_falloff = (1.0 - dist_sq) * (1.0 - dist_sq);
    let core_sharpen = exp(-3.5 * dist_sq * (1.0 + 2.0 * smoothstep(0.8, 1.2, sigma)));
    let core_intensity = mix(radial_falloff, core_sharpen, 0.45);

    // Multi-tier thermodynamic transfer (shbt11 / Thm 9.16): the C-infty
    // state (rho_b, sigma) -> (T_eff, I_vis). mass_seed is the
    // log-normalized mass of the nearest condensed defect, M_seed =
    // P_debt / 906 GW in solar units; (1 + 0.1 m_log) stays a bounded
    // corona gain rather than a raw solar-mass multiplier.
    let mass_seed_log = log(1.0 + max(in.seed_debt, 0.0) / 906.0) * 0.4342945;
    // Capacity-saturated branch: past sigma ~ 8 the debt activation and
    // Landauer luminance are fully in their asymptotes, so clamping the
    // transfer input keeps surface_brightness finite (and the tone map
    // NaN-free) without changing the thermodynamic knee.
    let thermo = compute_thermodynamic_state(rho_b, min(sigma, 8.0), mass_seed_log);
    let spectral_rgb = blackbody_to_linear_rgb(thermo.x);

    // Channel A: Visible Gauge Emission via the charge-state color map.
    let is_dark_branch = in.channel == 1u;
    var charge_col = resolve_charge_color(
        Particle(vec3<f32>(0.0), in.channel, vec3<f32>(0.0), in.charge_flags,
                 vec4<f32>(0.0), 0.0, in.mass, vec2<f32>(0.0)),
        in.linear_depth * 0.001,
        z
    );
    // Thermodynamic spectral radiance I_vis weighted by the blackbody
    // locus color at T_eff; ambient filaments sit warm-amber, shocked
    // cores run gold-white, capacity-saturated nodes bleach cyan-white.
    // I_vis spans ~0.5 in the field to ~1e5 inside over-capacity cores
    // (the 4.5*sigma^2*softplus Landauer term), far beyond what additive
    // splats plus the tone map can resolve — cores would fuse into
    // frame-scale white blowout. A display-domain soft knee
    // (x -> x*K/(x+K), K=6) preserves the field regime nearly
    // 1:1 while rolling the top end off at K: dense cores stay hot
    // and ranked brightest, but ~6x — not ~1e5x — the ambient floor,
    // so the anti-blowout ACES knee can hold chromaticity.
    let thermo_vis = thermo.y * 6.0 / (thermo.y + 6.0);
    var emit = spectral_rgb * max(thermo_vis, 0.15) * 1.25;
    let dark_ghost_weight = (1.0 - in.vis_factor) * in.mass;

    // Stinespring carrying fractions: eta_v ramps down while the
    // quenched share eta_d ramps up across the modular crossover.
    let w_vis_pre = evaluate_stinespring_channel(z).x;
    let quench_frac = 1.0 - w_vis_pre;
    let eta_v = max(0.05, w_vis_pre);
    let eta_d = quench_frac;

    if (is_dark_branch) {
        // Passive dark-completion flux carried by the quenched sector.
        emit = vec3<f32>(0.15, 0.08, 0.35)
            * max(dark_ghost_weight, 0.35) * (1.0 + eta_d * 0.75);
    } else if ((in.charge_flags & 1u) != 0u) {
        // charge_col.a carries the w_vis->0 cooling gate below z=1e9.
        emit = charge_col.rgb * 2.2 * (charge_col.a / 0.95) * max(thermo.y, 0.4);
    }
    let pulse = 0.75 + 0.25 * sin(in.branch_hash * 40.0 + z * 0.7);

    // Channel-A emission scales strictly by the thermal Stinespring
    // overlap w_vis(z); Channel B is the independent dark sector and
    // keeps its quadratic ghost dimming (shbt7 5.3).
    let w_vis = w_vis_pre;
    var emit_w = in.vis_factor * eta_v;
    if (is_dark_branch) {
        emit_w = in.vis_factor * in.vis_factor;
    }
    // Apply relativistic beaming: thermal tint keyed to the Doppler-
    // shifted effective temperature plus the cubic intensity boost.
    let doppler_boost = max(in.doppler_rgb, vec3<f32>(0.0));
    let emission = emit * doppler_boost * core_intensity * emit_w;
    // Ionization halo tinted by the branch color so the two populations
    // stay chromatically distinct (gold baryons vs violet anti-baryons).
    let ionization_halo = emit * vec3<f32>(0.35, 0.45, 0.7) * doppler_boost * pow(core_intensity, 2.0) * in.vis_factor;
    var glow_rgb = emission + ionization_halo;
    var glow_a = in.vis_factor * core_intensity;

    // Landauer heat map (Enhancement 2): Channel-B quenched particles
    // glow along the blackbody debt curve.
    if (is_dark_branch && in.landauer_debt > 0.0) {
        glow_rgb += compute_landauer_emission(in.landauer_debt) * core_intensity * 0.35;
    }

    // Precursor incubation glow (Thm 9.13): cyan emission ramps in as
    // the local congestion approaches the Cardy ceiling
    // (n_local/n_limit -> 1) before defects condense. n_limit is the
    // unscaled Cardy ceiling (percolation_scale is a sandbox knob, not
    // a physics parameter of the ceiling itself).
    let incubation_weight = smoothstep(0.4 * n_limit_cardy, n_limit_cardy, in.n_local);
    if (!is_dark_branch && incubation_weight > 0.0) {
        glow_rgb += vec3<f32>(0.10, 0.95, 0.95)
            * incubation_weight * core_intensity * 0.55;
    }

    // History crystallization shockwave (Enhancement 9).
    glow_rgb += sample_crystallization_flash(in.world_pos, z * 0.01 + in.linear_depth * 0.0) * core_intensity;

    // Emergent-seed starburst modulated by the instantaneous Landauer
    // dissipation P_debt = M_seed * 906 GW (shbt7 Phase 2): heavier
    // defects burn brighter amber/white cores with a radiant thermal
    // corona keyed to the same blackbody debt curve.
    if (in.seed_glow > 0.0) {
        let ring_dist = abs(dist_from_center - 0.40);
        let caustic_fringe = exp(-pow(ring_dist * 30.0, 2.0)) * (0.85 + 0.15 * sin(dist_from_center * 45.0 + z * 0.4));
        // Log-normalized debt luminosity: M_seed x 906 GW spans ~1e10-1e12
        // at these seed masses; normalize into [0, 1] on the debt curve.
        let debt_lum = clamp(log(max(in.seed_debt, 1.0)) / 27.63, 0.0, 1.0);
        let core_gain = 0.6 + 1.8 * debt_lum;
        let seed_core = vec3<f32>(1.0, 0.98, 0.85) * exp(-dist_from_center * 12.0) * pulse * core_gain;
        let seed_ring = vec3<f32>(1.0, 0.85, 0.35) * caustic_fringe * pulse * 1.5 + vec3<f32>(0.3, 0.8, 1.0) * caustic_fringe * 1.0;
        // Radiant thermal corona beyond the core: wide gaussian shoulder
        // in the Landauer blackbody palette, gain capped against fusion.
        let corona = compute_landauer_emission(in.seed_debt)
            * exp(-dist_from_center * 5.0) * 0.45;
        glow_rgb += (seed_core + seed_ring + corona) * in.seed_glow;
        glow_a = max(glow_a, in.seed_glow * (core_intensity + caustic_fringe));
    }
    // Near-camera emission fade: when the cinematic camera flies inside
    // the bulk, particles within ~10-35 code units would each cover a
    // 16 px splat disc; dozens of overlapping discs tile the frame as a
    // screen-door bokeh and drive runaway additive blowout. Fading
    // emission below the near-field regime keeps the fly-through legible:
    // mid/far particles at 1-4 px still carry the resolved structure.
    let near_fade = smoothstep(10.0, 35.0, in.linear_depth);

    // Channel A: spectral radiance in RGB, normalized line-of-sight
    // depth in A for the depth-aware bilateral post pass.
    let depth_norm = clamp(in.linear_depth / 4000.0, 0.0, 1.0);
    // Emission budget: ~10+ billboards overlap per pixel at these particle
    // counts; keep per-particle radiance low so the post tone map resolves
    // the emergent structure instead of saturating the whole frame.
    output.visible_gauge_glow = vec4<f32>(glow_rgb * 0.07 * near_fade, depth_norm * max(glow_a, 0.001));

    // Channel B: Passive Gravitational Ghost Distortion
    // Persists regardless of vis_factor, tracking total stress-energy
    // E_munu = 0.
    var shear_dir = vec2<f32>(0.0);
    if (length(in.world_pos.xy) > 1.0e-4) {
        shear_dir = normalize(in.world_pos.xy);
    }
    // Landauer-debt shear bias uses the log-normalized heat scale (0..3.8),
    // not the raw debt: accumulated debts exceed 1e9 late in the scrub and
    // an unbounded bias warps the lens field into a full-screen white-out.
    let debt_heat = compute_landauer_emission(in.landauer_debt).r;
    var gravitational_shear = vec2<f32>(
        shear_dir.x * core_intensity * 0.05,
        shear_dir.y * core_intensity * 0.05
    ) + vec2<f32>(debt_heat * 0.02);

    // Enhanced caustic lensing shear for condensed seed cores
    if (in.seed_glow > 0.0) {
        let seed_caustic_shear = 0.20 * in.seed_glow * exp(-dist_from_center * 6.0);
        gravitational_shear.x += shear_dir.x * seed_caustic_shear;
        gravitational_shear.y += shear_dir.y * seed_caustic_shear;
    }

    // Channel B packing (thin-screen lensing contract):
    //   RG: gravitational shear components (gamma_1, gamma_2)
    //   B:  convergence kappa = 1/2 nabla^2 psi (ghost mass weighted)
    //   A:  observer causal entropy budget (visibility share)
    // κ is summed across ~10+ overlapping billboards in the post pass;
    // keep per-particle weights small so stacked convergence resolves as
    // structure instead of saturating the dark-glow palette.
    let kappa = (in.mass * 0.05 + dark_ghost_weight * 0.2) * core_intensity
        + in.causal_env * 0.35 + in.seed_glow * 1.2 * core_intensity
        + debt_heat * 0.2 + incubation_weight * 0.30 * core_intensity;
    let causal_entropy = in.vis_factor;
    output.passive_metric_distortion = vec4<f32>(
        gravitational_shear.x + in.causal_env * 0.02,
        gravitational_shear.y + in.causal_env * 0.02,
        kappa,
        causal_entropy
    );

    return output;
}
