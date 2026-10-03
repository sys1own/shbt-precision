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

// Characteristic velocity scale used to normalize particle velocities
// into relativistic beta factors for the Doppler beaming model.
const C_SPEED: f32 = 60.0;

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

// Anti-baryon charge visualization (Enhancement 5): unquenched Channel-A
// anti-baryons fringe at ~410 nm violet; quenched Channel-B ghosts fall
// back to the dark-sector tint.
fn resolve_charge_color(p: Particle, time: f32) -> vec4<f32> {
    let has_unquenched_charge = (p.charge_flags & 1u) != 0u;

    if (p.channel == 0u) {
        if (has_unquenched_charge) {
            let osc = sin(time * 28.0 + dot(p.position, vec3<f32>(12.0)));
            let deep_violet = vec3<f32>(0.62, 0.12, 0.94);
            let charge_fringe = vec3<f32>(0.85, 0.45, 1.0) * (0.8 + 0.2 * osc);
            return vec4<f32>(mix(deep_violet, charge_fringe, 0.6), 0.95);
        }
        return vec4<f32>(0.92, 0.88, 0.82, 0.85); // Standard Channel A baryon
    }
    return vec4<f32>(0.12, 0.14, 0.22, 0.25);     // Quenched Channel B dark ghost
}

@group(0) @binding(0) var<storage, read> particles: array<Particle>;
@group(0) @binding(1) var<uniform> camera: Camera;

@group(1) @binding(0) var<storage, read> events: array<CrystallizationEvent>;
@group(1) @binding(1) var<uniform> event_count: u32;
@group(1) @binding(2) var<storage, read> active_seeds: array<SeedDefectRecord, 64>;
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
    let n_seeds = min(seed_state[1], 64u);
    for (var si = 0u; si < n_seeds; si = si + 1u) {
        let seed = active_seeds[si];
        if (seed.position.w > 0.5) {
            let s_norm = seed.position.xyz / camera.params.y;
            var dd = abs(u_norm - s_norm);
            dd = min(dd, vec3<f32>(1.0) - dd);
            d_min = min(d_min, length(dd));
        }
    }

    // Seed condensation window z in [2, 30]: particles co-moving with an
    // emergent basin render as oversized pulsing attractor markers.
    var psize = camera.params.x;
    var seed_glow = 0.0;
    // Radius kept small: 64 saturated defects otherwise overlap into a
    // screen-filling starburst wash.
    if (z_cam <= 30.0 && z_cam >= 2.0 && d_min < 0.025) {
        seed_glow = 1.0 - d_min / 0.025;
        psize = psize * 2.5;
    }
    // Active-baryon branch renders slightly larger so the 10/33 share stays
    // legible against the 23/33 anti-baryon population.
    if (branch_hash >= (23.0 / 33.0)) {
        psize = psize * 1.4;
    }

    // Causal-point projection envelope: faint spherical shell at
    // r = 0.10 box units around each emergent seed while GET clustering
    // (0 < z <= 7).
    var causal_env = 0.0;
    if (z_cam <= 7.0 && z_cam > 0.0) {
        causal_env = exp(-pow((d_min - 0.10) * 40.0, 2.0));
    }

    clip.x += corner.x * psize * clip.w;
    clip.y += corner.y * psize * clip.w;

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
    out.channel = p.channel;
    out.charge_flags = p.charge_flags;

    // Relativistic Doppler beaming: line-of-sight velocity beta_los,
    // Doppler factor D = sqrt(1 - beta^2) / (1 - beta_los), cubic
    // intensity boost I_obs = I_0 * D^3 with a blackbody color shift.
    let to_cam = camera.aux.xyz - world;
    let dist_cam = length(to_cam);
    let n_los = to_cam / max(dist_cam, 1.0e-5);
    let v_mag = length(p.velocity);
    let beta = clamp(v_mag / C_SPEED, 0.0, 0.999);
    let beta_los = clamp(dot(p.velocity, n_los) / C_SPEED, -0.999, 0.999);
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

    let dist_from_center = length(in.point_coord - vec2<f32>(0.5));

    let core_intensity = exp(-dist_from_center * 8.0);
    let z = camera.params.w;

    // Channel A: Visible Gauge Emission via the charge-state color map.
    let is_dark_branch = in.channel == 1u;
    var charge_col = resolve_charge_color(
        Particle(vec3<f32>(0.0), in.channel, vec3<f32>(0.0), in.charge_flags,
                 vec4<f32>(0.0), 0.0, in.mass, vec2<f32>(0.0)),
        in.linear_depth * 0.001
    );
    var emit = vec3<f32>(1.0, 0.85, 0.45) * 1.8;
    let dark_ghost_weight = (1.0 - in.vis_factor) * in.mass;

    if (is_dark_branch) {
        emit = vec3<f32>(0.12, 0.04, 0.28) * max(dark_ghost_weight, 0.35);
    } else if ((in.charge_flags & 1u) != 0u) {
        emit = charge_col.rgb * 2.2;
    }
    let pulse = 0.75 + 0.25 * sin(in.branch_hash * 40.0 + z * 0.7);

    // Dark-branch emission decays quadratically with vis_weight so the
    // de-rendering population dims faster than the baryon branch.
    var emit_w = in.vis_factor;
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

    // History crystallization shockwave (Enhancement 9).
    glow_rgb += sample_crystallization_flash(in.world_pos, z * 0.01 + in.linear_depth * 0.0) * core_intensity;

    // Emergent-seed starburst: glowing white/gold cores with caustic
    // lensing rings around freshly condensed topological defects.
    if (in.seed_glow > 0.0) {
        let ring_dist = abs(dist_from_center - 0.40);
        let caustic_fringe = exp(-pow(ring_dist * 30.0, 2.0)) * (0.85 + 0.15 * sin(dist_from_center * 45.0 + z * 0.4));
        // Gains trimmed so 64 concurrent defects read as distinct
        // starbursts rather than a merged white field.
        let seed_core = vec3<f32>(1.0, 0.98, 0.85) * exp(-dist_from_center * 12.0) * pulse * 2.0;
        let seed_ring = vec3<f32>(1.0, 0.85, 0.35) * caustic_fringe * pulse * 1.5 + vec3<f32>(0.3, 0.8, 1.0) * caustic_fringe * 1.0;
        glow_rgb += (seed_core + seed_ring) * in.seed_glow;
        glow_a = max(glow_a, in.seed_glow * (core_intensity + caustic_fringe));
    }
    // Channel A: spectral radiance in RGB, normalized line-of-sight
    // depth in A for the depth-aware bilateral post pass.
    let depth_norm = clamp(in.linear_depth / 4000.0, 0.0, 1.0);
    // Emission budget: ~10+ billboards overlap per pixel at these particle
    // counts; keep per-particle radiance low so the post tone map resolves
    // the emergent structure instead of saturating the whole frame.
    output.visible_gauge_glow = vec4<f32>(glow_rgb * 0.07, depth_norm * max(glow_a, 0.001));

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
        + debt_heat * 0.2;
    let causal_entropy = in.vis_factor;
    output.passive_metric_distortion = vec4<f32>(
        gravitational_shear.x + in.causal_env * 0.02,
        gravitational_shear.y + in.causal_env * 0.02,
        kappa,
        causal_entropy
    );

    return output;
}
