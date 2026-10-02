// Dual-channel particle render pass.
// Channel A (visible_gauge_glow): emission & ionization weighted by vis_weight.
// Channel B (passive_metric_distortion): gravitational lensing shear and ghost
// mass density, persisting regardless of vis_weight (E_munu = 0 conserved).

struct Particle {
    position: vec3<f32>,
    vis_weight: f32,
    velocity: vec3<f32>,
    grav_mass: f32,
};

struct Camera {
    view_proj: mat4x4<f32>,
    // x: point extent (clip units), y: box_size, z: unwrap_transition
    // (0 = comoving bulk 3D, 1 = boundary CFT torus), w: redshift.
    params: vec4<f32>,
};

@group(0) @binding(0) var<storage, read> particles: array<Particle>;
@group(0) @binding(1) var<uniform> camera: Camera;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) vis_factor: f32,
    @location(2) mass: f32,
    @location(3) point_coord: vec2<f32>,
    @location(4) branch_hash: f32,
    @location(5) seed_glow: f32,
    @location(6) causal_env: f32,
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
    let corner = corners[vertex_index];

    var clip = camera.view_proj * vec4<f32>(world, 1.0);

    let z_cam = camera.params.w;
    let branch_hash = fract(sin(f32(instance_index) * 12.9898) * 43758.5453);

    // Ghost-seed attractor wells (mirror make_force_grid in lib.rs) in
    // normalized box units; used for condensation markers and causal shells.
    var wells = array<vec3<f32>, 4>(
        vec3<f32>(0.25, 0.25, 0.25),
        vec3<f32>(0.75, 0.75, 0.25),
        vec3<f32>(0.25, 0.75, 0.75),
        vec3<f32>(0.75, 0.25, 0.75),
    );
    let u_norm = fract(p.position / camera.params.y);
    var d_min = 1.0e9;
    for (var wi = 0; wi < 4; wi = wi + 1) {
        d_min = min(d_min, distance(u_norm, wells[wi]));
    }

    // Seed condensation window z in [2, 30]: particles co-moving with a
    // well render as oversized pulsing attractor markers.
    var psize = camera.params.x;
    var seed_glow = 0.0;
    if (z_cam <= 30.0 && z_cam >= 2.0 && d_min < 0.06) {
        seed_glow = 1.0 - d_min / 0.06;
        psize = psize * 4.0;
    }
    // Active-baryon branch renders slightly larger so the 10/33 share stays
    // legible against the 23/33 anti-baryon population.
    if (branch_hash >= (23.0 / 33.0)) {
        psize = psize * 1.4;
    }

    // Causal-point projection envelope: faint spherical shell at
    // r = 0.10 box units around each well while GET clustering (0 < z <= 7).
    var causal_env = 0.0;
    if (z_cam <= 7.0 && z_cam > 0.0) {
        causal_env = exp(-pow((d_min - 0.10) * 40.0, 2.0));
    }

    clip.x += corner.x * psize * clip.w;
    clip.y += corner.y * psize * clip.w;

    out.clip_position = clip;
    out.world_pos = world;
    out.vis_factor = p.vis_weight;
    out.mass = p.grav_mass;
    out.point_coord = corner * 0.5 + vec2<f32>(0.5);
    out.branch_hash = branch_hash;
    out.seed_glow = seed_glow;
    out.causal_env = causal_env;
    return out;
}

@fragment
fn fs_render_dual_channel(in: VertexOutput) -> GBufferOutput {
    var output: GBufferOutput;

    let dist_from_center = length(in.point_coord - vec2<f32>(0.5));
    if (dist_from_center > 0.5) {
        discard;
    }

    let core_intensity = exp(-dist_from_center * 8.0);
    let z = camera.params.w;

    // Channel A: Visible Gauge Emission.
    // Branch split: active baryons (branch_hash >= 23/33) burn solar-gold;
    // anti-baryons de-render electric-violet -> deep ghost across the
    // Stinespring window z in [1e9, 1e12].
    let is_dark_branch = in.branch_hash < (23.0 / 33.0);
    var emit = vec3<f32>(1.0, 0.85, 0.45) * 1.8;
    let dark_ghost_weight = (1.0 - in.vis_factor) * in.mass;

    if (is_dark_branch) {
        if (z > 1.0e12) {
            emit = vec3<f32>(0.85, 0.25, 1.0);
        } else if (z > 1.0e9) {
            let blend = clamp((12.0 - log2(max(z, 1.0)) / 3.321928) / 3.0, 0.0, 1.0);
            let quenched_violet = vec3<f32>(0.12, 0.04, 0.28) * max(dark_ghost_weight, 0.35);
            emit = mix(vec3<f32>(0.85, 0.25, 1.0), quenched_violet, blend);
        } else {
            emit = vec3<f32>(0.12, 0.04, 0.28) * dark_ghost_weight;
        }
    }
    let pulse = 0.75 + 0.25 * sin(in.branch_hash * 40.0 + z * 0.7);

    // Dark-branch emission decays quadratically with vis_weight so the
    // de-rendering population dims faster than the baryon branch.
    var emit_w = in.vis_factor;
    if (is_dark_branch) {
        emit_w = in.vis_factor * in.vis_factor;
    }
    let emission = emit * core_intensity * emit_w;
    // Ionization halo tinted by the branch color so the two populations
    // stay chromatically distinct (gold baryons vs violet anti-baryons).
    let ionization_halo = emit * vec3<f32>(0.35, 0.45, 0.7) * pow(core_intensity, 2.0) * in.vis_factor;
    var glow_rgb = emission + ionization_halo;
    var glow_a = in.vis_factor * core_intensity;

    // Ghost-seed attractor marker: glowing white/gold cores with caustic lensing rings (M_seed ~ 10^9 M_sun)
    if (in.seed_glow > 0.0) {
        // High-contrast caustic lensing rings
        let ring_dist = abs(dist_from_center - 0.40);
        let caustic_fringe = exp(-pow(ring_dist * 30.0, 2.0)) * (0.85 + 0.15 * sin(dist_from_center * 45.0 + z * 0.4));
        let seed_core = vec3<f32>(1.0, 0.98, 0.85) * exp(-dist_from_center * 12.0) * pulse * 4.5;
        let seed_ring = vec3<f32>(1.0, 0.85, 0.35) * caustic_fringe * pulse * 3.0 + vec3<f32>(0.3, 0.8, 1.0) * caustic_fringe * 1.8;
        glow_rgb += (seed_core + seed_ring) * in.seed_glow;
        glow_a = max(glow_a, in.seed_glow * (core_intensity + caustic_fringe));
    }
    output.visible_gauge_glow = vec4<f32>(glow_rgb, glow_a);

    // Channel B: Passive Gravitational Ghost Distortion
    // Persists regardless of vis_factor, tracking total stress-energy E_munu = 0
    var shear_dir = vec2<f32>(0.0);
    if (length(in.world_pos.xy) > 1.0e-4) {
        shear_dir = normalize(in.world_pos.xy);
    }
    var gravitational_shear = vec2<f32>(
        shear_dir.x * core_intensity * 0.05,
        shear_dir.y * core_intensity * 0.05
    );

    // Enhanced caustic lensing shear for condensed seed cores
    if (in.seed_glow > 0.0) {
        let seed_caustic_shear = 0.20 * in.seed_glow * exp(-dist_from_center * 6.0);
        gravitational_shear.x += shear_dir.x * seed_caustic_shear;
        gravitational_shear.y += shear_dir.y * seed_caustic_shear;
    }

    // Store shear in RG, ghost density in B, total invariant mass in A.
    // Causal-point projection envelopes land in the ghost channel as faint
    // observer light-cone shells.
    output.passive_metric_distortion = vec4<f32>(
        gravitational_shear.x + in.causal_env * 0.02,
        gravitational_shear.y + in.causal_env * 0.02,
        dark_ghost_weight * core_intensity + in.causal_env * 0.35 + in.seed_glow * 1.2 * core_intensity,
        in.mass
    );

    return output;
}
