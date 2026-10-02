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
    // x: point extent (clip units), y: box_size, z: projection_mode
    // (0 = comoving bulk 3D, 1 = 2D boundary CFT plane), w: reserved.
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
    // 2D boundary projection: collapse the depth coordinate onto the screen.
    if (camera.params.z > 0.5) {
        world.z = 0.0;
    }

    var corners = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0),
        vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, 1.0), vec2<f32>(-1.0, 1.0),
    );
    let corner = corners[vertex_index];

    var clip = camera.view_proj * vec4<f32>(world, 1.0);
    clip.x += corner.x * camera.params.x * clip.w;
    clip.y += corner.y * camera.params.x * clip.w;

    out.clip_position = clip;
    out.world_pos = world;
    out.vis_factor = p.vis_weight;
    out.mass = p.grav_mass;
    out.point_coord = corner * 0.5 + vec2<f32>(0.5);
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

    // Channel A: Visible Gauge Emission
    // Decoupled anti-baryons fade to zero as vis_factor -> 0
    let baryonic_color = vec3<f32>(1.0, 0.65, 0.3) * core_intensity * in.vis_factor;
    let ionization_halo = vec3<f32>(0.2, 0.5, 1.0) * pow(core_intensity, 2.0) * in.vis_factor;
    output.visible_gauge_glow = vec4<f32>(baryonic_color + ionization_halo, in.vis_factor * core_intensity);

    // Channel B: Passive Gravitational Ghost Distortion
    // Persists regardless of vis_factor, tracking total stress-energy E_munu = 0
    let dark_ghost_weight = (1.0 - in.vis_factor) * in.mass;
    var shear_dir = vec2<f32>(0.0);
    if (length(in.world_pos.xy) > 1.0e-4) {
        shear_dir = normalize(in.world_pos.xy);
    }
    let gravitational_shear = vec2<f32>(
        shear_dir.x * core_intensity * 0.05,
        shear_dir.y * core_intensity * 0.05
    );

    // Store shear in RG, ghost density in B, total invariant mass in A
    output.passive_metric_distortion = vec4<f32>(
        gravitational_shear.x,
        gravitational_shear.y,
        dark_ghost_weight * core_intensity,
        in.mass
    );

    return output;
}
