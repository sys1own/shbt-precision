// ============================================================================
// File: src/shbt/visualizer/src/shaders/causal_sphere_render.wgsl
// Enhancement 8: Entropy Budget Spheres (shbt6, Eq. 191).
//
// An instanced billboard shell encloses each causal observer, scaled by the
// remaining entropy budget:
//   R_sphere = R_0 * (R_entropy / N_limit)^(1/3)
// The surface evaluates a Fresnel limb term and transitions emerald ->
// crimson as R_entropy -> 0 (observer freeze).
// ============================================================================

struct CausalObserver {
    position: vec3<f32>,
    radius: f32,
    entropy_budget: f32,
    active_get_flag: u32,
    cone_direction: vec3<f32>,
    cone_angle: f32,
};

struct CausalCam {
    view_proj: mat4x4<f32>,
    // x: box size, y: N_limit normalization, z: master scale, w: time.
    params: vec4<f32>,
};

@group(0) @binding(0) var<storage, read> observers: array<CausalObserver>;
@group(0) @binding(1) var<uniform> causal_cam: CausalCam;

struct SphereOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) quad_uv: vec2<f32>,
    @location(1) world_pos: vec3<f32>,
    @location(2) center: vec3<f32>,
    @location(3) r_entropy: f32,
};

@vertex
fn vs_entropy_sphere(
    @builtin(vertex_index) v_idx: u32,
    @builtin(instance_index) i_idx: u32
) -> SphereOut {
    let obs = observers[i_idx];
    var corners = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0),
        vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, 1.0), vec2<f32>(-1.0, 1.0),
    );
    let corner = corners[v_idx];

    let center_world = (obs.position - vec3<f32>(0.5)) * causal_cam.params.x;
    let ratio = clamp(obs.entropy_budget / max(causal_cam.params.y, 1.0), 0.0, 1.0);
    let r_sphere = obs.radius * pow(max(ratio, 1.0e-3), 1.0 / 3.0);

    var clip = causal_cam.view_proj * vec4<f32>(center_world, 1.0);
    // r_sphere is a normalized box fraction; keep the billboard a modest
    // screen fraction (~6% of the frame). Multiplying by params.x (box
    // size in world units) produced quads ~3x the screen width, and 64
    // additive shells fused into a frame-wide wash.
    let extent = r_sphere * 0.25 * f32(obs.active_get_flag);
    clip.x += corner.x * extent * clip.w;
    clip.y += corner.y * extent * clip.w;

    var out: SphereOut;
    out.clip_pos = clip;
    out.quad_uv = corner;
    out.world_pos = center_world + vec3<f32>(corner * extent, 0.0);
    out.center = center_world;
    out.r_entropy = obs.entropy_budget;
    return out;
}

struct SphereFragOut {
    @location(0) channel_a: vec4<f32>,
    @location(1) channel_b: vec4<f32>,
};

@fragment
fn fs_entropy_sphere(in: SphereOut) -> SphereFragOut {
    var out: SphereFragOut;
    let r2 = dot(in.quad_uv, in.quad_uv);
    if (r2 > 1.0) {
        discard;
    }
    // Fresnel limb term on the billboard shell.
    let fresnel = pow(1.0 - sqrt(max(r2, 0.0)), 3.0);

    let ratio = clamp(in.r_entropy / max(causal_cam.params.y, 1.0), 0.0, 1.0);
    let healthy_green = vec3<f32>(0.05, 0.95, 0.45);
    let depleted_red = vec3<f32>(0.95, 0.08, 0.08);
    let col = mix(depleted_red, healthy_green, ratio);

    let alpha = (fresnel * 0.5 + 0.06) * ratio;
    out.channel_a = vec4<f32>(col, alpha) * causal_cam.params.z;
    out.channel_b = vec4<f32>(0.0);
    return out;
}
