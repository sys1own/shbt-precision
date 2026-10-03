// ============================================================================
// File: src/shbt/visualizer/src/shaders/tracer_render.wgsl
// Enhancement 3 (render half): Entropy Gradient Streamlines.
//
// Renders the 10,000 tracer particles advected by
// entropy_tracer.wgsl/cs_integrate_streamlines as additive point sprites
// into Channel A. The tracer buffer is re-declared read-only here because
// WebGPU forbids read_write storage in the vertex stage.
// ============================================================================

struct Tracer {
    pos: vec3<f32>,
    lifetime: f32,
    vel: vec3<f32>,
    entropy_val: f32,
};

struct TracerCamera {
    view_proj: mat4x4<f32>,
    // x: point half-extent (clip), y: box_size, z: time, w: intensity.
    params: vec4<f32>,
};

@group(0) @binding(0) var<storage, read> tracers: array<Tracer>;
@group(1) @binding(0) var<uniform> tracer_cam: TracerCamera;

struct TracerOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) quad_uv: vec2<f32>,
    @location(1) entropy: f32,
};

@vertex
fn vs_tracer(
    @builtin(vertex_index) v_idx: u32,
    @builtin(instance_index) t_idx: u32,
) -> TracerOut {
    var out: TracerOut;
    let t = tracers[t_idx];

    var corners = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0),
        vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, 1.0), vec2<f32>(-1.0, 1.0),
    );
    let corner = corners[v_idx];

    // Tracers live in the centered unit cube [-1, 1]; scale to the world
    // box before projecting.
    let world = t.pos * (tracer_cam.params.y * 0.5);
    var clip = tracer_cam.view_proj * vec4<f32>(world, 1.0);
    clip.x += corner.x * tracer_cam.params.x * clip.w;
    clip.y += corner.y * tracer_cam.params.x * clip.w;

    out.clip_pos = clip;
    out.quad_uv = corner;
    out.entropy = t.entropy_val;
    return out;
}

struct TracerFragOut {
    @location(0) channel_a: vec4<f32>,
    @location(1) channel_b: vec4<f32>,
};

@fragment
fn fs_tracer(in: TracerOut) -> TracerFragOut {
    var out: TracerFragOut;
    let r2 = dot(in.quad_uv, in.quad_uv);
    if (r2 > 1.0) {
        discard;
    }
    let streak = (1.0 - r2) * clamp(in.entropy * 4.0 + 0.15, 0.0, 1.0);
    // Teal-amber entropic streamline tint.
    let col = mix(vec3<f32>(0.05, 0.55, 0.85), vec3<f32>(0.95, 0.55, 0.15),
                  clamp(in.entropy, 0.0, 1.0));
    out.channel_a = vec4<f32>(col * streak * 0.35 * tracer_cam.params.w, streak * 0.25);
    out.channel_b = vec4<f32>(0.0);
    return out;
}
