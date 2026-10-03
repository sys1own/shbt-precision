// ============================================================================
// File: src/shbt/visualizer/src/shaders/causal_cone_render.wgsl
// Enhancement 7: Past Light Cone Volumes (shbt6, Section 7).
//
// Each active causal point samples boundary registers within its past
// causal diamond (ds^2 = -c^2 dt^2 + dx^2 = 0). Rendered as instanced
// wireframe frustum spokes oriented along the observer's cone axis, apex
// at the observer coordinate expanding toward the boundary register.
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
    // x: box size, y: time, z: master scale, w: pad.
    params: vec4<f32>,
};

@group(0) @binding(0) var<storage, read> observers: array<CausalObserver>;
@group(0) @binding(1) var<uniform> causal_cam: CausalCam;

struct ConeOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) ray: vec3<f32>,
    @location(1) alpha: f32,
};

// Two spokes per vertex pair: even indices emit the apex, odd indices emit
// a rim point; 32 vertices per observer => 16 apex->rim lines.
@vertex
fn vs_cone_wireframe(
    @builtin(vertex_index) v_idx: u32,
    @builtin(instance_index) i_idx: u32
) -> ConeOut {
    let obs = observers[i_idx];
    let height = 0.5;
    let base_rad = height * tan(obs.cone_angle * 0.5);
    let spoke = f32(v_idx / 2u);
    let angle = spoke * (6.2831853 / 16.0);

    var local_pt = vec3<f32>(0.0);
    if ((v_idx & 1u) == 1u) {
        local_pt = vec3<f32>(cos(angle) * base_rad, sin(angle) * base_rad, -height);
        // Orient the cone along the observer's past-light-cone axis.
        local_pt = vec3<f32>(local_pt.xy, -local_pt.z);
        let axis = normalize(obs.cone_direction + vec3<f32>(0.0, 0.0, 1e-4));
        let up_ref = select(vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(1.0, 0.0, 0.0),
                         abs(axis.y) > 0.9);
        let t1 = normalize(cross(axis, up_ref));
        let t2 = cross(axis, t1);
        local_pt = t1 * local_pt.x + t2 * local_pt.y + axis * local_pt.z;
    }

    let world = (obs.position - vec3<f32>(0.5)) * causal_cam.params.x + local_pt * causal_cam.params.x * 0.1;

    var out: ConeOut;
    out.clip_pos = causal_cam.view_proj * vec4<f32>(world, 1.0);
    out.ray = local_pt;
    out.alpha = clamp(obs.entropy_budget / 1024.0, 0.05, 0.65) * f32(obs.active_get_flag);
    return out;
}

struct ConeFragOut {
    @location(0) channel_a: vec4<f32>,
    @location(1) channel_b: vec4<f32>,
};

@fragment
fn fs_cone_wireframe(in: ConeOut) -> ConeFragOut {
    var out: ConeFragOut;
    let wire_tint = vec3<f32>(0.05, 0.72, 0.98);
    out.channel_a = vec4<f32>(wire_tint, in.alpha * 0.4) * causal_cam.params.z;
    out.channel_b = vec4<f32>(0.0);
    return out;
}
