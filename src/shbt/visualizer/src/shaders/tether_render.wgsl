// ============================================================================
// File: src/shbt/visualizer/src/shaders/tether_render.wgsl
// Enhancement 4: Stinespring Transition Tethers (shbt6, Eq. 164).
//
// Line-list pass over the TetherVertexBuffer emitted by nbody_pm.wgsl when
// anti-baryons quench their gauge charge and transition to Channel B.
// Vertex alpha decays exponentially toward the dark-sector endpoint.
// ============================================================================

struct Camera {
    view_proj: mat4x4<f32>,
    params: vec4<f32>,
    aux: vec4<f32>,
};

@group(0) @binding(0) var<uniform> cam: Camera;
@group(0) @binding(1) var<storage, read> tether_verts: array<TetherVertex>;

struct TetherVertex {
    pos: vec3<f32>,
    alpha_decay: f32,
    color_tint: vec4<f32>,
};

struct TetherOutput {
    @location(0) channel_a: vec4<f32>,
    @location(1) channel_b: vec4<f32>,
};

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) color: vec4<f32>,
};

@vertex
fn vs_tether(@builtin(vertex_index) v_idx: u32) -> VertexOutput {
    var out: VertexOutput;
    let v = tether_verts[v_idx];
    // Tether endpoints are stored in comoving box coordinates [0, box];
    // recenter into world space like the particle pass.
    let world = v.pos - vec3<f32>(cam.params.y * 0.5);
    out.clip_pos = cam.view_proj * vec4<f32>(world, 1.0);
    out.color = vec4<f32>(v.color_tint.rgb, v.color_tint.a * v.alpha_decay);
    return out;
}

@fragment
fn fs_tether(in: VertexOutput) -> TetherOutput {
    var out: TetherOutput;
    out.channel_a = in.color;
    out.channel_b = vec4<f32>(0.0);
    return out;
}
