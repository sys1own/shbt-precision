// causal_lifecycle.wgsl
// 5-Phase Causal-Point Lifecycle and Stinespring Succession Shader.

struct CausalPointRecord {
    world_pos: vec3<f32>,
    lifecycle_phase: u32, // 0=Seeding, 1=Accretion, 2=Crystallizing, 3=Saturated, 4=Succession
    horizon_radius: f32,
    entropy_admissibility: f32,
    get_acceleration: vec2<f32>,
    projector_state: vec4<f32>,
    succession_arc: vec4<f32>,
};

struct LifecycleParams {
    dt_sim: f32,
    kappa_get: f32,
    entropy_rate: f32,
    _pad: f32,
};

@group(0) @binding(0) var<uniform> params: LifecycleParams;
@group(0) @binding(1) var<storage, read_write> causal_points: array<CausalPointRecord>;
@group(0) @binding(2) var<storage, read> grad_ln_rho: array<vec4<f32>>;

@compute @workgroup_size(64, 1, 1)
fn cs_update_causal_lifecycle(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&causal_points)) {
        return;
    }

    var cp = causal_points[idx];
    let dt_sim = params.dt_sim;

    switch (cp.lifecycle_phase) {
        case 0u: { // Horizon Seeding
            cp.horizon_radius += 0.1 * dt_sim;
            if (cp.horizon_radius >= 1.0) {
                cp.lifecycle_phase = 1u;
            }
        }
        case 1u: { // Past Light-Cone Accretion
            cp.horizon_radius += 0.5 * dt_sim;
            let norm_proj = dot(cp.projector_state, cp.projector_state);
            if (norm_proj >= 0.999) {
                cp.lifecycle_phase = 2u;
            }
        }
        case 2u: { // Projector Crystallization
            let grad = grad_ln_rho[idx].xy;
            cp.get_acceleration = -params.kappa_get * grad;
            cp.entropy_admissibility -= params.entropy_rate * dt_sim;
            if (cp.entropy_admissibility <= 0.001) {
                cp.entropy_admissibility = 0.0;
                cp.lifecycle_phase = 3u;
            }
        }
        case 3u: { // Admissibility Saturation & De-rendering
            cp.projector_state *= exp(-5.0 * dt_sim);
            if (cp.projector_state.x < 0.01) {
                cp.lifecycle_phase = 4u;
            }
        }
        case 4u: { // Stinespring Succession
            let t = cp.succession_arc.x + dt_sim / max(cp.succession_arc.y, 1e-4);
            cp.succession_arc.x = t;
            if (t >= 1.0) {
                cp.lifecycle_phase = 0u;
                cp.horizon_radius = 0.01;
                cp.entropy_admissibility = 1.0;
                cp.projector_state = vec4<f32>(1.0, 0.0, 0.0, 0.0);
            }
        }
        default: {}
    }

    causal_points[idx] = cp;
}

struct CausalVertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) uv: vec2<f32>,
};

struct RenderUniforms {
    view_proj: mat4x4<f32>,
};

@group(1) @binding(0) var<uniform> render_uniforms: RenderUniforms;

@vertex
fn vs_render_causal_point(
    @builtin(vertex_index) v_idx: u32,
    @builtin(instance_index) i_idx: u32
) -> CausalVertexOutput {
    var out: CausalVertexOutput;
    let cp = causal_points[i_idx];

    let quad = array<vec2<f32>, 4>(
        vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0),
        vec2<f32>(-1.0,  1.0), vec2<f32>(1.0,  1.0)
    );
    let offset = quad[v_idx] * cp.horizon_radius;
    out.uv = quad[v_idx];

    let world_pos = vec4<f32>(cp.world_pos + vec3<f32>(offset, 0.0), 1.0);
    out.position = render_uniforms.view_proj * world_pos;

    switch (cp.lifecycle_phase) {
        case 0u: { out.color = vec4<f32>(0.2, 0.8, 0.2, 0.5); } // Seeding: Emerald
        case 1u: { out.color = vec4<f32>(0.2, 0.4, 1.0, 0.7); } // Accretion: Azure
        case 2u: { out.color = vec4<f32>(1.0, 0.84, 0.0, 1.0); } // Crystallization: Gold
        case 3u: { out.color = vec4<f32>(0.8, 0.0, 0.0, cp.entropy_admissibility); } // Saturation: Crimson
        case 4u: { out.color = vec4<f32>(0.7, 0.2, 0.9, 0.8); } // Succession Arc: Violet
        default: { out.color = vec4<f32>(1.0); }
    }
    return out;
}

@fragment
fn fs_render_causal_point(in: CausalVertexOutput) -> @location(0) vec4<f32> {
    let r2 = dot(in.uv, in.uv);
    if (r2 > 1.0) {
        discard;
    }
    let ring = abs(sqrt(r2) - 0.85);
    let alpha = select(in.color.a * (1.0 - r2), 1.0, ring < 0.1);
    return vec4<f32>(in.color.rgb, alpha);
}
