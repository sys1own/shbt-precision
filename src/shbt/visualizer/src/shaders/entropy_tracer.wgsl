// ============================================================================
// File: src/shbt/visualizer/src/shaders/entropy_tracer.wgsl
// Enhancement 3: Entropy Gradient Streamlines (shbt6, Section 5).
//
// Advects 10,000 persistent tracer particles along the entropic transport
// field a_GET(x) = -kappa_GET * grad ln rho_proj(x) = -kappa_GET *
// grad rho_E(x) / rho_E(x), sampling central differences of the 3D grid
// potential. Tracers that exit the unit bounds or exceed their lifetime
// reset to a deterministic hashed boundary position.
//
// Stage 2 renders the tracers as additive point sprites so the entropic
// streamline topology is visible in Channel A.
// ============================================================================

struct Tracer {
    pos: vec3<f32>,
    lifetime: f32,
    vel: vec3<f32>,
    entropy_val: f32,
};

struct TracerParams {
    dt: f32,
    max_lifetime: f32,
    grid_dim: u32,
    step_scale: f32,
};

struct TracerCamera {
    view_proj: mat4x4<f32>,
    // x: point half-extent (clip), y: box_size, z: time, w: intensity.
    params: vec4<f32>,
};

@group(0) @binding(0) var<storage, read_write> tracers: array<Tracer>;
@group(0) @binding(1) var entropy_field_3d: texture_3d<f32>;
@group(0) @binding(2) var trilinear_sampler: sampler;
@group(0) @binding(3) var<uniform> params: TracerParams;

fn sample_entropy_gradient(pos: vec3<f32>) -> vec3<f32> {
    let uvw = pos * 0.5 + 0.5;
    let step = 1.0 / f32(params.grid_dim);

    let dx = textureSampleLevel(entropy_field_3d, trilinear_sampler, uvw + vec3<f32>(step, 0.0, 0.0), 0.0).r
           - textureSampleLevel(entropy_field_3d, trilinear_sampler, uvw - vec3<f32>(step, 0.0, 0.0), 0.0).r;
    let dy = textureSampleLevel(entropy_field_3d, trilinear_sampler, uvw + vec3<f32>(0.0, step, 0.0), 0.0).r
           - textureSampleLevel(entropy_field_3d, trilinear_sampler, uvw - vec3<f32>(0.0, step, 0.0), 0.0).r;
    let dz = textureSampleLevel(entropy_field_3d, trilinear_sampler, uvw + vec3<f32>(0.0, 0.0, step), 0.0).r
           - textureSampleLevel(entropy_field_3d, trilinear_sampler, uvw - vec3<f32>(0.0, 0.0, step), 0.0).r;

    return vec3<f32>(dx, dy, dz) / (2.0 * step);
}

@compute @workgroup_size(64, 1, 1)
fn cs_integrate_streamlines(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= 10000u) {
        return;
    }

    var t = tracers[idx];
    t.lifetime += params.dt;

    if (t.lifetime > params.max_lifetime || any(abs(t.pos) >= vec3<f32>(1.0))) {
        let seed = f32(idx) * 1.61803398875;
        t.pos = vec3<f32>(sin(seed * 2.1), cos(seed * 3.7), sin(seed * 5.3)) * 0.9;
        t.lifetime = 0.0;
        t.vel = vec3<f32>(0.0);
    } else {
        let grad = sample_entropy_gradient(t.pos);
        let dir = grad / (length(grad) + 1e-5);
        t.pos += dir * (params.step_scale * params.dt);
        t.vel = dir;
        let uvw = t.pos * 0.5 + 0.5;
        t.entropy_val = textureSampleLevel(entropy_field_3d, trilinear_sampler, uvw, 0.0).r;
    }
    tracers[idx] = t;
}

// ---------------------------------------------------------------------------
// Render stage: additive entropic-streamline sprites in Channel A.
// ---------------------------------------------------------------------------

// The render stage lives in tracer_render.wgsl: a vertex shader cannot
// bind this buffer read_write, so it re-declares `tracers` read-only.

