// ============================================================================
// File: src/shbt/visualizer/src/shaders/diagnostics.wgsl
// Module: sys1own/shbt-precision Real-Time Diagnostic Estimators (shbt9)
//
// GPU-side diagnostic observables that eliminate CPU telemetry stalls:
//   Pass 1 (clear_diagnostics): zero the 64-bin power spectrum accumulator.
//   Pass 2 (reduce_power_spectrum): workgroup-parallel reduction sorting
//     particle density modes into 64 logarithmic wavenumber shells
//     k in [k_min, k_max]; each thread's contribution is binned into
//     shared workgroup atomics and flushed once per workgroup.
//   Pass 3 (project_phase_space): rasterizes the (x, v_x) phase-space
//     density onto a 256x256 RGBA8 storage texture for the split-viewport
//     diagnostic pane (holographic_post.wgsl split_viewport_mode == 1).
// ============================================================================

struct DiagUniforms {
    particle_count: u32,
    k_min: f32,
    k_max: f32,
    v_max: f32,
};

struct Particle {
    position: vec3<f32>,       // comoving Mpc
    channel: u32,              // 0 = Channel A baryon, 1 = Channel B ghost
    velocity: vec3<f32>,       // supercomoving momentum
    charge_flags: u32,
    shear: vec4<f32>,
    landauer_debt: f32,
    grav_mass: f32,
    pad: vec2<f32>,
};

@group(0) @binding(0) var<uniform> uniforms: DiagUniforms;
@group(0) @binding(1) var<storage, read> particles: array<Particle>;
@group(0) @binding(2) var<storage, read_write> power_spectrum: array<atomic<u32>>;
@group(0) @binding(3) var phase_space_tex: texture_storage_2d<rgba8unorm, write>;

var<workgroup> wg_bins: array<atomic<u32>, 64>;
const DIAG_FIXED_POINT: f32 = 4096.0;

@compute @workgroup_size(64)
fn clear_diagnostics(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (global_id.x < 64u) {
        atomicStore(&power_spectrum[global_id.x], 0u);
    }
}

@compute @workgroup_size(256)
fn reduce_power_spectrum(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    if (local_id.x < 64u) {
        atomicStore(&wg_bins[local_id.x], 0u);
    }
    workgroupBarrier();

    let idx = global_id.x;
    if (idx < uniforms.particle_count) {
        let p = particles[idx];
        let offset = p.position - vec3<f32>(0.5);
        let k_mag = length(offset) * 6.283185307;

        if (k_mag >= uniforms.k_min && k_mag < uniforms.k_max) {
            let log_ratio = log(k_mag / uniforms.k_min) /
                log(uniforms.k_max / uniforms.k_min);
            let bin = clamp(u32(floor(log_ratio * 64.0)), 0u, 63u);
            let power_val = u32(p.grav_mass * p.grav_mass * DIAG_FIXED_POINT);
            atomicAdd(&wg_bins[bin], power_val);
        }
    }
    workgroupBarrier();

    if (local_id.x < 64u) {
        let local_acc = atomicLoad(&wg_bins[local_id.x]);
        if (local_acc > 0u) {
            atomicAdd(&power_spectrum[local_id.x], local_acc);
        }
    }
}

@compute @workgroup_size(256)
fn project_phase_space(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= uniforms.particle_count) {
        return;
    }

    let p = particles[idx];
    // x axis: box-coordinate x normalized to [0,1); v_x axis: momentum
    // mapped symmetric about 0 by +/-v_max.
    let coord_x = clamp(fract(p.position.x * 0.005), 0.0, 0.999);
    let norm_vx = clamp(
        (p.velocity.x / (2.0 * uniforms.v_max)) + 0.5,
        0.0, 0.999,
    );

    let tex_coords = vec2<i32>(
        i32(floor(coord_x * 256.0)),
        i32(floor(norm_vx * 256.0)),
    );

    // Channel A (baryons) in amber, Channel B (ghosts) in electric blue.
    let is_visible = p.channel == 0u;
    let color = select(
        vec4<f32>(0.05, 0.25, 0.80, 0.35),
        vec4<f32>(0.95, 0.55, 0.10, 0.60),
        is_visible
    );

    textureStore(phase_space_tex, tex_coords, color);
}
