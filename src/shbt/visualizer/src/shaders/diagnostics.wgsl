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

var<workgroup> wg_ps_bins: array<atomic<u32>, 1024>;

@compute @workgroup_size(256)
fn project_phase_space(@builtin(global_invocation_id) global_id: vec3<u32>, @builtin(local_invocation_id) local_id: vec3<u32>) {
    // Clear workgroup histogram tiles (1024 bins = 32x32 sub-grid)
    if (local_id.x < 256u) {
        atomicStore(&wg_ps_bins[local_id.x], 0u);
        atomicStore(&wg_ps_bins[local_id.x + 256u], 0u);
        atomicStore(&wg_ps_bins[local_id.x + 512u], 0u);
        atomicStore(&wg_ps_bins[local_id.x + 768u], 0u);
    }
    workgroupBarrier();

    let idx = global_id.x;
    if (idx < uniforms.particle_count) {
        let p = particles[idx];
        // Comoving periodic wrapping x_norm in [0, 1)
        let x_norm = fract(p.position.x);
        let bin_x = clamp(u32(x_norm * 256.0), 0u, 255u);

        // Conformal velocity mapping centered at v_norm = 0.5
        let v_x = p.velocity.x;
        let v_norm = (v_x / (2.0 * max(uniforms.v_max, 1e-4))) + 0.5;
        let clamped_v = clamp(v_norm, 0.0, 0.99999);
        let bin_v = clamp(u32(clamped_v * 256.0), 0u, 255u);
        let tex_coords = vec2<i32>(i32(bin_x), i32(bin_v));

            // Accumulate locally to estimate density
            let sub_x = bin_x % 32u;
            let sub_v = bin_v % 32u;
            let sub_idx = sub_v * 32u + sub_x;
            let local_count = f32(atomicAdd(&wg_ps_bins[sub_idx], 1u) + 1u);

            // Logarithmic dynamic range tone mapping: captures single-stream sheets and dense cores
            // I(u, v) = ln(1 + kappa * N) / ln(1 + kappa * N_sat) with kappa = 10.0, N_sat = 500.0
            let kappa = 10.0;
            let log_intensity = log(1.0 + kappa * local_count) / log(1.0 + kappa * 500.0);
            let intensity = clamp(log_intensity, 0.0, 1.0);

            // Scientific false-color colormap: Deep Void Blue -> Sheet Cyan -> Amber -> Caustic White
            let c1 = vec3<f32>(0.05, 0.15, 0.45);
            let c2 = vec3<f32>(0.10, 0.80, 0.75);
            let c3 = vec3<f32>(0.95, 0.60, 0.10);
            let c4 = vec3<f32>(1.00, 1.00, 1.00);

            var color = vec3<f32>(0.0);
            if (intensity < 0.33) {
                color = mix(c1, c2, intensity / 0.33);
            } else if (intensity < 0.66) {
                color = mix(c2, c3, (intensity - 0.33) / 0.33);
            } else {
                color = mix(c3, c4, (intensity - 0.66) / 0.34);
            }

            textureStore(phase_space_tex, tex_coords, vec4<f32>(color, 1.0));
    }
}

