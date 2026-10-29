// spectral_metrics.wgsl
// GPU reduction for real-time matter power spectrum, CIC deconvolution, and shot-noise subtraction.

struct SpectrumUniforms {
    box_size_mpc: f32,
    mesh_dim: u32,
    particle_count: f32,
    k_min: f32,
    k_max: f32,
    num_bins: u32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<uniform> spec_u: SpectrumUniforms;
@group(0) @binding(1) var delta_k_real: texture_3d<f32>;
@group(0) @binding(2) var delta_k_imag: texture_3d<f32>;
@group(0) @binding(3) var<storage, read_write> k_shell_power_accum: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> k_shell_modes_accum: array<atomic<u32>>;

const PI: f32 = 3.14159265358979323846;
const FIXED_POINT_SCALE: f32 = 1.0e6;

fn sinc_filter(q: f32) -> f32 {
    if (abs(q) < 1e-5) {
        return 1.0;
    }
    let s = sin(q) / q;
    return s * s;
}

@compute @workgroup_size(8, 8, 8)
fn cs_accumulate_power_spectrum(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let n = spec_u.mesh_dim;
    if (global_id.x >= n || global_id.y >= n || global_id.z >= n) {
        return;
    }

    let k_coords = vec3<f32>(
        select(f32(global_id.x), f32(global_id.x) - f32(n), global_id.x >= (n / 2u)),
        select(f32(global_id.y), f32(global_id.y) - f32(n), global_id.y >= (n / 2u)),
        select(f32(global_id.z), f32(global_id.z) - f32(n), global_id.z >= (n / 2u))
    );

    let k_mag = length(k_coords) * (2.0 * PI / spec_u.box_size_mpc);
    if (k_mag < spec_u.k_min || k_mag >= spec_u.k_max) {
        return;
    }

    let wx = sinc_filter(0.5 * k_coords.x * (2.0 * PI / f32(n)));
    let wy = sinc_filter(0.5 * k_coords.y * (2.0 * PI / f32(n)));
    let wz = sinc_filter(0.5 * k_coords.z * (2.0 * PI / f32(n)));
    let w_cic2 = pow(wx * wy * wz, 2.0);

    if (w_cic2 < 1e-7) {
        return;
    }

    let coord = vec3<i32>(global_id);
    let re = textureLoad(delta_k_real, coord, 0).r;
    let im = textureLoad(delta_k_imag, coord, 0).r;
    let power_raw = re * re + im * im;

    let p_deconv = (power_raw / w_cic2) - (1.0 / spec_u.particle_count);

    let log_ratio = log(k_mag / spec_u.k_min) / log(spec_u.k_max / spec_u.k_min);
    let bin_idx = clamp(u32(log_ratio * f32(spec_u.num_bins)), 0u, spec_u.num_bins - 1u);

    let fixed_val = u32(max(0.0, p_deconv) * FIXED_POINT_SCALE);
    atomicAdd(&k_shell_power_accum[bin_idx], fixed_val);
    atomicAdd(&k_shell_modes_accum[bin_idx], 1u);
}

// 2D Thermodynamic Phase Space Accumulation (64x64 bins)
struct PhaseSpaceParams {
    min_log_rho: f32,
    max_log_rho: f32,
    min_log_t: f32,
    max_log_t: f32,
};

@group(1) @binding(0) var<uniform> ps_u: PhaseSpaceParams;
@group(1) @binding(1) var<storage, read_write> phase_space_hist: array<atomic<u32>>;

@compute @workgroup_size(8, 8, 8)
fn cs_accumulate_phase_space(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let coord = vec3<i32>(global_id);
    let rho = textureLoad(delta_k_real, coord, 0).r + 1.0;
    if (rho <= 0.0) { return; }
    let log_rho = log(rho) * 0.4342944819;

    let t_val = max(100.0, textureLoad(delta_k_imag, coord, 0).r);
    let log_t = log(t_val) * 0.4342944819;

    if (log_rho < ps_u.min_log_rho || log_rho >= ps_u.max_log_rho ||
        log_t < ps_u.min_log_t || log_t >= ps_u.max_log_t) {
        return;
    }

    let bin_x = u32((log_rho - ps_u.min_log_rho) / (ps_u.max_log_rho - ps_u.min_log_rho) * 64.0);
    let bin_y = u32((log_t - ps_u.min_log_t) / (ps_u.max_log_t - ps_u.min_log_t) * 64.0);

    let flat_idx = clamp(bin_y, 0u, 63u) * 64u + clamp(bin_x, 0u, 63u);
    atomicAdd(&phase_space_hist[flat_idx], 1u);
}

// Stacked Void Profiling
struct VoidStackParams {
    void_center: vec3<f32>,
    r_void: f32,
    num_radial_bins: u32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
};

@group(2) @binding(0) var<uniform> void_u: VoidStackParams;
@group(2) @binding(1) var<storage, read> particle_positions: array<vec4<f32>>;
@group(2) @binding(2) var<storage, read_write> void_density_profile: array<atomic<u32>>;

@compute @workgroup_size(256, 1, 1)
fn cs_stacked_void_profile(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&particle_positions)) {
        return;
    }

    let dist = length(particle_positions[idx].xyz - void_u.void_center);
    let normalized_r = dist / max(void_u.r_void, 1e-4);

    if (normalized_r >= 3.0) {
        return;
    }

    let bin_idx = u32(normalized_r / 3.0 * f32(void_u.num_radial_bins));
    if (bin_idx < void_u.num_radial_bins) {
        atomicAdd(&void_density_profile[bin_idx], 1u);
    }
}
