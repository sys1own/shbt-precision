// Full-screen holographic composite pass: gravitational lensing UV
// displacement, dark-matter filament false-coloring, and holographic
// horizon boundary overlay driven by the SHBT-MMIO telemetry frame.

@group(0) @binding(0) var tex_visible: texture_2d<f32>;
@group(0) @binding(1) var tex_distortion: texture_2d<f32>;
@group(0) @binding(2) var default_sampler: sampler;
// params0: x: channel A enable, y: channel B enable, z: f_load (horizon
// fill), w: exposure.  params1.x: unwrap_transition (0 = comoving bulk,
// 1 = flat boundary CFT torus [0, 2pi)^2).
struct PostParams {
    params0: vec4<f32>,
    params1: vec4<f32>,
};
@group(0) @binding(3) var<uniform> post_params: PostParams;

// Conformal boundary unwrapping: roll the 3D comoving bulk onto the
// 2D CFT torus [0, 2pi)^2 as unwrap_transition goes 0 -> 1.
fn unwrap_torus_projection(uv: vec2<f32>) -> vec2<f32> {
    let pi2 = 6.28318530718;
    let theta1 = uv.x * pi2;
    let theta2 = uv.y * pi2;
    let torus = vec2<f32>((cos(theta1) + 1.0) * 0.5, (sin(theta2) + 1.0) * 0.5);
    return mix(uv, torus, post_params.params1.x);
}

struct ScreenQuadOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_fullscreen(@builtin(vertex_index) vertex_index: u32) -> ScreenQuadOutput {
    var out: ScreenQuadOutput;
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0),
    );
    let pos = positions[vertex_index];
    out.position = vec4<f32>(pos, 0.0, 1.0);
    out.uv = pos * 0.5 + vec2<f32>(0.5);
    out.uv.y = 1.0 - out.uv.y;
    return out;
}

@fragment
fn fs_composite_holography(in: ScreenQuadOutput) -> @location(0) vec4<f32> {
    let warped_uv = unwrap_torus_projection(in.uv);

    // Sample passive metric distortion (shear vectors stored in RG)
    let distortion_sample = textureSample(tex_distortion, default_sampler, warped_uv);
    let shear_offset = distortion_sample.xy * post_params.params0.y;
    let ghost_density = distortion_sample.z * post_params.params0.y;

    // Apply gravitational lensing displacement to visible texture sampling UV
    let lensed_uv = warped_uv + shear_offset;
    let visible_sample = textureSample(tex_visible, default_sampler, lensed_uv);

    // Color dark matter ghost filaments with gravitational blue shift
    let ghost_color = vec3<f32>(0.12, 0.04, 0.28) * ghost_density * 5.0;

    // Composite visible matter, gravitational lensing, and dark ghost background
    let final_rgb = visible_sample.rgb * post_params.params0.x + ghost_color;
    let alpha = max(visible_sample.a * post_params.params0.x, ghost_density);

    // Holographic horizon boundary overlay: the loaded screen fraction
    // brightens toward the canvas edge as f_load -> 1 (de Sitter freeze).
    let edge = max(abs(in.uv.x - 0.5), abs(in.uv.y - 0.5)) * 2.0;
    let horizon = pow(edge, 8.0) * post_params.params0.z;
    let horizon_rgb = vec3<f32>(0.05, 0.18, 0.35) * horizon;

    // Boundary screen grid: CFT torus lattice overlay fades in with the
    // unwrap transition.
    let grid_lines = abs(sin(warped_uv.x * 62.8318)) * abs(sin(warped_uv.y * 62.8318));
    let grid_rgb = vec3<f32>(0.0, 0.9, 0.7) * pow(grid_lines, 16.0) * post_params.params1.x;

    return vec4<f32>((final_rgb + horizon_rgb + grid_rgb) * post_params.params0.w, alpha);
}
