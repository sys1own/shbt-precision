// Full-screen holographic composite pass: gravitational lensing UV
// displacement, dark-matter filament false-coloring, and holographic
// horizon boundary overlay driven by the SHBT-MMIO telemetry frame.

@group(0) @binding(0) var tex_visible: texture_2d<f32>;
@group(0) @binding(1) var tex_distortion: texture_2d<f32>;
@group(0) @binding(2) var default_sampler: sampler;
// x: channel A enable, y: channel B enable, z: f_load (horizon fill), w: exposure
@group(0) @binding(3) var<uniform> post_params: vec4<f32>;

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
    // Sample passive metric distortion (shear vectors stored in RG)
    let distortion_sample = textureSample(tex_distortion, default_sampler, in.uv);
    let shear_offset = distortion_sample.xy * post_params.y;
    let ghost_density = distortion_sample.z * post_params.y;

    // Apply gravitational lensing displacement to visible texture sampling UV
    let lensed_uv = in.uv + shear_offset;
    let visible_sample = textureSample(tex_visible, default_sampler, lensed_uv);

    // Color dark matter ghost filaments with gravitational blue shift
    let ghost_color = vec3<f32>(0.12, 0.04, 0.28) * ghost_density * 2.5;

    // Composite visible matter, gravitational lensing, and dark ghost background
    let final_rgb = visible_sample.rgb * post_params.x + ghost_color;
    let alpha = max(visible_sample.a * post_params.x, ghost_density);

    // Holographic horizon boundary overlay: the loaded screen fraction
    // brightens toward the canvas edge as f_load -> 1 (de Sitter freeze).
    let edge = max(abs(in.uv.x - 0.5), abs(in.uv.y - 0.5)) * 2.0;
    let horizon = pow(edge, 8.0) * post_params.z;
    let horizon_rgb = vec3<f32>(0.05, 0.18, 0.35) * horizon;

    return vec4<f32>((final_rgb + horizon_rgb) * post_params.w, alpha);
}
