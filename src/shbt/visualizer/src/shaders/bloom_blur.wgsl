// ============================================================================
// File: src/shbt/visualizer/src/shaders/bloom_blur.wgsl
// Bloom pre-pass (Enhancement 12): separable 9-tap Gaussian over the
// HDR Channel-A scene texture at half resolution.
//
// Pass 1 (horizontal) applies a luminance bright-pass threshold so only
// emissive cores (seed starbursts, Landauer coronae, caustic fringes)
// feed the bloom; Pass 2 (vertical) completes the Gaussian kernel. The
// post pass adds the result to the composite before ACES tonemapping.
// ============================================================================

struct BloomUniforms {
    dir: vec2<f32>,        // blur axis in texel units ((1,0) or (0,1))
    texel: vec2<f32>,      // 1 / source resolution
    threshold: f32,        // luminance bright-pass (0 on the second pass)
    _pad: vec3<f32>,
};

@group(0) @binding(0) var src_tex: texture_2d<f32>;
@group(0) @binding(1) var smp: sampler;
@group(0) @binding(2) var<uniform> u: BloomUniforms;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_bloom(@builtin(vertex_index) v_idx: u32) -> VertexOutput {
    var out: VertexOutput;
    var pos = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 3.0, -1.0),
        vec2<f32>(-1.0,  3.0)
    );
    out.position = vec4<f32>(pos[v_idx], 0.0, 1.0);
    out.uv = pos[v_idx] * 0.5 + vec2<f32>(0.5);
    out.uv.y = 1.0 - out.uv.y;
    return out;
}

fn bright_pass(c: vec3<f32>) -> vec3<f32> {
    let luma = dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
    let knee = clamp(luma - u.threshold, 0.0, luma) / max(luma, 1.0e-4);
    return c * knee;
}

@fragment
fn fs_bloom(in: VertexOutput) -> @location(0) vec4<f32> {
    // Normalized 9-tap Gaussian (sigma ~ 2.5 texels).
    var weights = array<f32, 9>(
        0.016216, 0.054054, 0.1216216, 0.1945946,
        0.227027,
        0.1945946, 0.1216216, 0.054054, 0.016216,
    );

    var accum = vec3<f32>(0.0);
    for (var i = 0; i < 9; i = i + 1) {
        let offset = f32(i - 4) * u.dir * u.texel * 2.0;
        let c = textureSampleLevel(src_tex, smp, in.uv + offset, 0.0).rgb;
        accum = accum + bright_pass(c) * weights[i];
    }
    return vec4<f32>(accum, 1.0);
}
