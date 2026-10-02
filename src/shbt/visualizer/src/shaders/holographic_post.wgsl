// Full-screen holographic composite pass (shbt5 gravitational optics spec).
//
// Dual-scale gravitational lensing:
//   1. Macro-deflection from the Channel B convergence field via central
//      differences:  alpha_macro(u) = lambda_lens * (grad kappa + Gamma * grad kappa)
//   2. Micro-deflection from analytical softened point-mass seed defects:
//      alpha_seed(u) = sum_s theta_E,s^2 (u - u_s) / (|u - u_s|^2 + eps_core^2)
// A 3-tap chromatic dispersion splits the deflected sample into R/G/B
// coordinates scaled by delta_disp, producing spectral caustic fringes
// along the critical curves. A depth-aware bilateral blur over kappa
// produces the volumetric dark-matter halo glow, and sharp caustic rings
// accent the dominant Einstein radii.
//
// Channel A (visible_gauge_glow):   RGB spectral radiance, A = normalized depth.
// Channel B (passive_metric_distortion): RG = shear (gamma_1, gamma_2),
//   B = convergence kappa, A = observer causal entropy.

struct LensingUniforms {
    view_proj: mat4x4<f32>,       // offset   0
    inv_view_proj: mat4x4<f32>,   // offset  64
    cam_pos: vec4<f32>,           // offset 128
    screen_size: vec2<f32>,       // offset 144
    lensing_strength: f32,        // offset 152  (lambda_lens, 0.0 .. 5.0)
    dispersion_coeff: f32,        // offset 156  (delta_disp, 0.0 .. 1.0)
    dark_glow_intensity: f32,     // offset 160  (0.0 .. 2.0)
    dark_glow_radius: f32,        // offset 164  (bilateral radius, px)
    doppler_enabled: u32,         // offset 168
    seed_count: u32,              // offset 172  (S <= 64)
    time: f32,                    // offset 176
    _pad0: f32,                   // offset 180
    _pad1: vec2<f32>,             // offset 184  (192 bytes total)
    // Engine extras beyond the 192-byte shbt5 contract:
    post0: vec4<f32>,             // x: channel A on, y: channel B on,
                                  // z: f_load (horizon fill), w: exposure
    post1: vec4<f32>,             // x: unwrap_transition (0 bulk, 1 torus)
};

struct SeedDefect {
    screen_pos: vec2<f32>,        // normalized [0,1] screen coordinate
    theta_e: f32,                 // Einstein radius (screen units)
    core_radius: f32,             // softening core epsilon_core
};

@group(0) @binding(0) var channel_a_tex: texture_2d<f32>;
@group(0) @binding(1) var channel_b_tex: texture_2d<f32>;
@group(0) @binding(2) var tex_sampler: sampler;
@group(0) @binding(3) var<uniform> params: LensingUniforms;
@group(0) @binding(4) var<storage, read> seeds: array<SeedDefect>;

// Conformal boundary unwrapping: roll the 3D comoving bulk onto the
// 2D CFT torus [0, 2pi)^2 as unwrap_transition goes 0 -> 1.
fn unwrap_torus_projection(uv: vec2<f32>) -> vec2<f32> {
    let pi2 = 6.28318530718;
    let theta1 = uv.x * pi2;
    let theta2 = uv.y * pi2;
    let torus = vec2<f32>((cos(theta1) + 1.0) * 0.5, (sin(theta2) + 1.0) * 0.5);
    return mix(uv, torus, params.post1.x);
}

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_post(@builtin(vertex_index) v_idx: u32) -> VertexOutput {
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

// Depth-aware bilateral blur over the convergence field: spatial Gaussian
// times a range kernel on Channel-A depth, giving the volumetric dark
// halo glow term.
fn sample_bilateral_convergence(uv: vec2<f32>, texel: vec2<f32>, center_depth: f32) -> f32 {
    var total_weight = 0.0;
    var accum_conv = 0.0;
    let radius = i32(clamp(params.dark_glow_radius, 1.0, 8.0));
    let sigma_s = params.dark_glow_radius;
    let sigma_r = 0.08;

    for (var dy = -radius; dy <= radius; dy = dy + 1) {
        for (var dx = -radius; dx <= radius; dx = dx + 1) {
            let offset_uv = uv + vec2<f32>(f32(dx), f32(dy)) * texel;
            let sample_b = textureSampleLevel(channel_b_tex, tex_sampler, offset_uv, 0.0);
            let sample_a = textureSampleLevel(channel_a_tex, tex_sampler, offset_uv, 0.0);

            let spatial_dist2 = f32(dx * dx + dy * dy);
            let depth_diff = abs(sample_a.a - center_depth);

            let w_s = exp(-spatial_dist2 / (2.0 * sigma_s * sigma_s));
            let w_r = exp(-(depth_diff * depth_diff) / (2.0 * sigma_r * sigma_r));
            let weight = w_s * w_r;

            accum_conv = accum_conv + sample_b.z * weight;
            total_weight = total_weight + weight;
        }
    }

    return accum_conv / max(total_weight, 0.00001);
}

@fragment
fn fs_post(in: VertexOutput) -> @location(0) vec4<f32> {
    let uv = in.uv;
    let texel = vec2<f32>(1.0 / params.screen_size.x, 1.0 / params.screen_size.y);
    let warped_uv = unwrap_torus_projection(uv);

    let b_center = textureSampleLevel(channel_b_tex, tex_sampler, warped_uv, 0.0);
    let center_depth = textureSampleLevel(channel_a_tex, tex_sampler, warped_uv, 0.0).a;

    // Screen-space macro-deflection: central-difference gradient of the
    // convergence field, corrected by the shear tensor Gamma.
    let b_right = textureSampleLevel(channel_b_tex, tex_sampler, warped_uv + vec2<f32>(texel.x, 0.0), 0.0);
    let b_left  = textureSampleLevel(channel_b_tex, tex_sampler, warped_uv - vec2<f32>(texel.x, 0.0), 0.0);
    let b_up    = textureSampleLevel(channel_b_tex, tex_sampler, warped_uv + vec2<f32>(0.0, texel.y), 0.0);
    let b_down  = textureSampleLevel(channel_b_tex, tex_sampler, warped_uv - vec2<f32>(0.0, texel.y), 0.0);

    let d_kappa_dx = (b_right.z - b_left.z) / (2.0 * texel.x);
    let d_kappa_dy = (b_up.z - b_down.z) / (2.0 * texel.y);
    let grad_kappa = vec2<f32>(d_kappa_dx, d_kappa_dy);

    let gamma_1 = b_center.x;
    let gamma_2 = b_center.y;
    // Gamma . grad kappa with Gamma = [[gamma_1, gamma_2], [gamma_2, -gamma_1]]
    let sheared_grad = vec2<f32>(
        gamma_1 * grad_kappa.x + gamma_2 * grad_kappa.y,
        gamma_2 * grad_kappa.x - gamma_1 * grad_kappa.y
    );

    var alpha_macro = (grad_kappa + sheared_grad) * (params.lensing_strength * 0.0005);

    // Analytical softened point-mass micro-deflection over active seeds.
    var alpha_seeds = vec2<f32>(0.0, 0.0);
    let num_seeds = min(params.seed_count, 64u);
    for (var i = 0u; i < num_seeds; i = i + 1u) {
        let s = seeds[i];
        let diff = warped_uv - s.screen_pos;
        let r2 = dot(diff, diff);
        let core2 = s.core_radius * s.core_radius;
        let defl_mag = (s.theta_e * s.theta_e) / (r2 + core2);
        alpha_seeds = alpha_seeds + diff * (defl_mag * params.lensing_strength);
    }

    let alpha_total = alpha_macro + alpha_seeds;

    // Wave-optics 3-tap chromatic dispersion: R/G/B sample coordinates
    // scaled by delta_disp so critical curves fringe into spectra.
    let disp = params.dispersion_coeff * 0.08;
    let uv_r = clamp(warped_uv - alpha_total * (1.0 - disp), vec2<f32>(0.0), vec2<f32>(1.0));
    let uv_g = clamp(warped_uv - alpha_total, vec2<f32>(0.0), vec2<f32>(1.0));
    let uv_b = clamp(warped_uv - alpha_total * (1.0 + disp), vec2<f32>(0.0), vec2<f32>(1.0));

    let rad_r = textureSampleLevel(channel_a_tex, tex_sampler, uv_r, 0.0).r;
    let rad_g = textureSampleLevel(channel_a_tex, tex_sampler, uv_g, 0.0).g;
    let rad_b = textureSampleLevel(channel_a_tex, tex_sampler, uv_b, 0.0).b;
    let lensed_color = vec3<f32>(rad_r, rad_g, rad_b) * params.post0.x;

    // Depth-aware bilateral convergence -> volumetric dark-matter halo glow.
    var dark_glow_emission = vec3<f32>(0.0);
    if (params.post0.y > 0.5) {
        let smooth_conv = sample_bilateral_convergence(warped_uv, texel, center_depth);
        let dark_glow_palette = vec3<f32>(0.15, 0.35, 0.85);
        dark_glow_emission = dark_glow_palette * (smooth_conv * params.dark_glow_intensity);
    }

    // Sharp caustic rings around dominant Einstein radii.
    let aspect = params.screen_size.x / params.screen_size.y;
    var caustic_ring_accent = 0.0;
    for (var i = 0u; i < num_seeds; i = i + 1u) {
        let s = seeds[i];
        let d = warped_uv - s.screen_pos;
        let dist = sqrt(d.x * d.x * aspect * aspect + d.y * d.y);
        let ring_diff = abs(dist - s.theta_e);
        caustic_ring_accent = caustic_ring_accent + exp(-ring_diff * ring_diff * 4000.0) * 0.4;
    }
    let caustic_rgb = vec3<f32>(caustic_ring_accent * 0.4, caustic_ring_accent * 0.8, caustic_ring_accent)
        * params.lensing_strength * params.post0.y;

    // Observer causal-entropy shimmer: Channel-B alpha carries the causal
    // entropy budget; modulate a faint Fresnel ripple at observer nodes.
    let entropy = b_center.a;
    let ripple_phase = params.time * 2.0 - entropy * 40.0;
    let causal_ripple = (sin(ripple_phase) * 0.5 + 0.5) * entropy * 0.12 * params.post0.y;
    let causal_rgb = vec3<f32>(0.4, 0.75, 1.0) * causal_ripple;

    // Holographic horizon boundary overlay: the loaded screen fraction
    // brightens toward the canvas edge as f_load -> 1 (de Sitter freeze).
    let edge = max(abs(uv.x - 0.5), abs(uv.y - 0.5)) * 2.0;
    let horizon = pow(edge, 8.0) * params.post0.z;
    let horizon_rgb = vec3<f32>(0.05, 0.18, 0.35) * horizon;

    // Boundary screen grid: CFT torus lattice overlay fades in with the
    // unwrap transition, plus a faint permanent lattice.
    let grid_uv = warped_uv * vec2<f32>(80.0 * aspect, 80.0);
    let grid_line = step(0.97, fract(grid_uv.x)) + step(0.97, fract(grid_uv.y));
    let grid_rgb = vec3<f32>(0.02, 0.05, 0.08) * clamp(grid_line, 0.0, 1.0)
        * (0.35 + 0.65 * params.post1.x);

    let final_composite = (lensed_color + dark_glow_emission + caustic_rgb + causal_rgb + horizon_rgb + grid_rgb)
        * params.post0.w;

    return vec4<f32>(final_composite, 1.0);
}
