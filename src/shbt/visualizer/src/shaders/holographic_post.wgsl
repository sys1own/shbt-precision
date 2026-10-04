// Full-screen holographic composite pass (shbt6 gravitational optics spec).
//
// Dual-scale gravitational lensing:
//   1. Macro-deflection from the Channel B convergence field via central
//      differences:  alpha_macro(u) = lambda_lens * (grad kappa + Gamma * grad kappa)
//   2. Micro-deflection from emergent topological seed defects; the
//      Einstein radius is evaluated dynamically from each condensed mass:
//        theta_E,k = sqrt(4 G M_seed,k / c^2 * D_ds / (D_d D_s))
//      with D_d, D_s, D_ds evaluated on the Tier-1 H_SHBT background
//      (units::CosmologicalContext; Thm 9.12) and the screen-space radius
//      theta_E^screen = theta_E / Theta_FoV. The host projects the
//      emergent centroids and fills theta_e per seed each frame.
//      Boundary chromatic dispersion (shbt8 Eq. Thm 9.12):
//        delta_disp(lambda) = zeta_disp * [(550 nm / lambda)^2 - 1]
//      evaluated at the canonical optical bands B=436, G=546, R=700 nm.
//
// shbt6 visual enhancements composited here:
//   - Conformal Boundary Unwrap Overlay (Enhancement 1): lower-right inset
//     sampling BoundaryTex2D (R: rho_B loading density, G: rho_E
//     entanglement entropy density) over the CFT torus [0, 2pi)^2.
//   - Wave-Optics Caustic Fringing (Enhancement 10): Airy-regularized
//     chromatic dispersion across critical curves det A = 0, red (700 nm)
//     deflected further than blue (440 nm).
//   - Emergent Seed Glitch (Enhancement 11): localized UV quantization and
//     register-overflow tearing ahead of each condensation event.
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
    // shbt8 thin-screen metrology extras:
    //   x: Theta_FoV (radians), y: zeta_disp boundary dispersion
    //   coefficient, z: bloom lift gain (Enhancement 12),
    //   w: exponential depth-fog density (Enhancement 13).
    post2: vec4<f32>,
    // shbt9 multi-plane optics: D_ms / D_s distance ratios for the
    // 4-slice lens stack centered at z_m in {0.5, 1.2, 2.2, 3.5} with
    // source plane z_s = 4.0 (Tier-1 angular-diameter distances).
    post3: vec4<f32>,
    // shbt11 thermodynamic optics: x = telemetry gamma_max, y = kappa_max
    // (lensing-Jacobian det J drivers for caustic halo accentuation),
    // z = Landauer desaturation knee Y_desat, w = tone exposure for the
    // luminance-preserving ACES curve.
    post4: vec4<f32>,
};

struct SeedDefect {
    screen_pos: vec2<f32>,        // normalized [0,1] screen coordinate
    theta_e: f32,                 // Einstein radius (screen units)
    core_radius: f32,             // softening core epsilon_core
};

// Emergent condensation glitch record (Enhancement 11).
struct CondensationSeed {
    screen_pos: vec2<f32>,
    saturation: f32,              // N_local / N_limit saturation ratio
    lifetime: f32,
    _pad: f32,
};

struct GlitchUniforms {
    count: u32,
    u_glitch_intensity: f32,      // master gain on the glitch blend [0,1]
    u_glitch_enabled: u32,        // 0 disables the effect entirely
    _pad: vec4<f32>,
    _pad2: vec4<f32>,
};

@group(0) @binding(0) var channel_a_tex: texture_2d<f32>;
@group(0) @binding(1) var channel_b_tex: texture_2d<f32>;
@group(0) @binding(2) var tex_sampler: sampler;
@group(0) @binding(3) var<uniform> params: LensingUniforms;
@group(0) @binding(4) var<storage, read> seeds: array<SeedDefect>;
@group(0) @binding(5) var boundary_register_tex: texture_2d<f32>;
@group(0) @binding(6) var<storage, read> condensing_seeds: array<CondensationSeed>;
@group(0) @binding(7) var<uniform> glitch: GlitchUniforms;
@group(0) @binding(8) var bloom_tex: texture_2d<f32>;
// shbt9 diagnostics: (x, v_x) phase-space raster for the split viewport.
@group(0) @binding(9) var phase_space_tex: texture_2d<f32>;



// Conformal boundary unwrapping: roll the 3D comoving bulk onto the
// 2D CFT torus [0, 2pi)^2 as unwrap_transition goes 0 -> 1.
fn unwrap_torus_projection(uv: vec2<f32>) -> vec2<f32> {
    let pi2 = 6.28318530718;
    let theta1 = uv.x * pi2;
    let theta2 = uv.y * pi2;
    let torus = vec2<f32>((cos(theta1) + 1.0) * 0.5, (sin(theta2) + 1.0) * 0.5);
    return mix(uv, torus, params.post1.x);
}

// Screen-space edge filter (shbt7 Phase 2): analytic FXAA-style
// antialiasing for the torus lattice lines. The hard step() edges alias
// badly at 80+ cells/screen; feathering the line half-width by fwidth
// resolves sub-pixel lines to smooth grey instead of shimmer.
fn aa_lattice_line(u: f32) -> f32 {
    let d = abs(fract(u + 0.5) - 0.5); // distance to nearest integer line
    let w = max(fwidth(u) * 1.2, 1.0e-4);
    return 1.0 - smoothstep(0.03 - w, 0.03 + w, d);
}

// Enhancement 1: toroidal HUD inset unwrap of the 2D boundary register.
fn unwrap_torus_inset(screen_uv: vec2<f32>, inset_pos: vec2<f32>, inset_size: vec2<f32>) -> vec2<f32> {
    let local_uv = (screen_uv - inset_pos) / inset_size;
    let theta_1 = fract(local_uv.x) * 6.28318530718;
    let theta_2 = fract(local_uv.y) * 6.28318530718;
    return vec2<f32>(theta_1 / 6.28318530718, theta_2 / 6.28318530718);
}

fn render_boundary_overlay(screen_uv: vec2<f32>, in_color: vec4<f32>) -> vec4<f32> {
    let inset_pos = vec2<f32>(0.74, 0.74);
    let inset_size = vec2<f32>(0.24, 0.24);

    // Evaluated unconditionally: fwidth() inside aa_lattice_line requires
    // uniform control flow, so the inset interior is selected at the end
    // rather than computed inside a divergent branch.
    let torus_uv = unwrap_torus_inset(screen_uv, inset_pos, inset_size);
    let cft_sample = textureSampleLevel(boundary_register_tex, tex_sampler, torus_uv, 0.0);
    let rho_b = cft_sample.r;
    let rho_e = cft_sample.g;

    let entanglement_cyan = vec3<f32>(0.02, 0.45, 0.88);
    let saturation_magenta = vec3<f32>(0.98, 0.12, 0.45);
    let cft_color = mix(entanglement_cyan * rho_e, saturation_magenta * rho_b, clamp(rho_b - 0.5, 0.0, 1.0));

    let grid_lines = aa_lattice_line(torus_uv.x * 16.0) + aa_lattice_line(torus_uv.y * 16.0);
    let composed = cft_color + vec3<f32>(0.2) * grid_lines;

    let inside = screen_uv.x >= inset_pos.x && screen_uv.x <= (inset_pos.x + inset_size.x) &&
        screen_uv.y >= inset_pos.y && screen_uv.y <= (inset_pos.y + inset_size.y);
    return select(in_color, mix(in_color, vec4<f32>(composed, 0.95), 0.85), inside);
}

// Enhancement 10: lensing Jacobian det A = (1 - kappa)^2 - |gamma|^2 over
// the Channel-B shear/convergence G-buffer.
fn compute_lens_jacobian(uv: vec2<f32>) -> f32 {
    let s = textureSampleLevel(channel_b_tex, tex_sampler, uv, 0.0);
    let g_sq = dot(s.xy, s.xy);
    let kappa = s.z;
    return (1.0 - kappa) * (1.0 - kappa) - g_sq;
}

// Airy diffraction fringing: split R/G/B samples along grad(det A) with
// chromatic scaling (red x1.30, green x1.00, blue x0.70).
fn evaluate_caustic_fringing(uv: vec2<f32>, texel_delta: vec2<f32>, base: vec3<f32>) -> vec3<f32> {
    let j_center = compute_lens_jacobian(uv);
    let j_dx = compute_lens_jacobian(uv + vec2<f32>(texel_delta.x, 0.0));
    let j_dy = compute_lens_jacobian(uv + vec2<f32>(0.0, texel_delta.y));
    let grad_j = vec2<f32>(j_dx - j_center, j_dy - j_center);

    let caustic_weight = 1.0 - smoothstep(0.0, 0.065, abs(j_center));
    if (caustic_weight <= 0.001) {
        return base;
    }

    let dir = normalize(grad_j + vec2<f32>(1e-6));
    let fringe = 0.015 * caustic_weight;

    let col_r = textureSampleLevel(channel_a_tex, tex_sampler, uv + dir * (fringe * 1.30), 0.0).r;
    let col_g = textureSampleLevel(channel_a_tex, tex_sampler, uv + dir * (fringe * 1.00), 0.0).g;
    let col_b = textureSampleLevel(channel_a_tex, tex_sampler, uv + dir * (fringe * 0.70), 0.0).b;

    return vec3<f32>(col_r, col_g, col_b);
}

fn hash_noise(p: vec2<f32>) -> f32 {
    let p3 = fract(vec3<f32>(p.xyx) * 0.1031);
    let dot_val = dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * dot_val);
}

// Enhancement 11: subtle, localized UV quantization around condensing
// register cells (pre-nucleation overflow glitch). Refined (shbt8): the
// effect is a faint nuance — tight radius, fine pixelation, low blend
// opacity — and fully toggleable via u_glitch_enabled.
fn apply_condensation_glitch(uv: vec2<f32>, base_color: vec3<f32>, time: f32) -> vec3<f32> {
    if (glitch.u_glitch_enabled == 0u) { return base_color; }
    var out_col = base_color;
    let count = min(glitch.count, 1024u);

    for (var i = 0u; i < count; i = i + 1u) {
        let seed = condensing_seeds[i];
        let d = distance(uv, seed.screen_pos);
        let glitch_radius = 0.025 * clamp(seed.saturation, 0.0, 1.2);

        if (d < glitch_radius) {
            let falloff = 1.0 - (d / glitch_radius);
            let coarse_uv = floor(uv * 256.0) / 256.0;
            let noise = hash_noise(floor(uv * 128.0) + floor(time * 60.0));

            var glitch_sample = textureSampleLevel(channel_a_tex, tex_sampler, coarse_uv, 0.0).rgb;
            if (noise > 0.6) {
                glitch_sample = mix(glitch_sample, vec3<f32>(0.2, 0.9, 1.0) * 0.5 * noise, 0.4);
            }
            out_col = mix(out_col, glitch_sample,
                          falloff * clamp(seed.saturation * glitch.u_glitch_intensity, 0.0, 0.25));
        }
    }
    return out_col;
}

// Photometric Rec.709 relative luminance (shbt11): the scalar carrier
// for the cross-bilateral range kernel, the caustic Laplacian, and the
// luminance-preserving tone map.
fn calculate_luminance(color: vec3<f32>) -> f32 {
    return dot(color, vec3<f32>(0.2126, 0.7152, 0.0722));
}

// Anti-Blowout Luminance-Preserving ACES film transform (shbt11 /
// Thm 9.18): the Narkiewicz/Hill ACES fit is evaluated strictly on
// scalar luminance Y_in; pristine un-clipped chromaticity is then
// re-injected at the mapped luminance, and a controlled Landauer corona
// desaturation (white-cyan core tint) rolls in only above the knee
// Y_desat so galactic centers keep golden/cyan structure instead of
// collapsing into flat white discs.
fn tone_map_aces_anti_blowout(color_hdr: vec3<f32>, exposure: f32) -> vec3<f32> {
    let exposed_color = max(vec3<f32>(0.0), color_hdr * exposure);
    let y_in = calculate_luminance(exposed_color);

    // Narkiewicz/Hill ACES fit evaluated on scalar luminance.
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    let y_mapped = clamp(
        (y_in * (a * y_in + b)) / (y_in * (c * y_in + d) + e), 0.0, 1.0);

    // Re-inject pristine un-clipped chromaticity (guard the dark floor).
    let chromatic_color = exposed_color * (y_mapped / max(y_in, 1.0e-5));

    // Controlled Landauer corona high-end desaturation.
    let desat_ratio = pow(y_in / max(u_desat_threshold(), 1.0e-3), 1.6);
    let desat_factor = clamp(1.0 - exp(-desat_ratio), 0.0, 0.85);

    let landauer_core_tint = vec3<f32>(0.85 * y_mapped, 0.95 * y_mapped, 1.0 * y_mapped);
    let final_linear = mix(chromatic_color, landauer_core_tint, desat_factor);

    return clamp(final_linear, vec3<f32>(0.0), vec3<f32>(1.0));
}

fn u_desat_threshold() -> f32 {
    return params.post4.z;
}

// Accurate linear-to-sRGB electro-optical transfer function.
fn linear_to_srgb(c: vec3<f32>) -> vec3<f32> {
    let cutoff = vec3<f32>(0.0031308);
    let lower = c * 12.92;
    let higher = 1.055 * pow(c, vec3<f32>(1.0 / 2.4)) - 0.055;
    return select(higher, lower, c <= cutoff);
}

// 9-tap cross bilateral filter over the HDR Channel-A radiance (shbt11):
// spatial Gaussian sigma_space times a photometric range kernel on
// luminance sigma_range — edge-preserving noise attenuation feeding the
// high-frequency Laplacian for caustic accentuation.
fn sample_cross_bilateral(center_uv: vec2<f32>, texel: vec2<f32>, center_rgb: vec3<f32>) -> vec3<f32> {
    let sigma_space = 1.8;
    // Tightened photometric range (0.10 -> 0.025): with the looser kernel
    // dark-matter halo luminance bled across caustic edges into the
    // deep-field voids, lifting their black point to a chalky gray.
    let sigma_range = 0.025;
    let two_sigma_sq_space = 2.0 * sigma_space * sigma_space;
    let two_sigma_sq_range = 2.0 * sigma_range * sigma_range;

    let center_lum = calculate_luminance(center_rgb);
    var filtered_color = center_rgb;
    var total_weight = 1.0;

    var offsets = array<vec2<f32>, 8>(
        vec2<f32>(-1.0, -1.0), vec2<f32>(0.0, -1.0), vec2<f32>(1.0, -1.0),
        vec2<f32>(-1.0, 0.0),                          vec2<f32>(1.0, 0.0),
        vec2<f32>(-1.0, 1.0),  vec2<f32>(0.0, 1.0),  vec2<f32>(1.0, 1.0)
    );

    for (var i = 0u; i < 8u; i = i + 1u) {
        let sample_uv = center_uv + offsets[i] * texel * 1.25;
        let s_rgb = textureSampleLevel(channel_a_tex, tex_sampler, sample_uv, 0.0).rgb;
        let s_lum = calculate_luminance(s_rgb);

        let dist_s_sq = dot(offsets[i], offsets[i]);
        let lum_diff = s_lum - center_lum;

        let w = exp(-dist_s_sq / two_sigma_sq_space)
            * exp(-(lum_diff * lum_diff) / two_sigma_sq_range);
        filtered_color = filtered_color + s_rgb * w;
        total_weight = total_weight + w;
    }

    return filtered_color / total_weight;
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

// Depth-aware 13-tap bilateral filter over Channel B (shbt10 precursor
// field formalism): spatial Gaussian sigma_s = 3.2 px times a range
// kernel sigma_d = 0.04 on the normalized Channel-A depth. Diffuses
// precursor congestion within continuous distance shells while
// preserving sharp contrast across foreground filament boundaries.
fn sample_bilateral_channel_b(center_uv: vec2<f32>, texel: vec2<f32>, center_depth: f32) -> vec4<f32> {
    let sigma_s = 3.2;
    let sigma_d = 0.04;
    let two_sigma_s_sq = 2.0 * sigma_s * sigma_s;
    let two_sigma_d_sq = 2.0 * sigma_d * sigma_d;
    var offsets = array<vec2<f32>, 13>(
        vec2<f32>(0.0, 0.0),
        vec2<f32>(-1.5, 0.0), vec2<f32>(1.5, 0.0),
        vec2<f32>(0.0, -1.5), vec2<f32>(0.0, 1.5),
        vec2<f32>(-3.0, 0.0), vec2<f32>(3.0, 0.0),
        vec2<f32>(0.0, -3.0), vec2<f32>(0.0, 3.0),
        vec2<f32>(-2.0, -2.0), vec2<f32>(2.0, -2.0),
        vec2<f32>(-2.0, 2.0), vec2<f32>(2.0, 2.0)
    );
    var accum_col = vec4<f32>(0.0);
    var accum_weight = 0.0;
    for (var i = 0; i < 13; i = i + 1) {
        let sample_uv = center_uv + offsets[i] * texel;
        let s_depth = textureSampleLevel(channel_a_tex, tex_sampler, sample_uv, 0.0).a;
        let s_col = textureSampleLevel(channel_b_tex, tex_sampler, sample_uv, 0.0);
        let dist_s_sq = dot(offsets[i], offsets[i]);
        let dist_d = abs(center_depth - s_depth) / max(center_depth, 0.01);
        let weight = exp(-dist_s_sq / two_sigma_s_sq)
            * exp(-(dist_d * dist_d) / two_sigma_d_sq);
        accum_col = accum_col + s_col * weight;
        accum_weight = accum_weight + weight;
    }
    return accum_col / max(accum_weight, 0.00001);
}

// Depth-aware bilateral blur over the convergence field: spatial Gaussian
// times a range kernel on Channel-A depth, giving the volumetric dark
// halo glow term.
fn sample_bilateral_convergence(uv: vec2<f32>, texel: vec2<f32>, center_depth: f32) -> f32 {
    var total_weight = 0.0;
    var accum_conv = 0.0;
    let radius = i32(clamp(params.dark_glow_radius, 1.0, 8.0));
    let sigma_s = params.dark_glow_radius;
    let sigma_r = 0.025;

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

// Macro lensing field at a screen position: central-difference gradient
// of the Channel-B convergence field corrected by the shear tensor.
fn macro_deflection(uv: vec2<f32>, texel: vec2<f32>) -> vec2<f32> {
    let b_c = textureSampleLevel(channel_b_tex, tex_sampler, uv, 0.0);
    let b_right = textureSampleLevel(channel_b_tex, tex_sampler, uv + vec2<f32>(texel.x, 0.0), 0.0);
    let b_left  = textureSampleLevel(channel_b_tex, tex_sampler, uv - vec2<f32>(texel.x, 0.0), 0.0);
    let b_up    = textureSampleLevel(channel_b_tex, tex_sampler, uv + vec2<f32>(0.0, texel.y), 0.0);
    let b_down  = textureSampleLevel(channel_b_tex, tex_sampler, uv - vec2<f32>(0.0, texel.y), 0.0);

    let d_kappa_dx = (b_right.z - b_left.z) / (2.0 * texel.x);
    let d_kappa_dy = (b_up.z - b_down.z) / (2.0 * texel.y);
    let grad_kappa = vec2<f32>(d_kappa_dx, d_kappa_dy);

    let gamma_1 = b_c.x;
    let gamma_2 = b_c.y;
    let sheared_grad = vec2<f32>(
        gamma_1 * grad_kappa.x + gamma_2 * grad_kappa.y,
        gamma_2 * grad_kappa.x - gamma_1 * grad_kappa.y
    );
    return (grad_kappa + sheared_grad) * (params.lensing_strength * 0.0005);
}

// Per-slice seed micro-deflection + Shapiro delay accumulation (shbt9):
// each emergent defect contributes theta_E^2/(r^2+eps^2) to the ray bend
// and a (1+z_m) weighted potential-depth term to Delta t_Shapiro.
fn seed_deflection_shapiro(uv: vec2<f32>, z_m: f32, shapiro: ptr<function, f32>) -> vec2<f32> {
    var alpha_seeds = vec2<f32>(0.0, 0.0);
    let num_seeds = min(params.seed_count, 1024u);
    for (var i = 0u; i < num_seeds; i = i + 1u) {
        let s = seeds[i];
        let diff = uv - s.screen_pos;
        let r2 = dot(diff, diff);
        let core2 = s.core_radius * s.core_radius;
        // Clamp the near-core singularity: 64 emergent defects at
        // theta_E^2/core^2 ~ 20 otherwise drag every pixel into a core and
        // smear the frame into a white wash.
        let defl_mag = min((s.theta_e * s.theta_e) / (r2 + core2), 0.05);
        alpha_seeds = alpha_seeds + diff * (defl_mag * params.lensing_strength);
        // Shapiro slab integral: Delta t_m ~ (1+z_m) * 4G/c^3 * Psi,
        // approximated by the normalized deflection potential per seed.
        *shapiro = *shapiro + (1.0 + z_m) * defl_mag * s.theta_e;
    }
    return alpha_seeds;
}

@fragment
fn fs_post(in: VertexOutput) -> @location(0) vec4<f32> {
    let uv = in.uv;

    let texel = vec2<f32>(1.0 / params.screen_size.x, 1.0 / params.screen_size.y);
    let warped_uv = unwrap_torus_projection(uv);

    let b_center = textureSampleLevel(channel_b_tex, tex_sampler, warped_uv, 0.0);
    let center_depth = textureSampleLevel(channel_a_tex, tex_sampler, warped_uv, 0.0).a;

    // 4-slice depth-stratified multi-plane deflection (shbt9 Phase 1):
    // the optical ray is marched through lens planes at
    // z_m in {0.5, 1.2, 2.2, 3.5}; each slab contributes a deflection
    // scaled by its Tier-1 distance ratio D_ms/D_s (post3[m]), and the
    // accumulated Shapiro delay modulates the chromatic phase below.
    var ray_uv = warped_uv;
    var alpha_total = vec2<f32>(0.0, 0.0);
    var shapiro_delay = 0.0;
    // Lens slab centers z_m (dynamic indexing requires a function-scope
    // array, not a module-scope const).
    var lens_slices = array<f32, 4>(0.5, 1.2, 2.2, 3.5);
    var ratio_sum = 0.0;
    for (var m = 0u; m < 4u; m = m + 1u) {
        ratio_sum = ratio_sum + params.post3[m];
    }
    ratio_sum = max(ratio_sum, 1.0e-3);
    for (var m = 0u; m < 4u; m = m + 1u) {
        let slice_weight = params.post3[m] / ratio_sum;
        let d_alpha = macro_deflection(ray_uv, texel)
            + seed_deflection_shapiro(ray_uv, lens_slices[m], &shapiro_delay);
        // Multi-plane lens equation: the ray position on the next screen
        // is displaced by this slab's scaled deflection.
        ray_uv = ray_uv - d_alpha * slice_weight;
        alpha_total = alpha_total + d_alpha * slice_weight;
    }
    // Bound the summed warp to 20% of the frame so overlapping defects
    // produce ring structure rather than a global smear.
    let alpha_len = length(alpha_total);
    if (alpha_len > 0.2) {
        alpha_total = alpha_total * (0.2 / alpha_len);
    }

    // Boundary chromatic dispersion (shbt8 Thm 9.12): the screen
    // deflection for each optical band follows
    //   delta_disp(lambda) = zeta_disp * [(lambda_0 / lambda)^2 - 1]
    // with lambda_0 = 550 nm reference and canonical bands
    // B = 436 nm, G = 546 nm, R = 700 nm ->
    //   delta_B = +0.5917 zeta, delta_G = +0.0147 zeta, delta_R = -0.3820 zeta
    // (zeta_disp ~ 0.04 at full slider -> delta_B ~ +0.0237, matching
    // the boundary-dispersion table).
    let zeta = params.post2.y;
    let disp_b = zeta * 0.5917;   // (550/436)^2 - 1
    let disp_g = zeta * 0.0147;   // (550/546)^2 - 1
    let disp_r = zeta * -0.3820;  // (550/700)^2 - 1
    let uv_r = clamp(warped_uv - alpha_total * (1.0 + disp_r), vec2<f32>(0.0), vec2<f32>(1.0));
    let uv_g = clamp(warped_uv - alpha_total * (1.0 + disp_g), vec2<f32>(0.0), vec2<f32>(1.0));
    let uv_b = clamp(warped_uv - alpha_total * (1.0 + disp_b), vec2<f32>(0.0), vec2<f32>(1.0));

    var rad_r = textureSampleLevel(channel_a_tex, tex_sampler, uv_r, 0.0).r;
    var rad_g = textureSampleLevel(channel_a_tex, tex_sampler, uv_g, 0.0).g;
    var rad_b = textureSampleLevel(channel_a_tex, tex_sampler, uv_b, 0.0).b;

    // Wide-scale optical pedestal (shbt12 deep-space pass): a uniform
    // mass sheet is gravitationally invisible (mass-sheet degeneracy),
    // so a diffuse field seen from inside the bulk must not glow. A
    // 16-tap coarse ring estimates the local diffuse mean of the
    // Channel-A luminance and the Channel-B convergence/shear fields;
    // only CONTRAST above that pedestal emits — void floors collapse
    // to black and genuine filaments/seeds keep full brightness.
    var lum_wide = 0.0;
    var conv_wide = 0.0;
    var shear_wide = vec2<f32>(0.0);
    for (var wi = 0u; wi < 16u; wi = wi + 1u) {
        let wa = f32(wi) * 0.3926991; // 2*pi/16
        let woff = vec2<f32>(cos(wa), sin(wa))
            * vec2<f32>(0.055, 0.055) * (1.0 + 0.35 * f32(wi % 3u));
        let ca_w = textureSampleLevel(channel_a_tex, tex_sampler,
            clamp(warped_uv + woff, vec2<f32>(0.0), vec2<f32>(1.0)), 0.0);
        let cb_w = textureSampleLevel(channel_b_tex, tex_sampler,
            clamp(warped_uv + woff, vec2<f32>(0.0), vec2<f32>(1.0)), 0.0);
        lum_wide = lum_wide + calculate_luminance(ca_w.rgb);
        conv_wide = conv_wide + cb_w.z;
        shear_wide = shear_wide + cb_w.xy;
    }
    lum_wide = lum_wide / 16.0;
    conv_wide = conv_wide / 16.0;
    shear_wide = shear_wide / 16.0;

    // Wave-optics caustic fringing regularizes det A = 0 into Airy
    // patterns (Enhancement 10) applied on the deflected sample.
    let fringe_col = evaluate_caustic_fringing(warped_uv, texel, vec3<f32>(rad_r, rad_g, rad_b));
    rad_r = mix(rad_r, fringe_col.r, params.dispersion_coeff);
    rad_g = mix(rad_g, fringe_col.g, params.dispersion_coeff);
    rad_b = mix(rad_b, fringe_col.b, params.dispersion_coeff);

    var lensed_color = vec3<f32>(rad_r, rad_g, rad_b) * params.post0.x;

    // Mass-sheet gate on the direct emission: pixels at or below the
    // wide-scale pedestal luminance fade to black, peaks above ~1.8x
    // the diffuse mean keep full radiance. Hue-preserving (scalar
    // gate, no per-channel shift).
    let lum_local = calculate_luminance(lensed_color);
    let sheet_gate = smoothstep(lum_wide * 0.95 + 1.0e-6,
        lum_wide * 1.80 + 2.0e-6, lum_local);
    lensed_color = lensed_color * sheet_gate;

    // Emergent condensation glitch (Enhancement 11): pre-nucleation
    // register overflow tears the UV field around saturating cells.
    lensed_color = apply_condensation_glitch(warped_uv, lensed_color, params.time);

    // Relativistic Shapiro time delay (shbt9 multi-slice optics): the
    // accumulated Delta t_Shapiro(theta) from the 4-slab march modulates
    // a false-color chromatic phase — deep-field delayed rays shift
    // red while direct rays stay neutral.
    let delay_intensity = clamp(shapiro_delay * 6.0, 0.0, 1.0);
    let shap_rgb = vec3<f32>(
        delay_intensity,
        0.5 * sin(delay_intensity * 6.2831853) + 0.5,
        1.0 - delay_intensity,
    );
    lensed_color = mix(lensed_color, lensed_color * (0.5 + 0.9 * shap_rgb), delay_intensity * 0.5);

    // Depth-aware bilateral convergence -> volumetric dark-matter halo glow.
    let aspect = params.screen_size.x / params.screen_size.y;
    let num_seeds = min(params.seed_count, 1024u);
    var dark_glow_emission = vec3<f32>(0.0);
    if (params.post0.y > 0.5) {
        let smooth_conv = sample_bilateral_convergence(warped_uv, texel, center_depth);
        let dark_glow_palette = vec3<f32>(0.15, 0.35, 0.85);
        // Convergence structure gate (shbt12): the diffuse pedestal is
        // invisible, so emission follows only the locally overdense
        // convergence — filaments glow, uniform sheets stay black.
        let conv_gate = smoothstep(conv_wide * 1.05 + 1.0e-6,
            conv_wide * 2.10 + 2.0e-6, smooth_conv);
        // Soft-cap the summed convergence so stacked billboards can't
        // push the palette into the tone-map's white saturation point.
        dark_glow_emission = dark_glow_palette
            * min(smooth_conv * params.dark_glow_intensity, 0.6)
            * conv_gate;

        // Precursor congestion field (shbt10): the 13-tap depth-aware
        // bilateral over Channel B synthesizes continuous precursor
        // halos — indigo dark-metric potential (shear .y) and teal
        // congestion (shear .x) — without a volumetric Poisson solve.
        let precursor_filtered = sample_bilateral_channel_b(warped_uv, texel, center_depth);
        let shear_gate = smoothstep(length(shear_wide) * 1.05 + 1.0e-6,
            length(shear_wide) * 2.20 + 2.0e-6, length(precursor_filtered));
        let dm_potential_color = vec3<f32>(0.12, 0.05, 0.28)
            * min(precursor_filtered.y * 3.5, 0.35) * shear_gate;
        let precursor_congestion_color = vec3<f32>(0.02, 0.22, 0.35)
            * min(precursor_filtered.x * 2.8, 0.30) * shear_gate;

        // Multi-tier seed radiance (shbt10): amber-white Planckian cubic
        // core inside the Einstein radius plus a blue Landauer thermal
        // corona whose extent scales with P_debt = M_seed x 906 GW
        // (theta_e^2 proxies the condensed mass).
        var seed_radiance = vec3<f32>(0.0);
        for (var i = 0u; i < num_seeds; i = i + 1u) {
            let s = seeds[i];
            let d = warped_uv - s.screen_pos;
            let dist = sqrt(d.x * d.x * aspect * aspect + d.y * d.y);
            let core_radius = max(s.core_radius * 0.6, 0.0008);
            let core_falloff = 1.0 / (1.0 + pow(dist / core_radius, 3.0));
            let core_col = vec3<f32>(1.0, 0.96, 0.88) * core_falloff * 0.30;
            let corona_radius = 0.045;
            let debt_scale = clamp(s.theta_e * s.theta_e * 40.0, 0.0, 0.45);
            let corona_col = vec3<f32>(0.25, 0.65, 1.0)
                * exp(-dist / corona_radius) * debt_scale;
            seed_radiance = seed_radiance + core_col + corona_col;
        }
        seed_radiance = min(seed_radiance, vec3<f32>(1.0));
        dark_glow_emission = dark_glow_emission
            + dm_potential_color + precursor_congestion_color + seed_radiance;
    }

    // Sharp caustic rings around dominant Einstein radii.
    var caustic_ring_accent = 0.0;
    for (var i = 0u; i < num_seeds; i = i + 1u) {
        let s = seeds[i];
        let d = warped_uv - s.screen_pos;
        let dist = sqrt(d.x * d.x * aspect * aspect + d.y * d.y);
        let ring_diff = abs(dist - s.theta_e);
        caustic_ring_accent = caustic_ring_accent + exp(-ring_diff * ring_diff * 4000.0) * 0.15;
    }
    let caustic_rgb = vec3<f32>(caustic_ring_accent * 0.4, caustic_ring_accent * 0.8, caustic_ring_accent)
        * params.lensing_strength * params.post0.y;

    // Observer causal-entropy shimmer: Channel-B alpha carries the causal
    // entropy budget; modulate a faint Fresnel ripple at observer nodes.
    let entropy = b_center.a;
    let ripple_phase = params.time * 2.0 - entropy * 40.0;
    // Observer shimmer is gated by local convergence structure as well —
    // an ungated uniform ripple is the same ambient blue lift the
    // deep-space pass removes.
    let conv_gate_shim = smoothstep(conv_wide * 1.05 + 1.0e-6,
        conv_wide * 2.10 + 2.0e-6, b_center.z);
    let causal_ripple = (sin(ripple_phase) * 0.5 + 0.5) * entropy * 0.12
        * params.post0.y * conv_gate_shim;
    let causal_rgb = vec3<f32>(0.4, 0.75, 1.0) * causal_ripple;

    // Holographic horizon boundary overlay: the loaded screen fraction
    // brightens toward the canvas edge as f_load -> 1 (de Sitter freeze).
    let edge = max(abs(uv.x - 0.5), abs(uv.y - 0.5)) * 2.0;
    let horizon = pow(edge, 8.0) * params.post0.z;
    let horizon_rgb = vec3<f32>(0.05, 0.18, 0.35) * horizon;

    // Boundary screen grid: CFT torus lattice overlay fades in with the
    // unwrap transition, plus a faint permanent lattice.
    let grid_uv = warped_uv * vec2<f32>(80.0 * aspect, 80.0);
    let grid_line = aa_lattice_line(grid_uv.x) + aa_lattice_line(grid_uv.y);
    let grid_rgb = vec3<f32>(0.02, 0.05, 0.08) * clamp(grid_line, 0.0, 1.0)
        * (0.35 + 0.65 * params.post1.x);

    // Caustic halo accentuation (shbt11 / Thm 9.18): the high-frequency
    // luminance Laplacian nabla^2 Y — extracted as the residual between
    // the raw Channel-A radiance and its 9-tap cross-bilateral —
    // localizes gravitational caustics; the accent gain follows the
    // telemetry-driven lensing Jacobian
    //   det J = |(1 - kappa_max)^2 - gamma_max^2|,  mag = 1/(det J + 0.08).
    let filtered_a = sample_cross_bilateral(warped_uv, texel, vec3<f32>(rad_r, rad_g, rad_b));
    let laplacian = max(0.0,
        calculate_luminance(vec3<f32>(rad_r, rad_g, rad_b))
        - calculate_luminance(filtered_a));
    // Epsilon-softened caustic magnification (canonical thin-screen
    // form): A_caustic = 1 / sqrt(det J^2 + eps^2) — the quadratic floor
    // regularizes the critical-curve divergence without the linear
    // 1/(detJ + eps) bias that flattened ring contrast.
    let det_jacobian =
        (1.0 - params.post4.y) * (1.0 - params.post4.y)
        - params.post4.x * params.post4.x;
    let eps_caustic = 0.08;
    let caustic_magnification = 1.0 / sqrt(
        det_jacobian * det_jacobian + eps_caustic * eps_caustic);
    let caustic_halo_accent = vec3<f32>(0.25, 0.75, 1.00)
        * (laplacian * caustic_magnification * 0.35);

    let raw_composite = (lensed_color + dark_glow_emission + caustic_rgb
        + caustic_halo_accent + causal_rgb + horizon_rgb + grid_rgb)
        * params.post0.w;

    // Bloom lift (Enhancement 12): additive half-res Gaussian bloom over
    // the bright-passed Channel-A texture, intensity params.post2.z.
    let bloom_rgb = textureSampleLevel(bloom_tex, tex_sampler, warped_uv, 0.0).rgb
        * params.post2.z;

    // Depth fog (Enhancement 13): exponential attenuation along the
    // Channel-A depth field toward the far void ambient, density
    // params.post2.w — dim voids read as receding atmosphere rather
    // than flat black.
    let fog_amt = (1.0 - exp(-max(center_depth, 0.0) * params.post2.w)) * 0.6;
    // Void-preserving fog: attenuate toward true black rather than a
    // lifted ambient — any non-zero ambient offset reads as chalky
    // gray/cyan noise across the deep field once the sRGB EOTF expands
    // the toe. Zero-flux pixels must stay vec3(0.0) end to end.
    let fog_ambient = vec3<f32>(0.0, 0.0, 0.0);
    let fogged = mix(raw_composite + bloom_rgb, fog_ambient, clamp(fog_amt, 0.0, 1.0));

    // Anti-blowout luminance-preserving ACES tonemap (shbt11 /
    // Thm 9.18): the film curve acts on scalar luminance so saturated
    // Planckian hues keep their chromaticity through the highlight
    // rolloff, with the controlled Landauer corona desaturation knee at
    // post4.z and tone exposure post4.w — replacing per-channel ACES
    // (which desaturated dense cores into flat white discs).
    let mapped_linear = tone_map_aces_anti_blowout(fogged, params.post4.w);
    // Accurate sRGB electro-optical transfer for display presentation.
    let final_composite = linear_to_srgb(mapped_linear);

    // Conformal boundary unwrap HUD inset (Enhancement 1).
    var composed = render_boundary_overlay(uv, vec4<f32>(final_composite, 1.0));

    // Diagnostics inset (shbt12): the (x, v_x) phase-space raster written
    // by diagnostics.wgsl only draws inside a bounded bottom-right rect
    // when the HUD toggle enables it — no raw half-screen blit. The DOM
    // card (#phase-space-card) supplies the border/title/axis chrome over
    // the same rect. This MUST be a post-composite select, not an early
    // return — render_boundary_overlay uses fwidth(), which requires
    // uniform control flow in WGSL.
    if (params.post1.y > 0.5
        && uv.x > 0.60 && uv.x < 0.98
        && uv.y > 0.52 && uv.y < 0.96) {
        let diag_uv = vec2<f32>((uv.x - 0.60) / 0.38, (uv.y - 0.52) / 0.44);
        composed = textureSampleLevel(phase_space_tex, tex_sampler, diag_uv, 0.0);
    }
    return composed;
}
