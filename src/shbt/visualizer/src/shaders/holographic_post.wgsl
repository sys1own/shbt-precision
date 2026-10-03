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
    //   coefficient, z/w reserved.
    post2: vec4<f32>,
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
    _pad: vec4<f32>,
};

@group(0) @binding(0) var channel_a_tex: texture_2d<f32>;
@group(0) @binding(1) var channel_b_tex: texture_2d<f32>;
@group(0) @binding(2) var tex_sampler: sampler;
@group(0) @binding(3) var<uniform> params: LensingUniforms;
@group(0) @binding(4) var<storage, read> seeds: array<SeedDefect>;
@group(0) @binding(5) var boundary_register_tex: texture_2d<f32>;
@group(0) @binding(6) var<storage, read> condensing_seeds: array<CondensationSeed>;
@group(0) @binding(7) var<uniform> glitch: GlitchUniforms;

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

// Enhancement 11: localized UV quantization / tearing around condensing
// register cells (pre-nucleation overflow glitch).
fn apply_condensation_glitch(uv: vec2<f32>, base_color: vec3<f32>, time: f32) -> vec3<f32> {
    var out_col = base_color;
    let count = min(glitch.count, 256u);

    for (var i = 0u; i < count; i = i + 1u) {
        let seed = condensing_seeds[i];
        let d = distance(uv, seed.screen_pos);
        let glitch_radius = 0.075 * clamp(seed.saturation, 0.0, 1.2);

        if (d < glitch_radius) {
            let falloff = 1.0 - (d / glitch_radius);
            let coarse_uv = floor(uv * 48.0) / 48.0;
            let noise = hash_noise(floor(uv * 128.0) + floor(time * 60.0));

            var glitch_sample = textureSampleLevel(channel_a_tex, tex_sampler, coarse_uv, 0.0).rgb;
            if (noise > 0.6) {
                glitch_sample = mix(glitch_sample, vec3<f32>(0.2, 0.9, 1.0) * noise, 0.85);
            }
            out_col = mix(out_col, glitch_sample, falloff * clamp(seed.saturation, 0.1, 0.95));
        }
    }
    return out_col;
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

    // Analytical softened point-mass micro-deflection over emergent seeds:
    // theta_E,k expands dynamically as each defect accretes boundary bits.
    var alpha_seeds = vec2<f32>(0.0, 0.0);
    let num_seeds = min(params.seed_count, 256u);
    for (var i = 0u; i < num_seeds; i = i + 1u) {
        let s = seeds[i];
        let diff = warped_uv - s.screen_pos;
        let r2 = dot(diff, diff);
        let core2 = s.core_radius * s.core_radius;
        // Clamp the near-core singularity: 64 emergent defects at
        // theta_E^2/core^2 ~ 20 otherwise drag every pixel into a core and
        // smear the frame into a white wash.
        let defl_mag = min((s.theta_e * s.theta_e) / (r2 + core2), 0.05);
        alpha_seeds = alpha_seeds + diff * (defl_mag * params.lensing_strength);
    }

    var alpha_total = alpha_macro + alpha_seeds;
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

    // Wave-optics caustic fringing regularizes det A = 0 into Airy
    // patterns (Enhancement 10) applied on the deflected sample.
    let fringe_col = evaluate_caustic_fringing(warped_uv, texel, vec3<f32>(rad_r, rad_g, rad_b));
    rad_r = mix(rad_r, fringe_col.r, params.dispersion_coeff);
    rad_g = mix(rad_g, fringe_col.g, params.dispersion_coeff);
    rad_b = mix(rad_b, fringe_col.b, params.dispersion_coeff);

    var lensed_color = vec3<f32>(rad_r, rad_g, rad_b) * params.post0.x;

    // Emergent condensation glitch (Enhancement 11): pre-nucleation
    // register overflow tears the UV field around saturating cells.
    lensed_color = apply_condensation_glitch(warped_uv, lensed_color, params.time);

    // Depth-aware bilateral convergence -> volumetric dark-matter halo glow.
    var dark_glow_emission = vec3<f32>(0.0);
    if (params.post0.y > 0.5) {
        let smooth_conv = sample_bilateral_convergence(warped_uv, texel, center_depth);
        let dark_glow_palette = vec3<f32>(0.15, 0.35, 0.85);
        // Soft-cap the summed convergence so stacked billboards can't
        // push the palette into the tone-map's white saturation point.
        dark_glow_emission = dark_glow_palette * min(smooth_conv * params.dark_glow_intensity, 0.6);
    }

    // Sharp caustic rings around dominant Einstein radii.
    let aspect = params.screen_size.x / params.screen_size.y;
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
    let grid_line = aa_lattice_line(grid_uv.x) + aa_lattice_line(grid_uv.y);
    let grid_rgb = vec3<f32>(0.02, 0.05, 0.08) * clamp(grid_line, 0.0, 1.0)
        * (0.35 + 0.65 * params.post1.x);

    let raw_composite = (lensed_color + dark_glow_emission + caustic_rgb + causal_rgb + horizon_rgb + grid_rgb)
        * params.post0.w;

    // Reinhard HDR tonemap (shbt8 Phase 2): x -> x/(1+x) compresses the
    // multi-billboard accumulation without clipping, followed by a
    // gentle contrast S-curve c^2(3-2c) so dense structure stays
    // legible instead of saturating.
    var reinhard = raw_composite / (vec3<f32>(1.0) + raw_composite);
    reinhard = reinhard * reinhard * (vec3<f32>(3.0) - 2.0 * reinhard);
    let final_composite = reinhard;

    // Conformal boundary unwrap HUD inset (Enhancement 1).
    let composed = render_boundary_overlay(uv, vec4<f32>(final_composite, 1.0));
    return composed;
}
