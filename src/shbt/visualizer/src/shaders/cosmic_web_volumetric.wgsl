// cosmic_web_volumetric.wgsl
// Volumetric WHIM ray-marching compute shader evaluating Bremsstrahlung, Ly-alpha, tSZ, and ISW.

struct RayPayload {
    xray_flux: f32,
    lyman_alpha: f32,
    compton_y: f32,
    isw_delta_t: f32,
};

struct RenderUniforms {
    inv_view_proj: mat4x4<f32>,
    camera_pos: vec3<f32>,
    step_size: f32,
    max_steps: u32,
    redshift: f32,
    compton_t_baseline: f32,
    _pad: f32,
};

@group(0) @binding(0) var grid_rho_baryon: texture_3d<f32>;
@group(0) @binding(1) var grid_velocity_div: texture_3d<f32>;
@group(0) @binding(2) var grid_dphi_dt: texture_3d<f32>;
@group(0) @binding(3) var output_texture: texture_storage_2d<rgba16float, write>;
@group(0) @binding(4) var<uniform> render_params: RenderUniforms;

fn evaluate_bremsstrahlung_pade(ne: f32, T_eff: f32, photon_energy_kev: f32) -> f32 {
    let kt_kev = T_eff * 8.617333e-8;
    if (kt_kev < 0.01) { return 0.0; }
    let u = log(kt_kev) * 0.4342944819;
    let gaunt_ff = (1.201 + u * (0.142 + 0.051 * u)) / (1.0 + u * (0.082 + 0.039 * u));
    return 1.4e-27 * ne * ne * sqrt(T_eff) * exp(-photon_energy_kev / kt_kev) * gaunt_ff;
}

fn evaluate_lyman_alpha(ne: f32, n_hi: f32, T_eff: f32) -> f32 {
    if (T_eff < 5000.0 || T_eff > 500000.0) { return 0.0; }
    let excitation = exp(-118348.0 / T_eff);
    let t5 = T_eff * 1e-5;
    let suppression = 1.0 / (1.0 + t5 * sqrt(t5));
    return 3.7e-24 * ne * n_hi * excitation / (sqrt(T_eff) * (1.0 + suppression));
}

@compute @workgroup_size(8, 8, 1)
fn cs_volumetric_whim(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let screen_dim = textureDimensions(output_texture);
    if (global_id.x >= screen_dim.x || global_id.y >= screen_dim.y) {
        return;
    }

    let uv = (vec2<f32>(global_id.xy) + vec2<f32>(0.5)) / vec2<f32>(screen_dim);
    let clip_pos = vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0);
    let world_target = render_params.inv_view_proj * clip_pos;
    let ray_dir = normalize(world_target.xyz / world_target.w - render_params.camera_pos);

    var ray_pos = render_params.camera_pos;
    var acc = RayPayload(0.0, 0.0, 0.0, 0.0);

    for (var step = 0u; step < render_params.max_steps; step = step + 1u) {
        let uvw = ray_pos;
        if (any(uvw < vec3<f32>(0.0)) || any(uvw > vec3<f32>(1.0))) {
            ray_pos += ray_dir * render_params.step_size;
            continue;
        }

        let coord = vec3<i32>(uvw * 256.0);
        let rho_b = textureLoad(grid_rho_baryon, coord, 0).r;
        let div_v = textureLoad(grid_velocity_div, coord, 0).r;
        let dphi_dt = textureLoad(grid_dphi_dt, coord, 0).r;

        let compression = min(0.0, div_v);
        let t_shock = 40.34 * (compression * compression) / pow(1.0 + render_params.redshift, 2.0);
        let t_eff = max(render_params.compton_t_baseline + t_shock, 2.73);

        let n_e = rho_b * 0.88;
        let n_hi = rho_b * max(0.0, 1.0 - (t_eff / 100000.0));
        let dl = render_params.step_size * 100.0;

        acc.xray_flux += evaluate_bremsstrahlung_pade(n_e, t_eff, 1.0) * dl;
        acc.lyman_alpha += evaluate_lyman_alpha(n_e, n_hi, t_eff) * dl;
        acc.compton_y += 1.3328e-27 * n_e * t_eff * dl;
        acc.isw_delta_t += (2.0 / 27.0) * dphi_dt * dl;

        ray_pos += ray_dir * render_params.step_size;
    }

    textureStore(output_texture, vec2<i32>(global_id.xy), vec4<f32>(
        acc.xray_flux,
        acc.lyman_alpha,
        acc.compton_y * 1e5,
        acc.isw_delta_t
    ));
}
