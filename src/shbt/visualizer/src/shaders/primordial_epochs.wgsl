// primordial_epochs.wgsl
// Physical state mapping and procedural rendering for primordial phase transitions.

struct EpochUniforms {
    redshift_z: f32,
    plasma_temp_gev: f32,
    sound_horizon_mpc: f32,
    sphaleron_rate: f32,
    weil_seed_count: u32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
};

@group(0) @binding(0) var<uniform> epoch: EpochUniforms;
@group(0) @binding(1) var<storage, read> weil_seeds: array<vec4<f32>>;
@group(0) @binding(2) var out_field: texture_storage_3d<rgba16float, write>;

const WEIL_MODULAR_M: f32 = 2901360.0;

fn evaluate_ewsb_spinodal(p: vec3<f32>, z: f32) -> vec4<f32> {
    let k_spinodal = 42.0;
    let wave = sin(dot(p, vec3<f32>(k_spinodal)) + epoch.sphaleron_rate);
    let higgs_vev = clamp((1.0 - z / 1.0e15), 0.0, 1.0) * 246.0;
    return vec4<f32>(higgs_vev * abs(wave), epoch.sphaleron_rate, 0.0, 1.0);
}

fn evaluate_qcd_percolation(p: vec3<f32>, z: f32) -> vec4<f32> {
    let qcd_scale = 1.0 / (1.0 + exp((z - 1.0e12) * 1e-11));
    let noise = sin(p.x * 128.0) * cos(p.y * 128.0) * sin(p.z * 128.0);
    let chiral_condensate = select(0.0, 1.0, noise > 0.1) * qcd_scale;
    return vec4<f32>(chiral_condensate, 1.0 - chiral_condensate, 0.0, 1.0);
}

fn evaluate_recombination_bao(p: vec3<f32>, z: f32) -> vec4<f32> {
    var field = 0.0;
    let r_s = epoch.sound_horizon_mpc;
    let count = min(epoch.weil_seed_count, 64u);

    for (var i = 0u; i < count; i = i + 1u) {
        let seed = weil_seeds[i];
        let d = length(p - seed.xyz);
        let shell = exp(-pow((d - r_s) / 0.5, 2.0));
        field += shell * cos(seed.w * WEIL_MODULAR_M);
    }
    return vec4<f32>(field * 0.8, field * 0.4, field * 1.0, 1.0);
}

@compute @workgroup_size(4, 4, 4)
fn cs_render_primordial_epoch(@builtin(global_invocation_id) id: vec3<u32>) {
    let dim = textureDimensions(out_field);
    if (id.x >= dim.x || id.y >= dim.y || id.z >= dim.z) {
        return;
    }

    let p = vec3<f32>(id) / vec3<f32>(dim);
    let z = epoch.redshift_z;
    var result = vec4<f32>(0.0);

    if (z > 1.0e13) {
        result = evaluate_ewsb_spinodal(p, z);
    } else if (z > 1.0e10) {
        result = evaluate_qcd_percolation(p, z);
    } else {
        result = evaluate_recombination_bao(p, z);
    }

    textureStore(out_field, vec3<i32>(id), result);
}
