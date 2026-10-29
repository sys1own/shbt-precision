// cosmic_web_topology.wgsl
// Branchless PM-grid gravitational tidal tensor calculation and T-Web morphology.

struct TidalAnalysis {
    eigenvalues: vec3<f32>,     // lambda1 >= lambda2 >= lambda3
    e1: vec3<f32>,              // Eigenvector of lambda1 (primary compression)
    e2: vec3<f32>,              // Eigenvector of lambda2
    e3: vec3<f32>,              // Eigenvector of lambda3 (filament spine)
    morphology_class: u32,      // 0=Void, 1=Sheet, 2=Filament, 3=Knot
};

const PI: f32 = 3.14159265358979323846;
const TWO_PI_OVER_THREE: f32 = 2.0943951023931954923;
const LAMBDA_TH: f32 = 0.0198818;

fn extract_eigenvector(T: mat3x3<f32>, lambda: f32, fallback_ortho: vec3<f32>) -> vec3<f32> {
    let row0 = vec3<f32>(T[0][0] - lambda, T[0][1], T[0][2]);
    let row1 = vec3<f32>(T[0][1], T[1][1] - lambda, T[1][2]);
    let row2 = vec3<f32>(T[0][2], T[1][2], T[2][2] - lambda);

    let c01 = cross(row0, row1);
    let c02 = cross(row0, row2);
    let c12 = cross(row1, row2);

    let d01 = dot(c01, c01);
    let d02 = dot(c02, c02);
    let d12 = dot(c12, c12);

    var v = c01;
    var max_d = d01;
    if (d02 > max_d) { v = c02; max_d = d02; }
    if (d12 > max_d) { v = c12; max_d = d12; }

    if (max_d < 1e-8) {
        return normalize(cross(fallback_ortho, vec3<f32>(0.0, 1.0, 0.0) + vec3<f32>(0.001)));
    }
    return normalize(v);
}

fn solve_symmetric_eigensystem(T: mat3x3<f32>) -> TidalAnalysis {
    var out: TidalAnalysis;
    let i1 = T[0][0] + T[1][1] + T[2][2];
    let trace_third = i1 * 0.333333333333;

    let b00 = T[0][0] - trace_third;
    let b11 = T[1][1] - trace_third;
    let b22 = T[2][2] - trace_third;
    let b01 = T[0][1];
    let b02 = T[0][2];
    let b12 = T[1][2];

    let p = (b00*b00 + b11*b11 + b22*b22 + 2.0*(b01*b01 + b02*b02 + b12*b12)) * 0.166666666667;
    let q = 0.5 * (b00*(b11*b22 - b12*b12) - b01*(b01*b22 - b02*b12) + b02*(b01*b12 - b02*b11));

    if (p <= 1e-12) {
        out.eigenvalues = vec3<f32>(trace_third);
        out.e1 = vec3<f32>(1.0, 0.0, 0.0);
        out.e2 = vec3<f32>(0.0, 1.0, 0.0);
        out.e3 = vec3<f32>(0.0, 0.0, 1.0);
        out.morphology_class = select(0u, 3u, trace_third > LAMBDA_TH);
        return out;
    }

    let sqrt_p = sqrt(p);
    let phi = acos(clamp(q / (p * sqrt_p), -1.0, 1.0)) * 0.333333333333;

    let l1 = trace_third + 2.0 * sqrt_p * cos(phi);
    let l3 = trace_third + 2.0 * sqrt_p * cos(phi + TWO_PI_OVER_THREE);
    let l2 = i1 - l1 - l3;
    out.eigenvalues = vec3<f32>(l1, l2, l3);

    out.e1 = extract_eigenvector(T, l1, vec3<f32>(1.0, 0.0, 0.0));
    out.e2 = extract_eigenvector(T, l2, out.e1);
    out.e3 = cross(out.e1, out.e2);

    let c1 = u32(l1 > LAMBDA_TH);
    let c2 = u32(l2 > LAMBDA_TH);
    let c3 = u32(l3 > LAMBDA_TH);
    out.morphology_class = c1 + c2 + c3;

    return out;
}

@group(0) @binding(0) var grid_potential: texture_3d<f32>;
@group(0) @binding(1) var<storage, read_write> tidal_output: array<TidalAnalysis>;

@compute @workgroup_size(8, 8, 8)
fn cs_compute_tidal_tensor(@builtin(global_invocation_id) id: vec3<u32>) {
    let dim = textureDimensions(grid_potential);
    if (id.x >= dim.x || id.y >= dim.y || id.z >= dim.z) {
        return;
    }

    let i = vec3<i32>(id);
    let n = vec3<i32>(dim);

    let ip_x = (i.x + 1) % n.x;
    let im_x = (i.x - 1 + n.x) % n.x;
    let ip_y = (i.y + 1) % n.y;
    let im_y = (i.y - 1 + n.y) % n.y;
    let ip_z = (i.z + 1) % n.z;
    let im_z = (i.z - 1 + n.z) % n.z;

    let phi_c = textureLoad(grid_potential, i, 0).r;
    let phi_px = textureLoad(grid_potential, vec3<i32>(ip_x, i.y, i.z), 0).r;
    let phi_mx = textureLoad(grid_potential, vec3<i32>(im_x, i.y, i.z), 0).r;
    let phi_py = textureLoad(grid_potential, vec3<i32>(i.x, ip_y, i.z), 0).r;
    let phi_my = textureLoad(grid_potential, vec3<i32>(i.x, im_y, i.z), 0).r;
    let phi_pz = textureLoad(grid_potential, vec3<i32>(i.x, i.y, ip_z), 0).r;
    let phi_mz = textureLoad(grid_potential, vec3<i32>(i.x, i.y, im_z), 0).r;

    let t_xx = phi_px - 2.0 * phi_c + phi_mx;
    let t_yy = phi_py - 2.0 * phi_c + phi_my;
    let t_zz = phi_pz - 2.0 * phi_c + phi_mz;

    let phi_px_py = textureLoad(grid_potential, vec3<i32>(ip_x, ip_y, i.z), 0).r;
    let phi_px_my = textureLoad(grid_potential, vec3<i32>(ip_x, im_y, i.z), 0).r;
    let phi_mx_py = textureLoad(grid_potential, vec3<i32>(im_x, ip_y, i.z), 0).r;
    let phi_mx_my = textureLoad(grid_potential, vec3<i32>(im_x, im_y, i.z), 0).r;
    let t_xy = 0.25 * (phi_px_py - phi_px_my - phi_mx_py + phi_mx_my);

    let phi_px_pz = textureLoad(grid_potential, vec3<i32>(ip_x, i.y, ip_z), 0).r;
    let phi_px_mz = textureLoad(grid_potential, vec3<i32>(ip_x, i.y, im_z), 0).r;
    let phi_mx_pz = textureLoad(grid_potential, vec3<i32>(im_x, i.y, ip_z), 0).r;
    let phi_mx_mz = textureLoad(grid_potential, vec3<i32>(im_x, i.y, im_z), 0).r;
    let t_xz = 0.25 * (phi_px_pz - phi_px_mz - phi_mx_pz + phi_mx_mz);

    let phi_py_pz = textureLoad(grid_potential, vec3<i32>(i.x, ip_y, ip_z), 0).r;
    let phi_py_mz = textureLoad(grid_potential, vec3<i32>(i.x, ip_y, im_z), 0).r;
    let phi_my_pz = textureLoad(grid_potential, vec3<i32>(i.x, im_y, ip_z), 0).r;
    let phi_my_mz = textureLoad(grid_potential, vec3<i32>(i.x, im_y, im_z), 0).r;
    let t_yz = 0.25 * (phi_py_pz - phi_py_mz - phi_my_pz + phi_my_mz);

    let T = mat3x3<f32>(
        vec3<f32>(t_xx, t_xy, t_xz),
        vec3<f32>(t_xy, t_yy, t_yz),
        vec3<f32>(t_xz, t_yz, t_zz)
    );

    let flat_idx = (id.z * dim.y + id.y) * dim.x + id.x;
    tidal_output[flat_idx] = solve_symmetric_eigensystem(T);
}
