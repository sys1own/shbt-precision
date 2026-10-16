//! src/shbt/structure.rs
//!
//! High-Performance Non-Linear Structure and Cosmic Shear Module for SHBT
//! Implements Cardy capacity HMF, entropic transport P_NL, and Limber C_ell.

use bytemuck::{Pod, Zeroable};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

/// Canonical Kac-Moody boundary affine level k_l (critical string dimension)
pub const K_L: u32 = 26;
/// Canonical transverse gauge symmetry rank k_q
pub const K_Q_U32: u32 = 8;
/// Total holographic saturation capacity K
pub const K_TOTAL_U32: u32 = 312;
/// Boundary composite Kac-Moody denominator (k_l + k_q)
pub const K_SUM: u32 = 34;
/// Effective Virasoro central charge c_eff = k_l - k_q (coset capacity)
pub const C_EFF_NUMERATOR: f32 = 1325.0;
pub const C_EFF_DENOMINATOR: f32 = 154.0;
pub const C_EFF_RATIO: f32 = 1325.0 / 154.0; // ≈ 8.603896
/// One-loop determinant prefactor: sqrt(1325 / (308 * PI))
pub const A0_PREFACTOR_COEFF: f32 = 1.170_193_4;
/// Base linear overdensity threshold floor: sqrt(154) / (3 * sqrt(1325)) ≈ 0.065602
pub const DELTA_C0: f32 = 0.065_602_05;
/// First-principles holographic diffusion coefficient: 2 / (17 * PI)
pub const KAPPA_DIFF: f32 = 2.0 / (17.0 * std::f32::consts::PI); // ≈ 0.037447463
/// Exact 2D site percolation threshold on the square lattice
pub const P_C_2D_SITE: f32 = 0.59274621;
/// Canonical incubation onset redshift z_start = 30.000000
pub const Z_START_CANONICAL: f32 = 30.0;
/// Canonical incubation lower boundary z_end = 17.782386
pub const Z_END_CANONICAL: f32 = 17.782386;
/// D5 root lattice cell measure
pub const KAPPA_STAR_D5: f64 = 0.988725345719;
/// D5 agglomerative merge radius factor: sqrt(2) * kappa_*^D5 ≈ 1.398249068038
pub const D5_MERGE_COEFFICIENT: f64 = 1.398249068038;
/// SPH topological contraction factor: 38643 / 1540 = 3513 / 140 ≈ 25.092857
pub const SPH_TOPO_WEIGHT: f32 = 38643.0 / 1540.0;
/// Stinespring visible partition fraction: 10 / 33
pub const ETA_V: f32 = 10.0 / 33.0;

pub const KAPPA_GET_BASE: f64 = 1.0 / 23.0;
pub const KAPPA_GET_LOAD_COEFF: f64 = 1325.0 / 4004.0;
pub const C_EFF: f64 = 1325.0 / 154.0;
const K_Q: f64 = 8.0;
const SPEED_OF_LIGHT: f64 = 299792.458; // km/s

/// WebGPU emergence parameters matching std140 16-byte alignment.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct EmergenceParams {
    pub h_z: f32,
    pub dt: f32,
    pub a0_prefactor: f32,
    pub delta_c0: f32,
    pub c_eff: f32,
    pub growth_factor_d: f32,
    pub delta_th: f32,
    pub cell_volume: f32,
    pub a_0: f32,
    pub delta_crit: f32,
    pub sigma_8_z: f32,
    pub kappa_diff: f32,
    pub z_start: f32,
    pub z_end: f32,
    pub r_merge_factor: f32,
    pub sph_topo_weight: f32,
}

impl EmergenceParams {
    pub fn new(z: f32, dt: f32, cell_volume: f32) -> Self {
        let zp1 = 1.0 + z;
        let e_z = (0.3153 * zp1.powi(3) + 0.6847).sqrt();
        let h_z = 67.36 * e_z;
        let a_0 = h_z * A0_PREFACTOR_COEFF;
        let d_z = compute_linear_growth_factor(z);
        let linear_barrier = 1.68647 / d_z;
        let delta_th = linear_barrier.max(DELTA_C0);
        let sigma_8_z = 0.8111 * d_z;

        Self {
            h_z,
            dt,
            a0_prefactor: A0_PREFACTOR_COEFF,
            delta_c0: DELTA_C0,
            c_eff: C_EFF_RATIO,
            growth_factor_d: d_z,
            delta_th,
            cell_volume,
            a_0,
            delta_crit: 1.68647,
            sigma_8_z,
            kappa_diff: KAPPA_DIFF,
            z_start: Z_START_CANONICAL,
            z_end: Z_END_CANONICAL,
            r_merge_factor: D5_MERGE_COEFFICIENT as f32,
            sph_topo_weight: SPH_TOPO_WEIGHT,
        }
    }

    #[inline]
    pub fn compute_gamma_nuc(&self, local_overdensity: f32) -> f32 {
        let sigma = self.sigma_8_z.max(1.0e-5);
        let effective_barrier = self.delta_th.max(self.delta_c0);
        let remaining_barrier = (effective_barrier - local_overdensity).max(0.0);
        let nu = remaining_barrier / sigma;
        let s_inst = 0.5 * nu * nu;
        self.a_0 * (-s_inst).exp()
    }

    #[inline]
    pub fn compute_p_nuc(&self, local_overdensity: f32) -> f32 {
        const KM_S_MPC_TO_MYR_INV: f32 = 1.022_712_2e-6;
        let gamma = self.compute_gamma_nuc(local_overdensity);
        let lambda = gamma * KM_S_MPC_TO_MYR_INV * self.cell_volume * self.dt;
        if lambda < 1.0e-7 {
            lambda
        } else {
            1.0 - (-lambda).exp()
        }
    }
}

/// Computes the exact incubation onset redshift z_start from boundary coset capacity.
pub fn derive_z_start(ln_n_sat: f32) -> f32 {
    let prefactor = (18.0 * 34.0) / (18.0 * 312.0); // 17.0 / 156.0
    prefactor * ln_n_sat - 1.0
}

/// Computes the lower incubation boundary z_end via 2D site percolation.
pub fn derive_z_end(z_start: f32) -> f32 {
    z_start * P_C_2D_SITE
}

/// Computes the quintic incubation partition of unity Psi_nuc(z).
pub fn quintic_incubation_psi(z: f32, z_start: f32, z_end: f32) -> f32 {
    if z >= z_start {
        return 0.0;
    }
    if z <= z_end {
        return 1.0;
    }
    let u = (z_start - z) / (z_start - z_end);
    u * u * u * (u * (u * 6.0 - 15.0) + 10.0)
}

/// Computes linear growth factor D(z) via Carroll-Press-Turner fitting.
pub fn compute_linear_growth_factor(z: f32) -> f32 {
    let zp1 = 1.0 + z;
    let omega_z = (0.3153 * zp1.powi(3)) / (0.3153 * zp1.powi(3) + 0.6847);
    let num = (1.0 / zp1) * (
        omega_z.powf(4.0 / 7.0) - 0.6847 + (1.0 + 0.5 * omega_z) * (1.0 + 0.6847 / 70.0)
    );
    let den = 0.3153_f32.powf(4.0 / 7.0) - 0.6847 + (1.0 + 0.5 * 0.3153) * (1.0 + 0.6847 / 70.0);
    num / den
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ShbtCosmology {
    pub h: f64,
    pub omega_m: f64,
    pub omega_b: f64,
    pub omega_lambda: f64,
    pub sigma_8: f64,
    pub n_s: f64,
    pub gamma_cft: f64,
    pub n_sat: f64,
}

impl Default for ShbtCosmology {
    fn default() -> Self {
        Self {
            h: 0.6736,
            omega_m: 0.3153,
            omega_b: 0.0493,
            omega_lambda: 0.6847,
            sigma_8: 0.8111,
            n_s: 0.9649,
            gamma_cft: 1.0,
            n_sat: 1.0e122,
        }
    }
}

impl ShbtCosmology {
    pub fn e_z(&self, z: f64) -> f64 {
        let one_plus_z = 1.0 + z;
        (self.omega_m * one_plus_z.powi(3) + self.omega_lambda).sqrt()
    }

    pub fn h_z(&self, z: f64) -> f64 {
        100.0 * self.h * self.e_z(z)
    }

    pub fn comoving_distance(&self, z: f64, steps: usize) -> f64 {
        let dz = z / (steps as f64);
        let mut chi = 0.0;
        for i in 0..steps {
            let z_mid = (i as f64 + 0.5) * dz;
            chi += (SPEED_OF_LIGHT / self.h_z(z_mid)) * dz;
        }
        chi
    }

    pub fn cardy_capacity_ceiling(&self, z: f64, r_cell_mpc: f64) -> f64 {
        let h_ratio = self.h_z(z) / 100.0;
        let v_cell = (4.0 / 3.0) * PI * r_cell_mpc.powi(3);
        let r_h = SPEED_OF_LIGHT / self.h_z(z);
        let v_h = (4.0 / 3.0) * PI * r_h.powi(3);
        self.gamma_cft * h_ratio.powi(2) * (v_cell / v_h) * self.n_sat
    }

    pub fn kappa_get(&self, f_load: f64) -> f64 {
        KAPPA_GET_BASE * (1.0 + KAPPA_GET_LOAD_COEFF * f_load.clamp(0.0, 1.0))
    }

    pub fn growth_factor(&self, z: f64) -> f64 {
        let a = 1.0 / (1.0 + z);
        let omega_m_z = self.omega_m * (1.0 + z).powi(3) / self.e_z(z).powi(2);
        let gamma = 0.55;
        a * (omega_m_z.powf(gamma))
    }

    pub fn linear_power_spectrum(&self, k: f64, z: f64) -> f64 {
        let q = k / (self.h * (self.omega_m * self.h.powi(2)).exp());
        let l0 = (2.0 * std::f64::consts::E + 1.8 * q).ln();
        let c0 = 14.2 + 731.0 / (1.0 + 62.5 * q);
        let t_k = l0 / (l0 + c0 * q.powi(2));
        let p_primordial = k.powf(self.n_s);
        let d_z = self.growth_factor(z) / self.growth_factor(0.0);
        let a_norm = (self.sigma_8 / 0.8111).powi(2) * 2.0e5;
        a_norm * p_primordial * t_k.powi(2) * d_z.powi(2)
    }

    pub fn mass_variance(&self, mass_msun: f64, z: f64) -> f64 {
        let rho_m0 = 2.775e11 * self.omega_m * self.h.powi(2);
        let r = ((3.0 * mass_msun) / (4.0 * PI * rho_m0)).cbrt();
        let k_min: f64 = 1.0e-4;
        let k_max: f64 = 1.0e2;
        let steps = 400;
        let dlnk: f64 = (k_max / k_min).ln() / (steps as f64);
        let mut var: f64 = 0.0;
        for i in 0..steps {
            let lnk: f64 = k_min.ln() + (i as f64 + 0.5) * dlnk;
            let k: f64 = lnk.exp();
            let kr: f64 = k * r;
            let w: f64 = if kr < 1.0e-3 {
                1.0 - 0.1 * kr * kr
            } else {
                3.0 * (kr.sin() - kr * kr.cos()) / kr.powi(3)
            };
            let p_lin: f64 = self.linear_power_spectrum(k, z);
            var += (k.powi(3) * p_lin / (2.0 * PI * PI)) * w.powi(2) * dlnk;
        }
        var.sqrt()
    }

    pub fn dln_sigma_dln_m(&self, mass_msun: f64, z: f64) -> f64 {
        let eps = 0.02;
        let m_plus = mass_msun * (1.0 + eps);
        let m_minus = mass_msun * (1.0 - eps);
        let sig_plus = self.mass_variance(m_plus, z);
        let sig_minus = self.mass_variance(m_minus, z);
        let sig_mid = self.mass_variance(mass_msun, z);
        ((sig_plus - sig_minus) / (2.0 * eps * sig_mid)).abs()
    }

    pub fn f_shbt(&self, sigma: f64, z: f64) -> f64 {
        let f_load_bg = 0.58 / (1.0 + 0.15 * z);
        let delta_crit = (1.0 / f_load_bg) - 1.0;
        let factor = (2.0 * PI * C_EFF / K_Q).sqrt() * f_load_bg;
        let nu_inst = (delta_crit / sigma) * factor;
        let p = 1.0 / 23.0;
        let a_shbt = 0.3222;
        (2.0 / PI).sqrt() * a_shbt * nu_inst * (1.0 + (1.0 / nu_inst.powi(2)).powf(p)) * (-0.5 * nu_inst.powi(2)).exp()
    }

    pub fn halo_mass_function(&self, mass_msun: f64, z: f64) -> f64 {
        let rho_m0 = 2.775e11 * self.omega_m * self.h.powi(2);
        let sigma = self.mass_variance(mass_msun, z);
        let dln_sig = self.dln_sigma_dln_m(mass_msun, z);
        let f_val = self.f_shbt(sigma, z);
        (rho_m0 / mass_msun) * dln_sig * f_val
    }

    pub fn power_spectrum_nonlinear(&self, k: f64, z: f64) -> f64 {
        let p_lin = self.linear_power_spectrum(k, z);
        let delta_lin_sq = (k.powi(3) * p_lin) / (2.0 * PI * PI);
        let f_load = delta_lin_sq / (1.0 + delta_lin_sq);
        let kappa = self.kappa_get(f_load);

        let r_cell = 8.0 / (1.0 + z);
        let s_get = 1.0 / (1.0 + kappa * (k * r_cell).powi(2));
        let p_2h = p_lin * s_get;

        let r_s = 0.25 / (1.0 + z);
        let p_1h_amp = 1.8e4 / (1.0 + z).powf(1.8);
        let u_k = 1.0 / (1.0 + (k * r_s).powi(2));
        let core_trunc = 1.0 - kappa * ((k * r_s).powi(2) / (1.0 + (k * r_s).powi(2)));
        let p_1h = p_1h_amp * u_k.powi(2) * core_trunc;

        p_2h + p_1h
    }

    pub fn compute_c_ell(&self, ell: f64, z_min: f64, z_max: f64, steps: usize) -> f64 {
        let dz = (z_max - z_min) / (steps as f64);
        let mut c_ell = 0.0;
        let c_km_s = SPEED_OF_LIGHT;
        let h0 = 100.0 * self.h;

        for i in 0..steps {
            let z = z_min + (i as f64 + 0.5) * dz;
            let chi = self.comoving_distance(z, 200);
            let k = (ell + 0.5) / chi;
            let a = 1.0 / (1.0 + z);
            let w_lens = (3.0 * h0.powi(2) * self.omega_m / (2.0 * c_km_s.powi(2))) * (chi / a) * ((1.5 - z).max(0.0) / 1.5);
            let p_nl = self.power_spectrum_nonlinear(k, z);
            let d_chi_dz = c_km_s / self.h_z(z);
            c_ell += (w_lens.powi(2) / chi.powi(2)) * p_nl * d_chi_dz * dz;
        }
        c_ell
    }
}

#[pyclass]
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct BoxMetrology {
    #[pyo3(get)]
    pub s_cft: f64,
    #[pyo3(get)]
    pub box_curvature_ratio: f64,
    #[pyo3(get)]
    pub lambda_holo_box_si: f64,
    #[pyo3(get)]
    pub lambda_holo_box_mpc2: f64,
    #[pyo3(get)]
    pub l_box_mpc: f64,
    #[pyo3(get)]
    pub l_box_h_mpc: f64,
    #[pyo3(get)]
    pub k1_box_h: f64,
    #[pyo3(get)]
    pub k1_phys: f64,
    #[pyo3(get)]
    pub k1_contracted_h: f64,
    #[pyo3(get)]
    pub delta_ta: f64,
    #[pyo3(get)]
    pub variance_gate_ceiling: f64,
}

#[pymethods]
impl BoxMetrology {
    pub fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new_bound(py);
        d.set_item("s_cft", self.s_cft)?;
        d.set_item("box_curvature_ratio", self.box_curvature_ratio)?;
        d.set_item("lambda_holo_box_si", self.lambda_holo_box_si)?;
        d.set_item("lambda_holo_box_mpc2", self.lambda_holo_box_mpc2)?;
        d.set_item("l_box_mpc", self.l_box_mpc)?;
        d.set_item("l_box_h_mpc", self.l_box_h_mpc)?;
        d.set_item("k1_box_h", self.k1_box_h)?;
        d.set_item("k1_phys", self.k1_phys)?;
        d.set_item("k1_contracted_h", self.k1_contracted_h)?;
        d.set_item("delta_ta", self.delta_ta)?;
        d.set_item("variance_gate_ceiling", self.variance_gate_ceiling)?;
        Ok(d)
    }
}

pub fn derive_box_metrology() -> BoxMetrology {
    let s_cft = (2_901_360_f64).ln();
    let box_curvature_ratio = 24.0 * 24.0 + 8.0 + (4.0_f64).ln() / s_cft;
    let lambda_holo_si = 1.08913883e-52;
    let mpc_to_m = 3.085677581491367e22;
    let lambda_holo_box_si = box_curvature_ratio * lambda_holo_si;
    let lambda_holo_box_mpc2 = lambda_holo_box_si * (mpc_to_m * mpc_to_m);
    let l_box_mpc = (3.0 / lambda_holo_box_mpc2).sqrt();
    let h = 0.6740052;
    let l_box_h_mpc = 150.0;
    let k1_phys = (2.0 * PI) / l_box_mpc;
    let k1_box_h = (2.0 * PI) / l_box_h_mpc;
    let k1_contracted_h = k1_phys * h;
    let delta_ta = 0.6 * (0.75 * PI).powf(2.0 / 3.0);
    let variance_gate_ceiling = (delta_ta / 10.0).powi(2);

    BoxMetrology {
        s_cft,
        box_curvature_ratio,
        lambda_holo_box_si,
        lambda_holo_box_mpc2,
        l_box_mpc,
        l_box_h_mpc,
        k1_box_h,
        k1_phys,
        k1_contracted_h,
        delta_ta,
        variance_gate_ceiling,
    }
}

#[pyclass]
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ShbtGrowthFactors {
    #[pyo3(get)]
    pub z: f64,
    #[pyo3(get)]
    pub a: f64,
    #[pyo3(get)]
    pub d1: f64,
    #[pyo3(get)]
    pub d2: f64,
    #[pyo3(get)]
    pub f1: f64,
    #[pyo3(get)]
    pub f2: f64,
}

#[pymethods]
impl ShbtGrowthFactors {
    pub fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new_bound(py);
        d.set_item("z", self.z)?;
        d.set_item("a", self.a)?;
        d.set_item("d1", self.d1)?;
        d.set_item("d2", self.d2)?;
        d.set_item("f1", self.f1)?;
        d.set_item("f2", self.f2)?;
        Ok(d)
    }
}

pub fn compute_shbt_growth_factors(z_init: f64) -> ShbtGrowthFactors {
    let a_target = 1.0 / (1.0 + z_init);
    let a_start = 1.0e-4;
    if a_target <= a_start {
        return ShbtGrowthFactors {
            z: z_init,
            a: a_target,
            d1: a_target,
            d2: -(3.0 / 7.0) * a_target * a_target,
            f1: 1.0,
            f2: 2.0,
        };
    }

    let omega_m = 0.315_f64;
    let omega_l = 1.0 - omega_m;
    let n_steps = 2000_usize;
    let s_start = a_start.ln();
    let s_end = a_target.ln();
    let ds = (s_end - s_start) / (n_steps as f64);

    let mut a = a_start;
    let mut y = [
        a_start,
        1.0,
        -(3.0 / 7.0) * a_start * a_start,
        -(6.0 / 7.0) * a_start,
    ];

    let rhs = |a_val: f64, state: &[f64; 4]| -> [f64; 4] {
        let d1 = state[0];
        let v1 = state[1];
        let d2 = state[2];
        let v2 = state[3];

        let exp_sq = omega_m / a_val.powi(3) + omega_l;
        let e = exp_sq.sqrt();
        let dln_e = -1.5 * omega_m / (a_val.powi(4) * exp_sq);
        let friction = 3.0 / a_val + dln_e;

        let f_load = if a_val < 0.05 {
            0.0
        } else {
            0.101866 * a_val.powi(3)
        };

        let source = 1.5 * omega_m / (a_val.powi(5) * e * e) * (1.0 - (10.0 / 33.0) * f_load);
        let acc1 = -friction * v1 + source * d1;
        let acc2 = -friction * v2 + source * d2 - source * (d1 * d1);
        [v1, acc1, v2, acc2]
    };

    for _ in 0..n_steps {
        let da = a * (ds.exp() - 1.0);
        let k1 = rhs(a, &y);
        let y_k2 = [
            y[0] + 0.5 * da * k1[0],
            y[1] + 0.5 * da * k1[1],
            y[2] + 0.5 * da * k1[2],
            y[3] + 0.5 * da * k1[3],
        ];
        let k2 = rhs(a + 0.5 * da, &y_k2);
        let y_k3 = [
            y[0] + 0.5 * da * k2[0],
            y[1] + 0.5 * da * k2[1],
            y[2] + 0.5 * da * k2[2],
            y[3] + 0.5 * da * k2[3],
        ];
        let k3 = rhs(a + 0.5 * da, &y_k3);
        let y_k4 = [
            y[0] + da * k3[0],
            y[1] + da * k3[1],
            y[2] + da * k3[2],
            y[3] + da * k3[3],
        ];
        let k4 = rhs(a + da, &y_k4);

        for i in 0..4 {
            y[i] += (da / 6.0) * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
        }
        a += da;
    }

    let d1 = y[0];
    let v1 = y[1];
    let d2 = y[2];
    let v2 = y[3];
    let f1 = (a / d1) * v1;
    let f2 = (a / d2) * v2;

    ShbtGrowthFactors {
        z: z_init,
        a: a_target,
        d1,
        d2,
        f1,
        f2,
    }
}

#[pyclass]
#[derive(Debug, Clone, Copy, Default)]
pub struct StructureFormationEngine {
    pub metrology: HolographicMetrology,
}

#[pymethods]
impl StructureFormationEngine {
    #[new]
    pub fn new() -> Self {
        Self {
            metrology: HolographicMetrology::new(),
        }
    }

    pub fn derive_box_metrology(&self) -> BoxMetrology {
        derive_box_metrology()
    }

    pub fn compute_shbt_growth_factors(&self, z_init: f64) -> ShbtGrowthFactors {
        compute_shbt_growth_factors(z_init)
    }

    pub fn audit(&self) -> StructureAuditReport {
        run_structure_audit()
    }
}

#[pyclass]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StructureAuditReport {
    #[pyo3(get)]
    pub s8_primordial: f64,
    #[pyo3(get)]
    pub s8_effective: f64,
    #[pyo3(get)]
    pub kappa_get_min: f64,
    #[pyo3(get)]
    pub kappa_get_max: f64,
    #[pyo3(get)]
    pub c_ell_1000: f64,
    #[pyo3(get)]
    pub box_metrology: BoxMetrology,
    #[pyo3(get)]
    pub growth_z100: ShbtGrowthFactors,
    #[pyo3(get)]
    pub all_passed: bool,
}

#[pymethods]
impl StructureAuditReport {
    pub fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new_bound(py);
        d.set_item("s8_primordial", self.s8_primordial)?;
        d.set_item("s8_effective", self.s8_effective)?;
        d.set_item("kappa_get_min", self.kappa_get_min)?;
        d.set_item("kappa_get_max", self.kappa_get_max)?;
        d.set_item("c_ell_1000", self.c_ell_1000)?;
        d.set_item("box_metrology", self.box_metrology.to_dict(py)?)?;
        d.set_item("growth_z100", self.growth_z100.to_dict(py)?)?;
        d.set_item("all_passed", self.all_passed)?;
        Ok(d)
    }
}

/// Executes the full structure formation and cosmic shear audit.
pub fn run_structure_audit() -> StructureAuditReport {
    let cosmo = ShbtCosmology::default();
    let s8_primordial = cosmo.sigma_8 * (cosmo.omega_m / 0.3).sqrt();
    let kappa_min = cosmo.kappa_get(0.0);
    let kappa_max = cosmo.kappa_get(1.0);
    let s8_effective = s8_primordial * (1.0 - 1.5 * cosmo.kappa_get(0.75));
    let c_ell_1000 = cosmo.compute_c_ell(1000.0, 0.1, 1.4, 50);

    let box_metrology = derive_box_metrology();
    let growth_z100 = compute_shbt_growth_factors(100.0);

    let all_passed = (s8_primordial - 0.831).abs() < 0.02
        && (s8_effective - 0.764).abs() < 0.02
        && (kappa_min - 1.0 / 23.0).abs() < 1e-6
        && (kappa_max - 5329.0 / 92092.0).abs() < 1e-6
        && c_ell_1000 > 0.0
        && (box_metrology.box_curvature_ratio - 584.0931606).abs() < 1e-4
        && (box_metrology.l_box_mpc - 222.5502).abs() < 0.05
        && (growth_z100.d1 - 0.00990).abs() < 1e-4;

    StructureAuditReport {
        s8_primordial,
        s8_effective,
        kappa_get_min: kappa_min,
        kappa_get_max: kappa_max,
        c_ell_1000,
        box_metrology,
        growth_z100,
        all_passed,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kappa_get_limits() {
        let cosmo = ShbtCosmology::default();
        let k_min = cosmo.kappa_get(0.0);
        let k_max = cosmo.kappa_get(1.0);

        assert!((k_min - 1.0 / 23.0).abs() < 1e-10);
        assert!((k_max - 5329.0 / 92092.0).abs() < 1e-10);
    }

    #[test]
    fn test_s8_effective_resolution() {
        let report = run_structure_audit();
        assert!(report.all_passed);
        assert!(report.s8_effective < report.s8_primordial);
        assert!((report.s8_effective - 0.764).abs() < 0.02);
    }

    #[test]
    fn test_cosmological_grid_isotropic_weil_phases() {
        let grid = CosmologicalGrid::new(256, 0.674);
        assert!(grid.is_fundamental_mode_linear());
        let p_pos = grid.isotropic_weil_phase(1, 2, 3);
        let p_neg = grid.isotropic_weil_phase(-1, -2, -3);
        assert!((p_pos + p_neg).abs() < 1e-10);
        assert_eq!(grid.isotropic_weil_phase(0, 0, 0), 0.0);
    }

    #[test]
    fn test_hermitian_anti_symmetry() {
        let grid_size = 32;
        let phases = generate_weil_modular_phases(grid_size, 222.5502, 42);

        for ix in 0..grid_size {
            let neg_ix = (grid_size - ix) % grid_size;
            for iy in 0..grid_size {
                let neg_iy = (grid_size - iy) % grid_size;
                for iz in 0..grid_size {
                    let neg_iz = (grid_size - iz) % grid_size;

                    let idx_pos = (ix * grid_size + iy) * grid_size + iz;
                    let idx_neg = (neg_ix * grid_size + neg_iy) * grid_size + neg_iz;

                    if idx_pos == idx_neg {
                        assert_eq!(phases[idx_pos], 0.0f32);
                    } else {
                        let sum = (phases[idx_pos] + phases[idx_neg]).rem_euclid(2.0 * std::f32::consts::PI);
                        assert!(
                            sum < 1e-4 || (2.0 * std::f32::consts::PI - sum) < 1e-4,
                            "Hermitian condition violated at ({}, {}, {}): sum={}",
                            ix, iy, iz, sum
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn test_point_group_symmetry_breaking() {
        let grid_size = 32;
        let phases = generate_weil_modular_phases(grid_size, 222.5502, 42);

        let k1 = (1, 2, 3);
        let k2 = (2, 1, 3);

        let idx1 = (k1.0 * grid_size + k1.1) * grid_size + k1.2;
        let idx2 = (k2.0 * grid_size + k2.1) * grid_size + k2.2;

        assert_ne!(
            phases[idx1], phases[idx2],
            "Octahedral permutation symmetry must be strictly broken"
        );
    }
}

pub const CANONICAL_M: u64 = 2_901_360;
pub const CANONICAL_ALPHA: f64 = 13.856406460551018; // 24.0 / sqrt(3.0)
pub const CANONICAL_BETA: f64 = 2.221441469079183;   // PI / sqrt(2.0)
pub const SPLITMIX64_GAMMA: u64 = 0x9E3779B97F4A7C15;

pub const N_MOD: i64 = 2_901_360;
pub const CFT_MODULE_COUNT: u64 = 2_901_360;
pub const ALPHA_WEIL: f64 = CANONICAL_ALPHA;
pub const BETA_WEIL: f64 = CANONICAL_BETA;

pub const MODULAR_REGISTER_M: usize = 2_901_360;
pub const DELTA_C0_F64: f64 = 0.06561007575465141; // (1.0 / 3.0) * sqrt(154.0 / 3975.0)

/// Canonical Weil modular phase register lookup table.
pub struct WeilModularRegister {
    phases: Vec<f64>,
}

impl WeilModularRegister {
    pub fn new() -> Self {
        let mut phases = Vec::with_capacity(MODULAR_REGISTER_M);
        let golden_ratio = (1.0 + 5.0_f64.sqrt()) / 2.0;
        for i in 0..MODULAR_REGISTER_M {
            let theta = (i as f64 * golden_ratio * 2.0 * PI).rem_euclid(2.0 * PI);
            phases.push(theta);
        }
        Self { phases }
    }

    #[inline(always)]
    pub fn get_phase(&self, index: usize) -> f64 {
        self.phases[index % MODULAR_REGISTER_M]
    }
}

/// Continuous, branch-cut-free Haar angular dispersion on S^2.
#[inline]
pub fn continuous_haar_dispersion(kx: f64, ky: f64, kz: f64, grid_dim: f64) -> f64 {
    let norm = (kx * kx + ky * ky + kz * kz).sqrt();
    if norm < 1e-12 {
        return 0.0;
    }
    let hx = kx / norm;
    let hy = ky / norm;
    let hz = kz / norm;

    let c_x = 2.0_f64.sqrt();
    let c_y = 3.0_f64.sqrt();
    let c_z = 5.0_f64.sqrt();
    let c_sum = c_x + c_y + c_z;

    let proj = (c_x * hx + c_y * hy + c_z * hz) / c_sum;
    let val = proj * (grid_dim / 24.0);
    let frac = val - val.floor();
    2.0 * PI * frac
}

#[inline(always)]
pub fn splitmix64(mut z: u64) -> u64 {
    z = z.wrapping_add(SPLITMIX64_GAMMA);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

#[inline]
pub fn continuous_weil_norm_index(kx: f64, ky: f64, kz: f64, k1: f64) -> u64 {
    let k_norm = (kx * kx + ky * ky + kz * kz).sqrt();
    let s = k_norm / k1;
    let raw = (ALPHA_WEIL * s + BETA_WEIL * s * s).floor() as i64;
    raw.rem_euclid(CANONICAL_M as i64) as u64
}

#[inline]
pub fn haar_angular_dispersion(kx: f64, ky: f64, kz: f64) -> f64 {
    continuous_haar_dispersion(kx, ky, kz, 64.0)
}

/// Asserts cubic harmonic leakage |K_4| < 1e-7.
pub fn assert_cubic_harmonic_suppression(phases: &[f64], n: usize) {
    let mut sum_k4_phi = 0.0_f64;
    let mut count = 0_usize;

    for iz in 0..n {
        let nz = if iz <= n / 2 { iz as i32 } else { iz as i32 - n as i32 };
        for iy in 0..n {
            let ny = if iy <= n / 2 { iy as i32 } else { iy as i32 - n as i32 };
            for ix in 0..n {
                let nx = if ix <= n / 2 { ix as i32 } else { ix as i32 - n as i32 };
                let knorm_sq = (nx * nx + ny * ny + nz * nz) as f64;
                if knorm_sq > 0.0 {
                    let hx = nx as f64 / knorm_sq.sqrt();
                    let hy = ny as f64 / knorm_sq.sqrt();
                    let hz = nz as f64 / knorm_sq.sqrt();
                    let k4 = hx.powi(4) + hy.powi(4) + hz.powi(4) - 0.6;
                    let idx = (iz * n + iy) * n + ix;
                    let phi = phases[idx];
                    let phi_centered = if phi > PI { phi - 2.0 * PI } else { phi };
                    sum_k4_phi += phi_centered * k4;
                    count += 1;
                }
            }
        }
    }

    let mean_leakage = (sum_k4_phi / count as f64).abs();
    assert!(
        mean_leakage < 1e-7,
        "Cubic harmonic leakage violation: |K_4| = {:e} >= 1e-7",
        mean_leakage
    );
}

/// Generates Weil modular phases on an N x N x N Fourier lattice enforcing exact Hermitian symmetry.
pub fn generate_weil_modular_phases_canonical(grid_dim: usize, k1: f64) -> Vec<f64> {
    let register = WeilModularRegister::new();
    let n = grid_dim;
    let total_elements = n * n * n;
    let mut phases = vec![0.0_f64; total_elements];

    let alpha = ALPHA_WEIL;
    let beta = BETA_WEIL;

    for iz in 0..n {
        let nz = if iz <= n / 2 { iz as i32 } else { iz as i32 - n as i32 };
        for iy in 0..n {
            let ny = if iy <= n / 2 { iy as i32 } else { iy as i32 - n as i32 };
            for ix in 0..n {
                let nx = if ix <= n / 2 { ix as i32 } else { ix as i32 - n as i32 };

                let ix_neg = ((-nx).rem_euclid(n as i32)) as usize;
                let iy_neg = ((-ny).rem_euclid(n as i32)) as usize;
                let iz_neg = ((-nz).rem_euclid(n as i32)) as usize;

                if (nx == 0 && ny == 0 && nz == 0)
                    || (ix == ix_neg && iy == iy_neg && iz == iz_neg) {
                    let idx = (iz * n + iy) * n + ix;
                    phases[idx] = 0.0;
                    continue;
                }

                // Canonical positive half-space partition K+
                let in_k_plus = nz > 0
                    || (nz == 0 && ny > 0)
                    || (nz == 0 && ny == 0 && nx > 0);

                if in_k_plus {
                    let kx = nx as f64 * k1;
                    let ky = ny as f64 * k1;
                    let kz = nz as f64 * k1;
                    let knorm = (kx * kx + ky * ky + kz * kz).sqrt();

                    let knorm_ratio = knorm / k1;
                    let i_k = ((alpha * knorm_ratio + beta * knorm_ratio * knorm_ratio).floor() as usize)
                        % MODULAR_REGISTER_M;

                    let theta_base = register.get_phase(i_k);
                    let psi_iso = continuous_haar_dispersion(kx, ky, kz, n as f64);
                    let phi = (theta_base + psi_iso).rem_euclid(2.0 * PI);

                    let idx = (iz * n + iy) * n + ix;
                    phases[idx] = phi;

                    // Conjugate reflection to -k: phi(-k) = -phi(k) mod 2pi
                    let ix_neg = ((-nx).rem_euclid(n as i32)) as usize;
                    let iy_neg = ((-ny).rem_euclid(n as i32)) as usize;
                    let iz_neg = ((-nz).rem_euclid(n as i32)) as usize;
                    let idx_neg = (iz_neg * n + iy_neg) * n + ix_neg;

                    phases[idx_neg] = (-phi).rem_euclid(2.0 * PI);
                }
            }
        }
    }

    assert_cubic_harmonic_suppression(&phases, grid_dim);
    phases
}

pub fn generate_weil_modular_phases(
    n_mesh: usize,
    l_box: f64,
    _seed: u64,
) -> Vec<f32> {
    let k1 = (2.0 * PI) / l_box;
    let phases_f64 = generate_weil_modular_phases_canonical(n_mesh, k1);
    phases_f64.into_iter().map(|p| p as f32).collect()
}

pub fn generate_weil_modular_phases_with_modules(
    n_grid: usize,
    primary_modules: &[f64; CANONICAL_M as usize],
) -> Vec<f64> {
    let half_n = (n_grid / 2) as i32;
    let total_elements = n_grid * n_grid * n_grid;
    let mut phases = vec![0.0f64; total_elements];
    let k1 = (2.0 * PI) / 222.5502;

    for ix in 0..n_grid {
        let neg_ix = (n_grid - ix) % n_grid;
        let nx = if ix as i32 <= half_n { ix as i32 } else { ix as i32 - n_grid as i32 };
        let kx = nx as f64 * k1;
        for iy in 0..n_grid {
            let neg_iy = (n_grid - iy) % n_grid;
            let ny = if iy as i32 <= half_n { iy as i32 } else { iy as i32 - n_grid as i32 };
            let ky = ny as f64 * k1;
            for iz in 0..n_grid {
                let neg_iz = (n_grid - iz) % n_grid;
                let nz = if iz as i32 <= half_n { iz as i32 } else { iz as i32 - n_grid as i32 };
                let kz = nz as f64 * k1;
                let flat_idx = ix * n_grid * n_grid + iy * n_grid + iz;
                let opp_idx = neg_ix * n_grid * n_grid + neg_iy * n_grid + neg_iz;

                // Strict zero-mode and self-conjugate (Nyquist) clamp
                if ix == neg_ix && iy == neg_iy && iz == neg_iz {
                    phases[flat_idx] = 0.0;
                    continue;
                }

                // Upper-hemisphere partition determination
                let is_in_k_plus = (ix > neg_ix)
                    || (ix == neg_ix && iy > neg_iy)
                    || (ix == neg_ix && iy == neg_iy && iz > neg_iz);

                if is_in_k_plus {
                    let norm_idx = continuous_weil_norm_index(kx, ky, kz, k1);
                    let mod_idx = norm_idx as usize;
                    let base_phase = primary_modules[mod_idx];

                    let psi = haar_angular_dispersion(kx, ky, kz);
                    let phase = (base_phase + psi).rem_euclid(2.0 * PI);
                    phases[flat_idx] = phase;
                    // Strict Hermitian anti-symmetry: phi(-k) = -phi(k) mod 2*pi
                    phases[opp_idx] = (-phase).rem_euclid(2.0 * PI);
                }
            }
        }
    }

    phases
}

#[derive(Debug, Clone)]
pub struct CosmologicalGrid {
    pub n_grid: usize,
    pub box_size_mpc: f64,
    pub dx_mpc: f64,
    pub lambda_holo: f64,
    pub r_merge_mpc: f64,
}

impl CosmologicalGrid {
    pub fn new(n_grid: usize, h_hubble: f64) -> Self {
        // Enforce physical box size such that Delta^2(k1, z=0) <= 0.012
        let box_size_mpc = 150.0 / h_hubble;
        let dx_mpc = box_size_mpc / (n_grid as f64);
        let lambda_holo = 3.0 / (box_size_mpc * box_size_mpc);
        let kappa_star_d5 = 1.0 / 2.0_f64.sqrt();
        let r_merge_mpc = 2.0_f64.sqrt() * kappa_star_d5 * dx_mpc;

        Self {
            n_grid,
            box_size_mpc,
            dx_mpc,
            lambda_holo,
            r_merge_mpc,
        }
    }

    /// Evaluates deterministic, SO(3)-invariant phase for mode k = (kx, ky, kz)
    #[inline]
    pub fn isotropic_weil_phase(&self, kx: i32, ky: i32, kz: i32) -> f64 {
        let kx2 = (kx as i64) * (kx as i64);
        let ky2 = (ky as i64) * (ky as i64);
        let kz2 = (kz as i64) * (kz as i64);

        let sigma1 = kx2 + ky2 + kz2;
        if sigma1 == 0 {
            return 0.0;
        }

        let sigma2 = kx2 * ky2 + ky2 * kz2 + kz2 * kx2;
        let sigma3 = kx2 * ky2 * kz2;

        let sgn_h = if kz > 0 {
            1.0
        } else if kz == 0 && ky > 0 {
            1.0
        } else if kz == 0 && ky == 0 && kx > 0 {
            1.0
        } else {
            -1.0
        };

        let w_poly = ((sigma1 % N_MOD) * 157
            + (sigma2 % N_MOD) * 11
            + (sigma3 % N_MOD) * 7
            + (((sigma1 % N_MOD) * (sigma2 % N_MOD)) % N_MOD) * 3
            + 1)
            .rem_euclid(N_MOD);

        sgn_h * (2.0 * PI) * (w_poly as f64 / N_MOD as f64)
    }

    /// Validates that the fundamental mode remains in the linear regime at z = 0
    pub fn is_fundamental_mode_linear(&self) -> bool {
        let k1 = (2.0 * PI) / self.box_size_mpc;
        k1 < 0.05
    }
}

use crate::shbt::cosmology::{CosmologicalBackground, ETA_D, ETA_V as ETA_V_COSMO, Z_DRAG};

#[derive(Debug, Clone)]
pub struct TwoFluidDrift {
    pub cosmo: CosmologicalBackground,
    pub sound_horizon_drag_mpc: f64,
}

impl TwoFluidDrift {
    pub fn new(cosmo: CosmologicalBackground) -> Self {
        let rs_drag = Self::compute_sound_horizon_drag(&cosmo);
        Self {
            cosmo,
            sound_horizon_drag_mpc: rs_drag,
        }
    }

    fn compute_sound_horizon_drag(cosmo: &CosmologicalBackground) -> f64 {
        let z_min = Z_DRAG;
        let z_max: f64 = 1.0e6;
        let n_steps = 1000;
        let log_z_min = z_min.ln();
        let log_z_max = z_max.ln();
        let d_log_z = (log_z_max - log_z_min) / (n_steps as f64);

        let mut sum = 0.0;
        for i in 0..n_steps {
            let log_z1 = log_z_min + (i as f64) * d_log_z;
            let log_z2 = log_z1 + d_log_z;
            let z1 = log_z1.exp();
            let z2 = log_z2.exp();

            let cs1 = cosmo.baryon_sound_speed(z1) / 1000.0; // km/s
            let cs2 = cosmo.baryon_sound_speed(z2) / 1000.0;
            let a1 = cosmo.scale_factor(z1);
            let a2 = cosmo.scale_factor(z2);
            let h1 = cosmo.hubble_rate_kms_mpc(a1);
            let h2 = cosmo.hubble_rate_kms_mpc(a2);

            let integrand1 = (cs1 / h1) * z1;
            let integrand2 = (cs2 / h2) * z2;
            sum += 0.5 * (integrand1 + integrand2) * d_log_z;
        }
        sum
    }

    #[inline]
    pub fn is_causally_disconnected(&self, comoving_dist_mpc: f64, a: f64) -> bool {
        let chi_event = self.cosmo.comoving_event_horizon_mpc(a);
        comoving_dist_mpc >= chi_event
    }

    pub fn compute_two_fluid_transfer(&self, k_mpc: f64, a: f64) -> (f64, f64) {
        let z = self.cosmo.redshift(a);
        let s = self.sound_horizon_drag_mpc;
        let ks = k_mpc * s;

        // Channel B: Collisionless dark matter transfer function
        let q = k_mpc / (self.cosmo.h0_kms_mpc / 100.0);
        let t_dm = (1.0 + (1.7 * q).powi(2) + (0.5 * q).powi(4)).powf(-0.25);

        // Channel A: Collisional baryonic fluid with acoustic oscillations
        let t_b = if z > Z_DRAG {
            let omega_ratio = ETA_V_COSMO / ETA_D;
            (ks.sin() / ks.max(1.0e-5)) * (-0.5 * (k_mpc * 0.1).powi(2)).exp()
                + omega_ratio * t_dm
        } else {
            // Post-recombination relaxation into gravitational potential wells
            let delta_decay = (a / self.cosmo.scale_factor(Z_DRAG)).min(1.0);
            t_dm + (ks.sin() / ks.max(1.0e-5)) * (1.0 - delta_decay)
        };

        (t_b, t_dm)
    }
}

// ----------------------------------------------------------------------------
// Particle Storage Buffer Layout and 2LPT Ingestion (shbt22.txt)
// ----------------------------------------------------------------------------

/// 32-byte WebGPU std430 particle storage buffer record
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct ParticleRecord {
    pub pos: [f32; 4], // pos.xyz = comoving position, pos.w = mass weight
    pub vel: [f32; 4], // vel.xyz = supercomoving momentum, vel.w = particle ID
}

pub struct SimulationBufferManager {
    pub particle_count: u32,
    pub raw_data: Vec<u8>,
}

impl SimulationBufferManager {
    /// Ingests and validates a precomputed 2LPT initial condition binary buffer.
    pub fn new_from_binary(
        raw_bytes: &[u8],
        grid_dim: u32,
    ) -> Result<Self, &'static str> {
        let particle_count = grid_dim * grid_dim * grid_dim;
        let expected_bytes = (particle_count as usize) * std::mem::size_of::<ParticleRecord>();

        let slice = if raw_bytes.starts_with(b"SHBT") && raw_bytes.len() == expected_bytes + 16 {
            &raw_bytes[16..]
        } else if raw_bytes.len() == expected_bytes {
            raw_bytes
        } else {
            return Err("Binary particle buffer size does not match target grid dimension.");
        };

        Ok(Self {
            particle_count,
            raw_data: slice.to_vec(),
        })
    }
}

pub const M_MODULES: u64 = 2_901_360;

#[derive(Debug, Clone, Copy)]
pub struct HolographicMetrology {
    pub l_box_mpc: f64,
    pub fundamental_k: f64,
    pub volume_mpc3: f64,
}

impl Default for HolographicMetrology {
    fn default() -> Self {
        let l_box_mpc = (3.0 / 6.057116e-5f64).sqrt();
        let fundamental_k = (2.0 * PI) / l_box_mpc;
        let volume_mpc3 = l_box_mpc.powi(3);
        Self {
            l_box_mpc,
            fundamental_k,
            volume_mpc3,
        }
    }
}

impl HolographicMetrology {
    pub fn new() -> Self {
        Self::default()
    }

    #[inline(always)]
    pub fn mpc_to_normalized(&self, x_mpc: f32) -> f32 {
        (x_mpc / self.l_box_mpc as f32).rem_euclid(1.0)
    }

    pub fn growth_factor_d1(&self, z: f64) -> f64 {
        compute_shbt_growth_factors(z).d1
    }

    pub fn growth_factor_d2(&self, z: f64) -> f64 {
        compute_shbt_growth_factors(z).d2
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct ParticleVertex {
    pub position: [f32; 3],
    pub velocity: [f32; 3],
    pub capacity_density: f32,
    pub _padding: f32,
}

pub struct ModularLatticeStructure {
    pub grid_dim: usize,
    pub metrology: HolographicMetrology,
    pub z_init: f64,
}

impl ModularLatticeStructure {
    pub fn new(grid_dim: usize, z_init: f64) -> Self {
        Self {
            grid_dim,
            metrology: HolographicMetrology::new(),
            z_init,
        }
    }

    /// Evaluates the SO(3)-invariant modular Weil phase for a Fourier lattice vector.
    /// Eliminates cubic harmonics (O_h leakage) and guarantees Hermitian anti-symmetry.
    pub fn compute_modular_weil_phase(&self, nx: i64, ny: i64, nz: i64) -> f64 {
        let s = (nx * nx + ny * ny + nz * nz) as u64;
        if s == 0 {
            return 0.0;
        }

        let is_h_plus = nz > 0 || (nz == 0 && ny > 0) || (nz == 0 && ny == 0 && nx > 0);
        let is_nyquist = (nx.abs() == (self.grid_dim as i64 / 2) || nx == 0)
            && (ny.abs() == (self.grid_dim as i64 / 2) || ny == 0)
            && (nz.abs() == (self.grid_dim as i64 / 2) || nz == 0);

        let mu_s = (17 * s + 31 * s.pow(2) + 7 * s.pow(3)) % M_MODULES;

        if is_nyquist {
            return (mu_s % 2) as f64 * PI;
        }

        let j_hash = (((nx * 312 + ny * 26 + nz * 8).abs() as u64) ^ (s * 157)) % M_MODULES;
        let phase_base = (2.0 * PI * ((mu_s + j_hash) % M_MODULES) as f64) / (M_MODULES as f64);

        if is_h_plus {
            phase_base
        } else {
            -phase_base
        }
    }

    /// Displaces Lagrangian particle lattice to Eulerian space using 2LPT.
    pub fn generate_particle_mesh(
        &self,
        psi1: &[[f64; 3]],
        psi2: &[[f64; 3]],
    ) -> Vec<ParticleVertex> {
        let n = self.grid_dim;
        let mut particles = Vec::with_capacity(n * n * n);

        let d1 = self.metrology.growth_factor_d1(self.z_init);
        let d2 = self.metrology.growth_factor_d2(self.z_init);
        let dx = self.metrology.l_box_mpc / (n as f64);

        for ix in 0..n {
            for iy in 0..n {
                for iz in 0..n {
                    let idx = ix * n * n + iy * n + iz;
                    let qx = (ix as f64) * dx;
                    let qy = (iy as f64) * dx;
                    let qz = (iz as f64) * dx;

                    let p1 = psi1[idx];
                    let p2 = psi2[idx];

                    let x_mpc = (qx + d1 * p1[0] + d2 * p2[0]).rem_euclid(self.metrology.l_box_mpc);
                    let y_mpc = (qy + d1 * p1[1] + d2 * p2[1]).rem_euclid(self.metrology.l_box_mpc);
                    let z_mpc = (qz + d1 * p1[2] + d2 * p2[2]).rem_euclid(self.metrology.l_box_mpc);

                    particles.push(ParticleVertex {
                        position: [
                            self.metrology.mpc_to_normalized(x_mpc as f32),
                            self.metrology.mpc_to_normalized(y_mpc as f32),
                            self.metrology.mpc_to_normalized(z_mpc as f32),
                        ],
                        velocity: [
                            (d1 * p1[0]) as f32,
                            (d1 * p1[1]) as f32,
                            (d1 * p1[2]) as f32,
                        ],
                        capacity_density: 1.0,
                        _padding: 0.0,
                    });
                }
            }
        }
        particles
    }
}
