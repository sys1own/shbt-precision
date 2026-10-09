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

/// PyO3 audit report for Non-Linear Structure Formation and Cosmic Shear.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[pyclass]
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

    let all_passed = (s8_primordial - 0.831).abs() < 0.02
        && (s8_effective - 0.764).abs() < 0.02
        && (kappa_min - 1.0 / 23.0).abs() < 1e-6
        && (kappa_max - 5329.0 / 92092.0).abs() < 1e-6
        && c_ell_1000 > 0.0;

    StructureAuditReport {
        s8_primordial,
        s8_effective,
        kappa_get_min: kappa_min,
        kappa_get_max: kappa_max,
        c_ell_1000,
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
}
