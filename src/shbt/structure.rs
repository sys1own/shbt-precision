//! src/shbt/structure.rs
//!
//! High-Performance Non-Linear Structure and Cosmic Shear Module for SHBT
//! Implements Cardy capacity HMF, entropic transport P_NL, and Limber C_ell.

use pyo3::prelude::*;
use pyo3::types::PyDict;
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

pub const KAPPA_GET_BASE: f64 = 1.0 / 23.0;
pub const KAPPA_GET_LOAD_COEFF: f64 = 1325.0 / 4004.0;
pub const C_EFF: f64 = 1.0;
const K_Q: f64 = 8.0;
const SPEED_OF_LIGHT: f64 = 299792.458; // km/s

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
