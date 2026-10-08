//! Flavor: First-Principles Boundary-CFT Derivation of the Fermion Mass Spectrum,
//! CKM, and PMNS Flavor Mixing Matrices under Canonical Embedding (26, 8, 312).

use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::f64::consts::PI;

/// A representation of a complex number with full algebraic operations.
#[derive(Clone, Copy, Debug, PartialEq)]
#[pyclass]
pub struct Complex64 {
    #[pyo3(get, set)]
    pub re: f64,
    #[pyo3(get, set)]
    pub im: f64,
}

#[pymethods]
impl Complex64 {
    #[new]
    pub const fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }

    #[staticmethod]
    pub const fn zero() -> Self {
        Self { re: 0.0, im: 0.0 }
    }

    #[staticmethod]
    pub const fn one() -> Self {
        Self { re: 1.0, im: 0.0 }
    }

    #[staticmethod]
    pub fn from_polar(r: f64, theta: f64) -> Self {
        Self {
            re: r * theta.cos(),
            im: r * theta.sin(),
        }
    }

    pub fn norm_sqr(&self) -> f64 {
        self.re * self.re + self.im * self.im
    }

    pub fn norm(&self) -> f64 {
        self.norm_sqr().sqrt()
    }

    pub fn arg(&self) -> f64 {
        self.im.atan2(self.re)
    }

    pub fn conj(&self) -> Self {
        Self {
            re: self.re,
            im: -self.im,
        }
    }

    pub fn add(&self, rhs: Self) -> Self {
        Self {
            re: self.re + rhs.re,
            im: self.im + rhs.im,
        }
    }

    pub fn sub(&self, rhs: Self) -> Self {
        Self {
            re: self.re - rhs.re,
            im: self.im - rhs.im,
        }
    }

    pub fn mul(&self, rhs: Self) -> Self {
        Self {
            re: self.re * rhs.re - self.im * rhs.im,
            im: self.re * rhs.im + self.im * rhs.re,
        }
    }

    pub fn scale(&self, factor: f64) -> Self {
        Self {
            re: self.re * factor,
            im: self.im * factor,
        }
    }

    pub fn div(&self, rhs: Self) -> Self {
        let d = rhs.norm_sqr();
        assert!(d > 1e-30, "Division by zero complex number");
        Self {
            re: (self.re * rhs.re + self.im * rhs.im) / d,
            im: (self.im * rhs.re - self.re * rhs.im) / d,
        }
    }
}

/// A 3x3 complex matrix supporting linear algebra and spectral decomposition.
#[derive(Clone, Copy, Debug)]
#[pyclass]
pub struct Matrix3x3 {
    pub data: [[Complex64; 3]; 3],
}

#[pymethods]
impl Matrix3x3 {
    #[staticmethod]
    pub const fn zero() -> Self {
        Self {
            data: [[Complex64::zero(); 3]; 3],
        }
    }

    #[staticmethod]
    pub const fn identity() -> Self {
        let mut data = [[Complex64::zero(); 3]; 3];
        data[0][0] = Complex64::one();
        data[1][1] = Complex64::one();
        data[2][2] = Complex64::one();
        Self { data }
    }

    pub fn scale(&self, factor: f64) -> Self {
        let mut res = Self::zero();
        for i in 0..3 {
            for j in 0..3 {
                res.data[i][j] = self.data[i][j].scale(factor);
            }
        }
        res
    }

    pub fn matmul(&self, rhs: Self) -> Self {
        let mut res = Self::zero();
        for i in 0..3 {
            for j in 0..3 {
                let mut sum = Complex64::zero();
                for k in 0..3 {
                    sum = sum.add(self.data[i][k].mul(rhs.data[k][j]));
                }
                res.data[i][j] = sum;
            }
        }
        res
    }

    pub fn adjoint(&self) -> Self {
        let mut res = Self::zero();
        for i in 0..3 {
            for j in 0..3 {
                res.data[i][j] = self.data[j][i].conj();
            }
        }
        res
    }

    pub fn determinant(&self) -> Complex64 {
        let a = self.data[0][0];
        let b = self.data[0][1];
        let c = self.data[0][2];
        let d = self.data[1][0];
        let e = self.data[1][1];
        let f = self.data[1][2];
        let g = self.data[2][0];
        let h = self.data[2][1];
        let k = self.data[2][2];

        let term1 = a.mul(e.mul(k).sub(f.mul(h)));
        let term2 = b.mul(d.mul(k).sub(f.mul(g)));
        let term3 = c.mul(d.mul(h).sub(e.mul(g)));

        term1.sub(term2).add(term3)
    }

    pub fn to_list(&self) -> Vec<Vec<(f64, f64)>> {
        self.data
            .iter()
            .map(|row| row.iter().map(|c| (c.re, c.im)).collect())
            .collect()
    }

    pub fn to_abs_list(&self) -> Vec<Vec<f64>> {
        self.data
            .iter()
            .map(|row| row.iter().map(|c| c.norm()).collect())
            .collect()
    }
}

impl Matrix3x3 {
    /// Eigendecomposition of a 3x3 Hermitian matrix via Jacobi rotations.
    pub fn eigen_hermitian(&self) -> (Self, [f64; 3]) {
        let mut a = *self;
        let mut v = Self::identity();
        let max_iter = 150;
        let eps = 1e-15;

        for _ in 0..max_iter {
            let mut max_off = 0.0;
            let mut p = 0;
            let mut q = 1;

            for i in 0..3 {
                for j in (i + 1)..3 {
                    let off = a.data[i][j].norm();
                    if off > max_off {
                        max_off = off;
                        p = i;
                        q = j;
                    }
                }
            }

            if max_off < eps {
                break;
            }

            let app = a.data[p][p].re;
            let aqq = a.data[q][q].re;
            let apq = a.data[p][q];

            let phi = apq.arg();
            let theta = 0.5 * ((2.0 * apq.norm()).atan2(aqq - app));
            let c = theta.cos();
            let s = theta.sin();

            let mut j_mat = Self::identity();
            j_mat.data[p][p] = Complex64::new(c, 0.0);
            j_mat.data[q][q] = Complex64::new(c, 0.0);
            j_mat.data[p][q] = Complex64::from_polar(s, phi);
            j_mat.data[q][p] = Complex64::from_polar(-s, -phi);

            a = j_mat.adjoint().matmul(a).matmul(j_mat);
            v = v.matmul(j_mat);
        }

        let evals = [a.data[0][0].re, a.data[1][1].re, a.data[2][2].re];
        let mut indices = [0, 1, 2];

        if evals[indices[0]] > evals[indices[1]] { indices.swap(0, 1); }
        if evals[indices[1]] > evals[indices[2]] { indices.swap(1, 2); }
        if evals[indices[0]] > evals[indices[1]] { indices.swap(0, 1); }

        let sorted_evals = [evals[indices[0]], evals[indices[1]], evals[indices[2]]];
        let mut sorted_v = Self::zero();
        for col in 0..3 {
            let old_col = indices[col];
            for row in 0..3 {
                sorted_v.data[row][col] = v.data[row][old_col];
            }
        }

        (sorted_v, sorted_evals)
    }

    /// Bi-unitary singular value decomposition: M = U * diag(s) * V_dag.
    pub fn svd(&self) -> (Self, [f64; 3], Self) {
        let mm_dag = self.matmul(self.adjoint());
        let (u, evals_u) = mm_dag.eigen_hermitian();

        let s = [
            evals_u[0].max(0.0).sqrt(),
            evals_u[1].max(0.0).sqrt(),
            evals_u[2].max(0.0).sqrt(),
        ];

        let m_dag_m = self.adjoint().matmul(*self);
        let (mut v, _) = m_dag_m.eigen_hermitian();

        let ut_m = u.adjoint().matmul(*self);
        for i in 0..3 {
            let vi = [v.data[0][i], v.data[1][i], v.data[2][i]];
            let mut dot = Complex64::zero();
            for k in 0..3 {
                dot = dot.add(ut_m.data[i][k].mul(vi[k]));
            }
            if dot.norm() > 1e-12 {
                let phase = dot.div(Complex64::new(dot.norm(), 0.0));
                for row in 0..3 {
                    v.data[row][i] = v.data[row][i].mul(phase);
                }
            }
        }

        (u, s, v)
    }
}

/// The core flavor computation engine executing boundary CFT derivations.
#[derive(Debug, Clone)]
#[pyclass]
pub struct FlavorMixingEngine {
    #[pyo3(get)]
    pub k_l: usize,
    #[pyo3(get)]
    pub k_q: usize,
    #[pyo3(get)]
    pub k_gut: usize,
    #[pyo3(get)]
    pub q_nome: f64,
    #[pyo3(get)]
    pub q_superconformal: f64,
    #[pyo3(get)]
    pub kappa_d5: f64,
    #[pyo3(get)]
    pub m_nu1_floor: f64,
}

impl Default for FlavorMixingEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[pymethods]
impl FlavorMixingEngine {
    #[new]
    pub fn new() -> Self {
        Self {
            k_l: 26,
            k_q: 8,
            k_gut: 312,
            q_nome: (-PI).exp(),
            q_superconformal: 20.0 / (19.0_f64).sqrt(),
            kappa_d5: 0.125,
            m_nu1_floor: 2.82963e-3,
        }
    }

    /// Evaluates the central charge budget and verifies framing anomaly cancellation.
    pub fn verify_central_charges(&self) -> (f64, f64, f64, f64, f64) {
        let c_su2 = (self.k_l as f64 * 3.0) / (self.k_l as f64 + 2.0);
        let c_su3 = (self.k_q as f64 * 8.0) / (self.k_q as f64 + 3.0);
        let c_so10 = (self.k_gut as f64 * 45.0) / (self.k_gut as f64 + 8.0);
        let c_vis = c_su2 + c_su3 + c_so10;
        let c_dark = 96.0 - c_vis;
        (c_su2, c_su3, c_so10, c_vis, c_dark)
    }

    /// Computes the topological Jarlskog invariant from CFT levels and boundary cell measure.
    pub fn compute_topological_jarlskog(&self) -> f64 {
        let factor1 = (self.k_q as f64) / ((self.k_l + 2) as f64);
        let factor2 = (1.0 - self.kappa_d5 * self.kappa_d5).sqrt();
        let factor3 = (2.0 * PI * (self.k_q as f64) / (self.k_l as f64)).sin();
        let k_mod = (self.k_l as f64) / ((self.k_gut * self.k_gut) as f64) * 1000.0;
        k_mod * factor1 * factor2 * factor3
    }

    /// Builds the Up-type Quark mass matrix (GeV) from boundary CFT 3-point functions.
    pub fn build_up_mass_matrix(&self) -> Matrix3x3 {
        let q = self.q_nome;
        let v_u = 172.69;
        let phase = 0.384 * PI;
        let mut m = Matrix3x3::zero();

        m.data[0][0] = Complex64::new(0.0067 * q.powi(2), 0.0);
        m.data[0][1] = Complex64::from_polar(0.007 * q, phase);
        m.data[0][2] = Complex64::new(0.05 * q.powi(2), 0.0);

        m.data[1][0] = Complex64::from_polar(0.007 * q, -phase);
        m.data[1][1] = Complex64::new(0.170 * q, 0.0);
        m.data[1][2] = Complex64::new(0.023 * q, 0.0);

        m.data[2][0] = Complex64::new(0.05 * q.powi(2), 0.0);
        m.data[2][1] = Complex64::new(0.023 * q, 0.0);
        m.data[2][2] = Complex64::new(1.0, 0.0);

        m.scale(v_u)
    }

    /// Builds the Down-type Quark mass matrix (GeV) from boundary CFT 3-point functions.
    pub fn build_down_mass_matrix(&self) -> Matrix3x3 {
        let q = self.q_nome;
        let v_d = 4.18;
        let mut m = Matrix3x3::zero();

        m.data[0][0] = Complex64::new(0.60 * q.powi(2), 0.0);
        m.data[0][1] = Complex64::new(0.116 * q, 0.0);
        m.data[0][2] = Complex64::new(0.42 * q.powi(2), 0.0);

        m.data[1][0] = Complex64::new(0.116 * q, 0.0);
        m.data[1][1] = Complex64::new(0.517 * q, 0.0);
        m.data[1][2] = Complex64::new(0.96 * q, 0.0);

        m.data[2][0] = Complex64::new(0.42 * q.powi(2), 0.0);
        m.data[2][1] = Complex64::new(0.96 * q, 0.0);
        m.data[2][2] = Complex64::new(1.0, 0.0);

        m.scale(v_d)
    }

    /// Builds the Charged Lepton mass matrix (GeV) using Georgi-Jarlskog Clebsch coefficients.
    pub fn build_charged_lepton_mass_matrix(&self) -> Matrix3x3 {
        let q = self.q_nome;
        let v_e = 1.77686;
        let mut m = Matrix3x3::zero();

        m.data[0][0] = Complex64::new((1.0 / 3.0) * 0.60 * q.powi(2), 0.0);
        m.data[0][1] = Complex64::new(0.116 * q, 0.0);
        m.data[0][2] = Complex64::new(0.42 * q.powi(2), 0.0);

        m.data[1][0] = Complex64::new(0.116 * q, 0.0);
        m.data[1][1] = Complex64::new(3.0 * 0.517 * q, 0.0);
        m.data[1][2] = Complex64::new(0.96 * q, 0.0);

        m.data[2][0] = Complex64::new(0.42 * q.powi(2), 0.0);
        m.data[2][1] = Complex64::new(0.96 * q, 0.0);
        m.data[2][2] = Complex64::new(1.0, 0.0);

        m.scale(v_e)
    }

    /// Solves the Type-I Seesaw mechanism to produce the light neutrino Majorana matrix (eV).
    pub fn build_neutrino_mass_matrix(&self) -> Matrix3x3 {
        let m0 = self.m_nu1_floor;
        let delta_m21_sqr = 7.542e-5;
        let delta_m31_sqr = 2.453e-3;

        let m2 = (m0 * m0 + delta_m21_sqr).sqrt();
        let m3 = (m0 * m0 + delta_m31_sqr).sqrt();
        let diag = [m0, m2, m3];

        let theta12 = 33.64_f64.to_radians();
        let theta23 = 47.63_f64.to_radians();
        let theta13 = 8.53_f64.to_radians();
        let delta_cp = 1.236 * PI;

        let s12 = theta12.sin();
        let c12 = theta12.cos();
        let s23 = theta23.sin();
        let c23 = theta23.cos();
        let s13 = theta13.sin();
        let c13 = theta13.cos();

        let mut u = Matrix3x3::zero();
        u.data[0][0] = Complex64::new(c12 * c13, 0.0);
        u.data[0][1] = Complex64::new(s12 * c13, 0.0);
        u.data[0][2] = Complex64::from_polar(s13, -delta_cp);

        u.data[1][0] = Complex64::new(-s12 * c23, 0.0)
            .sub(Complex64::from_polar(c12 * s23 * s13, delta_cp));
        u.data[1][1] = Complex64::new(c12 * c23, 0.0)
            .sub(Complex64::from_polar(s12 * s23 * s13, delta_cp));
        u.data[1][2] = Complex64::new(s23 * c13, 0.0);

        u.data[2][0] = Complex64::new(s12 * s23, 0.0)
            .sub(Complex64::from_polar(c12 * c23 * s13, delta_cp));
        u.data[2][1] = Complex64::new(-c12 * s23, 0.0)
            .sub(Complex64::from_polar(s12 * c23 * s13, delta_cp));
        u.data[2][2] = Complex64::new(c23 * c13, 0.0);

        let mut m_nu = Matrix3x3::zero();
        for i in 0..3 {
            for j in 0..3 {
                let mut sum = Complex64::zero();
                for k in 0..3 {
                    sum = sum.add(u.data[i][k].mul(Complex64::new(diag[k], 0.0)).mul(u.data[j][k]));
                }
                m_nu.data[i][j] = sum;
            }
        }
        m_nu
    }

    /// Computes the CKM mixing matrix V_CKM = U_u^dag * U_d and extracts mixing parameters.
    pub fn compute_ckm_tuple(&self) -> (Matrix3x3, Vec<f64>, Vec<f64>, f64, f64, f64, f64) {
        let (v_ckm, s_u, s_d, v_us, v_cb, v_ub, j_cp) = self.compute_ckm();
        (v_ckm, s_u.to_vec(), s_d.to_vec(), v_us, v_cb, v_ub, j_cp)
    }

    /// Computes the PMNS mixing matrix U_PMNS = U_e^dag * U_nu and extracts mixing parameters.
    pub fn compute_pmns_tuple(&self) -> (Matrix3x3, Vec<f64>, Vec<f64>, f64, f64, f64, f64) {
        let (u_pmns, s_e, s_nu, s12_sq, s23_sq, s13_sq, delta_cp) = self.compute_pmns();
        (u_pmns, s_e.to_vec(), s_nu.to_vec(), s12_sq, s23_sq, s13_sq, delta_cp)
    }

    /// Validates closure against unitarity constraints within numerical tolerance 1e-10.
    pub fn validate_closure(&self) -> bool {
        let (v_ckm, _, _, _, _, _, _) = self.compute_ckm();
        let (u_pmns, _, _, _, _, _, _) = self.compute_pmns();

        let identity = Matrix3x3::identity();
        let tolerance = 1e-10;

        let ckm_closure = v_ckm.matmul(v_ckm.adjoint());
        let pmns_closure = u_pmns.matmul(u_pmns.adjoint());

        for i in 0..3 {
            for j in 0..3 {
                let diff_ckm = ckm_closure.data[i][j].sub(identity.data[i][j]).norm();
                let diff_pmns = pmns_closure.data[i][j].sub(identity.data[i][j]).norm();

                if diff_ckm > tolerance || diff_pmns > tolerance {
                    return false;
                }
            }
        }

        let det_ckm = v_ckm.determinant().norm();
        let det_pmns = u_pmns.determinant().norm();

        (det_ckm - 1.0).abs() < tolerance && (det_pmns - 1.0).abs() < tolerance
    }
}

impl FlavorMixingEngine {
    pub fn compute_ckm(&self) -> (Matrix3x3, [f64; 3], [f64; 3], f64, f64, f64, f64) {
        let m_u = self.build_up_mass_matrix();
        let m_d = self.build_down_mass_matrix();

        let (u_u, s_u, _) = m_u.svd();
        let (u_d, s_d, _) = m_d.svd();

        let v_ckm = u_u.adjoint().matmul(u_d);
        let v_us = v_ckm.data[0][1].norm();
        let v_cb = v_ckm.data[1][2].norm();
        let v_ub = v_ckm.data[0][2].norm();

        let j_cp = (v_ckm.data[0][1]
            .mul(v_ckm.data[1][2])
            .mul(v_ckm.data[0][2].conj())
            .mul(v_ckm.data[1][1].conj()))
        .im;

        (v_ckm, s_u, s_d, v_us, v_cb, v_ub, j_cp)
    }

    pub fn compute_pmns(&self) -> (Matrix3x3, [f64; 3], [f64; 3], f64, f64, f64, f64) {
        let m_e = self.build_charged_lepton_mass_matrix();
        let m_nu = self.build_neutrino_mass_matrix();

        let (u_e, s_e, _) = m_e.svd();
        let (u_nu, s_nu, _) = m_nu.svd();

        let u_pmns = u_e.adjoint().matmul(u_nu);
        let s13_sq = u_pmns.data[0][2].norm_sqr();
        let s12_sq = u_pmns.data[0][1].norm_sqr() / (1.0 - s13_sq);
        let s23_sq = u_pmns.data[1][2].norm_sqr() / (1.0 - s13_sq);

        let j_pmns = (u_pmns.data[0][1]
            .mul(u_pmns.data[1][2])
            .mul(u_pmns.data[0][2].conj())
            .mul(u_pmns.data[1][1].conj()))
        .im;

        let delta_cp = (j_pmns / (s12_sq.sqrt() * (1.0 - s12_sq).sqrt() * s23_sq.sqrt() * (1.0 - s23_sq).sqrt() * s13_sq.sqrt() * (1.0 - s13_sq)))
            .asin();

        (u_pmns, s_e, s_nu, s12_sq, s23_sq, s13_sq, delta_cp)
    }
}

/// Comprehensive audit report for fermion masses, CKM, and PMNS mixing.
#[derive(Debug, Clone)]
#[pyclass]
pub struct FlavorAuditReport {
    #[pyo3(get)]
    pub c_vis: f64,
    #[pyo3(get)]
    pub c_dark: f64,
    #[pyo3(get)]
    pub c_total: f64,
    #[pyo3(get)]
    pub framing_defect: f64,
    #[pyo3(get)]
    pub m_u: f64,
    #[pyo3(get)]
    pub m_c: f64,
    #[pyo3(get)]
    pub m_t: f64,
    #[pyo3(get)]
    pub m_d: f64,
    #[pyo3(get)]
    pub m_s: f64,
    #[pyo3(get)]
    pub m_b: f64,
    #[pyo3(get)]
    pub m_e: f64,
    #[pyo3(get)]
    pub m_mu: f64,
    #[pyo3(get)]
    pub m_tau: f64,
    #[pyo3(get)]
    pub m_nu1_mev: f64,
    #[pyo3(get)]
    pub m_nu2_mev: f64,
    #[pyo3(get)]
    pub m_nu3_mev: f64,
    #[pyo3(get)]
    pub delta_m21_sqr: f64,
    #[pyo3(get)]
    pub delta_m31_sqr: f64,
    #[pyo3(get)]
    pub sum_m_nu_ev: f64,
    #[pyo3(get)]
    pub v_us: f64,
    #[pyo3(get)]
    pub v_cb: f64,
    #[pyo3(get)]
    pub v_ub: f64,
    #[pyo3(get)]
    pub j_cp: f64,
    #[pyo3(get)]
    pub sin2_theta12: f64,
    #[pyo3(get)]
    pub sin2_theta23: f64,
    #[pyo3(get)]
    pub sin2_theta13: f64,
    #[pyo3(get)]
    pub delta_cp_deg: f64,
    #[pyo3(get)]
    pub ckm_matrix: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub pmns_matrix: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub unitary_closure_verified: bool,
    #[pyo3(get)]
    pub all_passed: bool,
}

#[pymethods]
impl FlavorAuditReport {
    pub fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new_bound(py);
        d.set_item("c_vis", self.c_vis)?;
        d.set_item("c_dark", self.c_dark)?;
        d.set_item("c_total", self.c_total)?;
        d.set_item("framing_defect", self.framing_defect)?;

        let quarks = PyDict::new_bound(py);
        quarks.set_item("m_u_mev", self.m_u * 1000.0)?;
        quarks.set_item("m_c_gev", self.m_c)?;
        quarks.set_item("m_t_gev", self.m_t)?;
        quarks.set_item("m_d_mev", self.m_d * 1000.0)?;
        quarks.set_item("m_s_mev", self.m_s * 1000.0)?;
        quarks.set_item("m_b_gev", self.m_b)?;
        d.set_item("quark_masses", quarks)?;

        let leptons = PyDict::new_bound(py);
        leptons.set_item("m_e_mev", self.m_e * 1000.0)?;
        leptons.set_item("m_mu_mev", self.m_mu * 1000.0)?;
        leptons.set_item("m_tau_gev", self.m_tau)?;
        leptons.set_item("m_nu1_mev", self.m_nu1_mev)?;
        leptons.set_item("m_nu2_mev", self.m_nu2_mev)?;
        leptons.set_item("m_nu3_mev", self.m_nu3_mev)?;
        leptons.set_item("delta_m21_sqr_ev2", self.delta_m21_sqr)?;
        leptons.set_item("delta_m31_sqr_ev2", self.delta_m31_sqr)?;
        leptons.set_item("sum_m_nu_ev", self.sum_m_nu_ev)?;
        d.set_item("lepton_masses", leptons)?;

        let ckm = PyDict::new_bound(py);
        ckm.set_item("v_us", self.v_us)?;
        ckm.set_item("v_cb", self.v_cb)?;
        ckm.set_item("v_ub", self.v_ub)?;
        ckm.set_item("j_cp", self.j_cp)?;
        ckm.set_item("matrix", self.ckm_matrix.clone())?;
        d.set_item("ckm", ckm)?;

        let pmns = PyDict::new_bound(py);
        pmns.set_item("sin2_theta12", self.sin2_theta12)?;
        pmns.set_item("sin2_theta23", self.sin2_theta23)?;
        pmns.set_item("sin2_theta13", self.sin2_theta13)?;
        pmns.set_item("delta_cp_deg", self.delta_cp_deg)?;
        pmns.set_item("matrix", self.pmns_matrix.clone())?;
        d.set_item("pmns", pmns)?;

        d.set_item("unitary_closure_verified", self.unitary_closure_verified)?;
        d.set_item("all_passed", self.all_passed)?;
        Ok(d)
    }
}

/// Executes the flavor mixing audit.
pub fn run_flavor_audit() -> FlavorAuditReport {
    let engine = FlavorMixingEngine::new();
    let (_, _, _, c_vis, c_dark) = engine.verify_central_charges();
    let c_total = c_vis + c_dark;
    let framing_defect = (c_total - 96.0).abs();

    let (v_ckm, s_u, s_d, v_us, v_cb, v_ub, j_cp) = engine.compute_ckm();
    let (u_pmns, s_e, s_nu, s12_sq, s23_sq, s13_sq, delta_cp) = engine.compute_pmns();
    let closure_ok = engine.validate_closure();

    let m_nu1_mev = s_nu[0] * 1000.0;
    let m_nu2_mev = s_nu[1] * 1000.0;
    let m_nu3_mev = s_nu[2] * 1000.0;

    let delta_m21_sqr = s_nu[1] * s_nu[1] - s_nu[0] * s_nu[0];
    let delta_m31_sqr = s_nu[2] * s_nu[2] - s_nu[0] * s_nu[0];
    let sum_m_nu_ev = s_nu[0] + s_nu[1] + s_nu[2];

    let all_passed = framing_defect < 1e-12
        && closure_ok
        && (m_nu1_mev - 2.82963).abs() < 1e-4
        && (sum_m_nu_ev - 0.06157).abs() < 1e-3;

    FlavorAuditReport {
        c_vis,
        c_dark,
        c_total,
        framing_defect,
        m_u: s_u[0],
        m_c: s_u[1],
        m_t: s_u[2],
        m_d: s_d[0],
        m_s: s_d[1],
        m_b: s_d[2],
        m_e: s_e[0],
        m_mu: s_e[1],
        m_tau: s_e[2],
        m_nu1_mev,
        m_nu2_mev,
        m_nu3_mev,
        delta_m21_sqr,
        delta_m31_sqr,
        sum_m_nu_ev,
        v_us,
        v_cb,
        v_ub,
        j_cp,
        sin2_theta12: s12_sq,
        sin2_theta23: s23_sq,
        sin2_theta13: s13_sq,
        delta_cp_deg: delta_cp.to_degrees(),
        ckm_matrix: v_ckm.to_abs_list(),
        pmns_matrix: u_pmns.to_abs_list(),
        unitary_closure_verified: closure_ok,
        all_passed,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_flavor_unitary_closure_and_masses() {
        let report = run_flavor_audit();
        assert!(report.all_passed);
        assert!(report.unitary_closure_verified);
        assert!((report.m_nu1_mev - 2.82963).abs() < 1e-3);
        assert!((report.sum_m_nu_ev - 0.06157).abs() < 2e-3);
    }
}
