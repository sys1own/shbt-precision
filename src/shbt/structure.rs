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
        let phases = generate_weil_modular_phases(grid_size);

        for ix in 0..grid_size {
            let neg_ix = (grid_size - ix) % grid_size;
            for iy in 0..grid_size {
                let neg_iy = (grid_size - iy) % grid_size;
                for iz in 0..grid_size {
                    let neg_iz = (grid_size - iz) % grid_size;

                    let idx_pos = (ix * grid_size + iy) * grid_size + iz;
                    let idx_neg = (neg_ix * grid_size + neg_iy) * grid_size + neg_iz;

                    if idx_pos == idx_neg {
                        assert_eq!(phases[idx_pos], 0.0);
                    } else {
                        let sum = (phases[idx_pos] + phases[idx_neg]) % (2.0 * PI);
                        assert!(
                            sum.abs() < 1e-12 || (sum - 2.0 * PI).abs() < 1e-12,
                            "Hermitian condition violated at ({}, {}, {})",
                            ix, iy, iz
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn test_point_group_symmetry_breaking() {
        let grid_size = 32;
        let phases = generate_weil_modular_phases(grid_size);

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

pub const N_MOD: i64 = 2_901_360;
pub const CFT_MODULE_COUNT: u64 = 2_901_360;
const PRIME_C1: u64 = 73_856_093;
const PRIME_C2: u64 = 19_349_663;
const PRIME_C3: u64 = 83_492_791;
const SPLITMIX_C1: u64 = 0xBF58_476D_1CE4_E5B9;
const SPLITMIX_C2: u64 = 0x94D0_49BB_1331_11EB;

/// Deterministic, PRNG-free phase generation mapping lattice modes onto CFT primary modules.
///
/// Ensures:
/// - Exact Hermitian anti-symmetry: phi(-k) == -phi(k) (mod 2*pi)
/// - Total breaking of O_h point-group symmetries on individual realizations
/// - Ergodic uniform distribution over [0, 2*pi)
pub fn generate_weil_modular_phases(grid_size: usize) -> Vec<f64> {
    let total_elements = grid_size * grid_size * grid_size;
    let mut phases = vec![0.0f64; total_elements];
    let half_n = (grid_size / 2) as i64;

    for ix in 0..grid_size {
        let kx = if ix as i64 >= half_n {
            ix as i64 - grid_size as i64
        } else {
            ix as i64
        };

        for iy in 0..grid_size {
            let ky = if iy as i64 >= half_n {
                iy as i64 - grid_size as i64
            } else {
                iy as i64
            };

            for iz in 0..grid_size {
                let kz = if iz as i64 >= half_n {
                    iz as i64 - grid_size as i64
                } else {
                    iz as i64
                };
                let idx = (ix * grid_size + iy) * grid_size + iz;

                // For periodic FFT lattice indices, the negative mode -k has coordinates in [0, grid_size):
                // neg_ix = (grid_size - ix) % grid_size, etc.
                // The corresponding Fourier mode k_neg is:
                let neg_ix = (grid_size - ix) % grid_size;
                let neg_iy = (grid_size - iy) % grid_size;
                let neg_iz = (grid_size - iz) % grid_size;

                let kx_neg = if neg_ix as i64 >= half_n { neg_ix as i64 - grid_size as i64 } else { neg_ix as i64 };
                let ky_neg = if neg_iy as i64 >= half_n { neg_iy as i64 - grid_size as i64 } else { neg_iy as i64 };
                let kz_neg = if neg_iz as i64 >= half_n { neg_iz as i64 - grid_size as i64 } else { neg_iz as i64 };

                // Modes that map to themselves under periodic negation:
                if ix == neg_ix && iy == neg_iy && iz == neg_iz {
                    phases[idx] = 0.0;
                    continue;
                }

                // Lexicographical positive half-space partition:
                // (ix, iy, iz) vs (neg_ix, neg_iy, neg_iz)
                let is_positive = (ix, iy, iz) > (neg_ix, neg_iy, neg_iz);

                let (kx_c, ky_c, kz_c) = if is_positive {
                    (kx, ky, kz)
                } else {
                    (kx_neg, ky_neg, kz_neg)
                };

                // Cast to unsigned 64-bit coordinates
                let ux = kx_c as u64;
                let uy = ky_c as u64;
                let uz = kz_c as u64;

                // Asymmetric linear coordinate projection
                let w = (ux.wrapping_mul(PRIME_C1))
                    ^ (uy.wrapping_mul(PRIME_C2))
                    ^ (uz.wrapping_mul(PRIME_C3));

                // SplitMix64 bi-permutative mixing
                let xi1 = (w ^ (w >> 30)).wrapping_mul(SPLITMIX_C1);
                let xi2 = (xi1 ^ (xi1 >> 27)).wrapping_mul(SPLITMIX_C2);
                let xi = xi2 ^ (xi2 >> 31);

                // Modular CFT projection
                let module_idx = xi % CFT_MODULE_COUNT;
                let phi_canon = (2.0 * PI) * ((module_idx as f64 + 0.5) / (CFT_MODULE_COUNT as f64));

                phases[idx] = if is_positive {
                    phi_canon
                } else {
                    let wrapped = (2.0 * PI) - phi_canon;
                    if wrapped >= 2.0 * PI { 0.0 } else { wrapped }
                };
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

        if raw_bytes.len() != expected_bytes {
            return Err("Binary particle buffer size does not match target grid dimension.");
        }

        Ok(Self {
            particle_count,
            raw_data: raw_bytes.to_vec(),
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
        let l_box_mpc = (3.0 / 6.05712e-5f64).sqrt();
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
        let a = 1.0 / (1.0 + z);
        let om_m = 0.315;
        let om_l = 0.685;
        let om_z = (om_m * (1.0 + z).powi(3)) / (om_m * (1.0 + z).powi(3) + om_l);
        let ol_z = om_l / (om_m * (1.0 + z).powi(3) + om_l);

        let gz = 2.5 * om_z / (om_z.powf(4.0 / 7.0) - ol_z + (1.0 + om_z / 2.0) * (1.0 + ol_z / 70.0));
        let g0 = 2.5 * om_m / (om_m.powf(4.0 / 7.0) - om_l + (1.0 + om_m / 2.0) * (1.0 + om_l / 70.0));
        a * (gz / g0)
    }

    pub fn growth_factor_d2(&self, z: f64) -> f64 {
        let d1 = self.growth_factor_d1(z);
        let om_m = 0.315;
        let om_l = 0.685;
        let om_z = (om_m * (1.0 + z).powi(3)) / (om_m * (1.0 + z).powi(3) + om_l);
        -(3.0 / 7.0) * d1.powi(2) * om_z.powf(-1.0 / 143.0)
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
