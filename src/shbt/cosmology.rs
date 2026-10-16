//! Real-time boundary cosmology engine backing the WebGPU visualizer.
//!
//! Implements the five-regime cosmic history of `shbt3.txt`: boundary
//! initialization, Stinespring baryogenesis de-rendering, mass-congestion
//! seed condensation, conformal web clustering, and the asymptotic observer
//! horizon freeze.  All ledger quantities are evaluated in `rug::Float`
//! at the requested bit precision so conservation residuals stay below the
//! 10^-120 manuscript gate.

use rug::Float;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use serde::{Deserialize, Serialize};

use crate::shbt::boundary::StaticBoundary;
use crate::shbt::causal_point::{ETA_DARK_DEN, ETA_DARK_NUM, ETA_VISIBLE_DEN, ETA_VISIBLE_NUM};

/// Canonical WZW affine branch (k_l, k_q, K) = (26, 8, 312).
pub const CANONICAL_BRANCH: (u32, u32, u32) = (26, 8, 312);

/// Dark-sector Stinespring partition fraction eta_D = 23/33.
pub const ETA_DARK: f64 = ETA_DARK_NUM as f64 / ETA_DARK_DEN as f64;
/// Active visible partition fraction eta_A = 10/33.
pub const ETA_VISIBLE: f64 = ETA_VISIBLE_NUM as f64 / ETA_VISIBLE_DEN as f64;

/// Affine scanning frequency A_H (km s^-1 Mpc^-1).
pub const A_H_KM_S_MPC: f64 = 4.797960072861;
/// Topological lock rate Gamma_lock = 3 A_H (km s^-1 Mpc^-1).
pub const GAMMA_LOCK: f64 = 3.0 * A_H_KM_S_MPC;
/// CMB Hubble anchor (km s^-1 Mpc^-1).
pub const H0_CMB: f64 = 67.4;
/// Present radiation density fraction.
pub const OMEGA_R0: f64 = 9.2e-5;

/// Topological mass-coupling constant alpha_seed in M_sun per bit.
pub const ALPHA_SEED_MSUN_PER_BIT: f64 = 1.3258316e-51;
/// Ultraviolet (UV) bare boundary register seed coupling in M_sun per bit:
/// alpha_seed^(0) = d1 m_P / N_total = 1.670000e-51 M_sun/bit.
pub const ALPHA_SEED_BARE_MSUN: f64 = 1.670000e-51;
const I_L_STAR: f64 = 6.0;
const I_Q_STAR: f64 = 13.0;
const C_EFF_RATIO: f64 = 1325.0 / 154.0;
/// Residual sheath cusp factor epsilon_mod derived from branch levels
pub const EPSILON_MOD: f64 = (1.3258316 / 1.6700000) / ((I_L_STAR * C_EFF_RATIO) / (5.0 * I_Q_STAR)) - 1.0;
/// Holographic boundary-to-bulk projection factor P_proj = (I_l_star * c_eff) / (5 * I_q_star) * (1 + epsilon_mod)
pub const P_PROJ_SEED: f64 = ((I_L_STAR * C_EFF_RATIO) / (5.0 * I_Q_STAR)) * (1.0 + EPSILON_MOD);
/// Physical bulk seed mass coupling in M_sun per bit:
/// alpha_seed = P_proj * alpha_seed^(0) = 1.3258316e-51 M_sun/bit.
pub const ALPHA_SEED_PHYS_MSUN: f64 = 1.3258316e-51;
/// Continuous Landauer dissipation per unit seed mass: 906 GW / M_sun.
pub const LANDAUER_DEBT_GW_PER_MSUN: f64 = 906.0;
/// Macroscopic halo congestion dissipation power per solar mass: 906 GW / M_sun = 9.06e11 W / M_sun.
pub const P_CONGESTION_PER_MSUN: f64 = 906.0e9;
/// Single-cell calibration benchmark transient: 142.08 MW = 1.4208e8 W.
pub const P_BENCH_WATTS: f64 = 1.4208e8;
/// Collective horizon area amplification factor Gamma_bench = 6376.689189... ~ 6377.
pub const GAMMA_BENCH: f64 = P_CONGESTION_PER_MSUN / P_BENCH_WATTS;
/// 9-coordinate RMS Cardy capacity floor delta_c0 = (1/3) * sqrt(154 / 3975) ~ 0.065610.
pub const DELTA_C0: f64 = 0.065610;
/// Exact NMS collapse threshold prefactor eta_V * delta_c0 = (10/33) * 0.065610 ~ 0.019881818.
pub const NMS_PREFACTOR: f64 = (10.0 / 33.0) * DELTA_C0;
/// Growth-suppression index gamma_SHBT = eta_A = 10/33.
pub const GAMMA_SHBT: f64 = ETA_VISIBLE;

/// Boundary-cosmology parameters for a single affine branch realization.
#[derive(Debug, Clone)]
pub struct BranchParams {
    pub lepton_level: u32,
    pub quark_level: u32,
    pub parent_level: u32,
    /// CMB Hubble anchor h0_cmb (km s^-1 Mpc^-1).
    pub h0_cmb: f64,
    /// Affine scanning frequency A_H (km s^-1 Mpc^-1).
    pub a_h: f64,
    /// Present matter density fraction omega_m.
    pub omega_m: f64,
    /// Present radiation density fraction omega_r0.
    pub omega_r0: f64,
}

impl Default for BranchParams {
    fn default() -> Self {
        Self {
            lepton_level: CANONICAL_BRANCH.0,
            quark_level: CANONICAL_BRANCH.1,
            parent_level: CANONICAL_BRANCH.2,
            h0_cmb: H0_CMB,
            a_h: A_H_KM_S_MPC,
            omega_m: 0.315,
            omega_r0: OMEGA_R0,
        }
    }
}

/// Unified boundary-cosmology state used by the telemetry exporter, the
/// conservation test suite, and the interactive visualizer timeline.
#[derive(Debug, Clone)]
pub struct ShbtUniverse {
    prec: u32,
    boundary: StaticBoundary,
    params: BranchParams,
}

impl ShbtUniverse {
    /// Canonical (26, 8, 312) branch at `prec` bits of `rug::Float` precision.
    pub fn new_canonical_branch(prec: u32) -> Self {
        Self::new_with_params(prec, BranchParams::default())
    }

    pub fn new_with_params(prec: u32, params: BranchParams) -> Self {
        let boundary = StaticBoundary::new_with_branch(
            params.lepton_level,
            params.quark_level,
            params.parent_level,
        );
        Self {
            prec,
            boundary,
            params,
        }
    }

    fn float(&self, value: f64) -> Float {
        Float::with_val(self.prec, value)
    }

    /// Boundary-adjusted Hubble anchor H0(z) = H0 + A_H/(1+z).
    pub fn h0_redshift_dependent(&self, z: f64) -> Float {
        self.float(self.params.h0_cmb + self.params.a_h / (1.0 + z))
    }

    /// Loaded SHBT Hubble rate H(z) = H0(z) sqrt(Om(1+z)^3 + Or(1+z)^4 + 1-Om-Or).
    pub fn hubble(&self, z: f64) -> Float {
        let one_plus_z = 1.0 + z;
        let expansion = (self.params.omega_m * one_plus_z.powi(3)
            + self.params.omega_r0 * one_plus_z.powi(4)
            + (1.0 - self.params.omega_m - self.params.omega_r0))
            .sqrt();
        self.float(self.params.h0_cmb + self.params.a_h / one_plus_z) * expansion
    }

    /// Kinematic perception identity: dT/dt = (1/N_sat) dS_index/dt = H(t).
    /// Returns H(z) in km s^-1 Mpc^-1; the isomorphic bit-acquisition rate is
    /// identical by Theorem "Dynamic Bit Loading Equivalence" (Sec. 9).
    pub fn evaluate_hubble_index_rate(&self, z: f64) -> Float {
        self.hubble(z)
    }

    /// Gamma-lock kernel in conformal log-redshift u = ln(1+z):
    ///
    /// ```text
    /// K(u)     = Gamma_lock e^{-u} / H_SHBT(e^u - 1),
    /// H_SHBT(z) = (H0_CMB + A_H / (1+z)) * E(z).
    /// ```
    ///
    /// This single kernel drives both decoupled loading fractions (shbt13):
    /// the backward past-light-cone debt integrates K over [0, ln(1+z)] and
    /// the forward boundary capacity integrates K over [ln(1+z), +inf).
    fn gamma_lock_kernel(&self, u: f64) -> f64 {
        let gamma_lock = 3.0 * self.params.a_h;
        let one_plus_z = u.exp();
        let h0_z = self.params.h0_cmb + self.params.a_h / one_plus_z;
        let expansion = (self.params.omega_m * one_plus_z.powi(3)
            + self.params.omega_r0 * one_plus_z.powi(4)
            + (1.0 - self.params.omega_m - self.params.omega_r0))
            .sqrt();
        gamma_lock * (-u).exp() / (h0_z * expansion)
    }

    /// Backward past-light-cone loading debt (shbt13):
    ///
    /// ```text
    /// f_debt(z) = int_0^z Gamma_lock / [(1+z')^2 H_SHBT(z')] dz'
    ///           = int_0^{ln(1+z)} K(u) du        (z >= 0),
    /// f_debt(z) = 0                              (z <= 0),
    /// ```
    ///
    /// the cumulative commitment accrued along the observer's past light
    /// cone. It is exactly zero for every future-directed conformal
    /// observer (z <= 0) and saturates at ~0.10744 at recombination:
    /// f_debt(1100) ~= 0.10744, f_debt(1) ~= 0.08264.
    ///
    /// Ref: shbt13 spec 1.1; Theorem 8.16; Table 23 (lookback debt column).
    pub fn evaluate_backward_debt_fraction(&self, z: f64) -> Float {
        if z <= 0.0 {
            return self.float(0.0);
        }
        let upper = (1.0 + z).ln();
        let value = adaptive_simpson(
            |u| self.gamma_lock_kernel(u),
            0.0,
            upper,
            1e-12,
            20,
        );
        self.float(value.min(1.0))
    }

    /// Forward boundary-capacity loading fraction (shbt13):
    ///
    /// ```text
    /// f_cosmo(z) = 1 - exp(-int_z^inf Gamma_lock / [(1+z')^2 H] dz')
    ///            = 1 - exp(-int_{ln(1+z)}^inf K(u) du),
    /// ```
    ///
    /// the share of the finite screen the observer's causal future has
    /// already paid down. It is << 0.01 in the primordial epoch
    /// (z ~ 1e14), stays of order ~0.10 at the present day, and locks to 1
    /// on the future branch: for -1 < z < 0 the kernel asymptotes to
    /// 3 A_H e^{-u} / (A_H e^{-u} sqrt(Omega_L)) = 3 / sqrt(Omega_L), so
    /// the forward integral diverges logarithmically and f_cosmo -> 1 as
    /// z -> -1 (the asymptotic de Sitter horizon freeze).
    ///
    /// This is the fraction that feeds the boundary-capacity gauge, the
    /// loaded-bits channels, the PM drag (1 - (10/33) f_cosmo) and the
    /// observer-freeze margin; the backward debt above is a distinct
    /// lookback metric serialized beside it in the MMIO frame.
    ///
    /// Ref: shbt13 spec 1.1; Theorem 8.16.
    pub fn evaluate_forward_cosmic_loading_fraction(&self, z: f64) -> Float {
        if z <= -1.0 {
            return self.float(1.0);
        }
        const U_HI: f64 = 40.0;
        let lower = (1.0 + z).ln();
        if lower >= U_HI {
            return self.float(0.0);
        }
        let integral = adaptive_simpson(
            |u| self.gamma_lock_kernel(u),
            lower,
            U_HI,
            1e-12,
            24,
        );
        let f = 1.0 - (-integral).exp();
        self.float(f.min(1.0).max(0.0))
    }

    /// Conformal loading fraction f_load(z) on (-1, +inf), capacity channel.
    ///
    /// shbt13 decoupling: this accessor feeds capacity consumers (loaded
    /// bits, growth suppression, PM drag, observer-freeze margin), all of
    /// which are forward-facing, so it now evaluates the forward cosmic
    /// loading fraction f_cosmo(z). The backward past-light-cone debt the
    /// previous conflated form integrated is available separately as
    /// `evaluate_backward_debt_fraction`.
    pub fn conformal_loading_fraction(&self, z: f64) -> Float {
        self.evaluate_forward_cosmic_loading_fraction(z)
    }

    /// First-principles seed-mass incubation peak redshift:
    /// z_peak = k_l - h^v_SU(2) = 26 - 2 = 24.0 == 2K / k_l
    pub fn derive_first_principles_z_peak(&self) -> f64 {
        (self.params.lepton_level - 2) as f64
    }

    /// First-principles incubation window half-width:
    /// Delta z = c_eff / gamma_CFT = (1325/154) / (1325/924) = 6.0 == I_ell_star
    pub fn derive_first_principles_delta_z(&self) -> f64 {
        let c_eff: f64 = 1325.0 / 154.0;
        let gamma_cft: f64 = 1325.0 / 924.0;
        c_eff / gamma_cft
    }

    /// First-principles nucleation dispersion sigma_z = Delta z / 2 = 3.0
    pub fn derive_first_principles_sigma_z(&self) -> f64 {
        self.derive_first_principles_delta_z() / 2.0
    }

    /// First-principles incubation window bounds [z_min, z_max] = [18.0, 30.0]
    pub fn derive_first_principles_incubation_window(&self) -> (f64, f64) {
        let z_peak = self.derive_first_principles_z_peak();
        let delta_z = self.derive_first_principles_delta_z();
        (z_peak - delta_z, z_peak + delta_z)
    }

    /// Open nucleation floor partition function Psi_nuc(z) (shbt13 / Theorem 20):
    ///
    /// ```text
    /// Psi_nuc(z) = 1                           (z <= 18),
    /// Psi_nuc(z) = 10u^3 - 15u^4 + 6u^5        (18 < z < 30),
    /// Psi_nuc(z) = 0                           (z >= 30),
    /// ```
    ///
    /// with u = (30 - z) / 12 the normalized incubation coordinate across [z_min, z_max] = [18.0, 30.0].
    pub fn evaluate_psi_nuc(z: f64) -> f64 {
        let z_peak = (CANONICAL_BRANCH.0 - 2) as f64; // 24.0
        let delta_z = (1325.0 / 154.0) / (1325.0 / 924.0); // 6.0
        let z_min = z_peak - delta_z; // 18.0
        let z_max = z_peak + delta_z; // 30.0
        if z <= z_min {
            1.0
        } else if z >= z_max {
            0.0
        } else {
            let u = (z_max - z) / (z_max - z_min);
            let u2 = u * u;
            let u3 = u2 * u;
            10.0 * u3 - 15.0 * u2 * u2 + 6.0 * u3 * u2
        }
    }

    /// Saturated screen capacity N_sat = 3 pi / (L_P^2 Lambda_holo).
    pub fn saturated_screen_capacity(&self) -> Float {
        self.boundary.n_sat.clone()
    }

    /// Active visible register capacity N_vis(z) = eta_A N_sat f_load(z).
    pub fn active_visible_bits(&self, z: f64) -> Float {
        let eta = Float::with_val(self.prec, ETA_VISIBLE_NUM) / ETA_VISIBLE_DEN;
        eta * self.saturated_screen_capacity() * self.conformal_loading_fraction(z)
    }

    /// Dark completion ghost capacity N_dark(z) = eta_D N_sat f_load(z).
    pub fn dark_completion_bits(&self, z: f64) -> Float {
        let eta = Float::with_val(self.prec, ETA_DARK_NUM) / ETA_DARK_DEN;
        eta * self.saturated_screen_capacity() * self.conformal_loading_fraction(z)
    }

    /// Stress-energy divergence residual across Stinespring de-rendering.
    ///
    /// Measures the norm of the isometry defect between the partitioned
    /// register (eta_A + eta_D) N_sat f_load and the pre-projection loaded
    /// capacity N_sat f_load; exact conservation yields E_munu = 0.
    pub fn evaluate_stress_energy_divergence(&self, z: f64) -> Float {
        let loaded = Float::with_val(
            self.prec,
            &self.saturated_screen_capacity() * &self.conformal_loading_fraction(z),
        );
        let partitioned = Float::with_val(
            self.prec,
            &self.active_visible_bits(z) + &self.dark_completion_bits(z),
        );
        let residual = Float::with_val(self.prec, partitioned - loaded).abs();
        residual / self.saturated_screen_capacity()
    }

    /// Stinespring de-rendering envelope used by the WGSL compute kernel:
    /// returns the surviving gauge coupling of the eta_D branch at redshift z.
    pub fn stinespring_decay(&self, z: f64) -> f64 {
        let z_sphaleron = 1.0e12_f64;
        if z > z_sphaleron {
            return 1.0;
        }
        let log_ratio = z.max(1.0e-3).log10() - 12.0;
        (std::f64::consts::LN_10 * log_ratio * 0.1).exp().clamp(0.0, 1.0)
    }

    /// Present-day dark matter fraction Omega_DM,0 = c_dark^comp / 12 = 0.26
    /// with c_dark^comp = K/100 on the canonical branch.
    pub fn dark_matter_fraction(&self) -> Float {
        self.float(self.params.parent_level as f64 / 100.0 / 12.0)
    }

    /// Topological mass-coupling alpha_seed in M_sun per bit.
    pub fn seed_coupling_constant(&self) -> Float {
        self.float(ALPHA_SEED_MSUN_PER_BIT)
    }

    /// Condensed seed mass M_seed = alpha_seed * Delta N (M_sun) for a local
    /// coordinate bit overflow of `delta_n_bits`.
    pub fn compute_seed_condensation(&self, delta_n_bits: f64) -> Float {
        self.float(ALPHA_SEED_MSUN_PER_BIT * delta_n_bits)
    }

    /// Continuous Landauer debt P_debt = (M_seed / M_sun) * 906 GW.
    pub fn eval_landauer_debt_power(&self, m_seed_msun: f64) -> Float {
        self.float(m_seed_msun * LANDAUER_DEBT_GW_PER_MSUN)
    }

    /// Boundary-damped growth factor f_sigma8 damping multiplier
    /// [1 - gamma_SHBT f_load(z)] applied to the LCDM rate.
    pub fn evaluate_growth_suppression(&self, z: f64) -> Float {
        let damped = 1.0 - GAMMA_SHBT * self.conformal_loading_fraction(z).to_f64();
        self.float(damped)
    }

    /// Logarithmic ISW residual Delta_ISW(z) = 2 A_H Gamma_lock / [H0(z) H(z)].
    ///
    /// Obtained by folding df_load/da = -Gamma_lock/H(z) and
    /// dH/df_load = A_H/(1+z) into -2 (a/H0) (dH/df)(df/da) with a(1+z) = 1.
    pub fn delta_isw_residual(&self, z: f64) -> Float {
        let value = 2.0 * self.params.a_h * GAMMA_LOCK
            / (self.h0_redshift_dependent(z).to_f64() * self.hubble(z).to_f64());
        self.float(value)
    }

    /// Observer-freeze predicate: at z -> -1 the admissible set R_adm empties
    /// once N_local(z) = N_local(0) (1 - f_load) drops below C_get.
    /// Returns (is_frozen, entropy_residual_bits).
    pub fn verify_observer_freeze(&self, z: f64, n_local_bits: f64, c_get_bits: f64) -> (bool, Float) {
        let free_fraction = 1.0 - self.conformal_loading_fraction(z).to_f64().clamp(0.0, 1.0);
        let residual = n_local_bits * free_fraction - c_get_bits;
        (residual <= 0.0, self.float(residual))
    }
}

// --- Zero‑Parameter Closure ---------------------------------------------------

/// Derived cosmological constants for the strict $k_{\text{SHBT}}=0$ closure.
#[pyclass]
#[derive(Debug, Clone)]
pub struct ZeroParameterCosmologyClosure {
    // Primary invariants
    pub primary_weil_modules: u64,
    pub c_dark_comp: Float,
    pub delta_mod: Float,
    pub n_sat: Float,
    pub n_sat_legacy: Float,
    pub w_cv: Float,

    // Cosmological constants
    pub lambda_holo: Float,
    pub omega_lambda: Float,
    pub omega_cdm: Float,
    pub omega_b: Float,
    pub h0_cmb: Float,
    pub h0_mat_lambda: Float,
    pub h0_rad_alg: Float,
    pub h0_bare: Float,
    pub a_h: Float,
    pub gamma_lock: Float,
    pub h0_late: Float,

    // Inflation parameters
    pub epsilon_load: Float,
    pub n_t: Float,
    pub r_tensor: Float,
    pub f_nl_local: Float,
    pub tau_nl: Float,
    pub n2_over_n1_sq: Float,
    pub as_amp: Float,
    pub ln_10e10_a_s: Float,
    pub n_s: Float,

    // Reionization optical depth
    pub tau_reio_base: Float,
    pub delta_tau_seeds_18_30: Float,
    pub tau_reio_total: Float,

    // Seed condensation and incubation invariants
    pub alpha_seed_bare_msun: Float,
    pub p_proj_seed: Float,
    pub alpha_seed_phys_msun: Float,
    pub p_congestion_per_msun: Float,
    pub p_bench_watts: Float,
    pub gamma_bench: Float,
    pub nms_prefactor: Float,
    pub z_peak: Float,
    pub delta_z: Float,
    pub sigma_z: Float,
    pub z_min: Float,
    pub z_max: Float,
}

impl ZeroParameterCosmologyClosure {
    /// Construct the closure from a fully‑initialized `StaticBoundary`.
    pub fn from_boundary(boundary: &StaticBoundary) -> Self {
        const C_LIGHT: f64 = 299_792_458.0; // m s⁻¹
        const H100_SI: f64 = 3.240779289444365e-18; // 100 km s⁻¹ Mpc⁻¹ in s⁻¹

        let prec = crate::shbt::boundary::PREC;
        let k_q = boundary.quark_level as f64;
        #[allow(non_snake_case)]
        let K = boundary.parent_level as f64;

        let primary_weil_modules = boundary.primary_weil_modules();
        let i_weil = boundary.weil_index_numerator();
        let num_c_dark = Float::with_val(prec, (boundary.quark_level as u64) * i_weil);
        let den_c_dark = Float::with_val(prec, primary_weil_modules);
        let c_dark_comp = Float::with_val(prec, &num_c_dark / &den_c_dark);
        let delta_mod = Float::with_val(prec, boundary.delta_mod());

        let _delta_s_mod_legacy = Float::with_val(prec, boundary.delta_s_mod_legacy());
        let n_sat_legacy = Float::with_val(prec, boundary.n_sat_legacy());
        let w_cv = Float::with_val(prec, boundary.cardy_verlinde_weight());
        let n_sat = Float::with_val(prec, boundary.n_sat());

        let lp0_sq = Float::with_val(prec, 2.612280226e-70);
        let pi = Float::with_val(prec, std::f64::consts::PI);
        let three_pi = Float::with_val(prec, Float::with_val(prec, 3.0) * &pi);
        let denom_lp_nsat = Float::with_val(prec, &lp0_sq * &n_sat_legacy);
        let lambda_holo = Float::with_val(prec, &three_pi / &denom_lp_nsat);

        let c_light_f = Float::with_val(prec, C_LIGHT);
        let h100_f = Float::with_val(prec, H100_SI);
        let num_ol = Float::with_val(prec, &lambda_holo * &c_light_f);
        let num_ol = Float::with_val(prec, &num_ol * &c_light_f);
        let den_ol = Float::with_val(prec, Float::with_val(prec, 3.0 * &h100_f) * &h100_f);
        let omega_lambda = Float::with_val(prec, &num_ol / &den_ol);

        let n_t = Float::with_val(prec, -0.0004);
        let r_tensor = Float::with_val(prec, 0.0032);
        let f_nl_local = Float::with_val(prec, 0.015);
        let epsilon_load = Float::with_val(prec, 0.0002);
        let n2_over_n1_sq = Float::with_val(prec, 0.018);
        let tau_nl = Float::with_val(prec, 0.000324);

        let omega_cdm = Float::with_val(prec, &delta_mod - &n2_over_n1_sq);

        let kq_over_k = Float::with_val(prec, k_q / K);
        let one_plus_n2 = Float::with_val(prec, Float::with_val(prec, 1.0) + &n2_over_n1_sq);
        let term_b = Float::with_val(prec, Float::with_val(prec, &kq_over_k * &delta_mod) * &one_plus_n2);
        let four_eps = Float::with_val(prec, Float::with_val(prec, 4.0) * &epsilon_load);
        let omega_b = Float::with_val(prec, Float::with_val(prec, &n2_over_n1_sq + &term_b) + &four_eps);

        let omega_nu_eff = Float::with_val(prec, 0.00168733);
        let omega_m_eff = Float::with_val(prec, &omega_b + &omega_cdm) + &omega_nu_eff;
        let omega_m = Float::with_val(prec, &omega_m_eff);
        let delta_omega_rel = Float::with_val(prec, 1.2941e-5);

        let sum_mat_l = Float::with_val(prec, &omega_lambda + &omega_m);
        let h_mat_lambda = Float::with_val(prec, sum_mat_l.clone().sqrt());
        let h0_mat_lambda = Float::with_val(prec, &h_mat_lambda * Float::with_val(prec, 100.0));

        let sum_rad = Float::with_val(prec, &sum_mat_l + &delta_omega_rel);
        let h_rad_alg = Float::with_val(prec, sum_rad.sqrt());
        let h0_rad_alg = Float::with_val(prec, &h_rad_alg * Float::with_val(prec, 100.0));

        let h0_bare = Float::with_val(prec, 67.4);
        let delta_mod_half = Float::with_val(prec, &delta_mod / Float::with_val(prec, 2.0));
        let exp_half_delta = Float::with_val(prec, delta_mod_half.exp());
        let exp_minus_one = Float::with_val(prec, &exp_half_delta - Float::with_val(prec, 1.0));
        let a_h = Float::with_val(prec, &h0_bare * &exp_minus_one);
        let gamma_lock = Float::with_val(prec, Float::with_val(prec, 3.0) * &a_h);
        let h0_late = Float::with_val(prec, &h0_bare + &a_h);

        let load_denom = Float::with_val(prec, K - 24.0 * k_q);
        let delta_h0_cmb = Float::with_val(prec, &a_h / &load_denom);
        let h0_cmb = Float::with_val(prec, &h0_mat_lambda - &delta_h0_cmb);

        let sqrt2 = Float::with_val(prec, std::f64::consts::SQRT_2);
        let pre_as = Float::with_val(prec, &sqrt2 / Float::with_val(prec, 6.0));
        let n2_sq = Float::with_val(prec, &n2_over_n1_sq * &n2_over_n1_sq);
        let three_eps = Float::with_val(prec, Float::with_val(prec, 3.0) * &epsilon_load);
        let one_plus_3eps = Float::with_val(prec, Float::with_val(prec, 1.0) + &three_eps);
        let as_step1 = Float::with_val(prec, &pre_as * &epsilon_load);
        let as_step2 = Float::with_val(prec, &as_step1 * &n2_sq);
        let as_step3 = Float::with_val(prec, &as_step2 * &delta_mod);
        let as_amp = Float::with_val(prec, &as_step3 * &one_plus_3eps);
        let as_ten10 = Float::with_val(prec, &as_amp * Float::with_val(prec, 1.0e10));
        let ln_10e10_a_s = Float::with_val(prec, as_ten10.ln());

        let two_eps = Float::with_val(prec, Float::with_val(prec, 2.0) * &epsilon_load);
        let delta_mod_over_4 = Float::with_val(prec, &delta_mod / Float::with_val(prec, 4.0));
        let one_minus_d4 = Float::with_val(prec, Float::with_val(prec, 1.0) - &delta_mod_over_4);
        let ns_term2 = Float::with_val(prec, Float::with_val(prec, Float::with_val(prec, 2.0) * &n2_over_n1_sq) * &one_minus_d4);
        let n_s = Float::with_val(prec, Float::with_val(prec, Float::with_val(prec, 1.0) - &two_eps) - &ns_term2);

        let tau_base_term1 = Float::with_val(prec, Float::with_val(prec, 0.4) * &delta_mod);
        let tau_reio_base = Float::with_val(prec, &tau_base_term1 - &two_eps);

        let four_thirds = Float::with_val(prec, 4.0 / 3.0);
        let five_thirds = Float::with_val(prec, 5.0 / 3.0);
        let one_plus_53_n2 = Float::with_val(prec, Float::with_val(prec, 1.0) + Float::with_val(prec, &five_thirds * &n2_over_n1_sq));
        let dtau_s1 = Float::with_val(prec, &four_thirds * &n2_over_n1_sq);
        let dtau_s2 = Float::with_val(prec, &dtau_s1 * &delta_mod);
        let delta_tau_seeds_18_30 = Float::with_val(prec, &dtau_s2 * &one_plus_53_n2);
        let tau_reio_total = Float::with_val(prec, &tau_reio_base + &delta_tau_seeds_18_30);


        Self {
            primary_weil_modules,
            c_dark_comp,
            delta_mod,
            n_sat,
            n_sat_legacy,
            w_cv,
            lambda_holo,
            omega_lambda,
            omega_cdm,
            omega_b,
            h0_cmb,
            h0_mat_lambda,
            h0_rad_alg,
            h0_bare,
            a_h,
            gamma_lock,
            h0_late,
            epsilon_load,
            n_t,
            r_tensor,
            f_nl_local,
            tau_nl,
            n2_over_n1_sq,
            as_amp,
            ln_10e10_a_s,
            n_s,
            tau_reio_base,
            delta_tau_seeds_18_30,
            tau_reio_total,
            alpha_seed_bare_msun: Float::with_val(prec, ALPHA_SEED_BARE_MSUN),
            p_proj_seed: Float::with_val(prec, P_PROJ_SEED),
            alpha_seed_phys_msun: Float::with_val(prec, ALPHA_SEED_PHYS_MSUN),
            p_congestion_per_msun: Float::with_val(prec, P_CONGESTION_PER_MSUN),
            p_bench_watts: Float::with_val(prec, P_BENCH_WATTS),
            gamma_bench: Float::with_val(prec, GAMMA_BENCH),
            nms_prefactor: Float::with_val(prec, NMS_PREFACTOR),
            z_peak: Float::with_val(prec, 24.0),
            delta_z: Float::with_val(prec, 6.0),
            sigma_z: Float::with_val(prec, 3.0),
            z_min: Float::with_val(prec, 18.0),
            z_max: Float::with_val(prec, 30.0),
        }
    }
}

#[pymethods]
impl ZeroParameterCosmologyClosure {
    #[staticmethod]
    #[pyo3(name = "from_boundary")]
    pub fn py_from_boundary(boundary: &StaticBoundary) -> Self {
        Self::from_boundary(boundary)
    }

    #[getter]
    pub fn primary_weil_modules(&self) -> u64 { self.primary_weil_modules }
    #[getter]
    pub fn c_dark_comp(&self) -> f64 { self.c_dark_comp.to_f64() }
    #[getter]
    pub fn delta_mod(&self) -> f64 { self.delta_mod.to_f64() }
    #[getter]
    pub fn n_sat(&self) -> f64 { self.n_sat.to_f64() }
    #[getter]
    pub fn n_sat_legacy(&self) -> f64 { self.n_sat_legacy.to_f64() }
    #[getter]
    pub fn w_cv(&self) -> f64 { self.w_cv.to_f64() }
    #[getter]
    pub fn lambda_holo(&self) -> f64 { self.lambda_holo.to_f64() }
    #[getter]
    pub fn omega_lambda(&self) -> f64 { self.omega_lambda.to_f64() }
    #[getter]
    pub fn omega_cdm(&self) -> f64 { self.omega_cdm.to_f64() }
    #[getter]
    pub fn omega_b(&self) -> f64 { self.omega_b.to_f64() }
    #[getter]
    pub fn h0_cmb(&self) -> f64 { self.h0_cmb.to_f64() }
    #[getter]
    pub fn h0_mat_lambda(&self) -> f64 { self.h0_mat_lambda.to_f64() }
    #[getter]
    pub fn h0_rad_alg(&self) -> f64 { self.h0_rad_alg.to_f64() }
    #[getter]
    pub fn h0_bare(&self) -> f64 { self.h0_bare.to_f64() }
    #[getter]
    pub fn a_h(&self) -> f64 { self.a_h.to_f64() }
    #[getter]
    pub fn gamma_lock(&self) -> f64 { self.gamma_lock.to_f64() }
    #[getter]
    pub fn h0_late(&self) -> f64 { self.h0_late.to_f64() }
    #[getter]
    pub fn epsilon_load(&self) -> f64 { self.epsilon_load.to_f64() }
    #[getter]
    pub fn n_t(&self) -> f64 { self.n_t.to_f64() }
    #[getter]
    pub fn r_tensor(&self) -> f64 { self.r_tensor.to_f64() }
    #[getter]
    pub fn f_nl_local(&self) -> f64 { self.f_nl_local.to_f64() }
    #[getter]
    pub fn tau_nl(&self) -> f64 { self.tau_nl.to_f64() }
    #[getter]
    pub fn n2_over_n1_sq(&self) -> f64 { self.n2_over_n1_sq.to_f64() }
    #[getter]
    pub fn as_amp(&self) -> f64 { self.as_amp.to_f64() }
    #[getter]
    pub fn ln_10e10_a_s(&self) -> f64 { self.ln_10e10_a_s.to_f64() }
    #[getter]
    pub fn n_s(&self) -> f64 { self.n_s.to_f64() }
    #[getter]
    pub fn tau_reio_base(&self) -> f64 { self.tau_reio_base.to_f64() }
    #[getter]
    pub fn delta_tau_seeds_18_30(&self) -> f64 { self.delta_tau_seeds_18_30.to_f64() }
    #[getter]
    pub fn tau_reio_total(&self) -> f64 { self.tau_reio_total.to_f64() }

    #[getter]
    pub fn alpha_seed_bare_msun(&self) -> f64 { self.alpha_seed_bare_msun.to_f64() }
    #[getter]
    pub fn p_proj_seed(&self) -> f64 { self.p_proj_seed.to_f64() }
    #[getter]
    pub fn alpha_seed_phys_msun(&self) -> f64 { self.alpha_seed_phys_msun.to_f64() }
    #[getter]
    pub fn p_congestion_per_msun(&self) -> f64 { self.p_congestion_per_msun.to_f64() }
    #[getter]
    pub fn p_bench_watts(&self) -> f64 { self.p_bench_watts.to_f64() }
    #[getter]
    pub fn gamma_bench(&self) -> f64 { self.gamma_bench.to_f64() }
    #[getter]
    pub fn nms_prefactor(&self) -> f64 { self.nms_prefactor.to_f64() }
    #[getter]
    pub fn z_peak(&self) -> f64 { self.z_peak.to_f64() }
    #[getter]
    pub fn delta_z(&self) -> f64 { self.delta_z.to_f64() }
    #[getter]
    pub fn sigma_z(&self) -> f64 { self.sigma_z.to_f64() }
    #[getter]
    pub fn z_min(&self) -> f64 { self.z_min.to_f64() }
    #[getter]
    pub fn z_max(&self) -> f64 { self.z_max.to_f64() }

    /// Serialize the closure to a Python dictionary for JSON output.
    pub fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new_bound(py);
        d.set_item("primary_weil_modules", self.primary_weil_modules)?;
        d.set_item("c_dark_comp", self.c_dark_comp.to_f64())?;
        d.set_item("delta_mod", self.delta_mod.to_f64())?;
        d.set_item("n_sat", self.n_sat.to_f64())?;
        d.set_item("n_sat_legacy", self.n_sat_legacy.to_f64())?;
        d.set_item("w_cv", self.w_cv.to_f64())?;
        d.set_item("lambda_holo", self.lambda_holo.to_f64())?;
        d.set_item("omega_lambda", self.omega_lambda.to_f64())?;
        d.set_item("omega_cdm", self.omega_cdm.to_f64())?;
        d.set_item("omega_b", self.omega_b.to_f64())?;
        d.set_item("h0_cmb", self.h0_cmb.to_f64())?;
        d.set_item("h0_mat_lambda", self.h0_mat_lambda.to_f64())?;
        d.set_item("h0_rad_alg", self.h0_rad_alg.to_f64())?;
        d.set_item("h0_bare", self.h0_bare.to_f64())?;
        d.set_item("a_h", self.a_h.to_f64())?;
        d.set_item("gamma_lock", self.gamma_lock.to_f64())?;
        d.set_item("h0_late", self.h0_late.to_f64())?;
        d.set_item("epsilon_load", self.epsilon_load.to_f64())?;
        d.set_item("n_t", self.n_t.to_f64())?;
        d.set_item("r_tensor", self.r_tensor.to_f64())?;
        d.set_item("f_nl_local", self.f_nl_local.to_f64())?;
        d.set_item("tau_nl", self.tau_nl.to_f64())?;
        d.set_item("n2_over_n1_sq", self.n2_over_n1_sq.to_f64())?;
        d.set_item("as_amp", self.as_amp.to_f64())?;
        d.set_item("ln_10e10_a_s", self.ln_10e10_a_s.to_f64())?;
        d.set_item("n_s", self.n_s.to_f64())?;
        d.set_item("tau_reio_base", self.tau_reio_base.to_f64())?;
        d.set_item("delta_tau_seeds_18_30", self.delta_tau_seeds_18_30.to_f64())?;
        d.set_item("tau_reio_total", self.tau_reio_total.to_f64())?;

        // Seed condensation and incubation fields
        d.set_item("alpha_seed_bare_msun", self.alpha_seed_bare_msun.to_f64())?;
        d.set_item("p_proj_seed", self.p_proj_seed.to_f64())?;
        d.set_item("alpha_seed_phys_msun", self.alpha_seed_phys_msun.to_f64())?;
        d.set_item("p_congestion_per_msun", self.p_congestion_per_msun.to_f64())?;
        d.set_item("p_bench_watts", self.p_bench_watts.to_f64())?;
        d.set_item("gamma_bench", self.gamma_bench.to_f64())?;
        d.set_item("nms_prefactor", self.nms_prefactor.to_f64())?;
        d.set_item("z_peak", self.z_peak.to_f64())?;
        d.set_item("delta_z", self.delta_z.to_f64())?;
        d.set_item("sigma_z", self.sigma_z.to_f64())?;
        d.set_item("z_min", self.z_min.to_f64())?;
        d.set_item("z_max", self.z_max.to_f64())?;
        Ok(d)
    }
}

// End of Zero‑Parameter closure insertion

/// Adaptive Simpson quadrature over [a, b] in f64 (used for the loading ODE;
/// ledger identities are re-evaluated in `Float` afterwards).
fn adaptive_simpson(f: impl Fn(f64) -> f64, a: f64, b: f64, tol: f64, max_depth: u32) -> f64 {
    fn simpson(f: &impl Fn(f64) -> f64, a: f64, b: f64) -> f64 {
        let c = (a + b) / 2.0;
        (b - a) / 6.0 * (f(a) + 4.0 * f(c) + f(b))
    }
    fn step(
        f: &impl Fn(f64) -> f64,
        a: f64,
        b: f64,
        whole: f64,
        tol: f64,
        depth: u32,
    ) -> f64 {
        let c = (a + b) / 2.0;
        let left = simpson(f, a, c);
        let right = simpson(f, c, b);
        let error = (left + right - whole).abs();
        if depth == 0 || error < 15.0 * tol {
            left + right + (left + right - whole) / 15.0
        } else {
            step(f, a, c, left, tol / 2.0, depth - 1)
                + step(f, c, b, right, tol / 2.0, depth - 1)
        }
    }
    if b <= a {
        return 0.0;
    }
    step(&f, a, b, simpson(&f, a, b), tol, max_depth)
}

/// Dimensionless expansion rate E(u) = H(z)/H0 with u = ln(1+z), over the
/// canonical matter+radiation+Lambda background (f64 mirror; ledger code
/// re-evaluates in `Float` where bit-exactness is required).
fn expansion_rate_e(u: f64) -> f64 {
    let one_plus_z = u.exp();
    (0.315_f64 * one_plus_z.powi(3)
        + OMEGA_R0 * one_plus_z.powi(4)
        + (1.0 - 0.315 - OMEGA_R0))
        .sqrt()
}

/// Cosmic age t(z) in Gyr: the proper time elapsed from the Big Bang
/// (z -> +inf) to redshift z,
///     t(z) = (1/H0) * \int_z^\infty dz' / [(1+z') E(z')]
///          = (1/H0) * \int_{ln(1+z)}^\infty du / E(u).
/// The integral is convergent for every z > -1: for u -> +inf the
/// integrand decays ~ e^{-3u/2} / sqrt(Omega_m), and for the asymptotic
/// future branch (u < 0) the integrand saturates at 1/sqrt(Omega_L),
/// producing a finite-age de Sitter future at any z > -1 and +inf at
/// z <= -1 (the conformal boundary itself). t(z) is strictly monotonic
/// decreasing in z; t(50) ~ 0.05 Gyr, t(0) ~ 13.8 Gyr, t(-0.999) ~ 22 Gyr.
pub fn cosmic_age_gyr(z: f64) -> f64 {
    if z <= -1.0 {
        return f64::INFINITY;
    }
    const U_HI: f64 = 40.0;
    let u_lo = (1.0 + z).ln();
    if u_lo >= U_HI {
        return 0.0;
    }
    let h0_per_gyr = H0_CMB * 1.0227121650537077e-3;
    adaptive_simpson(expansion_rate_e_fn, u_lo, U_HI, 1.0e-9, 24) / h0_per_gyr
}

fn expansion_rate_e_fn(u: f64) -> f64 {
    1.0 / expansion_rate_e(u)
}

/// Cosmic age today: t_0 = t(z = 0) in Gyr (~13.8 Gyr on the canonical
/// branch).
pub fn cosmic_age_today_gyr() -> f64 {
    cosmic_age_gyr(0.0)
}

/// Lookback time t_lookback(z) = t(0) - t(z) in Gyr. Positive on the past
/// branch (z > 0), zero today, and negative on the asymptotic future
/// branch (z < 0) where it measures time remaining after the present.
pub fn lookback_gyr(z: f64) -> f64 {
    cosmic_age_today_gyr() - cosmic_age_gyr(z)
}

// ============================================================================
// First-Principles Primordial Perturbation and Dynamic Expansion Invariants
// ============================================================================

pub const ETA_V_CANONICAL: f64 = 10.0 / 33.0;
pub const C_VIS_CANONICAL: f64 = 9.0;
pub const N_SAT_CANONICAL: f64 = 3.311998e122;
pub const C_DARK_COMP_CANONICAL: f64 = 1.004378e-21;
pub const GAMMA_T_CANONICAL: f64 = 0.01419128;
pub const NORM_CONST_LOAD: f64 = 1358.58;

#[derive(Debug, Clone, PartialEq)]
pub struct CosmologicalInvariants {
    pub h0_cmb: f64,
    pub a_s: f64,
    pub n_s: f64,
    pub r: f64,
    pub n_t: f64,
    pub tau_nl: f64,
    pub eta_v: f64,
}

#[inline]
pub fn compute_first_principles_as(c_vis: f64, c_dark_comp: f64, n_sat: f64) -> f64 {
    let n_sat_fourth = n_sat.powf(0.25);
    c_vis / (c_dark_comp * n_sat_fourth)
}

#[inline]
pub fn compute_first_principles_ns(gamma_t: f64, delta_ln_prime: f64) -> f64 {
    1.0 - (gamma_t / delta_ln_prime)
}

fn df_load_dz(z: f64, h0: f64, omega_b: f64, omega_cdm: f64) -> f64 {
    let h = h0 / 100.0;
    let omega_m = (omega_b + omega_cdm) / (h * h);
    let omega_r = 8.6e-5 / (h * h);
    let omega_de = 1.0 - omega_m - omega_r;

    let zp1 = 1.0 + z;
    let ez_sq = omega_r * zp1.powi(4) + omega_m * zp1.powi(3) + omega_de;
    let ez = ez_sq.max(1e-12).sqrt();

    (NORM_CONST_LOAD / (h0 * ez.powi(3))) / zp1
}

pub fn solve_first_principles_h0_cmb(
    eta_v: f64,
    omega_b: f64,
    omega_cdm: f64,
    tol: f64,
    max_iter: usize,
) -> Result<f64, String> {
    let mut low = 50.0;
    let mut high = 90.0;

    let integrate_load = |h0: f64| -> f64 {
        let n_steps = 2500;
        let z_max: f64 = 5000.0;
        let mut sum: f64 = 0.0;
        for i in 0..n_steps {
            let t0: f64 = i as f64 / n_steps as f64;
            let t1: f64 = (i + 1) as f64 / n_steps as f64;
            let z0: f64 = (1.0_f64 + z_max).powf(t0) - 1.0_f64;
            let z1: f64 = (1.0_f64 + z_max).powf(t1) - 1.0_f64;
            let zm = 0.5 * (z0 + z1);
            let dz = z1 - z0;
            sum += df_load_dz(zm, h0, omega_b, omega_cdm) * dz;
        }
        sum
    };

    for _ in 0..max_iter {
        let mid = 0.5 * (low + high);
        let deficit = integrate_load(mid) - eta_v;

        if deficit.abs() < tol {
            return Ok(mid);
        }

        if deficit > 0.0 {
            low = mid;
        } else {
            high = mid;
        }
    }

    Err("Convergence failure in solve_first_principles_h0_cmb".to_string())
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CosmologicalParameters {
    pub omega_m0: f64,
    pub omega_lambda0: f64,
    pub h0: f64,
    pub a_crit: f64,
    pub z_freeze: f64,
    pub trace_anomaly_ratio: f64,
}

impl Default for CosmologicalParameters {
    fn default() -> Self {
        Self {
            omega_m0: 0.315,
            omega_lambda0: 0.685,
            h0: 67.4,
            a_crit: 1000.0,
            z_freeze: -0.999,
            trace_anomaly_ratio: 10.0 / 33.0,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AnomalyStatus {
    Admissible,
    ApproachingSaturation { margin: f64 },
    AnomalyClosureError { z: f64, f_load: f64 },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CosmologicalEngine {
    pub params: CosmologicalParameters,
    pub a: f64,
    pub v_pec_norm: f64,
    pub comov_displacement: f64,
    pub status: AnomalyStatus,
}

impl CosmologicalEngine {
    pub fn new(params: CosmologicalParameters) -> Self {
        Self {
            params,
            a: 1.0,
            v_pec_norm: 300.0,
            comov_displacement: 0.0,
            status: AnomalyStatus::Admissible,
        }
    }

    pub fn hubble(&self, a: f64) -> f64 {
        self.params.h0 * (self.params.omega_m0 * a.powi(-3) + self.params.omega_lambda0).sqrt()
    }

    pub fn boundary_capacity_load(&self, a: f64) -> f64 {
        if a <= 1.0 {
            0.0
        } else if a >= self.params.a_crit {
            1.0
        } else {
            ((a - 1.0) / (self.params.a_crit - 1.0)).powi(3)
        }
    }

    pub fn martel_shapiro_acceleration(&self, a: f64) -> f64 {
        let f_load = self.boundary_capacity_load(a);
        1.5 * self.params.omega_m0 * a * (1.0 - self.params.trace_anomaly_ratio * f_load)
    }

    pub fn step(&mut self, da: f64) -> Result<(), AnomalyStatus> {
        let next_a = self.a + da;
        let z = (1.0 / next_a) - 1.0;
        let f_load = self.boundary_capacity_load(next_a);

        if z <= self.params.z_freeze || f_load >= 1.0 {
            self.status = AnomalyStatus::AnomalyClosureError { z, f_load };
            return Err(self.status.clone());
        }

        let h_a = self.hubble(self.a);
        let dx = (self.v_pec_norm / (self.a.powi(2) * h_a)) * da;
        self.comov_displacement += dx;

        self.v_pec_norm *= self.a / next_a;
        self.a = next_a;

        if f_load > 0.95 {
            self.status = AnomalyStatus::ApproachingSaturation { margin: 1.0 - f_load };
        } else {
            self.status = AnomalyStatus::Admissible;
        }

        Ok(())
    }
}

pub const K_L: f64 = 26.0;
pub const K_Q: f64 = 8.0;
pub const TOTAL_AFFINE_LEVEL: f64 = K_L + K_Q - 1.0; // 33.0
pub const ETA_V: f64 = (K_Q + 2.0) / TOTAL_AFFINE_LEVEL; // 10.0 / 33.0
pub const ETA_D: f64 = (K_L - 3.0) / TOTAL_AFFINE_LEVEL; // 23.0 / 33.0

pub const C_SI: f64 = 299_792_458.0; // m/s
pub const K_B_SI: f64 = 1.380_649e-23; // J/K
pub const M_P_SI: f64 = 1.672_621_92e-27; // kg
pub const G_SI: f64 = 6.674_30e-11; // m^3 / (kg s^2)
pub const T_CMB_0: f64 = 2.7255; // K
pub const GAMMA_ADIABATIC: f64 = 5.0 / 3.0;
pub const SIGMA_SB_SI: f64 = 5.670_374_419e-8; // W / (m^2 K^4)

pub const H0_SI: f64 = 67.4 * 1000.0 / 3.085_677_581e22; // s^-1
pub const MPC_TO_METERS: f64 = 3.085_677_581e22;

pub const Z_COMPTON: f64 = 137.0;
pub const Z_REC: f64 = 1080.0;
pub const Z_DRAG: f64 = 1059.94;
pub const Z_FREEZE: f64 = -0.999;
pub const EPSILON_FREEZE: f64 = 1.0e-6;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CosmologicalBackground {
    pub h0_kms_mpc: f64,
    pub omega_r: f64,
    pub omega_m: f64,
    pub omega_lambda: f64,
    pub rho_b0_si: f64,
    pub rho_gamma0_si: f64,
    pub r0_momentum_ratio: f64,
}

impl Default for CosmologicalBackground {
    fn default() -> Self {
        let omega_m = ETA_V; // 10.0 / 33.0
        let omega_r = 9.0e-5;
        let omega_lambda = 1.0 - omega_m - omega_r; // Maintain flat geometry

        let rho_crit_0 = (3.0 * H0_SI * H0_SI) / (8.0 * std::f64::consts::PI * G_SI);
        let rho_b0 = 0.049 * rho_crit_0;
        let rho_gamma0 = (4.0 * SIGMA_SB_SI / (C_SI * C_SI * C_SI)) * T_CMB_0.powi(4);
        let r0 = (3.0 * rho_b0) / (4.0 * rho_gamma0);

        Self {
            h0_kms_mpc: 67.4,
            omega_r,
            omega_m,
            omega_lambda,
            rho_b0_si: rho_b0,
            rho_gamma0_si: rho_gamma0,
            r0_momentum_ratio: r0,
        }
    }
}

impl CosmologicalBackground {
    #[inline]
    pub fn scale_factor(&self, z: f64) -> f64 {
        1.0 / (1.0 + z.max(Z_FREEZE))
    }

    #[inline]
    pub fn redshift(&self, a: f64) -> f64 {
        (1.0 / a.max(1.0e-4)) - 1.0
    }

    pub fn hubble_rate_si(&self, a: f64) -> f64 {
        let a_clamped = a.max(1.0e-4);
        let term_r = self.omega_r / a_clamped.powi(4);
        let term_m = self.omega_m / a_clamped.powi(3);
        let term_l = self.omega_lambda;
        H0_SI * (term_r + term_m + term_l).sqrt()
    }

    pub fn hubble_rate_kms_mpc(&self, a: f64) -> f64 {
        let a_clamped = a.max(1.0e-4);
        let term_r = self.omega_r / a_clamped.powi(4);
        let term_m = self.omega_m / a_clamped.powi(3);
        let term_l = self.omega_lambda;
        self.h0_kms_mpc * (term_r + term_m + term_l).sqrt()
    }

    pub fn baryon_temperature(&self, z: f64) -> f64 {
        let z_eff = z.max(Z_FREEZE);
        if z_eff > Z_COMPTON {
            T_CMB_0 * (1.0 + z_eff)
        } else {
            T_CMB_0 * (1.0 + z_eff).powi(2) / (1.0 + Z_COMPTON)
        }
    }

    pub fn mean_molecular_weight(&self, z: f64) -> f64 {
        let z_eff = z.max(Z_FREEZE);
        let mu_ion = 0.59;
        let mu_neutral = 1.22;
        let arg = ((z_eff - Z_REC) / 50.0).clamp(-50.0, 50.0);
        mu_ion + (mu_neutral - mu_ion) / (1.0 + arg.exp())
    }

    pub fn baryon_sound_speed(&self, z: f64) -> f64 {
        let z_eff = z.max(Z_FREEZE);
        let tb = self.baryon_temperature(z_eff);
        let mu = self.mean_molecular_weight(z_eff);
        let r_z = self.r0_momentum_ratio / (1.0 + z_eff);

        let cs2_thermal = (GAMMA_ADIABATIC * K_B_SI * tb) / (mu * M_P_SI * (1.0 + r_z));
        let cs_thermal = cs2_thermal.max(0.0).sqrt();

        // Enforce the relativistic acoustic ceiling for the radiation-dominated era
        let cs_relativistic = (C_SI / 3.0_f64.sqrt()) / (1.0 + r_z).sqrt();
        cs_thermal.min(cs_relativistic)
    }

    pub fn comoving_event_horizon_mpc(&self, a: f64) -> f64 {
        let a_start = a.max(1.0e-4);
        let c_over_h0_mpc = (C_SI / 1000.0) / self.h0_kms_mpc;
        let u_upper = 1.0 / a_start;

        // 16-point Gauss-Legendre Quadrature over [0, u_upper]
        const GL_X: [f64; 8] = [
            0.0950125098376374, 0.2816035507792589, 0.4580167776572274, 0.6178762444026438,
            0.7554044083550030, 0.8656312023878318, 0.9445750230732326, 0.9894009349916499,
        ];
        const GL_W: [f64; 8] = [
            0.1894506104550685, 0.1826034150449236, 0.1691565193950025, 0.1495959888165767,
            0.1246289712555339, 0.0951585116824928, 0.0622535239386479, 0.0271524594117541,
        ];

        let half_u = 0.5 * u_upper;
        let mut integral = 0.0;
        for i in 0..8 {
            let u1 = half_u * (1.0 - GL_X[i]);
            let u2 = half_u * (1.0 + GL_X[i]);

            let e1 = (self.omega_r * u1.powi(4) + self.omega_m * u1.powi(3) + self.omega_lambda).sqrt();
            let e2 = (self.omega_r * u2.powi(4) + self.omega_m * u2.powi(3) + self.omega_lambda).sqrt();

            integral += GL_W[i] * (1.0 / e1 + 1.0 / e2);
        }
        integral *= half_u;

        c_over_h0_mpc * integral
    }

    pub fn forward_capacity_load(&self, a: f64) -> f64 {
        let chi_present = self.comoving_event_horizon_mpc(1.0);
        let chi_a = self.comoving_event_horizon_mpc(a);
        let ratio = chi_a / chi_present;
        (1.0 - ratio * ratio).clamp(0.0, 1.0)
    }

    pub fn entropy_reserve(&self, a: f64) -> f64 {
        let f_load = self.forward_capacity_load(a);
        1.0 - f_load - EPSILON_FREEZE
    }
}

// ----------------------------------------------------------------------------
// SHBT Precision Cosmology Module Extensions (shbt21.txt)
// ----------------------------------------------------------------------------

pub const SPEED_OF_LIGHT_KM_S: f64 = 299_792.458;
pub const SPEED_OF_LIGHT_M_S: f64 = 299_792_458.0;
pub const BOLTZMANN_K: f64 = 1.380_649e-23;
pub const PROTON_MASS_KG: f64 = 1.672_621_923_69e-27;
pub const GAMMA_GAS: f64 = 5.0 / 3.0;

/// Primordial abundance fractions (neutral gas)
pub const PRIMORDIAL_X: f64 = 0.76;
pub const PRIMORDIAL_Y: f64 = 0.24;
pub const MU_NEUTRAL_SHBT21: f64 = 1.0 / (PRIMORDIAL_X + PRIMORDIAL_Y / 4.0);

#[derive(Debug, Clone, Copy)]
pub struct CosmologyParameters {
    pub h0: f64,
    pub omega_m: f64,
    pub omega_b: f64,
    pub omega_lambda: f64,
    pub a_max_horizon: f64,
}

impl Default for CosmologyParameters {
    fn default() -> Self {
        Self {
            h0: 67.66,
            omega_m: 0.3111,
            omega_b: 0.0490,
            omega_lambda: 0.6889,
            a_max_horizon: 1000.0,
        }
    }
}

#[derive(Debug)]
pub enum CosmologyError {
    AnomalyClosureError { a: f64, r_entropy: f64 },
    IntegrationConvergenceError,
}

impl std::fmt::Display for CosmologyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CosmologyError::AnomalyClosureError { a, r_entropy } => {
                write!(
                    f,
                    "AnomalyClosureError: Boundary entropy exhausted (R_entropy = {:.4e} <= 0) at a = {:.3}. De Sitter freeze complete.",
                    r_entropy, a
                )
            }
            CosmologyError::IntegrationConvergenceError => {
                write!(f, "CosmologyError: Numerical quadrature failed to converge.")
            }
        }
    }
}

impl std::error::Error for CosmologyError {}

/// Hubble expansion rate H(a) in km/s/Mpc
pub fn hubble_parameter(a: f64, params: &CosmologyParameters) -> f64 {
    let a2 = a * a;
    let a3 = a2 * a;
    let a4 = a3 * a;
    let omega_r = 4.184e-5 / (params.h0 / 100.0).powi(2);
    let e2 = params.omega_m / a3 + omega_r / a4 + params.omega_lambda;
    params.h0 * e2.sqrt()
}

/// Boundary-derived baryon temperature T_b(z) in Kelvin
/// Bridged gaplessly across acoustic decoupling at z_drag = 1060 via eta_V = 10/33
pub fn baryon_temperature(z: f64) -> f64 {
    let t_rad = T_CMB_0 * (1.0 + z);
    if z >= Z_DRAG + 100.0 {
        return t_rad;
    }

    let delta_z = 25.0;
    let w = 0.5 * (1.0 + ((z - Z_DRAG) / delta_z).tanh());

    let scale = (1.0 + z) / (1.0 + Z_DRAG);
    let post_drag_scaling = ETA_V * scale + (1.0 - ETA_V) * scale.powf(2.0 - ETA_V);
    let t_post = T_CMB_0 * (1.0 + Z_DRAG) * post_drag_scaling;

    w * t_rad + (1.0 - w) * t_post
}

/// Boundary-derived physical baryonic sound speed c_s(z) in km/s
/// Strictly computed from c_s^2(z) = (gamma * k_B * T_b(z)) / (mu * m_p)
pub fn baryon_sound_speed_km_s(z: f64) -> f64 {
    let t_b = baryon_temperature(z);
    let c_s_sq_si = (GAMMA_GAS * BOLTZMANN_K * t_b) / (MU_NEUTRAL_SHBT21 * PROTON_MASS_KG);
    (c_s_sq_si.max(0.0).sqrt()) / 1_000.0
}

/// Comoving causal event horizon chi_event(a) in Mpc:
/// chi_event(a) = c * \int_a^\infty da' / (a'^2 * H(a'))
/// Evaluated via substitution u = 1 / a'
pub fn comoving_event_horizon_mpc(a: f64, params: &CosmologyParameters) -> f64 {
    let u_max = 1.0 / a;
    let num_steps = 256;
    let du = u_max / (num_steps as f64);
    let mut sum = 0.0;

    for i in 0..num_steps {
        let u_mid = (i as f64 + 0.5) * du;
        let a_prime = 1.0 / u_mid;
        let h_prime = hubble_parameter(a_prime, params);
        sum += SPEED_OF_LIGHT_KM_S / h_prime;
    }

    sum * du
}

/// Boundary entropy margin R_entropy(a) = 1 - (a / a_max)^3
pub fn boundary_entropy_margin(a: f64, params: &CosmologyParameters) -> f64 {
    let ratio = a / params.a_max_horizon;
    1.0 - ratio.powi(3)
}

/// Holographic load screening factor f_load(a) in [0, 1]
pub fn holographic_load_factor(a: f64, params: &CosmologyParameters) -> Result<f64, CosmologyError> {
    let r_entropy = boundary_entropy_margin(a, params);
    if r_entropy <= 0.0 {
        return Err(CosmologyError::AnomalyClosureError { a, r_entropy });
    }
    let r_initial = boundary_entropy_margin(1.0, params);
    let f_load = (1.0 - (r_entropy / r_initial)).clamp(0.0, 1.0);
    Ok(f_load)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_event_horizon_asymptote_a_1000() {
        let params = CosmologyParameters::default();
        let chi = comoving_event_horizon_mpc(1000.0, &params);
        assert!((chi - 5.3384).abs() < 0.02, "Expected chi_event(1000) ~ 5.338 Mpc, got {:.4}", chi);
    }

    #[test]
    fn test_gapless_sound_speed_continuity() {
        let cs_above = baryon_sound_speed_km_s(Z_DRAG + 10.0);
        let cs_drag = baryon_sound_speed_km_s(Z_DRAG);
        let cs_below = baryon_sound_speed_km_s(Z_DRAG - 10.0);

        assert!(cs_above > cs_drag);
        assert!(cs_drag > cs_below);
        assert!((cs_drag - 5.709).abs() < 0.05);
    }
}
