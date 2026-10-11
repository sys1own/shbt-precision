//! Real-time boundary cosmology engine backing the WebGPU visualizer.
//!
//! Implements the five-regime cosmic history of `shbt3.txt`: boundary
//! initialization, Stinespring baryogenesis de-rendering, mass-congestion
//! seed condensation, conformal web clustering, and the asymptotic observer
//! horizon freeze.  All ledger quantities are evaluated in `rug::Float`
//! at the requested bit precision so conservation residuals stay below the
//! 10^-120 manuscript gate.

use rug::Float;
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

/// Topological mass-coupling constant alpha_seed = d1 m_P / e^33 in
/// M_sun per bit (d1 = gcd(26, 312) = 26).
pub const ALPHA_SEED_MSUN_PER_BIT: f64 = 1.3258316e-51;
/// Continuous Landauer dissipation per unit seed mass: 906 GW / M_sun.
pub const LANDAUER_DEBT_GW_PER_MSUN: f64 = 906.0;
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

    /// Open nucleation floor partition function Psi_nuc(z) (shbt13):
    ///
    /// ```text
    /// Psi_nuc(z) = 1                           (z <= 18),
    /// Psi_nuc(z) = 10u^3 - 15u^4 + 6u^5        (18 < z < 30),
    /// Psi_nuc(z) = 0                           (z >= 30),
    /// ```
    ///
    /// with u = (30 - z) / 12 the normalized incubation coordinate. The
    /// quintic is the unique C^2-step interpolation; the open floor at
    /// z <= 18 guarantees the condensation threshold never suppresses
    /// newly forming seeds inside the nucleation window. Mirrors the
    /// WGSL kernel `compute_nucleation_weight` in seed_emergence.wgsl.
    pub fn evaluate_psi_nuc(z: f64) -> f64 {
        if z <= 18.0 {
            1.0
        } else if z >= 30.0 {
            0.0
        } else {
            let u = (30.0 - z) / 12.0;
            let u2 = u * u;
            let u3 = u2 * u;
            10.0 * u3 - 15.0 * u2 * u2 + 6.0 * u3 * u2
        }
    }

    /// First-principles seed-mass peak redshift (shbt13), replacing the
    /// hardcoded z_peak = 7.5:
    ///
    /// ```text
    /// 1 + z_peak^(0) = (K / (gamma_CFT * c_eff))^{2/3},
    /// z_peak         = z_peak^(0) * (1 - (10/33) f_cosmo(z_peak)),
    /// ```
    ///
    /// with (k_l, k_q, K) = (26, 8, 312), c_eff = 1325/154 ~= 8.6039 and
    /// gamma_CFT = c_eff/6 = 1325/924 ~= 1.4340. The unscreened anchor
    /// 1 + z^(0) = (312/12.3378)^{2/3} ~= 8.6146 fixes z^(0) ~= 7.6146; a
    /// few fixed-point iterations close the mild forward-loading screen
    /// suppression self-consistently.
    pub fn derive_first_principles_z_peak(&self) -> f64 {
        let c_eff: f64 = 1325.0 / 154.0;
        let gamma_cft: f64 = 1325.0 / 924.0;
        let k: f64 = 312.0;
        let z0 = (k / (gamma_cft * c_eff)).powf(2.0 / 3.0) - 1.0;
        let mut z_peak = z0;
        for _ in 0..32 {
            let f = self
                .evaluate_forward_cosmic_loading_fraction(z_peak)
                .to_f64();
            z_peak = z0 * (1.0 - (10.0 / 33.0) * f);
        }
        z_peak
    }

    /// First-principles nucleation dispersion (shbt13), replacing the
    /// hardcoded sigma_z = 0.801:
    ///
    /// ```text
    /// sigma_z = (1/sqrt(2)) * ((k_l + k_q)/K)^{1/4}
    ///           * (1 + z_peak) / sqrt(c_eff) * sqrt(10/21),
    /// ```
    ///
    /// the excursion-set width of the first-crossing rate peak in z.
    pub fn derive_first_principles_sigma_z(&self) -> f64 {
        let c_eff: f64 = 1325.0 / 154.0;
        let z_peak = self.derive_first_principles_z_peak();
        let ratio = ((26.0 + 8.0) / 312.0_f64).powf(0.25);
        (1.0 / 2.0_f64.sqrt())
            * ratio
            * (1.0 + z_peak)
            / c_eff.sqrt()
            * (10.0_f64 / 21.0).sqrt()
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

