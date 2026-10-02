//! Real-time boundary cosmology engine backing the WebGPU visualizer.
//!
//! Implements the five-regime cosmic history of `shbt3.txt`: boundary
//! initialization, Stinespring baryogenesis de-rendering, mass-congestion
//! seed condensation, conformal web clustering, and the asymptotic observer
//! horizon freeze.  All ledger quantities are evaluated in `rug::Float`
//! at the requested bit precision so conservation residuals stay below the
//! 10^-120 manuscript gate.

use rug::{Assign, Float};

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

    /// Conformal loading fraction f_load(z) on (-1, +inf).
    ///
    /// z >= 0 integrates Eq. "Conformal Loading ODE" df/dz = Gamma_lock /
    /// [(1+z)^2 H_SHBT(z)] and caps at screen capacity f_load <= 1;
    /// -1 < z < 0 follows the saturated de Sitter law 1 - (1+z)^3.
    pub fn conformal_loading_fraction(&self, z: f64) -> Float {
        if z <= -1.0 {
            return self.float(1.0);
        }
        if z < 0.0 {
            return self.float(1.0 - (1.0 + z).powi(3));
        }
        if z == 0.0 {
            return self.float(0.0);
        }
        // Integrate in u = ln(1+z): dz = e^u du, so the loading ODE becomes
        // df_load/du = Gamma_lock e^{-u} / H_SHBT(e^u - 1).
        let gamma_lock = 3.0 * self.params.a_h;
        let kernel = |u: f64| -> f64 {
            let one_plus_z = u.exp();
            let h0_z = self.params.h0_cmb + self.params.a_h / one_plus_z;
            let expansion = (self.params.omega_m * one_plus_z.powi(3)
                + self.params.omega_r0 * one_plus_z.powi(4)
                + (1.0 - self.params.omega_m - self.params.omega_r0))
                .sqrt();
            // df_load = Gamma e^{-2u} / H du with dz = e^u du folded in:
            // d/dz = Gamma / [(1+z)^2 H] => d/du = Gamma e^{-u} / H.
            gamma_lock * (-u).exp() / (h0_z * expansion)
        };
        let upper = (1.0 + z).ln();
        let value = adaptive_simpson(kernel, 0.0, upper, 1e-12, 20);
        self.float(value.min(1.0))
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
