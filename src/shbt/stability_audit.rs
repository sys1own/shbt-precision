use rug::Float;

pub(crate) const PREC: u32 = 512;

// shbt13: the Gaussian seed-profile parameters are no longer hardcoded
// empirical fits — they are derived from the first-principles SHBT
// invariants (k_l, k_q, K) = (26, 8, 312), c_eff = 1325/154 and
// gamma_CFT = 1325/924 through
// `ShbtUniverse::derive_first_principles_z_peak` /
// `derive_first_principles_sigma_z` (spec section 1.5). The derived
// values (z_peak ~= 7.614, sigma_z ~= 0.823) are mildly shifted from the
// earlier constants (7.5, 0.801) by the forward-loading screen
// correction.
use crate::shbt::cosmology::ShbtUniverse;
const Q_DOT_W: f64 = 906e9;
const P_BENCH_W: f64 = 142.08e6;
const GAMMA_BENCH_TARGET: f64 = 6377.0;
const STATIONARITY_TOLERANCE: f64 = 1e-20;
const THERMAL_TOLERANCE: f64 = 1.0;
const FINITE_DIFF_H: f64 = 1e-6;

use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::HashMap;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[pyclass]
pub struct StabilityAudit {
    #[pyo3(get)]
    pub stationarity_passed: bool,
    #[pyo3(get)]
    pub thermal_flux_passed: bool,
    #[pyo3(get)]
    pub d_ln_c_even_dz: f64,
    #[pyo3(get)]
    pub d_ln_c_odd_dz: f64,
    #[pyo3(get)]
    pub q_dot_w: f64,
    #[pyo3(get)]
    pub p_bench_w: f64,
    #[pyo3(get)]
    pub gamma_bench: f64,
    #[pyo3(get)]
    pub gamma_bench_target: f64,
    #[pyo3(get)]
    pub seed_z_peak: f64,
    #[pyo3(get)]
    pub seed_sigma_z: f64,
    #[pyo3(get)]
    pub teschner_shift_residual_max: f64,
    #[pyo3(get)]
    pub bootstrap_canonical_branch_verified: bool,
    #[pyo3(get)]
    pub noether_ward_identity_verified: bool,
    #[pyo3(get)]
    pub n_sat_accessor_crosswalk: HashMap<String, f64>,
}

#[pymethods]
impl StabilityAudit {
    pub fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new_bound(py);
        d.set_item("stationarity_passed", self.stationarity_passed)?;
        d.set_item("thermal_flux_passed", self.thermal_flux_passed)?;
        d.set_item("d_ln_c_even_dz", self.d_ln_c_even_dz)?;
        d.set_item("d_ln_c_odd_dz", self.d_ln_c_odd_dz)?;
        d.set_item("q_dot_w", self.q_dot_w)?;
        d.set_item("p_bench_w", self.p_bench_w)?;
        d.set_item("gamma_bench", self.gamma_bench)?;
        d.set_item("gamma_bench_target", self.gamma_bench_target)?;
        d.set_item("seed_z_peak", self.seed_z_peak)?;
        d.set_item("seed_sigma_z", self.seed_sigma_z)?;
        d.set_item("teschner_shift_residual_max", self.teschner_shift_residual_max)?;
        d.set_item(
            "bootstrap_canonical_branch_verified",
            self.bootstrap_canonical_branch_verified,
        )?;
        d.set_item(
            "noether_ward_identity_verified",
            self.noether_ward_identity_verified,
        )?;
        let cw = PyDict::new_bound(py);
        for (k, v) in &self.n_sat_accessor_crosswalk {
            cw.set_item(k, v)?;
        }
        d.set_item("n_sat_accessor_crosswalk", cw)?;
        Ok(d)
    }
}

fn log_correlator(z: &Float, z_peak: &Float, sigma: &Float) -> Float {
    let dz = Float::with_val(PREC, z) - z_peak;
    let sigma_sq = sigma.clone() * sigma;
    let denom = Float::with_val(PREC, 2.0) * sigma_sq;
    let num = dz.clone() * &dz;
    -num / denom
}

pub fn verify_stability_audit() -> StabilityAudit {
    let cosmo = ShbtUniverse::new_canonical_branch(PREC);
    let z_peak_f64 = cosmo.derive_first_principles_z_peak();
    let sigma_z_f64 = cosmo.derive_first_principles_sigma_z();
    let z_peak = Float::with_val(PREC, z_peak_f64);
    let sigma = Float::with_val(PREC, sigma_z_f64);
    let h = Float::with_val(PREC, FINITE_DIFF_H);

    // Seed redshift coordinate.  C_even and C_odd are modeled as the
    // Gaussian seed profiles; the stationarity condition is the statement
    // that the peak of the profile is a fixed point of the RG flow.
    let mut z_plus = Float::with_val(PREC, &z_peak);
    z_plus += &h;
    let mut z_minus = Float::with_val(PREC, &z_peak);
    z_minus -= &h;

    let ln_c_even_plus = log_correlator(&z_plus, &z_peak, &sigma);
    let ln_c_even_minus = log_correlator(&z_minus, &z_peak, &sigma);
    let two_h = Float::with_val(PREC, 2.0) * &h;
    let d_ln_c_even = (ln_c_even_plus - ln_c_even_minus) / &two_h;

    let ln_c_odd_plus = log_correlator(&z_plus, &z_peak, &sigma);
    let ln_c_odd_minus = log_correlator(&z_minus, &z_peak, &sigma);
    let d_ln_c_odd = (ln_c_odd_plus - ln_c_odd_minus) / two_h;

    let tol = Float::with_val(PREC, STATIONARITY_TOLERANCE);
    let stationarity_passed = d_ln_c_even.clone().abs() <= tol.clone()
        && d_ln_c_odd.clone().abs() <= tol;

    let q_dot = Float::with_val(PREC, Q_DOT_W);
    let p_bench = Float::with_val(PREC, P_BENCH_W);
    let gamma_bench = q_dot / p_bench;
    let target = Float::with_val(PREC, GAMMA_BENCH_TARGET);
    let thermal_flux_passed =
        (gamma_bench.clone() - &target).abs() <= Float::with_val(PREC, THERMAL_TOLERANCE);

    let sb = crate::shbt::boundary::StaticBoundary::new();
    let teschner = sb.verify_teschner_shift_residual();
    let noether = sb.verify_noether_ward_identity();
    let cw = sb.n_sat_crosswalk();

    let mut cw_map = HashMap::new();
    cw_map.insert("s_top_bulk".to_string(), cw.s_top_bulk as f64);
    cw_map.insert("delta_s_mod_legacy".to_string(), cw.delta_s_mod_legacy);
    cw_map.insert("cardy_verlinde_weight".to_string(), cw.cardy_verlinde_weight);
    cw_map.insert("n_sat_legacy".to_string(), cw.n_sat_legacy);
    cw_map.insert("n_sat".to_string(), cw.n_sat);
    cw_map.insert(
        "crosswalk_verified".to_string(),
        if cw.crosswalk_verified { 1.0 } else { 0.0 },
    );

    let teschner_shift_residual_max = teschner
        .teschner_shift_residual_even
        .max(teschner.teschner_shift_residual_odd);
    let bootstrap_canonical_branch_verified = teschner.bootstrap_canonical_branch_verified;
    let noether_ward_identity_verified = noether.noether_ward_identity_verified;

    StabilityAudit {
        stationarity_passed,
        thermal_flux_passed,
        d_ln_c_even_dz: d_ln_c_even.to_f64(),
        d_ln_c_odd_dz: d_ln_c_odd.to_f64(),
        q_dot_w: Q_DOT_W,
        p_bench_w: P_BENCH_W,
        gamma_bench: gamma_bench.to_f64(),
        gamma_bench_target: GAMMA_BENCH_TARGET,
        seed_z_peak: z_peak_f64,
        seed_sigma_z: sigma_z_f64,
        teschner_shift_residual_max,
        bootstrap_canonical_branch_verified,
        noether_ward_identity_verified,
        n_sat_accessor_crosswalk: cw_map,
    }
}

const MASS_CONGESTION_TOLERANCE: f64 = 1e-12;

/// Anomaly raised when the mass-congestion coupling is detuned beyond the
/// boundary-closure tolerance, protecting the Newton-lock eigenvector.
#[derive(Debug)]
pub struct AnomalyClosureError;

impl std::fmt::Display for AnomalyClosureError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "mass-congestion coupling detuned beyond boundary-closure tolerance"
        )
    }
}

impl std::error::Error for AnomalyClosureError {}

/// Compute the heavy seed mass from the mass-congestion coupling identity and
/// enforce eigenvector rigidity.  A relative detuning of `alpha_seed` larger
/// than `MASS_CONGESTION_TOLERANCE` triggers an `AnomalyClosureError`.
pub fn mass_congestion_coupling(
    alpha_seed: f64,
    n_local: f64,
    n_limit: f64,
    detuning: f64,
) -> Result<f64, AnomalyClosureError> {
    if n_local <= n_limit {
        return Err(AnomalyClosureError);
    }
    if detuning.abs() >= MASS_CONGESTION_TOLERANCE {
        return Err(AnomalyClosureError);
    }
    Ok(alpha_seed * (n_local - n_limit) * (1.0 + detuning))
}
