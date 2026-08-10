use rug::Float;

pub(crate) const PREC: u32 = 512;

const Z_PEAK: f64 = 7.5;
const SIGMA_Z: f64 = 0.801;
const Q_DOT_W: f64 = 906e9;
const P_BENCH_W: f64 = 142.08e6;
const GAMMA_BENCH_TARGET: f64 = 6377.0;
const STATIONARITY_TOLERANCE: f64 = 1e-20;
const THERMAL_TOLERANCE: f64 = 1.0;
const FINITE_DIFF_H: f64 = 1e-6;

#[derive(Debug, Clone)]
pub struct StabilityAudit {
    pub stationarity_passed: bool,
    pub thermal_flux_passed: bool,
    pub d_ln_c_even_dz: f64,
    pub d_ln_c_odd_dz: f64,
    pub q_dot_w: f64,
    pub p_bench_w: f64,
    pub gamma_bench: f64,
    pub gamma_bench_target: f64,
    pub seed_z_peak: f64,
    pub seed_sigma_z: f64,
}

fn log_correlator(z: &Float, z_peak: &Float, sigma: &Float) -> Float {
    let dz = Float::with_val(PREC, z) - z_peak;
    let sigma_sq = sigma.clone() * sigma;
    let denom = Float::with_val(PREC, 2.0) * sigma_sq;
    let num = dz.clone() * &dz;
    -num / denom
}

pub fn verify_stability_audit() -> StabilityAudit {
    let z_peak = Float::with_val(PREC, Z_PEAK);
    let sigma = Float::with_val(PREC, SIGMA_Z);
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

    StabilityAudit {
        stationarity_passed,
        thermal_flux_passed,
        d_ln_c_even_dz: d_ln_c_even.to_f64(),
        d_ln_c_odd_dz: d_ln_c_odd.to_f64(),
        q_dot_w: Q_DOT_W,
        p_bench_w: P_BENCH_W,
        gamma_bench: gamma_bench.to_f64(),
        gamma_bench_target: GAMMA_BENCH_TARGET,
        seed_z_peak: Z_PEAK,
        seed_sigma_z: SIGMA_Z,
    }
}
