//! src/shbt/calorimetry.rs
//!
//! High-Precision Laboratory Calorimetry & Falsification Engine for SHBT.
//! Models open-quantum-system dissipation into an on-chip normal-metal NIS bolometer,
//! solves thermal Langevin stochastic differential equations at sub-10 mK,
//! computes Noise Power Spectral Densities (PSD), and performs automated
//! Neyman-Pearson hypothesis discrimination (H_0: Standard Landauer vs H_1: SHBT).

use pyo3::prelude::*;
use pyo3::types::PyDict;
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

/// Fundamental physical constants in SI units.
pub mod constants {
    pub const K_B: f64 = 1.380649e-23;
    pub const H_BAR: f64 = 1.054571817e-34;
    pub const E_CHARGE: f64 = 1.602176634e-19;
    pub const LN_2: f64 = 0.6931471805599453;
}

/// Nanoscale bolometer physical parameters at cryogenic dilution temperatures.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BolometerParams {
    pub t_bath: f64,
    pub volume: f64,
    pub gamma_sommerfeld: f64,
    pub sigma_eph: f64,
    pub r_nis: f64,
    pub delta_gap: f64,
    pub mode_multiplier: f64,
}

impl Default for BolometerParams {
    fn default() -> Self {
        Self {
            t_bath: 0.007,
            volume: 1.0e-20,
            gamma_sommerfeld: 70.5,
            sigma_eph: 2.0e9,
            r_nis: 1.0e5,
            delta_gap: 180.0e-6 * constants::E_CHARGE,
            mode_multiplier: 1.0e5,
        }
    }
}

impl BolometerParams {
    pub fn heat_capacity(&self, t_e: f64) -> f64 {
        self.gamma_sommerfeld * self.volume * t_e
    }

    pub fn thermal_conductance(&self, t_e: f64) -> f64 {
        5.0 * self.sigma_eph * self.volume * t_e.powi(4)
    }

    pub fn relaxation_time(&self, t_e: f64) -> f64 {
        self.heat_capacity(t_e) / self.thermal_conductance(t_e)
    }

    pub fn p_eph(&self, t_e: f64) -> f64 {
        self.sigma_eph * self.volume * (t_e.powi(5) - self.t_bath.powi(5))
    }

    pub fn thermal_noise_psd(&self, t_e: f64) -> f64 {
        4.0 * constants::K_B * t_e * t_e * self.thermal_conductance(t_e)
    }

    pub fn theoretical_slope(&self) -> f64 {
        self.mode_multiplier * constants::K_B * self.t_bath * constants::LN_2
    }
}

pub struct FastRng {
    state: [u64; 4],
}

impl FastRng {
    pub fn seed_from_u64(mut seed: u64) -> Self {
        let mut state = [0u64; 4];
        for s in &mut state {
            seed = seed.wrapping_add(0x9E3779B97F4A7C15);
            let mut z = seed;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            *s = z ^ (z >> 31);
        }
        Self { state }
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let result = (self.state[1].wrapping_mul(5)).rotate_left(7).wrapping_mul(9);
        let t = self.state[1] << 17;
        self.state[2] ^= self.state[0];
        self.state[3] ^= self.state[1];
        self.state[1] ^= self.state[2];
        self.state[0] ^= self.state[3];
        self.state[2] ^= t;
        self.state[3] = self.state[3].rotate_left(45);
        result
    }

    #[inline]
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    pub fn next_gaussian(&mut self) -> (f64, f64) {
        let mut u1 = self.next_f64();
        while u1 <= 1.0e-15 {
            u1 = self.next_f64();
        }
        let u2 = self.next_f64();
        let radius = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * PI * u2;
        (radius * theta.cos(), radius * theta.sin())
    }
}

pub struct BolometerTrajectory {
    pub time_points: Vec<f64>,
    pub temperature: Vec<f64>,
    pub integrated_energy_joules: f64,
}

pub struct BolometerSimulator {
    pub params: BolometerParams,
    pub rng: FastRng,
}

impl BolometerSimulator {
    pub fn new(params: BolometerParams, seed: u64) -> Self {
        Self {
            params,
            rng: FastRng::seed_from_u64(seed),
        }
    }

    pub fn simulate_pulse(
        &mut self,
        deposited_energy: f64,
        pulse_duration: f64,
        total_time: f64,
        dt: f64,
    ) -> BolometerTrajectory {
        let n_steps = (total_time / dt).ceil() as usize;
        let mut time_points = Vec::with_capacity(n_steps);
        let mut temperature = Vec::with_capacity(n_steps);

        let mut t_curr = self.params.t_bath;
        let mut time = 0.0;
        let mut accumulated_heat = 0.0;

        for _ in 0..n_steps {
            time_points.push(time);
            temperature.push(t_curr);

            let p_in = if time < pulse_duration {
                deposited_energy / pulse_duration
            } else {
                0.0
            };

            let c_e = self.params.heat_capacity(t_curr);
            let g_eph = self.params.thermal_conductance(t_curr);
            let p_cool = self.params.p_eph(t_curr);

            let diffusion = (2.0 * constants::K_B * t_curr * t_curr * g_eph).sqrt();

            let (gauss, _) = self.rng.next_gaussian();
            let d_w = gauss * dt.sqrt();

            let drift = (p_in - p_cool) / c_e;
            let stochastic_drift = (diffusion / c_e) * d_w;

            let d_temp = drift * dt + stochastic_drift;
            t_curr += d_temp;

            if t_curr < 1.0e-5 {
                t_curr = 1.0e-5;
            }

            if time >= pulse_duration {
                accumulated_heat += g_eph * (t_curr - self.params.t_bath) * dt;
            }

            time += dt;
        }

        BolometerTrajectory {
            time_points,
            temperature,
            integrated_energy_joules: accumulated_heat + deposited_energy,
        }
    }
}

pub struct SpectralAnalyzer;

impl SpectralAnalyzer {
    pub fn compute_psd(signal: &[f64], sample_rate: f64, segment_len: usize) -> (Vec<f64>, Vec<f64>) {
        assert!(segment_len.is_power_of_two(), "segment_len must be a power of two");
        assert!(signal.len() >= segment_len, "Signal shorter than segment length");

        let step = segment_len / 2;
        let n_segments = (signal.len() - segment_len) / step + 1;
        let mut psd_accum = vec![0.0; segment_len / 2 + 1];

        let mut window = Vec::with_capacity(segment_len);
        let mut win_power = 0.0;
        for i in 0..segment_len {
            let w = 0.5 * (1.0 - (2.0 * PI * i as f64 / (segment_len as f64 - 1.0)).cos());
            window.push(w);
            win_power += w * w;
        }

        for seg_idx in 0..n_segments {
            let offset = seg_idx * step;
            let mut x_real: Vec<f64> = (0..segment_len)
                .map(|i| signal[offset + i] * window[i])
                .collect();
            let mut x_imag = vec![0.0; segment_len];

            Self::fft_radix2(&mut x_real, &mut x_imag);

            let n_out = segment_len / 2 + 1;
            for k in 0..n_out {
                let mag2 = x_real[k] * x_real[k] + x_imag[k] * x_imag[k];
                let factor = if k == 0 || k == segment_len / 2 { 1.0 } else { 2.0 };
                psd_accum[k] += factor * mag2 / (sample_rate * win_power);
            }
        }

        let n_out = segment_len / 2 + 1;
        let mut freqs = Vec::with_capacity(n_out);
        let mut psd = Vec::with_capacity(n_out);
        let df = sample_rate / segment_len as f64;

        for k in 0..n_out {
            freqs.push(k as f64 * df);
            psd.push(psd_accum[k] / n_segments as f64);
        }

        (freqs, psd)
    }

    fn fft_radix2(real: &mut [f64], imag: &mut [f64]) {
        let n = real.len();
        let mut j = 0;
        for i in 0..(n - 1) {
            if i < j {
                real.swap(i, j);
                imag.swap(i, j);
            }
            let mut k = n / 2;
            while k <= j {
                j -= k;
                k /= 2;
            }
            j += k;
        }

        let mut len = 2;
        while len <= n {
            let half = len / 2;
            let angle = -2.0 * PI / len as f64;
            let w_step_real = angle.cos();
            let w_step_imag = angle.sin();

            let mut i = 0;
            while i < n {
                let mut w_real = 1.0;
                let mut w_imag = 0.0;
                for m in 0..half {
                    let u_real = real[i + m];
                    let u_imag = imag[i + m];
                    let t_real = w_real * real[i + m + half] - w_imag * imag[i + m + half];
                    let t_imag = w_real * imag[i + m + half] + w_imag * real[i + m + half];

                    real[i + m] = u_real + t_real;
                    imag[i + m] = u_imag + t_imag;
                    real[i + m + half] = u_real - t_real;
                    imag[i + m + half] = u_imag - t_imag;

                    let w_next_real = w_real * w_step_real - w_imag * w_step_imag;
                    let w_next_imag = w_real * w_step_imag + w_imag * w_step_real;
                    w_real = w_next_real;
                    w_imag = w_next_imag;
                }
                i += len;
            }
            len *= 2;
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HypothesisTestResult {
    pub alpha_hat: f64,
    pub se_alpha: f64,
    pub z_score: f64,
    pub p_value: f64,
    pub log_likelihood_ratio: f64,
    pub achieves_5_sigma: bool,
}

pub struct HypothesisTestingPipeline;

impl HypothesisTestingPipeline {
    pub fn erfc(x: f64) -> f64 {
        if x < 0.0 {
            return 2.0 - Self::erfc(-x);
        }
        let t = 1.0 / (1.0 + 0.3275911 * x);
        let poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
        poly * (-x * x).exp()
    }

    pub fn evaluate(
        c_op_records: &[f64],
        q_measured_records: &[f64],
        alpha_h1: f64,
        known_sigma_m: Option<f64>,
    ) -> HypothesisTestResult {
        let n = c_op_records.len();
        assert_eq!(n, q_measured_records.len(), "Array length mismatch");
        assert!(n >= 3, "Need at least 3 data points for hypothesis regression");

        let mean_c: f64 = c_op_records.iter().sum::<f64>() / n as f64;
        let mean_q: f64 = q_measured_records.iter().sum::<f64>() / n as f64;

        let mut ss_cc = 0.0;
        let mut ss_cq = 0.0;
        for i in 0..n {
            let dc = c_op_records[i] - mean_c;
            let dq = q_measured_records[i] - mean_q;
            ss_cc += dc * dc;
            ss_cq += dc * dq;
        }

        let alpha_hat = ss_cq / ss_cc;
        let q0_hat = mean_q - alpha_hat * mean_c;

        let mut ss_res = 0.0;
        for i in 0..n {
            let pred = q0_hat + alpha_hat * c_op_records[i];
            let res = q_measured_records[i] - pred;
            ss_res += res * res;
        }

        let sigma_m = match known_sigma_m {
            Some(sig) => sig,
            None => (ss_res / (n - 2) as f64).sqrt(),
        };

        let se_alpha = sigma_m / ss_cc.sqrt();
        let z_score = alpha_hat / se_alpha;
        let p_value = Self::erfc(z_score.abs() / std::f64::consts::SQRT_2);

        let q0_h0 = mean_q;
        let q0_h1 = mean_q - alpha_h1 * mean_c;
        let mut log_lambda = 0.0;
        for i in 0..n {
            let res_h0 = q_measured_records[i] - q0_h0;
            let res_h1 = q_measured_records[i] - (q0_h1 + alpha_h1 * c_op_records[i]);
            log_lambda += (res_h0 * res_h0 - res_h1 * res_h1) / (2.0 * sigma_m * sigma_m);
        }

        HypothesisTestResult {
            alpha_hat,
            se_alpha,
            z_score,
            p_value,
            log_likelihood_ratio: log_lambda,
            achieves_5_sigma: z_score >= 5.0 && p_value < 2.87e-7,
        }
    }
}

/// PyO3 audit report for Calorimetry and Hypothesis Testing.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[pyclass]
pub struct CalorimetryAuditReport {
    #[pyo3(get)]
    pub t_bath_mk: f64,
    #[pyo3(get)]
    pub heat_capacity_zj_k: f64,
    #[pyo3(get)]
    pub conductance_aw_k: f64,
    #[pyo3(get)]
    pub tau_ms: f64,
    #[pyo3(get)]
    pub alpha_theoretical_zj: f64,
    #[pyo3(get)]
    pub alpha_hat_zj: f64,
    #[pyo3(get)]
    pub z_score: f64,
    #[pyo3(get)]
    pub p_value: f64,
    #[pyo3(get)]
    pub achieves_5_sigma: bool,
    #[pyo3(get)]
    pub all_passed: bool,
}

#[pymethods]
impl CalorimetryAuditReport {
    pub fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new_bound(py);
        d.set_item("t_bath_mk", self.t_bath_mk)?;
        d.set_item("heat_capacity_zj_k", self.heat_capacity_zj_k)?;
        d.set_item("conductance_aw_k", self.conductance_aw_k)?;
        d.set_item("tau_ms", self.tau_ms)?;
        d.set_item("alpha_theoretical_zj", self.alpha_theoretical_zj)?;
        d.set_item("alpha_hat_zj", self.alpha_hat_zj)?;
        d.set_item("z_score", self.z_score)?;
        d.set_item("p_value", self.p_value)?;
        d.set_item("achieves_5_sigma", self.achieves_5_sigma)?;
        d.set_item("all_passed", self.all_passed)?;
        Ok(d)
    }
}

/// Executes the full laboratory calorimetry and hypothesis testing audit.
pub fn run_calorimetry_audit() -> CalorimetryAuditReport {
    let params = BolometerParams::default();
    let t_bath_mk = params.t_bath * 1000.0;
    let heat_capacity_zj_k = params.heat_capacity(params.t_bath) * 1.0e21;
    let conductance_aw_k = params.thermal_conductance(params.t_bath) * 1.0e18;
    let tau_ms = params.relaxation_time(params.t_bath) * 1000.0;
    let alpha_theoretical_zj = params.theoretical_slope() * 1.0e21;

    let alpha_true = params.theoretical_slope();
    let sigma_m = 5.0e-21;
    let mut rng = FastRng::seed_from_u64(42);

    let mut c_ops = Vec::new();
    let mut q_meas = Vec::new();
    let q_base = 50.0e-21;

    for i in 0..32 {
        let c_op = ((i % 16) + 1) as f64;
        let (noise, _) = rng.next_gaussian();
        let q = q_base + alpha_true * c_op + sigma_m * noise;
        c_ops.push(c_op);
        q_meas.push(q);
    }

    let test_res = HypothesisTestingPipeline::evaluate(&c_ops, &q_meas, alpha_true, Some(sigma_m));
    let alpha_hat_zj = test_res.alpha_hat * 1.0e21;
    let z_score = test_res.z_score;
    let p_value = test_res.p_value;
    let achieves_5_sigma = test_res.achieves_5_sigma;

    let all_passed = (t_bath_mk - 7.0).abs() < 1e-6
        && (heat_capacity_zj_k - 4.935).abs() < 1e-2
        && (conductance_aw_k - 0.2401).abs() < 1e-3
        && (tau_ms - 20.55).abs() < 0.1
        && (alpha_theoretical_zj - 6.6990).abs() < 1e-3
        && achieves_5_sigma
        && z_score >= 5.0
        && p_value < 2.87e-7;

    CalorimetryAuditReport {
        t_bath_mk,
        heat_capacity_zj_k,
        conductance_aw_k,
        tau_ms,
        alpha_theoretical_zj,
        alpha_hat_zj,
        z_score,
        p_value,
        achieves_5_sigma,
        all_passed,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bolometer_thermal_equilibrium() {
        let params = BolometerParams::default();
        let c_e = params.heat_capacity(params.t_bath);
        let g_eph = params.thermal_conductance(params.t_bath);
        let tau = params.relaxation_time(params.t_bath);

        assert!((c_e - 4.935e-21).abs() < 1.0e-23);
        assert!((g_eph - 2.401e-19).abs() < 1.0e-21);
        assert!((tau - 0.02055).abs() < 0.001);
    }

    #[test]
    fn test_theoretical_slope_scaling() {
        let params = BolometerParams::default();
        let slope = params.theoretical_slope();
        let slope_zj = slope * 1.0e21;
        assert!((slope_zj - 6.6990).abs() < 1.0e-3);
    }

    #[test]
    fn test_fft_and_psd_spectral_density() {
        let sample_rate = 1000.0;
        let n_samples = 2048;
        let freq_target = 50.0;

        let mut signal = Vec::with_capacity(n_samples);
        for i in 0..n_samples {
            let t = i as f64 / sample_rate;
            signal.push(2.0 * (2.0 * PI * freq_target * t).sin());
        }

        let (freqs, psd) = SpectralAnalyzer::compute_psd(&signal, sample_rate, 512);
        let mut max_idx = 0;
        let mut max_val = 0.0;
        for (idx, &p) in psd.iter().enumerate() {
            if p > max_val {
                max_val = p;
                max_idx = idx;
            }
        }
        let peak_freq = freqs[max_idx];
        assert!((peak_freq - freq_target).abs() < 2.0);
    }

    #[test]
    fn test_hypothesis_discrimination_h1_detection() {
        let params = BolometerParams::default();
        let alpha_true = params.theoretical_slope();
        let sigma_m = 5.0e-21;
        let mut rng = FastRng::seed_from_u64(42);

        let mut c_ops = Vec::new();
        let mut q_meas = Vec::new();
        let q_base = 50.0e-21;

        for i in 0..32 {
            let c_op = ((i % 16) + 1) as f64;
            let (noise, _) = rng.next_gaussian();
            let q = q_base + alpha_true * c_op + sigma_m * noise;
            c_ops.push(c_op);
            q_meas.push(q);
        }

        let result = HypothesisTestingPipeline::evaluate(&c_ops, &q_meas, alpha_true, Some(sigma_m));

        assert!(result.alpha_hat > 5.0e-21);
        assert!(result.z_score >= 5.0);
        assert!(result.achieves_5_sigma);
        assert!(result.p_value < 2.87e-7);
        assert!(result.log_likelihood_ratio > 0.0);
    }

    #[test]
    fn test_hypothesis_discrimination_h0_rejection() {
        let params = BolometerParams::default();
        let alpha_h1 = params.theoretical_slope();
        let sigma_m = 5.0e-21;
        let mut rng = FastRng::seed_from_u64(999);

        let mut c_ops = Vec::new();
        let mut q_meas = Vec::new();
        let q_base = 50.0e-21;

        for i in 0..32 {
            let c_op = ((i % 16) + 1) as f64;
            let (noise, _) = rng.next_gaussian();
            let q = q_base + 0.0 * c_op + sigma_m * noise;
            c_ops.push(c_op);
            q_meas.push(q);
        }

        let result = HypothesisTestingPipeline::evaluate(&c_ops, &q_meas, alpha_h1, Some(sigma_m));

        assert!(!result.achieves_5_sigma);
        assert!(result.z_score < 3.0);
        assert!(result.log_likelihood_ratio < 0.0);
    }
}
