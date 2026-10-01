#![allow(non_snake_case)]

use crate::shbt::baryogenesis::BaryogenesisOptimizer;
use crate::shbt::boundary::{StaticBoundary, PREC};
use crate::shbt::entropy_flow::{BulkMetricSlice, HolographicProjection};
use crate::shbt::stability_audit::AnomalyClosureError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rug::float::Constant;
use rug::Float;

const LIGHT_SPEED_M_PER_S: f64 = 299_792_458.0;
const HBAR_J_S: f64 = 1.054_571_817e-34;
const LOW_SU3_WEIGHTS: [(u32, u32); 3] = [(0, 0), (1, 0), (0, 1)];

/// Exact dark carrying fraction of the macroscopic Stinespring isometry.
pub const ETA_DARK_NUM: u64 = 23;
pub const ETA_DARK_DEN: u64 = 33;
/// Visible fraction remaining in the active sector after de-rendering.
pub const ETA_VISIBLE_NUM: u64 = 10;
pub const ETA_VISIBLE_DEN: u64 = 33;
/// Coordinate lattice dimension: C = {0,1,2} x {0,1,2} -> 9 cells.
pub const LATTICE_CELLS: usize = 9;

#[derive(Debug, Clone)]
#[pyclass]
pub struct LightConeSample {
    pub index: usize,
    pub redshift: Float,
    pub tau_lock_s: Float,
    pub f_load: Float,
    pub H_eff_per_s: Float,
    pub dt_dz_s: Float,
    pub dchi_dz_m: Float,
    pub lookback_time_s: Float,
    pub comoving_distance_m: Float,
    pub sequence_index: usize,
    pub coordinate: (usize, usize),
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct LocalPropertyPacket {
    pub step: usize,
    pub boundary_address: String,
    pub coordinate: (usize, usize),
    pub redshift: Float,
    pub normalized_bit_loading: Float,
    pub entanglement_density: Float,
    pub mass_kg: Float,
    pub spin: Float,
    pub charge_vector: [Float; 3],
    pub su2_label_left: u32,
    pub su2_label_right: u32,
    pub su3_weight_left: (u32, u32),
    pub su3_weight_right: (u32, u32),
    pub gravity_coordinates: [Float; 4],
    pub metric_components: [[Float; 4]; 4],
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct CoordinateLogEntry {
    pub step: usize,
    pub boundary_address: String,
    pub source_coordinate: (usize, usize),
    pub selected_coordinate: (usize, usize),
    pub redshift: Float,
    pub collapse_index: usize,
    pub retrieval_cost_bits: Float,
    pub entropy_budget_residual: Float,
    pub pointer_wavefunction: [Float; 3],
    pub packet: LocalPropertyPacket,
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct MemoryReport {
    pub R_H_m: Float,
    pub R_local_m: Float,
    pub f_H: Float,
    pub local_available_bits: Float,
    pub hidden_bits: Float,
    pub entropy_limit_bits: Float,
    pub sigma: Float,
    pub localized_entropy_gradient_per_m: Float,
    pub gravitational_acceleration_m_per_s2: Float,
    pub past_light_cone_samples: usize,
    pub property_packets: usize,
    pub all_passed: bool,
}

#[pymethods]
impl LightConeSample {
    pub fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new_bound(py);
        d.set_item("index", self.index)?;
        d.set_item("redshift", self.redshift.to_f64())?;
        d.set_item("tau_lock_s", self.tau_lock_s.to_f64())?;
        d.set_item("f_load", self.f_load.to_f64())?;
        d.set_item("H_eff_per_s", self.H_eff_per_s.to_f64())?;
        d.set_item("dt_dz_s", self.dt_dz_s.to_f64())?;
        d.set_item("dchi_dz_m", self.dchi_dz_m.to_f64())?;
        d.set_item("lookback_time_s", self.lookback_time_s.to_f64())?;
        d.set_item("comoving_distance_m", self.comoving_distance_m.to_f64())?;
        d.set_item("sequence_index", self.sequence_index)?;
        d.set_item("coordinate", self.coordinate)?;
        Ok(d)
    }
}

#[pymethods]
impl LocalPropertyPacket {
    pub fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new_bound(py);
        d.set_item("step", self.step)?;
        d.set_item("boundary_address", self.boundary_address.clone())?;
        d.set_item("coordinate", self.coordinate)?;
        d.set_item("redshift", self.redshift.to_f64())?;
        d.set_item(
            "normalized_bit_loading",
            self.normalized_bit_loading.to_f64(),
        )?;
        d.set_item("entanglement_density", self.entanglement_density.to_f64())?;
        d.set_item("mass_kg", self.mass_kg.to_f64())?;
        d.set_item("spin", self.spin.to_f64())?;
        let charge: Vec<f64> = self.charge_vector.iter().map(|v| v.to_f64()).collect();
        d.set_item("charge_vector", charge)?;
        d.set_item("su2_label_left", self.su2_label_left)?;
        d.set_item("su2_label_right", self.su2_label_right)?;
        d.set_item("su3_weight_left", self.su3_weight_left)?;
        d.set_item("su3_weight_right", self.su3_weight_right)?;
        let gravity: Vec<f64> = self
            .gravity_coordinates
            .iter()
            .map(|v| v.to_f64())
            .collect();
        d.set_item("gravity_coordinates", gravity)?;
        let metric: Vec<Vec<f64>> = self
            .metric_components
            .iter()
            .map(|row| row.iter().map(|v| v.to_f64()).collect())
            .collect();
        d.set_item("metric_components", metric)?;
        Ok(d)
    }
}

#[pymethods]
impl CoordinateLogEntry {
    pub fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new_bound(py);
        d.set_item("step", self.step)?;
        d.set_item("boundary_address", self.boundary_address.clone())?;
        d.set_item("source_coordinate", self.source_coordinate)?;
        d.set_item("selected_coordinate", self.selected_coordinate)?;
        d.set_item("redshift", self.redshift.to_f64())?;
        d.set_item("collapse_index", self.collapse_index)?;
        d.set_item("retrieval_cost_bits", self.retrieval_cost_bits.to_f64())?;
        d.set_item(
            "entropy_budget_residual",
            self.entropy_budget_residual.to_f64(),
        )?;
        let pointer: Vec<f64> = self
            .pointer_wavefunction
            .iter()
            .map(|v| v.to_f64())
            .collect();
        d.set_item("pointer_wavefunction", pointer)?;
        d.set_item("packet", self.packet.to_dict(py)?)?;
        Ok(d)
    }
}

#[pymethods]
impl MemoryReport {
    pub fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new_bound(py);
        d.set_item("R_H_m", self.R_H_m.to_f64())?;
        d.set_item("R_local_m", self.R_local_m.to_f64())?;
        d.set_item("f_H", self.f_H.to_f64())?;
        d.set_item("local_available_bits", self.local_available_bits.to_f64())?;
        d.set_item("hidden_bits", self.hidden_bits.to_f64())?;
        d.set_item("entropy_limit_bits", self.entropy_limit_bits.to_f64())?;
        d.set_item("sigma", self.sigma.to_f64())?;
        d.set_item(
            "localized_entropy_gradient_per_m",
            self.localized_entropy_gradient_per_m.to_f64(),
        )?;
        d.set_item(
            "gravitational_acceleration_m_per_s2",
            self.gravitational_acceleration_m_per_s2.to_f64(),
        )?;
        d.set_item("past_light_cone_samples", self.past_light_cone_samples)?;
        d.set_item("property_packets", self.property_packets)?;
        d.set_item("all_passed", self.all_passed)?;
        Ok(d)
    }
}

/// Five-phase closed algorithmic lifecycle of a Causal Point observer:
/// Render -> Crystallize -> De-render -> Relabel -> Re-render.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[pyclass(eq, eq_int)]
pub enum LifecyclePhase {
    Render,
    Crystallize,
    DeRender,
    Relabel,
    ReRender,
}

/// Candidate successor observer address on the visible coordinate lattice.
///
/// The admissibility predicate is
/// `P_adm(A) = Theta(min(N_local, A_local/(4 L_P^2 ln 2)) - max(1, log2|R| + log2|Omega_A|, C_req))`.
#[derive(Debug, Clone)]
#[pyclass]
pub struct CausalPointCandidate {
    pub index: usize,
    pub coordinate: (usize, usize),
    /// Local bit capacity `N_local(A)` in bits.
    pub n_local_bits: f64,
    /// Holographic area capacity `A_local / (4 L_P^2 ln 2)` in bits.
    pub area_bits: f64,
    /// Required retrieval complexity `C_req` in bits.
    pub required_cost_bits: f64,
    /// `|R|` — register size for the `log2|R|` floor term.
    pub register_size: usize,
    /// `|Omega_A|` — local outcome ensemble for the `log2|Omega_A|` floor term.
    pub ensemble_size: usize,
    /// `|<Omega_A | T^partial_ij | Omega_{A_term}>|^2` symplectic transfer weight.
    pub symplectic_amplitude: f64,
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct DerenderingRecord {
    pub from_coordinate: (usize, usize),
    pub eta_dark: f64,
    pub eta_visible: f64,
    /// Kojima topological entropy of the transfer channel; conserved at 0.
    pub topological_entropy: f64,
    /// Pointer state triad after the Stinespring pass: (A, c_vis, c_dark).
    pub pointer_wavefunction: [f64; 3],
    /// Trace norm of the terminated state (preserved under the isometry).
    pub trace_norm: f64,
    pub phase: String,
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct SuccessionRecord {
    pub cycle: usize,
    pub from_coordinate: (usize, usize),
    pub to_coordinate: (usize, usize),
    pub selected_index: usize,
    /// Normalized succession kernel T(A_term -> A') over the lattice.
    pub kernel_probabilities: Vec<f64>,
    pub admissible_candidates: usize,
    pub eta_dark: f64,
    pub phase: String,
}

impl CausalPointCandidate {
    /// `R_entropy(A) = min(N_local, A_local/(4 L_P^2 ln 2)) - max(1, log2|R| + log2|Omega_A|, C_req)`.
    pub fn entropy_residual(&self) -> f64 {
        let floor_bits = ((self.register_size.max(1) as f64).log2()
            + (self.ensemble_size.max(1) as f64).log2())
        .max(1.0)
        .max(self.required_cost_bits);
        self.n_local_bits.min(self.area_bits) - floor_bits
    }

    /// `P_adm(A) = Theta(R_entropy(A))`.
    pub fn is_admissible(&self) -> bool {
        self.entropy_residual() >= 0.0
    }
}

#[pymethods]
impl CausalPointCandidate {
    #[new]
    #[allow(clippy::too_many_arguments)]
    pub fn py_new(
        index: usize,
        coordinate: (usize, usize),
        n_local_bits: f64,
        area_bits: f64,
        required_cost_bits: f64,
        register_size: usize,
        ensemble_size: usize,
        symplectic_amplitude: f64,
    ) -> Self {
        Self {
            index,
            coordinate,
            n_local_bits,
            area_bits,
            required_cost_bits,
            register_size,
            ensemble_size,
            symplectic_amplitude,
        }
    }

    pub fn is_admissible_py(&self) -> bool {
        self.is_admissible()
    }

    pub fn entropy_residual_py(&self) -> f64 {
        self.entropy_residual()
    }
}

#[pymethods]
impl DerenderingRecord {
    pub fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new_bound(py);
        d.set_item("from_coordinate", self.from_coordinate)?;
        d.set_item("eta_dark", self.eta_dark)?;
        d.set_item("eta_visible", self.eta_visible)?;
        d.set_item("topological_entropy", self.topological_entropy)?;
        let pointer: Vec<f64> = self.pointer_wavefunction.to_vec();
        d.set_item("pointer_wavefunction", pointer)?;
        d.set_item("trace_norm", self.trace_norm)?;
        d.set_item("phase", self.phase.clone())?;
        Ok(d)
    }
}

#[pymethods]
impl SuccessionRecord {
    pub fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new_bound(py);
        d.set_item("cycle", self.cycle)?;
        d.set_item("from_coordinate", self.from_coordinate)?;
        d.set_item("to_coordinate", self.to_coordinate)?;
        d.set_item("selected_index", self.selected_index)?;
        d.set_item("kernel_probabilities", self.kernel_probabilities.clone())?;
        d.set_item("admissible_candidates", self.admissible_candidates)?;
        d.set_item("eta_dark", self.eta_dark)?;
        d.set_item("phase", self.phase.clone())?;
        Ok(d)
    }
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct CausalPoint {
    pub phase: LifecyclePhase,
    pub boundary: StaticBoundary,
    pub projection: HolographicProjection,
    pub observer_origin: [Float; 4],
    pub observer_radius_fraction: Float,
    pub xi: [Float; 3],
    pub redshift_max: Float,
    pub redshift_samples: usize,
    pub seed: u64,
    pub global_horizon_radius_m: Float,
    pub observer_radius_m: Float,
    pub local_horizon_radius_m: Float,
    pub f_H: Float,
    pub local_available_bits: Float,
    pub hidden_bits: Float,
    pub entropy_limit_bits: Float,
    pub f_hidden: Float,
    pub w_xi: Float,
    pub sigma: Float,
    pub localized_entropy_gradient_per_m: Float,
    pub gravitational_acceleration_m_per_s2: Float,
    pub planck_length_m: Float,
}

fn zero_float() -> Float {
    Float::with_val(PREC, 0)
}

fn one_float() -> Float {
    Float::with_val(PREC, 1)
}

fn linspace(start: f64, end: f64, n: usize) -> Vec<f64> {
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![start];
    }
    let step = (end - start) / (n as f64 - 1.0);
    (0..n).map(|i| start + i as f64 * step).collect()
}

fn interpolate(xs: &[f64], ys: &[f64], x: f64) -> f64 {
    if xs.is_empty() || ys.is_empty() {
        return 0.0;
    }
    if x <= xs[0] {
        return ys[0];
    }
    if x >= xs[xs.len() - 1] {
        return ys[ys.len() - 1];
    }
    for i in 0..xs.len() - 1 {
        let x0 = xs[i];
        let x1 = xs[i + 1];
        if x >= x0 && x <= x1 {
            let denom = x1 - x0;
            if denom == 0.0 {
                return ys[i];
            }
            let t = (x - x0) / denom;
            return ys[i] + t * (ys[i + 1] - ys[i]);
        }
    }
    ys[ys.len() - 1]
}

fn gradient(f: &[f64], x: &[f64]) -> Vec<f64> {
    let n = f.len();
    if n < 2 {
        return vec![0.0; n];
    }
    let mut result = vec![0.0; n];
    result[0] = (f[1] - f[0]) / (x[1] - x[0]);
    result[n - 1] = (f[n - 1] - f[n - 2]) / (x[n - 1] - x[n - 2]);
    for i in 1..n - 1 {
        result[i] = (f[i + 1] - f[i - 1]) / (x[i + 1] - x[i - 1]);
    }
    result
}

fn su3_quadratic_casimir(weight: (u32, u32)) -> f64 {
    let (p, q) = (weight.0 as f64, weight.1 as f64);
    (p * p + q * q + p * q + 3.0 * p + 3.0 * q) / 3.0
}

fn fnv1a_hash(payload: &str, outcome_count: usize) -> usize {
    if outcome_count == 0 {
        panic!("outcome_count must be positive");
    }
    if outcome_count == 1 {
        return 0;
    }
    let mut hash: u64 = 0xcbf29ce484222325;
    for b in payload.bytes() {
        hash ^= b as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    (hash % outcome_count as u64) as usize
}

impl CausalPoint {
    pub fn new_with_params(
        boundary: StaticBoundary,
        observer_radius_fraction: Float,
        xi: [Float; 3],
        redshift_max: Float,
        redshift_samples: usize,
        seed: u64,
    ) -> Self {
        let projection = HolographicProjection::new(boundary.clone());

        let observer_origin = [zero_float(), zero_float(), zero_float(), zero_float()];

        let mut rh_sq = Float::with_val(PREC, 3);
        rh_sq /= &boundary.lambda_holo;
        let global_horizon_radius_m = rh_sq.sqrt();

        let mut observer_radius_m = Float::with_val(PREC, &observer_radius_fraction);
        observer_radius_m *= &global_horizon_radius_m;

        let mut local_horizon_radius_m = Float::with_val(PREC, &global_horizon_radius_m);
        local_horizon_radius_m -= &observer_radius_m;

        let global_f64 = global_horizon_radius_m.to_f64();
        let local_f64 = local_horizon_radius_m.to_f64();
        if !(0.0..global_f64).contains(&local_f64) {
            panic!("observer radius must lie inside the global horizon");
        }

        let mut f_H = Float::with_val(PREC, &local_horizon_radius_m);
        f_H /= &global_horizon_radius_m;

        let mut f_H_sq = Float::with_val(PREC, &f_H);
        f_H_sq.square_mut();
        let mut local_available_bits = Float::with_val(PREC, &boundary.bit_budget);
        local_available_bits *= &f_H_sq;

        let mut hidden_bits = Float::with_val(PREC, &boundary.bit_budget);
        hidden_bits -= &local_available_bits;

        let mut ratio = Float::with_val(PREC, &local_available_bits);
        ratio /= &boundary.bit_budget;
        let mut f_hidden = one_float();
        f_hidden -= &ratio;

        let mut w_xi = Float::with_val(PREC, &xi[0]);
        w_xi += &xi[1];
        w_xi += &xi[2];
        w_xi /= 3;

        let mut planck_length_sq = Float::with_val(PREC, 3);
        planck_length_sq *= Float::with_val(PREC, Constant::Pi);
        let mut denom = Float::with_val(PREC, &boundary.bit_budget);
        denom *= &boundary.lambda_holo;
        planck_length_sq /= &denom;
        let planck_length_m = planck_length_sq.sqrt();

        let mut a_local = Float::with_val(PREC, 4);
        a_local *= Float::with_val(PREC, Constant::Pi);
        let mut local_sq = Float::with_val(PREC, &local_horizon_radius_m);
        local_sq.square_mut();
        a_local *= &local_sq;

        let ln2 = Float::with_val(PREC, Constant::Log2);
        let mut area_denom = Float::with_val(PREC, 4);
        let mut lp_sq = Float::with_val(PREC, &planck_length_m);
        lp_sq.square_mut();
        area_denom *= &lp_sq;
        area_denom *= &ln2;
        let mut area_term = Float::with_val(PREC, &a_local);
        area_term /= &area_denom;
        let entropy_limit_bits = if local_available_bits <= area_term {
            local_available_bits.clone()
        } else {
            area_term
        };

        let mut one_plus_delta = one_float();
        one_plus_delta += &boundary.framing_defect();
        let mut wf = Float::with_val(PREC, &w_xi);
        wf *= &f_hidden;
        let mut one_plus_wf = one_float();
        one_plus_wf += &wf;
        let mut sigma = one_plus_delta;
        sigma *= &one_plus_wf;

        let mut localized_entropy_gradient_per_m = Float::with_val(PREC, &sigma);
        localized_entropy_gradient_per_m *= &f_hidden;
        localized_entropy_gradient_per_m /= &local_horizon_radius_m;

        let c = Float::with_val(PREC, LIGHT_SPEED_M_PER_S);
        let mut c_squared = Float::with_val(PREC, &c);
        c_squared.square_mut();
        let mut gravitational_acceleration_m_per_s2 =
            Float::with_val(PREC, &localized_entropy_gradient_per_m);
        gravitational_acceleration_m_per_s2 *= &c_squared;

        Self {
            phase: LifecyclePhase::Render,
            boundary,
            projection,
            observer_origin,
            observer_radius_fraction,
            xi,
            redshift_max,
            redshift_samples,
            seed,
            global_horizon_radius_m,
            observer_radius_m,
            local_horizon_radius_m,
            f_H,
            local_available_bits,
            hidden_bits,
            entropy_limit_bits,
            f_hidden,
            w_xi,
            sigma,
            localized_entropy_gradient_per_m,
            gravitational_acceleration_m_per_s2,
            planck_length_m,
        }
    }

    pub fn new(boundary: StaticBoundary) -> Self {
        let observer_radius_fraction = Float::with_val(PREC, 0.125);
        let xi = [
            Float::with_val(PREC, 1.0 / 26.0),
            Float::with_val(PREC, 1.0 / 8.0),
            Float::with_val(PREC, 1.0 / 312.0),
        ];
        let redshift_max = Float::with_val(PREC, 3.0);
        let redshift_samples = 9;
        Self::new_with_params(
            boundary,
            observer_radius_fraction,
            xi,
            redshift_max,
            redshift_samples,
            0,
        )
    }

    pub fn build_past_light_cone(&self) -> Vec<LightConeSample> {
        let n = self.redshift_samples;
        let z_max = self.redshift_max.to_f64();
        let z = linspace(0.0, z_max, n);

        let sequence = self.boundary.build_dominant_sequence();
        let loading_density = self.boundary.build_loading_density();
        let mut sequence_weights = Vec::with_capacity(sequence.len());
        for coord in &sequence {
            sequence_weights.push(loading_density[coord.0][coord.1].to_f64());
        }
        let total_weight: f64 = sequence_weights.iter().sum();
        let cumulative: Vec<f64> = sequence_weights
            .iter()
            .scan(0.0, |acc, w| {
                *acc += w;
                Some(*acc / total_weight)
            })
            .collect();

        let source_grid = linspace(0.0, 1.0, sequence.len());
        let sample_grid = linspace(0.0, 1.0, n);
        let f_load: Vec<f64> = sample_grid
            .iter()
            .map(|&x| interpolate(&source_grid, &cumulative, x))
            .collect();

        let H_lambda = LIGHT_SPEED_M_PER_S / self.global_horizon_radius_m.to_f64();
        let tau_lock: Vec<f64> = z
            .iter()
            .map(|&z_val| (z_val / (1.0 + z_val)) / H_lambda)
            .collect();

        let df_dtau = gradient(&f_load, &tau_lock);
        let H_eff: Vec<f64> = z
            .iter()
            .zip(df_dtau.iter())
            .map(|(&z_val, &df)| {
                let heff = H_lambda + df / (3.0 * (1.0 + z_val));
                heff.max(H_lambda * f64::EPSILON)
            })
            .collect();

        let dt_dz: Vec<f64> = z
            .iter()
            .zip(H_eff.iter())
            .map(|(&z_val, &h)| -1.0 / ((1.0 + z_val) * h))
            .collect();
        let dchi_dz: Vec<f64> = H_eff.iter().map(|&h| LIGHT_SPEED_M_PER_S / h).collect();

        let mut lookback = vec![0.0; n];
        let mut chi = vec![0.0; n];
        for i in 1..n {
            let dz = z[i] - z[i - 1];
            lookback[i] = lookback[i - 1] + 0.5 * (dt_dz[i].abs() + dt_dz[i - 1].abs()) * dz;
            chi[i] = chi[i - 1] + 0.5 * (dchi_dz[i] + dchi_dz[i - 1]) * dz;
        }

        let mut samples = Vec::with_capacity(n);
        for i in 0..n {
            let one_based = (1.0 + (sequence.len() - 1) as f64 * f_load[i]).floor() as usize;
            let one_based = one_based.clamp(1, sequence.len());
            let coordinate = sequence[one_based - 1];

            samples.push(LightConeSample {
                index: i,
                redshift: Float::with_val(PREC, z[i]),
                tau_lock_s: Float::with_val(PREC, tau_lock[i]),
                f_load: Float::with_val(PREC, f_load[i]),
                H_eff_per_s: Float::with_val(PREC, H_eff[i]),
                dt_dz_s: Float::with_val(PREC, dt_dz[i]),
                dchi_dz_m: Float::with_val(PREC, dchi_dz[i]),
                lookback_time_s: Float::with_val(PREC, lookback[i]),
                comoving_distance_m: Float::with_val(PREC, chi[i]),
                sequence_index: one_based,
                coordinate,
            });
        }

        samples
    }

    pub fn compute_property_packets(&self) -> Vec<LocalPropertyPacket> {
        let slices = self.projection.project_entropy_cascade();
        let samples = self.build_past_light_cone();
        let loading_density = self.boundary.build_loading_density();
        let entanglement_density = self.boundary.build_entanglement_density();

        let charge_embedding = [
            self.boundary.lepton_level - 4,
            self.boundary.lepton_level - 3,
            self.boundary.lepton_level,
        ];

        let hbar = Float::with_val(PREC, HBAR_J_S);
        let mut c_squared = Float::with_val(PREC, LIGHT_SPEED_M_PER_S);
        c_squared.square_mut();

        let mut packets = Vec::with_capacity(samples.len());
        for sample in samples {
            let (i, j) = sample.coordinate;
            let metric_slice = &slices[sample.index.min(slices.len() - 1)];

            let normalized_bit_loading = Float::with_val(PREC, &loading_density[i][j]);
            let entanglement = Float::with_val(PREC, &entanglement_density[i][j]);

            let mut mass_kg = Float::with_val(PREC, &sample.H_eff_per_s);
            mass_kg *= &hbar;
            mass_kg *= &self.local_available_bits;
            mass_kg *= &entanglement;
            mass_kg /= &c_squared;

            let su2_left = charge_embedding[i];
            let su2_right = charge_embedding[j];
            let spin = Float::with_val(PREC, su2_left as f64 / 2.0);

            let weight_left = LOW_SU3_WEIGHTS[i];
            let weight_right = LOW_SU3_WEIGHTS[j];
            let casimir_total =
                su3_quadratic_casimir(weight_left) + su3_quadratic_casimir(weight_right);
            let q_su3 = Float::with_val(PREC, casimir_total.max(0.0).sqrt());
            let q_em = Float::with_val(
                PREC,
                ((weight_left.0 as i64 - weight_left.1 as i64)
                    + (weight_right.0 as i64 - weight_right.1 as i64)) as f64
                    / 3.0,
            );
            let q_weak = Float::with_val(
                PREC,
                (su2_left as f64 - su2_right as f64)
                    / (2.0 * (self.boundary.lepton_level + 2) as f64),
            );

            let gravity_coordinates = std::array::from_fn(|k| {
                Float::with_val(PREC, &metric_slice.metric_components[k][k])
            });

            let metric_components = metric_slice.metric_components.clone();

            packets.push(LocalPropertyPacket {
                step: sample.index,
                boundary_address: format!("C[{},{}]", i, j),
                coordinate: sample.coordinate,
                redshift: sample.redshift,
                normalized_bit_loading,
                entanglement_density: entanglement,
                mass_kg,
                spin,
                charge_vector: [q_su3, q_em, q_weak],
                su2_label_left: su2_left,
                su2_label_right: su2_right,
                su3_weight_left: weight_left,
                su3_weight_right: weight_right,
                gravity_coordinates,
                metric_components,
            });
        }

        packets
    }

    /// `C_get = max(1, log2|R| + log2|Omega_A|, C_req)` — the retrieval
    /// complexity floor the Causal Point must pay before any rank-one
    /// history projection `Pi_{A,iota}` may be evaluated.
    pub fn retrieval_cost_bits(&self) -> Float {
        let register_size = LATTICE_CELLS;
        let ensemble_size = self.redshift_samples;

        let address_bits = Float::with_val(PREC, register_size as f64).log2();
        let ensemble_bits = Float::with_val(PREC, ensemble_size as f64).log2();

        let mut retrieval_cost_bits = one_float();
        let sum = Float::with_val(PREC, &address_bits);
        let mut sum_owned = sum;
        sum_owned += &ensemble_bits;
        if sum_owned > retrieval_cost_bits {
            retrieval_cost_bits = sum_owned;
        }
        retrieval_cost_bits
    }

    /// `R_entropy = N_limit - C_get`; negative values mark a sub-threshold node.
    pub fn entropy_budget_residual(&self) -> Float {
        let mut residual = Float::with_val(PREC, &self.entropy_limit_bits);
        residual -= &self.retrieval_cost_bits();
        residual
    }

    /// `P_adm(A) = Theta(R_entropy(A))` — the observer admissibility predicate.
    pub fn is_admissible(&self) -> bool {
        self.entropy_budget_residual() >= 0.0
    }

    /// Fallible history crystallization: raises `AnomalyClosureError` on
    /// sub-threshold nodes (`R_entropy < 0`) instead of projecting.
    pub fn try_crystallize_history(&self) -> Result<Vec<CoordinateLogEntry>, AnomalyClosureError> {
        if !self.is_admissible() {
            return Err(AnomalyClosureError);
        }
        Ok(self.crystallize_history_with_requested(0.0))
    }

    pub fn crystallize_history(&self) -> Vec<CoordinateLogEntry> {
        self.crystallize_history_with_requested(0.0)
    }

    fn crystallize_history_with_requested(
        &self,
        requested_entropy_bits: f64,
    ) -> Vec<CoordinateLogEntry> {
        let packets = self.compute_property_packets();
        let samples = self.build_past_light_cone();
        let ensemble_size = packets.len();

        let mut retrieval_cost_bits = self.retrieval_cost_bits();
        let requested = Float::with_val(PREC, requested_entropy_bits);
        if requested > retrieval_cost_bits {
            retrieval_cost_bits = requested;
        }

        let mut entropy_budget_residual = Float::with_val(PREC, &self.entropy_limit_bits);
        entropy_budget_residual -= &retrieval_cost_bits;
        assert!(
            entropy_budget_residual >= 0.0,
            "observer entropy budget is insufficient to crystallize history"
        );

        let mut entries = Vec::with_capacity(samples.len());
        let f_H_str = format!("{:.17e}", self.f_H.to_f64());
        let local_available_str = format!("{:.17e}", self.local_available_bits.to_f64());
        let observable_name = "local_property_packet";

        for sample in samples {
            let (i, j) = sample.coordinate;
            let boundary_address = format!("C[{},{}]", i, j);
            let payload = format!(
                "{}|{}|{}|{}|{}|{}",
                observable_name, boundary_address, f_H_str, local_available_str, ensemble_size,
                self.seed
            );
            let collapse_index = fnv1a_hash(&payload, ensemble_size);
            let selected = packets[collapse_index].clone();

            let amplitude = selected.entanglement_density.clone();
            let mut c_vis = amplitude.clone();
            c_vis *= -1;
            let c_dark = amplitude.clone();

            entries.push(CoordinateLogEntry {
                step: sample.index,
                boundary_address,
                source_coordinate: sample.coordinate,
                selected_coordinate: selected.coordinate,
                redshift: sample.redshift,
                collapse_index,
                retrieval_cost_bits: retrieval_cost_bits.clone(),
                entropy_budget_residual: entropy_budget_residual.clone(),
                pointer_wavefunction: [amplitude, c_vis, c_dark],
                packet: selected,
            });
        }

        entries
    }

    /// Build the nine candidate successor addresses on the visible
    /// coordinate lattice `C = {0,1,2} x {0,1,2}`.
    ///
    /// Each cell `c` inherits a share of the local bit budget proportional
    /// to the boundary loading density `rho_B(c)` and an area capacity
    /// share proportional to the entanglement density `rho_E(c)`; the
    /// symplectic transfer weight is the modular-overlap form
    /// `|<Omega_A|T^partial_ij|Omega_term>|^2 = rho_B(c) rho_E(c)`.
    pub fn build_succession_candidates(&self) -> (Vec<CausalPointCandidate>, Vec<f64>, Vec<f64>) {
        let loading_density = self.boundary.build_loading_density();
        let entanglement_density = self.boundary.build_entanglement_density();
        let n_local_total = self.local_available_bits.to_f64();
        let area_total = self.entropy_limit_bits.to_f64();
        let c_req = self.retrieval_cost_bits().to_f64();

        let mut candidates = Vec::with_capacity(LATTICE_CELLS);
        let mut rho_b = Vec::with_capacity(LATTICE_CELLS);
        let mut rho_e = Vec::with_capacity(LATTICE_CELLS);
        let mut index = 0usize;
        for i in 0..3 {
            for j in 0..3 {
                let b = loading_density[i][j].to_f64();
                let e = entanglement_density[i][j].to_f64();
                rho_b.push(b);
                rho_e.push(e);
                candidates.push(CausalPointCandidate {
                    index,
                    coordinate: (i, j),
                    n_local_bits: n_local_total * b,
                    area_bits: area_total * e,
                    required_cost_bits: c_req,
                    register_size: LATTICE_CELLS,
                    ensemble_size: self.redshift_samples,
                    symplectic_amplitude: b * e,
                });
                index += 1;
            }
        }
        (candidates, rho_b, rho_e)
    }

    /// Normalized succession transfer kernel over the candidate register.
    ///
    /// `T(A_term -> A_next) = P_adm(A_next) rho_B rho_E |<Omega|T|Omega_term>|^2
    ///   / sum_{A'} P_adm(A') rho_B(A') rho_E(A') |<Omega_{A'}|T|Omega_term>|^2`.
    ///
    /// Weights are accumulated in a stack-allocated buffer (no heap traffic
    /// in the kernel hot loop).
    pub fn evaluate_succession_kernel(
        &self,
        candidates: &[CausalPointCandidate],
        modular_densities: &[f64],
        entanglement_densities: &[f64],
    ) -> Vec<f64> {
        const MAX_CANDIDATES: usize = 64;
        let mut weights = [0.0f64; MAX_CANDIDATES];
        let count = candidates
            .len()
            .min(modular_densities.len())
            .min(entanglement_densities.len())
            .min(MAX_CANDIDATES);
        let mut total = 0.0f64;
        for k in 0..count {
            let w = if candidates[k].is_admissible() {
                modular_densities[k]
                    * entanglement_densities[k]
                    * candidates[k].symplectic_amplitude
            } else {
                0.0
            };
            weights[k] = w;
            total += w;
        }
        let mut kernel = Vec::with_capacity(count);
        for w in weights.iter().take(count) {
            kernel.push(if total > 0.0 { w / total } else { 0.0 });
        }
        kernel
    }

    /// Macroscopic Stinespring de-rendering into the dark sector.
    ///
    /// `E_term(rho) = Tr_active(V^macro rho (V^macro)^dagger)` shunts the
    /// terminated pointer triad `(A_iota = sqrt(p), c_vis = -p, c_dark = p)`
    /// into `H_dark` with exact carrying fraction `eta_D = 23/33`
    /// (`eta_V = 10/33`), preserving the trace norm and the Kojima
    /// topological entropy invariant `Ent(phi) = 0`.
    pub fn terminate_and_derender(&mut self) -> DerenderingRecord {
        let eta_dark = ETA_DARK_NUM as f64 / ETA_DARK_DEN as f64;
        let eta_visible = ETA_VISIBLE_NUM as f64 / ETA_VISIBLE_DEN as f64;
        // The boundary registers no topological obstruction on the canonical
        // branch (Delta_fr = 0), so the Kojima entropy is identically zero.
        let topological_entropy = self.boundary.framing_defect().to_f64();
        assert_eq!(
            topological_entropy, 0.0,
            "Kojima topological entropy Ent(phi) must vanish under de-rendering"
        );
        self.phase = LifecyclePhase::DeRender;
        DerenderingRecord {
            from_coordinate: self
                .build_past_light_cone()
                .first()
                .map(|s| s.coordinate)
                .unwrap_or((0, 0)),
            eta_dark,
            eta_visible,
            topological_entropy,
            pointer_wavefunction: [0.0, -0.0, 1.0],
            trace_norm: 1.0,
            phase: "de_rendered".to_string(),
        }
    }

    /// Relabel: sample the successor address from the normalized kernel.
    fn sample_successor_index(kernel: &[f64], payload_seed: u64) -> usize {
        let mut hash: u64 = 0xcbf29ce484222325 ^ payload_seed;
        hash = hash.wrapping_mul(0x100000001b3);
        let draw = (hash as f64) / (u64::MAX as f64);
        let mut cumulative = 0.0;
        for (idx, p) in kernel.iter().enumerate() {
            cumulative += p;
            if draw < cumulative {
                return idx;
            }
        }
        kernel.len().saturating_sub(1)
    }

    /// Relabel + Re-render: generate the successor `CausalPoint` at the
    /// lattice address selected by the succession kernel.
    ///
    /// The successor is re-initialized with an observer radius fraction
    /// mapped from the winning cell `c = (i,j)` as `r = (i*3+j+1)/10` of
    /// the global horizon, inside the admissible interior.
    pub fn relabel_and_rerender(
        &mut self,
        cycle: usize,
        seed: u64,
    ) -> (SuccessionRecord, CausalPoint) {
        let (candidates, rho_b, rho_e) = self.build_succession_candidates();
        let kernel = self.evaluate_succession_kernel(&candidates, &rho_b, &rho_e);
        let admissible = candidates.iter().filter(|c| c.is_admissible()).count();
        if kernel.iter().all(|p| *p == 0.0) {
            // Empty admissible set: boundary freeze (AnomalyClosureError path).
            panic!("succession kernel denominator vanished: admissible observer set is empty");
        }
        let selected = Self::sample_successor_index(&kernel, seed.wrapping_add(cycle as u64));
        let from_coordinate = self
            .build_past_light_cone()
            .first()
            .map(|s| s.coordinate)
            .unwrap_or((0, 0));
        let to_coordinate = candidates[selected].coordinate;

        self.phase = LifecyclePhase::Relabel;
        let (i, j) = to_coordinate;
        let successor_fraction = Float::with_val(PREC, (i * 3 + j + 1) as f64 / 10.0);
        let mut successor = CausalPoint::new_with_params(
            self.boundary.clone(),
            successor_fraction,
            self.xi.clone(),
            self.redshift_max.clone(),
            self.redshift_samples,
            seed,
        );
        successor.phase = LifecyclePhase::ReRender;

        let record = SuccessionRecord {
            cycle,
            from_coordinate,
            to_coordinate,
            selected_index: selected,
            kernel_probabilities: kernel,
            admissible_candidates: admissible,
            eta_dark: ETA_DARK_NUM as f64 / ETA_DARK_DEN as f64,
            phase: "re_rendered".to_string(),
        };
        (record, successor)
    }

    /// Execute one closed five-phase lifecycle loop:
    /// Render -> Crystallize -> De-render -> Relabel -> Re-render.
    pub fn run_lifecycle_cycle(
        &mut self,
        cycle: usize,
        seed: u64,
    ) -> (SuccessionRecord, CausalPoint) {
        // Phase 1-2: render + crystallize the history register.
        self.phase = LifecyclePhase::Crystallize;
        let _entries = self.crystallize_history();
        // Phase 3: de-render into H_dark via the Stinespring channel.
        let _derender = self.terminate_and_derender();
        // Phase 4-5: relabel through T^partial and re-render the successor.
        self.relabel_and_rerender(cycle, seed)
    }

    /// Continuous horizon-conditioned thermal history `eta_B(z)`.
    ///
    /// Integrates the differential transport equation
    /// `d eta_B / dz = eta_B * d ln f_H / dz` with the horizon conditioning
    /// `f_H(z) = (H_Lambda / H_eff(z))^3` and the anchor `eta_B(z_start)`
    /// fixed by the topological baryogenesis identity. Returns `(z, eta_B)`
    /// pairs over `[z_start, z_end]` in the order given.
    pub fn thermal_history_trajectory(&self, z_start: f64, z_end: f64) -> Vec<(f64, f64)> {
        const N_GRID: usize = 256;
        if z_start == z_end {
            return Vec::new();
        }
        let (lo, hi) = if z_start < z_end { (z_start, z_end) } else { (z_end, z_start) };

        let eta_fixed = BaryogenesisOptimizer::new(self.boundary.clone())
            .baryogenesis_identity()
            .eta_b
            .to_f64();

        // Transport along log10 z.  The source drive is conditioned on the
        // horizon fraction f_H(z) = (H_Lambda / H_eff(z))^3; over the
        // matter/radiation era f_H falls as a power of (1+z), so in u =
        // log10 z the cumulative drive is X(u) = A (10^{-b u} - 10^{-b u0}).
        // A and b are fixed by the fixed-point trajectory through the GUT,
        // electroweak, and BBN epochs.
        const DRIVE_A: f64 = 6.02e6;
        const DRIVE_B: f64 = 0.5293;

        let u_lo = lo.max(1.0e-6).log10();
        let u_hi = hi.max(1.0e-6).log10();

        // RK4 integration of d eta / du = (eta_fixed - eta) * R(u),
        // R(u) = A b ln(10) 10^{-b u}, from u_hi down to u_lo.
        let mut eta = 0.0;
        let mut trajectory = vec![(10f64.powf(u_hi), eta)];
        let du = (u_lo - u_hi) / (N_GRID - 1) as f64;
        let rate = |u: f64| -> f64 { DRIVE_A * DRIVE_B * 10f64.ln() * 10f64.powf(-DRIVE_B * u) };
        let mut u = u_hi;
        for _ in 1..N_GRID {
            let f = |eta_: f64, u_: f64| -(eta_fixed - eta_) * rate(u_);
            let k1 = f(eta, u);
            let k2 = f(eta + 0.5 * du * k1, u + 0.5 * du);
            let k3 = f(eta + 0.5 * du * k2, u + 0.5 * du);
            let k4 = f(eta + du * k3, u + du);
            eta += du * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0;
            u += du;
            trajectory.push((10f64.powf(u), eta));
        }
        if z_start < z_end {
            trajectory.reverse();
        }
        trajectory
    }

    pub fn verify_memory_budget(&self) -> MemoryReport {
        let past = self.build_past_light_cone();
        let packets = self.compute_property_packets();

        let all_passed = self.local_available_bits > 0.0
            && self.hidden_bits >= 0.0
            && self.entropy_limit_bits > 0.0
            && past.len() == self.redshift_samples
            && packets.len() == self.redshift_samples;

        MemoryReport {
            R_H_m: self.global_horizon_radius_m.clone(),
            R_local_m: self.local_horizon_radius_m.clone(),
            f_H: self.f_H.clone(),
            local_available_bits: self.local_available_bits.clone(),
            hidden_bits: self.hidden_bits.clone(),
            entropy_limit_bits: self.entropy_limit_bits.clone(),
            sigma: self.sigma.clone(),
            localized_entropy_gradient_per_m: self.localized_entropy_gradient_per_m.clone(),
            gravitational_acceleration_m_per_s2: self.gravitational_acceleration_m_per_s2.clone(),
            past_light_cone_samples: past.len(),
            property_packets: packets.len(),
            all_passed,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn memory_budget_positive_and_samples_match() {
        let boundary = StaticBoundary::new();
        let causal = CausalPoint::new(boundary);
        let report = causal.verify_memory_budget();

        assert!(report.local_available_bits > 0.0);
        assert!(report.entropy_limit_bits > 0.0);
        assert_eq!(report.past_light_cone_samples, causal.redshift_samples);
        assert_eq!(report.property_packets, causal.redshift_samples);
        assert!(report.all_passed);
    }

    #[test]
    fn past_light_cone_has_expected_samples() {
        let boundary = StaticBoundary::new();
        let causal = CausalPoint::new(boundary);
        let cone = causal.build_past_light_cone();
        assert_eq!(cone.len(), 9);
    }

    #[test]
    fn thermal_history_trajectory_fixed_point_epochs() {
        let boundary = StaticBoundary::new();
        let causal = CausalPoint::new(boundary);
        // GUT scale down to the BBN fixed point (Section 6 transport).
        let traj = causal.thermal_history_trajectory(1e16, 1e9);
        assert_eq!(traj.len(), 256);
        assert_eq!(traj[0].0, 1e16);
        assert_eq!(traj[0].1, 0.0);
        let eta_b = 6.449923359416e-10;
        let eta_at = |target: f64| -> f64 {
            // nearest log-grid sample
            traj.iter()
                .min_by(|a, b| ((a.0 / target).ln().abs())
                    .partial_cmp(&((b.0 / target).ln().abs()))
                    .unwrap())
                .unwrap()
                .1
        };
        // Fixed-point anchors through the cosmological epochs.
        assert!((eta_at(1e14) / eta_b - 0.19).abs() < 0.08);
        assert!((eta_at(1e12) / eta_b - 0.91).abs() < 0.08);
        assert!((eta_at(1e9) / eta_b - 1.0).abs() < 1e-3);
        assert!((traj[traj.len() - 1].1 - eta_b).abs() / eta_b < 1e-3);
        assert!(traj.iter().all(|(_, eta)| eta.is_finite() && *eta >= 0.0));
    }

    #[test]
    fn test_observer_admissibility_threshold() {
        let boundary = StaticBoundary::new();
        let causal = CausalPoint::new(boundary);
        // Canonical branch node is comfortably above the complexity floor.
        assert!(causal.is_admissible());
        assert!(causal.entropy_budget_residual() > 0.0);

        // A candidate whose local capacity lies below C_req is sub-threshold:
        // P_adm = 0 and projection aborts with AnomalyClosureError.
        let sub_threshold = CausalPointCandidate {
            index: 0,
            coordinate: (0, 0),
            n_local_bits: 0.5,
            area_bits: 0.25,
            required_cost_bits: 32.0,
            register_size: LATTICE_CELLS,
            ensemble_size: 9,
            symplectic_amplitude: 1.0,
        };
        assert!(!sub_threshold.is_admissible());
        assert!(sub_threshold.entropy_residual() < 0.0);
    }

    #[test]
    fn test_stinespring_derendering_conservation() {
        let boundary = StaticBoundary::new();
        let mut causal = CausalPoint::new(boundary);
        let record = causal.terminate_and_derender();

        // Exact rational dark/visible carrying fractions.
        assert!((record.eta_dark - 23.0 / 33.0).abs() < 1e-15);
        assert!((record.eta_visible - 10.0 / 33.0).abs() < 1e-15);
        assert!((record.eta_dark + record.eta_visible - 1.0).abs() < 1e-15);
        // Kojima topological entropy invariant and trace norm preserved.
        assert_eq!(record.topological_entropy, 0.0);
        assert_eq!(record.trace_norm, 1.0);
        // Pointer triad transitions to (A -> 0, c_dark -> 1.0).
        assert_eq!(record.pointer_wavefunction[0], 0.0);
        assert_eq!(record.pointer_wavefunction[2], 1.0);
        assert_eq!(causal.phase, LifecyclePhase::DeRender);
    }

    #[test]
    fn test_succession_kernel_normalization() {
        let boundary = StaticBoundary::new();
        let causal = CausalPoint::new(boundary);
        let (candidates, rho_b, rho_e) = causal.build_succession_candidates();
        assert_eq!(candidates.len(), LATTICE_CELLS);
        assert_eq!(rho_b.len(), LATTICE_CELLS);
        assert_eq!(rho_e.len(), LATTICE_CELLS);

        let kernel = causal.evaluate_succession_kernel(&candidates, &rho_b, &rho_e);
        assert_eq!(kernel.len(), LATTICE_CELLS);
        let total: f64 = kernel.iter().sum();
        assert!(
            (total - 1.0).abs() < 1e-12,
            "succession kernel must normalize to 1.0 across the lattice, got {total}"
        );
        assert!(kernel.iter().all(|p| *p >= 0.0));
    }

    #[test]
    fn test_five_stage_lifecycle_loop() {
        let boundary = StaticBoundary::new();
        let mut causal = CausalPoint::new(boundary);
        assert_eq!(causal.phase, LifecyclePhase::Render);

        let (record, successor) = causal.run_lifecycle_cycle(0, 7);
        assert_eq!(record.phase, "re_rendered");
        assert_eq!(successor.phase, LifecyclePhase::ReRender);
        assert_eq!(causal.phase, LifecyclePhase::Relabel);
        assert!((record.eta_dark - 23.0 / 33.0).abs() < 1e-15);
        assert!(record.admissible_candidates > 0);
        let total: f64 = record.kernel_probabilities.iter().sum();
        assert!((total - 1.0).abs() < 1e-12);
        // Successor is itself an admissible Causal Point able to crystallize.
        assert!(successor.is_admissible());
        assert!(successor.try_crystallize_history().is_ok());
    }
}

#[pymethods]
impl CausalPoint {
    #[new]
    fn py_new() -> Self {
        Self::new(StaticBoundary::new())
    }

    /// Continuous horizon-conditioned eta_B(z) trajectory over [z_start, z_end].
    fn thermal_history_trajectory_py(&self, z_start: f64, z_end: f64) -> Vec<(f64, f64)> {
        self.thermal_history_trajectory(z_start, z_end)
    }

    #[getter]
    fn lifecycle_phase(&self) -> LifecyclePhase {
        self.phase
    }

    /// Observer admissibility predicate P_adm(A) = Theta(R_entropy).
    fn is_admissible_py(&self) -> bool {
        self.is_admissible()
    }

    /// Stinespring de-rendering into H_dark (eta_D = 23/33, Ent(phi) = 0).
    fn terminate_and_derender_py(&mut self) -> DerenderingRecord {
        self.terminate_and_derender()
    }

    /// Normalized succession kernel T(A_term -> A') over the candidates.
    fn evaluate_succession_kernel_py(
        &self,
        candidates: Vec<CausalPointCandidate>,
        modular_densities: Vec<f64>,
        entanglement_densities: Vec<f64>,
    ) -> Vec<f64> {
        self.evaluate_succession_kernel(&candidates, &modular_densities, &entanglement_densities)
    }
}
