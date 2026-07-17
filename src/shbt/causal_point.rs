#![allow(non_snake_case)]

use crate::shbt::boundary::{StaticBoundary, PREC};
use crate::shbt::entropy_flow::{BulkMetricSlice, HolographicProjection};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rug::float::Constant;
use rug::Float;

const LIGHT_SPEED_M_PER_S: f64 = 299_792_458.0;
const HBAR_J_S: f64 = 1.054_571_817e-34;
const LOW_SU3_WEIGHTS: [(u32, u32); 3] = [(0, 0), (1, 0), (0, 1)];

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

#[derive(Debug, Clone)]
pub struct CausalPoint {
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

    pub fn crystallize_history(&self) -> Vec<CoordinateLogEntry> {
        self.crystallize_history_with_requested(0.0)
    }

    fn crystallize_history_with_requested(
        &self,
        requested_entropy_bits: f64,
    ) -> Vec<CoordinateLogEntry> {
        let packets = self.compute_property_packets();
        let samples = self.build_past_light_cone();

        let register_size = 9;
        let ensemble_size = packets.len();

        let address_bits = Float::with_val(PREC, register_size as f64).log2();
        let ensemble_bits = Float::with_val(PREC, ensemble_size as f64).log2();

        let requested = Float::with_val(PREC, requested_entropy_bits);
        let mut retrieval_cost_bits = one_float();
        let sum = Float::with_val(PREC, &address_bits);
        let mut sum_owned = sum;
        sum_owned += &ensemble_bits;
        if sum_owned > retrieval_cost_bits {
            retrieval_cost_bits = sum_owned;
        }
        if requested > retrieval_cost_bits {
            retrieval_cost_bits = requested;
        }

        let mut entropy_budget_residual = Float::with_val(PREC, &self.entropy_limit_bits);
        entropy_budget_residual -= &retrieval_cost_bits;
        if entropy_budget_residual < 0.0 {
            panic!("observer entropy budget is insufficient to crystallize history");
        }

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
}
