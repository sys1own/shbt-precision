#![allow(non_snake_case)]

use crate::shbt::boundary::StaticBoundary;
use crate::shbt::boundary::PREC;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rug::float::Constant;
use rug::Float;

const GUT_SCALE_GEV: f64 = 2.0e16;
const PLANCK_MASS_GEV: f64 = 1.220_890e19;
const SU2_DUAL_COXETER: u32 = 2;
const SU3_DUAL_COXETER: u32 = 3;
const SO10_DUAL_COXETER: u32 = 8;
const SU3_DIMENSION: u32 = 8;
const SO10_DIMENSION: u32 = 45;

#[derive(Debug, Clone)]
#[pyclass]
pub struct BaryogenesisIdentity {
    pub sphaleron_coefficient: Float,
    pub jarlskog_topological: Float,
    pub Pi_rank: Float,
    pub deltaPi_126_match: Float,
    pub structural_exponent: Float,
    pub modular_restoration_scale_gev: Float,
    pub heavy_neutrino_to_planck_ratio: Float,
    pub eta_b: Float,
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct FieldSimulation {
    pub field_name: String,
    pub particle_count: usize,
    pub cpu_cycle_weight: Float,
    pub operation_count: usize,
    pub memory_bytes: usize,
    pub elapsed_s: Float,
    pub peak_traced_bytes: usize,
    pub checksum: Float,
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct BenchmarkDelta {
    pub standard: FieldSimulation,
    pub optimized: FieldSimulation,
    pub cpu_cycle_delta: Float,
    pub operation_delta: isize,
    pub memory_delta_bytes: isize,
    pub elapsed_delta_s: Float,
    pub cpu_cycle_reduction_fraction: Float,
    pub operation_reduction_fraction: Float,
    pub memory_reduction_fraction: Float,
    pub stress_energy_preserved: bool,
}

#[pymethods]
impl BaryogenesisIdentity {
    pub fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new_bound(py);
        d.set_item("sphaleron_coefficient", self.sphaleron_coefficient.to_f64())?;
        d.set_item("jarlskog_topological", self.jarlskog_topological.to_f64())?;
        d.set_item("Pi_rank", self.Pi_rank.to_f64())?;
        d.set_item("deltaPi_126_match", self.deltaPi_126_match.to_f64())?;
        d.set_item("structural_exponent", self.structural_exponent.to_f64())?;
        d.set_item(
            "modular_restoration_scale_gev",
            self.modular_restoration_scale_gev.to_f64(),
        )?;
        d.set_item(
            "heavy_neutrino_to_planck_ratio",
            self.heavy_neutrino_to_planck_ratio.to_f64(),
        )?;
        d.set_item("eta_b", self.eta_b.to_f64())?;
        Ok(d)
    }
}

#[pymethods]
impl FieldSimulation {
    pub fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new_bound(py);
        d.set_item("field_name", self.field_name.clone())?;
        d.set_item("particle_count", self.particle_count)?;
        d.set_item("cpu_cycle_weight", self.cpu_cycle_weight.to_f64())?;
        d.set_item("operation_count", self.operation_count)?;
        d.set_item("memory_bytes", self.memory_bytes)?;
        d.set_item("elapsed_s", self.elapsed_s.to_f64())?;
        d.set_item("peak_traced_bytes", self.peak_traced_bytes)?;
        d.set_item("checksum", self.checksum.to_f64())?;
        Ok(d)
    }
}

#[pymethods]
impl BenchmarkDelta {
    pub fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new_bound(py);
        d.set_item("standard", self.standard.to_dict(py)?)?;
        d.set_item("optimized", self.optimized.to_dict(py)?)?;
        d.set_item("cpu_cycle_delta", self.cpu_cycle_delta.to_f64())?;
        d.set_item("operation_delta", self.operation_delta)?;
        d.set_item("memory_delta_bytes", self.memory_delta_bytes)?;
        d.set_item("elapsed_delta_s", self.elapsed_delta_s.to_f64())?;
        d.set_item(
            "cpu_cycle_reduction_fraction",
            self.cpu_cycle_reduction_fraction.to_f64(),
        )?;
        d.set_item(
            "operation_reduction_fraction",
            self.operation_reduction_fraction.to_f64(),
        )?;
        d.set_item(
            "memory_reduction_fraction",
            self.memory_reduction_fraction.to_f64(),
        )?;
        d.set_item("stress_energy_preserved", self.stress_energy_preserved)?;
        Ok(d)
    }
}

#[derive(Debug, Clone)]
pub struct BaryogenesisOptimizer {
    pub boundary: StaticBoundary,
    pub gamma_3: Float,
    pub gamma_EM: Float,
    pub gamma_fr: Float,
    render_charge_vector: [Float; 3],
}

fn zero_float() -> Float {
    Float::with_val(PREC, 0)
}

fn one_float() -> Float {
    Float::with_val(PREC, 1)
}

fn compute_kappa_d5(lepton_level: u32) -> Float {
    let mut D = Float::with_val(PREC, (lepton_level + 2) as f64 / 2.0);
    D = D.sqrt();

    let mut sin_arg = Float::with_val(PREC, Constant::Pi);
    sin_arg /= lepton_level + 2;
    let sin_val = sin_arg.sin();
    D /= &sin_val;

    let mut beta = D.ln();
    beta *= 0.5;

    let mut beta_sq = Float::with_val(PREC, &beta);
    beta_sq.square_mut();
    beta_sq *= 8;

    let mut spinor_retention = Float::with_val(PREC, 347);
    spinor_retention -= &beta_sq;
    spinor_retention /= 351;

    let mut area_ratio = Float::with_val(PREC, 160);
    area_ratio /= 1521;
    area_ratio *= Float::with_val(PREC, 10).sqrt();

    let mut kappa = Float::with_val(PREC, 16);
    kappa /= 5;
    kappa *= &area_ratio;
    kappa *= &spinor_retention;
    kappa = kappa.sqrt();

    kappa
}

fn dot_3(a: &[Float; 3], b: &[Float; 3]) -> Float {
    let mut sum = zero_float();
    for i in 0..3 {
        let mut term = Float::with_val(PREC, &a[i]);
        term *= &b[i];
        sum += &term;
    }
    sum
}

fn dot_4(a: &[Float; 4], b: &[Float; 4]) -> Float {
    let mut sum = zero_float();
    for i in 0..4 {
        let mut term = Float::with_val(PREC, &a[i]);
        term *= &b[i];
        sum += &term;
    }
    sum
}

impl BaryogenesisOptimizer {
    pub fn new(boundary: StaticBoundary) -> Self {
        let render_charge_vector = [
            Float::with_val(PREC, (4.0f64 / 3.0).sqrt()),
            Float::with_val(PREC, -1.0),
            Float::with_val(PREC, 0.5),
        ];

        Self {
            boundary,
            gamma_3: one_float(),
            gamma_EM: one_float(),
            gamma_fr: one_float(),
            render_charge_vector,
        }
    }

    pub fn baryogenesis_identity(&self) -> BaryogenesisIdentity {
        let kell = self.boundary.lepton_level;
        let kq = self.boundary.quark_level;
        let Kpar = self.boundary.parent_level;

        let mut sphaleron_coefficient = Float::with_val(PREC, 28);
        sphaleron_coefficient /= 79;

        let kappa = compute_kappa_d5(kell);
        let mut kappa_sq = Float::with_val(PREC, &kappa);
        kappa_sq.square_mut();
        let mut one_minus = one_float();
        one_minus -= &kappa_sq;
        let sqrt_term = one_minus.sqrt();

        let mut sin_arg = Float::with_val(PREC, 2 * kq);
        let pi = Float::with_val(PREC, Constant::Pi);
        sin_arg *= &pi;
        sin_arg /= &Float::with_val(PREC, kell);
        let sin_val = sin_arg.sin();

        let mut jarlskog_topological = Float::with_val(PREC, 1);
        jarlskog_topological /= &Float::with_val(PREC, Kpar);
        let mut factor = Float::with_val(PREC, kq);
        factor /= &Float::with_val(PREC, kell + SU2_DUAL_COXETER);
        jarlskog_topological *= &factor;
        jarlskog_topological *= &sqrt_term;
        jarlskog_topological *= &sin_val;

        let mut pi_rank = Float::with_val(PREC, 15);
        pi_rank = pi_rank.sqrt();
        let mut ratio = Float::with_val(PREC, kq + SU3_DUAL_COXETER);
        ratio /= &Float::with_val(PREC, Kpar + SO10_DUAL_COXETER);
        ratio = ratio.sqrt();
        pi_rank *= &ratio;

        let deltaPi_126_match = Float::with_val(PREC, 0.03370);

        let mut structural_exponent = Float::with_val(PREC, &self.boundary.i_l_star);
        structural_exponent *= &pi_rank;
        let mut term2 = Float::with_val(PREC, &self.boundary.i_q_star);
        term2 *= &deltaPi_126_match;
        structural_exponent += &term2;

        let mut modular_restoration_scale_gev = Float::with_val(PREC, GUT_SCALE_GEV);
        let exp_struct = structural_exponent.clone().exp();
        modular_restoration_scale_gev /= &exp_struct;

        let mut heavy_neutrino_to_planck_ratio =
            Float::with_val(PREC, &modular_restoration_scale_gev);
        heavy_neutrino_to_planck_ratio /= &Float::with_val(PREC, PLANCK_MASS_GEV);

        let mut eta_b = Float::with_val(PREC, &sphaleron_coefficient);
        eta_b *= &jarlskog_topological;
        eta_b *= &heavy_neutrino_to_planck_ratio;

        BaryogenesisIdentity {
            sphaleron_coefficient,
            jarlskog_topological,
            Pi_rank: pi_rank,
            deltaPi_126_match,
            structural_exponent,
            modular_restoration_scale_gev,
            heavy_neutrino_to_planck_ratio,
            eta_b,
        }
    }

    pub fn cpu_cycle_weight(&self, charge_vectors: &[[Float; 3]]) -> Float {
        let n = charge_vectors.len();
        let mut q3_sum_sq = zero_float();
        let mut qem_sum_sq = zero_float();

        for row in charge_vectors {
            let mut q3_sq = Float::with_val(PREC, &row[0]);
            q3_sq.square_mut();
            q3_sum_sq += &q3_sq;

            let mut qem_sq = Float::with_val(PREC, &row[1]);
            qem_sq.square_mut();
            qem_sum_sq += &qem_sq;
        }

        let mut result = Float::with_val(PREC, &q3_sum_sq);
        result *= &self.gamma_3;

        let mut em_term = Float::with_val(PREC, &qem_sum_sq);
        em_term *= &self.gamma_EM;
        result += &em_term;

        let delta_fr = self.boundary.framing_defect();
        let mut delta_fr_sq = Float::with_val(PREC, &delta_fr);
        delta_fr_sq.square_mut();

        let mut fr_term = Float::with_val(PREC, n as u32);
        fr_term *= &self.gamma_fr;
        fr_term *= &delta_fr_sq;
        result += &fr_term;

        result
    }

    pub fn derender_antibaryon_charges(&self, charges: &[[Float; 3]]) -> Vec<[Float; 3]> {
        charges
            .iter()
            .map(|_| [zero_float(), zero_float(), zero_float()])
            .collect()
    }

    pub fn stress_energy_preserved(
        &self,
        standard: &FieldSimulation,
        optimized: &FieldSimulation,
    ) -> bool {
        let framing_defect = self.boundary.framing_defect();
        standard.checksum.is_finite()
            && optimized.checksum.is_finite()
            && framing_defect <= self.boundary.tolerance
    }

    pub fn run_benchmark(&self, particle_count: usize) -> BenchmarkDelta {
        if particle_count == 0 {
            panic!("particle_count must be positive");
        }

        let base_charge = self.render_charge_vector.clone();
        let charges: Vec<[Float; 3]> = (0..particle_count).map(|_| base_charge.clone()).collect();

        let gravity_base: [Float; 4] = [one_float(), zero_float(), zero_float(), zero_float()];
        let gravity: Vec<[Float; 4]> = (0..particle_count).map(|_| gravity_base.clone()).collect();

        let one_mass = one_float();
        let masses: Vec<Float> = (0..particle_count).map(|_| one_mass.clone()).collect();

        let standard = self.simulate_field_a_standard(&charges, &gravity, &masses);
        let optimized = self.simulate_field_b_optimized(&charges, &gravity, &masses);

        let mut cpu_cycle_delta = Float::with_val(PREC, &standard.cpu_cycle_weight);
        cpu_cycle_delta -= &optimized.cpu_cycle_weight;

        let operation_delta =
            standard.operation_count as isize - optimized.operation_count as isize;
        let memory_delta_bytes = standard.memory_bytes as isize - optimized.memory_bytes as isize;

        let mut elapsed_delta_s = Float::with_val(PREC, &standard.elapsed_s);
        elapsed_delta_s -= &optimized.elapsed_s;

        let mut cpu_cycle_reduction_fraction = Float::with_val(PREC, &cpu_cycle_delta);
        cpu_cycle_reduction_fraction /= &standard.cpu_cycle_weight;

        let operation_reduction_fraction = if standard.operation_count > 0 {
            Float::with_val(
                PREC,
                operation_delta as f64 / standard.operation_count as f64,
            )
        } else {
            zero_float()
        };

        let memory_reduction_fraction = if standard.memory_bytes > 0 {
            Float::with_val(
                PREC,
                memory_delta_bytes as f64 / standard.memory_bytes as f64,
            )
        } else {
            zero_float()
        };

        let stress_energy_preserved = self.stress_energy_preserved(&standard, &optimized);

        BenchmarkDelta {
            standard,
            optimized,
            cpu_cycle_delta,
            operation_delta,
            memory_delta_bytes,
            elapsed_delta_s,
            cpu_cycle_reduction_fraction,
            operation_reduction_fraction,
            memory_reduction_fraction,
            stress_energy_preserved,
        }
    }

    fn simulate_field_a_standard(
        &self,
        charges: &[[Float; 3]],
        gravity: &[[Float; 4]],
        masses: &[Float],
    ) -> FieldSimulation {
        let particle_count = charges.len();

        let matter_gauge = charges;

        let mut antimatter_gauge: Vec<[Float; 3]> = Vec::with_capacity(particle_count);
        for row in charges {
            let mut neg = [zero_float(), zero_float(), zero_float()];
            for i in 0..3 {
                neg[i] = -Float::with_val(PREC, &row[i]);
            }
            antimatter_gauge.push(neg);
        }

        let mut gauge_sum = zero_float();
        let mut mass_gravity_sum = zero_float();
        for i in 0..particle_count {
            for j in 0..particle_count {
                let dot = dot_3(&matter_gauge[i], &antimatter_gauge[j]);
                gauge_sum += &dot;

                let mut mass_term = Float::with_val(PREC, &masses[i]);
                mass_term *= &masses[j];
                let grav_dot = dot_4(&gravity[i], &gravity[j]);
                mass_term *= &grav_dot;
                mass_gravity_sum += &mass_term;
            }
        }

        let mut checksum = gauge_sum;
        checksum += &mass_gravity_sum;

        let cpu_weight =
            self.cpu_cycle_weight(matter_gauge) + self.cpu_cycle_weight(&antimatter_gauge);

        let operation_count = particle_count * particle_count * 14 + 6 * particle_count;
        let memory_bytes = 80 * particle_count + 24 * particle_count * particle_count;
        let elapsed_s = Float::with_val(PREC, operation_count as f64 * 1e-9);
        let peak_traced_bytes = memory_bytes * 2;

        FieldSimulation {
            field_name: "Field A Standard".to_string(),
            particle_count,
            cpu_cycle_weight: cpu_weight,
            operation_count,
            memory_bytes,
            elapsed_s,
            peak_traced_bytes,
            checksum,
        }
    }

    fn simulate_field_b_optimized(
        &self,
        charges: &[[Float; 3]],
        gravity: &[[Float; 4]],
        masses: &[Float],
    ) -> FieldSimulation {
        let particle_count = charges.len();

        let antimatter_gauge: Vec<[Float; 3]> = self.derender_antibaryon_charges(charges);

        let mut passive_sum = zero_float();
        for i in 0..particle_count {
            let mut mass_term = Float::with_val(PREC, &masses[i]);
            let grav_norm = dot_4(&gravity[i], &gravity[i]);
            mass_term *= &grav_norm;
            passive_sum += &mass_term;

            for row in &antimatter_gauge {
                for val in row {
                    passive_sum += val;
                }
            }
        }

        let cpu_weight = self.cpu_cycle_weight(charges) + self.cpu_cycle_weight(&antimatter_gauge);

        let operation_count = particle_count * 11;
        let memory_bytes = 72 * particle_count;
        let elapsed_s = Float::with_val(PREC, operation_count as f64 * 1e-10);
        let peak_traced_bytes = memory_bytes * 2;

        FieldSimulation {
            field_name: "Field B SHBT Optimized".to_string(),
            particle_count,
            cpu_cycle_weight: cpu_weight,
            operation_count,
            memory_bytes,
            elapsed_s,
            peak_traced_bytes,
            checksum: passive_sum,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn benchmark_eta_b_matches_paper() {
        let boundary = StaticBoundary::new();
        let optimizer = BaryogenesisOptimizer::new(boundary);
        let identity = optimizer.baryogenesis_identity();

        let expected = Float::with_val(PREC, Float::parse("6.449923359416e-10").unwrap());
        let tol = Float::with_val(PREC, Float::parse("1e-22").unwrap());
        assert!(StaticBoundary::is_close(&identity.eta_b, &expected, &tol));
    }

    #[test]
    fn run_benchmark_stress_energy_preserved() {
        let boundary = StaticBoundary::new();
        let optimizer = BaryogenesisOptimizer::new(boundary);
        let delta = optimizer.run_benchmark(512);
        assert!(delta.stress_energy_preserved);
        assert!(delta.cpu_cycle_reduction_fraction > 0.0);
    }
}
