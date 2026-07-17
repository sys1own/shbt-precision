#![allow(non_snake_case)]

use crate::shbt::boundary::PREC;
use crate::shbt::boundary::{EntropyUpdate, StaticBoundary};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rug::Float;

#[derive(Debug, Clone)]
#[pyclass]
pub struct BulkMetricSlice {
    pub tau: Float,
    pub load_vector: [Float; 5],
    pub euler_flux: [Float; 4],
    pub execution_vectors: [[Float; 4]; 4],
    pub unstabilized_metric: [[Float; 4]; 4],
    pub metric_components: [[Float; 4]; 4],
    pub spatial_metric: [[Float; 3]; 3],
    pub epsilon_eig: Float,
    pub eigenvalues: [Float; 4],
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct ProjectionReport {
    pub slice_count: usize,
    pub projector_rank: usize,
    pub symmetric: bool,
    pub trace_normalized: bool,
    pub positive_definite: bool,
    pub spatial_metric_shape: bool,
    pub all_passed: bool,
}

#[pymethods]
impl BulkMetricSlice {
    pub fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new_bound(py);
        d.set_item("tau", self.tau.to_f64())?;
        let load: Vec<f64> = self.load_vector.iter().map(|v| v.to_f64()).collect();
        d.set_item("load_vector", load)?;
        let euler: Vec<f64> = self.euler_flux.iter().map(|v| v.to_f64()).collect();
        d.set_item("euler_flux", euler)?;
        let exec: Vec<Vec<f64>> = self
            .execution_vectors
            .iter()
            .map(|row| row.iter().map(|v| v.to_f64()).collect())
            .collect();
        d.set_item("execution_vectors", exec)?;
        let unstab: Vec<Vec<f64>> = self
            .unstabilized_metric
            .iter()
            .map(|row| row.iter().map(|v| v.to_f64()).collect())
            .collect();
        d.set_item("unstabilized_metric", unstab)?;
        let metric: Vec<Vec<f64>> = self
            .metric_components
            .iter()
            .map(|row| row.iter().map(|v| v.to_f64()).collect())
            .collect();
        d.set_item("metric_components", metric)?;
        let spatial: Vec<Vec<f64>> = self
            .spatial_metric
            .iter()
            .map(|row| row.iter().map(|v| v.to_f64()).collect())
            .collect();
        d.set_item("spatial_metric", spatial)?;
        d.set_item("epsilon_eig", self.epsilon_eig.to_f64())?;
        let eigs: Vec<f64> = self.eigenvalues.iter().map(|v| v.to_f64()).collect();
        d.set_item("eigenvalues", eigs)?;
        Ok(d)
    }
}

#[pymethods]
impl ProjectionReport {
    pub fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new_bound(py);
        d.set_item("slice_count", self.slice_count)?;
        d.set_item("projector_rank", self.projector_rank)?;
        d.set_item("symmetric", self.symmetric)?;
        d.set_item("trace_normalized", self.trace_normalized)?;
        d.set_item("positive_definite", self.positive_definite)?;
        d.set_item("spatial_metric_shape", self.spatial_metric_shape)?;
        d.set_item("all_passed", self.all_passed)?;
        Ok(d)
    }
}

#[derive(Debug, Clone)]
pub struct HolographicProjection {
    pub boundary: StaticBoundary,
    pub lambda_closure: Float,
    P: [[Float; 4]; 3],
}

fn zero_float() -> Float {
    Float::with_val(PREC, 0)
}

fn one_float() -> Float {
    Float::with_val(PREC, 1)
}

fn zero_matrix_3x3() -> [[Float; 3]; 3] {
    std::array::from_fn(|_| std::array::from_fn(|_| zero_float()))
}

fn zero_matrix_4x4() -> [[Float; 4]; 4] {
    std::array::from_fn(|_| std::array::from_fn(|_| zero_float()))
}

fn clone_matrix_4x4(m: &[[Float; 4]; 4]) -> [[Float; 4]; 4] {
    std::array::from_fn(|i| std::array::from_fn(|j| Float::with_val(PREC, &m[i][j])))
}

fn trace_4x4(m: &[[Float; 4]; 4]) -> Float {
    let mut trace = zero_float();
    for i in 0..4 {
        trace += &m[i][i];
    }
    trace
}

fn normalize_vector(v: &[Float; 5]) -> [Float; 5] {
    let floor = Float::with_val(PREC, f64::MIN_POSITIVE);
    let mut floored: [Float; 5] = std::array::from_fn(|i| {
        let mut x = Float::with_val(PREC, &v[i]);
        if x < floor {
            x = floor.clone();
        }
        x
    });

    let mut total = zero_float();
    for i in 0..5 {
        total += &floored[i];
    }
    for i in 0..5 {
        floored[i] /= &total;
    }
    floored
}

fn jacobi_eigenvalues(a: &[[Float; 4]; 4]) -> [Float; 4] {
    let mut m = clone_matrix_4x4(a);
    let threshold = Float::with_val(PREC, 1e-40);

    for _ in 0..40 {
        let mut max_off = zero_float();
        let mut p = 0usize;
        let mut q = 1usize;

        for i in 0..4 {
            for j in (i + 1)..4 {
                let mut abs_val = Float::with_val(PREC, &m[i][j]);
                abs_val.abs_mut();
                if abs_val > max_off {
                    max_off = abs_val;
                    p = i;
                    q = j;
                }
            }
        }

        if max_off < threshold {
            break;
        }

        let app = Float::with_val(PREC, &m[p][p]);
        let aqq = Float::with_val(PREC, &m[q][q]);
        let apq = Float::with_val(PREC, &m[p][q]);

        let mut tau_num = Float::with_val(PREC, &aqq);
        tau_num -= &app;
        let mut tau_den = Float::with_val(PREC, 2);
        tau_den *= &apq;
        let mut tau_jacobi = Float::with_val(PREC, &tau_num);
        tau_jacobi /= &tau_den;

        let mut tau_abs = Float::with_val(PREC, &tau_jacobi);
        tau_abs.abs_mut();
        let mut tau2 = Float::with_val(PREC, &tau_jacobi);
        tau2.square_mut();
        tau2 += 1;
        let mut denom_t = Float::with_val(PREC, 1);
        let sqrt_one_plus_tau2 = tau2.sqrt();
        denom_t += &tau_abs;
        denom_t += &sqrt_one_plus_tau2;

        let mut t = Float::with_val(PREC, 1);
        t /= &denom_t;
        if tau_jacobi < 0.0 {
            t *= -1;
        }

        let t2 = {
            let mut v = Float::with_val(PREC, &t);
            v.square_mut();
            v += 1;
            v
        };
        let mut c = one_float();
        let sqrt_one_plus_t2 = t2.sqrt();
        c /= &sqrt_one_plus_t2;
        let mut s = Float::with_val(PREC, &t);
        s *= &c;

        let mut new_pp = {
            let mut v = Float::with_val(PREC, &c);
            v.square_mut();
            v *= &app;
            v
        };
        let mut s2_aqq = Float::with_val(PREC, &s);
        s2_aqq.square_mut();
        s2_aqq *= &aqq;
        new_pp += &s2_aqq;

        let mut term = Float::with_val(PREC, 2);
        term *= &s;
        term *= &c;
        term *= &apq;
        new_pp -= &term;

        let mut new_qq = {
            let mut v = Float::with_val(PREC, &s);
            v.square_mut();
            v *= &app;
            v
        };
        let mut c2_aqq = Float::with_val(PREC, &c);
        c2_aqq.square_mut();
        c2_aqq *= &aqq;
        new_qq += &c2_aqq;
        new_qq += &term;

        m[p][p] = new_pp;
        m[q][q] = new_qq;
        m[p][q] = zero_float();
        m[q][p] = zero_float();

        for k in 0..4 {
            if k != p && k != q {
                let apk = Float::with_val(PREC, &m[p][k]);
                let aqk = Float::with_val(PREC, &m[q][k]);

                let mut new_pk = Float::with_val(PREC, &c);
                new_pk *= &apk;
                let mut s_aqk = Float::with_val(PREC, &s);
                s_aqk *= &aqk;
                new_pk -= &s_aqk;

                let mut new_qk = Float::with_val(PREC, &s);
                new_qk *= &apk;
                let mut c_aqk = Float::with_val(PREC, &c);
                c_aqk *= &aqk;
                new_qk += &c_aqk;

                m[p][k] = new_pk;
                m[k][p] = Float::with_val(PREC, &m[p][k]);
                m[q][k] = new_qk;
                m[k][q] = Float::with_val(PREC, &m[q][k]);
            }
        }
    }

    let mut values: Vec<Float> = (0..4).map(|i| Float::with_val(PREC, &m[i][i])).collect();
    values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    std::array::from_fn(|i| values[i].clone())
}

impl HolographicProjection {
    pub fn new(boundary: StaticBoundary) -> Self {
        let P: [[Float; 4]; 3] = [
            [
                Float::with_val(PREC, 0),
                Float::with_val(PREC, 1),
                Float::with_val(PREC, 0),
                Float::with_val(PREC, 0),
            ],
            [
                Float::with_val(PREC, 0),
                Float::with_val(PREC, 0),
                Float::with_val(PREC, 1),
                Float::with_val(PREC, 0),
            ],
            [
                Float::with_val(PREC, 0),
                Float::with_val(PREC, 0),
                Float::with_val(PREC, 0),
                Float::with_val(PREC, 1),
            ],
        ];

        Self {
            boundary,
            lambda_closure: Float::with_val(PREC, 1),
            P,
        }
    }

    pub fn derive_load_vector(&self, state: &[[Float; 3]; 3]) -> [Float; 5] {
        let mut load: [Float; 5] = std::array::from_fn(|_| zero_float());
        let sequence = self.boundary.build_dominant_sequence();

        for (index, coordinate) in sequence.iter().enumerate() {
            let mut value = Float::with_val(PREC, &state[coordinate.0][coordinate.1]);
            if value < 0.0 {
                value = zero_float();
            }
            load[index % 5] += &value;
        }

        normalize_vector(&load)
    }

    pub fn metric_from_load_vector(&self, load: &[Float; 5], tau: Float) -> BulkMetricSlice {
        let ell = normalize_vector(load);
        let primes: [Float; 5] = [
            Float::with_val(PREC, 2),
            Float::with_val(PREC, 3),
            Float::with_val(PREC, 5),
            Float::with_val(PREC, 7),
            Float::with_val(PREC, 11),
        ];

        let mut Phi: [Float; 4] = std::array::from_fn(|_| zero_float());
        let mut W: [Float; 4] = std::array::from_fn(|_| zero_float());
        let mut execution_vectors = zero_matrix_4x4();
        let mut unstab = zero_matrix_4x4();
        let mut diagonal: [Float; 4] = std::array::from_fn(|_| zero_float());

        for s in 0..4 {
            let mut diff = Float::with_val(PREC, &ell[s + 1]);
            diff -= &ell[s];
            let mut ratio = Float::with_val(PREC, &primes[s + 1]);
            ratio /= &primes[s];
            let ln_ratio = ratio.ln();
            diff /= &ln_ratio;
            Phi[s] = diff;

            let mut ell_bar = Float::with_val(PREC, &ell[s]);
            ell_bar += &ell[s + 1];
            ell_bar /= &Float::with_val(PREC, 2);

            let mut h_s = one_float();
            h_s /= &primes[s];
            let mut term = one_float();
            term /= &primes[s + 1];
            h_s += &term;

            let mut phi_abs = Float::with_val(PREC, &Phi[s]);
            phi_abs.abs_mut();
            let mut w_s = Float::with_val(PREC, &phi_abs);
            w_s += &ell_bar;
            w_s += &h_s;
            w_s *= &self.lambda_closure;
            W[s] = w_s;

            let mut ev: [Float; 4] = std::array::from_fn(|_| zero_float());
            for comp in 0..4 {
                let idx = (s + comp) % 5;
                let mut component = Float::with_val(PREC, &ell[idx]);
                if comp == s {
                    component += &one_float();
                    component += &one_float();
                }
                if comp == 3 {
                    let mut pa = Float::with_val(PREC, &Phi[s]);
                    pa.abs_mut();
                    component += &pa;
                }
                ev[comp] = component;
            }
            for i in 0..4 {
                for j in 0..4 {
                    let mut contrib = Float::with_val(PREC, &ev[i]);
                    contrib *= &ev[j];
                    contrib *= &W[s];
                    unstab[i][j] += &contrib;
                }
            }

            execution_vectors[s] = ev;

            let mut d = Float::with_val(PREC, &self.lambda_closure);
            d += &W[s];
            d += &ell[s];
            d += &ell[s + 1];
            diagonal[s] = d;
            unstab[s][s] += &diagonal[s];
        }

        let mut sym = zero_matrix_4x4();
        for i in 0..4 {
            for j in 0..4 {
                let mut val = Float::with_val(PREC, &unstab[i][j]);
                val += &unstab[j][i];
                val /= &Float::with_val(PREC, 2);
                sym[i][j] = val;
            }
        }

        let eigs_sym = jacobi_eigenvalues(&sym);
        let min_eig = Float::with_val(PREC, &eigs_sym[0]);
        let mut epsilon_eig = zero_float();
        if min_eig <= 0.0 {
            epsilon_eig = Float::with_val(PREC, &min_eig);
            epsilon_eig.abs_mut();
            epsilon_eig += &Float::with_val(PREC, f64::EPSILON);
        }

        let mut stabilized = clone_matrix_4x4(&sym);
        for i in 0..4 {
            stabilized[i][i] += &epsilon_eig;
        }
        let mut stabilized_sym = zero_matrix_4x4();
        for i in 0..4 {
            for j in 0..4 {
                let mut val = Float::with_val(PREC, &stabilized[i][j]);
                val += &stabilized[j][i];
                val /= &Float::with_val(PREC, 2);
                stabilized_sym[i][j] = val;
            }
        }

        let trace = trace_4x4(&stabilized_sym);
        let mut metric = clone_matrix_4x4(&stabilized_sym);
        for i in 0..4 {
            for j in 0..4 {
                metric[i][j] /= &trace;
            }
        }

        let spatial_metric = self.project_static_block_to_bulk(&metric);
        let eigenvalues = jacobi_eigenvalues(&metric);

        BulkMetricSlice {
            tau,
            load_vector: ell,
            euler_flux: Phi,
            execution_vectors,
            unstabilized_metric: unstab,
            metric_components: metric,
            spatial_metric,
            epsilon_eig,
            eigenvalues,
        }
    }

    pub fn project_static_block_to_bulk(&self, metric: &[[Float; 4]; 4]) -> [[Float; 3]; 3] {
        let mut result = zero_matrix_3x3();
        for i in 0..3 {
            for j in 0..3 {
                let mut sum = zero_float();
                for k in 0..4 {
                    for l in 0..4 {
                        let mut term = Float::with_val(PREC, &self.P[i][k]);
                        term *= &metric[k][l];
                        term *= &self.P[j][l];
                        sum += &term;
                    }
                }
                result[i][j] = sum;
            }
        }
        result
    }

    pub fn project_entropy_cascade(&self) -> Vec<BulkMetricSlice> {
        let updates = self.boundary.entropy_self_resolution();
        let mut state: [[Float; 3]; 3] =
            std::array::from_fn(|_| std::array::from_fn(|_| zero_float()));
        let mut slices = Vec::with_capacity(updates.len());

        for update in updates {
            let (i, j) = update.coordinate;
            state[i][j] += &update.delta_T;
            let load = self.derive_load_vector(&state);
            let tau = Float::with_val(PREC, update.n as u32);
            slices.push(self.metric_from_load_vector(&load, tau));
        }

        slices
    }

    pub fn verify_projection(&self, slices: &[BulkMetricSlice]) -> ProjectionReport {
        let tol = Float::with_val(PREC, 1e-12);
        let one = one_float();

        let mut symmetric = true;
        let mut trace_normalized = true;
        let mut positive_definite = true;

        for slice in slices {
            for i in 0..4 {
                for j in 0..4 {
                    if !StaticBoundary::is_close(
                        &slice.metric_components[i][j],
                        &slice.metric_components[j][i],
                        &tol,
                    ) {
                        symmetric = false;
                    }
                }
            }

            let trace = trace_4x4(&slice.metric_components);
            if !StaticBoundary::is_close(&trace, &one, &tol) {
                trace_normalized = false;
            }

            if slice.eigenvalues[0] <= 0.0 {
                positive_definite = false;
            }
        }

        let spatial_metric_shape = slices.iter().all(|s| s.spatial_metric.len() == 3);

        ProjectionReport {
            slice_count: slices.len(),
            projector_rank: 3,
            symmetric,
            trace_normalized,
            positive_definite,
            spatial_metric_shape,
            all_passed: symmetric && trace_normalized && positive_definite && spatial_metric_shape,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn derive_load_vector_positive_and_normalized() {
        let boundary = StaticBoundary::new();
        let projection = HolographicProjection::new(boundary.clone());
        let state = boundary.build_entanglement_density();
        let load = projection.derive_load_vector(&state);

        for entry in &load {
            assert!(*entry > 0.0, "load vector entry must be positive");
        }

        let mut sum = zero_float();
        for entry in &load {
            sum += entry;
        }
        assert!(StaticBoundary::is_close(
            &sum,
            &one_float(),
            &Float::with_val(PREC, 1e-12)
        ));
    }

    #[test]
    fn metric_from_load_vector_is_valid_metric() {
        let boundary = StaticBoundary::new();
        let projection = HolographicProjection::new(boundary);
        let load: [Float; 5] = std::array::from_fn(|_| Float::with_val(PREC, 0.1));
        let tau = Float::with_val(PREC, 0);
        let slice = projection.metric_from_load_vector(&load, tau);

        let tol = Float::with_val(PREC, 1e-12);
        let one = one_float();

        for i in 0..4 {
            for j in 0..4 {
                assert!(StaticBoundary::is_close(
                    &slice.metric_components[i][j],
                    &slice.metric_components[j][i],
                    &tol
                ));
            }
        }

        let trace = trace_4x4(&slice.metric_components);
        assert!(StaticBoundary::is_close(&trace, &one, &tol));
        assert!(
            slice.eigenvalues[0] > 0.0,
            "metric must be positive definite"
        );
    }

    #[test]
    fn project_entropy_cascade_nine_slices() {
        let boundary = StaticBoundary::new();
        let projection = HolographicProjection::new(boundary);
        let slices = projection.project_entropy_cascade();
        assert_eq!(slices.len(), 9);
    }

    #[test]
    fn verify_projection_all_passed_for_benchmark() {
        let boundary = StaticBoundary::new();
        let projection = HolographicProjection::new(boundary);
        let slices = projection.project_entropy_cascade();
        let report = projection.verify_projection(&slices);
        assert!(report.all_passed);
    }
}
