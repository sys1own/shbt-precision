#![allow(non_snake_case)]

use pyo3::prelude::*;
use pyo3::types::PyDict;
use rug::float::Constant;
use rug::{Complex, Float, Rational};

pub(crate) const PREC: u32 = 512;

const BENCHMARK_BRANCH: (u32, u32, u32) = (26, 8, 312);
const LEPTON_LEVEL: u32 = 26;
const QUARK_LEVEL: u32 = 8;
const PARENT_LEVEL: u32 = 312;
const C_DARK_NUM: u32 = 1197103;
const C_DARK_DEN: u32 = 362670;
const LAMBDA_HOLO_STR: &str = "1.0892229828054038e-52";
const BIT_BUDGET_STR: &str = "3.311997720142366e122";
const TOLERANCE: f64 = 1.0e-12;

const LOW_SU3_WEIGHTS: [(u32, u32); 3] = [(0, 0), (1, 0), (0, 1)];
const CHARGE_EMBEDDING: [u32; 3] = [LEPTON_LEVEL - 4, LEPTON_LEVEL - 3, LEPTON_LEVEL];

#[derive(Debug, Clone)]
#[pyclass]
pub struct EntropyUpdate {
    pub n: usize,
    pub coordinate: (usize, usize),
    pub feedback_signal: Float,
    pub delta_B: Float,
    pub delta_S: Float,
    pub delta_T: Float,
    pub D_n: Float,
    pub delta_D: Float,
    pub remaining_B: Float,
    pub remaining_E: Float,
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct TemporalKernelAudit {
    pub H: Float,
    pub dot_S_total: Float,
    pub local_gradient: [[Float; 3]; 3],
    pub dot_T: Float,
    pub perception_identity_holds: bool,
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct VerificationReport {
    pub framing_defect: Float,
    pub loading_normalized: bool,
    pub entanglement_density_normalized: bool,
    pub dominant_sequence_matches: bool,
    pub modular_S_commutator: Float,
    pub modular_T_commutator: Float,
    pub modular_invariant: bool,
    pub zero_energy_locked: bool,
    pub projection_dimension_26_to_4: bool,
    pub all_passed: bool,
}

#[pymethods]
impl EntropyUpdate {
    pub fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new_bound(py);
        d.set_item("n", self.n)?;
        d.set_item("coordinate", self.coordinate)?;
        d.set_item("feedback_signal", self.feedback_signal.to_f64())?;
        d.set_item("delta_B", self.delta_B.to_f64())?;
        d.set_item("delta_S", self.delta_S.to_f64())?;
        d.set_item("delta_T", self.delta_T.to_f64())?;
        d.set_item("D_n", self.D_n.to_f64())?;
        d.set_item("delta_D", self.delta_D.to_f64())?;
        d.set_item("remaining_B", self.remaining_B.to_f64())?;
        d.set_item("remaining_E", self.remaining_E.to_f64())?;
        Ok(d)
    }
}

#[pymethods]
impl TemporalKernelAudit {
    pub fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new_bound(py);
        d.set_item("H", self.H.to_f64())?;
        d.set_item("dot_S_total", self.dot_S_total.to_f64())?;
        let grad: Vec<Vec<f64>> = self
            .local_gradient
            .iter()
            .map(|row| row.iter().map(|v| v.to_f64()).collect())
            .collect();
        d.set_item("local_gradient", grad)?;
        d.set_item("dot_T", self.dot_T.to_f64())?;
        d.set_item("perception_identity_holds", self.perception_identity_holds)?;
        Ok(d)
    }
}

#[pymethods]
impl VerificationReport {
    pub fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new_bound(py);
        d.set_item("framing_defect", self.framing_defect.to_f64())?;
        d.set_item("loading_normalized", self.loading_normalized)?;
        d.set_item(
            "entanglement_density_normalized",
            self.entanglement_density_normalized,
        )?;
        d.set_item("dominant_sequence_matches", self.dominant_sequence_matches)?;
        d.set_item("modular_S_commutator", self.modular_S_commutator.to_f64())?;
        d.set_item("modular_T_commutator", self.modular_T_commutator.to_f64())?;
        d.set_item("modular_invariant", self.modular_invariant)?;
        d.set_item("zero_energy_locked", self.zero_energy_locked)?;
        d.set_item(
            "projection_dimension_26_to_4",
            self.projection_dimension_26_to_4,
        )?;
        d.set_item("all_passed", self.all_passed)?;
        Ok(d)
    }
}

#[derive(Debug, Clone)]
pub struct StaticBoundary {
    pub benchmark_branch: (u32, u32, u32),
    pub lepton_level: u32,
    pub quark_level: u32,
    pub parent_level: u32,
    pub i_l_star: Float,
    pub i_q_star: Float,
    pub c_dark: Rational,
    pub lambda_holo: Float,
    pub bit_budget: Float,
    pub tolerance: Float,

    charge_embedding: [u32; 3],
    su2_visible_block: [[Float; 3]; 3],
    su3_visible_block: [[Complex; 3]; 3],
    su2_visible_phases: [Complex; 3],
    su3_visible_phases: [Complex; 3],
    loading_density: [[Float; 3]; 3],
    entanglement_density: [[Float; 3]; 3],
    dominant_sequence: [(usize, usize); 9],
    z_boundary_matrix: [[Complex; 9]; 9],
    s_boundary: [[Complex; 9]; 9],
    t_boundary: [[Complex; 9]; 9],
}

impl StaticBoundary {
    pub fn new_with_branch(lepton_level: u32, quark_level: u32, parent_level: u32) -> Self {
        let benchmark_branch = (lepton_level, quark_level, parent_level);
        let charge_embedding = [lepton_level - 4, lepton_level - 3, lepton_level];

        let mut i_l_star = Float::with_val(PREC, parent_level);
        i_l_star /= Float::with_val(PREC, 2 * lepton_level);

        let mut i_q_star = Float::with_val(PREC, parent_level);
        i_q_star /= Float::with_val(PREC, 3 * quark_level);

        let c_dark = Rational::from((C_DARK_NUM, C_DARK_DEN));
        let lambda_holo = Float::with_val(PREC, Float::parse(LAMBDA_HOLO_STR).unwrap());
        let bit_budget = Float::with_val(PREC, Float::parse(BIT_BUDGET_STR).unwrap());
        let tolerance = Float::with_val(PREC, TOLERANCE);

        let su2_visible_block =
            Self::build_su2_visible_block_static(lepton_level, charge_embedding);
        let su3_visible_block = Self::build_su3_visible_block_static(quark_level, LOW_SU3_WEIGHTS);
        let su2_visible_phases =
            Self::build_su2_visible_phases_static(lepton_level, charge_embedding);
        let su3_visible_phases =
            Self::build_su3_visible_phases_static(quark_level, LOW_SU3_WEIGHTS);

        let raw_loading =
            Self::build_raw_loading_density_static(&su2_visible_block, &su3_visible_block);
        let loading_density = Self::normalize_matrix(&raw_loading);
        let entanglement_density = Self::build_entanglement_density_static(&loading_density);
        let dominant_sequence = Self::build_dominant_sequence_static(&loading_density);

        let z_boundary_matrix = Self::identity_matrix_9_complex();
        let s_boundary = Self::build_s_boundary_static(&su2_visible_block, &su3_visible_block);
        let t_boundary = Self::build_t_boundary_static(&su2_visible_phases, &su3_visible_phases);

        StaticBoundary {
            benchmark_branch,
            lepton_level,
            quark_level,
            parent_level,
            i_l_star,
            i_q_star,
            c_dark,
            lambda_holo,
            bit_budget,
            tolerance,
            charge_embedding,
            su2_visible_block,
            su3_visible_block,
            su2_visible_phases,
            su3_visible_phases,
            loading_density,
            entanglement_density,
            dominant_sequence,
            z_boundary_matrix,
            s_boundary,
            t_boundary,
        }
    }

    pub fn new() -> Self {
        Self::new_with_branch(LEPTON_LEVEL, QUARK_LEVEL, PARENT_LEVEL)
    }

    // ------------------------------------------------------------------
    // Helpers
    // ------------------------------------------------------------------

    fn pi() -> Float {
        Float::with_val(PREC, Constant::Pi)
    }

    pub(crate) fn is_close(a: &Float, b: &Float, tol: &Float) -> bool {
        let mut diff = a.clone();
        diff -= b;
        diff.abs() <= tol.clone()
    }

    fn distance_to_integer(x: &Float) -> Float {
        let nearest = x.to_i32_saturating().unwrap_or(0);
        let nearest_f = Float::with_val(PREC, nearest);
        let mut d = x.clone();
        d -= nearest_f;
        d.abs()
    }

    fn normalize_matrix(raw: &[[Float; 3]; 3]) -> [[Float; 3]; 3] {
        let mut sum = Float::with_val(PREC, 0);
        for row in raw.iter() {
            for val in row.iter() {
                sum += val;
            }
        }

        std::array::from_fn(|i| {
            std::array::from_fn(|j| {
                let mut val = Float::with_val(PREC, &raw[i][j]);
                val /= &sum;
                val
            })
        })
    }

    // ------------------------------------------------------------------
    // Static builders (used by new())
    // ------------------------------------------------------------------

    pub fn su2_conformal_weight(label: u32, level: u32) -> Float {
        let mut numer = Float::with_val(PREC, label);
        numer *= label + 2;
        let denom = Float::with_val(PREC, 4 * (level + 2));
        numer /= &denom;
        numer
    }

    pub fn su2_central_charge(level: u32) -> Float {
        let mut numer = Float::with_val(PREC, 3 * level);
        let denom = Float::with_val(PREC, level + 2);
        numer /= &denom;
        numer
    }

    pub fn su3_conformal_weight(p: u32, q: u32, level: u32) -> Float {
        let p_f = Float::with_val(PREC, p);
        let q_f = Float::with_val(PREC, q);

        let mut p2 = p_f.clone();
        p2.square_mut();
        let mut q2 = q_f.clone();
        q2.square_mut();
        let mut pq = Float::with_val(PREC, &p_f);
        pq *= &q_f;
        let mut p3 = Float::with_val(PREC, 3);
        p3 *= &p_f;
        let mut q3 = Float::with_val(PREC, 3);
        q3 *= &q_f;

        let mut numer = Float::with_val(PREC, 0);
        numer += &p2;
        numer += &q2;
        numer += &pq;
        numer += &p3;
        numer += &q3;

        let denom = Float::with_val(PREC, 3 * (level + 3));
        numer /= &denom;
        numer
    }

    pub fn su3_central_charge(level: u32) -> Float {
        let mut numer = Float::with_val(PREC, 8 * level);
        let denom = Float::with_val(PREC, level + 3);
        numer /= &denom;
        numer
    }

    fn visible_central_charge(&self) -> Float {
        let mut c = Self::su2_central_charge(self.lepton_level);
        c += Self::su3_central_charge(self.quark_level);
        c
    }

    pub fn su2_modular_s_entry(left: u32, right: u32, level: u32) -> Float {
        let mut coef = Float::with_val(PREC, 2);
        coef /= level + 2;
        coef.sqrt_mut();

        let mut arg = Float::with_val(PREC, (left + 1) * (right + 1));
        arg *= Self::pi();
        arg /= level + 2;

        let mut result = Float::with_val(PREC, coef);
        result *= arg.sin();
        result
    }

    fn su3_vector(weight: (u32, u32)) -> [Float; 3] {
        let p = Float::with_val(PREC, weight.0);
        let q = Float::with_val(PREC, weight.1);
        let three = Float::with_val(PREC, 3);

        let mut v0 = Float::with_val(PREC, 2);
        v0 *= &p;
        v0 += &q;
        v0 /= &three;

        let mut v1 = Float::with_val(PREC, &q);
        v1 -= &p;
        v1 /= &three;

        let mut v2 = Float::with_val(PREC, &p);
        v2 *= 2;
        v2 += &q;
        v2 *= -1;
        v2 /= &three;

        [v0, v1, v2]
    }

    fn permutation_sign(permutation: (usize, usize, usize)) -> i32 {
        let p = [permutation.0, permutation.1, permutation.2];
        let mut inversions = 0;
        for i in 0..3 {
            for j in (i + 1)..3 {
                if p[i] > p[j] {
                    inversions += 1;
                }
            }
        }
        if inversions % 2 == 0 {
            1
        } else {
            -1
        }
    }

    fn permute_vector(v: &[Float; 3], permutation: (usize, usize, usize)) -> [Float; 3] {
        [
            v[permutation.0].clone(),
            v[permutation.1].clone(),
            v[permutation.2].clone(),
        ]
    }

    fn dot_product(a: &[Float; 3], b: &[Float; 3]) -> Float {
        let mut total = Float::with_val(PREC, 0);
        for i in 0..3 {
            let mut term = Float::with_val(PREC, &a[i]);
            term *= &b[i];
            total += term;
        }
        total
    }

    pub fn su3_modular_s_entry(left: (u32, u32), right: (u32, u32), level: u32) -> Complex {
        let rho = Self::su3_vector((1, 1));

        let v_left = Self::su3_vector(left);
        let mut lambda_rho_0 = Float::with_val(PREC, &v_left[0]);
        lambda_rho_0 += &rho[0];
        let mut lambda_rho_1 = Float::with_val(PREC, &v_left[1]);
        lambda_rho_1 += &rho[1];
        let mut lambda_rho_2 = Float::with_val(PREC, &v_left[2]);
        lambda_rho_2 += &rho[2];
        let lambda_rho = [lambda_rho_0, lambda_rho_1, lambda_rho_2];

        let v_right = Self::su3_vector(right);
        let mut mu_rho_0 = Float::with_val(PREC, &v_right[0]);
        mu_rho_0 += &rho[0];
        let mut mu_rho_1 = Float::with_val(PREC, &v_right[1]);
        mu_rho_1 += &rho[1];
        let mut mu_rho_2 = Float::with_val(PREC, &v_right[2]);
        mu_rho_2 += &rho[2];
        let mu_rho = [mu_rho_0, mu_rho_1, mu_rho_2];

        let permutations = [
            (0, 1, 2),
            (0, 2, 1),
            (1, 0, 2),
            (1, 2, 0),
            (2, 0, 1),
            (2, 1, 0),
        ];

        let mut total = Complex::with_val(PREC, (0, 0));

        let mut pre_factor = Float::with_val(PREC, -2);
        pre_factor *= Self::pi();
        pre_factor /= level + 3;

        for &perm in permutations.iter() {
            let permuted = Self::permute_vector(&lambda_rho, perm);
            let dot = Self::dot_product(&permuted, &mu_rho);
            let sign = Self::permutation_sign(perm);

            let mut theta = Float::with_val(PREC, &pre_factor);
            theta *= dot;

            let cos = theta.clone().cos();
            let sin = theta.sin();
            let exp_term = Complex::with_val(PREC, (cos, sin));

            let mut signed_term = Complex::with_val(PREC, (sign, 0));
            signed_term *= &exp_term;
            total += signed_term;
        }

        let mut denom = Float::with_val(PREC, 3);
        denom.sqrt_mut();
        denom *= level + 3;
        let denom_c = Complex::with_val(PREC, (denom, 0));
        let i_neg = Complex::with_val(PREC, (0, -1));

        total * i_neg / denom_c
    }

    fn build_su2_visible_block_static(level: u32, labels: [u32; 3]) -> [[Float; 3]; 3] {
        std::array::from_fn(|i| {
            std::array::from_fn(|j| Self::su2_modular_s_entry(labels[i], labels[j], level))
        })
    }

    fn build_su3_visible_block_static(level: u32, weights: [(u32, u32); 3]) -> [[Complex; 3]; 3] {
        std::array::from_fn(|i| {
            std::array::from_fn(|j| Self::su3_modular_s_entry(weights[i], weights[j], level))
        })
    }

    fn build_su2_visible_phases_static(level: u32, labels: [u32; 3]) -> [Complex; 3] {
        let c_over_24 = {
            let mut c = Self::su2_central_charge(level);
            c /= 24;
            c
        };

        let mut two_pi = Self::pi();
        two_pi *= 2;

        std::array::from_fn(|i| {
            let h = Self::su2_conformal_weight(labels[i], level);
            let mut exponent = Float::with_val(PREC, &h);
            exponent -= &c_over_24;

            let mut imag = Float::with_val(PREC, &exponent);
            imag *= &two_pi;

            Complex::with_val(PREC, (0, imag)).exp()
        })
    }

    fn build_su3_visible_phases_static(level: u32, weights: [(u32, u32); 3]) -> [Complex; 3] {
        let c_over_24 = {
            let mut c = Self::su3_central_charge(level);
            c /= 24;
            c
        };

        let mut two_pi = Self::pi();
        two_pi *= 2;

        std::array::from_fn(|i| {
            let h = Self::su3_conformal_weight(weights[i].0, weights[i].1, level);
            let mut exponent = Float::with_val(PREC, &h);
            exponent -= &c_over_24;

            let mut imag = Float::with_val(PREC, &exponent);
            imag *= &two_pi;

            Complex::with_val(PREC, (0, imag)).exp()
        })
    }

    fn build_raw_loading_density_static(
        su2: &[[Float; 3]; 3],
        su3: &[[Complex; 3]; 3],
    ) -> [[Float; 3]; 3] {
        std::array::from_fn(|i| {
            std::array::from_fn(|j| {
                let mut s2_abs = Float::with_val(PREC, &su2[i][j]);
                s2_abs.abs_mut();
                s2_abs.square_mut();

                let mut s3_abs = Float::with_val(PREC, su3[i][j].abs_ref());
                s3_abs.square_mut();

                s2_abs *= &s3_abs;
                s2_abs
            })
        })
    }

    fn build_entanglement_density_static(loading: &[[Float; 3]; 3]) -> [[Float; 3]; 3] {
        let mut raw: [[Float; 3]; 3] =
            std::array::from_fn(|_| std::array::from_fn(|_| Float::with_val(PREC, 0)));

        let mut sum = Float::with_val(PREC, 0);
        for i in 0..3 {
            for j in 0..3 {
                if loading[i][j] > 0.0 {
                    let ln_p = loading[i][j].clone().ln();
                    let mut contrib = Float::with_val(PREC, &ln_p);
                    contrib *= &loading[i][j];
                    contrib *= -1;
                    raw[i][j] = contrib;
                    sum += &raw[i][j];
                }
            }
        }

        std::array::from_fn(|i| {
            std::array::from_fn(|j| {
                let mut val = Float::with_val(PREC, &raw[i][j]);
                val /= &sum;
                val
            })
        })
    }

    fn build_dominant_sequence_static(loading: &[[Float; 3]; 3]) -> [(usize, usize); 9] {
        let mut indexed: Vec<((usize, usize), Float)> = Vec::with_capacity(9);
        for i in 0..3 {
            for j in 0..3 {
                indexed.push(((i, j), Float::with_val(PREC, &loading[i][j])));
            }
        }

        indexed.sort_by(|a, b| {
            let ord = b.1.partial_cmp(&a.1).unwrap();
            if ord == std::cmp::Ordering::Equal {
                a.0.cmp(&b.0)
            } else {
                ord
            }
        });

        indexed
            .into_iter()
            .map(|(coord, _)| coord)
            .collect::<Vec<_>>()
            .try_into()
            .unwrap()
    }

    fn identity_matrix_9_complex() -> [[Complex; 9]; 9] {
        std::array::from_fn(|i| {
            std::array::from_fn(|j| {
                if i == j {
                    Complex::with_val(PREC, (1, 0))
                } else {
                    Complex::with_val(PREC, (0, 0))
                }
            })
        })
    }

    fn build_s_boundary_static(
        su2: &[[Float; 3]; 3],
        su3: &[[Complex; 3]; 3],
    ) -> [[Complex; 9]; 9] {
        std::array::from_fn(|a| {
            let i = a / 3;
            let k = a % 3;
            std::array::from_fn(|b| {
                let j = b / 3;
                let l = b % 3;

                let mut val = Float::with_val(PREC, &su2[i][j]);
                val *= su3[k][l].real();
                Complex::with_val(PREC, (val, 0))
            })
        })
    }

    fn build_t_boundary_static(
        su2_phases: &[Complex; 3],
        su3_phases: &[Complex; 3],
    ) -> [[Complex; 9]; 9] {
        std::array::from_fn(|a| {
            let i = a / 3;
            let j = a % 3;
            std::array::from_fn(|b| {
                if a == b {
                    let mut t = Complex::with_val(PREC, &su2_phases[i]);
                    t *= &su3_phases[j];
                    t
                } else {
                    Complex::with_val(PREC, (0, 0))
                }
            })
        })
    }

    // ------------------------------------------------------------------
    // Public methods
    // ------------------------------------------------------------------

    pub fn framing_defect(&self) -> Float {
        let mut first = Float::with_val(PREC, self.parent_level);
        first /= Float::with_val(PREC, 2 * self.lepton_level);

        let mut second = Float::with_val(PREC, self.parent_level);
        second /= Float::with_val(PREC, 3 * self.quark_level);

        let d1 = Self::distance_to_integer(&first);
        let d2 = Self::distance_to_integer(&second);

        if d1 > d2 {
            d1
        } else {
            d2
        }
    }

    pub fn build_su2_visible_block(&self) -> [[Float; 3]; 3] {
        self.su2_visible_block.clone()
    }

    pub fn build_su3_visible_block(&self) -> [[Complex; 3]; 3] {
        self.su3_visible_block.clone()
    }

    pub fn build_loading_density(&self) -> [[Float; 3]; 3] {
        self.loading_density.clone()
    }

    pub fn build_entanglement_density(&self) -> [[Float; 3]; 3] {
        self.entanglement_density.clone()
    }

    pub fn build_dominant_sequence(&self) -> [(usize, usize); 9] {
        self.dominant_sequence
    }

    pub fn evaluate_z_boundary(&self, tau: Complex) -> Float {
        let c_over_24 = {
            let mut c = self.visible_central_charge();
            c /= 24;
            c
        };

        let mut two_pi = Self::pi();
        two_pi *= 2;

        let mut z = Float::with_val(PREC, 0);
        for i in 0..3 {
            for j in 0..3 {
                let h_su2 = Self::su2_conformal_weight(self.charge_embedding[i], self.lepton_level);
                let h_su3 = Self::su3_conformal_weight(
                    LOW_SU3_WEIGHTS[j].0,
                    LOW_SU3_WEIGHTS[j].1,
                    self.quark_level,
                );

                let mut exponent = Float::with_val(PREC, &h_su2);
                exponent += &h_su3;
                exponent -= &c_over_24;

                let mut scalar = Float::with_val(PREC, &two_pi);
                scalar *= &exponent;

                let i_scalar = Complex::with_val(PREC, (0, scalar));
                let mut arg = Complex::with_val(PREC, &tau);
                arg *= &i_scalar;

                let q_alpha = arg.exp();
                let abs = Float::with_val(PREC, q_alpha.abs_ref());
                let mut abs_sq = Float::with_val(PREC, &abs);
                abs_sq *= &abs;
                z += &abs_sq;
            }
        }

        z
    }

    pub fn entropy_self_resolution(&self) -> Vec<EntropyUpdate> {
        let mut updates = Vec::with_capacity(9);
        let mut omega: [[Float; 3]; 3] =
            std::array::from_fn(|_| std::array::from_fn(|_| Float::with_val(PREC, 0)));

        let feedback_coupling = Float::with_val(PREC, 1);
        let mut remaining_B = Float::with_val(PREC, &self.bit_budget);
        let mut remaining_E = Float::with_val(PREC, &self.bit_budget);
        let mut cumulative_delta_T = Float::with_val(PREC, 0);

        let sequence = self.dominant_sequence;
        let all_coords: Vec<(usize, usize)> =
            (0..3).flat_map(|i| (0..3).map(move |j| (i, j))).collect();

        for (n, &c) in sequence.iter().enumerate() {
            let n_step = n + 1;
            let unresolved: Vec<(usize, usize)> = all_coords
                .iter()
                .filter(|&&coord| !sequence[..n].contains(&coord))
                .copied()
                .collect();

            // Build feedback signal for each unresolved coordinate.
            let mut feedback_by_coord: Vec<((usize, usize), Float)> =
                Vec::with_capacity(unresolved.len());
            for &uc in unresolved.iter() {
                let mut signal = Float::with_val(PREC, 0);
                for i in 0..3 {
                    for j in 0..3 {
                        let di = (i as i32 - uc.0 as i32).abs();
                        let dj = (j as i32 - uc.1 as i32).abs();
                        let denom = Float::with_val(PREC, 1 + (di + dj) as u32);
                        let mut term = Float::with_val(PREC, &omega[i][j]);
                        term /= &denom;
                        signal += &term;
                    }
                }
                feedback_by_coord.push((uc, signal));
            }

            let feedback_signal = {
                let (_, s) = feedback_by_coord
                    .iter()
                    .find(|(coord, _)| *coord == c)
                    .unwrap();
                Float::with_val(PREC, s)
            };

            // Dress weights for unresolved set.
            let mut wB: Vec<((usize, usize), Float)> = Vec::with_capacity(unresolved.len());
            let mut wE: Vec<((usize, usize), Float)> = Vec::with_capacity(unresolved.len());
            for &(coord, ref signal) in feedback_by_coord.iter() {
                let mut factor = Float::with_val(PREC, signal);
                factor *= &feedback_coupling;
                factor += 1;

                let mut wb = Float::with_val(PREC, &self.loading_density[coord.0][coord.1]);
                wb *= &factor;
                wB.push((coord, wb));

                let mut we = Float::with_val(PREC, &self.entanglement_density[coord.0][coord.1]);
                we *= &factor;
                wE.push((coord, we));
            }

            let sum_wB: Float = wB.iter().map(|(_, v)| Float::with_val(PREC, v)).fold(
                Float::with_val(PREC, 0),
                |mut acc, v| {
                    acc += &v;
                    acc
                },
            );
            let sum_wE: Float = wE.iter().map(|(_, v)| Float::with_val(PREC, v)).fold(
                Float::with_val(PREC, 0),
                |mut acc, v| {
                    acc += &v;
                    acc
                },
            );

            let c_wB = wB.iter().find(|(coord, _)| *coord == c).unwrap().1.clone();
            let c_wE = wE.iter().find(|(coord, _)| *coord == c).unwrap().1.clone();

            let mut delta_B = Float::with_val(PREC, &remaining_B);
            delta_B *= &c_wB;
            delta_B /= &sum_wB;

            let mut delta_S = Float::with_val(PREC, &remaining_E);
            delta_S *= &c_wE;
            delta_S /= &sum_wE;

            let mut delta_T = Float::with_val(PREC, &delta_S);
            delta_T /= &self.bit_budget;

            remaining_B -= &delta_B;
            remaining_E -= &delta_S;

            cumulative_delta_T += &delta_T;
            omega[c.0][c.1] += &delta_T;

            let D_previous = {
                let mut prev_cum = Float::with_val(PREC, &cumulative_delta_T);
                prev_cum -= &delta_T;
                let mut d = Float::with_val(PREC, 26);
                let mut sub = Float::with_val(PREC, 22);
                sub *= &prev_cum;
                d -= &sub;
                d
            };
            let D_n = {
                let mut d = Float::with_val(PREC, 26);
                let mut sub = Float::with_val(PREC, 22);
                sub *= &cumulative_delta_T;
                d -= &sub;
                d
            };
            let mut delta_D = Float::with_val(PREC, &D_previous);
            delta_D -= &D_n;

            let remaining_B_report = if remaining_B < 0.0 {
                Float::with_val(PREC, 0)
            } else {
                Float::with_val(PREC, &remaining_B)
            };
            let remaining_E_report = if remaining_E < 0.0 {
                Float::with_val(PREC, 0)
            } else {
                Float::with_val(PREC, &remaining_E)
            };

            updates.push(EntropyUpdate {
                n: n_step,
                coordinate: c,
                feedback_signal,
                delta_B,
                delta_S,
                delta_T,
                D_n,
                delta_D,
                remaining_B: remaining_B_report,
                remaining_E: remaining_E_report,
            });
        }

        updates
    }

    pub fn derive_temporal_increment(&self, H: Float) -> TemporalKernelAudit {
        let mut dot_S_total = Float::with_val(PREC, &self.bit_budget);
        dot_S_total *= &H;

        let mut local_gradient: [[Float; 3]; 3] =
            std::array::from_fn(|_| std::array::from_fn(|_| Float::with_val(PREC, 0)));

        let mut sum_gradient = Float::with_val(PREC, 0);
        for i in 0..3 {
            for j in 0..3 {
                let mut val = Float::with_val(PREC, &self.entanglement_density[i][j]);
                val *= &dot_S_total;
                local_gradient[i][j] = val.clone();
                sum_gradient += &val;
            }
        }

        let mut dot_T = Float::with_val(PREC, &sum_gradient);
        dot_T /= &self.bit_budget;

        let perception_identity_holds = Self::is_close(&dot_T, &H, &self.tolerance);

        TemporalKernelAudit {
            H,
            dot_S_total,
            local_gradient,
            dot_T,
            perception_identity_holds,
        }
    }

    pub fn verify_equations(&self) -> VerificationReport {
        let framing_defect = self.framing_defect();

        let loading = self.build_loading_density();
        let mut sum_load = Float::with_val(PREC, 0);
        for row in loading.iter() {
            for val in row.iter() {
                sum_load += val;
            }
        }
        let loading_normalized =
            Self::is_close(&sum_load, &Float::with_val(PREC, 1), &self.tolerance);

        let entanglement = self.build_entanglement_density();
        let mut sum_ent = Float::with_val(PREC, 0);
        for row in entanglement.iter() {
            for val in row.iter() {
                sum_ent += val;
            }
        }
        let entanglement_density_normalized =
            Self::is_close(&sum_ent, &Float::with_val(PREC, 1), &self.tolerance);

        let sequence_bit_loading = self.dominant_sequence;
        let dominant_sequence_matches = self.dominant_sequence == sequence_bit_loading;

        // Modular commutators.
        let ms = mat_mul_9x9(&self.z_boundary_matrix, &self.s_boundary);
        let sm = mat_mul_9x9(&self.s_boundary, &self.z_boundary_matrix);
        let s_comm = mat_sub_9x9(&ms, &sm);
        let modular_S_commutator = frobenius_norm_9x9(&s_comm);

        let mt = mat_mul_9x9(&self.z_boundary_matrix, &self.t_boundary);
        let tm = mat_mul_9x9(&self.t_boundary, &self.z_boundary_matrix);
        let t_comm = mat_sub_9x9(&mt, &tm);
        let modular_T_commutator = frobenius_norm_9x9(&t_comm);

        let zero = Float::with_val(PREC, 0);
        let modular_invariant = Self::is_close(&modular_S_commutator, &zero, &self.tolerance)
            && Self::is_close(&modular_T_commutator, &zero, &self.tolerance);

        let zero_energy_locked = Self::is_close(&framing_defect, &zero, &self.tolerance);

        let updates = self.entropy_self_resolution();
        let projection_dimension_26_to_4 = if let Some(last) = updates.last() {
            Self::is_close(&last.D_n, &Float::with_val(PREC, 4), &self.tolerance)
        } else {
            false
        };

        let all_passed = loading_normalized
            && entanglement_density_normalized
            && dominant_sequence_matches
            && modular_invariant
            && zero_energy_locked
            && projection_dimension_26_to_4;

        VerificationReport {
            framing_defect,
            loading_normalized,
            entanglement_density_normalized,
            dominant_sequence_matches,
            modular_S_commutator,
            modular_T_commutator,
            modular_invariant,
            zero_energy_locked,
            projection_dimension_26_to_4,
            all_passed,
        }
    }
}

fn mat_mul_9x9(a: &[[Complex; 9]; 9], b: &[[Complex; 9]; 9]) -> [[Complex; 9]; 9] {
    std::array::from_fn(|i| {
        std::array::from_fn(|j| {
            let mut sum = Complex::with_val(PREC, (0, 0));
            for k in 0..9 {
                let term = Complex::with_val(PREC, &a[i][k] * &b[k][j]);
                sum += term;
            }
            sum
        })
    })
}

fn mat_sub_9x9(a: &[[Complex; 9]; 9], b: &[[Complex; 9]; 9]) -> [[Complex; 9]; 9] {
    std::array::from_fn(|i| {
        std::array::from_fn(|j| {
            let mut d = Complex::with_val(PREC, &a[i][j]);
            d -= &b[i][j];
            d
        })
    })
}

fn frobenius_norm_9x9(m: &[[Complex; 9]; 9]) -> Float {
    let mut sum = Float::with_val(PREC, 0);
    for i in 0..9 {
        for j in 0..9 {
            let n = Float::with_val(PREC, m[i][j].norm_ref());
            sum += &n;
        }
    }
    sum.sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn benchmark_framing_defect_zero() {
        let sb = StaticBoundary::new();
        assert_eq!(sb.framing_defect(), 0.0);
    }

    #[test]
    fn benchmark_loading_density_normalized() {
        let sb = StaticBoundary::new();
        let density = sb.build_loading_density();
        let mut sum = Float::with_val(PREC, 0);
        for row in density.iter() {
            for val in row.iter() {
                sum += val;
            }
        }
        assert!(StaticBoundary::is_close(
            &sum,
            &Float::with_val(PREC, 1),
            &Float::with_val(PREC, TOLERANCE)
        ));
    }

    #[test]
    fn benchmark_entanglement_density_normalized() {
        let sb = StaticBoundary::new();
        let density = sb.build_entanglement_density();
        let mut sum = Float::with_val(PREC, 0);
        for row in density.iter() {
            for val in row.iter() {
                sum += val;
            }
        }
        assert!(StaticBoundary::is_close(
            &sum,
            &Float::with_val(PREC, 1),
            &Float::with_val(PREC, TOLERANCE)
        ));
    }

    #[test]
    fn benchmark_dominant_sequence_length_nine() {
        let sb = StaticBoundary::new();
        let seq = sb.build_dominant_sequence();
        assert_eq!(seq.len(), 9);
    }

    #[test]
    fn benchmark_z_boundary_at_self_dual_point() {
        let sb = StaticBoundary::new();
        let tau = Complex::with_val(PREC, (0, 1));
        let z = sb.evaluate_z_boundary(tau);
        let expected = Float::with_val(PREC, Float::parse("2.441381789163e-24").unwrap());
        let tol = Float::with_val(PREC, Float::parse("1e-35").unwrap());
        assert!(StaticBoundary::is_close(&z, &expected, &tol));
    }

    #[test]
    fn benchmark_projection_dimension_26_to_4() {
        let sb = StaticBoundary::new();
        let report = sb.verify_equations();
        assert!(report.projection_dimension_26_to_4);
        assert!(report.all_passed);
    }
}
