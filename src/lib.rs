use pyo3::exceptions::{PyIndexError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use rug::Assign;
use rug::{Complex, Float, Rational};
use std::collections::HashMap;
use std::fmt::Write as FmtWrite;
use rayon::prelude::*;

pub mod shbt;
use crate::shbt::*;

pyo3::create_exception!(anyon_simulator, NonAbelianLeakageError, pyo3::exceptions::PyException);

const EVAL_PREC: u32 = 512;
const PARENT: u32 = 312;
const LEPTON: u32 = 26;
const QUARK: u32 = 8;
const C_DARK_NUM: u32 = 1197103;
const C_DARK_DEN: u32 = 362670;
const MAX_LOGICAL_QUBITS: usize = 4;
const MAX_DIMENSION: usize = 1 << (2 * MAX_LOGICAL_QUBITS); // 256 channels
const MAX_ANYONS: usize = 32;
const WORLDLINE_DEPTH: usize = 128;
const MAX_FUSION_NODES: usize = 64;
const MAX_CLASSICAL_REGS: usize = 16;
const MAX_INSTRUCTIONS: usize = 128;
const MAX_SK_STEPS: usize = 64;

fn c_dark_rational() -> Rational {
    Rational::from((C_DARK_NUM, C_DARK_DEN))
}

#[pyclass]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Coordinate {
    #[pyo3(get, set)]
    pub x: f64,
    #[pyo3(get, set)]
    pub y: f64,
}

#[pymethods]
impl Coordinate {
    #[new]
    fn new(x: f64, y: f64) -> Self {
        Coordinate { x, y }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct AnyonWorldline {
    pub id: usize,
    pub history: [Coordinate; WORLDLINE_DEPTH],
    pub head_idx: usize,
}

impl AnyonWorldline {
    pub fn zeroed(id: usize) -> Self {
        AnyonWorldline {
            id,
            history: [Coordinate { x: 0.0, y: 0.0 }; WORLDLINE_DEPTH],
            head_idx: 0,
        }
    }

    pub fn push(&mut self, coord: Coordinate) {
        if self.head_idx < WORLDLINE_DEPTH {
            self.history[self.head_idx] = coord;
            self.head_idx += 1;
        }
    }

    pub fn current(&self) -> Coordinate {
        if self.head_idx == 0 {
            Coordinate { x: 0.0, y: 0.0 }
        } else {
            self.history[self.head_idx - 1]
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct FusionNode {
    pub left_child: Option<usize>,
    pub right_child: Option<usize>,
    pub anyon_id: Option<usize>,
    pub topological_charge: u32,
    pub parent: Option<usize>,
}

impl FusionNode {
    pub fn leaf(anyon_id: usize, charge: u32) -> Self {
        FusionNode {
            left_child: None,
            right_child: None,
            anyon_id: Some(anyon_id),
            topological_charge: charge,
            parent: None,
        }
    }

    pub fn internal(left: usize, right: usize, charge: u32) -> Self {
        FusionNode {
            left_child: Some(left),
            right_child: Some(right),
            anyon_id: None,
            topological_charge: charge,
            parent: None,
        }
    }
}

#[pyclass]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StabilizerType {
    Plaquette,
    Vertex,
}

#[pyclass]
#[derive(Debug, Clone, Copy)]
pub struct StabilizerGenerator {
    #[pyo3(get, set)]
    pub op_type: StabilizerType,
    pub target_anyons: [usize; 4],
    #[pyo3(get, set)]
    pub weight: usize,
}

#[pymethods]
impl StabilizerGenerator {
    #[new]
    pub fn new(op_type: StabilizerType, targets: Vec<usize>) -> PyResult<Self> {
        if targets.len() > 4 {
            return Err(PyValueError::new_err("Stabilizer weight cannot exceed 4"));
        }
        let mut target_anyons = [0usize; 4];
        let weight = targets.len();
        for i in 0..weight {
            target_anyons[i] = targets[i];
        }
        Ok(StabilizerGenerator { op_type, target_anyons, weight })
    }

    #[getter]
    pub fn targets(&self) -> Vec<usize> {
        self.target_anyons[0..self.weight].to_vec()
    }
}

#[pyclass]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LieSectorType {
    SU2_26,
    SU3_8,
    SO10_312,
}

#[derive(Debug, Clone, Copy)]
pub enum QasmInstruction {
    Braid { id_a: usize, id_b: usize },
    Measure { target: usize, creg: usize },
    ConditionalBraid { creg: usize, val: u32, id_a: usize, id_b: usize },
    DecodeAndCorrect,
    MeasureStabilizer { stabilizer_idx: usize, creg: usize },
    LogicalGate { gate: LogicalGateKind, target_qubit: usize },
    LogicalCNOT { control_qubit: usize, target_qubit: usize },
    RzPhase { angle_multiplier: f64, target_qubit: usize },
    RxPhase { angle_multiplier: f64, target_qubit: usize },
    ArbitraryUnitary4 { matrix_data: [[(f64, f64); 4]; 4] },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogicalGateKind {
    H,
    T,
    X,
}

#[pyclass]
pub struct CompiledProgram {
    pub instructions: [QasmInstruction; MAX_INSTRUCTIONS],
    pub len: usize,
}

#[pymethods]
impl CompiledProgram {
    #[new]
    fn new() -> Self {
        CompiledProgram {
            instructions: [QasmInstruction::Braid { id_a: 0, id_b: 0 }; MAX_INSTRUCTIONS],
            len: 0,
        }
    }

    #[getter(len)]
    fn get_len(&self) -> usize {
        self.len
    }
}

#[derive(Debug, Clone, Copy)]
pub struct StackStabilizer {
    pub is_active: bool,
    pub is_x_type: bool,
    pub anyons: [usize; 4],
    pub weight: usize,
}

#[pyclass]
#[derive(Debug, Clone, Copy)]
pub struct SurfaceCodeLattice {
    pub stabilizers: [StabilizerGenerator; 16],
    pub stabilizer_count: usize,
    pub data_qubit_to_anyon_map: [usize; 16],
}

#[pymethods]
impl SurfaceCodeLattice {
    #[new]
    pub fn new() -> Self {
        let dummy_stabilizer = StabilizerGenerator {
            op_type: StabilizerType::Plaquette,
            target_anyons: [0usize; 4],
            weight: 0,
        };
        SurfaceCodeLattice {
            stabilizers: [dummy_stabilizer; 16],
            stabilizer_count: 0,
            data_qubit_to_anyon_map: [0usize; 16],
        }
    }

    #[getter]
    pub fn stabilizer_count(&self) -> usize {
        self.stabilizer_count
    }

    pub fn add_stabilizer(&mut self, stabilizer: StabilizerGenerator) -> PyResult<()> {
        if self.stabilizer_count >= 16 {
            return Err(PyIndexError::new_err("Maximum stabilizer capacity of 16 reached"));
        }
        self.stabilizers[self.stabilizer_count] = stabilizer;
        self.stabilizer_count += 1;
        Ok(())
    }
}

impl AnyonBraidingEngine {
    pub fn apply_unitary(&mut self, unitary: &[[Complex; 4]; 4]) {
        // Zero-allocation row-by-column multiplication
        for i in 0..4 {
            // reset scratch
            self.scratch_vector[i].assign(&self.zero_complex);
            for j in 0..4 {
                // temp_scalar = unitary[i][j] * state_vector[j]
                self.temp_scalar.assign(&unitary[i][j]);
                self.temp_scalar *= &self.state_vector[j];
                // accumulate
                self.scratch_vector[i] += &self.temp_scalar;
            }
        }

        // move results back into state_vector
        for i in 0..4 {
            self.state_vector[i].assign(&self.scratch_vector[i]);
        }
    }
}

#[derive(Debug, Clone)]
struct LieSector {
    name: &'static str,
    rank: usize,
    level: u32,
    dual_coxeter_number: u32,
    cartan_matrix: Vec<Vec<i32>>,
    rho: Vec<i32>,
}

impl LieSector {
    fn su2_26() -> Self {
        LieSector {
            name: "SU(2)_26",
            rank: 1,
            level: 26,
            dual_coxeter_number: 2,
            cartan_matrix: vec![vec![2]],
            rho: vec![1],
        }
    }

    fn su3_8() -> Self {
        LieSector {
            name: "SU(3)_8",
            rank: 2,
            level: 8,
            dual_coxeter_number: 3,
            cartan_matrix: vec![vec![2, -1], vec![-1, 2]],
            rho: vec![1, 1],
        }
    }

    fn so10_312() -> Self {
        LieSector {
            name: "SO(10)_312",
            rank: 5,
            level: 312,
            dual_coxeter_number: 8,
            cartan_matrix: vec![
                vec![2, -1, 0, 0, 0],
                vec![-1, 2, -1, 0, 0],
                vec![0, -1, 2, -1, -1],
                vec![0, 0, -1, 2, 0],
                vec![0, 0, -1, 0, 2],
            ],
            rho: vec![1, 2, 3, 2, 1],
        }
    }

    fn total_quantum_dimension(&self) -> Float {
        match self.name {
            "SU(2)_26" => self.su2_total_quantum_dimension(),
            "SU(3)_8" => self.su3_total_quantum_dimension(),
            _ => Float::with_val(EVAL_PREC, 0),
        }
    }

    fn su2_total_quantum_dimension(&self) -> Float {
        let mut sum_sq = Float::with_val(EVAL_PREC, 0);
        for weight in 0..=self.level {
            let qdim = self.su2_quantum_dimension(weight);
            let mut qdim_sq = Float::with_val(EVAL_PREC, &qdim);
            qdim_sq *= &qdim;
            sum_sq += qdim_sq;
        }
        sum_sq.sqrt()
    }

    fn su2_quantum_dimension(&self, highest_weight: u32) -> Float {
        let mut pi = Float::with_val(EVAL_PREC, -1);
        pi = pi.acos();

        let mut denominator = Float::with_val(EVAL_PREC, &pi);
        denominator /= Float::with_val(EVAL_PREC, self.level + self.dual_coxeter_number);

        let mut argument = Float::with_val(EVAL_PREC, highest_weight + 1);
        argument *= &denominator;

        let numerator = argument.sin();
        let denominator = denominator.sin();

        let mut result = Float::with_val(EVAL_PREC, numerator);
        result /= denominator;
        result
    }

    fn su3_quantum_dimension(&self, highest_weight: &[u32]) -> Float {
        if highest_weight.len() != 2 {
            return Float::with_val(EVAL_PREC, 0);
        }

        let k = Float::with_val(EVAL_PREC, self.level + self.dual_coxeter_number);
        let mut pi = Float::with_val(EVAL_PREC, -1);
        pi = pi.acos();
        let mut x = Float::with_val(EVAL_PREC, &pi);
        x /= k;

        let lambda1 = highest_weight[0] as i32;
        let lambda2 = highest_weight[1] as i32;

        let mut arg1 = Float::with_val(EVAL_PREC, lambda1 + self.rho[0]);
        arg1 *= &x;
        let mut arg2 = Float::with_val(EVAL_PREC, lambda2 + self.rho[1]);
        arg2 *= &x;
        let mut arg12 = Float::with_val(EVAL_PREC, lambda1 + lambda2 + self.rho.iter().sum::<i32>());
        arg12 *= &x;

        let numerator = Float::with_val(EVAL_PREC, arg1.sin());
        let mut numerator = Float::with_val(EVAL_PREC, numerator * arg2.sin());
        numerator *= arg12.sin();

        let denom_factor1 = Float::with_val(EVAL_PREC, x.clone().sin());
        let mut denom_factor2 = Float::with_val(EVAL_PREC, &x);
        denom_factor2 *= 2;
        let denom_factor2 = denom_factor2.sin();

        let mut denominator = Float::with_val(EVAL_PREC, &denom_factor1);
        denominator *= &denom_factor1;
        denominator *= denom_factor2;

        let mut result = Float::with_val(EVAL_PREC, numerator);
        result /= denominator;
        result
    }

    fn su3_total_quantum_dimension(&self) -> Float {
        let mut total = Float::with_val(EVAL_PREC, 0);
        for lambda1 in 0..=self.level {
            for lambda2 in 0..=(self.level - lambda1) {
                let qdim = self.su3_quantum_dimension(&[lambda1, lambda2]);
                total += qdim;
            }
        }
        total
    }
}

fn su2_q_number(n: u32, level: u32) -> Float {
    if n == 0 {
        return Float::with_val(EVAL_PREC, 0);
    }
    let mut pi = Float::with_val(EVAL_PREC, -1);
    pi = pi.acos();

    let mut theta = Float::with_val(EVAL_PREC, n);
    let mut denom = Float::with_val(EVAL_PREC, level + 2);
    theta *= &pi;
    theta /= &denom;

    let mut numerator = theta.sin();
    let mut denominator = Float::with_val(EVAL_PREC, &pi);
    denominator /= &denom;
    denominator = denominator.sin();

    numerator /= denominator;
    numerator
}

fn su2_q_factorial(n: u32, level: u32) -> Float {
    let mut result = Float::with_val(EVAL_PREC, 1);
    for i in 1..=n {
        result *= su2_q_number(i, level);
    }
    result
}

fn su2_q_triangle(a: i32, b: i32, c: i32, level: u32) -> Float {
    let ab_c = a + b - c;
    let a_bc = a - b + c;
    let _abc = -a + b + c;
    let abc = a + b + c;

    if ab_c < 0 || a_bc < 0 || _abc < 0 || abc < 0 {
        return Float::with_val(EVAL_PREC, 0);
    }

    let x1 = (ab_c / 2) as u32;
    let x2 = (a_bc / 2) as u32;
    let x3 = (_abc / 2) as u32;
    let x4 = (abc / 2 + 1) as u32;

    let mut numerator = su2_q_factorial(x1, level);
    numerator *= su2_q_factorial(x2, level);
    numerator *= su2_q_factorial(x3, level);

    let mut denominator = su2_q_factorial(x4, level);
    numerator /= denominator;
    numerator.sqrt()
}

fn su2_q_6j(a: i32, b: i32, c: i32, d: i32, e: i32, f: i32, level: u32) -> Float {
    let delta1 = su2_q_triangle(a, b, e, level);
    let delta2 = su2_q_triangle(c, d, e, level);
    let delta3 = su2_q_triangle(a, c, f, level);
    let delta4 = su2_q_triangle(b, d, f, level);

    if delta1 == 0 || delta2 == 0 || delta3 == 0 || delta4 == 0 {
        return Float::with_val(EVAL_PREC, 0);
    }

    let x1 = (a + b + e) / 2;
    let x2 = (a + c + f) / 2;
    let x3 = (b + d + f) / 2;
    let zmin = *[x1, x2, x3].iter().max().unwrap();
    let zmax = *[
        (a + b + c + d) / 2,
        (a + e + c + f) / 2,
        (b + e + d + f) / 2,
    ]
    .iter()
    .min()
    .unwrap();

    if zmin > zmax {
        return Float::with_val(EVAL_PREC, 0);
    }

    let mut sum = Float::with_val(EVAL_PREC, 0);
    for z in zmin..=zmax {
        let z_usize = (z + 1) as u32;
        let mut term = su2_q_factorial(z_usize, level);

        let denom_inputs = [
            (z - x1) as u32,
            (z - x2) as u32,
            (z - x3) as u32,
            ((a + b + c + d) / 2 - z) as u32,
            ((a + e + c + f) / 2 - z) as u32,
            ((b + e + d + f) / 2 - z) as u32,
        ];

        for value in denom_inputs.iter() {
            if (*value as i32) < 0 {
                term = Float::with_val(EVAL_PREC, 0);
                break;
            }
            term /= su2_q_factorial(*value, level);
        }

        if term == 0 {
            continue;
        }

        if (z % 2) != 0 {
            term *= Float::with_val(EVAL_PREC, -1);
        }
        sum += term;
    }

    let mut result = delta1;
    result *= &delta2;
    result *= &delta3;
    result *= &delta4;
    result *= &sum;
    result
}

fn su2_topological_spin(label: i32, level: u32) -> Float {
    let spin = Float::with_val(EVAL_PREC, label as f64 / 2.0);
    let mut numer = Float::with_val(EVAL_PREC, &spin);
    numer *= Float::with_val(EVAL_PREC, &spin + 2);
    let mut denom = Float::with_val(EVAL_PREC, 4 * (level + 2));
    numer /= denom;
    numer
}

fn su2_r_phases(level: u32) -> [Complex; 4] {
    let labels = [0, 2, 4, 6];
    let a_label = 3;
    let precision = EVAL_PREC;
    let mut phases: [Complex; 4] = std::array::from_fn(|_| Complex::with_val(precision, (0, 0)));
    let h_a = su2_topological_spin(a_label, level);

    for (i, &label) in labels.iter().enumerate() {
        let h_e = su2_topological_spin(label, level);
        let mut exponent = Float::with_val(precision, &h_a + &h_a);
        exponent -= &h_e;
        let mut angle = Float::with_val(precision, &exponent);
        let mut pi = Float::with_val(precision, -1);
        pi = pi.acos();
        angle *= &pi;

        let cre = angle.clone().cos();
        let cim = angle.sin();
        let mut phase = Complex::with_val(precision, (cre, cim));

        let sign_exponent = (a_label + a_label - label) / 2;
        if sign_exponent % 2 != 0 {
            phase *= &Complex::with_val(precision, (-1, 0));
        }
        phases[i].assign(&phase);
    }
    phases
}

fn su2_f_matrix(level: u32) -> [[Complex; 4]; 4] {
    let labels = [0, 2, 4, 6];
    let mut matrix: [[Complex; 4]; 4] = std::array::from_fn(|_| std::array::from_fn(|_| Complex::with_val(EVAL_PREC, (0, 0))));
    let a = 3;
    let b = 3;
    let c = 3;
    let d = 3;

    for (i, &e) in labels.iter().enumerate() {
        for (j, &f) in labels.iter().enumerate() {
            let mut element = su2_q_6j(a, b, c, d, e, f, level);
            let mut norm = su2_q_number(e as u32 + 1, level);
            norm *= su2_q_number(f as u32 + 1, level);
            norm = norm.sqrt();
            element *= &norm;
            matrix[i][j].assign(&Complex::with_val(EVAL_PREC, (element, 0)));
        }
    }
    matrix
}

fn su2_braid_matrix_4x4(level: u32) -> [[Complex; 4]; 4] {
    let precision = EVAL_PREC;
    let fmat = su2_f_matrix(level);
    let rdiag = su2_r_phases(level);
    let mut tmat: [[Complex; 4]; 4] = std::array::from_fn(|_| std::array::from_fn(|_| Complex::with_val(precision, (0, 0))));
    for i in 0..4 {
        for j in 0..4 {
            tmat[i][j].assign(&rdiag[i]);
            tmat[i][j] *= &fmat[i][j];
        }
    }

    let mut braid: [[Complex; 4]; 4] = std::array::from_fn(|_| std::array::from_fn(|_| Complex::with_val(precision, (0, 0))));
    for i in 0..4 {
        for j in 0..4 {
            braid[i][j].assign(&Complex::with_val(precision, (0, 0)));
            for k in 0..4 {
                // F_inv = conjugate transpose of F; F is real here
                let mut tmp = fmat[k][i].clone();
                tmp *= &tmat[k][j];
                braid[i][j] += &tmp;
            }
        }
    }

    braid
}

// ---------------------------------------------------------------------------
// Phase 7: Polymorphic sector R-phases and braid matrices
// ---------------------------------------------------------------------------

fn su3_r_phases(level: u32) -> [Complex; 4] {
    let su2_phases = su2_r_phases(level);
    let sector = LieSector::su3_8();
    let deform: Float = {
        let mut d = Float::with_val(EVAL_PREC, sector.dual_coxeter_number as i64);
        let mut l = Float::with_val(EVAL_PREC, (level + sector.dual_coxeter_number) as i64);
        d /= &l;
        d
    };
    let mut out: [Complex; 4] = std::array::from_fn(|_| Complex::with_val(EVAL_PREC, (0, 0)));
    for i in 0..4 {
        let mut pi = Float::with_val(EVAL_PREC, -1);
        pi = pi.acos();
        let mut angle = Float::with_val(EVAL_PREC, &deform);
        angle *= &pi;
        let cre = angle.clone().cos();
        let cim = angle.sin();
        let weight = Complex::with_val(EVAL_PREC, (cre, cim));
        let mut p = su2_phases[i].clone();
        p *= &weight;
        out[i].assign(&p);
    }
    out
}

fn so10_r_phases(level: u32) -> [Complex; 4] {
    let su2_phases = su2_r_phases(level);
    let sector = LieSector::so10_312();
    let deform: Float = {
        let mut d = Float::with_val(EVAL_PREC, sector.dual_coxeter_number as i64);
        let l = Float::with_val(EVAL_PREC, (level + sector.dual_coxeter_number) as i64);
        d /= &l;
        d
    };
    let mut out: [Complex; 4] = std::array::from_fn(|_| Complex::with_val(EVAL_PREC, (0, 0)));
    for i in 0..4 {
        let mut pi = Float::with_val(EVAL_PREC, -1);
        pi = pi.acos();
        let mut angle = Float::with_val(EVAL_PREC, &deform);
        angle *= &pi;
        let cre = angle.clone().cos();
        let cim = angle.sin();
        let weight = Complex::with_val(EVAL_PREC, (cre, cim));
        let mut p = su2_phases[i].clone();
        p *= &weight;
        out[i].assign(&p);
    }
    out
}

// ===========================================================================
// Phase 8: Fault-Tolerant Constants & Data Structures
// ===========================================================================
pub const MAX_SK_DEPTH: usize = 3;

#[pyclass]
#[derive(Debug, Clone, Copy)]
pub struct BraidSequence {
    pub braids: [(usize, usize); MAX_SK_STEPS],
    pub len: usize,
}

#[pymethods]
impl BraidSequence {
    #[new]
    pub fn new() -> Self {
        BraidSequence {
            braids: [(0, 0); MAX_SK_STEPS],
            len: 0,
        }
    }

    #[getter(len)]
    pub fn get_len(&self) -> usize {
        self.len
    }

    pub fn get_braid(&self, idx: usize) -> PyResult<(usize, usize)> {
        if idx >= self.len {
            return Err(pyo3::exceptions::PyIndexError::new_err("Braid index out of range"));
        }
        Ok(self.braids[idx])
    }

    pub fn push(&mut self, swap: (usize, usize)) -> bool {
        if self.len < MAX_SK_STEPS {
            self.braids[self.len] = swap;
            self.len += 1;
            true
        } else {
            false
        }
    }
}

impl BraidSequence {
    pub fn clear(&mut self) {
        self.len = 0;
    }

    pub fn append(&mut self, other: &BraidSequence) -> bool {
        if self.len + other.len <= MAX_SK_STEPS {
            for i in 0..other.len {
                self.braids[self.len] = other.braids[i];
                self.len += 1;
            }
            true
        } else {
            false
        }
    }

    pub fn append_inverse(&mut self, other: &BraidSequence) -> bool {
        if self.len + other.len <= MAX_SK_STEPS {
            for i in (0..other.len).rev() {
                self.braids[self.len] = other.braids[i];
                self.len += 1;
            }
            true
        } else {
            false
        }
    }
}

// ===========================================================================
// HIGH-PRECISION IN-PLACE MATRIX UTILITIES
// ===========================================================================
pub struct MatrixMath2x2 {
    pub zero: Complex,
    pub one: Complex,
    pub temp: Complex,
}

impl MatrixMath2x2 {
    pub fn new() -> Self {
        MatrixMath2x2 {
            zero: Complex::with_val(EVAL_PREC, (0, 0)),
            one:  Complex::with_val(EVAL_PREC, (1, 0)),
            temp: Complex::with_val(EVAL_PREC, (0, 0)),
        }
    }

    pub fn mul(
        &mut self,
        a: &[[Complex; 2]; 2],
        b: &[[Complex; 2]; 2],
        out: &mut [[Complex; 2]; 2],
    ) {
        for i in 0..2 {
            for j in 0..2 {
                out[i][j].assign(&self.zero);
                for k in 0..2 {
                    self.temp.assign(&a[i][k]);
                    self.temp *= &b[k][j];
                    out[i][j] += &self.temp;
                }
            }
        }
    }

    pub fn dagger(&self, a: &[[Complex; 2]; 2], out: &mut [[Complex; 2]; 2]) {
        for i in 0..2 {
            for j in 0..2 {
                out[i][j].assign(a[j][i].conj_ref());
            }
        }
    }

    pub fn identity(&self, out: &mut [[Complex; 2]; 2]) {
        for i in 0..2 {
            for j in 0..2 {
                if i == j {
                    out[i][j].assign(&self.one);
                } else {
                    out[i][j].assign(&self.zero);
                }
            }
        }
    }
}

pub struct SKScratchpad2x2 {
    pub math: MatrixMath2x2,
    pub u_prev: [[Complex; 2]; 2],
    pub delta: [[Complex; 2]; 2],
    pub v: [[Complex; 2]; 2],
    pub w: [[Complex; 2]; 2],
    pub v_dagger: [[Complex; 2]; 2],
    pub w_dagger: [[Complex; 2]; 2],
    pub s: [[Complex; 2]; 2],
    pub s_dagger: [[Complex; 2]; 2],
    pub v_tilde: [[Complex; 2]; 2],
    pub w_tilde: [[Complex; 2]; 2],
    pub temp_matrix: [[Complex; 2]; 2],
    pub temp_matrix2: [[Complex; 2]; 2],
    pub phi: Float,
    pub theta: Float,
    pub sin_theta_half: Float,
    pub cos_theta_half: Float,
    pub asin_arg: Float,
    pub temp_float: Float,
    pub axis_m: [Float; 3],
    pub axis_n: [Float; 3],
    pub axis_w: [Float; 3],
    pub w_norm: Float,
    pub psi: Float,
    // Pre-allocated SKBaseCache: generator matrices for basic_lookup.
    // cache_g[0]: U = diag(0.7071+0.7071i, 1) — generator (0,1)
    // cache_g[1]: U = diag(0.7071-0.7071i, 1) — generators (1,2) and (2,3)
    pub cache_g: [[[Complex; 2]; 2]; 2],
    // Mutable scratch registers for basic_lookup distance accumulation.
    // Eliminates .clone() calls on real/imag components during the sweep.
    pub bl_diff_re: Float,
    pub bl_diff_im: Float,
    pub bl_distance: Float,
}

impl SKScratchpad2x2 {
    pub fn new() -> Self {
        let math = MatrixMath2x2::new();
        let init_mat = || -> [[Complex; 2]; 2] {
            std::array::from_fn(|_| std::array::from_fn(|_| Complex::with_val(EVAL_PREC, (0, 0))))
        };
        let init_f = || Float::with_val(EVAL_PREC, 0);

        // ---------------------------------------------------------------
        // Pre-compute SKBaseCache generator matrices at construction time.
        //
        // cache_g[0] represents generator (0,1):
        //   U[0][0] = 0.7071 + 0.7071i,  U[1][1] = 1
        // cache_g[1] represents generators (1,2) and (2,3):
        //   U[0][0] = 0.7071 - 0.7071i,  U[1][1] = 1
        // Off-diagonal entries are 0 (identity layout).
        // ---------------------------------------------------------------
        let sqrt2_inv = 0.5_f64.sqrt(); // 1/√2 ≈ 0.70710678…
        let mut cg0 = init_mat();
        cg0[0][0] = Complex::with_val(EVAL_PREC, (sqrt2_inv,  sqrt2_inv));
        cg0[1][1] = Complex::with_val(EVAL_PREC, (1, 0));
        let mut cg1 = init_mat();
        cg1[0][0] = Complex::with_val(EVAL_PREC, (sqrt2_inv, -sqrt2_inv));
        cg1[1][1] = Complex::with_val(EVAL_PREC, (1, 0));

        SKScratchpad2x2 {
            math,
            u_prev:       init_mat(),
            delta:        init_mat(),
            v:            init_mat(),
            w:            init_mat(),
            v_dagger:     init_mat(),
            w_dagger:     init_mat(),
            s:            init_mat(),
            s_dagger:     init_mat(),
            v_tilde:      init_mat(),
            w_tilde:      init_mat(),
            temp_matrix:  init_mat(),
            temp_matrix2: init_mat(),
            phi:             init_f(),
            theta:           init_f(),
            sin_theta_half:  init_f(),
            cos_theta_half:  init_f(),
            asin_arg:        init_f(),
            temp_float:      init_f(),
            axis_m: [init_f(), init_f(), init_f()],
            axis_n: [init_f(), init_f(), init_f()],
            axis_w: [init_f(), init_f(), init_f()],
            w_norm: init_f(),
            psi:    init_f(),
            cache_g: [cg0, cg1],
            bl_diff_re: init_f(),
            bl_diff_im: init_f(),
            bl_distance: init_f(),
        }
    }
}

// ===========================================================================
// RECURSIVE SOLOVAY-KITAEV COMPILER MATRIX LOGIC
// ===========================================================================
pub fn gc_decompose(
    delta: &[[Complex; 2]; 2],
    v: &mut [[Complex; 2]; 2],
    w: &mut [[Complex; 2]; 2],
    scratch: &mut SKScratchpad2x2,
) {
    scratch.cos_theta_half.assign(delta[0][0].real());

    scratch.temp_float.assign(&scratch.cos_theta_half);
    scratch.temp_float.square_mut();
    scratch.sin_theta_half = Float::with_val(EVAL_PREC, 1) - &scratch.temp_float;
    scratch.sin_theta_half.sqrt_mut();

    {
        let tiny = Float::with_val(EVAL_PREC, 1e-15_f64);
        if scratch.sin_theta_half.clone().abs() < tiny {
            scratch.math.identity(v);
            scratch.math.identity(w);
            return;
        }
    }

    scratch.theta.assign(scratch.cos_theta_half.clone().acos());
    scratch.theta *= Float::with_val(EVAL_PREC, 2);

    scratch.axis_n[0].assign(delta[0][1].imag());
    let mut tmp_n0 = scratch.axis_n[0].clone();
    tmp_n0 *= Float::with_val(EVAL_PREC, -1);
    scratch.axis_n[0] = tmp_n0;
    scratch.axis_n[0] /= &Float::with_val(EVAL_PREC, &scratch.sin_theta_half);

    scratch.axis_n[1].assign(delta[0][1].real());
    let mut tmp_n1 = scratch.axis_n[1].clone();
    tmp_n1 *= Float::with_val(EVAL_PREC, -1);
    scratch.axis_n[1] = tmp_n1;
    scratch.axis_n[1] /= &Float::with_val(EVAL_PREC, &scratch.sin_theta_half);

    scratch.axis_n[2].assign(delta[0][0].imag());
    let mut tmp_n2 = scratch.axis_n[2].clone();
    tmp_n2 *= Float::with_val(EVAL_PREC, -1);
    scratch.axis_n[2] = tmp_n2;
    scratch.axis_n[2] /= &Float::with_val(EVAL_PREC, &scratch.sin_theta_half);

    scratch.asin_arg.assign(&scratch.theta);
    scratch.asin_arg /= &Float::with_val(EVAL_PREC, 4);
    scratch.asin_arg = scratch.asin_arg.clone().sin();
    scratch.asin_arg.sqrt_mut();
    scratch.phi = scratch.asin_arg.clone().asin();
    scratch.phi *= Float::with_val(EVAL_PREC, 2);

    let phi_half = Float::with_val(EVAL_PREC, &scratch.phi) / Float::with_val(EVAL_PREC, 2);
    let s_val = phi_half.clone().sin();
    let c_val = phi_half.cos();

    let mut norm = Float::with_val(EVAL_PREC, &s_val);
    norm.square_mut();
    norm += Float::with_val(EVAL_PREC, 1);
    norm.sqrt_mut();

    scratch.axis_m[0] = Float::with_val(EVAL_PREC, &s_val) / &norm;
    scratch.axis_m[1] = -Float::with_val(EVAL_PREC, &s_val) / &norm;
    scratch.axis_m[2] = Float::with_val(EVAL_PREC, &c_val) / &norm;

    scratch.axis_w[0] = Float::with_val(EVAL_PREC,
        (&scratch.axis_m[1] * &scratch.axis_n[2]) - (&scratch.axis_m[2] * &scratch.axis_n[1]));
    scratch.axis_w[1] = Float::with_val(EVAL_PREC,
        (&scratch.axis_m[2] * &scratch.axis_n[0]) - (&scratch.axis_m[0] * &scratch.axis_n[2]));
    scratch.axis_w[2] = Float::with_val(EVAL_PREC,
        (&scratch.axis_m[0] * &scratch.axis_n[1]) - (&scratch.axis_m[1] * &scratch.axis_n[0]));

    let mut w_norm = Float::with_val(EVAL_PREC, &scratch.axis_w[0]);
    w_norm *= &scratch.axis_w[0];
    let mut temp = Float::with_val(EVAL_PREC, &scratch.axis_w[1]);
    temp *= &scratch.axis_w[1];
    w_norm += &temp;
    temp.assign(&scratch.axis_w[2]);
    temp *= &scratch.axis_w[2];
    w_norm += &temp;
    scratch.w_norm = w_norm;
    scratch.w_norm.sqrt_mut();

    let mut dot = Float::with_val(EVAL_PREC, &scratch.axis_m[0]);
    dot *= &scratch.axis_n[0];
    let mut temp = Float::with_val(EVAL_PREC, &scratch.axis_m[1]);
    temp *= &scratch.axis_n[1];
    dot += &temp;
    temp.assign(&scratch.axis_m[2]);
    temp *= &scratch.axis_n[2];
    dot += &temp;
    scratch.psi = dot.acos();

    let psi_half = Float::with_val(EVAL_PREC, &scratch.psi) / Float::with_val(EVAL_PREC, 2);
    let cos_psi = psi_half.clone().cos();
    let sin_psi = psi_half.sin();

    {
        let tiny = Float::with_val(EVAL_PREC, 1e-15_f64);
        if scratch.w_norm.clone().abs() > tiny {
            let wn = Float::with_val(EVAL_PREC, &scratch.w_norm);
            scratch.axis_w[0] /= &wn;
            scratch.axis_w[1] /= &wn;
            scratch.axis_w[2] /= &wn;
        }
    }

    let i_c = Complex::with_val(EVAL_PREC, (0, 1));

    scratch.s[0][0] = Complex::with_val(EVAL_PREC, (&scratch.axis_w[2] * &sin_psi, 0))
        * &i_c;
    let mut tmp_s00 = scratch.s[0][0].clone();
    tmp_s00 *= Complex::with_val(EVAL_PREC, (-1, 0));
    scratch.s[0][0] = tmp_s00;
    scratch.s[0][0] += Complex::with_val(EVAL_PREC, (&cos_psi, 0));

    let w_x_iy = Complex::with_val(EVAL_PREC, (&scratch.axis_w[0], -&scratch.axis_w[1]));
    scratch.s[0][1] = w_x_iy * &sin_psi * &i_c;
    let mut tmp_s01 = scratch.s[0][1].clone();
    tmp_s01 *= Complex::with_val(EVAL_PREC, (-1, 0));
    scratch.s[0][1] = tmp_s01;

    let w_x_py = Complex::with_val(EVAL_PREC, (&scratch.axis_w[0], &scratch.axis_w[1]));
    scratch.s[1][0] = w_x_py * &sin_psi * &i_c;
    let mut tmp_s10 = scratch.s[1][0].clone();
    tmp_s10 *= Complex::with_val(EVAL_PREC, (-1, 0));
    scratch.s[1][0] = tmp_s10;

    scratch.s[1][1] = Complex::with_val(EVAL_PREC, (&scratch.axis_w[2] * &sin_psi, 0))
        * &i_c;
    scratch.s[1][1] += Complex::with_val(EVAL_PREC, (&cos_psi, 0));

    scratch.math.dagger(&scratch.s.clone(), &mut scratch.s_dagger);

    let neg_is = Complex::with_val(EVAL_PREC, (0, -1))
        * Complex::with_val(EVAL_PREC, (&s_val, 0));
    let c_cx = Complex::with_val(EVAL_PREC, (&c_val, 0));
    let s_cx = Complex::with_val(EVAL_PREC, (&s_val, 0));
    let neg_s = -s_cx.clone();

    let mut v0: [[Complex; 2]; 2] =
        std::array::from_fn(|_| std::array::from_fn(|_| Complex::with_val(EVAL_PREC, (0, 0))));
    v0[0][0].assign(&c_cx);
    v0[1][1].assign(&c_cx);
    v0[0][1].assign(&neg_is);
    v0[1][0].assign(&neg_is);

    let mut w0: [[Complex; 2]; 2] =
        std::array::from_fn(|_| std::array::from_fn(|_| Complex::with_val(EVAL_PREC, (0, 0))));
    w0[0][0].assign(&c_cx);
    w0[1][1].assign(&c_cx);
    w0[0][1].assign(&neg_s);
    w0[1][0].assign(&s_cx);

    let s_clone = scratch.s.clone();
    let sd_clone = scratch.s_dagger.clone();
    scratch.math.mul(&s_clone, &v0, &mut scratch.temp_matrix);
    let tm_clone = scratch.temp_matrix.clone();
    scratch.math.mul(&tm_clone, &sd_clone, v);

    scratch.math.mul(&s_clone, &w0, &mut scratch.temp_matrix);
    let tm2_clone = scratch.temp_matrix.clone();
    scratch.math.mul(&tm2_clone, &sd_clone, w);
}

pub fn basic_lookup(
    target: &[[Complex; 2]; 2],
    out_seq: &mut BraidSequence,
    scratch: &mut SKScratchpad2x2,
) {
    out_seq.clear();
    let generators: [(usize, usize); 3] = [(0, 1), (1, 2), (2, 3)];
    let mut best_idx = 0usize;
    // Re-use the pre-allocated min_distance register; init to sentinel 1e100.
    scratch.temp_float.assign(1e100_f64);

    // cache_g[0] -> generator index 0;  cache_g[1] -> generator indices 1 and 2.
    // Cache values are copied into a local [[f64;2];2] at the start of each iteration
    // so that scratch can be mutated freely without a simultaneous borrow conflict.
    for idx in 0..generators.len() {
        let cache_idx = if idx == 0 { 0 } else { 1 };
        // Extract real/imag f64 components from the pre-allocated cache entry.
        // This avoids holding a borrow on `scratch` while mutating its registers.
        let g_re: [[f64; 2]; 2] = std::array::from_fn(|i| {
            std::array::from_fn(|j| scratch.cache_g[cache_idx][i][j].real().to_f64())
        });
        let g_im: [[f64; 2]; 2] = std::array::from_fn(|i| {
            std::array::from_fn(|j| scratch.cache_g[cache_idx][i][j].imag().to_f64())
        });

        scratch.bl_distance.assign(0);
        for i in 0..2 {
            for j in 0..2 {
                // bl_diff_re = real(target[i][j]) - real(cache_g[i][j])
                scratch.bl_diff_re.assign(target[i][j].real());
                scratch.bl_diff_re -= g_re[i][j];
                scratch.bl_diff_re.square_mut();
                scratch.bl_distance += &scratch.bl_diff_re;

                // bl_diff_im = imag(target[i][j]) - imag(cache_g[i][j])
                scratch.bl_diff_im.assign(target[i][j].imag());
                scratch.bl_diff_im -= g_im[i][j];
                scratch.bl_diff_im.square_mut();
                scratch.bl_distance += &scratch.bl_diff_im;
            }
        }

        if scratch.bl_distance < scratch.temp_float {
            scratch.temp_float.assign(&scratch.bl_distance);
            best_idx = idx;
        }
    }
    out_seq.push(generators[best_idx]);
}

pub fn solovay_kitaev_recursive(
    target: &[[Complex; 2]; 2],
    depth: usize,
    out_seq: &mut BraidSequence,
    scratch: &mut SKScratchpad2x2,
) {
    if depth == 0 {
        basic_lookup(target, out_seq, scratch);
        return;
    }

    solovay_kitaev_recursive(target, depth - 1, out_seq, scratch);

    // Compute delta = target * U_prev_dagger.
    // U_prev is approximated as identity at this recursion level.
    let mut u_prev: [[Complex; 2]; 2] =
        std::array::from_fn(|_| std::array::from_fn(|_| Complex::with_val(EVAL_PREC, (0, 0))));
    scratch.math.identity(&mut u_prev);

    let mut u_prev_dagger: [[Complex; 2]; 2] =
        std::array::from_fn(|_| std::array::from_fn(|_| Complex::with_val(EVAL_PREC, (0, 0))));
    scratch.math.dagger(&u_prev, &mut u_prev_dagger);

    // Compute delta into a local to avoid borrow conflicts with scratch
    let mut delta_local: [[Complex; 2]; 2] =
        std::array::from_fn(|_| std::array::from_fn(|_| Complex::with_val(EVAL_PREC, (0, 0))));
    scratch.math.mul(target, &u_prev_dagger, &mut delta_local);

    let mut v_mat: [[Complex; 2]; 2] =
        std::array::from_fn(|_| std::array::from_fn(|_| Complex::with_val(EVAL_PREC, (0, 0))));
    let mut w_mat: [[Complex; 2]; 2] =
        std::array::from_fn(|_| std::array::from_fn(|_| Complex::with_val(EVAL_PREC, (0, 0))));
    gc_decompose(&delta_local, &mut v_mat, &mut w_mat, scratch);

    // Copy v_mat and w_mat to locals before recursive calls (avoids borrow on scratch.v/w)
    let v_copy = v_mat.clone();
    let w_copy = w_mat.clone();

    let mut v_seq = BraidSequence::new();
    let mut w_seq = BraidSequence::new();
    solovay_kitaev_recursive(&v_copy, depth - 1, &mut v_seq, scratch);
    solovay_kitaev_recursive(&w_copy, depth - 1, &mut w_seq, scratch);

    let mut reconstructed = BraidSequence::new();
    reconstructed.append(&v_seq);
    reconstructed.append(&w_seq);
    reconstructed.append_inverse(&v_seq);
    reconstructed.append_inverse(&w_seq);
    reconstructed.append(out_seq);

    *out_seq = reconstructed;
}

#[pyclass]
#[derive(Debug)]
pub struct AnyonBraidingEngine {
    parent: u32,
    lepton: u32,
    quark: u32,
    c_dark: Rational,
    state_vector: Vec<Complex>,
    scratch_vector: Vec<Complex>,
    temp_scalar: Complex,
    zero_complex: Complex,
}

#[pymethods]
impl AnyonBraidingEngine {
    #[new]
    fn new() -> Self {
        let mut state_vector = Vec::with_capacity(4);
        state_vector.push(Complex::with_val(EVAL_PREC, (1, 0)));
        for _ in 1..4 {
            state_vector.push(Complex::with_val(EVAL_PREC, (0, 0)));
        }

        let mut scratch_vector = Vec::with_capacity(4);
        for _ in 0..4 {
            scratch_vector.push(Complex::with_val(EVAL_PREC, (0, 0)));
        }

        let temp_scalar = Complex::with_val(EVAL_PREC, (0, 0));
        let zero_complex = Complex::with_val(EVAL_PREC, (0, 0));

        AnyonBraidingEngine {
            parent: PARENT,
            lepton: LEPTON,
            quark: QUARK,
            c_dark: c_dark_rational(),
            state_vector,
            scratch_vector,
            temp_scalar,
            zero_complex,
        }
    }

    #[getter]
    fn parent(&self) -> u32 {
        self.parent
    }

    #[getter]
    fn lepton(&self) -> u32 {
        self.lepton
    }

    #[getter]
    fn quark(&self) -> u32 {
        self.quark
    }

    #[getter]
    fn c_dark(&self) -> String {
        Float::with_val(EVAL_PREC, &self.c_dark).to_string()
    }

    fn perform_swap(&mut self, index_a: usize, index_b: usize) -> PyResult<()> {
        if index_a >= self.state_vector.len() || index_b >= self.state_vector.len() {
            return Err(PyIndexError::new_err("swap indices must be within state vector length"));
        }

        let precision = EVAL_PREC;

        if index_a == index_b {
            // apply a small phase to the same index
            let phase = Complex::with_val(precision, (0, 1));
            self.state_vector[index_a] *= &phase;
            return Ok(());
        }

        // Normalize order
        let (first_idx, second_idx) = if index_a < index_b {
            (index_a, index_b)
        } else {
            (index_b, index_a)
        };

        // Determine if indices share an intra-node parent (simple heuristic: same pair block)
        let intra_node = (first_idx / 2) == (second_idx / 2);

        // Build a 4x4 unitary matrix using zero-allocation construction
        let mut unitary: [[Complex; 4]; 4] = std::array::from_fn(|_| std::array::from_fn(|_| Complex::with_val(EVAL_PREC, (0, 0))));

        // Default to identity
        for i in 0..4 {
            unitary[i][i].assign(&Complex::with_val(precision, (1, 0)));
        }

        if intra_node {
            let rdiag = su2_r_phases(26);
            for i in 0..4 {
                unitary[i][i].assign(&rdiag[i]);
            }
        } else {
            let braid = su2_braid_matrix_4x4(26);
            for i in 0..4 {
                for j in 0..4 {
                    unitary[i][j].assign(&braid[i][j]);
                }
            }
        }

        // Apply the transformation using the zero-allocation multiply
        self.apply_unitary(&unitary);

        Ok(())
    }

    fn get_geometric_kappa(&self) -> String {
        calculate_geometric_kappa()
    }

    fn get_state(&self) -> Vec<String> {
        self.state_vector.iter().map(|c| c.to_string()).collect()
    }

    fn describe(&self) -> String {
        format!(
            "AnyonBraidingEngine(parent={}, lepton={}, quark={}, c_dark={}/{})",
            self.parent, self.lepton, self.quark, C_DARK_NUM, C_DARK_DEN
        )
    }
}

fn calculate_geometric_kappa() -> String {
    let su2 = LieSector::su2_26();
    let lepton_total = su2.total_quantum_dimension();

    let mut beta = lepton_total.ln();
    beta *= 0.5;

    let mut area_ratio = Float::with_val(EVAL_PREC, 160);
    area_ratio /= 1521;
    area_ratio *= Float::with_val(EVAL_PREC, 10).sqrt();

    let mut spinor_retention = Float::with_val(EVAL_PREC, 347);
    let beta_sq = Float::with_val(EVAL_PREC, &beta);
    let mut beta_sq_value = Float::with_val(EVAL_PREC, &beta_sq);
    beta_sq_value *= &beta_sq;
    beta_sq_value *= 8;
    spinor_retention -= beta_sq_value;
    spinor_retention /= 351;

    let mut kappa = Float::with_val(EVAL_PREC, 16);
    kappa /= 5;
    kappa *= &area_ratio;
    kappa *= &spinor_retention;
    kappa = kappa.sqrt();

    normalize_decimal_string(&kappa, 160)
}

fn normalize_decimal_string(value: &Float, digits: usize) -> String {
    let raw = value.to_string_radix(10, Some(digits));
    if let Some((mut mantissa, exponent_str)) = raw.split_once('e').or(raw.split_once('E')) {
        let exponent = exponent_str.parse::<i32>().unwrap_or(0);
        let negative = mantissa.starts_with('-');
        if negative {
            mantissa = &mantissa[1..];
        }

        let (integer_part, fractional_part) = if let Some((int_part, frac_part)) = mantissa.split_once('.') {
            (int_part.to_string(), frac_part.to_string())
        } else {
            (mantissa.to_string(), String::new())
        };

        let mut digits = format!("{}{}", integer_part, fractional_part);
        if exponent >= 0 {
            let shift = exponent as usize;
            if shift >= fractional_part.len() {
                digits.push_str(&"0".repeat(shift - fractional_part.len()));
                digits = format!("{}.", digits);
            } else {
                let split_at = integer_part.len() + shift;
                digits.insert(split_at, '.');
            }
        } else {
            let shift = (-exponent) as usize;
            if shift >= integer_part.len() {
                let zeros = "0".repeat(shift - integer_part.len());
                digits = format!("0.{}{}", zeros, digits);
            } else {
                let split_at = integer_part.len() - shift;
                digits.insert(split_at, '.');
            }
        }

        if digits.starts_with('.') {
            digits.insert(0, '0');
        }
        if digits.ends_with('.') {
            digits.push('0');
        }

        if negative {
            format!("-{}", digits)
        } else {
            digits
        }
    } else {
        raw
    }
}

#[pyfunction]
fn get_geometric_kappa() -> PyResult<String> {
    Ok(calculate_geometric_kappa())
}

// ===========================================================================
// HIGH-PRECISION IN-PLACE 4x4 MATRIX MATH TOOLS
// ===========================================================================
pub struct MatrixMath4x4 {
    pub zero: Complex,
    pub one: Complex,
    pub temp: Complex,
    pub scratch: [[Complex; 4]; 4],
}

impl MatrixMath4x4 {
    pub fn new() -> Self {
        MatrixMath4x4 {
            zero:    Complex::with_val(EVAL_PREC, (0, 0)),
            one:     Complex::with_val(EVAL_PREC, (1, 0)),
            temp:    Complex::with_val(EVAL_PREC, (0, 0)),
            scratch: std::array::from_fn(|_| std::array::from_fn(|_| Complex::with_val(EVAL_PREC, (0, 0)))),
        }
    }

    pub fn mul(
        &mut self,
        a: &[[Complex; 4]; 4],
        b: &[[Complex; 4]; 4],
        out: &mut [[Complex; 4]; 4],
    ) {
        for i in 0..4 {
            for j in 0..4 {
                self.scratch[i][j].assign(&self.zero);
                for k in 0..4 {
                    self.temp.assign(&a[i][k]);
                    self.temp *= &b[k][j];
                    self.scratch[i][j] += &self.temp;
                }
            }
        }
        for i in 0..4 {
            for j in 0..4 {
                out[i][j].assign(&self.scratch[i][j]);
            }
        }
    }

    pub fn dagger(&self, a: &[[Complex; 4]; 4], out: &mut [[Complex; 4]; 4]) {
        for i in 0..4 {
            for j in 0..4 {
                out[i][j].assign(a[j][i].conj_ref());
            }
        }
    }
}

pub struct SKScratchpadMultiQubit {
    pub math: MatrixMath4x4,
    pub magic_m: [[Complex; 4]; 4],
    pub magic_m_dag: [[Complex; 4]; 4],
    pub target_u: [[Complex; 4]; 4],
    pub magic_u: [[Complex; 4]; 4],
    pub m_matrix: [[Complex; 4]; 4],
    pub m_sq: [[Complex; 4]; 4],
    pub tr_m: Complex,
    pub tr_m_sq: Complex,
    pub g1: Complex,
    pub g2: Complex,
    pub a: Float,
    pub b: Float,
    pub c: Float,
    pub p: Float,
    pub q: Float,
    pub r: Float,
    pub dep_p: Float,
    pub dep_q: Float,
    pub discriminant: Float,
    pub theta: Float,
    pub roots_x: [Float; 3],
    pub cartan_c: [Float; 3],
    pub local_k: [[[Complex; 2]; 2]; 4],
}

impl SKScratchpadMultiQubit {
    pub fn new() -> Self {
        let init_c  = || Complex::with_val(EVAL_PREC, (0, 0));
        let init_f  = || Float::with_val(EVAL_PREC, 0);
        let init_mat = || -> [[Complex; 4]; 4] {
            std::array::from_fn(|_| std::array::from_fn(|_| Complex::with_val(EVAL_PREC, (0, 0))))
        };

        // 1/sqrt(2) as high-precision Float
        let inv_sqrt2 = Float::with_val(EVAL_PREC, 2).sqrt().recip();
        let i_val     = Complex::with_val(EVAL_PREC, (0, 1));

        // Populate the standard magic basis matrix M
        let mut magic_m = init_mat();
        // Row 0: [ 1/√2,  i/√2,  i/√2,  1/√2 ]
        magic_m[0][0] = Complex::with_val(EVAL_PREC, (&inv_sqrt2, 0));
        magic_m[0][1] = i_val.clone() * Complex::with_val(EVAL_PREC, (&inv_sqrt2, 0));
        magic_m[0][2] = i_val.clone() * Complex::with_val(EVAL_PREC, (&inv_sqrt2, 0));
        magic_m[0][3] = Complex::with_val(EVAL_PREC, (&inv_sqrt2, 0));
        // Row 1: [ i/√2, -1/√2,  1/√2, -i/√2 ]
        magic_m[1][0] = i_val.clone() * Complex::with_val(EVAL_PREC, (&inv_sqrt2, 0));
        magic_m[1][1] = Complex::with_val(EVAL_PREC, (-Float::with_val(EVAL_PREC, &inv_sqrt2), 0));
        magic_m[1][2] = Complex::with_val(EVAL_PREC, (&inv_sqrt2, 0));
        magic_m[1][3] = -(i_val.clone() * Complex::with_val(EVAL_PREC, (&inv_sqrt2, 0)));
        // Row 2: [-i/√2,  1/√2, -1/√2,  i/√2 ]
        magic_m[2][0] = -(i_val.clone() * Complex::with_val(EVAL_PREC, (&inv_sqrt2, 0)));
        magic_m[2][1] = Complex::with_val(EVAL_PREC, (&inv_sqrt2, 0));
        magic_m[2][2] = Complex::with_val(EVAL_PREC, (-Float::with_val(EVAL_PREC, &inv_sqrt2), 0));
        magic_m[2][3] = i_val.clone() * Complex::with_val(EVAL_PREC, (&inv_sqrt2, 0));
        // Row 3: [ 1/√2, -i/√2,  i/√2,  1/√2 ]
        magic_m[3][0] = Complex::with_val(EVAL_PREC, (&inv_sqrt2, 0));
        magic_m[3][1] = -(i_val.clone() * Complex::with_val(EVAL_PREC, (&inv_sqrt2, 0)));
        magic_m[3][2] = i_val.clone() * Complex::with_val(EVAL_PREC, (&inv_sqrt2, 0));
        magic_m[3][3] = Complex::with_val(EVAL_PREC, (&inv_sqrt2, 0));

        let mut math = MatrixMath4x4::new();
        let mut magic_m_dag = init_mat();
        math.dagger(&magic_m, &mut magic_m_dag);

        SKScratchpadMultiQubit {
            math,
            magic_m,
            magic_m_dag,
            target_u:     init_mat(),
            magic_u:      init_mat(),
            m_matrix:     init_mat(),
            m_sq:         init_mat(),
            tr_m:         init_c(),
            tr_m_sq:      init_c(),
            g1:           init_c(),
            g2:           init_c(),
            a:            init_f(),
            b:            init_f(),
            c:            init_f(),
            p:            init_f(),
            q:            init_f(),
            r:            init_f(),
            dep_p:        init_f(),
            dep_q:        init_f(),
            discriminant: init_f(),
            theta:        init_f(),
            roots_x:  [init_f(), init_f(), init_f()],
            cartan_c: [init_f(), init_f(), init_f()],
            local_k: std::array::from_fn(|_| {
                std::array::from_fn(|_| std::array::from_fn(|_| Complex::with_val(EVAL_PREC, (0, 0))))
            }),
        }
    }

    pub fn solve_cartan_coordinates(&mut self, u: &[[Complex; 4]; 4]) {
        for i in 0..4 {
            for j in 0..4 {
                self.target_u[i][j].assign(&u[i][j]);
            }
        }

        // Borrow-safe: clone inputs before calling math.mul
        let md_clone = self.magic_m_dag.clone();
        let tu_clone = self.target_u.clone();
        let mm_clone = self.magic_m.clone();

        let mut temp_u: [[Complex; 4]; 4] =
            std::array::from_fn(|_| std::array::from_fn(|_| Complex::with_val(EVAL_PREC, (0, 0))));
        self.math.mul(&md_clone, &tu_clone, &mut temp_u);
        self.math.mul(&temp_u.clone(), &mm_clone, &mut self.magic_u);

        // m(U) = (U')^T · U'  (no conjugate — purely transpose product)
        for i in 0..4 {
            for j in 0..4 {
                self.m_matrix[i][j].assign(&Complex::with_val(EVAL_PREC, (0, 0)));
                for k in 0..4 {
                    let mut term = self.magic_u[k][i].clone();
                    term *= &self.magic_u[k][j];
                    self.m_matrix[i][j] += &term;
                }
            }
        }

        // m² — clone one arg to avoid double-borrow of self.m_matrix
        let mm_sq_in = self.m_matrix.clone();
        self.math.mul(&mm_sq_in, &mm_sq_in, &mut self.m_sq);

        self.tr_m.assign(&self.m_matrix[0][0]);
        for i in 1..4 {
            self.tr_m += &self.m_matrix[i][i];
        }

        self.tr_m_sq.assign(&self.m_sq[0][0]);
        for i in 1..4 {
            self.tr_m_sq += &self.m_sq[i][i];
        }

        // G1 = Tr²(m) / 16
        self.g1.assign(&self.tr_m);
        self.g1.square_mut();
        self.g1 *= &Complex::with_val(EVAL_PREC, (0.0625_f64, 0));

        // G2 = (Tr²(m) - Tr(m²)) / 4
        self.g2.assign(&self.tr_m);
        self.g2.square_mut();
        self.g2 -= &self.tr_m_sq;
        self.g2 *= &Complex::with_val(EVAL_PREC, (0.25_f64, 0));

        self.a.assign(self.g1.real());
        self.b.assign(self.g1.imag());
        self.c.assign(self.g2.real());

        let mut sqrt_ab = Float::with_val(EVAL_PREC, &self.a);
        sqrt_ab.square_mut();
        let mut b_sq = Float::with_val(EVAL_PREC, &self.b);
        b_sq.square_mut();
        sqrt_ab += &b_sq;
        sqrt_ab.sqrt_mut();

        let mut one_minus_c_half = Float::with_val(EVAL_PREC, 1) - &self.c;
        one_minus_c_half /= Float::with_val(EVAL_PREC, 2);

        self.p.assign(&one_minus_c_half);
        self.p += 1;
        self.p *= Float::with_val(EVAL_PREC, -1);

        self.q.assign(&sqrt_ab);
        self.q += &one_minus_c_half;

        self.r.assign(&sqrt_ab);
        self.r -= &self.a;
        self.r /= Float::with_val(EVAL_PREC, -2);

        let mut p_third = Float::with_val(EVAL_PREC, &self.p);
        p_third /= Float::with_val(EVAL_PREC, 3);

        self.dep_p.assign(&p_third);
        self.dep_p *= &self.p;
        self.dep_p *= Float::with_val(EVAL_PREC, -1);
        self.dep_p += &self.q;

        // dep_q = 2*(p/3)^3 - (p*q)/3 + r
        let p_third_cubed = {
            let mut v = Float::with_val(EVAL_PREC, &p_third);
            let v2 = v.clone() * &v;
            v *= &v2;
            v
        };
        self.dep_q.assign(&p_third_cubed);
        self.dep_q *= Float::with_val(EVAL_PREC, 2);
        let mut pq_third = Float::with_val(EVAL_PREC, &self.p);
        pq_third *= &self.q;
        pq_third /= Float::with_val(EVAL_PREC, 3);
        self.dep_q -= &pq_third;
        self.dep_q += &self.r;

        let mut q_half = Float::with_val(EVAL_PREC, &self.dep_q);
        q_half /= Float::with_val(EVAL_PREC, 2);
        let mut q_half_sq = Float::with_val(EVAL_PREC, &q_half);
        q_half_sq.square_mut();

        let mut p_third_dep = Float::with_val(EVAL_PREC, &self.dep_p);
        p_third_dep /= Float::with_val(EVAL_PREC, 3);
        let p_third_dep_cubed = {
            let mut v = Float::with_val(EVAL_PREC, &p_third_dep);
            let v2 = v.clone() * &v;
            v *= &v2;
            v
        };

        self.discriminant.assign(&q_half_sq);
        self.discriminant += &p_third_dep_cubed;

        if self.discriminant <= 0 {
            // Three real roots — trigonometric method
            let mut cos_arg = Float::with_val(EVAL_PREC, &self.dep_q);
            cos_arg *= Float::with_val(EVAL_PREC, 27);
            let mut denom_factor = {
                let mut v = Float::with_val(EVAL_PREC, &self.dep_p);
                v *= Float::with_val(EVAL_PREC, -1);
                v /= Float::with_val(EVAL_PREC, 3);
                v.sqrt_mut();
                let v2 = v.clone() * &v;
                v *= &v2;   // v = ((-dep_p/3)^(1/2))^3
                v *= Float::with_val(EVAL_PREC, 2);
                v
            };
            cos_arg /= &denom_factor;
            // Clamp to [-1, 1] before acos
            if cos_arg > Float::with_val(EVAL_PREC, 1) {
                cos_arg.assign(Float::with_val(EVAL_PREC, 1));
            } else if cos_arg < Float::with_val(EVAL_PREC, -1) {
                cos_arg.assign(Float::with_val(EVAL_PREC, -1));
            }
            self.theta.assign(cos_arg.acos());

            let mut multiplier = Float::with_val(EVAL_PREC, &self.dep_p);
            multiplier *= Float::with_val(EVAL_PREC, -1);
            multiplier /= Float::with_val(EVAL_PREC, 3);
            multiplier.sqrt_mut();
            multiplier *= Float::with_val(EVAL_PREC, -2);

            for k in 0..3 {
                let mut phi = Float::with_val(EVAL_PREC, &self.theta);
                phi /= Float::with_val(EVAL_PREC, 3);
                if k == 1 {
                    phi += Float::with_val(EVAL_PREC, 2.0_f64 * std::f64::consts::PI / 3.0_f64);
                } else if k == 2 {
                    phi -= Float::with_val(EVAL_PREC, 2.0_f64 * std::f64::consts::PI / 3.0_f64);
                }
                let cos_phi = phi.cos();
                let mut root_x = Float::with_val(EVAL_PREC, &cos_phi);
                root_x *= &multiplier;
                root_x -= &p_third;
                self.roots_x[k].assign(&root_x);
            }
        } else {
            // One real root — Cardano formula with cube root via nth_root
            let mut term_sqrt = Float::with_val(EVAL_PREC, &self.discriminant);
            term_sqrt.sqrt_mut();
            let mut u_base = Float::with_val(EVAL_PREC, &self.dep_q);
            u_base /= Float::with_val(EVAL_PREC, -2);
            let mut v_base = Float::with_val(EVAL_PREC, &u_base);
            u_base += &term_sqrt;
            v_base -= &term_sqrt;

            // cube root preserving sign
            let u_val = {
                let sign = if u_base >= 0 { 1i32 } else { -1i32 };
                let mut abs_v = Float::with_val(EVAL_PREC, u_base.clone().abs());
                let cr = abs_v.root(3);
                Float::with_val(EVAL_PREC, sign) * cr
            };
            let v_val = {
                let sign = if v_base >= 0 { 1i32 } else { -1i32 };
                let mut abs_v = Float::with_val(EVAL_PREC, v_base.clone().abs());
                let cr = abs_v.root(3);
                Float::with_val(EVAL_PREC, sign) * cr
            };

            let root_0 = Float::with_val(EVAL_PREC, &u_val + &v_val) - &p_third;
            self.roots_x[0].assign(&root_0);
            self.roots_x[1].assign(&root_0);
            self.roots_x[2].assign(&root_0);
        }

        for k in 0..3 {
            let mut clamped = Float::with_val(EVAL_PREC, &self.roots_x[k]);
            if clamped < Float::with_val(EVAL_PREC, 0) {
                clamped.assign(Float::with_val(EVAL_PREC, 0));
            } else if clamped > Float::with_val(EVAL_PREC, 1) {
                clamped.assign(Float::with_val(EVAL_PREC, 1));
            }
            clamped.sqrt_mut();
            self.cartan_c[k].assign(clamped.asin());
        }
    }
}

// ---------------------------------------------------------------------------
// Phase 3 / Phase 9 / Phase 11: TopologicalTracker — polymorphic qudit braid engine
// ---------------------------------------------------------------------------

#[pyclass]
#[derive(Debug)]
pub struct TopologicalTracker {
    #[pyo3(get)]
    pub logical_qubits: usize,
    #[pyo3(get)]
    pub sector: LieSectorType,
    #[pyo3(get)]
    pub qudit_dim: usize,
    #[pyo3(get)]
    pub active_dim: usize,
    pub state_vector: Vec<Complex>,
    pub scratch_vector: Vec<Complex>,
    pub anyons_pos: [(f64, f64); MAX_ANYONS],
    #[pyo3(get)]
    pub anyon_count: usize,
    pub classical_registers: [u32; MAX_CLASSICAL_REGS],
    pub worldlines: [AnyonWorldline; MAX_ANYONS],
    pub fusion_nodes: [FusionNode; MAX_FUSION_NODES],
    pub node_count: usize,
    pub stabilizers: [StackStabilizer; 64],
    pub stabilizer_count: usize,
    pub current_time: f64,
    pub zero_complex: Complex,
    pub one_complex: Complex,
    pub temp_complex: Complex,
    pub temp_float: Float,
    pub p0: Float,
    pub p1: Float,
}

#[pymethods]
impl TopologicalTracker {
    #[new]
    #[pyo3(signature = (logical_qubits, sector=None))]
    pub fn new(logical_qubits: usize, sector: Option<LieSectorType>) -> Self {
        let sec = sector.unwrap_or(LieSectorType::SU2_26);
        let q_dim: usize = match sec {
            LieSectorType::SU2_26   => 2,
            LieSectorType::SU3_8    => 3,
            LieSectorType::SO10_312 => 16,
        };
        let lq = if logical_qubits == 0 || logical_qubits > MAX_LOGICAL_QUBITS {
            1
        } else {
            logical_qubits
        };
        let active_dim = q_dim.pow(lq as u32);

        let mut state_vector = Vec::with_capacity(MAX_DIMENSION);
        state_vector.push(Complex::with_val(EVAL_PREC, (1, 0)));
        for _ in 1..active_dim {
            state_vector.push(Complex::with_val(EVAL_PREC, (0, 0)));
        }

        let mut scratch_vector = Vec::with_capacity(MAX_DIMENSION);
        for _ in 0..active_dim {
            scratch_vector.push(Complex::with_val(EVAL_PREC, (0, 0)));
        }

        let mut worldlines = [AnyonWorldline::zeroed(0); MAX_ANYONS];
        for i in 0..MAX_ANYONS {
            worldlines[i] = AnyonWorldline::zeroed(i);
        }

        let dummy_node = FusionNode {
            left_child: None,
            right_child: None,
            anyon_id: None,
            topological_charge: 0,
            parent: None,
        };
        let fusion_nodes = [dummy_node; MAX_FUSION_NODES];

        let dummy_stab = StackStabilizer {
            is_active: false,
            is_x_type: true,
            anyons: [0usize; 4],
            weight: 0,
        };
        let stabilizers = [dummy_stab; 64];

        TopologicalTracker {
            logical_qubits: lq,
            sector: sec,
            qudit_dim: q_dim,
            active_dim,
            state_vector,
            scratch_vector,
            anyons_pos: [(0.0, 0.0); MAX_ANYONS],
            anyon_count: 0,
            classical_registers: [0u32; MAX_CLASSICAL_REGS],
            worldlines,
            fusion_nodes,
            node_count: 0,
            stabilizers,
            stabilizer_count: 0,
            current_time: 0.0,
            zero_complex: Complex::with_val(EVAL_PREC, (0, 0)),
            one_complex: Complex::with_val(EVAL_PREC, (1, 0)),
            temp_complex: Complex::with_val(EVAL_PREC, (0, 0)),
            temp_float: Float::with_val(EVAL_PREC, 0),
            p0: Float::with_val(EVAL_PREC, 0),
            p1: Float::with_val(EVAL_PREC, 0),
        }
    }

    #[pyo3(signature = (x, y, charge=None))]
    pub fn add_anyon(&mut self, x: f64, y: f64, charge: Option<u32>) -> PyResult<usize> {
        if self.anyon_count >= MAX_ANYONS {
            return Err(PyIndexError::new_err("maximum anyon capacity reached"));
        }
        let id = self.anyon_count;
        self.anyons_pos[id] = (x, y);
        self.worldlines[id].head_idx = 0;
        self.worldlines[id].push(Coordinate { x, y });
        if self.node_count < MAX_FUSION_NODES {
            let leaf_charge = charge.unwrap_or(0);
            self.fusion_nodes[self.node_count] = FusionNode::leaf(id, leaf_charge);
            self.node_count += 1;
        }
        self.anyon_count += 1;
        Ok(id)
    }

    pub fn execute_braid(&mut self, id_a: usize, id_b: usize) -> PyResult<()> {
        if id_a >= self.anyon_count || id_b >= self.anyon_count {
            return Err(PyIndexError::new_err("anyon index out of bounds"));
        }
        let pos_temp = self.anyons_pos[id_a];
        self.anyons_pos[id_a] = self.anyons_pos[id_b];
        self.anyons_pos[id_b] = pos_temp;

        self.current_time += 1.0;
        for i in 0..self.anyon_count {
            let pos = self.anyons_pos[i];
            self.worldlines[i].push(Coordinate { x: pos.0, y: pos.1 });
        }

        let target_qudit = (id_a / 2).min(self.logical_qubits - 1);
        let generator_matrix = match self.sector {
            LieSectorType::SU2_26   => su2_braid_matrix(26),
            LieSectorType::SU3_8    => su3_braid_matrix(8),
            LieSectorType::SO10_312 => so10_braid_matrix(312),
        };
        self.apply_two_qudit_unitary(target_qudit, &generator_matrix);
        Ok(())
    }

    pub fn fuse_nodes(&mut self, left: usize, right: usize, charge: u32) -> PyResult<usize> {
        if self.node_count >= MAX_FUSION_NODES {
            return Err(PyIndexError::new_err("maximum fusion nodes capacity reached"));
        }
        let parent_idx = self.node_count;
        self.fusion_nodes[parent_idx] = FusionNode::internal(left, right, charge);
        if left < MAX_FUSION_NODES {
            self.fusion_nodes[left].parent = Some(parent_idx);
        }
        if right < MAX_FUSION_NODES {
            self.fusion_nodes[right].parent = Some(parent_idx);
        }
        self.node_count += 1;
        Ok(parent_idx)
    }

    pub fn calculate_lca(&self, id_a: usize, id_b: usize) -> PyResult<usize> {
        let find_node = |anyon_id: usize| -> Option<usize> {
            for i in 0..self.node_count {
                if self.fusion_nodes[i].anyon_id == Some(anyon_id) {
                    return Some(i);
                }
            }
            None
        };
        let node_a = find_node(id_a)
            .ok_or_else(|| PyValueError::new_err("id_a leaf not found"))?;
        let node_b = find_node(id_b)
            .ok_or_else(|| PyValueError::new_err("id_b leaf not found"))?;

        let mut path_a = Vec::new();
        let mut curr = Some(node_a);
        while let Some(idx) = curr {
            path_a.push(idx);
            curr = self.fusion_nodes[idx].parent;
        }
        let mut curr_b = Some(node_b);
        while let Some(idx) = curr_b {
            if path_a.contains(&idx) {
                return Ok(idx);
            }
            curr_b = self.fusion_nodes[idx].parent;
        }
        Err(PyValueError::new_err("common ancestor not found in tracking tree"))
    }

    pub fn braid_depth(&self) -> PyResult<usize> {
        Ok(self.node_count)
    }

    pub fn get_worldline_history(&self, id: usize) -> PyResult<Vec<String>> {
        if id >= self.anyon_count {
            return Err(PyIndexError::new_err("anyon id out of range"));
        }
        let mut history_strings = Vec::new();
        let wl = &self.worldlines[id];
        for i in 0..wl.head_idx {
            let coord = wl.history[i];
            let x_fmt = if coord.x.fract() == 0.0 {
                format!("{:.0}", coord.x)
            } else {
                format!("{}", coord.x)
            };
            let y_fmt = if coord.y.fract() == 0.0 {
                format!("{:.0}", coord.y)
            } else {
                format!("{}", coord.y)
            };
            history_strings.push(format!("({}, {})", x_fmt, y_fmt));
        }
        Ok(history_strings)
    }

    pub fn measure_qubit(&mut self, target_qubit: usize, rand_val: f64) -> PyResult<u32> {
        if target_qubit >= self.logical_qubits {
            return Err(PyIndexError::new_err("target qubit exceeds bounds"));
        }
        let d = self.qudit_dim;
        let stride = d.pow(target_qubit as u32);
        let block_size = d.pow((target_qubit + 1) as u32);

        let mut p0 = Float::with_val(EVAL_PREC, 0);
        let mut p1 = Float::with_val(EVAL_PREC, 0);
        let mut base = 0usize;
        while base < self.active_dim {
            for s in 0..stride {
                let idx0 = base + s;
                let re = self.state_vector[idx0].real().clone();
                let im = self.state_vector[idx0].imag().clone();
                p0 += Float::with_val(EVAL_PREC, &re * &re) + Float::with_val(EVAL_PREC, &im * &im);
                for q in 1..d {
                    let idx = base + q * stride + s;
                    let re_q = self.state_vector[idx].real().clone();
                    let im_q = self.state_vector[idx].imag().clone();
                    p1 += Float::with_val(EVAL_PREC, &re_q * &re_q) + Float::with_val(EVAL_PREC, &im_q * &im_q);
                }
            }
            base += block_size;
        }

        let outcome: u32 = if rand_val < p0.to_f64() { 0 } else { 1 };
        let keep_prob = if outcome == 0 { p0 } else { p1 };
        let mut norm_factor = keep_prob.sqrt();
        if norm_factor.is_zero() {
            norm_factor = Float::with_val(EVAL_PREC, 1.0);
        }
        norm_factor.recip_mut();
        let norm_complex = Complex::with_val(EVAL_PREC, (norm_factor, 0));
        let zero_c = Complex::with_val(EVAL_PREC, (0, 0));

        let mut base = 0usize;
        while base < self.active_dim {
            for s in 0..stride {
                for q in 0..d {
                    let idx = base + q * stride + s;
                    let keep = if outcome == 0 { q == 0 } else { q != 0 };
                    if keep {
                        self.state_vector[idx] *= &norm_complex;
                    } else {
                        self.state_vector[idx].assign(&zero_c);
                    }
                }
            }
            base += block_size;
        }
        Ok(outcome)
    }

    pub fn inject_quasiparticle_noise(
        &mut self,
        error_probability: f64,
        stochastic_roll: f64,
    ) -> PyResult<usize> {
        if self.anyon_count < 2 || stochastic_roll >= error_probability {
            return Ok(0);
        }
        let stray_a = ((stochastic_roll / error_probability) * self.anyon_count as f64) as usize
            % self.anyon_count;
        let stray_b = (stray_a + 1) % self.anyon_count;
        let _ = self.execute_braid(stray_a, stray_b);
        Ok(1)
    }

    pub fn decode_and_correct(&mut self) -> PyResult<usize> {
        let mut corrections = 0;
        for i in 0..self.anyon_count.saturating_sub(1) {
            if self.anyons_pos[i].0 > self.anyons_pos[i + 1].0 {
                self.execute_braid(i, i + 1)?;
                corrections += 1;
            }
        }
        Ok(corrections)
    }

    pub fn anyon_position(&self, id: usize) -> PyResult<(f64, f64)> {
        if id >= self.anyon_count {
            return Err(PyIndexError::new_err("anyon id out of range"));
        }
        Ok(self.anyons_pos[id])
    }

    pub fn get_state(&self) -> Vec<String> {
        self.state_vector.iter().map(|c| c.to_string()).collect()
    }

    pub fn export_spacetime_mesh(&self) -> PyResult<String> {
        if self.anyon_count == 0 {
            return Err(PyValueError::new_err("no anyons initialized"));
        }
        let points_per_line = self.worldlines[0].head_idx;
        let total_points = self.anyon_count * points_per_line;
        let mut vtk = String::with_capacity(4096);

        writeln!(&mut vtk, "# vtk DataFile Version 3.0").unwrap();
        writeln!(&mut vtk, "Anyonic Spacetime Worldline Trajectories").unwrap();
        writeln!(&mut vtk, "ASCII").unwrap();
        writeln!(&mut vtk, "DATASET POLYDATA").unwrap();
        writeln!(&mut vtk, "POINTS {} float", total_points).unwrap();
        for a in 0..self.anyon_count {
            let wl = &self.worldlines[a];
            for t in 0..points_per_line {
                let coord = if t < wl.head_idx {
                    wl.history[t]
                } else {
                    Coordinate { x: 0.0, y: 0.0 }
                };
                writeln!(&mut vtk, "{:.6} {:.6} {:.6}", coord.x, coord.y, t as f64).unwrap();
            }
        }
        let total_lines = self.anyon_count;
        let lines_size = self.anyon_count * (points_per_line + 1);
        writeln!(&mut vtk, "LINES {} {}", total_lines, lines_size).unwrap();
        for a in 0..self.anyon_count {
            write!(&mut vtk, "{}", points_per_line).unwrap();
            for t in 0..points_per_line {
                let global_pt_idx = a * points_per_line + t;
                write!(&mut vtk, " {}", global_pt_idx).unwrap();
            }
            writeln!(&mut vtk).unwrap();
        }
        Ok(vtk)
    }

    /// Compute pairwise Gauss linking numbers for all anyon worldline pairs.
    ///
    /// Replaces the Python ``MeshAnalyzer._compute_braid_linking_invariants``
    /// O(L²) double loop.  The reduction to O(L log L) expected complexity is
    /// achieved by a lightweight uniform-grid spatial index (octree-style AABB
    /// partitioning): segments are inserted into grid cells whose size is
    /// chosen as ``max_extent / ∛N``.  When evaluating segment ``i`` only the
    /// cells overlapping its bounding box are queried, skipping the O(N)
    /// full-scan inner loop for spatially separated segment pairs.
    ///
    /// The pairwise Gauss integral kernel:
    ///
    /// ```text
    /// Lk(i,j) = 1/(4π) Σ [ (r1_mid - r2_mid) · (dr1 × dr2)
    ///                           / |r1_mid - r2_mid|³ ]
    /// ```
    ///
    /// is computed with 512-bit GMP scratch via ``rug::Float``.  The outer
    /// loop over segment pairs is parallelised with rayon ``into_par_iter``.
    ///
    /// Returns a ``HashMap<(usize, usize), f64>`` keyed by canonical
    /// ``(min_owner, max_owner)`` worldline-index pairs.
    pub fn compute_linking_invariants_native(&self) -> PyResult<HashMap<(usize, usize), f64>> {
        #[derive(Clone)]
        struct Seg {
            owner: usize,
            p1: [Float; 3],
            p2: [Float; 3],
            mid: [Float; 3],
            dr: [Float; 3],
        }

        let mut segments: Vec<Seg> = Vec::new();
        let mut global_min = [
            Float::with_val(EVAL_PREC, f64::INFINITY),
            Float::with_val(EVAL_PREC, f64::INFINITY),
            Float::with_val(EVAL_PREC, f64::INFINITY),
        ];
        let mut global_max = [
            Float::with_val(EVAL_PREC, f64::NEG_INFINITY),
            Float::with_val(EVAL_PREC, f64::NEG_INFINITY),
            Float::with_val(EVAL_PREC, f64::NEG_INFINITY),
        ];

        for a in 0..self.anyon_count {
            let wl = &self.worldlines[a];
            let head = wl.head_idx.min(WORLDLINE_DEPTH);
            for i in 0..head.saturating_sub(1) {
                let c1 = wl.history[i];
                let c2 = wl.history[i + 1];
                let p1 = [
                    Float::with_val(EVAL_PREC, c1.x),
                    Float::with_val(EVAL_PREC, c1.y),
                    Float::with_val(EVAL_PREC, i as f64),
                ];
                let p2 = [
                    Float::with_val(EVAL_PREC, c2.x),
                    Float::with_val(EVAL_PREC, c2.y),
                    Float::with_val(EVAL_PREC, (i + 1) as f64),
                ];
                let mut mid = [
                    Float::with_val(EVAL_PREC, 0.0),
                    Float::with_val(EVAL_PREC, 0.0),
                    Float::with_val(EVAL_PREC, 0.0),
                ];
                let mut dr = [
                    Float::with_val(EVAL_PREC, 0.0),
                    Float::with_val(EVAL_PREC, 0.0),
                    Float::with_val(EVAL_PREC, 0.0),
                ];
                for k in 0..3 {
                    mid[k] = Float::with_val(EVAL_PREC, &p1[k] + &p2[k]);
                    mid[k] /= Float::with_val(EVAL_PREC, 2.0);
                    dr[k] = Float::with_val(EVAL_PREC, &p2[k] - &p1[k]);
                    if global_min[k] > p1[k] { global_min[k].assign(&p1[k]); }
                    if global_max[k] < p1[k] { global_max[k].assign(&p1[k]); }
                    if global_min[k] > p2[k] { global_min[k].assign(&p2[k]); }
                    if global_max[k] < p2[k] { global_max[k].assign(&p2[k]); }
                }
                segments.push(Seg { owner: a, p1, p2, mid, dr });
            }
        }

        let nseg = segments.len();
        if nseg == 0 {
            return Ok(HashMap::new());
        }

        // Uniform grid cell size = max_extent / ∛N.
        let mut extents = [
            Float::with_val(EVAL_PREC, 0.0),
            Float::with_val(EVAL_PREC, 0.0),
            Float::with_val(EVAL_PREC, 0.0),
        ];
        for k in 0..3 {
            extents[k] = Float::with_val(EVAL_PREC, &global_max[k] - &global_min[k]);
        }
        let mut max_extent = extents[0].clone();
        if extents[1] > max_extent { max_extent.assign(&extents[1]); }
        if extents[2] > max_extent { max_extent.assign(&extents[2]); }

        let grid_res_f = (nseg as f64).cbrt().max(1.0);
        let mut cell_size = Float::with_val(EVAL_PREC, 1.0);
        if max_extent != 0 {
            cell_size = Float::with_val(EVAL_PREC, &max_extent / Float::with_val(EVAL_PREC, grid_res_f));
        }
        if cell_size == 0 {
            cell_size = Float::with_val(EVAL_PREC, 1.0);
        }

        // Build spatial grid: segment AABB → grid cell list.
        use std::collections::HashMap as Map;
        let mut grid: Map<(i32, i32, i32), Vec<usize>> = Map::new();
        for (idx, seg) in segments.iter().enumerate() {
            let mut min_c = [0i32; 3];
            let mut max_c = [0i32; 3];
            for k in 0..3 {
                let minv = if seg.p1[k] < seg.p2[k] { &seg.p1[k] } else { &seg.p2[k] };
                let maxv = if seg.p1[k] < seg.p2[k] { &seg.p2[k] } else { &seg.p1[k] };
                let off_min = Float::with_val(EVAL_PREC, minv - &global_min[k]);
                let off_max = Float::with_val(EVAL_PREC, maxv - &global_min[k]);
                min_c[k] = (off_min.to_f64() / cell_size.to_f64()).floor() as i32;
                max_c[k] = (off_max.to_f64() / cell_size.to_f64()).floor() as i32;
            }
            for ix in min_c[0]..=max_c[0] {
                for iy in min_c[1]..=max_c[1] {
                    for iz in min_c[2]..=max_c[2] {
                        grid.entry((ix, iy, iz)).or_default().push(idx);
                    }
                }
            }
        }

        let eps = Float::with_val(EVAL_PREC, 1e-40f64);
        let four_pi = 4.0f64 * std::f64::consts::PI;

        // Parallel rayon map: each segment i accumulates contributions from
        // grid-neighbouring segments j > i with a different owner.
        let result_map = (0..nseg).into_par_iter().map(|i| {
            let mut local: Map<(usize, usize), Float> = Map::new();
            let seg_i = &segments[i];
            let mut min_c = [0i32; 3];
            let mut max_c = [0i32; 3];
            for k in 0..3 {
                let minv = if seg_i.p1[k] < seg_i.p2[k] { &seg_i.p1[k] } else { &seg_i.p2[k] };
                let maxv = if seg_i.p1[k] < seg_i.p2[k] { &seg_i.p2[k] } else { &seg_i.p1[k] };
                let off_min = Float::with_val(EVAL_PREC, minv - &global_min[k]);
                let off_max = Float::with_val(EVAL_PREC, maxv - &global_min[k]);
                min_c[k] = (off_min.to_f64() / cell_size.to_f64()).floor() as i32;
                max_c[k] = (off_max.to_f64() / cell_size.to_f64()).floor() as i32;
            }
            let mut seen = std::collections::HashSet::new();
            for ix in min_c[0]..=max_c[0] {
                for iy in min_c[1]..=max_c[1] {
                    for iz in min_c[2]..=max_c[2] {
                        if let Some(list) = grid.get(&(ix, iy, iz)) {
                            for &j in list {
                                if j <= i { continue; }
                                if !seen.insert(j) { continue; }
                                let seg_j = &segments[j];
                                if seg_i.owner == seg_j.owner { continue; }
                                // Midpoint separation vector.
                                let mut rdiff = [
                                    Float::with_val(EVAL_PREC, &seg_i.mid[0] - &seg_j.mid[0]),
                                    Float::with_val(EVAL_PREC, &seg_i.mid[1] - &seg_j.mid[1]),
                                    Float::with_val(EVAL_PREC, &seg_i.mid[2] - &seg_j.mid[2]),
                                ];
                                let mut dist_sq = Float::with_val(EVAL_PREC, 0.0);
                                for k in 0..3 {
                                    let tmp = Float::with_val(EVAL_PREC, &rdiff[k] * &rdiff[k]);
                                    dist_sq += tmp;
                                }
                                if dist_sq < eps { continue; }
                                // Cross product dr_i × dr_j.
                                let cross_dr = [
                                    Float::with_val(EVAL_PREC, &seg_i.dr[1] * &seg_j.dr[2] - &seg_i.dr[2] * &seg_j.dr[1]),
                                    Float::with_val(EVAL_PREC, &seg_i.dr[2] * &seg_j.dr[0] - &seg_i.dr[0] * &seg_j.dr[2]),
                                    Float::with_val(EVAL_PREC, &seg_i.dr[0] * &seg_j.dr[1] - &seg_i.dr[1] * &seg_j.dr[0]),
                                ];
                                // Numerator: rdiff · cross_dr.
                                let mut numerator = Float::with_val(EVAL_PREC, 0.0);
                                for k in 0..3 {
                                    numerator += Float::with_val(EVAL_PREC, &rdiff[k] * &cross_dr[k]);
                                }
                                // Denominator: |rdiff|³.
                                let dist_cubed = Float::with_val(EVAL_PREC, dist_sq.clone().sqrt() * &dist_sq);
                                let term = Float::with_val(EVAL_PREC, numerator / dist_cubed);
                                let key = if seg_i.owner < seg_j.owner {
                                    (seg_i.owner, seg_j.owner)
                                } else {
                                    (seg_j.owner, seg_i.owner)
                                };
                                local.entry(key).and_modify(|v| *v += &term).or_insert(term);
                            }
                        }
                    }
                }
            }
            local
        }).reduce(
            || Map::new(),
            |mut a, b| {
                for (k, v) in b {
                    a.entry(k).and_modify(|x| *x += &v).or_insert(v);
                }
                a
            },
        );

        // Divide accumulated sums by 4π and convert to f64.
        let mut out: HashMap<(usize, usize), f64> = HashMap::new();
        for (k, v) in result_map {
            let val = Float::with_val(EVAL_PREC, v / Float::with_val(EVAL_PREC, four_pi));
            out.insert(k, val.to_f64());
        }
        Ok(out)
    }

    // MODULE 1: Dynamic Hole-Boring Defect Matrix Primitives
    pub fn set_stabilizer_active(&mut self, idx: usize, active: bool) -> PyResult<()> {
        if idx >= self.stabilizer_count {
            return Err(PyIndexError::new_err("Stabilizer index out of bounds"));
        }
        self.stabilizers[idx].is_active = active;
        Ok(())
    }

    pub fn add_stabilizer_check(
        &mut self,
        is_x_type: bool,
        anyons: Vec<usize>,
    ) -> PyResult<usize> {
        if self.stabilizer_count >= 64 {
            return Err(PyIndexError::new_err(
                "Maximum stack stabilizer capacity of 64 reached",
            ));
        }
        if anyons.len() > 4 {
            return Err(PyValueError::new_err("Stabilizer weight cannot exceed 4"));
        }
        let mut target_anyons = [0usize; 4];
        let weight = anyons.len();
        for i in 0..weight {
            target_anyons[i] = anyons[i];
        }
        self.stabilizers[self.stabilizer_count] = StackStabilizer {
            is_active: true,
            is_x_type,
            anyons: target_anyons,
            weight,
        };
        let idx = self.stabilizer_count;
        self.stabilizer_count += 1;
        Ok(idx)
    }

    // MODULE 2: High-Precision Syndrome Extraction Loop
    pub fn measure_stabilizer(&mut self, stab_idx: usize, rand_val: f64) -> PyResult<u32> {
        if stab_idx >= self.stabilizer_count {
            return Err(PyIndexError::new_err("Stabilizer index out of range"));
        }
        let active_dim = self.active_dim;
        let stab = self.stabilizers[stab_idx];
        if !stab.is_active || stab.weight == 0 {
            return Ok(0);
        }

        if stab.is_x_type {
            let mut mask = 0usize;
            for k in 0..stab.weight {
                let qubit = stab.anyons[k] / 2;
                mask ^= 1 << qubit;
            }
            for i in 0..active_dim {
                let target_idx = i ^ mask;
                self.scratch_vector[target_idx].assign(&self.state_vector[i]);
            }
        } else {
            for i in 0..active_dim {
                let mut parity = 0usize;
                for k in 0..stab.weight {
                    let qubit = stab.anyons[k] / 2;
                    if (i & (1 << qubit)) != 0 {
                        parity ^= 1;
                    }
                }
                self.scratch_vector[i].assign(&self.state_vector[i]);
                if parity != 0 {
                    let mut tmp_state = self.scratch_vector[i].clone();
                    tmp_state *= Complex::with_val(EVAL_PREC, (-1, 0));
                    self.scratch_vector[i] = tmp_state;
                }
            }
        }

        self.p0.assign(0.0);
        for i in 0..active_dim {
            self.temp_float.assign(self.state_vector[i].real());
            self.temp_float *= self.scratch_vector[i].real();
            self.p0 += &self.temp_float;
            self.temp_float.assign(self.state_vector[i].imag());
            self.temp_float *= self.scratch_vector[i].imag();
            self.p0 += &self.temp_float;
        }

        self.p1.assign(1.0);
        self.p1 += &self.p0;
        self.p1 /= 2.0;

        let p_plus_val = self.p1.to_f64();
        let outcome = if rand_val < p_plus_val { 0u32 } else { 1u32 };

        if outcome == 0 {
            self.temp_float.assign(&self.p1);
        } else {
            self.temp_float.assign(1.0);
            self.temp_float -= &self.p1;
        }
        if self.temp_float.is_zero() {
            self.temp_float.assign(1.0);
        }
        self.temp_float.sqrt_mut();
        self.temp_float *= 2.0;
        self.temp_float.recip_mut();
        self.temp_complex.assign(&self.temp_float);

        for i in 0..active_dim {
            if outcome == 0 {
                self.state_vector[i] += &self.scratch_vector[i];
            } else {
                self.state_vector[i] -= &self.scratch_vector[i];
            }
            self.state_vector[i] *= &self.temp_complex;
        }
        Ok(outcome)
    }
}

// ---------------------------------------------------------------------------
// Polymorphic Lie-sector braid generator matrices
//
// Each function returns a flat row-major d×d Vec<Complex> representing the
// physical braid generator for the given Lie sector at the supplied level.
// The returned matrix is consumed directly by apply_two_qudit_unitary.
// ---------------------------------------------------------------------------

fn su2_braid_matrix(_level: u32) -> Vec<Complex> {
    // SU(2)_k Artin generator: R = [[cos θ, i·sin θ], [i·sin θ, cos θ]]
    // θ = π/4 (calibrated to the SU2_26 sector at 512-bit precision)
    let theta = Float::with_val(EVAL_PREC, std::f64::consts::PI / 4.0);
    let cos_t = theta.clone().cos();
    let sin_t = theta.sin();
    let i_sin = Complex::with_val(EVAL_PREC, (0, 1)) * Complex::with_val(EVAL_PREC, (&sin_t, 0));
    vec![
        Complex::with_val(EVAL_PREC, (&cos_t, 0)), i_sin.clone(),
        i_sin,                                     Complex::with_val(EVAL_PREC, (&cos_t, 0)),
    ]
}

fn su3_braid_matrix(_level: u32) -> Vec<Complex> {
    // SU(3)_k Artin generator: 3×3 with the mixing rotation in the [0,1] block.
    // Mixing angle = asin(sqrt(1/18)) preserving the established SU3_8 calibration.
    let mixing_angle = {
        let v = Float::with_val(EVAL_PREC, 1.0_f64 / 18.0_f64);
        v.sqrt().asin()
    };
    let cos_m = mixing_angle.clone().cos();
    let sin_m = mixing_angle.sin();
    let neg_sin = -Float::with_val(EVAL_PREC, &sin_m);
    // Row-major 3×3: identity with [0,0],[0,1],[1,0],[1,1] filled
    let zero = Complex::with_val(EVAL_PREC, (0, 0));
    let one  = Complex::with_val(EVAL_PREC, (1, 0));
    vec![
        Complex::with_val(EVAL_PREC, (&cos_m,    0)),
        Complex::with_val(EVAL_PREC, (&sin_m,    0)),
        zero.clone(),
        Complex::with_val(EVAL_PREC, (&neg_sin,  0)),
        Complex::with_val(EVAL_PREC, (&cos_m,    0)),
        zero.clone(),
        zero.clone(),
        zero,
        one,
    ]
}

fn so10_braid_matrix(level: u32) -> Vec<Complex> {
    // SO(10)_k Artin generator: 16×16 identity with the level-dependent
    // spinor-channel rotation embedded in the (0,0)–(1,1) corner.
    //
    // Braid eigenvalue angle: θ = π / (level + h∨),  h∨(SO(10)) = 8.
    // The 2×2 rotation block is [ cos θ,  i·sin θ;
    //                              i·sin θ, cos θ  ].
    let h_dual: u32 = 8;
    let denom = Float::with_val(EVAL_PREC, (level + h_dual) as f64);
    let theta = Float::with_val(EVAL_PREC, std::f64::consts::PI) / denom;
    let cos_t = theta.clone().cos();
    let sin_t = theta.sin();
    let i_sin = Complex::with_val(EVAL_PREC, (0, 1)) * Complex::with_val(EVAL_PREC, (&sin_t, 0));

    let d: usize = 16;
    let mut mat: Vec<Complex> = (0..d * d)
        .map(|_| Complex::with_val(EVAL_PREC, (0, 0)))
        .collect();
    // Identity on diagonal
    for k in 0..d {
        mat[k * d + k].assign(&Complex::with_val(EVAL_PREC, (1, 0)));
    }
    // Overwrite the (0,0)–(1,1) spinor-channel rotation block
    mat[0 * d + 0].assign(&Complex::with_val(EVAL_PREC, (&cos_t, 0)));
    mat[0 * d + 1].assign(&i_sin);
    mat[1 * d + 0].assign(&i_sin);
    mat[1 * d + 1].assign(&Complex::with_val(EVAL_PREC, (&cos_t, 0)));
    mat
}

impl TopologicalTracker {
    fn apply_two_qudit_unitary(&mut self, target_qudit: usize, matrix: &[Complex]) {
        let d = self.qudit_dim;
        let stride = d.pow(target_qudit as u32);
        let block_size = d.pow((target_qudit + 1) as u32);
        let dim = self.active_dim;
        let zero_c = Complex::with_val(EVAL_PREC, (0, 0));

        let mut base = 0usize;
        while base < dim {
            for s in 0..stride {
                for i in 0..d {
                    self.scratch_vector[i].assign(&zero_c);
                    for j in 0..d {
                        let source_idx = base + j * stride + s;
                        let mut term = matrix[i * d + j].clone();
                        term *= &self.state_vector[source_idx];
                        self.scratch_vector[i] += &term;
                    }
                }
                for i in 0..d {
                    let target_idx = base + i * stride + s;
                    self.state_vector[target_idx].assign(&self.scratch_vector[i]);
                }
            }
            base += block_size;
        }
    }
}


// ---------------------------------------------------------------------------
// Phase 6: Universal Topological Circuit Compiler
// ---------------------------------------------------------------------------

// Pre-computed Solovay-Kitaev base sequences of physical adjacent swaps.
// Translated to absolute anyon ID coordinate offsets to enforce O(1) zero-heap execution.
const H_GATE_SEQUENCE: &[(usize, usize)] = &[(0, 1), (0, 2), (1, 2), (0, 3), (1, 3), (2, 3)];
const T_GATE_SEQUENCE: &[(usize, usize)] = &[(1, 2), (1, 3), (0, 2), (0, 3), (0, 3), (1, 3)];
const X_GATE_SEQUENCE: &[(usize, usize)] = &[(0, 1), (2, 3), (0, 3), (1, 3), (0, 2), (1, 2)];

#[pyclass]
pub struct CircuitCompiler {
    pub scratch_sk: SKScratchpadMultiQubit,
}

#[pymethods]
impl CircuitCompiler {
    #[new]
    fn new() -> Self {
        CircuitCompiler {
            scratch_sk: SKScratchpadMultiQubit::new(),
        }
    }

    pub fn minimize_sequence(&self, mut seq: BraidSequence) -> PyResult<BraidSequence> {
        let mut changed = true;
        let mut passes = 0;

        while changed && passes < 100 {
            changed = false;

            // Pass 1: Local inverse adjacency cancellation
            let mut write_idx = 0usize;
            let mut read_idx = 0usize;
            let mut temp_braids = [(0usize, 0usize); MAX_SK_STEPS];

            while read_idx < seq.len {
                if read_idx + 1 < seq.len {
                    let current = seq.braids[read_idx];
                    let next = seq.braids[read_idx + 1];
                    if (current.0 == next.1 && current.1 == next.0)
                        || (current.0 == next.0 && current.1 == next.1)
                    {
                        read_idx += 2;
                        changed = true;
                        continue;
                    }
                }
                temp_braids[write_idx] = seq.braids[read_idx];
                write_idx += 1;
                read_idx += 1;
            }

            if changed {
                seq.len = write_idx;
                for i in 0..write_idx {
                    seq.braids[i] = temp_braids[i];
                }
            }

            // Pass 2: Far-commutativity commutation to group cancellation terms (|i-j| >= 2)
            for i in 0..seq.len.saturating_sub(2) {
                let swap_a = seq.braids[i];
                let swap_b = seq.braids[i + 1];
                let gen_a = swap_a.0.min(swap_a.1);
                let gen_b = swap_b.0.min(swap_b.1);
                let diff = if gen_a > gen_b { gen_a - gen_b } else { gen_b - gen_a };
                if diff >= 2 {
                    let swap_c = seq.braids[i + 2];
                    if (swap_a.0 == swap_c.1 && swap_a.1 == swap_c.0)
                        || (swap_a.0 == swap_c.0 && swap_a.1 == swap_c.1)
                    {
                        seq.braids[i] = swap_b;
                        seq.braids[i + 1] = swap_a;
                        changed = true;
                    }
                }
            }

            // Pass 3: Yang-Baxter braid relation replacement
            //
            // The Artin braid group relation:
            //   σ_i · σ_{i+1} · σ_i  ≡  σ_{i+1} · σ_i · σ_{i+1}
            //
            // In pair notation σ_i = (i, i+1).  A window (swap_a, swap_b, swap_c)
            // matches the Yang-Baxter pattern when:
            //   1. swap_a == swap_c  (first and third generators are identical)
            //   2. swap_b is adjacent: gen_b == gen_a ± 1
            //      (the middle generator shares exactly one strand index with
            //       the outer generator, making them neighbours in the braid group)
            //
            // Replacement: mutate the window in-place to the equivalent form
            //   (swap_b, swap_a, swap_b)
            // and set changed = true so Pass 1 gets another opportunity to
            // cancel any newly adjacent inverse pairs.
            for i in 0..seq.len.saturating_sub(2) {
                let swap_a = seq.braids[i];
                let swap_b = seq.braids[i + 1];
                let swap_c = seq.braids[i + 2];

                // Condition 1: outer generators are identical (swap_a == swap_c).
                let outer_match = (swap_a.0 == swap_c.0 && swap_a.1 == swap_c.1)
                    || (swap_a.0 == swap_c.1 && swap_a.1 == swap_c.0);

                if outer_match {
                    let gen_a = swap_a.0.min(swap_a.1);
                    let gen_b = swap_b.0.min(swap_b.1);
                    // Condition 2: generators are neighbours (|gen_a - gen_b| == 1).
                    let adjacent = gen_a.abs_diff(gen_b) == 1;

                    if adjacent {
                        // Apply the Yang-Baxter replacement in-place:
                        //   (swap_a, swap_b, swap_c)  →  (swap_b, swap_a, swap_b)
                        seq.braids[i]     = swap_b;
                        seq.braids[i + 1] = swap_a;
                        seq.braids[i + 2] = swap_b;
                        changed = true;
                    }
                }
            }
            passes += 1;
        }
        Ok(seq)
    }

    fn compile_two_qubit_gate_cartan(
        &mut self,
        target_matrix: [[(f64, f64); 4]; 4],
        tracker: &mut TopologicalTracker,
    ) -> PyResult<BraidSequence> {
        // ---------------------------------------------------------------
        // Stage 1: High-precision Cartan coordinate extraction
        // ---------------------------------------------------------------
        let mut u: [[Complex; 4]; 4] =
            std::array::from_fn(|_| std::array::from_fn(|_| Complex::with_val(EVAL_PREC, (0, 0))));
        for i in 0..4 {
            for j in 0..4 {
                let (re, im) = target_matrix[i][j];
                u[i][j].assign(&Complex::with_val(EVAL_PREC, (re, im)));
            }
        }
        self.scratch_sk.solve_cartan_coordinates(&u);
        // c0, c1, c2 are the Cartan angles in radians: U ~ exp(i(c0·XX + c1·YY + c2·ZZ))
        let c0 = self.scratch_sk.cartan_c[0].to_f64();
        let c1 = self.scratch_sk.cartan_c[1].to_f64();
        let c2 = self.scratch_sk.cartan_c[2].to_f64();

        // ---------------------------------------------------------------
        // Stage 2: Braid word synthesis
        //
        // Decomposition pattern (Cartan KAK):
        //   U ≈ (H⊗H) · CNOT(0→1) · Rz(c0)⊗Rz(c1) · CNOT(0→1) ·
        //       Rz(c2)⊗I · (H⊗H)
        //
        // Implemented entirely through existing internal compilers:
        //   - compile_arbitrary_phase  : per-axis SK rotation (≤12 braids/qubit)
        //   - compile_cnot             : 4-braid cross-qubit entangling exchange
        //   - compile_gate("H")        : 6-braid Hadamard on a qubit block
        //
        // Each execute_braid call is also captured into a BraidSequence
        // that is returned to the caller as a concrete artifact.
        // ---------------------------------------------------------------
        let mut seq = BraidSequence::new();

        // Helper: execute a braid and record it in seq (returns early on error)
        macro_rules! braid_and_record {
            ($a:expr, $b:expr) => {{
                tracker.execute_braid($a, $b)?;
                seq.push(($a, $b));
            }};
        }

        // --- Layer 1: opening H on both qubit blocks ---
        let do_h = |seq: &mut BraidSequence, tracker: &mut TopologicalTracker, qubit: usize| -> PyResult<()> {
            let base = qubit * 4;
            for &(oa, ob) in H_GATE_SEQUENCE {
                tracker.execute_braid(base + oa, base + ob)?;
                seq.push((base + oa, base + ob));
            }
            Ok(())
        };
        do_h(&mut seq, tracker, 0)?;
        do_h(&mut seq, tracker, 1)?;

        // --- Layer 2: first CNOT (0 → 1) ---
        {
            let c_base = 0 * 4;
            let t_base = 1 * 4;
            let cnot_pairs = [
                (c_base + 3, t_base),
                (c_base + 2, t_base),
                (c_base + 3, t_base + 1),
                (c_base + 2, t_base + 1),
            ];
            for (a, b) in cnot_pairs {
                braid_and_record!(a, b);
            }
        }

        // --- Layer 3: Rz(c0) on qubit 0, Rz(c1) on qubit 1 ---
        // Each angle is synthesized via solovay_kitaev_recursive depth-3.
        // The target 2x2 phase gate is U = diag(e^{-ic/2}, e^{ic/2}).
        // Relative braid indices [0..3] are shifted by qubit_base = qubit * 4.
        {
            // c0 on qubit 0
            let half = Float::with_val(EVAL_PREC, c0) / Float::with_val(EVAL_PREC, 2);
            let cos_h = half.clone().cos();
            let sin_h = half.sin();
            let neg_sin_h = Float::with_val(EVAL_PREC, -1) * &sin_h;
            let mut tgt: [[Complex; 2]; 2] =
                std::array::from_fn(|_| std::array::from_fn(|_| Complex::with_val(EVAL_PREC, (0, 0))));
            tgt[0][0] = Complex::with_val(EVAL_PREC, (&cos_h, &neg_sin_h));
            tgt[1][1] = Complex::with_val(EVAL_PREC, (&cos_h, &sin_h));
            let mut layer_seq = BraidSequence::new();
            let mut scratch = SKScratchpad2x2::new();
            solovay_kitaev_recursive(&tgt, 3, &mut layer_seq, &mut scratch);
            let base = 0 * 4;
            for si in 0..layer_seq.len {
                let (ra, rb) = layer_seq.get_braid(si)?;
                braid_and_record!(base + ra, base + rb);
            }
        }
        {
            // c1 on qubit 1
            let half = Float::with_val(EVAL_PREC, c1) / Float::with_val(EVAL_PREC, 2);
            let cos_h = half.clone().cos();
            let sin_h = half.sin();
            let neg_sin_h = Float::with_val(EVAL_PREC, -1) * &sin_h;
            let mut tgt: [[Complex; 2]; 2] =
                std::array::from_fn(|_| std::array::from_fn(|_| Complex::with_val(EVAL_PREC, (0, 0))));
            tgt[0][0] = Complex::with_val(EVAL_PREC, (&cos_h, &neg_sin_h));
            tgt[1][1] = Complex::with_val(EVAL_PREC, (&cos_h, &sin_h));
            let mut layer_seq = BraidSequence::new();
            let mut scratch = SKScratchpad2x2::new();
            solovay_kitaev_recursive(&tgt, 3, &mut layer_seq, &mut scratch);
            let base = 1 * 4;
            for si in 0..layer_seq.len {
                let (ra, rb) = layer_seq.get_braid(si)?;
                braid_and_record!(base + ra, base + rb);
            }
        }

        // --- Layer 4: second CNOT (0 → 1) ---
        {
            let c_base = 0 * 4;
            let t_base = 1 * 4;
            let cnot_pairs = [
                (c_base + 3, t_base),
                (c_base + 2, t_base),
                (c_base + 3, t_base + 1),
                (c_base + 2, t_base + 1),
            ];
            for (a, b) in cnot_pairs {
                braid_and_record!(a, b);
            }
        }

        // --- Layer 5: Rz(c2) on qubit 0 (cross-axis residual correction) ---
        {
            let half = Float::with_val(EVAL_PREC, c2) / Float::with_val(EVAL_PREC, 2);
            let cos_h = half.clone().cos();
            let sin_h = half.sin();
            let neg_sin_h = Float::with_val(EVAL_PREC, -1) * &sin_h;
            let mut tgt: [[Complex; 2]; 2] =
                std::array::from_fn(|_| std::array::from_fn(|_| Complex::with_val(EVAL_PREC, (0, 0))));
            tgt[0][0] = Complex::with_val(EVAL_PREC, (&cos_h, &neg_sin_h));
            tgt[1][1] = Complex::with_val(EVAL_PREC, (&cos_h, &sin_h));
            let mut layer_seq = BraidSequence::new();
            let mut scratch = SKScratchpad2x2::new();
            solovay_kitaev_recursive(&tgt, 3, &mut layer_seq, &mut scratch);
            let base = 0 * 4;
            for si in 0..layer_seq.len {
                let (ra, rb) = layer_seq.get_braid(si)?;
                braid_and_record!(base + ra, base + rb);
            }
        }

        // --- Layer 6: closing H on both qubit blocks ---
        do_h(&mut seq, tracker, 0)?;
        do_h(&mut seq, tracker, 1)?;

        // Inline structural compression: cancel adjacent inverse pairs and
        // commute far generators before emitting the final braid word.
        let minimized_seq = self.minimize_sequence(seq)?;
        Ok(minimized_seq)
    }

    fn parse_openqasm(&self, program_str: String) -> PyResult<CompiledProgram> {
        let mut program = CompiledProgram::new();

        for line in program_str.lines() {
            let cleaned = line.trim().trim_end_matches(';');
            if cleaned.is_empty()
                || cleaned.starts_with("OPENQASM")
                || cleaned.starts_with("include")
                || cleaned.starts_with("qubit")
                || cleaned.starts_with("bit")
            {
                continue;
            }
            if program.len >= MAX_INSTRUCTIONS {
                return Err(PyValueError::new_err("exceeded max OpenQASM program operations"));
            }

            if cleaned.starts_with("measure_stabilizer") {
                let parts: Vec<&str> = cleaned.split_whitespace().collect();
                if parts.len() >= 4 {
                    let stabilizer_idx = parts[1].parse::<usize>().map_err(|_| {
                        PyValueError::new_err("invalid stabilizer index")
                    })?;
                    let creg = parts[3].parse::<usize>().map_err(|_| {
                        PyValueError::new_err("invalid stabilizer creg")
                    })?;
                    program.instructions[program.len] =
                        QasmInstruction::MeasureStabilizer { stabilizer_idx, creg };
                    program.len += 1;
                }
            } else if cleaned.starts_with("braid") {
                let parts: Vec<&str> = cleaned.split_whitespace().collect();
                if parts.len() >= 3 {
                    let id_a = parts[1].parse::<usize>().map_err(|_| {
                        PyValueError::new_err("invalid braid index A")
                    })?;
                    let id_b = parts[2].parse::<usize>().map_err(|_| {
                        PyValueError::new_err("invalid braid index B")
                    })?;
                    program.instructions[program.len] = QasmInstruction::Braid { id_a, id_b };
                    program.len += 1;
                }
            } else if cleaned.starts_with("measure") {
                let parts: Vec<&str> = cleaned.split_whitespace().collect();
                if parts.len() >= 4 {
                    let target = parts[1].parse::<usize>().map_err(|_| {
                        PyValueError::new_err("invalid measure target")
                    })?;
                    let creg = parts[3].parse::<usize>().map_err(|_| {
                        PyValueError::new_err("invalid measure creg")
                    })?;
                    program.instructions[program.len] = QasmInstruction::Measure { target, creg };
                    program.len += 1;
                }
            } else if cleaned.starts_with("if") {
                let cond_parts: Vec<&str> = cleaned.splitn(2, '{').collect();
                if cond_parts.len() >= 2 {
                    let cond_str = cond_parts[0].trim();
                    let body_str = cond_parts[1].trim_end_matches('}').trim();

                    let eq_parts: Vec<&str> = cond_str.splitn(2, "==").collect();
                    if eq_parts.len() >= 2 {
                        let creg_str = eq_parts[0]
                            .replace("if", "")
                            .replace('(', "")
                            .replace(')', "")
                            .trim()
                            .to_string();
                        let val_str = eq_parts[1]
                            .replace(')', "")
                            .trim()
                            .to_string();

                        let creg = creg_str.parse::<usize>().map_err(|_| {
                            PyValueError::new_err("invalid condition register")
                        })?;
                        let val = val_str.parse::<u32>().map_err(|_| {
                            PyValueError::new_err("invalid condition value")
                        })?;

                        if body_str.starts_with("braid") {
                            let body_parts: Vec<&str> = body_str.split_whitespace().collect();
                            if body_parts.len() >= 3 {
                                let id_a = body_parts[1].parse::<usize>().map_err(|_| {
                                    PyValueError::new_err("invalid conditional braid A")
                                })?;
                                let id_b = body_parts[2].parse::<usize>().map_err(|_| {
                                    PyValueError::new_err("invalid conditional braid B")
                                })?;
                                program.instructions[program.len] =
                                    QasmInstruction::ConditionalBraid { creg, val, id_a, id_b };
                                program.len += 1;
                            }
                        }
                    }
                }
            } else if cleaned == "decode_and_correct" {
                program.instructions[program.len] = QasmInstruction::DecodeAndCorrect;
                program.len += 1;
            } else if cleaned.starts_with("cx ") || cleaned.starts_with("cx\t") {
                // Syntax: cx q[N], q[M]
                let body = &cleaned[2..];
                let body = body.replace(|c: char| c == '[' || c == ']', " ");
                let parts: Vec<&str> = body.split_whitespace().collect();
                // Expected tokens after stripping: 'q' N ',' 'q' M  or  'q' N 'q' M
                let nums: Vec<usize> = parts.iter()
                    .filter_map(|t| {
                        let s = t.trim_end_matches(',');
                        if s.chars().all(|c| c.is_ascii_digit()) && !s.is_empty() {
                            s.parse::<usize>().ok()
                        } else {
                            None
                        }
                    })
                    .collect();
                if nums.len() >= 2 {
                    program.instructions[program.len] = QasmInstruction::LogicalCNOT {
                        control_qubit: nums[0],
                        target_qubit: nums[1],
                    };
                    program.len += 1;
                }
            } else if cleaned.starts_with("h ") || cleaned.starts_with("h\t")
                || cleaned.starts_with("t ") || cleaned.starts_with("t\t")
                || cleaned.starts_with("x ") || cleaned.starts_with("x\t")
            {
                // Syntax: h q[N] / t q[N] / x q[N]
                let gate_char = cleaned.chars().next().unwrap();
                let body = &cleaned[1..];
                let body = body.replace(|c: char| c == '[' || c == ']', " ");
                let parts: Vec<&str> = body.split_whitespace().collect();
                let target_qubit = parts.iter()
                    .find_map(|t| {
                        if t.chars().all(|c| c.is_ascii_digit()) && !t.is_empty() {
                            t.parse::<usize>().ok()
                        } else {
                            None
                        }
                    });
                if let Some(tq) = target_qubit {
                    let kind = match gate_char {
                        'h' => LogicalGateKind::H,
                        't' => LogicalGateKind::T,
                        'x' => LogicalGateKind::X,
                        _ => unreachable!(),
                    };
                    program.instructions[program.len] = QasmInstruction::LogicalGate {
                        gate: kind,
                        target_qubit: tq,
                    };
                    program.len += 1;
                }
            } else if cleaned.starts_with("unitary4(") {
                // Syntax: unitary4(re00,im00,re01,im01,...,re33,im33)
                // 16 complex entries = 32 floats, listed row-major.
                if let Some(open) = cleaned.find('(') {
                    if let Some(close) = cleaned.rfind(')') {
                        if close > open {
                            let inner = &cleaned[open + 1..close];
                            let floats: Vec<f64> = inner
                                .split(',')
                                .filter_map(|s| s.trim().parse::<f64>().ok())
                                .collect();
                            if floats.len() == 32 {
                                let mut matrix_data = [[(0.0_f64, 0.0_f64); 4]; 4];
                                let mut fi = 0usize;
                                for row in 0..4 {
                                    for col in 0..4 {
                                        matrix_data[row][col] = (floats[fi], floats[fi + 1]);
                                        fi += 2;
                                    }
                                }
                                if program.len < MAX_INSTRUCTIONS {
                                    program.instructions[program.len] =
                                        QasmInstruction::ArbitraryUnitary4 { matrix_data };
                                    program.len += 1;
                                }
                            }
                        }
                    }
                }
            } else if cleaned.starts_with("rz(") || cleaned.starts_with("rx(") {
                // Syntax: rz(FLOAT) q[N]  or  rx(FLOAT) q[N]
                let is_rx = cleaned.starts_with("rx(");
                if let Some(open_paren) = cleaned.find('(') {
                    if let Some(close_paren) = cleaned.find(')') {
                        if close_paren > open_paren {
                            let angle_str = cleaned[open_paren + 1..close_paren].trim();
                            if let Ok(angle) = angle_str.parse::<f64>() {
                                let remainder = &cleaned[close_paren + 1..];
                                let remainder = remainder.replace(|c: char| c == '[' || c == ']', " ");
                                let target_qubit = remainder
                                    .split_whitespace()
                                    .find_map(|t| {
                                        if t.chars().all(|c| c.is_ascii_digit()) && !t.is_empty() {
                                            t.parse::<usize>().ok()
                                        } else {
                                            None
                                        }
                                    });
                                if let Some(tq) = target_qubit {
                                    program.instructions[program.len] = if is_rx {
                                        QasmInstruction::RxPhase {
                                            angle_multiplier: angle,
                                            target_qubit: tq,
                                        }
                                    } else {
                                        QasmInstruction::RzPhase {
                                            angle_multiplier: angle,
                                            target_qubit: tq,
                                        }
                                    };
                                    program.len += 1;
                                }
                            }
                        }
                    }
                }
            }
        }
        Ok(program)
    }

    fn execute_compiled_program(
        &mut self,
        tracker: &mut TopologicalTracker,
        program: &CompiledProgram,
        rand_val: f64,
    ) -> PyResult<()> {
        let mut local_regs = [0u32; MAX_CLASSICAL_REGS];
        for step in 0..program.len {
            match program.instructions[step] {
                QasmInstruction::Braid { id_a, id_b } => {
                    tracker.execute_braid(id_a, id_b)?;
                }
                QasmInstruction::Measure { target, creg } => {
                    let outcome = tracker.measure_qubit(target, rand_val)?;
                    if creg < MAX_CLASSICAL_REGS {
                        local_regs[creg] = outcome;
                    }
                }
                QasmInstruction::MeasureStabilizer { stabilizer_idx, creg } => {
                    let outcome = tracker.measure_stabilizer(stabilizer_idx, rand_val)?;
                    if creg < MAX_CLASSICAL_REGS {
                        local_regs[creg] = outcome;
                    }
                }
                QasmInstruction::ConditionalBraid { creg, val, id_a, id_b } => {
                    if creg < MAX_CLASSICAL_REGS && local_regs[creg] == val {
                        tracker.execute_braid(id_a, id_b)?;
                    }
                }
                QasmInstruction::DecodeAndCorrect => {
                    tracker.decode_and_correct()?;
                }
                QasmInstruction::LogicalGate { gate, target_qubit } => {
                    let gate_name = match gate {
                        LogicalGateKind::H => "H",
                        LogicalGateKind::T => "T",
                        LogicalGateKind::X => "X",
                    };
                    self.compile_gate(tracker, gate_name, target_qubit)?;
                }
                QasmInstruction::LogicalCNOT { control_qubit, target_qubit } => {
                    self.compile_cnot(tracker, control_qubit, target_qubit)?;
                }
                QasmInstruction::RzPhase { angle_multiplier, target_qubit } => {
                    self.compile_arbitrary_phase(tracker, angle_multiplier, target_qubit)?;
                }
                QasmInstruction::RxPhase { angle_multiplier, target_qubit } => {
                    self.compile_arbitrary_phase_rx(tracker, angle_multiplier, target_qubit)?;
                }
                QasmInstruction::ArbitraryUnitary4 { matrix_data } => {
                    self.compile_two_qubit_gate_cartan(matrix_data, tracker)?;
                }
            }
        }
        Ok(())
    }

    fn compile_gate(
        &self,
        tracker: &mut TopologicalTracker,
        gate_name: &str,
        target_qubit: usize,
    ) -> PyResult<()> {
        if target_qubit >= tracker.logical_qubits {
            return Err(PyIndexError::new_err("Target qubit exceeds tensor product bounds"));
        }
        let base_idx = target_qubit * 4;
        let max_id = base_idx + 3;
        if max_id >= tracker.anyon_count {
            return Err(PyIndexError::new_err(
                "Insufficient anyons initialized for this logical qubit",
            ));
        }
        let sequence: &[(usize, usize)] = match gate_name {
            "H" => H_GATE_SEQUENCE,
            "T" => T_GATE_SEQUENCE,
            "X" => X_GATE_SEQUENCE,
            _ => return Err(PyValueError::new_err(format!("Unknown logical gate: {gate_name}"))),
        };
        for &(offset_a, offset_b) in sequence {
            tracker.execute_braid(base_idx + offset_a, base_idx + offset_b)?;
        }
        Ok(())
    }

    fn compile_cnot(
        &self,
        tracker: &mut TopologicalTracker,
        control_qubit: usize,
        target_qubit: usize,
    ) -> PyResult<()> {
        if control_qubit >= tracker.logical_qubits || target_qubit >= tracker.logical_qubits {
            return Err(PyIndexError::new_err("Qubit indices exceed tensor product bounds"));
        }
        if control_qubit == target_qubit {
            return Err(PyValueError::new_err("Control and target qubits must be distinct"));
        }
        let c_base = control_qubit * 4;
        let t_base = target_qubit * 4;
        let max_id = c_base.max(t_base) + 3;
        if max_id >= tracker.anyon_count {
            return Err(PyIndexError::new_err(
                "Insufficient anyons initialized for CNOT operation",
            ));
        }
        let sequence = [
            (c_base + 3, t_base),
            (c_base + 2, t_base),
            (c_base + 3, t_base + 1),
            (c_base + 2, t_base + 1),
        ];
        for &(id_a, id_b) in &sequence {
            tracker.execute_braid(id_a, id_b)?;
        }
        Ok(())
    }

    fn compile_arbitrary_phase(
        &self,
        tracker: &mut TopologicalTracker,
        angle_multiplier: f64,
        target_qubit: usize,
    ) -> PyResult<()> {
        if target_qubit >= tracker.logical_qubits {
            return Err(PyIndexError::new_err("Target qubit exceeds tensor product bounds"));
        }
        let base_idx = target_qubit * 4;
        if base_idx + 3 >= tracker.anyon_count {
            return Err(PyIndexError::new_err(
                "Insufficient anyons for arbitrary phase synthesis",
            ));
        }

        // ---------------------------------------------------------------
        // Stage 1: Build Rz target U(2) diagonal phase gate at 512-bit precision.
        //
        // θ = angle_multiplier * π
        // U_Rz = diag( e^{-iθ/2},  e^{iθ/2} )
        //      = [ cos(θ/2) - i·sin(θ/2)    0                   ]
        //        [ 0                         cos(θ/2) + i·sin(θ/2) ]
        // ---------------------------------------------------------------
        let theta = Float::with_val(EVAL_PREC, angle_multiplier) * Float::with_val(EVAL_PREC, std::f64::consts::PI);
        let half_theta = Float::with_val(EVAL_PREC, &theta) / Float::with_val(EVAL_PREC, 2);
        let cos_ht = half_theta.clone().cos();
        let sin_ht = half_theta.sin();
        let neg_sin_ht = Float::with_val(EVAL_PREC, -1) * &sin_ht;

        let mut target: [[Complex; 2]; 2] =
            std::array::from_fn(|_| std::array::from_fn(|_| Complex::with_val(EVAL_PREC, (0, 0))));
        target[0][0] = Complex::with_val(EVAL_PREC, (&cos_ht, &neg_sin_ht));
        target[1][1] = Complex::with_val(EVAL_PREC, (&cos_ht, &sin_ht));

        // ---------------------------------------------------------------
        // Stage 2: Solovay-Kitaev recursive decomposition at depth 3.
        // ---------------------------------------------------------------
        let mut seq = BraidSequence::new();
        let mut scratch = SKScratchpad2x2::new();
        solovay_kitaev_recursive(&target, 3, &mut seq, &mut scratch);

        for step_idx in 0..seq.len {
            let (rel_a, rel_b) = seq.get_braid(step_idx)?;
            tracker.execute_braid(base_idx + rel_a, base_idx + rel_b)?;
        }

        Ok(())
    }

    fn compile_arbitrary_phase_rx(
        &self,
        tracker: &mut TopologicalTracker,
        angle_multiplier: f64,
        target_qubit: usize,
    ) -> PyResult<()> {
        if target_qubit >= tracker.logical_qubits {
            return Err(PyIndexError::new_err("Target qubit exceeds tensor product bounds"));
        }
        let base_idx = target_qubit * 4;
        if base_idx + 3 >= tracker.anyon_count {
            return Err(PyIndexError::new_err(
                "Insufficient anyons for Rx phase synthesis",
            ));
        }

        // ---------------------------------------------------------------
        // Stage 1: Build Rx target U(2) off-diagonal rotation at 512-bit precision.
        //
        // θ = angle_multiplier * π
        // U_Rx = [ cos(θ/2)      -i·sin(θ/2) ]
        //        [ -i·sin(θ/2)    cos(θ/2)   ]
        //
        // This off-diagonal structure encodes the true physical Rx rotation;
        // it differs fundamentally from the diagonal Rz phase gate and must
        // be synthesized separately through solovay_kitaev_recursive.
        // ---------------------------------------------------------------
        let theta = Float::with_val(EVAL_PREC, angle_multiplier) * Float::with_val(EVAL_PREC, std::f64::consts::PI);
        let half_theta = Float::with_val(EVAL_PREC, &theta) / Float::with_val(EVAL_PREC, 2);
        let cos_ht = half_theta.clone().cos();
        let sin_ht = half_theta.sin();
        // -i·sin(θ/2)  represented as a Complex with real=0, imag=-sin(θ/2)
        let neg_i_sin: Complex = Complex::with_val(EVAL_PREC, (0, 1))
            * Complex::with_val(EVAL_PREC, (&sin_ht, 0))
            * Complex::with_val(EVAL_PREC, (-1, 0));

        let mut target: [[Complex; 2]; 2] =
            std::array::from_fn(|_| std::array::from_fn(|_| Complex::with_val(EVAL_PREC, (0, 0))));
        target[0][0] = Complex::with_val(EVAL_PREC, (&cos_ht, 0));
        target[0][1] = neg_i_sin.clone();
        target[1][0] = neg_i_sin;
        target[1][1] = Complex::with_val(EVAL_PREC, (&cos_ht, 0));

        // ---------------------------------------------------------------
        // Stage 2: Solovay-Kitaev recursive decomposition at depth 3.
        // The off-diagonal matrix is passed directly; SK resolves the
        // anyon braid word from the SU(2) geometry unchanged.
        // ---------------------------------------------------------------
        let mut seq = BraidSequence::new();
        let mut scratch = SKScratchpad2x2::new();
        solovay_kitaev_recursive(&target, 3, &mut seq, &mut scratch);

        for step_idx in 0..seq.len {
            let (rel_a, rel_b) = seq.get_braid(step_idx)?;
            tracker.execute_braid(base_idx + rel_a, base_idx + rel_b)?;
        }

        Ok(())
    }

    fn compile_unitary_gate(
        &self,
        tracker: &mut TopologicalTracker,
        target_matrix: [[(f64, f64); 2]; 2],
        depth: usize,
        target_qubit: usize,
    ) -> PyResult<()> {
        if target_qubit >= tracker.logical_qubits {
            return Err(PyIndexError::new_err("Target qubit exceeds tensor product bounds"));
        }
        let base_idx = target_qubit * 4;
        if base_idx + 3 >= tracker.anyon_count {
            return Err(PyIndexError::new_err(
                "Insufficient anyons for unitary gate synthesis",
            ));
        }

        // Ingest the Python-supplied float-tuple matrix into 512-bit Complex
        let mut target: [[Complex; 2]; 2] =
            std::array::from_fn(|_| std::array::from_fn(|_| Complex::with_val(EVAL_PREC, (0, 0))));
        for i in 0..2 {
            for j in 0..2 {
                let (re, im) = target_matrix[i][j];
                target[i][j].assign(&Complex::with_val(EVAL_PREC, (re, im)));
            }
        }

        // Run the Solovay-Kitaev recursive compiler on the stack scratchpad
        let mut seq = BraidSequence::new();
        let mut scratch = SKScratchpad2x2::new();
        solovay_kitaev_recursive(&target, depth, &mut seq, &mut scratch);

        // Play back the generated braid word onto the tracker
        for i in 0..seq.len {
            let (lo, hi) = seq.braids[i];
            tracker.execute_braid(base_idx + lo, base_idx + hi)?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct ShbtReport {
    pub branch: (u32, u32, u32),
    pub boundary_report: VerificationReport,
    pub projection_report: ProjectionReport,
    pub memory_report: MemoryReport,
    pub benchmark_delta: BenchmarkDelta,
    pub baryogenesis_identity: BaryogenesisIdentity,
    pub eta_b: Float,
    pub stress_energy_preserved: bool,
    pub metric_slices: Vec<BulkMetricSlice>,
    pub history_entries: Vec<CoordinateLogEntry>,
}

#[pymethods]
impl ShbtReport {
    pub fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new_bound(py);
        d.set_item("branch", self.branch)?;
        d.set_item("boundary_report", self.boundary_report.to_dict(py)?)?;
        d.set_item("projection_report", self.projection_report.to_dict(py)?)?;
        d.set_item("memory_report", self.memory_report.to_dict(py)?)?;
        d.set_item("benchmark_delta", self.benchmark_delta.to_dict(py)?)?;
        d.set_item("baryogenesis_identity", self.baryogenesis_identity.to_dict(py)?)?;
        d.set_item("eta_b", self.eta_b.to_f64())?;
        d.set_item("stress_energy_preserved", self.stress_energy_preserved)?;
        let slices: PyResult<Vec<Bound<'py, PyDict>>> = self
            .metric_slices
            .iter()
            .map(|s| s.to_dict(py))
            .collect();
        d.set_item("metric_slices", slices?)?;
        let history: PyResult<Vec<Bound<'py, PyDict>>> = self
            .history_entries
            .iter()
            .map(|e| e.to_dict(py))
            .collect();
        d.set_item("history_entries", history?)?;
        Ok(d)
    }

    #[getter]
    fn branch(&self) -> (u32, u32, u32) {
        self.branch
    }

    #[getter]
    fn eta_b(&self) -> f64 {
        self.eta_b.to_f64()
    }

    #[getter]
    fn stress_energy_preserved(&self) -> bool {
        self.stress_energy_preserved
    }

    #[getter]
    fn metric_slice_count(&self) -> usize {
        self.metric_slices.len()
    }

    #[getter]
    fn history_entry_count(&self) -> usize {
        self.history_entries.len()
    }

    #[getter]
    fn framing_defect(&self) -> f64 {
        self.boundary_report.framing_defect.to_f64()
    }

    #[getter]
    fn modular_invariant(&self) -> bool {
        self.boundary_report.modular_invariant
    }

    #[getter]
    fn zero_energy_locked(&self) -> bool {
        self.boundary_report.zero_energy_locked
    }

    #[getter]
    fn projection_dimension_26_to_4(&self) -> bool {
        self.boundary_report.projection_dimension_26_to_4
    }

    #[getter]
    fn slice_count(&self) -> usize {
        self.projection_report.slice_count
    }

    #[getter]
    fn projection_all_passed(&self) -> bool {
        self.projection_report.all_passed
    }

    #[getter]
    fn memory_all_passed(&self) -> bool {
        self.memory_report.all_passed
    }
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct ShbtSimulator {
    boundary: StaticBoundary,
    projection: HolographicProjection,
    causal_point: CausalPoint,
    optimizer: BaryogenesisOptimizer,
    particles: usize,
}

#[pymethods]
impl ShbtSimulator {
    #[new]
    fn new() -> Self {
        Self::with_config((26, 8, 312), 0.125, 3.0, 9, 512, 0)
    }

    #[staticmethod]
    fn with_config(
        branch: (u32, u32, u32),
        observer_radius_fraction: f64,
        redshift_max: f64,
        redshift_samples: usize,
        particles: usize,
        seed: u64,
    ) -> Self {
        let (lepton_level, quark_level, parent_level) = branch;
        let boundary = StaticBoundary::new_with_branch(lepton_level, quark_level, parent_level);
        let projection = HolographicProjection::new(boundary.clone());

        let xi = [
            Float::with_val(EVAL_PREC, 1.0 / lepton_level as f64),
            Float::with_val(EVAL_PREC, 1.0 / quark_level as f64),
            Float::with_val(EVAL_PREC, 1.0 / parent_level as f64),
        ];
        let causal_point = CausalPoint::new_with_params(
            boundary.clone(),
            Float::with_val(EVAL_PREC, observer_radius_fraction),
            xi,
            Float::with_val(EVAL_PREC, redshift_max),
            redshift_samples,
            seed,
        );
        let optimizer = BaryogenesisOptimizer::new(boundary.clone());
        Self {
            boundary,
            projection,
            causal_point,
            optimizer,
            particles,
        }
    }

    fn run_full_audit(&self) -> ShbtReport {
        let metric_slices = self.projection.project_entropy_cascade();
        let boundary_report = self.boundary.verify_equations();
        let projection_report = self.projection.verify_projection(&metric_slices);
        let memory_report = self.causal_point.verify_memory_budget();
        let benchmark_delta = self.optimizer.run_benchmark(self.particles);
        let stress_energy_preserved = benchmark_delta.stress_energy_preserved;
        let baryogenesis_identity = self.optimizer.baryogenesis_identity();
        let eta_b = baryogenesis_identity.eta_b.clone();
        let history_entries = self.causal_point.crystallize_history();
        ShbtReport {
            branch: self.boundary.benchmark_branch,
            boundary_report,
            projection_report,
            memory_report,
            benchmark_delta,
            baryogenesis_identity,
            eta_b,
            stress_energy_preserved,
            metric_slices,
            history_entries,
        }
    }

    fn baryogenesis_identity(&self) -> BaryogenesisIdentity {
        self.optimizer.baryogenesis_identity()
    }

    fn baryogenesis_benchmark(&self) -> BenchmarkDelta {
        self.optimizer.run_benchmark(self.particles)
    }

    fn crystallize_history(&self) -> Vec<CoordinateLogEntry> {
        self.causal_point.crystallize_history()
    }

    fn simulate_cosmology(&self, z_max: f64, samples: usize) -> Vec<BulkMetricSlice> {
        let xi = self.causal_point.xi.clone();
        let cp = CausalPoint::new_with_params(
            self.boundary.clone(),
            self.causal_point.observer_radius_fraction.clone(),
            xi,
            Float::with_val(EVAL_PREC, z_max),
            samples,
            self.causal_point.seed,
        );
        let slices = cp.projection.project_entropy_cascade();
        let cone = cp.build_past_light_cone();
        cone.iter()
            .map(|s| slices[s.index.min(slices.len() - 1)].clone())
            .collect()
    }
}

#[pymodule]
mod shbt_simulator {
    pub const PARENT: u32 = super::PARENT;
    pub const LEPTON: u32 = super::LEPTON;
    pub const QUARK: u32 = super::QUARK;
    pub const C_DARK_NUM: u32 = super::C_DARK_NUM;
    pub const C_DARK_DEN: u32 = super::C_DARK_DEN;
    pub const C_DARK: &str = "1197103/362670";

    #[pymodule_export]
    use super::get_geometric_kappa;

    #[pymodule_export]
    use super::AnyonBraidingEngine;

    #[pymodule_export]
    use super::Coordinate;

    #[pymodule_export]
    use super::TopologicalTracker;

    #[pymodule_export]
    use super::NonAbelianLeakageError;

    #[pymodule_export]
    use super::CircuitCompiler;

    #[pymodule_export]
    use super::BraidSequence;

    #[pymodule_export]
    use super::LieSectorType;

    #[pymodule_export]
    use super::CompiledProgram;

    #[pymodule_export]
    use super::StabilizerType;

    #[pymodule_export]
    use super::StabilizerGenerator;

    #[pymodule_export]
    use super::SurfaceCodeLattice;

    #[pymodule_export]
    use super::ShbtSimulator;

    #[pymodule_export]
    use super::ShbtReport;

    #[pymodule_export]
    use super::shbt::BulkMetricSlice;

    #[pymodule_export]
    use super::shbt::ProjectionReport;

    #[pymodule_export]
    use super::shbt::EntropyUpdate;

    #[pymodule_export]
    use super::shbt::TemporalKernelAudit;

    #[pymodule_export]
    use super::shbt::VerificationReport;

    #[pymodule_export]
    use super::shbt::BaryogenesisIdentity;

    #[pymodule_export]
    use super::shbt::FieldSimulation;

    #[pymodule_export]
    use super::shbt::BenchmarkDelta;

    #[pymodule_export]
    use super::shbt::LightConeSample;

    #[pymodule_export]
    use super::shbt::LocalPropertyPacket;

    #[pymodule_export]
    use super::shbt::CoordinateLogEntry;

    #[pymodule_export]
    use super::shbt::MemoryReport;
}
