# Static Holographic Boundary Theory (SHBT) — Precision Boundary Simulator & Cosmology Engine

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22844471.svg)](https://doi.org/10.5281/zenodo.22844471)
[![Release](https://img.shields.io/badge/release-v2.0.0-blue.svg)](https://github.com/sys1own/shbt-precision/releases)
[![Rust](https://img.shields.io/badge/rust-1.80+-blue.svg)](https://www.rust-lang.org/)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**`shbt-precision`** is the definitive computational mathematics core and foundational physics authority for the eight-repository SHBT ecosystem. It provides 512-bit arbitrary-precision proofs (via `rug`/MPFR, 492-bit mantissa) and symplectic Yoshida-6 integrators verifying boundary Conformal Field Theory (CFT) projections, baryogenesis, dark-matter topological ghosts, first-principles inflation, and precision cosmology.

The theory is fully documented in the accompanying publications [`main.pdf`](main.pdf) and [`supplementary.pdf`](supplementary.pdf). This repository is the **executable proof** of the theory: every claim, equation, and numerical prediction in the papers is audited by the Rust/Python code herein.

---

## Table of Contents

- [Complete Repository Topology](#complete-repository-topology)
- [Core Physics & Invariant Verification Matrix](#core-physics--invariant-verification-matrix)
- [Two-Tier Simulation Engine Architecture](#two-tier-simulation-engine-architecture)
- [CLI & Execution Reference](#cli--execution-reference)
- [Section 12 Verification & Traceability Matrices](#section-12-verification--traceability-matrices)
- [Build, Packaging & Document Compilation](#build-packaging--document-compilation)
- [Sibling Repository Crosswalk](#sibling-repository-crosswalk)

---

## Complete Repository Topology

```text
.
├── Cargo.toml                # Rust workspace manifest (PyO3, rug/MPFR, ndarray, rayon)
├── Cargo.lock                # Deterministic dependency lockfile
├── pyproject.toml            # PEP 517/621 build definition for maturin wheel generation
├── Makefile                  # Automated build pipeline (LaTeX, Cargo, Maturin, verification)
├── requirements.txt          # Python runtime dependencies (scipy, matplotlib, h5py, pandas)
├── config.default.yaml       # Default simulation configuration template
├── LICENSE                   # MIT license
├── src/
│   ├── lib.rs                # PyO3 module bindings & high-precision C-ABI export
│   └── shbt/
│       ├── mod.rs            # SHBT module root (re-exports, shared types)
│       ├── boundary.rs       # StaticBoundary (modular data, framing defect, dark pairing)
│       ├── entropy_flow.rs   # HolographicProjection (Fefferman-Graham RG flow, bulk metric)
│       ├── baryogenesis.rs   # BaryogenesisOptimizer (topological anti-baryon decoupling)
│       ├── causal_point.rs   # CausalPoint (observer memory, light-cone history crystallization)
│       ├── provenance.rs     # Run provenance & reproducibility metadata capture
│       ├── stability_audit.rs# Numerical stability audit (condition numbers, tolerances)
│       └── data/
│           └── chronometer_data.json  # Cosmic-chronometer H(z) dataset (32 points)
├── boltzmann_shbt.py         # Full Boltzmann pipeline, non-Gaussianity templates, CMB & P(k)
├── precision_cosmology.py    # Section 9 precision cosmology audit, 7-param MCMC, H0 tension
├── shbt_simulate.py          # Unified CLI/API orchestration engine & calorimetry OLS/MLE
├── main.tex                  # Primary publication LaTeX manuscript (5-pillar architecture)
├── main.pdf                  # Compiled primary publication PDF
├── supplementary.tex         # Standalone supplementary publication LaTeX manuscript
├── supplementary.pdf         # Compiled standalone supplementary monograph PDF
├── paper_references.md       # Traceability crosswalk mapping paper sections to source methods
├── examples/
│   ├── run_audit.py          # Minimal foundation verification script
│   └── shbt_notebook.ipynb   # Interactive analysis and visualization notebook
└── tests/
    ├── test_shbt.rs          # Rust integration test suite (modular closure, symplectic norms)
    └── test_simulator.py     # Python integration tests (pytest)
```

---

## Core Physics & Invariant Verification Matrix

The platform proves the following foundational physics invariants:

| Invariant | Result | Source |
| :--- | :--- | :--- |
| **Canonical Affine Algebra** | WZW branch (*k*<sub>*ℓ*</sub>, *k*<sub>*q*</sub>, *K*) = (26, 8, 312) for SU(2)<sub>26</sub>, SU(3)<sub>8</sub>, SO(10)<sub>312</sub>; exact framing cancellation Δ<sub>fr</sub> = 0 and vanishing stress-energy trace *E*<sub>*μν*</sub> = 0 | `src/shbt/boundary.rs` |
| **Topological Baryogenesis** | Non-perturbative de-rendering generating η<sub>*B*</sub> ≃ 6.45 × 10<sup>−10</sup> with no arbitrary parameters | `src/shbt/baryogenesis.rs` |
| **Microscopic Dark CFT** | Invariant rational capacity partitioning η<sub>A</sub> = 10/33 (visible), η<sub>D</sub> = 23/33 (dark); dark matter resolved as a topological gravitational ghost, *c*<sub>darkresidual</sub> ≃ 2.3008 | `src/shbt/boundary.rs` |
| **Holographic Dark Energy** | First-principles cosmological scale Λ<sub>holo</sub> ≃ 1.09 × 10<sup>−52</sup> m<sup>−2</sup> from total boundary bit budget *N* ≃ 3.31 × 10<sup>122</sup> | `precision_cosmology.py` |
| **Causal Point & Landauer Limit** | Observer-state crystallization bounded by local operational entropy *C*<sub>op</sub> ≤ *C*<sub>local</sub>; Landauer address dissipation limit *P*<sub>debt</sub> = 906.00 kW | `src/shbt/causal_point.rs` |

Boundary-isometry constraint consumed by downstream hardware:

> ‖*W*<sup>†</sup>*W* − *P*<sub>code</sub>‖<sub>op</sub> ≤ 3.430 × 10<sup>−3</sup>

---

## Two-Tier Simulation Engine Architecture

### Tier 1 — Rust/PyO3 High-Precision Foundation (`src/`, `shbt_simulator`)

- High-precision arithmetic via `rug`/MPFR: 512-bit floats, 492-bit mantissas, resolving 1/*N* ≃ 10<sup>−122</sup> against unit values.
- Symplectic Yoshida-6 integrators and zero-allocation hot loops.
- Core types: `StaticBoundary`, `HolographicProjection`, `BulkMetricSlice`, `BaryogenesisOptimizer`, `CausalPoint`, `AnomalyClosureError`.
- Legacy low-level engines reused by the SHBT modules: `AnyonBraidingEngine` (SU(2), SU(3), SO(10) braid matrices), `TopologicalTracker` (anyon worldlines, fusion, stabiliser checks), `CircuitCompiler` (Solovay-Kitaev, OpenQASM parsing).

### Tier 2 — Python Precision Cosmology & Boltzmann Pipeline

- `boltzmann_shbt.py`: first-principles inflation dynamics and perturbation spectra; primordial non-Gaussianity templates (local, equilateral, orthogonal *f*<sub>NL</sub> and *g*<sub>NL</sub> bispectra/trispectra); Boltzmann hierarchy integration producing scalar temperature (*C*<sub>*ℓ*</sub><sup>TT</sup>), polarization (*C*<sub>*ℓ*</sub><sup>EE</sup>, *C*<sub>*ℓ*</sub><sup>TE</sup>), matter power *P*(*k*, *z*), and tensor B-modes (*C*<sub>*ℓ*</sub><sup>BB</sup>).
- `precision_cosmology.py`: Section 9 precision-cosmology audit, 7-parameter MCMC, *H*<sub>0</sub>-tension quantification, cosmic-chronometer covariance, and the sub-10 mK Landauer calorimetry experiment (`simulate_calorimetry_experiment`).
- `shbt_simulate.py`: unified CLI/API orchestrator; evaluates address sweeps and heat-dissipation hypotheses via OLS/MLE regression and emits the `shbt_run_*` data products.

---

## CLI & Execution Reference

```bash
# Foundation audit (modular closure, projection, memory, baryogenesis)
python shbt_simulate.py --mode audit

# Full unified simulation (foundation + precision cosmology)
python shbt_simulate.py --mode all --output result.json

# Boltzmann CMB and matter spectra generation
python boltzmann_shbt.py                      # runs the full pipeline, prints CMB spectra
python boltzmann_shbt.py --run-tests          # embedded Boltzmann/chronometer unit tests

# Precision cosmology report and parameter sweeps
python precision_cosmology.py --json
python precision_cosmology.py --run-tests
python shbt_simulate.py --mode cosmology --sweep sweep.json --plot --output sweep_result.json

# Sub-10 mK Landauer calorimetry audit (writes shbt_run_calorimetry_sim.csv)
python shbt_simulate.py --mode all --output result.json
# or programmatically:
python -c "import precision_cosmology as pc; pc.simulate_calorimetry_experiment(n_pulses=1000000)"
```

Additional modes: `baryogenesis` (topological asymmetry benchmark), `history` (Causal-Point observer crystallization), `cosmology-test`. Exports support `--format json|csv|hdf5`, `--plot`, `--sweep`, `--config` (YAML/JSON), `--seed`, and structured `--log-format json` logging.

---

## Section 12 Verification & Traceability Matrices

### Code & Equation Traceability Matrix (Table 38)

| Publication claim | Implementation | Output artifact |
| :--- | :--- | :--- |
| Bispectrum & trispectrum templates | `boltzmann_shbt.py` | `result.json: precision_pipeline.non_gaussianity` |
| GET capacity & cutoff | `src/shbt/causal_point.rs` (`CausalPoint.crystallize_history`) | `result.json: foundation_audit.memory_report` |
| Landauer calorimetry regression | `shbt_simulate.py` / `precision_cosmology.py` | `shbt_run_calorimetry_sim.csv` |

### Generated-Artifact Data Product Crosswalk (Table 39)

| Artifact | Columns | Content |
| :--- | :--- | :--- |
| `shbt_run_cmb_cls.csv` | `Dl_TT`, `Dl_EE`, `Dl_TE` | Scalar temperature & polarization spectra |
| `shbt_run_matter_pk.csv` | `Pk_z0`, `Pk_z05`, `Pk_z1` | Matter power spectrum at *z* = 0, 0.5, 1 |
| `shbt_run_tensor_cls.csv` | `Cl_BB`, `Dl_BB`, `Cl_TT_tensor` | Primordial tensor B-modes |
| `shbt_run_calorimetry_sim.csv` | `k_bits`, `R_addresses`, `Q_H0_zJ`, `Q_H1_zJ`, `Q_noise_zJ` | Sub-10 mK calorimetry simulation |
| `result.json` | nested ledger | Master simulation result tree |

### Software Interface Reproduction Contract (Table 40)

| Symbol | Kind |
| :--- | :--- |
| `shbt_simulator` | Master PyO3 module |
| `StaticBoundary` | Boundary CFT modular data, framing defect, dark ledger |
| `ShbtSimulator` | Orchestrating runtime (`run_full_audit`) |
| `HolographicProjection` | RG flow → `BulkMetricSlice` |
| `CausalPoint` | Observer memory & history crystallization |
| `AnomalyClosureError` | Algebraic anomaly failure type |

See [`paper_references.md`](paper_references.md) for the complete paper-section → method crosswalk.

---

## Build, Packaging & Document Compilation

```bash
# Prerequisites: Rust 1.80+ (rustup), Python 3.8+, pip install maturin
pip install -r requirements.txt

# Build Rust library and PyO3 bindings
cargo build --release
maturin develop --release          # installs shbt_simulator into the active env
maturin build --release            # produces target/wheels/shbt_simulator-*.whl

# Unit and integration tests
cargo test --release               # Rust suite (tests/test_shbt.rs)
pytest tests/                      # Python suite (tests/test_simulator.py)

# Compile publication documents
make pdf                           # pdflatex main.tex (primary manuscript)
latexmk -pdf -jobname=main main.tex
latexmk -pdf -jobname=supplementary supplementary.tex

# Whole pipeline: build + test + audit + pdf
make all
```

---

## Sibling Repository Crosswalk

The SHBT program is a federated ecosystem of eight configuration-controlled repositories. `shbt-precision` is the mathematical core; downstream repositories consume its audited invariants.

| Repository | Primary Domain | Interface to `shbt-precision` |
| :--- | :--- | :--- |
| [**`sys1own/shbt-power`**](https://github.com/sys1own/shbt-power) | Commercial fusion power plant digital twin | Consumes bremsstrahlung suppression factor *S* = 100/1089 |
| [**`sys1own/shbt-cf`**](https://github.com/sys1own/shbt-cf) | Solid-state cold fusion LANR starter grid | 1,800 modules, 999.054 kW net DC, 3D multiphase helium cooling |
| [**`sys1own/shbt-qc`**](https://github.com/sys1own/shbt-qc) | Photonic quantum processor | Bare-metal C11 `shbt-os` microkernel, SHBT-MMIO-1 register map |
| [**`sys1own/shbt-ghost`**](https://github.com/sys1own/shbt-ghost) | Optical interlocks & metric control | Sub-2.5 ns PCSS crowbars, 94.20% SiC shunts, 3+1 CCZ4/ADM stabilization |
| [**`sys1own/shbt-exotic`**](https://github.com/sys1own/shbt-exotic) | Boundary CFT & spacetime engineering | Heegaard-Floer relabeling, dark ledger partition η<sub>D</sub> = 23/33 |
| [**`sys1own/shbt-recon`**](https://github.com/sys1own/shbt-recon) | Macroscopic Stinespring state translocation | 128-byte dual-cacheline C-ABI DMA streaming |
| [**`sys1own/shbt-sglt`**](https://github.com/sys1own/shbt-sglt) | Synthetic gravitational lensing telescope array at SE-L2 | 2PN beam optics, sub-SQL squeezed heterodyne metrology |
| **`sys1own/shbt-precision`** (this repo) | Computational mathematics core | 512-bit MPFR framework, cosmological closure, verification ledgers |

---

## License

MIT — see [`LICENSE`](LICENSE).
