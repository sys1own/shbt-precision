# Static Holographic Boundary Theory (SHBT) — Precision Boundary Simulator & Cosmology Engine

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22844471.svg)](https://doi.org/10.5281/zenodo.22844471)
[![Release](https://img.shields.io/badge/release-v2.0.0-blue.svg)](https://github.com/sys1own/shbt-precision/releases)
[![Rust](https://img.shields.io/badge/rust-1.80+-blue.svg)](https://www.rust-lang.org/)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**`shbt-precision`** is the definitive computational mathematics core and foundational physics authority for the nine-repository SHBT ecosystem. It provides 512-bit arbitrary-precision proofs (via `rug`/MPFR, 492-bit mantissa) and symplectic Yoshida-6 integrators verifying boundary Conformal Field Theory (CFT) projections, baryogenesis, dark-matter topological ghosts, first-principles inflation, and precision cosmology.

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
│       ├── causal_point.rs   # CausalPoint (observer memory, crystallization, succession lifecycle)
│       ├── cosmology.rs      # ShbtUniverse (branch cosmology, loading, ghost partition, seed condensation)
│       ├── export.rs         # SHBT-MMIO telemetry serialization (128-byte frame + seed records)
│       ├── provenance.rs     # Run provenance & reproducibility metadata capture
│       ├── stability_audit.rs# Numerical stability audit (condition numbers, tolerances)
│       ├── visualizer/       # Tier-2 shbt-visualizer crate (wgpu/wasm32, WGSL shaders, headless)
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
├── visualizer/
│   ├── index.html            # Browser harness: HUD, timeline scrub, channel toggles
│   ├── index.js              # Engine driver + SHBT-MMIO telemetry decode
│   └── pkg/                  # wasm-bindgen output (shbt_visualizer_bg.wasm)
├── data/
│   └── telemetry.bin         # SHBT-MMIO telemetry frame (generated)
├── examples/
│   ├── run_audit.py          # Minimal foundation verification script
│   └── shbt_notebook.ipynb   # Interactive analysis and visualization notebook
└── tests/
    ├── test_shbt.rs          # Rust integration test suite (modular closure, symplectic norms)
    ├── cosmological_invariants.rs  # Cosmological invariant tests (loading, ghost partition)
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
| **Observer Succession Lifecycle** | Closed-loop admissibility predicate *P*<sub>adm</sub>(*A*) = Θ(*R*<sub>entropy</sub>); Stinespring de-rendering into *H*<sub>dark</sub> with η<sub>D</sub> = 23/33, η<sub>V</sub> = 10/33, Kojima Ent(φ) = 0; normalized transfer kernel *T*(*A*<sub>term</sub> → *A*<sub>next</sub>) over lattice C = {0,1,2}²; five-phase Render → Crystallize → De-render → Relabel → Re-render loop | `src/shbt/causal_point.rs` |
| **Asymptotic Observer Freeze** | As z → −1 the loading fraction saturates (f<sub>load</sub> → 1) and the admissible observer set empties, *R*<sub>adm</sub> → ∅ — a deterministic horizon freeze recorded at `foundation_audit.asymptotic_observer_freeze` | `precision_cosmology.py` |

Boundary-isometry constraint consumed by downstream hardware:

> ‖*W*<sup>†</sup>*W* − *P*<sub>code</sub>‖<sub>op</sub> ≤ 3.430 × 10<sup>−3</sup>

---

## Two-Tier Simulation Engine Architecture

### Tier 1 — Rust/PyO3 High-Precision Foundation (`src/`, `shbt_simulator`)

- High-precision arithmetic via `rug`/MPFR: 512-bit floats, 492-bit mantissas, resolving 1/*N* ≃ 10<sup>−122</sup> against unit values.
- Symplectic Yoshida-6 integrators and zero-allocation hot loops.
- Core types: `StaticBoundary`, `HolographicProjection`, `BulkMetricSlice`, `BaryogenesisOptimizer`, `CausalPoint`, `CausalPointCandidate`, `DerenderingRecord`, `SuccessionRecord`, `LifecyclePhase`, `AnomalyClosureError`.
- Closed-loop observer succession: `CausalPoint.is_admissible` evaluates *P*<sub>adm</sub>(*A*); `terminate_and_derender` applies the macroscopic Stinespring channel (η<sub>D</sub> = 23/33, pointer triad Ψ<sub>ι</sub> → (0, 0, 1), Ent(φ) = 0); `evaluate_succession_kernel` returns the normalized *T*(*A*<sub>term</sub> → *A*<sub>next</sub>) distribution over the 3×3 visible coordinate lattice; `relabel_and_rerender` / `ShbtSimulator.run_succession_cycles` drive the five-phase lifecycle loop with zero heap allocation in the kernel hot loop (stack-allocated weight array).
- Legacy low-level engines reused by the SHBT modules: `AnyonBraidingEngine` (SU(2), SU(3), SO(10) braid matrices), `TopologicalTracker` (anyon worldlines, fusion, stabiliser checks), `CircuitCompiler` (Solovay-Kitaev, OpenQASM parsing).

### Tier 2 — Python Precision Cosmology & Boltzmann Pipeline

- `boltzmann_shbt.py`: first-principles inflation dynamics and perturbation spectra; primordial non-Gaussianity templates (local, equilateral, orthogonal *f*<sub>NL</sub> and *g*<sub>NL</sub> bispectra/trispectra); Boltzmann hierarchy integration producing scalar temperature (*C*<sub>*ℓ*</sub><sup>TT</sup>), polarization (*C*<sub>*ℓ*</sub><sup>EE</sup>, *C*<sub>*ℓ*</sub><sup>TE</sup>), matter power *P*(*k*, *z*), and tensor B-modes (*C*<sub>*ℓ*</sub><sup>BB</sup>); SHBT-MMIO telemetry export via `export_webgpu_telemetry`.
- `precision_cosmology.py`: Section 9 precision-cosmology audit, 7-parameter MCMC, *H*<sub>0</sub>-tension quantification, cosmic-chronometer covariance, and the sub-10 mK Landauer calorimetry experiment (`simulate_calorimetry_experiment`).
- `shbt_simulate.py`: unified CLI/API orchestrator; evaluates address sweeps and heat-dissipation hypotheses via OLS/MLE regression and emits the `shbt_run_*` data products.

### Tier 2 — WebGPU Interactive Visualizer (`src/shbt/visualizer`, `visualizer/`)

- `shbt-visualizer` crate compiles to `wasm32-unknown-unknown` and binds to `visualizer/index.html` (`<canvas id="shbt-canvas">`). The same crate runs natively via the `headless` binary for CI benchmarking.
- `ShbtWebGpuEngine` drives four WGSL stages: `causal_point_get.wgsl` (active Causal-Point GET dynamics — `evaluate_causal_points` applies the *R*<sub>entropy</sub> = *N*<sub>limit</sub> − *C*<sub>get</sub> ≥ 0 admissibility gate and `apply_get_acceleration` the emergent clustering acceleration **a**<sub>GET</sub>(**x**) = −κ<sub>GET</sub> ∇ln ρ<sub>proj</sub>(**x**)), `nbody_pm.wgsl` (Fast-PM/2LPT `kick_drift_kernel` under loaded conformal friction 1 + *f*<sub>load</sub>·10/33 with the Stinespring anti-baryon de-render envelope and softened supermassive-seed attraction), `dual_channel_render.wgsl` (Channel A visible emission weighted by `vis_weight`; Channel B passive ghost shear/density conserved independently of it), and `holographic_post.wgsl` (lensed-UV composite, ghost false-color, `unwrap_torus_projection` conformal unwrapping onto the CFT torus [0, 2π)², horizon overlay driven by *f*<sub>load</sub>).
- `WasmShbtEngine` owns the zero-copy simulation buffers — `ParticleRecord` (32 B), `CausalPointRecord` (64 B), `SeedDefectRecord` (128 B) — exposed via raw `get_*_buffer_ptr` accessors, with `update_epoch(z)` driven by `HorizonLedger` (*N*<sub>sat</sub> = 3.3119977 × 10<sup>122</sup> bits, Γ<sub>lock</sub>, Stinespring partition η<sub>A</sub> = 10/33 / η<sub>D</sub> = 23/33, seed inventory, Landauer debt, observer admissibility ℛ<sub>adm</sub>, conservation residual).
- Double-buffered storage holds 2<sup>20</sup> particles in 32-byte `Particle` records (< 256 MB); the HUD decodes the 128-byte SHBT-MMIO telemetry frame (Horizon Bar, Boundary Capacity Gauge, Congestion & Seed Ledger, Landauer Debt Monitor, Δ<sub>fr</sub> = 0 / *E*<sub>*μν*</sub> = 0 / horizon-freeze indicators).
- Interactive controls: timeline scrub z = 10<sup>14</sup> → −1 (ghost-seed condensation highlighted across z ≈ 30 → 7 with Δ*N* ≈ 6 × 10<sup>59</sup> bits and *P*<sub>debt</sub> ≈ 9.06 × 10<sup>20</sup> W), playback speeds 1×/10×/100×, continuous `unwrap_transition` torus slider plus comoving-bulk / boundary-CFT projection switch, and Channel A/B toggles. The HUD adds a Causal-Point Observer Activity monitor (ℛ<sub>adm</sub> cardinality and *R*<sub>entropy</sub> ≥ 0 / freeze status).

---

## CLI & Execution Reference

```bash
# Foundation audit (modular closure, projection, memory, baryogenesis)
python shbt_simulate.py --mode audit

# Full unified simulation (foundation + precision cosmology)
python shbt_simulate.py --mode all --output result.json

# Full pipeline with closed-loop observer succession (5-phase lifecycle,
# Stinespring de-rendering, transfer kernel over the 3x3 coordinate lattice)
python shbt_simulate.py --mode all --enable-succession --succession-cycles 3 --output result.json

# Boltzmann CMB and matter spectra generation
python boltzmann_shbt.py                      # runs the full pipeline, prints CMB spectra
python boltzmann_shbt.py --run-tests          # embedded Boltzmann/chronometer unit tests

# Precision cosmology report and parameter sweeps
python precision_cosmology.py --json
python precision_cosmology.py --run-tests
python shbt_simulate.py --mode cosmology --sweep sweep.json --plot --output sweep_result.json

# WebGPU visualizer: telemetry frame + wasm bundle + serve
python shbt_simulate.py --mode visualize --particles 262144 \
    --export-webgpu-telemetry data/telemetry.bin --sim-speed 1.0 --output viz.json
cargo build --release --manifest-path src/shbt/visualizer/Cargo.toml \
    --target wasm32-unknown-unknown
wasm-bindgen --target web --out-dir visualizer/pkg \
    src/shbt/visualizer/target/wasm32-unknown-unknown/release/shbt_visualizer.wasm
python -m http.server 8080 --directory visualizer   # open http://localhost:8080

# Headless GPU benchmark (no browser): frames + particles
cargo run --release --manifest-path src/shbt/visualizer/Cargo.toml \
    --bin headless -- 60 262144

# Sub-10 mK Landauer calorimetry audit (writes shbt_run_calorimetry_sim.csv)
python shbt_simulate.py --mode all --output result.json
# or programmatically:
python -c "import precision_cosmology as pc; pc.simulate_calorimetry_experiment(n_pulses=1000000)"
```

Additional modes: `baryogenesis` (topological asymmetry benchmark), `visualize` (SHBT-MMIO telemetry export + `cosmology_visualization` and `causal_point_simulation` records; flags `--particles`, `--export-webgpu-telemetry`, `--sim-speed`), `history` (Causal-Point observer crystallization; with `--enable-succession` runs the multi-cycle lifecycle engine and exports `succession` records to `result.json`), `cosmology-test`. Exports support `--format json|csv|hdf5`, `--plot`, `--sweep`, `--config` (YAML/JSON), `--seed`, and structured `--log-format json` logging.

---

## Section 12 Verification & Traceability Matrices

### Code & Equation Traceability Matrix (Table 38)

| Publication claim | Implementation | Output artifact |
| :--- | :--- | :--- |
| Bispectrum & trispectrum templates | `boltzmann_shbt.py` | `result.json: precision_pipeline.non_gaussianity` |
| GET capacity & cutoff | `src/shbt/causal_point.rs` (`CausalPoint.crystallize_history`) | `result.json: foundation_audit.memory_report` |
| Observer admissibility & de-rendering | `src/shbt/causal_point.rs` (`CausalPoint.is_admissible`, `terminate_and_derender`) | `result.json: succession.records[].eta_dark`, `.pointer_triad` |
| Succession transfer kernel | `src/shbt/causal_point.rs` (`evaluate_succession_kernel`, `ShbtSimulator.run_succession_cycles`) | `result.json: succession.records[].kernel_probabilities`, `.kernel_normalized` |
| Asymptotic observer freeze | `precision_cosmology.py` (`asymptotic_observer_freeze`) | `result.json: foundation_audit.asymptotic_observer_freeze` |
| Landauer calorimetry regression | `shbt_simulate.py` / `precision_cosmology.py` | `shbt_run_calorimetry_sim.csv` |
| Dynamic bit-loading Ṫ = *H*(*t*) (no inflaton) | `src/shbt/cosmology.rs` (`ShbtUniverse.evaluate_hubble_index_rate`) | `result.json: cosmology_visualization`; `tests/cosmological_invariants.rs` |
| Passive ghost dark matter (η<sub>D</sub> = 23/33, Ω<sub>DM,0</sub> = 0.260000) | `src/shbt/cosmology.rs` (`dark_completion_bits`, `dark_matter_fraction`) | `data/telemetry.bin` header fields |
| Mass-congestion condensation (*M*<sub>seed</sub> = α<sub>seed</sub> Δ*N*) | `src/shbt/cosmology.rs` (`compute_seed_condensation`, `eval_landauer_debt_power`) | SHBT-MMIO `SeedDefectRecord` payload |
| SHBT-MMIO telemetry frame | `src/shbt/export.rs` (`serialize_mmio_frame_py`, `MmioTelemetryHeader.to_bytes`) | `data/telemetry.bin` |
| Two-tier WebGPU visualizer | `src/shbt/visualizer` (`ShbtWebGpuEngine`, WGSL pipeline) | `visualizer/pkg/shbt_visualizer_bg.wasm` |

### Generated-Artifact Data Product Crosswalk (Table 39)

| Artifact | Columns | Content |
| :--- | :--- | :--- |
| `shbt_run_cmb_cls.csv` | `ell`, `Dl_TT_muK2`, `Dl_EE_muK2`, `Dl_TE_muK2`, `Cl_TT`, `Cl_EE`, `Cl_TE`, `Cl_BB` | Scalar temperature & polarization spectra |
| `shbt_run_matter_pk.csv` | `k_Mpc_inv`, `Pk_z0`, `Pk_z05`, `Pk_z1` | Matter power spectrum at *z* = 0, 0.5, 1 |
| `shbt_run_tensor_cls.csv` | `ell`, `Cl_BB`, `Dl_BB`, `Cl_TT_tensor` | Primordial tensor B-modes |
| `shbt_run_calorimetry_sim.csv` | `k_bits`, `R_addresses`, `Q_H0_zJ`, `Q_H1_zJ`, `Q_noise_zJ` | Sub-10 mK calorimetry simulation |
| `result.json` | nested ledger | Master simulation result tree |
| `data/telemetry.bin` | 128-byte SHBT-MMIO header + *T*/*P*<sub>m</sub> grids + seed records | Tier-2 visualizer telemetry contract |
| `visualizer/pkg/` | `shbt_visualizer.js`, `shbt_visualizer_bg.wasm` | Browser engine bundle |

### Software Interface Reproduction Contract (Table 40)

| Symbol | Kind |
| :--- | :--- |
| `shbt_simulator` | Master PyO3 module |
| `StaticBoundary` | Boundary CFT modular data, framing defect, dark ledger |
| `ShbtSimulator` | Orchestrating runtime (`run_full_audit`) |
| `HolographicProjection` | RG flow → `BulkMetricSlice` |
| `CausalPoint` | Observer memory & history crystallization |
| `CausalPoint` (succession) | `is_admissible`, `terminate_and_derender`, `evaluate_succession_kernel`, `relabel_and_rerender`, `run_lifecycle_cycle` |
| `CausalPointCandidate` / `DerenderingRecord` / `SuccessionRecord` / `LifecyclePhase` | Observer-succession record types (Stinespring de-rendering, transfer kernel, 5-phase lifecycle) |
| `AnomalyClosureError` | Algebraic anomaly failure type |
| `ShbtUniverse` | Branch cosmology: loading, Hubble, ghost partition, seed condensation, horizon freeze (`src/shbt/cosmology.rs`) |
| `MmioTelemetryHeader` / `SeedDefectRecord` | SHBT-MMIO serialization (`src/shbt/export.rs`) |
| `ShbtWebGpuEngine` / `Particle` / `HudMetrics` / `TimelineController` | Tier-2 visualizer (`src/shbt/visualizer`) |

See [`paper_references.md`](paper_references.md) for the complete paper-section → method crosswalk.

---

## Build, Packaging & Document Compilation

```bash
# Prerequisites: Rust 1.80+ (rustup), Python 3.8+, pip install maturin
pip install -r requirements.txt

# For the WebGPU visualizer additionally:
rustup target add wasm32-unknown-unknown
cargo install wasm-bindgen-cli   # matches wasm-bindgen crate version

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

The SHBT program is a federated ecosystem of nine configuration-controlled repositories. `shbt-precision` is the mathematical core; downstream repositories consume its audited invariants.

```text
                              [shbt-precision]
                       Computational Math & Cosmology
                       (512-bit MPFR / WZW Characters)
                                     │
    ┌────────────────────────────────┼───────────────────────────────┐
    ▼                                ▼                               ▼
 [shbt-power]                     [shbt-cf]                       [shbt-qc]
 Commercial Fusion Grid         1,800-Module LANR Array         Bare-Metal Microkernel &
 (8,750 MW p-11B Twin)          & Thermal-Hydraulics            Photonic Quantum Bus
        │                            │                               │
        └────────────────────────┬───┴───────────────────────────────┘
                                 ▼
        ┌────────────────────────────────────────────────────────────────┐
        │                  SPECIALIZED VEHICLE TWINS                     │
        │  • shbt-ghost : Reactionless Propulsion & Local Gravity Wells  │
        │  • shbt-recon : Macroscopic State Translocation Gateway        │
        │  • shbt-sglt  : Synthetic Gravitational Lensing Telescope      │
        │  • shbt-warp  : Holographic Warp Metric & 3+1D Flight Twin     │
        └────────────────────────┬───────────────────────────────────────┘
                                 │
                                 ▼
        ┌──────────────────────────────────────────────────────────────────────────┐
        │                               shbt-exotic                                │
        │        MULTI-PROTOCOL SPACETIME ENGINEERING CO-SIMULATION BENCH          │
        │  • Cross-Protocol Field Coupling (Warp + Stasis + Translocation + Wells) │
        │  • Global Energy Condition & Ford-Roman Quantum Inequality Auditing      │
        │  • Dynamic 5-Stage Multi-Technology Flight Director                      │
        └──────────────────────────────────────────────────────────────────────────┘
```

### Standardized 9-Pillar Ecosystem Crosswalk

| Repository | Domain Role & Platform Scope | Shared Invariants & Interface Contracts |
| :--- | :--- | :--- |
| [`shbt-precision`](https://github.com/sys1own/shbt-precision) | Computational Math & Cosmological Foundation Core | 512-bit MPFR numerics, canonical WZW (26, 8, 312), Δ<sub>fr</sub> ≡ 0, Landauer debt P<sub>debt</sub> = 906.00 kW. |
| [`shbt-power`](https://github.com/sys1own/shbt-power) | Commercial p-¹¹B Aneutronic Fusion Power Plant Twin | 8,750 MW fusion / 7,832.903 MW net export, 70-gate audit, closed-loop thermal ledger, 128-byte SHBT-MMIO-POWER. |
| [`shbt-cf`](https://github.com/sys1own/shbt-cf) | LANR Cold Fusion Reactor Workbench & Thermal-Hydraulics | 1,800-module LANR starter grid (999.054 kW net DC), dual-stage CoSb<sub>3</sub>/ZrNiSn TEG, Kapitza resistance ΔT<sub>K</sub> = 3.546 K. |
| [`shbt-qc`](https://github.com/sys1own/shbt-qc) | Photonic Quantum Computer Twin & C11 Microkernel | Bare-metal C11 shbt-os microkernel, base 56-byte SHBT-MMIO-1 at 0x70000000, SECDED Hamming(72,64) ECC, AVX-512 interlocks. |
| [`shbt-ghost`](https://github.com/sys1own/shbt-ghost) | Ghost Seed Reactionless Propulsion & Metric Stabilization | Sub-2.5 ns PCSS crowbars, 94.20% SiC inductive recovery, 3+1 CCZ4/ADM stabilization (β<sup>i</sup> → 0, ‖det(g)+1‖ ≤ 10<sup>-12</sup>). |
| [`shbt-recon`](https://github.com/sys1own/shbt-recon) | Macroscopic State Translocation & Gateway Twin | Macroscopic Stinespring dilation (V<sub>unified</sub><sup>macro</sup>), dark ledger η<sub>D</sub> = 23/33, 128-byte C-ABI DMA streaming, 78-gate audit. |
| [`shbt-sglt`](https://github.com/sys1own/shbt-sglt) | Synthetic Gravitational Lensing Telescope (SE-L2) Stack | 2PN relativistic beam optics, TMSV heterodyne metrology (r = 2.50, 21.715 dB), 5th-order minimum-jerk flight profiles. |
| [`shbt-exotic`](https://github.com/sys1own/shbt-exotic) | Multi-Protocol Spacetime Engineering Co-Simulation | Cross-protocol metric coupling (all 6 phenomena), Ford-Roman QI dark-ledger auditing, Heegaard-Floer boundary relabeling. |
| [`shbt-warp`](https://github.com/sys1own/shbt-warp) | Holographic Warp Drive Digital Twin & 3+1D ADM Engine | Alcubierre metric foliation (α = 1.0, γ<sub>ij</sub> = δ<sub>ij</sub>), 500 TJ ¹⁷⁸ᵐ²Hf graser battery (109 TW burst), 128-gate audit, 8 Z3 proofs. |

---

## License

MIT — see [`LICENSE`](LICENSE).
