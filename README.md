# SHBT Simulator — Static Holographic Boundary Theory

[![Rust](https://img.shields.io/badge/rust-1.80+-blue.svg)](https://www.rust-lang.org/)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**SHBT Simulator** is an executable implementation of the Static Holographic Boundary Theory (SHBT), a first‑principles framework that derives gravitational closure, baryogenesis, and observer histories from a completed modular‑invariant boundary CFT. The theory is fully documented in the accompanying paper [`main.pdf`](main.pdf). This simulator serves as an **executable proof** of the theory: every claim, equation, and numerical prediction in the paper is audited by the Rust/Python code in this repository.

---

## The Big Picture

SHBT postulates that the universe is described by a finite boundary register whose modular‑invariant pairing fixes:

- The canonical branch $(k_\ell, k_q, K) = (26, 8, 312)$ with zero framing defect.
- The holographic dark‑energy scale $\Lambda_{\rm holo} \simeq 1.09\times 10^{-52}\,\text{m}^{-2}$.
- A finite bit budget $N \simeq 3.31\times 10^{122}$.
- A holographic RG flow from boundary entropy densities to symmetric, trace‑normalised metric slices.
- A topological baryogenesis identity yielding $\eta_B \simeq 6.45\times 10^{-10}$.
- A Causal Point observer interface that crystallises histories only within a local entropy budget.

The **paper** (`main.pdf`) is the mathematical formulation. The **simulator** is the computational verification.

---

## Quick Start (TL;DR)

Clone, build, and run the full paper audit:

```bash
git clone https://github.com/sys1own/shbt-simulator.git
cd shbt-simulator

# Install Rust (if not already installed)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"

# Install maturin and build the Python bindings
pip install maturin
maturin develop --release

# Run the audit
python shbt_simulate.py --mode audit
```

## Paper ⇄ Simulator Relationship

The paper and simulator are developed **in lockstep**:

- Every equation in the paper is implemented in the Rust code.
- Every numerical audit table (Sections 8.1–8.4) can be reproduced by running the simulator.
- The paper explicitly references simulator objects, e.g.  
  `shbt_simulator.ShbtSimulator().run_full_audit()`,  
  `shbt_simulator.StaticBoundary`,  
  `shbt_simulator.HolographicProjection`, etc.
- The file [`paper_references.md`](paper_references.md) provides a complete mapping from paper sections to simulator methods, making it easy to navigate between the theory and its computational realisation.

---

## Repository Structure

```
.
├── Cargo.toml              # Rust project manifest
├── src/
│   ├── lib.rs              # Main Rust library (PyO3 bindings)
│   └── shbt/               # SHBT core modules
│       ├── boundary.rs      # StaticBoundary (modular data, defect, entropy)
│       ├── entropy_flow.rs  # HolographicProjection (RG flow)
│       ├── baryogenesis.rs  # BaryogenesisOptimizer (η_B, de-rendering)
│       └── causal_point.rs  # CausalPoint (observer memory, history)
├── examples/
│   ├── run_audit.py              # Thin wrapper around shbt_simulate.py --mode audit
│   └── shbt_notebook.ipynb       # Jupyter / Colab example
├── shbt_simulate.py             # Customisable CLI/API for research runs
├── config.default.yaml          # Default simulation configuration template
├── requirements.txt             # Optional Python dependencies (plots/HDF5/pandas)
├── tests/
│   └── test_shbt.rs             # Unit tests for all SHBT components
├── main.pdf                     # The SHBT paper (formal theory)
├── paper_references.md          # Mapping from paper to code
└── README.md                    # This file
```

---

## Prerequisites

- **Rust** (1.80 or later) – [install via rustup](https://rustup.rs/)
- **Python** (3.8 or later) – with `pip`
- **maturin** (for building the Python module) – `pip install maturin`
- (Optional) **cargo‑test** for running unit tests
- (Optional) `matplotlib`, `h5py`, `pandas` for data export and plots – `pip install -r requirements.txt`

---

## Build & Install

### 1. Build the Rust library

```bash
cargo build --release
```

This produces a shared library in `target/release/`.

### 2. Build the Python bindings (via maturin)

```bash
maturin build --release
```

This will create a wheel in `target/wheels/`. Install it with:

```bash
pip install target/wheels/shbt_simulator-*.whl
```

You can also install directly from the local folder using:

```bash
maturin develop
```

Now you can import the module in Python:

```python
import shbt_simulator
```

---

## Run the Full Audit

The main entry point is the Python script `examples/run_audit.py` (a thin wrapper around `shbt_simulate.py --mode audit`). It constructs a `ShbtSimulator` object, runs the complete audit, and prints all key results.

```bash
python examples/run_audit.py
```

Expected output:

```json
{
  "branch": [26, 8, 312],
  "framing_defect (delta_fr)": 0.0,
  "modular_invariant": true,
  "zero_energy_locked": true,
  "projection_dimension_26_to_4": true,
  "eta_b": 6.449923359416e-10,
  "stress_energy_preserved": true,
  "projection_all_passed": true,
  "memory_all_passed": true,
  "metric_slices": 9,
  "history_entries": 9
}
```

## Precision cosmology integration

The standalone module `precision_cosmology.py` implements the Section 9 equations and can now pull the canonical constants directly from the compiled simulator:

```python
import precision_cosmology
constants = precision_cosmology.load_default_constants()
print(constants.h0_cmb, constants.lambda_holo_si_m2, constants.n_sat, constants.c_dark_residual)
```

Run its embedded audit with:

```bash
python precision_cosmology.py --run-tests
```

## Run custom simulations with `shbt_simulate.py`

`shbt_simulate.py` is the unified CLI and API entry point for research runs. It wraps the Rust/PyO3 simulator and the `precision_cosmology.py` Section 9 audit in one interface.

```bash
python shbt_simulate.py --mode audit
python shbt_simulate.py --mode all --output result.json
python shbt_simulate.py --mode cosmology --output cosmology.json
python shbt_simulate.py --mode cosmology-test
python shbt_simulate.py --mode baryogenesis --particles 1024
python shbt_simulate.py --mode history --observer-radius-fraction 0.2
```

Available modes are `audit` (foundation audit only), `cosmology` (Section 9 precision-cosmology report), `cosmology-test` (embedded precision-cosmology unit tests), `baryogenesis`, `history`, and `all` (default; foundation + precision cosmology).

`--mode cosmology` accepts optional precision-cosmology parameters:

```bash
python shbt_simulate.py --mode cosmology \
  --h0-cmb 67.4 \
  --omega-m 0.315 \
  --omega-r0 9.2e-5 \
  --z-samples 0 0.5 1 2 10 1100 \
  --output cosmology.json
```

Programmatically:

```python
import shbt_simulate
result = shbt_simulate.simulate({
    "mode": "all",
    "branch": (26, 8, 312),
    "observer_radius_fraction": 0.125,
    "redshift_max": 3.0,
    "redshift_samples": 9,
    "particles": 512,
})
print(result["audit"]["eta_b"])
```

### Export formats

`--output` writes the full result to a file. The format is inferred from the extension (`json`, `csv`, `h5`/`hdf5`) or set explicitly with `--format`. If no format is specified, JSON is used.

```bash
python shbt_simulate.py --mode all --output result.json
python shbt_simulate.py --mode cosmology --output cosmology.json
python shbt_simulate.py --mode cosmology --format csv --output cosmology
python shbt_simulate.py --mode all --format hdf5 --output result.h5
```

- JSON (default) stores the complete nested result tree.
- CSV writes table files: `{prefix}_metric_slices.csv`, `{prefix}_history.csv`, and for cosmology `{prefix}_precision_summary.csv`, `{prefix}_redshift_ladder.csv`, `{prefix}_growth_suppression.csv`, `{prefix}_lightcone_entropy_debt.csv`, and `{prefix}_isw_stability.csv`.
- HDF5 requires `h5py` and stores the full result tree.

### Optional plots

If `matplotlib` is installed, `--plot` writes PNG figures alongside the data export:

```bash
python shbt_simulate.py --mode all --output result.json --plot
python shbt_simulate.py --mode cosmology --output cosmology.json --plot
```

For `cosmology`/`all`, this produces:

- `result_eigenvalues.png` (metric eigenvalues vs redshift step)
- `result_spatial_metric.png` (spatial metric heatmap)
- `result_hubble_ladder.png` (`H0(z)` vs redshift)
- `result_growth_suppression.png` (`fσ8` vs redshift)
- `result_isw_residual.png` (ISW residual vs redshift)

For sweeps, `--plot` also visualises the scanned parameter.

### Parameter sweeps

Write a JSON file with list-valued parameters. The sweep runner detects cosmology keys (`h0_cmb`, `omega_m`, `omega_r0`, `delta_mod`, `z_samples`, `precision`) and runs the precision cosmology audit for each combination; otherwise it runs the foundation simulator.

```json
{
  "h0_cmb": [67.0, 67.4, 68.0],
  "omega_m": [0.30, 0.315, 0.33]
}
```

Then run:

```bash
python shbt_simulate.py --mode cosmology --sweep sweep.json --output sweep_result.json
python shbt_simulate.py --mode cosmology --sweep sweep.json --plot --output sweep_result.json
```

The first command scans nine combinations of `h0_cmb` and `omega_m`. The second also writes sweep summary plots.

Foundation sweeps use the original keys:

```json
{
  "redshift_samples": [5, 9, 13],
  "observer_radius_fraction": [0.1, 0.125]
}
```

Then run as before.

### Configuration files

Simulation setups can be stored in YAML or JSON files and reused:

```yaml
# my_config.yaml
mode: all
branch: [26, 8, 312]
observer_radius_fraction: 0.125
redshift_max: 3.0
redshift_samples: 9
particles: 512
seed: 0
# Optional precision-cosmology overrides (used when mode is cosmology or all)
h0_cmb: 67.4
omega_m: 0.315
omega_r0: 9.2e-5
z_samples: [0, 0.5, 1, 2, 10, 1100]
precision: 80
output_dir: ./simulation_results
export_formats: [json, csv]
plot: true
verbose: false
log_level: INFO
log_format: text
quiet: false
```

Run it with:

```bash
python shbt_simulate.py --config my_config.yaml
```

CLI flags override config file values, so you can iterate quickly:

```bash
python shbt_simulate.py --config my_config.yaml --mode baryogenesis --particles 1024 --seed 42
```

A default configuration is provided in [`config.default.yaml`](config.default.yaml). The config is validated against a schema; if `jsonschema` is installed it is used, otherwise a manual validator runs.

Results are written to `output_dir/<timestamp>/result.<fmt>` so repeated runs are organised automatically. Each run directory also contains a reproducibility log (`result.log`) and `result_run_info.json` with the simulator version, git commit/branch (when available), config, and summary.

Use `--seed` or the `seed` config key to make Causal Point collapse selections reproducible.

### Logging and output control

`shbt_simulate.py` uses Python's `logging` module. The log level, format, and output destination are configurable:

```bash
python shbt_simulate.py --mode audit --log-level DEBUG
python shbt_simulate.py --mode all --log-format json --log-file run.log
python shbt_simulate.py --mode audit --quiet
python shbt_simulate.py --mode audit --verbose
```

- `--log-level` accepts `DEBUG`, `INFO`, `WARNING`, or `ERROR` (default: `INFO`).
- `--log-format json` emits structured JSON lines for key events, e.g. `{"event": "audit_complete", "eta_b": 6.449923359416e-10}`.
- `--log-file` appends log messages to a file as well as the console.
- `--quiet` (`-q`) suppresses non-essential console output and defaults the log level to `WARNING`.
- `--verbose` (`-v`) is an alias for `--log-level DEBUG`.

For `audit` and `all` modes, a clean ASCII summary table is printed at the end:

```
+------------------------------------------------------+
| SHBT Audit Summary                                   |
+------------------------------------------------------+
| Branch                       | (26, 8, 312)           |
| Framing defect (delta_fr)    | 0.0                    |
| Modular invariant            | True                   |
| Zero energy locked           | True                   |
| Projection dimension 26 -> 4 | True                   |
| eta_b                        | 6.449923359416131e-10  |
| Stress energy preserved      | True                   |
| Metric slices                | 9                      |
| History entries              | 9                      |
+------------------------------------------------------+
```

### Jupyter / Colab

See [`examples/shbt_notebook.ipynb`](examples/shbt_notebook.ipynb) for a notebook that loads the simulator, runs a custom configuration, and plots the results.

## Python bindings

### Run all unit tests

```bash
cargo test --release
```

All tests should pass, confirming that the simulator satisfies the algebraic constraints derived in the paper.

---

## Verifying the Paper’s Tables

The simulator’s audit reports (`boundary_report`, `projection_report`, `memory_report`, `baryogenesis_identity`, `benchmark_delta`) contain every value that appears in the paper’s tables. You can compare them directly with the printed outputs from `run_audit.py`.

- **Table 1** – Boundary closure audit → `report.boundary_report`
- **Table 2** – Holographic projection → `report.projection_report`
- **Table 3** – Causal Point memory → `report.memory_report`
- **Table 4** – Baryogenesis benchmark → `report.benchmark_delta`

The paper is written so that the reader can, at any point, refer to the code and see that the mathematics is implemented exactly.

---

## Navigating the Code

### Core SHBT components

| Module | Structure | Paper Section |
|--------|-----------|---------------|
| `boundary.rs` | `StaticBoundary` | Sections 2–4 |
| `entropy_flow.rs` | `HolographicProjection`, `BulkMetricSlice` | Section 5 |
| `baryogenesis.rs` | `BaryogenesisOptimizer`, `BaryogenesisIdentity` | Section 6 |
| `causal_point.rs` | `CausalPoint`, `LightConeSample`, etc. | Section 7 |

### Existing (legacy) code in `lib.rs`

The original `lib.rs` already implements the low‑level components used by SHBT:
- High‑precision modular arithmetic (`rug::Float`, `rug::Complex`)
- `AnyonBraidingEngine` (SU(2), SU(3), SO(10) braid matrices)
- `TopologicalTracker` (anyon worldlines, fusion, stabiliser checks)
- `CircuitCompiler` (Solovay‑Kitaev, OpenQASM parsing)

These are reused by the new SHBT modules where appropriate.

---

## License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.

---
