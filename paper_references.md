# SHBT Paper-to-Code Reference Mapping

This file extracts the code-related constants, class/method references, audit tables, and numerical benchmark results from `main.pdf` and maps them to the new Rust/Python simulator. It is intended as a guide for updating the paper so that old `shbt_core.py` references are replaced by references to `shbt_simulator`.

---

## 1. Constants and Branch Parameters

| Name | Paper Value / Definition | Paper Location | Rust/Python Location |
|------|--------------------------|----------------|----------------------|
| `BENCHMARK_BRANCH` | `(26, 8, 312)` | Eq. (3), Section 2 | `StaticBoundary.benchmark_branch` / `ShbtSimulator` default |
| `LEPTON_LEVEL` (`k_ℓ`) | `26` | Eq. (3), Section 2 | `StaticBoundary.lepton_level` |
| `QUARK_LEVEL` (`k_q`) | `8` | Eq. (3), Section 2 | `StaticBoundary.quark_level` |
| `PARENT_LEVEL` (`K`) | `312` | Eq. (3), Section 2 | `StaticBoundary.parent_level` |
| `I_L_STAR` | `6` (`K / (2 k_ℓ)`) | Eq. (3), Section 2 | `StaticBoundary.i_l_star` |
| `I_Q_STAR` | `13` (`K / (3 k_q)`) | Eq. (3), Section 2 | `StaticBoundary.i_q_star` |
| `C_DARK_FRACTION` (`c_dark`) | `834433 / 362670 ≈ 2.300805139658643` | Eq. (5), Section 2 | `StaticBoundary.c_dark` (Rational) |
| `lambda_holo` (`Λ_holo`) | `1.0892229828054038e-52` m⁻² | Eq. (7), Section 2 | `StaticBoundary.lambda_holo` |
| `bit_budget` (`N` or `N_sat`) | `3.311997720142366e122` bits | Eq. (7), Section 2 | `StaticBoundary.bit_budget` |
| `LIGHT_SPEED_M_PER_S` (`c`) | `299_792_458.0` m/s | Eq. (124), Section 7 | `causal_point.rs` constant |
| `HBAR_J_S` (`ℏ`) | `1.054_571_817e-34` J·s | Eq. (124), Section 7 | `causal_point.rs` constant |
| `PLANCK_MASS_GEV` | `1.220_890e19` GeV | Section 6 | `baryogenesis.rs` constant |
| `GUT_SCALE_GEV` | `2.0e16` GeV | Section 6 | `baryogenesis.rs` constant |
| `PRIME_LATTICE` | `[2.0, 3.0, 5.0, 7.0, 11.0]` | Eq. (92), Section 5 | `entropy_flow.rs` constant |
| `LOW_SU3_WEIGHTS` | `[(0, 0), (1, 0), (0, 1)]` | Eq. (45), Section 3 | `boundary.rs` / `causal_point.rs` constant |
| `CHARGE_EMBEDDING` | `(k_ℓ − 4, k_ℓ − 3, k_ℓ) = (22, 23, 26)` | Eq. (45), Section 3 | `boundary.rs` constant |
| `DEFAULT_XI` | `(1/26, 1/8, 1/312)` | Eq. (121), Section 7 | `CausalPoint.xi` default |
| `DEFAULT_OBSERVER_RADIUS_FRACTION` | `0.125` | Section 7 | `CausalPoint.observer_radius_fraction` default |
| `DEFAULT_REDSHIFT_MAX` | `3.0` | Section 7 | `CausalPoint.redshift_max` default |
| `DEFAULT_REDSHIFT_SAMPLES` | `9` | Section 7 | `CausalPoint.redshift_samples` default |
| `KAPPA_D5` | geometric factor from `SU(2)_26` (see `lib.rs` / `baryogenesis.rs`) | Eq. (108)–(112), Section 6 | `BaryogenesisOptimizer` internal |

Key derived horizon/memory quantities:

| Quantity | Paper Definition | Paper Location | Rust/Python Accessor |
|----------|------------------|----------------|----------------------|
| Planck length `L_P` | `sqrt(3π / (N Λ_holo))` | Section 2 / 7 | `CausalPoint.planck_length_m` |
| Global horizon `R_H` | `sqrt(3 / Λ_holo)` | Eq. (118), Section 7 | `MemoryReport.R_H_m` |
| Observer radius `R_obs` | `0.125 R_H` | Section 7 | `CausalPoint.observer_radius_m` |
| Local horizon `R_local` | `R_H − R_obs` | Section 7 | `MemoryReport.R_local_m` |
| Horizon fraction `f_H` | `R_local / R_H` | Eq. (119), Section 7 | `MemoryReport.f_H` |
| Local available bits `N_local` | `N f_H²` | Eq. (120), Section 7 | `MemoryReport.local_available_bits` |
| Hidden bits `N_hidden` | `N − N_local` | Eq. (120), Section 7 | `MemoryReport.hidden_bits` |
| Entropy limit `N_limit` | `min(N_local, A_local / (4 L_P² ln 2))` | Eq. (120), Section 7 | `MemoryReport.entropy_limit_bits` |
| Hidden fraction `f_hidden` | `N_hidden / N` | Eq. (121), Section 7 | `CausalPoint.f_hidden` |
| `w(ξ)` | `(ξ_ℓ + ξ_q + ξ_s) / 3` | Eq. (121), Section 7 | `CausalPoint.w_xi` |
| Self-valuation `Σ` | `(1 + Δ_frame)(1 + w(ξ) f_hidden)` | Eq. (122), Section 7 | `MemoryReport.sigma` |
| Localized entropy gradient `∇_obs Σ` | `Σ f_hidden / R_local` | Eq. (123), Section 7 | `MemoryReport.localized_entropy_gradient_per_m` |
| Apparent acceleration `a_obs` | `c² ∇_obs Σ` | Eq. (124), Section 7 | `MemoryReport.gravitational_acceleration_m_per_s2` |

---

## 2. Class and Method References

### 2.1 StaticBoundary (Section 2–3)

| Paper Location | Current Python Reference | Mathematical Object / Result | Suggested Rust/Python Reference |
|----------------|--------------------------|------------------------------|---------------------------------|
| Section 2, Eqs. (3)–(4) | `StaticBoundary` | Canonical branch and framing defect | `ShbtSimulator.boundary` or `StaticBoundary` |
| Section 2 | `StaticBoundary.framing_defect()` | `Δ_fr = max(|K/(2k_ℓ) − I_L_STAR|, |K/(3k_q) − I_Q_STAR|)` | `report.framing_defect` / `StaticBoundary.framing_defect()` |
| Section 3 | `StaticBoundary._build_su2_visible_phases()` / `_build_su3_visible_phases()` | Visible modular phases for `S` and `T` kernels | `StaticBoundary` internal (exposed as implementation detail) |
| Section 3 | `StaticBoundary._build_su2_visible_block()` / `_build_su3_visible_block()` | Visible 3×3 modular blocks | `StaticBoundary.build_su2_visible_block()` / `build_su3_visible_block()` |
| Section 3, Eq. (59) | `StaticBoundary.evaluate_Z_boundary(tau)` | Boundary partition function `Z_code^∂(τ)` | `StaticBoundary.evaluate_z_boundary()` |
| Section 4, Eqs. (66)–(70) | `StaticBoundary._build_raw_loading_density()` | Loading density `ρ_B` | `StaticBoundary.build_loading_density()` |
| Section 4, Eqs. (66)–(70) | `StaticBoundary._build_entanglement_density()` | Entanglement density `ρ_E` | `StaticBoundary.build_entanglement_density()` |
| Section 4, Eq. (70) | `StaticBoundary._build_dominant_loading_sequence()` | Dominant ordering `σ` (length 9) | `StaticBoundary.build_dominant_sequence()` |
| Section 4, Eq. (76) | `StaticBoundary.entropy_self_resolution()` | Iterative entropy-resolution updates | `StaticBoundary.entropy_self_resolution()` |
| Section 4, Eqs. (81)–(84) | `StaticBoundary.derive_temporal_increment(H)` | Temporal kernel `Ḣ = H(t)` perception identity | `StaticBoundary.derive_temporal_increment()` |
| Section 3, Table 1 | `StaticBoundary.verify_equations()` | Full boundary closure audit | `ShbtSimulator.run_full_audit().boundary_report` or `StaticBoundary.verify_equations()` |

### 2.2 HolographicProjection (Section 4–5)

| Paper Location | Current Python Reference | Mathematical Object / Result | Suggested Rust/Python Reference |
|----------------|--------------------------|------------------------------|---------------------------------|
| Section 5 | `HolographicProjection` | RG projection controller | `ShbtSimulator.projection` or `HolographicProjection` |
| Section 5, Eq. (92) | `HolographicProjection.derive_load_vector()` | `ρ_E, σ, Ω_τ → ℓ_r` | `HolographicProjection.derive_load_vector()` |
| Section 5, Eqs. (94)–(102) | `HolographicProjection.metric_from_load_vector()` | `ℓ_r → Φ_s, W_s, v^s_μ, g_μν` | `HolographicProjection.metric_from_load_vector()` |
| Section 5, Eq. (105) | `HolographicProjection.project_static_block_to_bulk()` | `P_a^μ g_μν P_b^ν` | `HolographicProjection.project_static_block_to_bulk()` |
| Section 5, Eq. (103) | `HolographicProjection.verify_projection()` | Symmetry, trace-1, positive-definiteness, 3×3 shape | `HolographicProjection.verify_projection()` / `ShbtSimulator.run_full_audit().projection_report` |
| Section 5 | `BulkMetricSlice` | Metric slice record | `BulkMetricSlice` / `ShbtReport.metric_slices` |

### 2.3 BaryogenesisOptimizer (Section 6)

| Paper Location | Current Python Reference | Mathematical Object / Result | Suggested Rust/Python Reference |
|----------------|--------------------------|------------------------------|---------------------------------|
| Section 6 | `BaryogenesisOptimizer` | Baryogenesis / anti-baryon de-rendering controller | `ShbtSimulator.optimizer` or `BaryogenesisOptimizer` |
| Section 6, Eqs. (108)–(112) | `BaryogenesisOptimizer.baryogenesis_identity()` | `C_sph, J_CP^topo, M_N/M_P, η_B` | `ShbtSimulator.run_full_audit().baryogenesis_identity` |
| Section 6, Eq. (113) | `BaryogenesisOptimizer.cpu_cycle_weight()` | Active cost of rendered field | `BaryogenesisOptimizer.cpu_cycle_weight()` |
| Section 6, Eq. (115) | `BaryogenesisOptimizer.derender_antibaryon_charges()` | Charge-stripping constraints `D_B` | `BaryogenesisOptimizer.derender_antibaryon_charges()` |
| Section 6, Eq. (116) | `BaryogenesisOptimizer.stress_energy_preserved()` | Passive stress-energy equality | `BaryogenesisOptimizer.stress_energy_preserved()` / `report.stress_energy_preserved` |
| Section 6, Table 4 | `BaryogenesisOptimizer.run_benchmark()` | Field A vs. Field B cost comparison | `ShbtSimulator.run_full_audit().benchmark_delta` |
| Section 6 | `BaryogenesisIdentity` | Dataclass for identity results | `BaryogenesisIdentity` |
| Section 6 | `FieldSimulation` / `BenchmarkDelta` | Benchmark records | `FieldSimulation` / `BenchmarkDelta` |

### 2.4 CausalPoint (Section 7)

| Paper Location | Current Python Reference | Mathematical Object / Result | Suggested Rust/Python Reference |
|----------------|--------------------------|------------------------------|---------------------------------|
| Section 7 | `CausalPoint` | Observer / Causal Point interface | `ShbtSimulator.causal_point` or `CausalPoint` |
| Section 7, Eqs. (127)–(129) | `CausalPoint._build_past_light_cone()` | Redshift grid → boundary coordinate samples | `CausalPoint.build_past_light_cone()` |
| Section 7 | `CausalPoint.property_packets` | Local anisotropy and packet data | `CausalPoint.compute_property_packets()` |
| Section 7, Eqs. (120)–(132) | `CausalPoint.verify_memory_budget()` | Entropy-budget admissibility | `CausalPoint.verify_memory_budget()` / `ShbtSimulator.run_full_audit().memory_report` |
| Section 7, Eqs. (133)–(137) | `CausalPoint.crystallize_history()` | History projection and pointer packet | `CausalPoint.crystallize_history()` / `ShbtReport.history_entries` |
| Section 7 | `LightConeSample` | Past-light-cone sample record | `LightConeSample` |
| Section 7 | `LocalPropertyPacket` | Per-sample property record | `LocalPropertyPacket` |
| Section 7 | `CoordinateLogEntry` | Crystallized history log entry | `CoordinateLogEntry` |
| Section 7 | `MemoryReport` | Memory-budget audit report | `MemoryReport` |

### 2.5 Unified Harness

| Paper Location | Current Python Reference | Mathematical Object / Result | Suggested Rust/Python Reference |
|----------------|--------------------------|------------------------------|---------------------------------|
| Section 8 | `shbt_core.py` main lifecycle | Full boundary → bulk → observer → baryogenesis audit | `ShbtSimulator.run_full_audit()` |
| Section 8 | `shbt_core.py` cosmology wrapper | Redshift-sampled metric slices | `ShbtSimulator.simulate_cosmology(z_max, samples)` |

---

## 3. Audit Tables from the Paper

### Table 1: Static Boundary Verification Results

| Audit quantity | Expected condition | Result |
|----------------|--------------------|--------|
| Branch levels `(k_ℓ, k_q, K) = (26, 8, 312)` | verified | verified |
| Framing defect `Δ_fr = 0` | verified | verified |
| Loading density `Σ_i ρ_load,i = 1` | true | true |
| Entanglement density `Σ_i ρ_ent,i = 1` | true | true |
| Dominant sequence | benchmark ordering is reproduced | true |
| Modular S closure `‖[M, S_∂]‖ ≃ 0` | true | true |
| Modular T closure `‖[M, T_∂]‖ ≃ 0` | true | true |
| Modular invariant | completed partition function is invariant | true |
| Zero-energy lock | boundary Hamiltonian constraint closes | true |
| Visible projection `26 → 4` | projected dimension check | true |

**Paper location:** Section 8.1, Eq. (139)–(141).

### Table 2: Holographic Projection Verification Results

| Audit quantity | Expected condition | Result |
|----------------|--------------------|--------|
| Entropy cascade length | `slice_count = 9` | verified |
| Spatial projector rank | `projector_rank = 3` | verified |
| Metric symmetry | `g_ij = g_ji` | true |
| Trace normalization | `Tr(g) = 1` slice-by-slice | true |
| Positive definiteness | all eigenvalues are positive | true |

**Paper location:** Section 8.2, Eqs. (142)–(143).

### Table 3: Causal Point Verification Results

| Audit quantity | Expected condition | Result |
|----------------|--------------------|--------|
| Local available bits `N_local ≃ N f_H²` | verified | verified |
| Hidden bits `N_hidden = N − N_local` | verified | verified |
| Entropy limit `N_limit > 0` | true | true |
| Light-cone samples | samples match the redshift grid | true |
| Memory admissibility | residual entropy is nonnegative | true |
| History construction | collapse index and pointer packet returned | true |

**Paper location:** Section 8.3, Eqs. (144)–(145).

### Table 4: Normalized Standard-Versus-Optimized Field Simulation Cost

| Simulation mode | Active visible channels | CPU cycles | Memory footprint |
|-----------------|-------------------------|------------|------------------|
| Standard field simulation | baryon and anti-baryon | 1.00 | 1.00 |
| Optimized SHBT simulation | baryon rendered, anti-baryon de-rendered | 0.20–0.30 | 0.30–0.40 |
| Reduction fraction | removed active anti-baryon render | 70–80% | 60–70% |

**Paper location:** Section 8.4, Eq. (146).

### Table 5: Physical Accounting in the Standard and Optimized Simulations

| Quantity | Standard simulation | Optimized SHBT simulation |
|----------|---------------------|---------------------------|
| Visible baryon channel | actively rendered | actively rendered |
| Visible anti-baryon channel | actively rendered | de-rendered |
| Passive stress-energy | explicitly evolved | preserved in dark completion |
| Baryon asymmetry output | dynamical target `η_B ≃ 6.1 × 10⁻¹⁰` | `η_B ≃ 6.1 × 10⁻¹⁰` |
| Stress-energy check | baseline conservation test | true |

**Paper location:** Section 8.4, after Table 4.

---

## 4. Numerical Benchmark Outputs

| Quantity | Paper Expected Value | Rust/Python Simulator Output | Accessor |
|----------|----------------------|------------------------------|----------|
| Branch | `(26, 8, 312)` | `(26, 8, 312)` | `report.branch` |
| Framing defect `Δ_fr` | `0` | `0.0` | `report.framing_defect` |
| Modular S commutator norm | `≃ 0` | `0.0` | `boundary_report.modular_S_commutator` |
| Modular T commutator norm | `≃ 0` | `0.0` | `boundary_report.modular_T_commutator` |
| Modular invariant | `true` | `True` | `report.modular_invariant` |
| Zero-energy lock | `true` | `True` | `report.zero_energy_locked` |
| Visible projection `26 → 4` | `true` | `True` | `report.projection_dimension_26_to_4` |
| `Z_boundary(i)` | `2.441381789163 × 10⁻²⁴` | `2.441381789163e-24` | `StaticBoundary.evaluate_z_boundary()` test |
| Loading density sum | `1` | `1.0` | `StaticBoundary.build_loading_density()` |
| Entanglement density sum | `1` | `1.0` | `StaticBoundary.build_entanglement_density()` |
| Dominant sequence length | `9` | `9` | `StaticBoundary.build_dominant_sequence()` |
| Bulk metric slices | `slice_count = 9` | `9` | `report.metric_slice_count` / `projection_report.slice_count` |
| Projector rank | `projector_rank = 3` | `3` | `projection_report.projector_rank` |
| Metric symmetry | `true` | `True` | `projection_report.symmetric` |
| Trace normalization | `true` slice-by-slice | `True` | `projection_report.trace_normalized` |
| Positive definiteness | all eigenvalues positive | `True` | `projection_report.positive_definite` |
| Local available bits `N_local` | `≃ N f_H²` | `2.535748254483999e+122` | `memory_report.local_available_bits` |
| Hidden bits `N_hidden` | `N − N_local` | `7.76249465658367e+121` | `memory_report.hidden_bits` |
| Entropy limit `N_limit` | `> 0` | `2.535748254483999e+122` | `memory_report.entropy_limit_bits` |
| Past-light-cone samples | `redshift_samples = 9` | `9` | `memory_report.past_light_cone_samples` |
| Property packets | `9` | `9` | `memory_report.property_packets` |
| Memory all passed | `true` | `True` | `report.memory_all_passed` |
| Baryon asymmetry `η_b` | `6.449923359416 × 10⁻¹⁰` | `6.449923359416e-10` | `report.eta_b` |
| Stress-energy preserved | `true` | `True` | `report.stress_energy_preserved` |
| CPU-cycle reduction | `50%` (benchmark) | `0.5` | `benchmark_delta.cpu_cycle_reduction_fraction` |
| Operation reduction | `≈ 99.85%` | `0.9984666852523` | `benchmark_delta.operation_reduction_fraction` |
| Memory reduction | `≈ 99.42%` | `0.9941785252263907` | `benchmark_delta.memory_reduction_fraction` |

---

## 5. Suggested Paper-Reference Update Mapping

Use the following replacements when rewriting paper paragraphs that currently point to `shbt_core.py`.

| Old `shbt_core.py` Reference | New Rust/Python Reference | Notes |
|------------------------------|---------------------------|-------|
| `StaticBoundary` | `shbt_simulator.ShbtSimulator().boundary` or `shbt_simulator.StaticBoundary` | Constructor fixes branch data automatically. |
| `StaticBoundary.framing_defect()` | `report.framing_defect` | Scalar `Float`/Python `float`. |
| `StaticBoundary.verify_equations()` | `ShbtSimulator.run_full_audit()` then `report.boundary_report` / `report.to_dict()['boundary_report']` | Returns `VerificationReport`. |
| `StaticBoundary.evaluate_Z_boundary(tau)` | `StaticBoundary.evaluate_z_boundary()` | Rust method name lowercased. |
| `StaticBoundary._build_su2_visible_block()` | `StaticBoundary.build_su2_visible_block()` | Public in Rust. |
| `StaticBoundary._build_su3_visible_block()` | `StaticBoundary.build_su3_visible_block()` | Public in Rust. |
| `StaticBoundary._build_raw_loading_density()` | `StaticBoundary.build_loading_density()` | Public in Rust. |
| `StaticBoundary._build_entanglement_density()` | `StaticBoundary.build_entanglement_density()` | Public in Rust. |
| `StaticBoundary._build_dominant_loading_sequence()` | `StaticBoundary.build_dominant_sequence()` | Public in Rust. |
| `StaticBoundary.entropy_self_resolution()` | `StaticBoundary.entropy_self_resolution()` | Returns `Vec<EntropyUpdate>`. |
| `StaticBoundary.derive_temporal_increment(H)` | `StaticBoundary.derive_temporal_increment()` | Accepts `Float` in Rust. |
| `HolographicProjection` | `ShbtSimulator.projection` or `shbt_simulator.HolographicProjection` |  |
| `HolographicProjection.derive_load_vector()` | `HolographicProjection.derive_load_vector()` |  |
| `HolographicProjection.metric_from_load_vector()` | `HolographicProjection.metric_from_load_vector()` |  |
| `HolographicProjection.project_static_block_to_bulk()` | `HolographicProjection.project_static_block_to_bulk()` |  |
| `HolographicProjection.verify_projection()` | `ShbtSimulator.run_full_audit().projection_report` | Returns `ProjectionReport`. |
| `BulkMetricSlice` | `shbt_simulator.BulkMetricSlice` / `report.metric_slices` | Each slice has `to_dict()`. |
| `CausalPoint` | `ShbtSimulator.causal_point` or `shbt_simulator.CausalPoint` |  |
| `CausalPoint._build_past_light_cone()` | `CausalPoint.build_past_light_cone()` | Public in Rust. |
| `CausalPoint.verify_memory_budget()` | `ShbtSimulator.run_full_audit().memory_report` | Returns `MemoryReport`. |
| `CausalPoint.crystallize_history()` | `CausalPoint.crystallize_history()` / `report.history_entries` | Returns `Vec<CoordinateLogEntry>`. |
| `LightConeSample` | `shbt_simulator.LightConeSample` | `#[pyclass]` with `to_dict()`. |
| `LocalPropertyPacket` | `shbt_simulator.LocalPropertyPacket` | `#[pyclass]` with `to_dict()`. |
| `CoordinateLogEntry` | `shbt_simulator.CoordinateLogEntry` | `#[pyclass]` with `to_dict()`. |
| `MemoryReport` | `shbt_simulator.MemoryReport` | `#[pyclass]` with `to_dict()`. |
| `BaryogenesisOptimizer` | `ShbtSimulator.optimizer` or `shbt_simulator.BaryogenesisOptimizer` |  |
| `BaryogenesisOptimizer.baryogenesis_identity()` | `ShbtSimulator.run_full_audit().baryogenesis_identity` | Returns `BaryogenesisIdentity`. |
| `BaryogenesisOptimizer.cpu_cycle_weight()` | `BaryogenesisOptimizer.cpu_cycle_weight()` |  |
| `BaryogenesisOptimizer.derender_antibaryon_charges()` | `BaryogenesisOptimizer.derender_antibaryon_charges()` |  |
| `BaryogenesisOptimizer.stress_energy_preserved()` | `report.stress_energy_preserved` | Also in `BenchmarkDelta`. |
| `BaryogenesisOptimizer.run_benchmark()` | `ShbtSimulator.run_full_audit().benchmark_delta` | Returns `BenchmarkDelta`. |
| `BaryogenesisIdentity` | `shbt_simulator.BaryogenesisIdentity` | `#[pyclass]` with `to_dict()`. |
| `FieldSimulation` / `BenchmarkDelta` | `shbt_simulator.FieldSimulation` / `shbt_simulator.BenchmarkDelta` | `#[pyclass]` with `to_dict()`. |
| `python shbt_core.py` | `python -c "import shbt_simulator; shbt_simulator.ShbtSimulator().run_full_audit()"` or `python examples/run_audit.py` | Also `python shbt_core.py` now auto-uses Rust if installed. |

---

## 6. Quick Reproducibility Checklist

```bash
# Rust tests
cargo test --release

# Rust build
cargo build --release

# Python audit (requires the .so to be importable as shbt_simulator)
cp target/release/libshbt_simulator.so target/release/shbt_simulator.so
PYTHONPATH=target/release python3 examples/run_audit.py

# Run from a configuration file (YAML or JSON)
PYTHONPATH=target/release python3 shbt_simulate.py --config config.default.yaml

# Override a config value from the command line
PYTHONPATH=target/release python3 shbt_simulate.py \
  --config config.default.yaml --mode baryogenesis --particles 1024 --seed 42

# Reproducible history with a fixed seed
PYTHONPATH=target/release python3 shbt_simulate.py --mode history --seed 123
```

Expected outputs: `branch = (26, 8, 312)`, `framing_defect = 0.0`, `modular_invariant = True`, `zero_energy_locked = True`, `projection_dimension_26_to_4 = True`, `eta_b = 6.449923359416e-10`, `stress_energy_preserved = True`, 9 metric slices, 9 crystallized history entries.

Result directories now include:
- `result.json` (or `result.h5` / `result_*.csv` depending on `export_formats`)
- `result.log` (plain-text reproducibility log with version, git commit, config summary)
- `result_run_info.json` (machine-readable version of the same metadata)

Set `seed` in the config or via `--seed` to make Causal Point collapse selections reproducible.
