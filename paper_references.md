# SHBT Paper-to-Code Reference Mapping

This file maps the major equations, audit tables, and numerical benchmarks in `main.pdf` (Sections 2–9, Tables 1–18) to the exact implemented code objects in the `shbt-precision` repository. It is intended as the single source of truth for later polishing passes (e.g. Prism) that replace generic code references with exact paths and names.

**Repository layout**
- Rust source: `src/shbt/` (boundary, entropy flow, baryogenesis, causal point) and `src/lib.rs` (module exports and `ShbtSimulator`).
- Python audit layer: `precision_cosmology.py` (the only Section 9 audit module; it subsumes the old `noether_bridge.py` and `precision_cosmology_engine.py` references in the paper text).
- Foundation-audit examples: `examples/run_audit.py`.

**Naming convention**
- `shbt_simulator` is the PyO3 module exposed by the Rust build.
- `StaticBoundary.c_dark` returns the *completed* ledger `1197103/362670`.
- `StaticBoundary.c_dark_residual` returns the *residual* ledger `834433/362670`.
- `StaticBoundary.c_dark_completion` is a Python alias for `c_dark`.

---

## 1. Global Entry Points

| Operation | Command / API |
|-----------|---------------|
| Import the PyO3 module | `import shbt_simulator` |
| Construct and run the full foundation audit | `shbt_simulator.ShbtSimulator().run_full_audit()` |
| Access a `ShbtReport` as a Python dict | `report.to_dict()` |
| Run Rust unit tests | `cargo test` or `cargo test --release` |
| Run the Python precision-cosmology tests | `python precision_cosmology.py --run-tests` |
| Run the Python simulator tests | `pytest tests/test_simulator.py -q` |
| Run the foundation-audit example | `python examples/run_audit.py` |
| Build the Python wheel | `maturin build --release` |

---

## 2. Branch and Constant Definitions

| Paper symbol | Paper value | Paper location | Rust / Python accessor |
|--------------|-------------|----------------|------------------------|
| Benchmark branch `b*` | `(26, 8, 312)` | Eq. (3), Eq. (21), Eq. (22) | `StaticBoundary.benchmark_branch` |
| `k_ℓ` (lepton level) | `26` | Eq. (3) | `StaticBoundary.lepton_level` |
| `k_q` (quark level) | `8` | Eq. (3) | `StaticBoundary.quark_level` |
| `K` (parent level) | `312` | Eq. (3) | `StaticBoundary.parent_level` |
| `I_ℓ*` | `6` | Eq. (22) | `StaticBoundary.i_l_star` |
| `I_q*` | `13` | Eq. (22) | `StaticBoundary.i_q_star` |
| `c_dark^res` | `834433/362670 ≈ 2.300805139659` | Eq. (5), Eq. (173) | `StaticBoundary.c_dark_residual` |
| `c_dark^comp` | `1197103/362670 ≈ 3.300805139659` | Eq. (6), Eq. (173) | `StaticBoundary.c_dark`; `precision_cosmology.load_completed_ledger()` |
| `Λ_holo` | `1.08913883e-52 m⁻²` | Eq. (175) | `StaticBoundary.lambda_holo_si_m2` |
| `N_sat` / bit budget | `≈ 3.312593327986e122` bits | Eq. (175) | `StaticBoundary.n_sat`; `StaticBoundary.bit_budget`; `precision_cosmology.load_default_constants().n_sat` |
| `H_0^CMB` | `67.4` km/s/Mpc | Eq. (175) | `StaticBoundary.h0_cmb`; `precision_cosmology.load_default_constants().h0_cmb` |
| `c` (speed of light) | `299_792_458.0` m/s | Section 7 | `causal_point.rs` `LIGHT_SPEED_M_PER_S` |
| `ℏ` | `1.054_571_817e-34` J·s | Section 7 | `causal_point.rs` `HBAR_J_S` |
| Planck mass `M_P` | `1.220_890e19` GeV | Section 6 | `baryogenesis.rs` `PLANCK_MASS_GEV` |
| GUT scale `M_GUT` | `2.0e16` GeV | Section 6 | `baryogenesis.rs` `GUT_SCALE_GEV` |
| Prime lattice `(p_0,…,p_4)` | `(2, 3, 5, 7, 11)` | Eq. (109) | `entropy_flow.rs` `metric_from_load_vector()` hard-coded prime array |
| `SU(3)` low weights | `[(0,0), (1,0), (0,1)]` | Eq. (53) | `boundary.rs` / `causal_point.rs` `LOW_SU3_WEIGHTS` |
| Charge embedding | `(22, 23, 26)` | Eq. (52) | `boundary.rs` `CHARGE_EMBEDDING` |

---

## 3. Equation-to-Code Index

### 3.1 Section 2 — Foundation Axioms

| Eq. | Mathematical object | Rust implementation | Python accessor / report field |
|-----|---------------------|---------------------|--------------------------------|
| (1)–(2) | Completed partition `Z_∂`, pairing matrix `M`, modular kernels `S_∂`, `T_∂` | `StaticBoundary` internal `z_boundary_matrix`, `s_boundary`, `t_boundary` | `ShbtReport.to_dict()['boundary_report']` |
| (3) | Canonical branch `(26,8,312)` | `StaticBoundary::new_with_branch` / `StaticBoundary::new` | `StaticBoundary.benchmark_branch` |
| (4) / (23) | Framing defect `Δ_fr` | `StaticBoundary.framing_defect()` in `src/shbt/boundary.rs` | `StaticBoundary.framing_defect_py`; `report.framing_defect` |
| (5) | `c_dark^res` | `StaticBoundary.c_dark_residual` | `StaticBoundary.c_dark_residual` |
| (6) | `c_dark^comp` | `StaticBoundary.c_dark` | `StaticBoundary.c_dark`; `precision_cosmology.load_completed_ledger()` |
| (7)–(8) | `N_sat = 3π/(L_P² Λ_holo)` | `StaticBoundary` constructor computes `bit_budget` / `n_sat` | `StaticBoundary.n_sat`; `StaticBoundary.bit_budget` |
| (9)–(11) | Closure tensor / topological Einstein equation | `StaticBoundary.verify_equations()` | `StaticBoundary.verify_equations_py()`; `report.boundary_report` |
| (12)–(17) | Integral spin closure | `StaticBoundary.build_s_boundary_static`, `build_t_boundary_static` | via `report.boundary_report['integral_spin_closure']` / `modular_invariant` |
| (18)–(19) | Affine central charges | `StaticBoundary.su2_central_charge(level)`, `su3_central_charge(level)` | not directly exposed; used by `verify_equations` |
| (20)–(22) | Integer locks `I_ℓ*`, `I_q*` | `StaticBoundary` fields `i_l_star`, `i_q_star` | `StaticBoundary.i_l_star`, `i_q_star` |
| (23)–(35) | Framing defect → Einstein lock chain | `StaticBoundary.verify_equations()` | `report.boundary_report` (`zero_energy_locked`, `modular_invariant`, etc.) |
| (36)–(40) | Static Hamiltonian cancellation | checked as part of `verify_equations` / `run_full_audit` | `report.boundary_report.all_passed` |

### 3.2 Section 3 — Modular Data and Partition Function

| Eq. | Mathematical object | Rust implementation | Python accessor / report field |
|-----|---------------------|---------------------|--------------------------------|
| (41)–(43) | `SU(2)` conformal weights / central charge | `StaticBoundary.su2_conformal_weight(label, level)`, `su2_central_charge(level)` | not directly exposed; used internally |
| (44) | `SU(2)` modular S/T entries | `StaticBoundary.su2_modular_s_entry(left, right, level)` | not directly exposed; used in visible block |
| (45)–(46) | `SU(3)` conformal weights / central charge | `StaticBoundary.su3_conformal_weight(p, q, level)`, `su3_central_charge(level)` | not directly exposed |
| (47)–(49) | `SU(3)` modular S/T via Weyl sum | `StaticBoundary.su3_modular_s_entry(left, right, level)`; internal `build_t_boundary_static` | not directly exposed |
| (50)–(51) | `SO(10)` central charge (parent) | `StaticBoundary` `parent_level` used in `BaryogenesisIdentity.Pi_rank` | `BaryogenesisIdentity.Pi_rank` via `report.baryogenesis_identity` |
| (52)–(55) | Visible 3×3 S/T blocks | `StaticBoundary.build_su2_visible_block()`, `build_su3_visible_block()`; internal `build_s_boundary_static`, `build_t_boundary_static` | not directly exposed; see `report.boundary_report` |
| (56)–(57) | Benchmark central charges / weights | `StaticBoundary.su2_conformal_weight`, `su3_conformal_weight`, `visible_central_charge()` | not directly exposed |
| (58)–(59) | Visible S block / T phases | `StaticBoundary.build_su2_visible_block()`, `build_su3_visible_block()` | not directly exposed |
| (60)–(65) | Modular invariance of `Z_∂` | `StaticBoundary.evaluate_z_boundary(tau)` | `StaticBoundary.evaluate_z_boundary_py(tau_re, tau_im)` |
| (66)–(72) | Pairing matrix commutant `M_∂` | `StaticBoundary` internal `z_boundary_matrix` | `report.boundary_report.modular_invariant` |
| (73)–(76) | Partition-function values at `τ = i`, `i+1`, `-1/i` | `StaticBoundary.evaluate_z_boundary(tau)` | `StaticBoundary.evaluate_z_boundary_py(...)`; tested in Rust `test_z_boundary_modular_invariance` |

### 3.3 Section 4 — Boundary Entropy Densities and Self-Resolution

| Eq. | Mathematical object | Rust implementation | Python accessor / report field |
|-----|---------------------|---------------------|--------------------------------|
| (78)–(79) | Visible coordinate lattice `C` | `StaticBoundary` internal indices | `StaticBoundary.build_dominant_sequence()` |
| (80)–(82) | Raw / normalized loading density `ρ_B` | `StaticBoundary.build_raw_loading_density_static`; `build_loading_density()` | `StaticBoundary.build_loading_density()` (Rust-only) |
| (83) | Entanglement density `ρ_E` | `StaticBoundary.build_entanglement_density()` | `StaticBoundary.build_entanglement_density()` (Rust-only) |
| (84) | Dominant loading sequence `σ` | `StaticBoundary.build_dominant_sequence()` | `StaticBoundary.build_dominant_sequence()` (Rust-only) |
| (85)–(94) | Entropy self-resolution, `D_n`, terminal `D_9 = 4` | `StaticBoundary.entropy_self_resolution()` | `report.boundary_report.projection_dimension_26_to_4` |
| (95)–(98) | Perception identity `Ṫ = H(t)` | `StaticBoundary.derive_temporal_increment(H)` | `StaticBoundary.derive_temporal_increment()` (Rust-only) |

### 3.4 Section 5 — Holographic RG Flow

| Eq. | Mathematical object | Rust implementation | Python accessor / report field |
|-----|---------------------|---------------------|--------------------------------|
| (99)–(103) | RG parameter `τ_n`, entropy cascade | `HolographicProjection.project_entropy_cascade()` | `report.metric_slices` / `report.to_dict()['metric_slices']` |
| (104)–(108) | Loading and entanglement densities | `StaticBoundary.build_loading_density()`, `build_entanglement_density()` | used by `project_entropy_cascade` |
| (109)–(117) | Prime lattice, load vector `ℓ_r`, Euler flux `Φ_s` | `HolographicProjection.derive_load_vector(state)`, `metric_from_load_vector(load, tau)` | `BulkMetricSlice` fields `load_vector`, `euler_flux` |
| (118)–(126) | Metric construction, stabilization, trace-1 normalization | `HolographicProjection.metric_from_load_vector()` | `BulkMetricSlice.metric_components`, `eigenvalues` |
| (127) | Metric-slice checks | `HolographicProjection.verify_projection(slices)` | `report.projection_report` |
| (128)–(129) | Spatial projector `P`, bulk metric `g_ab^bulk` | `HolographicProjection.project_static_block_to_bulk(metric)` | `BulkMetricSlice.spatial_metric` |
| (130)–(133) | Full holographic RG pipeline | `HolographicProjection.project_entropy_cascade()` | `report.metric_slices` |

### 3.5 Section 6 — Topological Baryogenesis

| Eq. | Mathematical object | Rust implementation | Python accessor / report field |
|-----|---------------------|---------------------|--------------------------------|
| (134) | Sphaleron coefficient `C_sph = 28/79` | `BaryogenesisOptimizer.baryogenesis_identity()` | `BaryogenesisIdentity.sphaleron_coefficient` |
| (135) | Topological Jarlskog `J_CP^topo` | `BaryogenesisOptimizer.baryogenesis_identity()` | `BaryogenesisIdentity.jarlskog_topological` |
| (136)–(137) | Rank projection `Π_rank`, modular restoration scale `M_N` | `BaryogenesisOptimizer.baryogenesis_identity()` | `BaryogenesisIdentity.Pi_rank`, `modular_restoration_scale_gev` |
| (138) | Baryon asymmetry `η_B` | `BaryogenesisOptimizer.baryogenesis_identity()` | `BaryogenesisIdentity.eta_b`; `ShbtReport.eta_b`; `report.eta_b` |
| (139) | Active render cost `C_B̄` | `BaryogenesisOptimizer.cpu_cycle_weight(charges)` | `BaryogenesisOptimizer.cpu_cycle_weight()` (Rust-only) |
| (140)–(143) | De-rendering operator, fixed point, stress-energy preservation | `BaryogenesisOptimizer.derender_antibaryon_charges()`, `stress_energy_preserved()` | `report.stress_energy_preserved`; `BenchmarkDelta.stress_energy_preserved` |

### 3.6 Section 7 — Causal Point and History Crystallization

| Eq. | Mathematical object | Rust implementation | Python accessor / report field |
|-----|---------------------|---------------------|--------------------------------|
| (144) | Horizon radius `R_H` | `CausalPoint.new_with_params()` | `MemoryReport.R_H_m` |
| (145) | Horizon fraction `f_H`, local area | `CausalPoint` fields | `MemoryReport.f_H` |
| (146) | `N_local`, `N_hidden`, `N_limit` | `CausalPoint` constructor / `verify_memory_budget()` | `MemoryReport.local_available_bits`, `hidden_bits`, `entropy_limit_bits` |
| (147)–(150) | Self-valuation `Σ`, `∇_obs Σ`, `a_obs` | `CausalPoint` constructor | `MemoryReport.sigma`, `localized_entropy_gradient_per_m`, `gravitational_acceleration_m_per_s2` |
| (151)–(152) | Observer Jacobian | `CausalPoint` internal (applied in `compute_property_packets` and `verify_memory_budget`) | `LocalPropertyPacket.metric_components` |
| (153)–(155) | Past-light-cone loading and effective expansion | `CausalPoint.build_past_light_cone()` | `LightConeSample` fields (`redshift`, `f_load`, `H_eff_per_s`, etc.) |
| (156)–(158) | Entropy cost, GET admissibility | `CausalPoint.verify_memory_budget()` | `MemoryReport.all_passed` |
| (159)–(163) | Collapse index, history density matrix, pointer packet | `CausalPoint.crystallize_history()` | `ShbtReport.history_entries`; `CoordinateLogEntry` fields |

---

## 4. Foundation Audit Tables

### Table 1: Affine algebraic parameters and central charges

| Affine factor | Level | Central charge | Rust source |
|---------------|-------|----------------|-------------|
| `SU(2)_kℓ` | `k_ℓ = 26` | `39/14` | `StaticBoundary.su2_central_charge(26)` |
| `SU(3)_kq` | `k_q = 8` | `64/11` | `StaticBoundary.su3_central_charge(8)` |
| `SO(10)_K` | `K = 312` | `351/8` | `StaticBoundary.parent_level` used by `BaryogenesisIdentity.Pi_rank` |
| `H_dark^comp` | – | `1197103/362670` | `StaticBoundary.c_dark`; `precision_cosmology.load_completed_ledger()` |

### Table 2: Boundary closure audit

| Audit quantity | Report accessor / method | Expected | Actual |
|----------------|--------------------------|----------|--------|
| Branch levels | `report.branch` | `(26, 8, 312)` | `(26, 8, 312)` |
| Framing defect | `report.framing_defect` | `0.0` | `0.0` |
| `Z_∂code(i)` | `StaticBoundary.evaluate_z_boundary_py(0.0, 1.0)` | `2.441381789163e-24` | `2.441381789163e-24` |
| Loading density sum | `report.boundary_report.loading_normalized` | `true` | `true` |
| Entanglement density sum | `report.boundary_report.entanglement_density_normalized` | `true` | `true` |
| Dominant sequence length | `StaticBoundary.build_dominant_sequence()` | `9` | `9` |
| Modular S commutator norm | `report.boundary_report.modular_S_commutator` | `≈ 0` | `0.0` |
| Modular T commutator norm | `report.boundary_report.modular_T_commutator` | `≈ 0` | `0.0` |
| Modular invariant | `report.modular_invariant` | `true` | `true` |
| Zero-energy lock | `report.zero_energy_locked` | `true` | `true` |
| Visible projection `26 → 4` | `report.projection_dimension_26_to_4` | `true` | `true` |

### Table 3: Holographic projection audit

| Audit quantity | Report accessor | Expected | Actual |
|----------------|-----------------|----------|--------|
| Entropy cascade length | `report.projection_report.slice_count` | `9` | `9` |
| Spatial projector rank | `report.projection_report.projector_rank` | `3` | `3` |
| Metric symmetry | `report.projection_report.symmetric` | `true` | `true` |
| Trace normalization | `report.projection_report.trace_normalized` | `true` | `true` |
| Positive definiteness | `report.projection_report.positive_definite` | `true` | `true` |

### Table 4: Causal Point audit

| Audit quantity | Report accessor | Expected | Actual |
|----------------|-----------------|----------|--------|
| Local available bits | `report.memory_report.local_available_bits` | `≃ N f_H²` | `2.535748254483999e122` |
| Hidden bits | `report.memory_report.hidden_bits` | `N − N_local` | `7.76249465658367e121` |
| Entropy limit | `report.memory_report.entropy_limit_bits` | `> 0` | `2.535748254483999e122` |
| Light-cone samples | `report.memory_report.past_light_cone_samples` | `9` | `9` |
| Property packets | `report.memory_report.property_packets` | `9` | `9` |
| Memory all passed | `report.memory_all_passed` | `true` | `true` |

### Table 5: Standard-versus-optimized field simulation cost

| Simulation mode | Active channels | CPU cycles | Operations | Memory |
|-----------------|-----------------|------------|------------|--------|
| Standard | baryon + anti-baryon | `1.0` | `1.0` | `1.0` |
| Optimized SHBT | baryon rendered, anti-baryon de-rendered | `0.5` | `0.001533314747` | `0.0058214747736093` |
| Reduction fraction | removed anti-baryon render | `0.5` | `0.998466685252` | `0.994178525226` |

*Code:* `BaryogenesisOptimizer.run_benchmark(particle_count)` → `BenchmarkDelta` → `ShbtReport.benchmark_delta`.

### Table 6: Physical accounting

| Quantity | Standard simulation | Optimized SHBT simulation |
|----------|---------------------|---------------------------|
| Visible baryon channel | actively rendered | actively rendered |
| Visible anti-baryon channel | actively rendered | de-rendered |
| Passive stress-energy | explicitly evolved | preserved in dark completion |
| Baryon asymmetry output | baseline | `η_B = 6.449923359416e-10` |
| Stress-energy check | baseline | `true` |

---

## 5. Precision Cosmology (`precision_cosmology.py`)

### 5.1 File-name note

The paper text mentions `noether_bridge.py` and `precision_cosmology_engine.py`. In this repository all of their functionality is consolidated into a single module: **`precision_cosmology.py`**. The tables below point only to objects in that file.

### 5.2 Section 9 constants and bridge quantities

| Symbol | Value | Paper equation | Python source |
|--------|-------|----------------|---------------|
| `c_dark^comp` | `1197103/362670` | Eq. (173) | `precision_cosmology.load_completed_ledger()` |
| `Δ_mod` | `c_dark^comp / 24` | Eq. (173) | `precision_cosmology.load_completed_ledger() / 24` |
| `Λ_holo` | `1.08913883e-52 m⁻²` | Eq. (175) | `precision_cosmology.load_default_constants().lambda_holo_si_m2` |
| `N_sat` | `3.312593327986e122` | Eq. (175) | `precision_cosmology.load_default_constants().n_sat` |
| `H_0^CMB` | `67.4` km/s/Mpc | Eq. (175) | `precision_cosmology.load_default_constants().h0_cmb` |
| `κ_D5` | `0.988769793998` | Appendix C / Table 7 | `src/shbt/baryogenesis.rs` `compute_kappa_d5` (private helper used by `baryogenesis_identity`); result reflected in `BaryogenesisIdentity.jarlskog_topological` |

### 5.3 Section 9 equation-to-function index

| Eq. | Description | Function / Report key | File |
|-----|-------------|-----------------------|------|
| (173) | `c_dark^res`, `c_dark^comp`, `Δ_mod` | `precision_cosmology.load_completed_ledger()`; `precision_cosmology.entropy_debt_uplift_factor(delta_mod)` | `precision_cosmology.py` |
| (174)–(175) | Bekenstein-Hawking / holographic bit bound | `precision_cosmology.load_default_constants()` | `precision_cosmology.py` |
| (176) | Loading fraction `f_load(z)`, entropy debt `S_debt(z)` | `precision_cosmology.compute_loading_fraction(...)`; `precision_cosmology.compute_entropy_debt(...)` | `precision_cosmology.py` |
| (177)–(180) | Light-cone clock and loading ODE | `precision_cosmology.loading_fraction_ode(...)`; `precision_cosmology.shbt_hubble_rate(...)` | `precision_cosmology.py` |
| (181)–(183) | Lock rate `Γ_lock = 3 A_H` | `precision_cosmology.lock_rate(A_H)`; `precision_cosmology.loading_amplitude(...)` | `precision_cosmology.py` |
| (194)–(197) | Local Hubble uplift `H_0^loc`, amplitude `A_H` | `precision_cosmology.h0_local(...)`; `precision_cosmology.loading_amplitude(...)` | `precision_cosmology.py` |
| (198)–(199) | Redshift-dependent intercept `H_0(z)` | `precision_cosmology.h0_redshift_dependent(z, ...)` | `precision_cosmology.py` |
| (200)–(203) | Linear growth ODE | `precision_cosmology.growth_ode_system(...)` | `precision_cosmology.py` |
| (204) | SHBT Hubble rate `H_SHBT(z)` | `precision_cosmology.shbt_hubble_rate(z, ...)` | `precision_cosmology.py` |
| (206) | Growth suppression `(fσ_8)_SHBT / (fσ_8)_ΛCDM − 1` | `precision_cosmology.compute_growth_suppression(z, ...)` | `precision_cosmology.py` |
| (200)–(206) + spherical collapse | Cluster-collapse linear threshold `δc(z)`, mass variance `σM(z)`, peak height `ν`, and Press-Schechter abundance ratio | `precision_cosmology.compute_cluster_collapse(z, ...)`; `precision_cosmology._cluster_collapse_delta_c(...)` | `precision_cosmology.py` |
| (207)–(211) | ISW residual `Δ_ISW(z)` | `precision_cosmology.isw_residual(z, ...)` | `precision_cosmology.py` |
| (212)–(217) | BBN loading shift / stability | `precision_cosmology.bbn_stability_check(z_bbn, ...)`; internal `_bbn_components` | `precision_cosmology.py` |
| (188)–(193) | Neutrino hierarchy masses and tensions | `precision_cosmology.neutrino_hierarchy_masses(...)` | `precision_cosmology.py` |
| (222)–(226) | GET measurement cost | `precision_cosmology.get_measurement_cost(...)` | `precision_cosmology.py` |
| (227) | Deterministic collapse index `ι` | `precision_cosmology.collapse_index(...)` | `precision_cosmology.py` |
| (173)–(231) | Full Section 9 audit | `precision_cosmology.build_precision_cosmology_report(...)` | `precision_cosmology.py` |

### 5.4 Section 9 table mappings

| Paper table | Report key / function | Notes |
|-------------|-----------------------|-------|
| Table 7 (Noether bridge) | `precision_cosmology.load_default_constants()`; `precision_cosmology.load_completed_ledger()`; `precision_cosmology.neutrino_hierarchy_masses(...)` | `κ_D5` is internal to `baryogenesis_identity`; `m_ν,1` and hierarchy sums are produced by `neutrino_hierarchy_masses` |
| Table 8 (Neutrino hierarchy) | `precision_cosmology.neutrino_hierarchy_masses(...)` | Returns `normal_hierarchy` and `inverted_hierarchy` dicts with `m1..m3`, `sum_meV`, `tension_sigma` |
| Table 9 (Code-to-math map) | Use the equation index above; cluster-collapse entry maps `δc, σM, abundance ratio` → `precision_cosmology.compute_cluster_collapse(...)` → `report['cluster_collapse']` (Table 13) | This table is a cross-reference; the new canonical source is `precision_cosmology.py` |
| Table 10 (Redshift ladder) | `report['redshift_ladder']` from `build_precision_cosmology_report(...)` | Each row contains `z`, `loading_term_km_s_mpc`, `h0_z_km_s_mpc` |
| Table 11 (Growth suppression) | `report['growth_suppression']` from `build_precision_cosmology_report(...)` | Produced by `compute_growth_suppression` for `GROWTH_AUDIT_REDSHIFTS` |
| Table 12 (High-redshift mirage) | `report['cpl_template']` from `build_precision_cosmology_report(...)` | Internal `_cpl_template` computes `w0`, `wa`, `density_zero_crossing_redshift`, `density_ratio_z1_5` |
| Table 13 (Cluster-collapse) | `precision_cosmology.compute_cluster_collapse(z, h0_cmb, A_H, omega_m, sigma8, ...)`; `report['cluster_collapse']` from `build_precision_cosmology_report(...)` | Rows for `z = 0, 0.5, 1.0` contain `z`, `scale_factor`, `lcdm_delta_c`, `shbt_delta_c`, `lcdm_sigma_mass`, `shbt_sigma_mass`, `lcdm_peak_height`, `shbt_peak_height`, `abundance_ratio`, `delta_c_shift_percent`, `sigma_mass_suppression_percent` |
| Table 14 (Light-cone ledger) | `report['lightcone_entropy_debt']` from `build_precision_cosmology_report(...)` | Rows contain `z`, `h0_z`, `h_shbt`, `f_load`, `S_debt_bits` |
| Table 15 (ISW stability) | `report['isw_stability']` from `build_precision_cosmology_report(...)` | Produced by `isw_residual` for `ISW_AUDIT_REDSHIFTS` |
| Table 16 (Cosmic chronometers) | `report['cosmic_chronometer_validation']` | Hard-coded `χ² = 30.16`, `ν = 29`, `χ²_ν = 1.04` from paper |
| Table 17 (Summary ledger) | `report['summary_table_17']` from `build_precision_cosmology_report(...)` | Includes `local_uplift`, `gradient_target`, `cpl_template`, `bbn_loading_shift`, `chronometer`, `forecast_chi2_sensitivity`, `cosmic_age_gyr`, `thermodynamic_arrow`, `overall_precision_audit` |
| Table 18 (GET cost metrics) | `precision_cosmology.get_measurement_cost(...)` and `precision_cosmology.collapse_index(...)` | Implements `C_addr`, `C_ens`, `C_get`, `R_entropy`, collapse index `ι` |

### 5.5 Cluster-collapse model (Table 13)

The dedicated cluster-collapse audit was previously marked as not implemented. It is now exposed through `precision_cosmology.compute_cluster_collapse(...)`, which uses the same background expansion and growth ODE as the other Section 9 audits.

| Quantity | Equation / model | Code implementation | Report field |
|----------|------------------|---------------------|--------------|
| Spherical top-hat linear overdensity `δc(z)` | Calibrated spherical-collapse threshold using the matter density `Ωm(z)` and Hubble intercept slope `dln H0(z) / d ln a` from Eqs. (200)–(206) | `precision_cosmology._cluster_collapse_delta_c(...)`; called by `compute_cluster_collapse(...)` | `row['lcdm_delta_c']`, `row['shbt_delta_c']` |
| Mass variance `σM(z) = σM(0) D(z) / D(0)` | Linear growth factor `D(z)` from Eq. (203) / `compute_growth_suppression`; `σM(0)` normalized to the reference value `reference_sigma_mass_z0 * (sigma8 / 0.812)` | `precision_cosmology.compute_cluster_collapse(...)` | `row['lcdm_sigma_mass']`, `row['shbt_sigma_mass']` |
| Peak height `ν = δc / σM` | Direct ratio of calibrated `δc` and `σM(z)` | `precision_cosmology.compute_cluster_collapse(...)` | `row['lcdm_peak_height']`, `row['shbt_peak_height']` |
| Abundance ratio `n_SHBT / n_LCDM` | Press-Schechter multiplicity function `f(ν) ∝ ν exp(-ν²/2)`; ratio cancels the common prefactors | `precision_cosmology.compute_cluster_collapse(...)` | `row['abundance_ratio']` |

---

## 6. Bidirectional Replacement Guide (Old → New)

When rewriting paper paragraphs, replace the left-hand references with the right-hand references.

| Old reference in `main.pdf` | New exact reference |
|-----------------------------|---------------------|
| `shbt_core.py` | `precision_cosmology.py` or `examples/run_audit.py` |
| `noether_bridge.py` | `precision_cosmology.py` (functions `load_default_constants`, `load_completed_ledger`, `neutrino_hierarchy_masses`) |
| `precision_cosmology_engine.py` | `precision_cosmology.py` |
| `StaticBoundary.evaluate_Z_boundary(tau)` | `StaticBoundary.evaluate_z_boundary_py(tau_re, tau_im)` (Python) or `StaticBoundary.evaluate_z_boundary(tau: Complex)` (Rust) |
| `StaticBoundary._build_su2_visible_block()` / `_build_su3_visible_block()` | `StaticBoundary.build_su2_visible_block()` / `build_su3_visible_block()` |
| `StaticBoundary._build_raw_loading_density()` | `StaticBoundary.build_loading_density()` |
| `StaticBoundary._build_entanglement_density()` | `StaticBoundary.build_entanglement_density()` |
| `StaticBoundary._build_dominant_loading_sequence()` | `StaticBoundary.build_dominant_sequence()` |
| `CausalPoint._build_past_light_cone()` | `CausalPoint.build_past_light_cone()` |
| `shbt_simulator.StaticBoundary.build_loading_density()` (Python call) | *Not directly exposed to Python*; use `ShbtSimulator().run_full_audit().to_dict()` and read `boundary_report` / `metric_slices` / `history_entries` |
| `report.c_dark` (residual) | `StaticBoundary.c_dark_residual` or `report['simulator_constants']['c_dark_residual']` |
| `report.c_dark` (completion) | `StaticBoundary.c_dark` or `report['completed_ledger']` |

---

## 7. Python API Notes

- `StaticBoundary` Python-exposed methods / getters:
  - `__init__()` (default branch) and `with_branch(lepton, quark, parent)`.
  - `benchmark_branch`, `lepton_level`, `quark_level`, `parent_level`, `i_l_star`, `i_q_star`.
  - `c_dark`, `c_dark_residual`, `c_dark_completion`, `lambda_holo`, `lambda_holo_si_m2`, `bit_budget`, `n_sat`, `h0_cmb`.
  - `framing_defect_py()`.
  - `verify_equations_py()`.
  - `evaluate_z_boundary_py(tau_re, tau_im)`.
- `HolographicProjection`, `BaryogenesisOptimizer`, `CausalPoint` are exported as Python classes but their core methods (`derive_load_vector`, `metric_from_load_vector`, `cpu_cycle_weight`, `build_past_light_cone`, etc.) are **Rust-only**. Consume their outputs through `ShbtSimulator().run_full_audit()`.
- `ShbtReport` exposes getters: `branch`, `eta_b`, `stress_energy_preserved`, `metric_slice_count`, `history_entry_count`, `framing_defect`, `modular_invariant`, `zero_energy_locked`, `projection_dimension_26_to_4`, `slice_count`, `projection_all_passed`, `memory_all_passed`, and `to_dict()`.

---

## 8. Quick Verification

Run these commands from the repository root to reproduce the mappings above.

```bash
# Rust foundation tests (boundary, projection, baryogenesis, causal point)
cargo test

# Rust release build
cargo build --release

# Python simulator tests (requires a compiled/importable shbt_simulator)
pytest tests/test_simulator.py -q

# Foundation audit example (uses ShbtSimulator.run_full_audit)
python examples/run_audit.py

# Section 9 precision-cosmology unit tests
python precision_cosmology.py --run-tests

# Section 9 full report (JSON to stdout)
python precision_cosmology.py --json

# Wheel build (optional)
maturin build --release
```

**Expected key outputs**
- `report.branch == (26, 8, 312)`
- `report.framing_defect == 0.0`
- `report.modular_invariant == True`
- `report.zero_energy_locked == True`
- `report.projection_dimension_26_to_4 == True`
- `report.projection_report.slice_count == 9`
- `report.eta_b == 6.449923359416e-10`
- `report.stress_energy_preserved == True`
- `precision_cosmology.py --run-tests` prints `OK` (12 tests)
- `report['cluster_collapse'][0]['lcdm_delta_c']` ≈ `1.6760` at `z = 0`
- `report['cluster_collapse'][0]['shbt_delta_c']` ≈ `1.6733` at `z = 0`
- `report['cluster_collapse'][0]['abundance_ratio']` ≈ `6.10E-1` at `z = 0`

---

## 9. Audit Report Field Reference

### `ShbtReport.to_dict()` top-level fields

| Field | Rust type | Description |
|-------|-----------|-------------|
| `branch` | `(u32, u32, u32)` | Canonical branch `(26, 8, 312)` |
| `boundary_report` | `VerificationReport` | Boundary closure audit (Table 2) |
| `projection_report` | `ProjectionReport` | Metric projection audit (Table 3) |
| `memory_report` | `MemoryReport` | Causal Point audit (Table 4) |
| `benchmark_delta` | `BenchmarkDelta` | Standard vs. optimized field simulation (Tables 5–6) |
| `baryogenesis_identity` | `BaryogenesisIdentity` | Section 6 identity values |
| `eta_b` | `f64` | Baryon asymmetry `6.449923359416e-10` |
| `stress_energy_preserved` | `bool` | Passive stress-energy equality |
| `metric_slices` | `List[BulkMetricSlice]` | 9 entropy-cascade metric slices |
| `history_entries` | `List[CoordinateLogEntry]` | 9 crystallized history entries |

### `build_precision_cosmology_report(...)` top-level keys

| Key | Source function | Paper table / equation |
|-----|-----------------|------------------------|
| `completed_ledger` | `load_completed_ledger()` | Eq. (173) / Table 7 |
| `uplift_factor` | `entropy_debt_uplift_factor(delta_mod)` | Eq. (194) |
| `h0_local_km_s_mpc` | `h0_local(...)` | Eq. (196) |
| `A_H_km_s_mpc` | `loading_amplitude(...)` | Eq. (197) |
| `Gamma_lock_km_s_mpc` | `lock_rate(A_H)` | Eq. (183) |
| `redshift_ladder` | `h0_redshift_dependent(...)` | Table 10 / Eq. (199) |
| `lightcone_entropy_debt` | `compute_loading_fraction(...)` / `compute_entropy_debt(...)` / `shbt_hubble_rate(...)` | Table 14 / Eqs. (176), (180), (204) |
| `growth_suppression` | `compute_growth_suppression(...)` | Table 11 / Eq. (206) |
| `cluster_collapse` | `compute_cluster_collapse(...)` | Table 13 |
| `isw_stability` | `isw_residual(...)` | Table 15 / Eq. (211) |
| `bbn_stability` | `bbn_stability_check(...)` | Eqs. (214)–(217) |
| `neutrino_hierarchy` | `neutrino_hierarchy_masses(...)` | Table 8 / Eqs. (188)–(193) |
| `cpl_template` | `_cpl_template(...)` | Table 12 |
| `forecast_sensitivity` | `_forecast_sensitivity(...)` | Table 17 |
| `cosmic_chronometer_validation` | hard-coded | Table 16 / Eqs. (219)–(221) |
| `cosmic_age_gyr` | `_cosmic_age_gyr(...)` | Table 17 |
| `thermodynamic_arrow` | `_thermodynamic_arrow(...)` | Table 17 / Eq. (178) |
| `summary_table_17` | composite of above | Table 17 |

---

*Last updated to match the merged `ShbtSimulator` API and `precision_cosmology.py` as of the current session.*
