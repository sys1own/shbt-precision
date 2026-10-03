# SHBT Equation-to-Code Traceability Contract & Verification Ledger

**Repository:** `sys1own/shbt-precision`
**Primary publication:** `main.tex` → `main.pdf`
**Supplementary monograph:** `supplementary.tex` → `supplementary.pdf`
**Computational core:** `src/shbt/` (Rust), `src/lib.rs` (PyO3 bindings), `boltzmann_shbt.py`, `precision_cosmology.py`, `shbt_simulate.py`

This document is the authoritative bidirectional contract between the published SHBT manuscripts and their executable proof. Every equation, audit table, and numerical claim in `main.pdf` / `supplementary.pdf` is bound to an implementing function and a serialized artifact; conversely, every reported value in `result.json` and the `shbt_run_*.csv` data products is traceable to a paper-level identity. All values in Section 4 were extracted from a live `cargo test --release` + `python3 shbt_simulate.py --mode all --output result.json` run — nothing is estimated.

---

## 1. Header & Traceability Scope

The contract is three-way:

- **Theory → Code.** Each numbered equation and audit table in `main.tex` (foundation axioms §2, modular data §3, entropy densities §4, holographic RG flow §5, topological baryogenesis §6, causal point §7, numerical verification §8, precision cosmology §9, calorimetry §10) and each proof in `supplementary.tex` (modular framing phases, character asymptotics, dark-sector ledger arithmetic) maps to a concrete Rust or Python entry point listed in Section 2.
- **Code → Artifacts.** The Rust core executes through the PyO3 module `shbt_simulator`; the Python layer serializes results to `result.json` and four `shbt_run_*.csv` products (Section 3, Table 39).
- **Artifacts → Verification.** `result.json` is the machine-readable ledger. Its `audit` record carries the live `ShbtReport`; its `precision_cosmology` record carries the Section 9 report; its `precision_pipeline` record carries spectra, calorimetry, and non-Gaussianity outputs; its `stability_audit` record carries the thermodynamic debt schedule.

### 1.1 Reproduction mechanisms

| Mechanism | Command / entry point |
|-----------|------------------------|
| Rust unit tests (boundary, projection, baryogenesis, causal point, stability) | `cargo test --release` |
| Build + stage the PyO3 module | `python3 shbt_simulate.py --build` (runs `cargo build --release`, copies `target/release/libshbt_simulator.so` → `shbt_simulator.so`) |
| Foundation audit only | `python3 shbt_simulate.py --mode audit --output result.json` or `python3 examples/run_audit.py` |
| Full pipeline (audit + cosmology + spectra + calorimetry + non-Gaussianity) | `PYTHONPATH=target/release:. python3 shbt_simulate.py --mode all --output result.json` |
| Section 9 report standalone | `python3 precision_cosmology.py --json` |
| Python unit suites | `pytest tests/ -q`; `python3 precision_cosmology.py --run-tests`; `python3 boltzmann_shbt.py --run-tests` |
| Document compilation | `make pdf` (`pdflatex` × 2 on `main.tex`) |
| Visualizer telemetry frame | `python3 shbt_simulate.py --mode visualize --export-webgpu-telemetry data/telemetry.bin` |
| WebGPU harness | `wasm-bindgen --target web --out-dir visualizer/pkg src/shbt/visualizer/target/wasm32-unknown-unknown/release/shbt_visualizer.wasm`, then serve `visualizer/` |
| Headless GPU benchmark | `cargo run --release --manifest-path src/shbt/visualizer/Cargo.toml --bin headless -- 60 262144` |

**Naming conventions**

- `shbt_simulator` is the PyO3 extension module produced by `src/lib.rs` (crate `shbt_simulator`, module name `anyon_simulator` internally for exception types).
- `StaticBoundary.c_dark` returns the *completed* dark ledger 1197103/362670; `StaticBoundary.c_dark_residual` returns the *residual* ledger 834433/362670; `c_dark_completion` is an alias of `c_dark`.
- `HolographicProjection`, `BaryogenesisOptimizer`, and `CausalPoint` expose their deep methods in Rust only; in Python their results are consumed through `ShbtSimulator().run_full_audit().to_dict()`.

---

## 2. Master Equation-to-Code Traceability Matrix

This is the Markdown mirror of the paper's master code-traceability ledger (`tab:master-code-traceability-28` in `main.tex`).

| Paper Section / Eq. No. | Physical Phenomenon / Formal Identity | Source File & Function / Struct | Simulation Artifact / JSON Key |
|---|---|---|---|
| §2, Eqs. (1)–(11) | Completed boundary partition Z<sub>∂</sub>, pairing matrix M, modular kernels S<sub>∂</sub>, T<sub>∂</sub>; topological Einstein lock | `src/shbt/boundary.rs` — `StaticBoundary` internal `z_boundary_matrix`, `s_boundary`, `t_boundary`; `StaticBoundary::verify_equations` | `result.json → audit.boundary_report` |
| §2, Eq. (3) / (21)–(22) | WZW affine branch (*k*<sub>*ℓ*</sub>, *k*<sub>*q*</sub>, *K*) = (26, 8, 312); integer center lifts (*I*<sub>*ℓ*</sub>, *I*<sub>*q*</sub>) = (6, 13) | `src/shbt/boundary.rs` — `StaticBoundary::new_with_branch`, `StaticBoundary::new` | `result.json → audit.branch`; getters `benchmark_branch`, `i_l_star`, `i_q_star` |
| §2, Eq. (4) / (23); supp. §1 | Framing defect Δ<sub>fr</sub> ≡ 0 ⟹ *E*<sub>μν</sub> ≡ 0 | `src/shbt/boundary.rs` — `StaticBoundary::framing_defect` (PyO3: `framing_defect_py`), `verify_equations` | `result.json → audit.boundary_report.framing_defect` (= 0.0), `audit.boundary_report.zero_energy_locked` |
| §3, Eqs. (41)–(59) | Affine central charges and modular S/T entries for SU(2)<sub>26</sub>, SU(3)<sub>8</sub>, SO(10)<sub>312</sub> | `src/shbt/boundary.rs` — `su2_conformal_weight`, `su3_conformal_weight`, `su2_central_charge`, `su3_central_charge`, `su2_modular_s_entry`, `su3_modular_s_entry`, `build_su2_visible_block`, `build_su3_visible_block` | consumed by `verify_equations`; `audit.boundary_report.modular_invariant` |
| §3, Eqs. (60)–(76); supp. §3 | Modular invariance Z<sub>∂</sub>(τ+1) = Z<sub>∂</sub>(−1/τ) = Z<sub>∂</sub>(τ); Weil-orthogonal dark pairing | `src/shbt/boundary.rs` — `evaluate_z_boundary`, `evaluate_z_dark`, `build_dark_modular_data`, `dark_weil_orthogonality_check`, `classify_modular_completions`, `defect_free_completion_count` | `result.json → audit.boundary_report.modular_S_commutator`, `.modular_T_commutator` (both 0.0) |
| §4, Eqs. (78)–(98) | Boundary entropy densities ρ<sub>B</sub>, ρ<sub>E</sub>; dominant loading sequence; entropy self-resolution D<sub>9</sub> = 4; perception identity Ṫ = *H*(*t*) | `src/shbt/boundary.rs` — `build_loading_density`, `build_entanglement_density`, `build_dominant_sequence`, `entropy_self_resolution`, `derive_temporal_increment` | `result.json → audit.boundary_report.loading_normalized`, `.entanglement_density_normalized`, `.projection_dimension_26_to_4` |
| §5, Eqs. (99)–(133) | Holographic RG flow; entropy cascade; prime lattice load vector; Fefferman–Graham metric slices | `src/shbt/entropy_flow.rs` — `HolographicProjection::project_entropy_cascade`, `derive_load_vector`, `metric_from_load_vector`, `verify_projection`, `project_static_block_to_bulk` | `result.json → audit.metric_slices` (9 slices), `audit.projection_report` |
| Numerics | 512-bit arbitrary-precision arithmetic (`rug`/MPFR, `PREC`/`EVAL_PREC` = 512); symplectic-order energy-momentum bookkeeping with residuals below 10<sup>−122</sup> | `src/shbt/boundary.rs` (`PREC`), `src/lib.rs` (`EVAL_PREC`), `src/shbt/entropy_flow.rs` (Float kernels) | `result.json → audit.*` reports computed entirely in 512-bit Float |
| §6, Eqs. (134)–(143) | Topological baryogenesis: *C*<sub>sph</sub> = 28/79, *J*<sub>CP</sub><sup>topo</sup>, rank projection Π<sub>rank</sub>, restoration scale *M*<sub>N</sub>, asymmetry η<sub>B</sub> | `src/shbt/baryogenesis.rs` — `BaryogenesisOptimizer::baryogenesis_identity` → `BaryogenesisIdentity`; `derender_antibaryon_charges`, `stress_energy_preserved`, `cpu_cycle_weight`, `run_benchmark` | `result.json → audit.baryogenesis_identity` (`eta_b`, `sphaleron_coefficient`, `Pi_rank`, `jarlskog_topological`, `modular_restoration_scale_gev`), `audit.eta_b`, `audit.benchmark_delta` |
| §7, Eqs. (144)–(163) | Causal point memory: horizon *R*<sub>H</sub>, fraction *f*<sub>H</sub>, *N*<sub>local</sub>/*N*<sub>hidden</sub>/*N*<sub>limit</sub>; GET admissibility *C*<sub>get</sub> ≤ max(1, log<sub>2</sub> \|*R*\|); Landauer bound *Q*<sub>H</sub> ≥ *k*<sub>B</sub>*T* ln 2 · *C*<sub>op</sub>; history crystallization | `src/shbt/causal_point.rs` — `CausalPoint::new_with_params`, `build_past_light_cone`, `verify_memory_budget`, `crystallize_history` → `MemoryReport`, `LightConeSample`, `CoordinateLogEntry` | `result.json → audit.memory_report`, `audit.history_entries` (9 entries) |
| §7, observer-succession eqs. | Admissibility predicate *P*<sub>adm</sub>(*A*) = Θ(*R*<sub>entropy</sub>) with complexity floor *C*<sub>get</sub> = max(1, log<sub>2</sub>\|*R*\| + log<sub>2</sub>\|Ω<sub>A</sub>\|, *C*<sub>req</sub>); sub-threshold nodes abort with `AnomalyClosureError` | `src/shbt/causal_point.rs` — `CausalPoint::is_admissible`, `entropy_budget_residual`, `retrieval_cost_bits`; `CausalPointCandidate` | `result.json → succession.records[].entropy_residual_bits` |
| §7, observer-succession eqs. | Stinespring de-rendering channel *E*<sub>term</sub>(ρ) = Tr<sub>active</sub>(V ρ V†) ∈ *H*<sub>dark</sub>; pointer triad Ψ<sub>ι</sub> → (0, 0, 1); η<sub>D</sub> = 23/33, η<sub>V</sub> = 10/33; Kojima Ent(φ) = 0 | `src/shbt/causal_point.rs` — `CausalPoint::terminate_and_derender` → `DerenderingRecord` | `result.json → succession.records[].eta_dark`, `.trace_norm` |
| §7, observer-succession eqs. | Succession transfer kernel *T*(*A*<sub>term</sub> → *A*<sub>next</sub>) over lattice C = {0,1,2}² mediated by symplectic *T*<sup>∂</sup><sub>ij</sub> ∈ Sp(2g, ℤ); Σ<sub>*A*′</sub> *T* = 1 | `src/shbt/causal_point.rs` — `CausalPoint::build_succession_candidates`, `evaluate_succession_kernel`, `relabel_and_rerender`, `run_lifecycle_cycle`; `ShbtSimulator::run_succession_cycles` | `result.json → succession.records[].kernel_probabilities`, `.kernel_normalized` |
| §7, observer-succession eqs. | Five-phase closed lifecycle Render → Crystallize → De-render → Relabel → Re-render | `src/shbt/causal_point.rs` — `LifecyclePhase`, `phase` field on `CausalPoint` | `result.json → succession.records[].phase` |
| §7, observer-succession eqs. | Cosmological horizon saturated freeze: *R*<sub>adm</sub>(z) → ∅ as z → −1 (f<sub>load</sub> → 1); deterministic horizon freeze without Friedmann bounce | `precision_cosmology.py` — `loading_fraction_asymptotic`, `observer_admissible_set`, `asymptotic_observer_freeze` | `result.json → foundation_audit.asymptotic_observer_freeze` (freeze redshift, kernel denominator → 0) |
| §9, Eqs. (173)–(231) | Precision cosmology: completed ledger, Hubble loading law, growth suppression *f*σ<sub>8</sub>, cluster collapse, dark-matter ghost density, ISW, BBN, neutrinos, GET cost, 7-parameter joint MCMC over cosmic chronometers | `precision_cosmology.py` — `load_completed_ledger`, `entropy_debt_uplift_factor`, `h0_local`, `h0_redshift_dependent`, `shbt_hubble_rate`, `compute_loading_fraction`, `compute_growth_suppression`, `compute_cluster_collapse`, `compute_dark_matter_density`, `compute_dm_baryon_ratio`, `isw_residual`, `bbn_stability_check`, `neutrino_hierarchy_masses`, `get_measurement_cost`, `collapse_index`, `run_mcmc_analysis`, `build_precision_cosmology_report` | `result.json → precision_cosmology.*` (see Table 39) |
| §9 Boltzmann pipeline | Scalar CMB *C*<sub>ℓ</sub><sup>TT</sup>, *C*<sub>ℓ</sub><sup>EE</sup>, *C*<sub>ℓ</sub><sup>TE</sup>; matter *P*(*k*, *z*); tensor *C*<sub>ℓ</sub><sup>BB</sup> | `boltzmann_shbt.py` — `compute_cmb_power_spectra`, `compute_tensor_power_spectra` | `shbt_run_cmb_cls.csv`, `shbt_run_matter_pk.csv`, `shbt_run_tensor_cls.csv`; `result.json → precision_pipeline.spectra` |
| §9 non-Gaussianity | Bispectrum/trispectrum templates *f*<sub>NL</sub>, *g*<sub>NL</sub>, τ<sub>NL</sub> | `precision_cosmology.py` — `compute_non_gaussianity_shapes` | `result.json → precision_pipeline.non_gaussianity` |
| §10 calorimetry | Sub-10 mK Landauer calorimetry; address-scaled heat ledger *Q*<sub>H0</sub> vs *Q*<sub>H1</sub>; OLS/MLE regression and model selection | `precision_cosmology.py` — `simulate_calorimetry_experiment`; orchestrated by `shbt_simulate.py` — `run_simulation_pipeline` | `shbt_run_calorimetry_sim.csv`; `result.json → precision_pipeline.calorimetry_csv` |
| §9, Thm. (Dynamic Bit-Loading Equivalence) | Ṫ = (1/*N*<sub>sat</sub>) d*S*<sub>index</sub>/d*t* = *H*(*t*); no inflaton sector | `src/shbt/cosmology.rs` — `ShbtUniverse::evaluate_hubble_index_rate`, `conformal_loading_fraction`, `hubble` | `result.json → cosmology_visualization`; `tests/cosmological_invariants.rs` |
| §9, Thm. (Passive Ghost Dark Matter) | η<sub>D</sub> = 23/33, η<sub>A</sub> = 10/33; *w*<sub>DM</sub> = 0; Ω<sub>DM,0</sub> = *c*<sub>dark</sub><sup>comp</sup>/12 = 0.260000 | `src/shbt/cosmology.rs` — `ShbtUniverse::dark_completion_bits`, `active_visible_bits`, `dark_matter_fraction` | `data/telemetry.bin` header (`dark_bits`, `active_bits`) |
| §9, Thm. (Mass-Congestion Condensation) | *M*<sub>seed</sub> = α<sub>seed</sub> Δ*N*, α<sub>seed</sub> = 1.3258316 × 10<sup>−51</sup> *M*<sub>☉</sub>/bit; *P*<sub>debt</sub> = (*M*<sub>seed</sub>/*M*<sub>☉</sub>) × 906 GW | `src/shbt/cosmology.rs` — `ShbtUniverse::compute_seed_condensation`, `eval_landauer_debt_power`, `ALPHA_SEED_MSUN_PER_BIT` | SHBT-MMIO `SeedDefectRecord` payload |
| §9, Thm. (Causal-Point History Crystallization and GET Clustering) | *R*<sub>entropy</sub> = *N*<sub>limit</sub> − *C*<sub>get</sub> ≥ 0; rank-one Π<sub>*A*,*ι*</sub>; **a**<sub>GET</sub>(**x**) = −κ<sub>GET</sub> ∇ln ρ<sub>proj</sub>(**x**); ℛ<sub>adm</sub>(*z* → −1) → ∅ | `src/shbt/visualizer/src/shaders/causal_point_get.wgsl` — `cs_causal_point_get`, `compute_kappa_get`; `src/shbt/visualizer/src/engine.rs` — `WasmShbtEngine::update_epoch`; `src/hud.rs` — `HorizonLedger` | `result.json → causal_point_simulation`; `src/shbt/visualizer/tests/invariant_gates.rs` |
| §9, Thm. 9.8 (First-Principles κ<sub>GET</sub>) | κ<sub>GET</sub>(*f*<sub>load</sub>) = (1/(*d*<sub>1</sub> − *h*<sup>∨</sup><sub>su(3)</sub>))(1 + (*c*<sub>eff</sub>/*d*<sub>1</sub>)*f*<sub>load</sub>) = (1/23)(1 + (1325/4004)*f*<sub>load</sub>) | `src/shbt/visualizer/src/shaders/causal_point_get.wgsl` — `compute_kappa_get`, `TheoryInvariants` uniform (*k*<sub>*ℓ*</sub>, *k*<sub>*q*</sub>, *K*, *c*<sub>eff</sub>, *d*<sub>1</sub>, *h*<sup>∨</sup>, *N*<sub>sat</sub>) | `tests/first_principles_kernels.rs::test_stress_energy_conservation_in_get_transport` |
| §9, Thm. 9.9 (Instanton-Gated Seed Nucleation) | *S*<sub>inst</sub> = (2π*c*<sub>eff</sub>/*k*<sub>*q*</sub>)((*N*<sub>limit</sub> − *N*<sub>local</sub>)/*N*<sub>limit</sub>)²; Γ<sub>nuc</sub> = *A*<sub>0</sub>·e<sup>−*S*<sub>inst</sub></sup>; *N*<sub>limit</sub> = γ<sub>CFT</sub>(*H*/*H*<sub>ref</sub>)²(*V*<sub>cell</sub>/*V*<sub>*H*</sub>)*N*<sub>sat</sub>, γ<sub>CFT</sub> = 1325/924 ≈ 1.4339826; overflow mass bounded by *N*<sub>local</sub> − *N*<sub>limit</sub> | `src/shbt/visualizer/src/shaders/seed_emergence.wgsl` — `cardy_limit_norm`, `instanton_action`, `cs_detect_condensation` (PCG tunneling draw, 26-NMS, atomic append into `GlobalSeedBuffer`, `SEED_POOL_CAP` = 256) | `tests/first_principles_kernels.rs::test_seed_nucleation_instanton_dynamics` |
| §9, Thm. 9.10 (Stinespring Thermal De-Rendering Expectation) | *w*<sub>vis</sub>(*z*) = (1 − η<sub>*D*</sub>) + η<sub>*D*</sub>/(1 + (*z*<sub>*N*</sub>/*z*)<sup>Δ<sub>*B̄*</sub></sup>), η<sub>*D*</sub> = 23/33, *z*<sub>*N*</sub> = 7.356 × 10<sup>10</sup>, Δ<sub>*B̄*</sub> = 26/3 | `src/shbt/visualizer/src/shaders/nbody_pm.wgsl` — `evaluate_stinespring_channel`, `evaluate_quench_fraction`; `dual_channel_render.wgsl` — Channel-A emission scaled by *w*<sub>vis</sub>; `lib.rs`/`hud.rs` — `stinespring_w_vis`, `stinespring_quench_fraction` | `tests/first_principles_kernels.rs::test_bit_conservation_across_stinespring_transition` |
| §9, Thm. 9.11 (Symplectic Supercomoving Integration and Adaptive Plummer Softening) | x̃ = **x**/*L*<sub>box</sub>, *d*τ = *H*₀*a*<sup>−2</sup>*dt*, p̃ = *a***v**/*V*₀, *V*₀ = *H*₀*L*<sub>box</sub>; Δτ = Δ*a*/(*a*<sub>½</sub>³*H*(*a*<sub>½</sub>)/H₀); *A*(*a*) = (3/2)Ω<sub>m,0</sub>*a*(1 − (10/33)*f*<sub>load</sub>); ε̃ = η<sub>soft</sub>/*N*<sub>grid</sub>, η<sub>soft</sub> ≈ 0.3333 | `src/shbt/visualizer/src/units.rs` — `CosmologicalContext`, `MetrologyPipeline::prepare_step_uniforms`, `GpuSimulationUniforms` (byte-exact `#[repr(C)]`); `nbody_pm.wgsl` — `cs_advance_particles` KDK, `compute_seed_force`, `compute_get_force` | `tests/test_harness_metrology.py` (Keplerian KDK energy drift < 10⁻⁴ over 10⁴ steps, *f*σ₈ table) |
| §9, Thm. 9.12 (Distance-Duality Thin-Screen Gravitational Optics and Boundary Dispersion) | θ<sub>E,k</sub> = √(4*GM*/c² · *D*<sub>ds</sub>/(*D*<sub>d</sub>*D*<sub>s</sub>)), θ<sub>E</sub><sup>screen</sup> = θ<sub>E</sub>/Θ<sub>FoV</sub>; χ(*z*) = *c*∫₀<sup>z</sup>*dz*′/*H*<sub>SHBT</sub>(*z*′); δ<sub>disp</sub>(λ) = ζ[(550 nm/λ)² − 1] at (*B*, *G*, *R*) = (436, 546, 700) nm | `src/shbt/visualizer/src/units.rs` — `comoving_distance_mpc`, `angular_diameter_distance_mpc`, `compute_einstein_radius_rad`; `holographic_post.wgsl` — `fs_post` band deflections + `post2` metrology slot, `bloom_blur.wgsl` half-res separable-Gaussian bloom (`post2.z`), exponential depth fog (`post2.w`), ACES filmic tonemap; `dual_channel_render.wgsl` — solar-gold 10/33 baryons, electric-violet 23/33 anti-baryons, Gaussian splats exp(−3.5*d*²) | `visualizer/recordings/05_cosmic_web_lensing.png`; HUD `max_einstein_radius` (arcsec) |
| §9/§12, Fast-PM + seed attraction | conformal coupling *A*(*a*) = (3/2)Ω<sub>m,0</sub>*a*(1 − (10/33)*f*<sub>load</sub>); softened **a**<sub>seed</sub> sum over `SeedDefectRecord` | `src/shbt/visualizer/src/shaders/nbody_pm.wgsl` — `cs_advance_particles` | SHBT-MMIO `SeedDefectRecord` payload |
| §12, conformal torus unwrapping | `unwrap_transition` 0 → 1 maps comoving bulk onto CFT torus [0, 2π)² | `src/shbt/visualizer/src/shaders/holographic_post.wgsl` — `unwrap_torus_projection`; `ShbtWebGpuEngine::set_unwrap_transition` | `visualizer/index.html` unwrap slider |
| §9 audits (Python) | α<sub>seed</sub> = 1.3258316 × 10<sup>−51</sup> *M*<sub>☉</sub>/bit; η<sub>*B*</sub> ≃ 6.449923 × 10<sup>−10</sup>; 60 FPS frame budget | `scripts/verify_cosmology.py` — `audit_seed_coupling`, `audit_framing_defect_and_stinespring`, `benchmark_simulation_frame_budget` | stdout audit ledger |
| §9/§12, SHBT-MMIO frame | 128-byte header (magic 0x54424853, schema 0x00020000), 256-point *T*(*k*,*z*) and *P*<sub>m</sub>(*k*,*z*) grids, 128-byte seed records | `src/shbt/export.rs` — `MmioTelemetryHeader`, `SeedDefectRecord`, `serialize_mmio_frame`, `serialize_mmio_frame_py`; `boltzmann_shbt.export_webgpu_telemetry` | `data/telemetry.bin` |
| §12, two-tier WebGPU visualizer | `ShbtWebGpuEngine` + `WasmShbtEngine` + WGSL pipeline (`causal_point_get`, `nbody_pm`, `dual_channel_render`, `holographic_post`); 32-byte `ParticleRecord`, 64-byte `CausalPointRecord`, 128-byte `SeedDefectRecord`; `HorizonLedger` (*N*<sub>sat</sub>, Γ<sub>lock</sub>, Stinespring partition, seed ledger, ℛ<sub>adm</sub>); HUD + timeline *z* ∈ [−1, 10<sup>14</sup>] | `src/shbt/visualizer/` — `src/lib.rs`, `src/engine.rs`, `src/particle.rs`, `src/hud.rs`, `src/telemetry.rs`, `src/shaders/*.wgsl`, `src/bin/headless.rs` | `visualizer/pkg/shbt_visualizer_bg.wasm`, `visualizer/index.html` |
| §10 thermodynamic debt | Entropy-debt power schedule Q̇ = 9.06 × 10<sup>11</sup> W (≃ 906 GW); benchmark ratio Γ<sub>bench</sub> ≃ 6377 | `src/shbt/stability_audit.rs` | `result.json → stability_audit.Q_dot_W`, `.P_bench_W`, `.Gamma_bench` |

**Implementation-name reconciliation.** The functions named in earlier drafts of this contract (`calculate_framing_defect`, `verify_modular_invariance`, `integrate_yoshida6`, `compute_asymmetry`, `PrecisionPipeline.compute_non_gaussianity`, `BoltzmannSolver.integrate_cl_pk`, `CalorimetryEngine.run_regression`) are specification shorthand, not literal symbols. Their live equivalents are:

| Contract shorthand | Live symbol |
|---|---|
| `StaticBoundary::calculate_framing_defect` | `StaticBoundary::framing_defect` / `framing_defect_py` (Δ<sub>fr</sub>); `verify_equations` / `verify_equations_py` for the full closure chain |
| `StaticBoundary::verify_modular_invariance` | `StaticBoundary::verify_equations` + `verify_dark_modular_closure` (S/T commutators reported via `boundary_report`) |
| `HolographicProjection::integrate_yoshida6` | not present in this repo — 512-bit symplectic integration is realized by the `rug`-backed cascade kernels in `entropy_flow.rs`; the Yoshida-6 integrator is a `sys1own/shbt-cf` export |
| `BaryogenesisOptimizer::compute_asymmetry` | `BaryogenesisOptimizer::baryogenesis_identity` (returns `BaryogenesisIdentity.eta_b`) |
| `PrecisionPipeline::compute_non_gaussianity` | `precision_cosmology.compute_non_gaussianity_shapes` |
| `BoltzmannSolver::integrate_cl_pk` | `boltzmann_shbt.compute_cmb_power_spectra` / `compute_tensor_power_spectra` |
| `CalorimetryEngine::run_regression` | `precision_cosmology.simulate_calorimetry_experiment` + calorimetry regression pipeline in `shbt_simulate.py` |
| `--mode full` | `--mode all` (valid modes: `audit`, `cosmology`, `cosmology-test`, `baryogenesis`, `history`, `all`) |

---

## 3. Section 12 Publication Reproduction Ledgers

These correspond to the paper's Table 28 (code traceability), Table 29 (data crosswalk), and the `tab:code-availability` interface contract in `main.tex` §12.

### Table 38 — Verification Ledger (theory → code entry points)

| Verification quantity | Expected | Live value | Code entry point |
|---|---|---|---|
| Branch (*k*<sub>*ℓ*</sub>, *k*<sub>*q*</sub>, *K*) | (26, 8, 312) | (26, 8, 312) | `StaticBoundary::new` / `audit.branch` |
| Framing defect Δ<sub>fr</sub> | 0 | 0.0 | `StaticBoundary::framing_defect` → `audit.boundary_report.framing_defect` |
| Modular S commutator norm | ≃ 0 | 0.0 | `audit.boundary_report.modular_S_commutator` |
| Modular T commutator norm | ≃ 0 | 0.0 | `audit.boundary_report.modular_T_commutator` |
| Modular invariant Z<sub>∂</sub> | true | true | `audit.boundary_report.modular_invariant` |
| Zero-energy lock *E*<sub>μν</sub> ≡ 0 | true | true | `audit.boundary_report.zero_energy_locked` |
| Visible projection 26 → 4 | true | true | `audit.boundary_report.projection_dimension_26_to_4` |
| Metric slices | 9 | 9 | `audit.projection_report.slice_count` |
| Projector rank / symmetry / trace-1 / positive-definite | 3 / true / true / true | 3 / true / true / true | `audit.projection_report` |
| Memory budget all-passed | true | true | `audit.memory_report.all_passed` |
| History entries | 9 | 9 | `audit.history_entries` |
| Stress-energy preservation | true | true | `audit.stress_energy_preserved` |
| Baryon asymmetry η<sub>B</sub> | 6.449923359416 × 10<sup>−10</sup> | 6.449923359416131 × 10<sup>−10</sup> | `audit.eta_b` |
| Dark-sector fraction η<sub>D</sub> after de-rendering | 23/33 | 23/33 = 0.696969… | `succession.records[].eta_dark` / `CausalPoint::terminate_and_derender` |
| Succession kernel normalization Σ<sub>*A*′</sub> *T* | 1.0 | 1.0 (all cycles) | `succession.kernel_normalized` / `evaluate_succession_kernel` |
| Admissible observer set *R*<sub>adm</sub> at z → −1 | ∅ (freeze) | empty | `foundation_audit.asymptotic_observer_freeze.asymptotic_admissible_set_empty` |
| Dynamic bit loading Ṫ = *H*(*t*) | exact | equal to < 10<sup>−30</sup> | `ShbtUniverse::evaluate_hubble_index_rate` / `cosmological_invariants` test |
| Ghost partition Ω<sub>DM,0</sub> | 0.260000 | 0.260000 | `ShbtUniverse::dark_matter_fraction` |
| Seed condensation *M*<sub>seed</sub>(Δ*N* = 6 × 10<sup>59</sup>) | ≃ 7.955 × 10<sup>8</sup> *M*<sub>☉</sub> | 7.9549896 × 10<sup>8</sup> | `ShbtUniverse::compute_seed_condensation` → `cosmology_visualization.seed_condensation` |
| Landauer debt at 10<sup>9</sup> *M*<sub>☉</sub> | 9.06 × 10<sup>20</sup> W | 9.06 × 10<sup>11</sup> GW | `ShbtUniverse::eval_landauer_debt_power` |
| κ<sub>GET</sub>(*f*<sub>load</sub>) first-principles coefficient | (1/23)(1 + (1325/4004)*f*<sub>load</sub>) | exact at *f*<sub>load</sub> ∈ {0, 1}; *E*<sub>μν</sub> < 10<sup>−120</sup> | `causal_point_get.wgsl → compute_kappa_get`; `tests/first_principles_kernels.rs` |
| Instanton nucleation dynamics | *S*<sub>inst</sub> ≥ 0, = 0 at overflow; γ<sub>CFT</sub> = 1325/924 | verified | `seed_emergence.wgsl → instanton_action`; `tests/first_principles_kernels.rs::test_seed_nucleation_instanton_dynamics` |
| Stinespring thermal expectation *w*<sub>vis</sub>(*z*) | η<sub>*D*</sub> = 23/33, *z*<sub>*N*</sub> = 7.356 × 10<sup>10</sup>, Δ<sub>*B̄*</sub> = 26/3 | bit conservation < 10<sup>−30</sup>; derivative match | `nbody_pm.wgsl → evaluate_stinespring_channel`; `tests/first_principles_kernels.rs::test_bit_conservation_across_stinespring_transition` |
| Dynamic seed pool (GlobalSeedBuffer) | atomic append, no hardcoded 64-seed cap | pool scales to `SEED_POOL_CAP` = 256 | `seed_emergence.wgsl → tracking_state.candidate_count`; `lib.rs → MAX_SEEDS` |
| SHBT-MMIO header layout | 128 bytes, magic 0x54424853 | verified | `tests/cosmological_invariants.rs::test_mmio_header_serialization_layout` |
| Supercomoving KDK integrator | Δτ = Δ*a*/(*a*<sub>½</sub>³*H*(*a*<sub>½</sub>)); symplectic half-kick → drift(fract wrap) → half-kick | \|Δ*E*\|/*E*₀ < 10⁻⁴ over 10⁴ steps | `units.rs → MetrologyPipeline`; `nbody_pm.wgsl → cs_advance_particles`; `tests/test_harness_metrology.py` |
| Adaptive Plummer softening | ε̃ = η<sub>soft</sub>/*N*<sub>grid</sub>, η<sub>soft</sub> ≈ 0.3333 | implemented | `nbody_pm.wgsl → eps_soft_sq` uniform; `tests/test_harness_metrology.py` |
| Physical thin-screen optics | θ<sub>E,k</sub> from Tier-1 angular diameter distances; δ<sub>disp</sub>(λ) = ζ[(550/λ)² − 1] | implemented; bands (*B*, *G*, *R*) = (436, 546, 700) nm | `units.rs → compute_einstein_radius_rad`; `holographic_post.wgsl → fs_post`, `post2` |
| Screen-space deflection β(θ) (§9 Eq.) | **β** = **θ** − λ<sub>lens</sub>[∇κ + **Γ**·∇κ] − Σ θ<sub>E,s</sub>²(**θ**−**θ**<sub>s</sub>)/(|**θ**−**θ**<sub>s</sub>|²+ε<sub>core</sub>²) | implemented | `holographic_post.wgsl → fs_post` (central-difference ∇κ, shear matrix **Γ**, emergent seed loop *S* ≤ 256) |
| Dominant seed Einstein radius θ<sub>E</sub> | emergent, ≤ 0.09 rad measured | θ<sub>E,k</sub> from condensed seed masses | `active_seeds` pool (`SEED_POOL_CAP` = 256) / `LensingUniforms.seeds` / HUD `max_einstein_radius` |
| Doppler boost 𝒟 = √(1−β²)/(1−β<sub>los</sub>) (§9) | *I*<sub>obs</sub> = *I*<sub>0</sub>𝒟³ | applied per-particle | `dual_channel_render.wgsl → kelvin_to_rgb`, `vs_particle_billboard` |
| Bilateral convergence smoothing (§9) | depth-aware σ<sub>r</sub> = 0.08 | pass | `holographic_post.wgsl → sample_bilateral_convergence` |
| Causal Fresnel ripple gate | *R*<sub>entropy</sub> ≥ 0 only | pass | `causal_point_get.wgsl → vs_causal` / `fs_causal` |
| Post-pass frame budget τ<sub>post</sub> | ≤ 1.18 ms @1440p | 0.92 ms @1080p | `holographic_post.wgsl`; §5 performance benchmark |
| `LensingUniforms` layout | 224 bytes, 16-aligned | `size_of::<LensingUniforms>() == 224` | `hud.rs::lensing_tests::lensing_uniform_layout_is_wgsl_contract` |
| `SeedDefect` layout | 16 bytes | `size_of::<SeedDefect>() == 16` | `hud.rs::lensing_tests::seed_registration_and_telemetry` |

### Table 39 — Generated-Artifact Data Product Crosswalk

| Artifact | Columns / nested records | Paper-level observable | Producer |
|---|---|---|---|
| `shbt_run_cmb_cls.csv` | `ell`, `Dl_TT_muK2`, `Dl_EE_muK2`, `Dl_TE_muK2`, `Cl_TT`, `Cl_EE`, `Cl_TE`, `Cl_BB` | scalar TT/EE/TE spectra (and BB column) | `boltzmann_shbt.compute_cmb_power_spectra` |
| `shbt_run_matter_pk.csv` | `k_Mpc_inv`, `Pk_z0`, `Pk_z05`, `Pk_z1` | matter power at *z* = 0, 0.5, 1 | `boltzmann_shbt.compute_cmb_power_spectra` (`_matter_rows`) |
| `shbt_run_tensor_cls.csv` | `ell`, `Cl_BB`, `Dl_BB`, `Cl_TT_tensor` | primordial tensor B modes and tensor TT | `boltzmann_shbt.compute_tensor_power_spectra` |
| `shbt_run_calorimetry_sim.csv` | `k_bits`, `R_addresses`, `Q_H0_zJ`, `Q_H1_zJ`, `Q_noise_zJ` | Landauer address sweep, competing heat laws, simulated noise | `precision_cosmology.simulate_calorimetry_experiment` |
| `result.json` | `audit` (full `ShbtReport` dict), `baryogenesis`, `history`, `succession` (with `--enable-succession`: `cycles`, `records`, `kernel_normalized`), `foundation_audit` (`asymptotic_observer_freeze`), `precision_cosmology`, `precision_pipeline` (`spectra`, `calorimetry_csv`, `non_gaussianity`), `stability_audit`, `summary`, `metadata`, `config` | complete machine-readable simulation report | `shbt_simulate.py` main pipeline |
| `data/telemetry.bin` | 128-byte SHBT-MMIO header; 256-point *T*(*k*,*z*) + *P*<sub>m</sub>(*k*,*z*) grids; 128-byte `SeedDefectRecord` entries | two-tier visualizer telemetry contract | `boltzmann_shbt.export_webgpu_telemetry` / `shbt_simulate.py --export-webgpu-telemetry` |
| `visualizer/pkg/` | `shbt_visualizer.js`, `shbt_visualizer_bg.wasm` | Tier-2 browser engine bundle | `cargo build --target wasm32-unknown-unknown --release` + `wasm-bindgen --target web` |
| `visualizer/recordings/` | `01_primordial_bit_loading` … `06_asymptotic_horizon_freeze` PNGs + per-event `.webm` clips, `full_cosmic_evolution.webm`, `cosmic_event_telemetry.json` | six-event cinematic record: bit loading → Stinespring quench → instanton nucleation → κ<sub>GET</sub> clustering → cosmic-web lensing → horizon freeze | `visualizer/record_simulation_events.py` (DOM-telemetry asserted milestones, `--verify` artifact gate) |

### Table 40 — Software Interface Contract

| Interface | Role / principal accessors |
|---|---|
| `shbt_simulator` (PyO3) | Module exposing `ShbtSimulator`, report getters, and record classes; `import shbt_simulator` after `--build` |
| `ShbtSimulator` | `run_full_audit()` → `ShbtReport`; `crystallize_history()`; `is_observer_admissible()`; `build_succession_candidates()`; `evaluate_succession_kernel()`; `terminate_and_derender()`; `run_succession_cycles(cycles, seed)` → `SuccessionRecord` list; `to_dict()` serialization |
| `ShbtReport` | Getters: `branch`, `eta_b`, `stress_energy_preserved`, `framing_defect`, `modular_invariant`, `zero_energy_locked`, `projection_dimension_26_to_4`, `metric_slice_count`, `history_entry_count`, `memory_all_passed`, `to_dict()` |
| `StaticBoundary` | `benchmark_branch`, `lepton_level`, `quark_level`, `parent_level`, `i_l_star`, `i_q_star`, `c_dark`, `c_dark_residual`, `c_dark_completion`, `lambda_holo_si_m2`, `n_sat`, `bit_budget`, `h0_cmb`, `framing_defect_py`, `verify_equations_py`, `evaluate_z_boundary_py`, `evaluate_z_dark_py`, `dark_modular_data`, `s_dark`, `t_dark`, `dark_conformal_weights`, `verify_dark_modular_closure`, `to_c_abi` |
| `HolographicProjection` / `BulkMetricSlice` | `project_entropy_cascade`, `derive_load_vector`, `metric_from_load_vector`, `verify_projection`, `project_static_block_to_bulk` (Rust-level; consumed via `audit.metric_slices`) |
| `BaryogenesisOptimizer` / `BaryogenesisIdentity` / `BenchmarkDelta` | `baryogenesis_identity`, `thermal_lindblad_evolution`, `derender_antibaryon_charges`, `stress_energy_preserved`, `run_benchmark` |
| `CausalPoint` / `MemoryReport` / `LightConeSample` / `CoordinateLogEntry` | `build_past_light_cone`, `verify_memory_budget`, `crystallize_history`; `AnomalyClosureError` raised on finite-capacity violation |
| `CausalPoint` (succession) | `is_admissible`, `entropy_budget_residual`, `retrieval_cost_bits`, `terminate_and_derender`, `build_succession_candidates`, `evaluate_succession_kernel`, `relabel_and_rerender`, `run_lifecycle_cycle` |
| `CausalPointCandidate` / `DerenderingRecord` / `SuccessionRecord` / `LifecyclePhase` | Observer-succession record types; `DerenderingRecord` carries `eta_dark`, `pointer_triad`, `trace_norm`; `SuccessionRecord` carries `kernel_probabilities`, `successor_index`, `phase`; `to_dict()` on all |
| `precision_cosmology.py` | `build_precision_cosmology_report`, `run_mcmc_analysis`, `simulate_calorimetry_experiment`, `compute_non_gaussianity_shapes`, plus all Section 9 equation functions |
| `boltzmann_shbt.py` | `compute_cmb_power_spectra`, `compute_tensor_power_spectra` |
| `shbt_simulate.py` CLI | `--mode {audit, cosmology, cosmology-test, baryogenesis, history, all}`, `--enable-succession`, `--succession-cycles N`, `--branch K_L K_Q K`, `--observer-radius-fraction`, `--redshift-max`, `--redshift-samples`, `--particles`, `--seed`, `--h0-cmb`, `--omega-m`, `--omega-r0`, `--delta-mod`, `--z-samples`, `--precision`, `--output`, `--output-dir`, `--format {json,csv,hdf5,h5}`, `--sweep`, `--plot`, `--verbose`, `--log-level` |
| `ShbtUniverse` (`src/shbt/cosmology.rs`) | `new_canonical_branch`, `hubble`, `h0_redshift_dependent`, `conformal_loading_fraction`, `saturated_screen_capacity`, `active_visible_bits`, `dark_completion_bits`, `dark_matter_fraction`, `compute_seed_condensation`, `eval_landauer_debt_power`, `evaluate_growth_suppression`, `delta_isw_residual`, `verify_observer_freeze` |
| `src/shbt/export.rs` | `MmioTelemetryHeader` (128-byte SHBT-MMIO frame), `SeedDefectRecord`, `serialize_mmio_frame`, `serialize_mmio_frame_py` (PyO3) |
| `shbt-visualizer` crate (`src/shbt/visualizer`) | `units.rs` metrology — `CosmologicalContext` (*L*<sub>box</sub>, *M*<sub>box</sub>, *V*₀ = *H*₀*L*<sub>box</sub>, *G*<sub>code</sub> = (3/2)Ω<sub>m,0</sub>, angular diameter distances, `compute_einstein_radius_rad`), `MetrologyPipeline`, `GpuSimulationUniforms`/`GpuLensUniforms`/`GpuSeedLensBuffer` (`#[repr(C)]` + `bytemuck`), `kappa_get`; `Particle` (32-byte, 16-aligned), `ShbtWebGpuEngine` (`new`, `update_frame_telemetry`, `step_frame`, `set_redshift`, `set_speed`, `set_projection`, `set_unwrap_transition`, `set_channels`, `set_lensing_enabled`, `set_lensing_scale`, `set_dispersion`, `set_doppler_enabled`, `set_dark_glow`, `set_glitch_enabled`, `set_glitch_intensity`, `hud_json`), `HudMetrics`, `TimelineController`, `headless` binary; `LensingUniforms` (224-byte), `SeedDefect` (16-byte), `VisualizerEngine`, `VisualizerTelemetry` (`peak_shear`, `peak_convergence`, `max_einstein_radius`, `active_caustics`); `WasmShbtEngine` (`get_*_buffer_ptr` zero-copy accessors, `update_epoch`, `get_hud_telemetry_json`), `HorizonLedger`, `CausalPointRecord` (64-byte), `SeedDefectRecord` (128-byte) |
| WGSL shader suite (`src/shbt/visualizer/src/shaders`) | `dual_channel_render.wgsl` (`kelvin_to_rgb` black-body map, Doppler beaming 𝒟³ boost, `w_vis`-weighted Channel-A emission, Landauer heat glow from `seed_debt`, Channel B shear/convergence/causal-entropy write), `holographic_post.wgsl` (`fs_post` screen-space deflection, `aa_lattice_line` FXAA on torus grid + horizon lattice, `sample_bilateral_convergence` depth-aware bilateral glow, 3-tap chromatic dispersion, caustic rings), `causal_point_get.wgsl` (`cs_causal_point_get` entropic erasure + center kick via `compute_kappa_get`, `vs_causal`/`fs_causal` Fresnel ripple shells gated by *R*<sub>entropy</sub> ≥ 0), `seed_emergence.wgsl` (`cs_accumulate_cic`, `cs_detect_condensation`, `cs_temporal_tracking`), `nbody_pm.wgsl` (`evaluate_stinespring_channel`, progressive quench draw, `cs_advance_particles` supercomoving KDK with adaptive Plummer softening ε̃ = η/*N*<sub>grid</sub> and `compute_seed_force`/`compute_get_force`), `units.rs` host metrology |
| `shbt-visualizer-headless` CLI flags | `--enable-lensing`, `--lensing-scale S` ([0, 5]), `--dispersion D` ([0, 1]), `--enable-doppler`, `--dark-glow G` ([0, 2]), `--width PX`, `--height PX` |
| `visualizer/index.html` + `index.js` | Horizon Bar, Boundary Capacity Gauge, Congestion & Seed Ledger, Landauer Debt Monitor, Causal-Point Observer Activity monitor (ℛ<sub>adm</sub>, *R*<sub>entropy</sub> status), Gravitational Optics Telemetry ledger (γ<sub>max</sub>, κ<sub>max</sub>, θ<sub>E</sub>, caustics), invariant icons; timeline scrub *z* = 10<sup>14</sup> → −1; speeds 1×/10×/100×; `unwrap_transition` torus slider and projection switch; Channel A/B toggles; lensing/doppler/dark-glow toggles + λ<sub>lens</sub>/δ<sub>disp</sub>/glow sliders; seed-glitch toggle + intensity slider (`set_glitch_enabled`/`set_glitch_intensity`, `glitchEnabled`/`glitchIntensity` in the hud_json frame); keybindings `L`, `D`, `[`, `]`, `-`, `=`, `G`, `H` |
| `shbt_simulate.py` CLI additions | `--mode visualize`, `--export-webgpu-telemetry PATH`, `--sim-speed FLOAT`, `--particles N` (shared with the visualizer contract); `result.json → cosmology_visualization` + `result.json → causal_point_simulation` |
| `visualizer/record_simulation_events.py` | Cinematic six-event recording suite (`?capture=1` overlay path; Chromium flags `--enable-unsafe-webgpu --use-angle=vulkan --enable-features=Vulkan`); per-milestone DOM telemetry assertions; outputs `full_cosmic_evolution.webm`, six milestone PNGs/WebM clips, `cosmic_event_telemetry.json`; `--verify` checks artifact existence, size, and non-blankness |
| `examples/run_audit.py` | Minimal foundation-audit entry point |

---

## 4. Ground-Truth Invariant Ledger (live simulation values)

Extracted from `python3 shbt_simulate.py --mode all --output result.json` and `cargo test --release` (27 tests, all passing) on this checkout. `result.json` paths are exact key chains.

| Invariant | Paper specification | Live value | Source JSON path / accessor |
|---|---|---|---|
| Visible central charge *c*<sub>vis</sub> | 1325/154 ≃ 8.603896103896 | 8.603896103896 (39/14 + 64/11) | `main.tex` Eq. *c*<sub>vis</sub>; `StaticBoundary.su2_central_charge(26) + su3_central_charge(8)` |
| Parent central charge *c*<sub>parent</sub> | 351/8 = 43.875 (SO(10)<sub>312</sub>) | 351/8 | `main.tex` Table `section-three-affine-central-charges`; affine formula *k*·dim/(*k*+*h*∨) = 312·45/320 |
| Completed dark ledger *c*<sub>dark</sub> | 1197103/362670 ≃ 3.300805139659 | 3.3008051396586424 | `result.json → precision_cosmology.completed_ledger` (`"1197103/362670"`); `StaticBoundary.c_dark` |
| Residual dark ledger *c*<sub>dark</sub><sup>res</sup> | 834433/362670 ≃ 2.300805139659 | 2.3008051396586424 | `result.json → precision_cosmology.simulator_constants.c_dark_residual`; `StaticBoundary.c_dark_residual` |
| Framing defect Δ<sub>fr</sub> | 0.000000000000000000 | 0.0 | `result.json → audit.boundary_report.framing_defect`; `StaticBoundary.framing_defect_py` |
| Baryon asymmetry η<sub>B</sub> | 6.449923359416 × 10<sup>−10</sup> | 6.449923359416131 × 10<sup>−10</sup> | `result.json → audit.eta_b`; `audit.baryogenesis_identity.eta_b` |
| Entropy-debt power Q̇ | 9.06 × 10<sup>11</sup> W (906 GW) | 906 000 000 000 W | `result.json → stability_audit.Q_dot_W` |
| Benchmark transient ratio Γ<sub>bench</sub> | ≃ 6377 | 6376.689189189189 | `result.json → stability_audit.Gamma_bench` |
| Saturated screen *N*<sub>sat</sub> | 3π/(*L*<sub>P</sub><sup>2</sup>Λ<sub>holo</sub>) | 3.311997720142366 × 10<sup>122</sup> bits | `result.json → precision_cosmology.N_sat_bits` |
| Local Hubble uplift *H*<sub>0</sub><sup>loc</sup> | 72.197960 km s<sup>−1</sup> Mpc<sup>−1</sup> | 72.19796007286148 | `result.json → precision_cosmology.h0_local_km_s_mpc` |
| Loading amplitude *A*<sub>H</sub> | 4.797960 km s<sup>−1</sup> Mpc<sup>−1</sup> | 4.797960072861485 | `result.json → precision_cosmology.A_H_km_s_mpc` |
| Local observer bits *N*<sub>local</sub> | ≃ *N* *f*<sub>H</sub><sup>2</sup> | 2.535748254483999 × 10<sup>122</sup> | `result.json → audit.memory_report.local_available_bits` |
| Dark-matter abundance Ω<sub>DM</sub>/Ω<sub>b</sub> | ≃ 5.34 | 5.343450862978382 | `result.json → precision_cosmology.dark_matter.abundance_ratio` |
| Growth suppression *f*σ<sub>8</sub> at *z* = 0.5 | negative suppression | −9.819 % | `result.json → precision_cosmology.growth_suppression[z=0.5].suppression_fraction` |
| Non-Gaussianity | *f*<sub>NL</sub><sup>loc</sup> = 0.015, *f*<sub>NL</sub><sup>equil</sup> = −0.042, *f*<sub>NL</sub><sup>ortho</sup> = −0.018, τ<sub>NL</sub> = 0.000324, *g*<sub>NL</sub> = −1.2 × 10<sup>−5</sup> | identical | `result.json → precision_pipeline.non_gaussianity` |
| Tensor parameters | *r* = 0.0032, *n*<sub>t</sub> = −0.0004 | identical | `result.json → precision_pipeline.spectra.r`, `.n_t` |
| Cosmic age | 13.277 Gyr | 13.276616557 Gyr | `result.json → precision_cosmology.cosmic_age_gyr` |
| Chronometer χ²/ν | ≃ 1.00 | 2491.49/2482 = 1.004 | `result.json → precision_cosmology.summary_table_17.chronometer` |

**Downstream-only invariants.** Two quantities named in the contract scope are defined by consumer repositories, not emitted by `shbt-precision`:

- η<sub>A</sub> = 10/33 — the Stinespring active capacity fraction; η<sub>D</sub> = 23/33 is now computed in-repo by `CausalPoint::terminate_and_derender` (`succession.records[].eta_dark`) in addition to its downstream use in `sys1own/shbt-recon`.
- *P*<sub>debt</sub> = 906.00 kW — the scaled Landauer-debt schedule used by `sys1own/shbt-power`. The native value here is Q̇ = 906 GW (`stability_audit.Q_dot_W`); downstream repos rescale it for plant-level ledgers. These entries are listed for crosswalk completeness and must not be quoted as `result.json` outputs.

---

## 5. SHBT Ecosystem Downstream Export Crosswalk

Bidirectional technology transfer: `shbt-precision` is the computational authority; siblings consume its verified constants and kernels.

| Repository | Consumed export | Mechanism |
|---|---|---|
| `sys1own/shbt-power` | Bremsstrahlung suppression factor *S* = 100/1089; Landauer-debt power schedule (native Q̇ = 906 GW, rescaled to the 906.00 kW plant ledger) | constant import from `stability_audit` / §10 ledger |
| `sys1own/shbt-cf` | Symplectic integrator conventions (Yoshida-6) and WZW affine character tables for LANR non-equilibrium screening | `su2/su3` character and modular-entry formulas in `boundary.rs` |
| `sys1own/shbt-qc` | Canonical affine branch (26, 8, 312) and boundary code projection norm bounds | `audit.projection_report` bounds; `StaticBoundary` branch getters |
| `sys1own/shbt-ghost` | 512-bit MPFR arithmetic kernels and Landauer debt scaling for mass-seed coupling | `rug` Float infrastructure (`PREC` = 512); `stability_audit` debt schedule |
| `sys1own/shbt-recon` | Stinespring dilation capacity partition (η<sub>A</sub> = 10/33, η<sub>D</sub> = 23/33) and trace-norm invariants | dark-ledger completion arithmetic (`c_dark`, `c_dark_residual`); `CausalPoint::terminate_and_derender` records |
| `sys1own/shbt-sglt` | Arbitrary-precision register math and 2PN relativistic optics integration bounds | `rug`/MPFR Float kernels in `entropy_flow.rs` |
| `sys1own/shbt-exotic` | Boundary CFT partition algebra and modular closure operators | `evaluate_z_boundary` / `evaluate_z_dark`; `build_dark_modular_data` |
| `sys1own/shbt-warp` | Framing-defect identity Δ<sub>fr</sub> ≡ 0 ⟹ *E*<sub>μν</sub> ≡ 0 and 512-bit MPFR foliation wrappers | `StaticBoundary.framing_defect`, `verify_equations`; `EVAL_PREC` |

---

## 6. Quick Verification

```bash
# Build the Rust core and stage the PyO3 module
python3 shbt_simulate.py --build

# Rust foundation tests (22 + 5 passing)
cargo test --release

# Full live audit: regenerates result.json + shbt_run_*.csv
# (add --enable-succession to run the closed-loop observer succession engine)
PYTHONPATH=target/release:. python3 shbt_simulate.py --mode all --enable-succession --output result.json
test -s result.json

# Python suites
pytest tests/ -q
python3 precision_cosmology.py --run-tests
python3 boltzmann_shbt.py --run-tests

# Visualizer telemetry + wasm bundle + headless benchmark
python3 shbt_simulate.py --mode visualize --export-webgpu-telemetry data/telemetry.bin
cargo build --release --manifest-path src/shbt/visualizer/Cargo.toml --target wasm32-unknown-unknown
wasm-bindgen --target web --out-dir visualizer/pkg \
    src/shbt/visualizer/target/wasm32-unknown-unknown/release/shbt_visualizer.wasm
cargo run --release --manifest-path src/shbt/visualizer/Cargo.toml --bin headless -- 60 262144
```
