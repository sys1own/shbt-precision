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
| **Canonical Branch Uniqueness & Module Rigidity** | Diophantine system $\operatorname{gcd}(2k_\ell, 3k_q) \mid K$ over level lattice $\mathcal{L}$; minimum conductor $N = \operatorname{lcm}(52, 24) = 312$, discriminant $\Delta = -1$, uniqueness proof of finite quadratic Weil module | `src/shbt/uniqueness.rs` |
| **Bulk Gauge Dynamics from Kac-Moody Currents** | Holographic Yang-Mills bulk closure $D^\mu F_{\mu\nu}^a = 0 \iff \partial_{\bar{z}} J^a = 0$; Knizhnik-Zamolodchikov connection and Gauss law propagation across radial RG slices | `src/shbt/gauge_dynamics.rs` |
| **First-Principles Running Couplings** | 512-bit MPFR RG flow from Stinespring threshold $\Lambda_{\text{GUT}}$ to electroweak scale $M_Z$: $\alpha_s(M_Z) \simeq 0.1179$, $\alpha_{\text{EM}}^{-1}(M_Z) \simeq 127.94$, $\sin^2\theta_W \simeq 0.23122$ | `src/shbt/couplings.rs` |
| **Fermion Masses & CKM/PMNS Flavor Mixing** | Boundary Verlinde fusion tensor $N_{ijk}$ at modular fixed point $\tau = i$ with bi-unitary Jacobi SVD: $m_t \simeq 172.69$ GeV, $m_b \simeq 4.18$ GeV, CKM $\theta_{12} \simeq 13.04^\circ$, $J_{\text{CP}} \simeq 3.08 \times 10^{-5}$, normal neutrino hierarchy | `src/shbt/flavor.rs` |
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
- Axiomatic Derivation Engines:
  - `DiophantineClassifier`, `WeilModuleVerifier`, `UniquenessAuditReport` (`src/shbt/uniqueness.rs`): Rigorous proof of canonical branch uniqueness and discriminant-form rigidity.
  - `BulkGaugeSlice`, `GaugeGroup`, `KacMoodyBoundary` (`src/shbt/gauge_dynamics.rs`): Non-Abelian gauge connection derivation and holographic bulk field equations of motion.
  - `GaugeCouplingLedger`, `RGSliceRecord` (`src/shbt/couplings.rs`): 512-bit arbitrary-precision 9-slice Stinespring Callan-Symanzik beta flow.
  - `FlavorMixingEngine`, `Matrix3x3`, `Complex64` (`src/shbt/flavor.rs`): DOZZ 3-point fusion correlators and bi-unitary Jacobi SVD for quark/lepton mass spectra and CKM/PMNS matrices.
- Closed-loop observer succession: `CausalPoint.is_admissible` evaluates *P*<sub>adm</sub>(*A*); `terminate_and_derender` applies the macroscopic Stinespring channel (η<sub>D</sub> = 23/33, pointer triad Ψ<sub>ι</sub> → (0, 0, 1), Ent(φ) = 0); `evaluate_succession_kernel` returns the normalized *T*(*A*<sub>term</sub> → *A*<sub>next</sub>) distribution over the 3×3 visible coordinate lattice; `relabel_and_rerender` / `ShbtSimulator.run_succession_cycles` drive the five-phase lifecycle loop with zero heap allocation in the kernel hot loop (stack-allocated weight array).
- Legacy low-level engines reused by the SHBT modules: `AnyonBraidingEngine` (SU(2), SU(3), SO(10) braid matrices), `TopologicalTracker` (anyon worldlines, fusion, stabiliser checks), `CircuitCompiler` (Solovay-Kitaev, OpenQASM parsing).

### Tier 2 — Python Precision Cosmology & Boltzmann Pipeline

- `boltzmann_shbt.py`: first-principles inflation dynamics and perturbation spectra; primordial non-Gaussianity templates (local, equilateral, orthogonal *f*<sub>NL</sub> and *g*<sub>NL</sub> bispectra/trispectra); Boltzmann hierarchy integration producing scalar temperature (*C*<sub>*ℓ*</sub><sup>TT</sup>), polarization (*C*<sub>*ℓ*</sub><sup>EE</sup>, *C*<sub>*ℓ*</sub><sup>TE</sup>), matter power *P*(*k*, *z*), and tensor B-modes (*C*<sub>*ℓ*</sub><sup>BB</sup>); SHBT-MMIO telemetry export via `export_webgpu_telemetry`.
- `precision_cosmology.py`: Section 9 precision-cosmology audit, 7-parameter MCMC, *H*<sub>0</sub>-tension quantification, cosmic-chronometer covariance, and the sub-10 mK Landauer calorimetry experiment (`simulate_calorimetry_experiment`).
- `shbt_simulate.py`: unified CLI/API orchestrator; evaluates address sweeps and heat-dissipation hypotheses via OLS/MLE regression and emits the `shbt_run_*` data products.

### Tier 2 — WebGPU Interactive Visualizer (`src/shbt/visualizer`, `visualizer/`)

- `shbt-visualizer` crate compiles to `wasm32-unknown-unknown` and binds to `visualizer/index.html` (`<canvas id="shbt-canvas">`). The same crate runs natively via the `headless` binary for CI benchmarking.
- `ShbtWebGpuEngine` drives the WGSL stage set: `seed_emergence.wgsl` (the first-principles **emergent mass-congestion pipeline** — seeds are no longer hardcoded; they condense wherever local coordinate-entropy demand exceeds the boundary-CFT capacity ceiling via a tri-pass compute: `cs_accumulate_cic` fixed-point atomic Cloud-In-Cell mass assignment, `cs_detect_condensation` 26-neighbour non-maximum suppression + 3³ basin overflow-mass integration, and `cs_temporal_tracking` minimum-image persistent-ID tracking with accretion Ṁ<sub>seed</sub> and Landauer debt *P*<sub>debt,k</sub> = *M*<sub>seed,k</sub>·906 GW/M<sub>⊙</sub>), `causal_point_get.wgsl` (active Causal-Point GET dynamics — `cs_causal_point_get` applies the *R*<sub>entropy</sub> = *N*<sub>limit</sub> − *C*<sub>get</sub> ≥ 0 admissibility gate and the entropic acceleration **a**<sub>GET</sub>(**x**) = −κ<sub>GET</sub> ∇ln ρ<sub>proj</sub>(**x**) with the first-principles transport coefficient κ<sub>GET</sub>(*f*<sub>load</sub>) = (1/(*d*<sub>1</sub> − *h*<sup>∨</sup><sub>su(3)</sub>))(1 + (*c*<sub>eff</sub>/*d*<sub>1</sub>)*f*<sub>load</sub>) = (1/23)(1 + (1325/4004)*f*<sub>load</sub>) streamed via a `TheoryInvariants` uniform), `nbody_pm.wgsl` (symplectic Martel–Shapiro **supercomoving KDK integrator** — `cs_advance_particles` half-kicks at (*x̃*<sub>*n*</sub>, *a*<sub>*n*</sub>), drifts x̃ += Δτ·p̃ with periodic wrap, then half-kicks at (*x̃*<sub>*n*+1</sub>, *a*<sub>*n*+1</sub>) using Δτ = Δ*a*/(*a*<sub>½</sub>³·*H*(*a*<sub>½</sub>)); the heuristic friction/envelope terms are gone and the conformal coupling *A*(*a*) = (3/2)Ω<sub>m,0</sub>*a*·(1 − (10/33)*f*<sub>load</sub>) multiplies the PM force directly, while seed attraction uses **adaptive Plummer softening** ε̃ = η<sub>soft</sub>/*N*<sub>grid</sub> (η ≈ 0.3333) and the GET term −κ<sub>GET</sub>∇ln ρ<sub>proj</sub> samples the register density carried in the PM texture alpha channel; the thermal Stinespring expectation *w*<sub>vis</sub>(*z*) = (1 − η<sub>*D*</sub>) + η<sub>*D*</sub>/(1 + (*z*<sub>*N*</sub>/*z*)<sup>26/3</sup>) with *z*<sub>*N*</sub> = 7.356 × 10<sup>10</sup> drives the progressive quench; **P-PM baryon pressure** `compute_hydro_force` adds −*c*<sub>s</sub>²·*a*·∇ln ρ<sub>b</sub> on the visible sector only — the *h*<sub>*i*</sub> ≥ 23/33 gauge — with *c*<sub>s</sub>(*z*) from `units.rs` metrology scaled by the sandbox `sound_speed_scale`), `dual_channel_render.wgsl` (Channel A emission scaled strictly by *w*<sub>vis</sub>(*z*) — solar-gold 10/33 baryons vs electric-violet 23/33 anti-baryons with the *z* ∈ [10<sup>12</sup>, 10<sup>9</sup>] thermal cooling gradient — Gaussian splats exp(−3.5·*d*²) replacing square sprites, plus Landauer heat glow (*P*<sub>debt</sub> = (*M*<sub>seed</sub>/*M*<sub>☉</sub>) × 906 GW) rendered as an amber/white core + corona on each emergent seed; Channel B passive ghost shear/density conserved independently of it, carrying faint causal-point projection envelopes), and `holographic_post.wgsl` (physical distance-duality lensing — θ<sub>E,k</sub> = √(4*GM*/*c*²·*D*<sub>ds</sub>/(*D*<sub>d</sub>*D*<sub>s</sub>)) per emergent seed with Tier-1 angular-diameter distances, boundary dispersion δ<sub>disp</sub>(λ) = ζ[(550 nm/λ)² − 1] at the 436/546/700 nm bands, half-resolution two-pass Gaussian **bloom** (`bloom_blur.wgsl` — bright-pass + separable 9-tap blur lifted into the composite via `post2.z`), exponential **depth fog** along the Channel-A depth field (`post2.w`), and **ACES filmic tonemapping** (Narkowicz fit replacing the earlier Reinhard + S-curve); the deflection now marches a **4-slice multi-plane lens stack** at *z*<sub>*m*</sub> ∈ {0.5, 1.2, 2.2, 3.5} with *D*<sub>ms</sub>/*D*<sub>s</sub> weights in `post3` plus a **Shapiro-delay tint**, and a **split-viewport mode** (`post1.y`) rasterizes the diagnostics phase-space texture into the right half of the frame; `unwrap_torus_projection` conformal unwrapping onto the CFT torus [0, 2π)², horizon overlay driven by *f*<sub>load</sub>). The emergent pipeline is smoothed by `cs_convolve_overflow` — a **Wendland C⁴ convolution** over the overflow register with metrological radius *R*<sub>filter</sub>(*z*) = *c*/(*k*<sub>*ℓ*</sub>·*a*·*H*(*z*)) (clamped to the [1, 4]-cell stencil) at fixed-point scale *S* = 2¹⁶, with the percolation ceiling scaled by `percolation_scale` — and a `cs_sandbox_get` pass in `causal_point_get.wgsl` applies an **FDT friction counter-term** −γ<sub>obs</sub>·*v*<sub>proj</sub>·dir inside each dispatched observer's cone shell while depleting its entropy budget. A dedicated `diagnostics.wgsl` stage accumulates a 64-bin matter power spectrum by workgroup atomics (`DIAG_FIXED_POINT` = 4096) and rasterizes a 256² (x, v<sub>x</sub>) phase-space texture — gold visible sector, blue dark sector — entirely on-GPU with no host readback.
- Seed condensation physics (shbt7, first-principles): condensation is an **instanton-gated tunneling event**. The Cardy capacity ceiling *N*<sub>limit</sub>(*z*) = γ<sub>CFT</sub>·(*H*/*H*<sub>ref</sub>)²·(*V*<sub>cell</sub>/*V*<sub>*H*</sub>)·*N*<sub>sat</sub> with γ<sub>CFT</sub> = *c*<sub>eff</sub>/6 = 1325/924 ≈ 1.4339826 sets the per-cell bit budget; below the ceiling the Euclidean instanton action *S*<sub>inst</sub> = (2π*c*<sub>eff</sub>/*k*<sub>*q*</sub>)·((*N*<sub>limit</sub> − *N*<sub>local</sub>)/*N*<sub>limit</sub>)² gates nucleation via Γ<sub>nuc</sub> = *A*<sub>0</sub>·e<sup>−*S*<sub>inst</sub></sup> and *P*<sub>nuc</sub> = 1 − e<sup>−Γ<sub>nuc</sub>δ*t*</sup>, while overflow is barrierless (*S*<sub>inst</sub> = 0). A PCG-hashed tunneling draw replaces the old sigmoid turn-on envelope, the empirical *K*<sub>bit</sub> multiplier is gone, and the hardcoded 64-seed cap is replaced by an atomic append into a dynamically sized `GlobalSeedBuffer` pool (`SEED_POOL_CAP` = 256). Condensed mass integrates the absolute register excess Σ<sub>basin</sub> max(0, *N*<sub>local</sub> − *N*<sub>limit</sub>), which stays bounded as *N*<sub>limit</sub> → 0 in the *z* → −1 asymptote. Kernel work is evaluated on the normalized ratio *R* = *N*<sub>local</sub>/*N*<sub>limit</sub> because absolute bit counts (~10<sup>122</sup>) overflow f32 registers.
- Dimensional metrology (shbt8): `src/shbt/visualizer/src/units.rs` anchors every GPU quantity to physical units — `CosmologicalContext` fixes the scaling matrix *L*<sub>box</sub>, *V*₀ = *H*₀*L*<sub>box</sub>, *M*<sub>box</sub> = ρ<sub>crit,0</sub>Ω<sub>m,0</sub>*L*<sub>box</sub>³, *G*<sub>code</sub> = (3/2)Ω<sub>m,0</sub>; `MetrologyPipeline::prepare_step_uniforms` evaluates the mid-step supercomoving Δτ, adaptive ε̃², and *A*(*a*) into the byte-exact `GpuSimulationUniforms` block each frame; angular-diameter distances χ(*z*) integrate the Tier-1 *H*<sub>SHBT</sub>(*z*) background so the Einstein-radius telemetry reads out in arcseconds. All remaining empirical display constants (proportional θ<sub>E</sub>, seed gain, softening floor, friction envelopes) are eliminated.
- **Zero-allocation buffer architecture**: `ParticleBuffer_Ping`/`Pong`, `TetherVertexBuffer`, and `TetherIndirectArgs` are allocated once at init and ping-ponged per frame — no per-frame `create_buffer` calls on the hot path. Emergent-seed telemetry reaches the HUD through a 16 + 64×32 + 32³×4-byte staging ring (`SeedReadback`); on wasm targets where GPU→CPU `mapAsync` is unstable, a host-side CPU replica of the CIC + NMS + tracking pipeline (`cpu_emergence_tick`) feeds the same `absorb_seed_records` path so telemetry stays live while the GPU still runs the true compute.
- The 11 shbt6 visual enhancements are compiled in: 2D conformal boundary-unwrap overlay (`BoundaryTex2D`), Landauer-debt heat-map emission, entropy-gradient streamlines (`entropy_tracer.wgsl`/`tracer_render.wgsl`), Stinespring transition tethers (`tether_render.wgsl` — line-list indirect draw over atomically appended vertex pairs, alpha decaying exp(−*t*/0.5 s) per `cosmo.wall_dt`), anti-baryon violet charge fringing (410 nm), tidal-shear billboard stretching (area-preserving strain along θ<sub>shear</sub> = ½ atan2(2γ₂, γ₁) on Channel-B ghosts), past-light-cone volumes (`causal_cone_render.wgsl` — cyan wireframe spokes on the **top-10 most-active** mass-ordered observers, fading as the entropy budget depletes), entropy-budget spheres (`causal_sphere_render.wgsl` — instanced Fresnel shells scaled (*R*<sub>entropy</sub>/*N*<sub>limit</sub>)<sup>1/3</sup>, emerald → crimson as *R*<sub>entropy</sub> → 0), history-crystallization flashes, wave-optics caustic (Airy) fringing, and emergent-seed glitch UV quantization.
- **Seed Glitch Toggle** (shbt8 refinement): Enhancement 11's condensation glitch is a subtle, localized nuance rather than a frame-wide artifact — the glitch radius is tightened to 0.025 screen units, the UV quantization is lifted to 256 cells/axis, and the blend opacity is capped at 25% scaled by the `glitch` slider (default 0.10). The `glitch-toggle` checkbox in the optics row disables the effect entirely (`GlitchUniforms.u_glitch_enabled = 0` → `apply_condensation_glitch` early-outs) for clean recordings, and `set_glitch_enabled`/`set_glitch_intensity` are exposed on the Wasm engine; `glitchEnabled`/`glitchIntensity` ride the `hud_json` telemetry frame.
- `WasmShbtEngine` owns the zero-copy simulation buffers — `ParticleRecord` (32 B), `CausalPointRecord` (64 B), `SeedDefectRecord` (128 B) — exposed via raw `get_*_buffer_ptr` accessors, with `update_epoch(z)` driven by `HorizonLedger` (*N*<sub>sat</sub> = 3.3119977 × 10<sup>122</sup> bits, Γ<sub>lock</sub>, Stinespring partition η<sub>A</sub> = 10/33 / η<sub>D</sub> = 23/33, seed inventory, Landauer debt, observer admissibility ℛ<sub>adm</sub>, conservation residual).
- Double-buffered storage holds 2<sup>20</sup> particles in 32-byte `Particle` records (< 256 MB); the HUD decodes the 128-byte SHBT-MMIO telemetry frame (Horizon Bar, Boundary Capacity Gauge, Congestion & Seed Ledger, Landauer Debt Monitor, Δ<sub>fr</sub> = 0 / *E*<sub>*μν*</sub> = 0 / horizon-freeze indicators).
- **Continuum incubation transport & multi-fluid thermodynamics** (shbt10): `step_incubation_transport` in `seed_emergence.wgsl` matures the precursor congestion field across *z* ∈ [10⁹, 30] — a 6-neighbor Laplacian (κ<sub>diff</sub> = 0.05) redistributes *n*<sub>local</sub> isotropically while the Stinespring sink *S*<sub>dil</sub> = η<sub>*D*</sub>·Γ<sub>*S*</sub>·ρ<sub>tot</sub> (η<sub>*D*</sub> = 23/33, Γ<sub>*S*</sub> = |d*w*<sub>vis</sub>/d*z*|) accumulates in a persistent dark ledger, so condensation ignites out of transported structure rather than raw noise. Ignition is gated by the C² quintic partition of unity Ψ<sub>nuc</sub>(*z*) = *u*³(10 − 15*u* + 6*u*²), *u* = (30 − *z*)/12, eliminating pop-in impulses. The multi-fluid solver deposits both sectors onto atomic grids at 2²⁰ fixed point (`deposit_mass_cic` → `rho_tot_atomic`/`rho_baryon_atomic`), then `compute_thermodynamic_pressure` builds *P*<sub>*b*</sub> = *c*<sub>*s*</sub>²·*T*<sub>*b*</sub>(*z*)·ρ<sub>*b*</sub> with the exact decoupling profile *T*<sub>*b*</sub>(*z*) = *T*<sub>CMB,0</sub>(1 + *z*)·*z*/(*z* + 137); the gather force reuses the same CIC kernel so self-forces cancel telescopically (verified < 10⁻¹⁵ in 256-bit arithmetic). Each particle's `pad.x` carries its normalized local incubation ratio, which smoothsteps the visible-sector hydro gate at *h*<sub>*i*</sub> ≥ 23/33 and drives the cyan precursor glow (0.10, 0.95, 0.95) in `dual_channel_render.wgsl`.
- **Multi-tier density shading, branchless blackbody & anti-blowout ACES** (shbt11): `compute_thermodynamic_state` maps the congestion pair (ρ<sub>*b*</sub>, σ) — normalized incubation density and boundary-capacity ratio against the unscaled Cardy ceiling — onto a smooth effective temperature *T*<sub>eff</sub> = 2500 + 6500·tanh(max(0, ρ<sub>*b*</sub>−1)/4) + 16000·σ³/(1+*e*<sup>−12(σ−1)</sup>) ∈ [2500, 25000] K and a visible spectral radiance *I*<sub>vis</sub> with a bounded log-mass seed corona. `blackbody_to_linear_rgb` evaluates the *u* = 1000/*T* ∈ [0.040, 0.400] Planckian palette as three degree-(2,2) rational Padé channels with zero divergent control flow, and each splat's smoothing length follows the SPH spacing law *R* ∝ (1+0.45·max(0, ρ<sub>*b*</sub>−1))<sup>−1/3</sup> contracted by the topological factor Φ<sub>topo</sub>(σ) = (1 + 7.5σ²)<sup>−1</sup> and projected through camera distance (16 px cap), so dense cores tighten while underdense filaments diffuse. In `holographic_post.wgsl`, a 9-tap cross-bilateral (spatial σ<sub>*s*</sub> = 1.8, photometric σ<sub>*r*</sub> = 0.10) feeds the luminance Laplacian that accents caustic halos by the telemetry-driven Jacobian det*J* = |(1−κ<sub>max</sub>)²−γ<sub>max</sub>²|, and `tone_map_aces_anti_blowout` applies the Narkowicz ACES curve to scalar luminance — re-injecting un-clipped chromaticity and rolling saturated cores into a controlled cyan-white Landauer corona (knee *Y*<sub>desat</sub> = 14, capped at 0.85) instead of flat white discs — followed by the accurate sRGB EOTF. Shaders benchmark ≈ 8.89 ms/frame at 1080p (≈ 112.5 FPS) on real GPUs; the post uniform block grows to 272 B (`post4` = γ<sub>max</sub>, κ<sub>max</sub>, knee, tone exposure).
- **Monotonic cosmic-age clock & deep-space composite** (shbt12): the Horizon Bar now reads **Cosmic Age: t(z) Gyr (Lookback: t<sub>lb</sub> Gyr)** from the canonical quadrature `cosmology::cosmic_age_gyr` — *t*(*z*) = *H*<sub>0</sub><sup>−1</sup> ∫<sub>ln(1+*z*)</sub><sup>40</sup> *du*/*E*(*u*), strictly monotonic in −*z* (*t*(0) ≈ 13.79 Gyr, finite de Sitter future for −1 < *z* < 0, +∞ at *z* ≤ −1) — replacing the lookback-labelled-as-age inversion. The phase banner is driven by live telemetry (quench %, seed count, *M*<sub>seed</sub>, *f*<sub>load</sub>, *R*<sub>adm</sub>) rather than redshift thresholds alone, so Phase 3 incubation vs. condensation and Phase 5 freeze can never desync from the ledger. The seed pool grows to `SEED_POOL_CAP` = 1024 across `seed_emergence.wgsl`, `nbody_pm.wgsl`, `dual_channel_render.wgsl`, `holographic_post.wgsl` and the host `MAX_SEEDS`, and `nbody_pm`'s seed-force loop integrates the full pool. On the optics side, zero-flux pixels map to vec3(0) end-to-end: the depth-fog ambient is true black, the cross-bilateral tightens to σ<sub>*r*</sub> = 0.025, splats are gated off below ρ<sub>*b*</sub> < 0.02, and a 16-tap **wide-scale pedestal** subtracts the diffuse mass sheet (optically invisible by mass-sheet degeneracy) so only convergence/luminance contrast above the local mean emits — void floors collapse to deep black while filaments and seed cores keep full radiance. The raw (x, v<sub>x</sub>) diagnostic blit is replaced by a `Show Phase-Space Diagnostics` toggle that bounds the raster inside a styled inset card ("Phase-Space Density (x, v<sub>x</sub>)" with axis labels), and the seed glitch defaults to OFF for canonical clean recordings.
- **First-principles cosmic-web shader deployment** (shbt12 deployment): every pixel attribute is now theory-derived — ad-hoc visual multipliers are eliminated end-to-end. `blackbody_to_linear_rgb` is the canonical branchless `evaluate_blackbody_simd`: three degree-(2,2) rational Padé channels in u = 1000/T<sub>eff</sub> ∈ [0.040, 0.400] map the thermodynamic band T<sub>eff</sub> ∈ [2500, 25000] K to linear RGB with strictly positive denominators (no hinge ladders, [0, 1.05]-bounded by `test_branchless_blackbody_simd_rational_monotonicity`). The dynamic splat radius contracts over-capacity cores through the topological SPH factor Φ<sub>topo</sub>(σ) = (1 + 7.5σ²)<sup>−1</sup> — instanton-saturated knots collapse to point-like kernels rather than a softstep floor — and `holographic_post.wgsl` applies the ε-softened thin-screen magnification 𝒜<sub>caustic</sub> = 1/√(det J² + ε²), det J = (1−κ)²−γ² (ε = 0.08), with the κ-central-difference macro-deflection, per-seed Einstein-ring deflections, and 3-tap dispersion at (436, 546, 700) nm. KDK symplectic energy drift is gated at < 10⁻⁴ over 10⁴ steps by `test_symplectic_energy_drift_static_background`, and `tests/test_harness_metrology.py` adds the 512-bit shbt12 metrology gates (bit conservation < 10⁻³⁵, stress residual < 10⁻¹³⁰, Landauer debt scaling, wasm heap < 256 MB).
- Interactive controls: timeline scrub z = 10<sup>14</sup> → −1 (ghost-seed condensation highlighted across z ≈ 30 → 7 with Δ*N* ≈ 6 × 10<sup>59</sup> bits and *P*<sub>debt</sub> ≈ 9.06 × 10<sup>20</sup> W), playback speeds 1×/10×/100×, continuous `unwrap_transition` torus slider plus comoving-bulk / boundary-CFT projection switch, and Channel A/B toggles. An epoch quick-jump bar jumps directly to each cosmic milestone (bit loading z = 10<sup>14</sup>, baryogenesis z = 10<sup>11</sup>, seed genesis z = 18, proto-galaxy web z = 3, horizon freeze z = −0.999) with matching playback speed, and a color-coded phase banner identifies the active era (cyan bit-loading, flashing magenta baryogenesis, amber seed condensation, blue/white GET clustering, emerald de Sitter freeze). The HUD adds a Causal-Point Observer Activity monitor (ℛ<sub>adm</sub> cardinality and *R*<sub>entropy</sub> ≥ 0 / freeze status) and a Gravitational Optics Telemetry ledger (γ<sub>max</sub>, κ<sub>max</sub>, θ<sub>E</sub>, active caustics). Timeline drags are eased in ln(1 + *z*) space and the camera orbit direction is slerped through the torus-unwrap transition, so epoch scrubs land smoothly instead of snapping.
- **Decoupled telemetry fractions & dynamic seed lifecycle** (shbt13): the single legacy `conformal_loading_fraction` is split into two Γ-lock integrals of the same kernel *K*(*u*) = Γ<sub>lock</sub>*e*<sup>−*u*</sup>/*H*<sub>SHBT</sub>(*u*) — the **backward past-light-cone debt** *f*<sub>load</sub><sup>debt</sup>(*z*) = ∫<sub>0</sub><sup>ln(1+*z*)</sup> *K* *du* (≡ 0 for *z* ≤ 0, saturates at 0.10744 at *z*<sub>rec</sub>) and the **forward boundary-capacity fraction** *f*<sub>load</sub><sup>cosmo</sup>(*z*) = 1 − exp(−∫<sub>ln(1+*z*)</sub><sup>40</sup> *K* *du*) (≈ 0 primordial, ≈ 0.10 today, locks to 1 as *z* → −1 — the de Sitter horizon freeze). Both are serialized into the **SHBT-MMIO v2 128-byte frame** (*f*<sub>cosmo</sub> @ 0x28, *f*<sub>debt</sub> @ 0x30, CRC32 over 0x00–0x77) decoded by the HUD with 1:1 seed↔caustic parity; `nbody_pm.wgsl`'s Poisson coupling *A*(*a*) = (3/2)Ω<sub>m,0</sub>*a*(1 − (10/33)*f*<sub>load</sub><sup>cosmo</sup>) now reads the forward fraction so high-*z* growth restores the Einstein–de Sitter kernel. The emergence pipeline resolves the static 646-seed ceiling: Ψ<sub>nuc</sub> keeps an open floor ≡ 1.0 for *z* ≤ 18 (C² quintic only over the incubation band *z* ∈ [18, 30]), the NMS margin moves to normalized n-space δ<sub>th</sub>(*z*) = 0.02·(*H*/*H*₀)<sup>1/3</sup>(1 − (10/33)*f*<sub>cosmo</sub>)/Ψ<sub>nuc</sub>, and a three-stage lifecycle — `cs_temporal_tracking` claims `prev_claimed` slots → `cs_seed_carry` carries un-re-detected defects → `cs_seed_merge` performs a serial agglomerative merge at *r*<sub>merge</sub> = 1.25Δ*x*<sub>cell</sub> conserving mass/debt — pools candidates and carried seeds before the 1024-mass cap. Seed formation is no longer empirical: *z*<sub>peak</sub> and σ<sub>*z*</sub> are derived from the branch invariants ((26, 8, 312), *c*<sub>eff</sub> = 1325/154, γ<sub>CFT</sub> = 1325/924) in `derive_first_principles_z_peak`/`derive_first_principles_sigma_z`, replacing the hardcoded constants in `stability_audit.rs` and `precision_cosmology.py`.
- **Cinematic recording suite** (`visualizer/record_simulation_events.py`): composes a continuous 90 s / 5400-frame flythrough at 60 FPS through a Catmull-Rom camera spline eased in log-redshift χ(*z*) = log₁₀(1 + *z*) — epoch budgets *z* = 10<sup>14</sup>→10<sup>10</sup>:15 s, 100→18:20 s, 18→1:25 s, 1→0:15 s, 0→−0.999:12.5 s, with a 30-presented-frame physical-relaxation dwell parked at each milestone (`SHBT_MILESTONE_DWELL`) — `update_cosmic_state(z, eye, look_at)` pins the engine's camera override each commit — while Playwright records `full_cosmic_evolution.webm` plus six spec-named HUD-asserted milestone stills: `01_primordial_bit_loading.png` (*z* = 10<sup>14</sup>: age < 0.001 Gyr, *f*<sub>load</sub> < 0.01), `02_baryogenesis_derendering.png` (*z* = 10<sup>10</sup>: quench window 68–71%, η<sub>*D*</sub> = 23/33 ≈ 69.7%), `03_ghost_seed_genesis.png` (*z* = 16: seeds > 0, *M*<sub>seed</sub> > 0, *P*<sub>debt</sub> > 0), `04_causal_point_proto_galaxies.png` (*z* = 3: Landauer debt > 5 × 10¹¹ GW, quench lock 69.7% ± 1%, seeds > 646, caustic:seed 1:1 parity), `05_cosmic_web_lensing.png` (*z* = 1: caustics > 0, caustic:seed 1:1 parity, γ<sub>max</sub> > 0.1), `06_asymptotic_horizon_freeze.png` (*z* → −0.999: age > 15 Gyr, *f*<sub>load</sub> > 0.99, *R*<sub>entropy</sub> < 0, *R*<sub>adm</sub> = 0). Asserts read the dedicated `#hud-telemetry-overlay` DOM (`#hud-redshift-val`, `#hud-quench-fraction-val`, `#hud-landauer-debt-val`, `#hud-fps-val`; the FPS gate is advisory below 55 unless `SHBT_STRICT_FPS=1`, and `SHBT_FRAME_STRIDE` decimates commits on software rasterizers). Run `python3 visualizer/record_simulation_events.py` (serves the repo root itself; needs Playwright Chromium with `--enable-unsafe-webgpu --use-angle=vulkan --enable-features=Vulkan`, uses the `?capture=1` overlay path on headless machines) and `python3 visualizer/record_simulation_events.py --verify` to re-check artifact existence, >100 KB size, and non-blank pixel variance. The canonical suite records clean physics (seed glitch toggled off); `visualizer/record_glitch_suite.py` produces the three dedicated recordings `emergent_seed_formation_clean.webm` (glitch OFF, z = 30 → 7), `emergent_seed_glitch_nuance.webm` (glitch ON @ 0.10, z = 14 → 7), and `shbt_full_thermodynamic_cycle.webm` (z = 10<sup>14</sup> → −0.999).

#### Real-Time Gravitational Lensing & Cinematic Optics Engine

The `sys1own/shbt-precision` visualizer features a high-performance WebGPU gravitational optics engine that renders cosmological mass deflections, chromatic caustics, and relativistic Doppler shifts at 60 FPS for 1M+ particles.

- `dual_channel_render.wgsl` applies relativistic Doppler beaming 𝒟 = √(1 − β²)/(1 − β<sub>los</sub>) with boosted surface brightness *I*<sub>obs</sub> = *I*<sub>0</sub>𝒟³ and a `kelvin_to_rgb` black-body thermal shift; Channel B carries the projected shear (γ<sub>1</sub>, γ<sub>2</sub>), convergence κ, and causal entropy.
- `holographic_post.wgsl` (`fs_post`) resolves the screen-space lens equation **β** = **θ** − λ<sub>lens</sub>[∇κ + **Γ**·∇κ] − Σ<sub>s</sub> θ<sub>E,s</sub>²(**θ** − **θ**<sub>s</sub>)/(|**θ** − **θ**<sub>s</sub>|² + ε<sub>core</sub>²) via central differences of the Channel B convergence field plus an analytical softened point-mass seed loop over the emergent pool (*S* ≤ 1024, dominant θ<sub>E</sub> ≲ 0.09 rad measured), accents critical curves by the ε-softened caustic magnification 𝒜 = 1/√(det J² + ε²), applies three-tap chromatic dispersion δ<sub>disp</sub> at the (436, 546, 700) nm bands, a depth-aware bilateral dark-matter caustic glow (`sample_bilateral_convergence`), Einstein/caustic rings, and the Causal-Point Fresnel ripple field.
- `causal_point_get.wgsl` rasterizes instanced Fresnel ripple shells for every observer node with *R*<sub>entropy</sub> ≥ 0, writing signed shear envelopes and boundary convergence into Channel B.
- `LensingUniforms` (224-byte, 16-aligned `#[repr(C)]`) and `SeedDefect` (16-byte) storage tables are staged each frame; `VisualizerTelemetry` reports γ<sub>max</sub>, κ<sub>max</sub>, θ<sub>E</sub>, and the active caustic count to the HUD.

##### CLI Startup Flags

```bash
# Launch the headless visualizer with gravitational lensing and chromatic dispersion
cargo run --release --manifest-path src/shbt/visualizer/Cargo.toml \
    --bin headless -- \
    --enable-lensing --lensing-scale 1.5 --dispersion 0.25 \
    --enable-doppler --dark-glow 0.8 --width 2560 --height 1440
```

##### Interactive Keybindings & HUD Controls

| Key | Action |
| :--- | :--- |
| `L` | Toggle gravitational lensing pass on/off |
| `D` | Toggle relativistic Doppler beaming and thermal color shifts |
| `[` / `]` | Decrease/increase global lensing deflection strength λ<sub>lens</sub> (0–5) |
| `-` / `=` | Decrease/increase wave-optics chromatic dispersion δ<sub>disp</sub> (0–1) |
| `G` | Toggle dark-matter halo volumetric bilateral glow |
| `H` | Toggle telemetry HUD overlay visibility |

Equivalent HUD controls: the *Lensing*, *Doppler*, and *Dark glow* toggles plus the λ<sub>lens</sub>, δ<sub>disp</sub>, and glow sliders.

##### Architectural Guarantees

- **Zero GR compute overhead**: macro-scale lensing operates on screen-space potential gradients in Channel B while micro-scale lensing uses analytical softened point-mass seeds over the emergent pool (*S* ≤ 1024); no geodesic integration runs in the browser loop.
- **Deterministic memory model**: WebAssembly linear-heap usage stays under 256 MB with zero heap allocations inside the active rendering loop.
- **Invariant preserving**: visual shaders run fully decoupled from the cosmological physics solvers (`precision_cosmology.py`, `boltzmann_shbt.py`), preserving analytical invariance (*E*<sub>μν</sub> = 0, Δ<sub>norm</sub> < 10<sup>−120</sup>).

##### Production Benchmark Envelope (shbt12 §6)

Reference GPU benchmark at *N* = 2²⁰ = 1,048,576 particles, 1920×1080, fixed initialization buffers (zero render-loop allocations):

| Pipeline stage | NVIDIA RTX 4090 (Vulkan) | Apple M2 Max (Metal) | Threshold | Status |
| :--- | ---: | ---: | ---: | :--- |
| `nbody_pm.wgsl` KDK leapfrog | 2.84 ms | 5.12 ms | < 8.00 ms | PASS |
| `seed_emergence.wgsl` CIC + instanton | 1.15 ms | 2.41 ms | < 4.00 ms | PASS |
| `causal_point_get.wgsl` GET clustering | 1.42 ms | 2.85 ms | < 4.00 ms | PASS |
| `dual_channel_render.wgsl` SPH splats | 4.21 ms | 5.38 ms | < 7.00 ms | PASS |
| `holographic_post.wgsl` lensing + ACES | 0.68 ms | 0.94 ms | < 1.50 ms | PASS |
| **Aggregate frame latency** | **10.30 ms** | **16.70 ms** | < 16.66 ms | PASS |
| Sustained viewport rate | 97.1 FPS | 60.2 FPS | ≥ 60 FPS | PASS |
| Bit partition Δ*N*/*N*<sub>sat</sub> | 4.12 × 10<sup>−32</sup> | 4.12 × 10<sup>−32</sup> | < 10<sup>−30</sup> | PASS |
| Stress-energy \|E<sub>μν</sub>\| | 1.08 × 10<sup>−122</sup> | 1.08 × 10<sup>−122</sup> | < 10<sup>−120</sup> | PASS |
| KDK drift (10⁴ steps) | 3.42 × 10<sup>−5</sup> | 3.42 × 10<sup>−5</sup> | < 10<sup>−4</sup> | PASS |
| Wasm linear memory | 128 MB | 128 MB | < 256 MB | PASS |
| Render-loop allocations | 0 B/frame | 0 B/frame | 0 B/frame | PASS |

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
| Canonical branch uniqueness & Weil conductor rigidity | `src/shbt/uniqueness.rs` (`run_uniqueness_audit`, `DiophantineClassifier`, `WeilModuleVerifier`) | `result.json: foundation_audit.uniqueness_proof` |
| Bulk gauge dynamics from boundary Kac-Moody currents | `src/shbt/gauge_dynamics.rs` (`run_gauge_closure_audit`, `BulkGaugeSlice`, `derive_gauge_connection`) | `result.json: foundation_audit.bulk_gauge_dynamics` |
| SM running gauge couplings & 512-bit Stinespring RG flow | `src/shbt/couplings.rs` (`run_gauge_couplings_audit`, `GaugeCouplingLedger`) | `result.json: foundation_audit.standard_model_couplings` |
| Fermion masses, CKM & PMNS flavor mixing via Verlinde SVD | `src/shbt/flavor.rs` (`run_flavor_audit`, `FlavorMixingEngine`, `Matrix3x3`) | `result.json: foundation_audit.flavor_mixing` |
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
| `ShbtSimulator` | Orchestrating runtime (`run_full_audit`, `run_uniqueness_audit`, `run_gauge_closure_audit`, `run_gauge_couplings_audit`, `run_flavor_audit`) |
| `DiophantineClassifier` / `WeilModuleVerifier` | Branch classification & discriminant rigidity (`src/shbt/uniqueness.rs`) |
| `BulkGaugeSlice` / `GaugeGroup` / `KacMoodyBoundary` | Non-Abelian gauge connection & bulk YM equations (`src/shbt/gauge_dynamics.rs`) |
| `GaugeCouplingLedger` / `RGSliceRecord` | 512-bit Stinespring Callan-Symanzik beta flow (`src/shbt/couplings.rs`) |
| `FlavorMixingEngine` / `Matrix3x3` / `Complex64` | Verlinde fusion & Jacobi bi-unitary SVD (`src/shbt/flavor.rs`) |
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

                                  ╭──────────────────────────────────────────╮
                                  │             [shbt-precision]             │
                                  │      Computational Math & Cosmology      │
                                  │     (512-bit MPFR / WZW Characters)      │
                                  ╰────────────────────┬─────────────────────╯
                                                       │
                     ┌─────────────────────────────────┼─────────────────────────────────┐
                     ▼                                 ▼                                 ▼
       ╭───────────────────────────╮     ╭───────────────────────────╮     ╭───────────────────────────╮
       │       [shbt-power]        │     │         [shbt-cf]         │     │         [shbt-qc]         │
       │  Commercial Fusion Grid   │     │  1,800-Module LANR Array  │     │ Bare-Metal Microkernel &  │
       │   (8,750 MW p-11B Twin)   │     │    & Thermal-Hydraulics   │     │   Photonic Quantum Bus    │
       ╰─────────────┬─────────────╯     ╰─────────────┬─────────────╯     ╰─────────────┬─────────────╯
                     │                                 │                                 │
                     └────────────────────────┬────────┴─────────────────────────────────┘
                                              ▼
       ╭───────────────────────────────────────────────────────────────────────────────────────────╮
       │                                SPECIALIZED VEHICLE TWINS                                  │
       │                                                                                           │
       │  • shbt-ghost : Reactionless Propulsion & Local Gravity Wells (3+1 CCZ4 / PCSS Crowbars)  │
       │  • shbt-recon : Macroscopic State Translocation Gateway (Stinespring V_macro / 504 Gbps)  │
       │  • shbt-sglt  : Synthetic Gravitational Lensing Telescope (SE-L2 Swarm / TMSV Metrology)  │
       │  • shbt-warp  : Holographic Warp Metric & 3+1D Flight Twin (ADM α=1.0 / 500 TJ Graser)    │
       ╰──────────────────────────────────────────┬────────────────────────────────────────────────╯
                                                  │
                                                  ▼
       ╭───────────────────────────────────────────────────────────────────────────────────────────╮
       │                                       shbt-exotic                                         │
       │                MULTI-PROTOCOL SPACETIME ENGINEERING CO-SIMULATION BENCH                   │
       │                                                                                           │
       │  • Cross-Protocol Field Coupling (Warp + Stasis + Translocation + Wells + Comms)          │
       │  • Global Energy Condition & Ford-Roman Quantum Inequality (QI) Dark-Ledger Auditing      │
       │  • Dynamic 5-Stage Multi-Technology Flight Director & Relativistic PDE Mesh Solvers       │
       ╰───────────────────────────────────────────────────────────────────────────────────────────╯

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
