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
- `ShbtWebGpuEngine` drives the WGSL stage set: `seed_emergence.wgsl` (the first-principles **emergent mass-congestion pipeline** — seeds are no longer hardcoded; they condense wherever local coordinate-entropy demand exceeds the boundary-CFT capacity ceiling via a tri-pass compute: `cs_accumulate_cic` fixed-point atomic Cloud-In-Cell mass assignment, `cs_detect_condensation` 26-neighbour non-maximum suppression + 3³ basin overflow-mass integration, and `cs_temporal_tracking` minimum-image persistent-ID tracking with accretion Ṁ<sub>seed</sub> and Landauer debt *P*<sub>debt,k</sub> = *M*<sub>seed,k</sub>·906 GW/M<sub>⊙</sub>), `causal_point_get.wgsl` (active Causal-Point GET dynamics — `cs_causal_point_get` applies the *R*<sub>entropy</sub> = *N*<sub>limit</sub> − *C*<sub>get</sub> ≥ 0 admissibility gate and the entropic acceleration **a**<sub>GET</sub>(**x**) = −κ<sub>GET</sub> ∇ln ρ<sub>proj</sub>(**x**) with the first-principles transport coefficient κ<sub>GET</sub>(*f*<sub>load</sub>) = (1/(*d*<sub>1</sub> − *h*<sup>∨</sup><sub>su(3)</sub>))(1 + (*c*<sub>eff</sub>/*d*<sub>1</sub>)*f*<sub>load</sub>) = (1/23)(1 + (1325/4004)*f*<sub>load</sub>) streamed via a `TheoryInvariants` uniform), `nbody_pm.wgsl` (symplectic Martel–Shapiro **supercomoving KDK integrator** — `cs_advance_particles` half-kicks at (*x̃*<sub>*n*</sub>, *a*<sub>*n*</sub>), drifts x̃ += Δτ·p̃ with periodic wrap, then half-kicks at (*x̃*<sub>*n*+1</sub>, *a*<sub>*n*+1</sub>) using Δτ = Δ*a*/(*a*<sub>½</sub>³·*H*(*a*<sub>½</sub>)); the heuristic friction/envelope terms are gone and the conformal coupling *A*(*a*) = (3/2)Ω<sub>m,0</sub>*a*·(1 − (10/33)*f*<sub>load</sub>) multiplies the PM force directly, while seed attraction uses **adaptive Plummer softening** ε̃ = η<sub>soft</sub>/*N*<sub>grid</sub> (η ≈ 0.3333) and the GET term −κ<sub>GET</sub>∇ln ρ<sub>proj</sub> samples the register density carried in the PM texture alpha channel; the thermal Stinespring expectation *w*<sub>vis</sub>(*z*) = (1 − η<sub>*D*</sub>) + η<sub>*D*</sub>/(1 + (*z*<sub>*N*</sub>/*z*)<sup>26/3</sup>) with *z*<sub>*N*</sub> = 7.356 × 10<sup>10</sup> drives the progressive quench), `dual_channel_render.wgsl` (Channel A emission scaled strictly by *w*<sub>vis</sub>(*z*) — solar-gold 10/33 baryons vs electric-violet 23/33 anti-baryons with the *z* ∈ [10<sup>12</sup>, 10<sup>9</sup>] thermal cooling gradient — Gaussian splats exp(−3.5·*d*²) replacing square sprites, plus Landauer heat glow (*P*<sub>debt</sub> = (*M*<sub>seed</sub>/*M*<sub>☉</sub>) × 906 GW) rendered as an amber/white core + corona on each emergent seed; Channel B passive ghost shear/density conserved independently of it, carrying faint causal-point projection envelopes), and `holographic_post.wgsl` (physical distance-duality lensing — θ<sub>E,k</sub> = √(4*GM*/*c*²·*D*<sub>ds</sub>/(*D*<sub>d</sub>*D*<sub>s</sub>)) per emergent seed with Tier-1 angular-diameter distances, boundary dispersion δ<sub>disp</sub>(λ) = ζ[(550 nm/λ)² − 1] at the 436/546/700 nm bands, Reinhard HDR + contrast S-curve tonemapping, `unwrap_torus_projection` conformal unwrapping onto the CFT torus [0, 2π)², horizon overlay driven by *f*<sub>load</sub>).
- Seed condensation physics (shbt7, first-principles): condensation is an **instanton-gated tunneling event**. The Cardy capacity ceiling *N*<sub>limit</sub>(*z*) = γ<sub>CFT</sub>·(*H*/*H*<sub>ref</sub>)²·(*V*<sub>cell</sub>/*V*<sub>*H*</sub>)·*N*<sub>sat</sub> with γ<sub>CFT</sub> = *c*<sub>eff</sub>/6 = 1325/924 ≈ 1.4339826 sets the per-cell bit budget; below the ceiling the Euclidean instanton action *S*<sub>inst</sub> = (2π*c*<sub>eff</sub>/*k*<sub>*q*</sub>)·((*N*<sub>limit</sub> − *N*<sub>local</sub>)/*N*<sub>limit</sub>)² gates nucleation via Γ<sub>nuc</sub> = *A*<sub>0</sub>·e<sup>−*S*<sub>inst</sub></sup> and *P*<sub>nuc</sub> = 1 − e<sup>−Γ<sub>nuc</sub>δ*t*</sup>, while overflow is barrierless (*S*<sub>inst</sub> = 0). A PCG-hashed tunneling draw replaces the old sigmoid turn-on envelope, the empirical *K*<sub>bit</sub> multiplier is gone, and the hardcoded 64-seed cap is replaced by an atomic append into a dynamically sized `GlobalSeedBuffer` pool (`SEED_POOL_CAP` = 256). Condensed mass integrates the absolute register excess Σ<sub>basin</sub> max(0, *N*<sub>local</sub> − *N*<sub>limit</sub>), which stays bounded as *N*<sub>limit</sub> → 0 in the *z* → −1 asymptote. Kernel work is evaluated on the normalized ratio *R* = *N*<sub>local</sub>/*N*<sub>limit</sub> because absolute bit counts (~10<sup>122</sup>) overflow f32 registers.
- Dimensional metrology (shbt8): `src/shbt/visualizer/src/units.rs` anchors every GPU quantity to physical units — `CosmologicalContext` fixes the scaling matrix *L*<sub>box</sub>, *V*₀ = *H*₀*L*<sub>box</sub>, *M*<sub>box</sub> = ρ<sub>crit,0</sub>Ω<sub>m,0</sub>*L*<sub>box</sub>³, *G*<sub>code</sub> = (3/2)Ω<sub>m,0</sub>; `MetrologyPipeline::prepare_step_uniforms` evaluates the mid-step supercomoving Δτ, adaptive ε̃², and *A*(*a*) into the byte-exact `GpuSimulationUniforms` block each frame; angular-diameter distances χ(*z*) integrate the Tier-1 *H*<sub>SHBT</sub>(*z*) background so the Einstein-radius telemetry reads out in arcseconds. All remaining empirical display constants (proportional θ<sub>E</sub>, seed gain, softening floor, friction envelopes) are eliminated.
- **Zero-allocation buffer architecture**: `ParticleBuffer_Ping`/`Pong`, `TetherVertexBuffer`, and `TetherIndirectArgs` are allocated once at init and ping-ponged per frame — no per-frame `create_buffer` calls on the hot path. Emergent-seed telemetry reaches the HUD through a 16 + 64×32 + 32³×4-byte staging ring (`SeedReadback`); on wasm targets where GPU→CPU `mapAsync` is unstable, a host-side CPU replica of the CIC + NMS + tracking pipeline (`cpu_emergence_tick`) feeds the same `absorb_seed_records` path so telemetry stays live while the GPU still runs the true compute.
- The 11 shbt6 visual enhancements are compiled in: 2D conformal boundary-unwrap overlay (`BoundaryTex2D`), Landauer-debt heat-map emission, entropy-gradient streamlines (`entropy_tracer.wgsl`/`tracer_render.wgsl`), Stinespring transition tethers (`tether_render.wgsl`), anti-baryon violet charge fringing (410 nm), tidal-shear billboard stretching, past-light-cone volumes (`causal_cone_render.wgsl`), entropy-budget spheres (`causal_sphere_render.wgsl`), history-crystallization flashes, wave-optics caustic (Airy) fringing, and emergent-seed glitch UV quantization.
- **Seed Glitch Toggle** (shbt8 refinement): Enhancement 11's condensation glitch is a subtle, localized nuance rather than a frame-wide artifact — the glitch radius is tightened to 0.025 screen units, the UV quantization is lifted to 256 cells/axis, and the blend opacity is capped at 25% scaled by the `glitch` slider (default 0.10). The `glitch-toggle` checkbox in the optics row disables the effect entirely (`GlitchUniforms.u_glitch_enabled = 0` → `apply_condensation_glitch` early-outs) for clean recordings, and `set_glitch_enabled`/`set_glitch_intensity` are exposed on the Wasm engine; `glitchEnabled`/`glitchIntensity` ride the `hud_json` telemetry frame.
- `WasmShbtEngine` owns the zero-copy simulation buffers — `ParticleRecord` (32 B), `CausalPointRecord` (64 B), `SeedDefectRecord` (128 B) — exposed via raw `get_*_buffer_ptr` accessors, with `update_epoch(z)` driven by `HorizonLedger` (*N*<sub>sat</sub> = 3.3119977 × 10<sup>122</sup> bits, Γ<sub>lock</sub>, Stinespring partition η<sub>A</sub> = 10/33 / η<sub>D</sub> = 23/33, seed inventory, Landauer debt, observer admissibility ℛ<sub>adm</sub>, conservation residual).
- Double-buffered storage holds 2<sup>20</sup> particles in 32-byte `Particle` records (< 256 MB); the HUD decodes the 128-byte SHBT-MMIO telemetry frame (Horizon Bar, Boundary Capacity Gauge, Congestion & Seed Ledger, Landauer Debt Monitor, Δ<sub>fr</sub> = 0 / *E*<sub>*μν*</sub> = 0 / horizon-freeze indicators).
- Interactive controls: timeline scrub z = 10<sup>14</sup> → −1 (ghost-seed condensation highlighted across z ≈ 30 → 7 with Δ*N* ≈ 6 × 10<sup>59</sup> bits and *P*<sub>debt</sub> ≈ 9.06 × 10<sup>20</sup> W), playback speeds 1×/10×/100×, continuous `unwrap_transition` torus slider plus comoving-bulk / boundary-CFT projection switch, and Channel A/B toggles. An epoch quick-jump bar jumps directly to each cosmic milestone (bit loading z = 10<sup>14</sup>, baryogenesis z = 10<sup>11</sup>, seed genesis z = 18, proto-galaxy web z = 3, horizon freeze z = −0.999) with matching playback speed, and a color-coded phase banner identifies the active era (cyan bit-loading, flashing magenta baryogenesis, amber seed condensation, blue/white GET clustering, emerald de Sitter freeze). The HUD adds a Causal-Point Observer Activity monitor (ℛ<sub>adm</sub> cardinality and *R*<sub>entropy</sub> ≥ 0 / freeze status) and a Gravitational Optics Telemetry ledger (γ<sub>max</sub>, κ<sub>max</sub>, θ<sub>E</sub>, active caustics). Timeline drags are eased in ln(1 + *z*) space and the camera orbit direction is slerped through the torus-unwrap transition, so epoch scrubs land smoothly instead of snapping.
- **Cinematic recording suite** (`visualizer/record_simulation_events.py`): captures a continuous 1920×1080 `full_cosmic_evolution.webm` plus six telemetry-asserted milestone stills/clips under `visualizer/recordings/` — primordial bit loading (*z* = 10<sup>14</sup>), baryogenesis de-rendering (*z* = 10<sup>10</sup> → 10<sup>9</sup>), ghost-seed genesis (*z* = 18 → 14), causal-point proto-galaxies (*z* = 3), cosmic-web lensing (*z* = 0.5), and the asymptotic horizon freeze (*z* → −0.999, *f*<sub>load</sub> = 1, ℛ<sub>adm</sub> → ∅). Each milestone parks for ≥ 30 presented frames so the gravitational dynamics relax before the capture. Run `python3 visualizer/record_simulation_events.py` (serves the repo root on :8080 itself; needs Playwright Chromium with `--enable-unsafe-webgpu --use-angle=vulkan --enable-features=Vulkan`, uses the `?capture=1` overlay path on headless machines) and `python3 visualizer/record_simulation_events.py --verify` to re-check artifact existence, >100 KB size, and non-blank pixel variance. The canonical suite records clean physics (seed glitch toggled off); `visualizer/record_glitch_suite.py` produces the three dedicated recordings `emergent_seed_formation_clean.webm` (glitch OFF, z = 30 → 7), `emergent_seed_glitch_nuance.webm` (glitch ON @ 0.10, z = 14 → 7), and `shbt_full_thermodynamic_cycle.webm` (z = 10<sup>14</sup> → −0.999).

#### Real-Time Gravitational Lensing & Cinematic Optics Engine

The `sys1own/shbt-precision` visualizer features a high-performance WebGPU gravitational optics engine that renders cosmological mass deflections, chromatic caustics, and relativistic Doppler shifts at 60 FPS for 1M+ particles.

- `dual_channel_render.wgsl` applies relativistic Doppler beaming 𝒟 = √(1 − β²)/(1 − β<sub>los</sub>) with boosted surface brightness *I*<sub>obs</sub> = *I*<sub>0</sub>𝒟³ and a `kelvin_to_rgb` black-body thermal shift; Channel B carries the projected shear (γ<sub>1</sub>, γ<sub>2</sub>), convergence κ, and causal entropy.
- `holographic_post.wgsl` (`fs_post`) resolves the screen-space lens equation **β** = **θ** − λ<sub>lens</sub>[∇κ + **Γ**·∇κ] − Σ<sub>s</sub> θ<sub>E,s</sub>²(**θ** − **θ**<sub>s</sub>)/(|**θ** − **θ**<sub>s</sub>|² + ε<sub>core</sub>²) via central differences of the Channel B convergence field plus an analytical softened point-mass seed loop over the emergent pool (*S* ≤ 256, dominant θ<sub>E</sub> ≲ 0.09 rad measured), applies three-tap chromatic dispersion δ<sub>disp</sub> near critical curves, a depth-aware bilateral dark-matter caustic glow (`sample_bilateral_convergence`), Einstein/caustic rings, and the Causal-Point Fresnel ripple field.
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

- **Zero GR compute overhead**: macro-scale lensing operates on screen-space potential gradients in Channel B while micro-scale lensing uses analytical softened point-mass seeds over the emergent pool (*S* ≤ 256); no geodesic integration runs in the browser loop.
- **Deterministic memory model**: WebAssembly linear-heap usage stays under 256 MB with zero heap allocations inside the active rendering loop.
- **Invariant preserving**: visual shaders run fully decoupled from the cosmological physics solvers (`precision_cosmology.py`, `boltzmann_shbt.py`), preserving analytical invariance (*E*<sub>μν</sub> = 0, Δ<sub>norm</sub> < 10<sup>−120</sup>).

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
