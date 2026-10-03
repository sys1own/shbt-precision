# Emergent Condensation — Theoretical & Algorithmic Traceability Matrix

Maps the formal paper equations onto their WGSL implementation identifiers,
Rust engine structures, and the mathematical invariant each one targets.
Source specification: shbt6 (Emergent Mass-Congestion roadmap, Report 1).

## Invariant non-regression gates

| Invariant | Bound | Verified by |
|---|---|---|
| Bit conservation \|(*N*<sub>vis</sub> + *N*<sub>dark</sub>) − *N*<sub>sat</sub>·*f*<sub>load</sub>(*z*)\| / *N*<sub>sat</sub> | < 1.0 × 10<sup>−30</sup> | `cargo test` (visualizer crate), `python examples/run_audit.py` |
| Stress-energy conservation ℰ<sub>*μν*</sub> = ∇<sup>*μ*</sup>(*T*<sub>*μν*</sub><sup>vis</sup> + *T*<sub>*μν*</sub><sup>dark</sup> + *T*<sub>*μν*</sub><sup>seed</sup>) | < 1.0 × 10<sup>−120</sup> | `cargo test` (visualizer crate), `run_audit.py` |
| Zero-hardcoding gate: δ ≡ 0 ⇒ Δ*N* ≡ 0 ⇒ *N*<sub>seeds</sub> = 0 at *z* = 30 | exact | Playwright `tests/browser/test_emergent_seeds.spec.js` |
| Emergence & accretion: *z* = 30→7 scrub nucleates *M*<sub>seed</sub> > 10<sup>8</sup> M<sub>⊙</sub>, *P*<sub>debt</sub> > 9.0 × 10<sup>10</sup> GW, debt = mass × 906 to < 10<sup>−4</sup> | relative | same spec |
| Wasm heap | < 256 MB | same spec (memory gate) |

## Equation → implementation matrix

| Paper equation | Physical / theoretical derivation | WGSL identifier | Rust struct & member | Invariant target |
|---|---|---|---|---|
| Eq. (9.14) | Trilinear mass assignment ρ<sub>cell</sub> = Σ m<sub>i</sub> W(**x**<sub>i</sub> − **x**<sub>g</sub>) | `cs_accumulate_cic` | `GpuSimulationParameters.fixed_point_scale` | Σρ<sub>grid</sub> = *M*<sub>total</sub> ± 10<sup>−7</sup> |
| Eq. (9.19) | Local coordinate demand *N*<sub>local</sub> = 𝒦<sub>bit</sub>·ρ·(1 + δ) | `cell_entropy_demand` in `cs_detect_condensation` | `GpuSimulationParameters.particle_count` | Bit-density monotonicity |
| Eq. (9.23) | Ceiling capacity *N*<sub>limit</sub> = *N*<sub>sat</sub>·*f*<sub>load</sub>·*V*<sub>cell</sub>·γ<sub>geom</sub> / *V*<sub>box</sub> | `params.n_limit_per_cell` | `GpuSimulationParameters.n_limit_per_cell` | Homogeneous stability (δ = 0) |
| Eq. (9.28) | Congestion overflow Δ*N* = max(0, *N*<sub>local</sub> − *N*<sub>limit</sub>) | `overflow` in `cs_detect_condensation` | `GpuSimulationParameters.delta_n_thresh` | Non-negative bit overflow |
| Eq. (9.34) | Spatial centroid **x**<sub>seed</sub> = Σ **u**Δ*N* / ΣΔ*N* | `centroid` / `wrapped_centroid` | `SeedDefectRecord.position` | Periodic translation invariance |
| Eq. (9.41) | Emergent mass *M*<sub>seed</sub> = α<sub>seed</sub> Σ Δ*N* | `m_seed` / `params.alpha_seed` | `SeedDefectRecord.dynamics[0]` | Σ*M*<sub>seed</sub> = α<sub>seed</sub> ΣΔ*N* |
| Eq. (9.48) | Landauer dissipation *P*<sub>debt</sub> = (*M*<sub>seed</sub>/M<sub>⊙</sub>) × 906 GW | `p_debt` / `params.landauer_rate` | `SeedDefectRecord.dynamics[2]` | Dissipation-rate linearity |
| Eq. (9.56) | Active entropic acceleration **a**<sub>GET</sub> = −κ<sub>GET</sub> ∇ ln ρ<sub>proj</sub> | `compute_seed_gravitational_acceleration` | `WasmShbtEngine` emergence pass | Momentum conservation |

## Canonical constants

| Constant | Value | Role |
|---|---|---|
| WZW triple (*k*<sub>l</sub>, *k*<sub>q</sub>, *K*) | (26, 8, 312) | Modular boundary register; growth-amplitude sigmoid key |
| γ<sub>geom</sub> | π²/4 ≈ 2.4674 | Geometric density→capacity conversion |
| *N*<sub>sat</sub> | 3.3119977 × 10<sup>122</sup> bits | Saturated de Sitter screen capacity |
| α<sub>seed</sub> | 1.3258316 × 10<sup>−51</sup> M<sub>⊙</sub>/bit | Holographic overflow→mass coupling |
| Landauer rate | 906 GW/M<sub>⊙</sub> | Bit-erasure debt power per seed mass |
| Condensation window | *z* = 30 → 7 | 0 seeds above *z* ≈ 17.5; 64 defects by *z* ≤ 14 |
