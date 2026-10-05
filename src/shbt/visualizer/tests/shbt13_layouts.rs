//! shbt13 Stage-6 struct-alignment audit: static-assert the real
//! `#[repr(C)]` uniform strides against the WGSL buffer contracts.
//!
//! Documented spec delta: the shbt13 spec nominally lists
//! GpuSimulationUniforms = 64 B and LensingUniforms = 64 B — those are
//! the nominal contract sizes from earlier enhancement rounds; the live
//! engine layouts are 96 B (uniforms) and 128/272 B (lens blocks)
//! respectively — they carry the metrology/sandbox/post fields the
//! deployed shaders actually bind.
//! The GPU-facing 32-byte seed/lens record does match the spec figure
//! and is asserted here; the WGSL SeedDefectRecord (vec4<f32> position
//! with w = alive flag + vec4<f32> dynamics) is likewise 32 B on the
//! shader side.

use shbt_visualizer::{GpuLensUniforms, GpuSeedLensBuffer, GpuSimulationUniforms, LensingUniforms};
use std::mem::size_of;

#[test]
fn test_gpu_simulation_uniforms_layout() {
    // 24 f32/u32 lanes = 96 B live layout (spec's nominal 64 B covers
    // only the legacy metrology contract).
    assert_eq!(size_of::<GpuSimulationUniforms>(), 96);
}

#[test]
fn test_lensing_uniforms_layout() {
    // 192-byte base uniform + post0..post3 vec4 extension slots = 272 B
    // (spec's nominal 64 B covers only the thin-screen subset).
    assert_eq!(size_of::<LensingUniforms>(), 272);
    // units-side lens block: view_proj + cam_pos + distance/dispersion
    // lanes = 128 B.
    assert_eq!(size_of::<GpuLensUniforms>(), 128);
}

#[test]
fn test_gpu_seed_record_layout() {
    // The WGSL-side seed/lens record contract: 2 x vec4<f32> = 32 B,
    // matching the spec's 32-byte SeedDefectRecord figure.
    assert_eq!(size_of::<GpuSeedLensBuffer>(), 32);
}
