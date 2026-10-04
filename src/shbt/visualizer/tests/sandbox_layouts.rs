//! Byte-layout bindings between the live engine types and the WGSL
//! buffer contracts introduced in shbt9 Phase 1-2: the 24-byte
//! SimulationControls block, the 32-byte sandbox CausalPointRecord,
//! the 48-byte engine Particle stride, and the 272-byte post uniform.

use shbt_visualizer::{
    HudCausalRecord, LensingUniforms, Particle, SimulationControls,
};
use std::mem::size_of;

#[test]
fn test_simulation_controls_contract() {
    assert_eq!(size_of::<SimulationControls>(), 24);

    let clamped = SimulationControls {
        sound_speed_scale: 5.0,
        percolation_threshold_scale: 0.2,
        lensing_strength: 8.0,
        chromatic_dispersion: 0.9,
        target_redshift: -5.0,
        viewport_split_mode: 1,
    }
    .clamped();

    assert_eq!(clamped.sound_speed_scale, 3.0);
    assert_eq!(clamped.percolation_threshold_scale, 0.5);
    assert_eq!(clamped.lensing_strength, 5.0);
    assert_eq!(clamped.chromatic_dispersion, 0.5);
    assert_eq!(clamped.target_redshift, -0.999);
    assert_eq!(clamped.viewport_split_mode, 1);
}

#[test]
fn test_causal_point_record_contract() {
    assert_eq!(size_of::<HudCausalRecord>(), 32);
}

#[test]
fn test_particle_and_post_uniform_strides() {
    // Engine particle stride bound into every nbody/emergence pass
    // (16 x f32: position, channel, velocity, charge_flags, shear,
    // landauer_debt, grav_mass, pad).
    assert_eq!(size_of::<Particle>(), 64);
    // 192-byte base uniform + post0..post3 vec4 extension slots.
    assert_eq!(size_of::<LensingUniforms>(), 272);
}
