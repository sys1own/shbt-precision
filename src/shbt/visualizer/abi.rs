//! Deterministic C-ABI structs strictly aligned to 16-byte boundaries for WebGPU MMIO.
//! Specification: src/shbt/visualizer/abi.rs

#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ShbtMmioHeaderV2 {
    pub magic: u32,             // 0x54424853 ("SHBT")
    pub schema: u32,            // 0x00020000 (v2.0)
    pub frame_index: u32,
    pub particle_count: u32,
    pub causal_point_count: u32,
    pub seed_defect_count: u32,
    pub scale_factor_a: f32,
    pub redshift_z: f32,
    pub h_over_h0: f32,
    pub eta_a: f32,             // 10.0 / 33.0
    pub eta_d: f32,             // 23.0 / 33.0
    pub landauer_debt_gw: f32,
    pub n_sat_hi: f32,          // Multi-word float: hi-part of 3.3119977e122
    pub n_sat_lo: f32,          // Multi-word float: lo-part
    pub framing_defect: f32,    // Strictly 0.0
    pub padding: u32,
    pub _reserved: [u8; 64],    // Pads 64 -> 128 bytes
}

#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ParticleRecord {
    pub position: [f32; 3],     // Comoving coordinates [Mpc/h]
    pub mass: f32,              // Code mass units
    pub velocity: [f32; 3],     // Peculiar velocity [km/s]
    pub fluid_type: u32,        // 0 = Baryon (A), 1 = Passive Ghost DM (D)
}

#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CausalPointRecord {
    pub world_pos: [f32; 3],
    pub lifecycle_phase: u32,   // 0=Seeding, 1=Accretion, 2=Crystallizing, 3=Saturated, 4=Succession
    pub horizon_radius: f32,
    pub entropy_admissibility: f32, // R_entropy in [0, 1]
    pub get_acceleration: [f32; 2], // Packed a_GET vector
    pub projector_state: [f32; 4],  // Rank-one history projector |A, omega_iota>
    pub succession_arc: [f32; 4],   // Geodesic parameterization [t0, t1, curvature, target_id]
}

#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SeedDefectRecord {
    pub root_anchor: [f32; 3],
    pub seed_mass_msun: f32,
    pub cardy_c_eff: f32,
    pub weil_phase_idx: u32,
    pub landauer_emission_rate: f32,
    pub d5_kappa_star: f32,
    pub lorentz_sigma: f32,
    pub topological_contraction: f32,
    pub padding: [f32; 2],
    pub defect_tensor: [f32; 16],   // 4x4 metric perturbation defect (64 bytes)
    pub _reserved: [f32; 4],        // 16 bytes -> total 128 bytes
}

#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct MeshCellAtomic {
    pub density_fixed: u32,
    pub velocity_x_fixed: u32,
    pub velocity_y_fixed: u32,
    pub velocity_z_fixed: u32,
}

#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CosmicBandHistogram {
    pub bins: [u32; 4096],       // 64x64 phase-space bins
}

unsafe impl bytemuck::Pod for CosmicBandHistogram {}
unsafe impl bytemuck::Zeroable for CosmicBandHistogram {}

const _: () = assert!(std::mem::size_of::<ShbtMmioHeaderV2>() == 128);
const _: () = assert!(std::mem::size_of::<ParticleRecord>() == 32);
const _: () = assert!(std::mem::size_of::<CausalPointRecord>() == 64);
const _: () = assert!(std::mem::size_of::<SeedDefectRecord>() == 128);
const _: () = assert!(std::mem::size_of::<MeshCellAtomic>() == 16);
const _: () = assert!(std::mem::size_of::<CosmicBandHistogram>() == 16384);
