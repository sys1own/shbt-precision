//! SHBT WebGPU visualizer engine (Tier 2 of the two-tier topology).
//!
//! Consumes 128-byte SHBT-MMIO telemetry frames produced by the Tier-1
//! `shbt_simulator` core and drives the Fast-PM compute pipeline, the
//! emergent mass-congestion condensation pipeline (`seed_emergence.wgsl`,
//! shbt6 Sections 1-3), the dual-channel render pass, and the holographic
//! composite pass.
//!
//! Emergent seeds (no hardcoded positions): every frame the tri-pass
//! kernel accumulates a fixed-point Cloud-In-Cell mass field
//! (`cs_accumulate_cic`), runs 26-neighborhood Non-Maximum Suppression with
//! 3x3x3 basin integration (`cs_detect_condensation`), and resolves
//! persistent identities via minimum-image tracking (`cs_temporal_tracking`)
//! into the `active_seeds` buffer that `nbody_pm.wgsl`,
//! `dual_channel_render.wgsl`, and `holographic_post.wgsl` all read.

mod engine;
mod hud;
mod particle;
mod telemetry;
mod units;

pub use engine::{CausalPointRecord, ParticleRecord, SeedDefectRecord, WasmShbtEngine};
pub use hud::{
    HorizonLedger, HudMetrics, LensingUniforms, SeedDefect, TimelineController,
    VisualizerEngine, VisualizerTelemetry, GAMMA_LOCK, N_SAT,
};
pub use particle::Particle;
pub use telemetry::encode_mmio_frame;

use bytemuck::{Pod, Zeroable};
use std::cell::RefCell;
use std::rc::Rc;
use std::sync::{
    atomic::{AtomicBool, AtomicU32, Ordering},
    Arc,
};
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::*;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

const GRID_DIM: u32 = 32;
const BOX_SIZE: f32 = 200.0; // comoving Mpc/h
const TARGET_FORMAT: TextureFormat = TextureFormat::Rgba16Float;
const MAX_SEEDS: usize = 256;
const CAUSAL_NODE_COUNT: usize = 8;
const TRACER_COUNT: u32 = 10_000;
const TETHER_CAP: u32 = 65_536;
const BOUNDARY_RES: u32 = 256;

// Canonical WZW triple (k_l, k_q, K) = (26, 8, 312) and the geometric
// saturation constants from shbt6 Section 2.
#[allow(dead_code)]
const WZW_K_L: f64 = 26.0;
#[allow(dead_code)]
const WZW_K_Q: f64 = 8.0;
#[allow(dead_code)]
const WZW_K: f64 = 312.0;
const GAMMA_GEOM: f64 = std::f64::consts::PI * std::f64::consts::PI / 4.0; // ~2.467401
/// N_sat expressed in scaled bits (1 scaled bit = 1e30 physical bits) so
/// all GPU register arithmetic stays inside f32 dynamic range.
#[allow(dead_code)]
const N_SAT_SCALED: f64 = 3.3119977e122 / 1.0e30; // 3.3119977e92
/// alpha_seed scaled: 1.3258316e-51 M_sun/bit * 1e30 = M_sun per scaled bit.
const ALPHA_SEED_SCALED: f64 = 1.3258316e-21;
/// Delta N condensation threshold: 1e57 bits = 1e27 scaled bits.
#[allow(dead_code)]
const DELTA_N_THRESH_SCALED: f64 = 1.0e27;
/// Normalized condensation threshold (fraction of the N_limit ceiling the
/// 3x3x3 basin must exceed before a candidate seed is registered).
const DELTA_N_THRESH_NORM: f32 = 0.02;
/// Seed mass per unit of normalized overflow sum_{Omega} (N_local/N_limit - 1),
/// chosen so a condensation basin integrates to ~1e8-1e9 M_sun per the
/// canonical alpha_seed coupling in scaled-bit units.
const SEED_MASS_NORM: f32 = 2.0e7;
const LANDAUER_RATE: f64 = 906.0; // GW per M_sun of collapsed register mass
const FIXED_POINT_SCALE: f64 = 1024.0;
const TRACK_RADIUS_MPC: f32 = 12.0;

// shbt7 first-principles invariants (coset CFT SO(10)_312/SU(3)_8).
#[allow(dead_code)]
const C_EFF: f64 = 1325.0 / 154.0;            // coset central charge
const GAMMA_CFT: f64 = 1325.0 / 924.0;        // c_eff/6 Cardy ceiling
const Z_REF: f64 = 17.0;                       // modular onset reference
const S_INST_PREFACTOR: f64 = 6.7576509;      // 2*pi*c_eff/k_q = 1325pi/616
/// Instanton nucleation attempt rate A_0 (per unit normalized timestep).
const ATTEMPT_FREQ: f64 = 0.5;
/// Thermal Stinespring channel constants (shbt7 Section 4 / Thm 9.10).
const ETA_D: f64 = 23.0 / 33.0;
const Z_N: f64 = 7.356e10;
const DELTA_BBAR: f64 = 26.0 / 3.0;

/// Thermal Stinespring visible overlap w_vis(z) = (1-eta_D) +
/// eta_D/(1+(z_N/z)^Delta_Bbar) — shared by the HUD ledger, the engine
/// channel-assignment quench fraction, and the CPU telemetry replica.
fn stinespring_w_vis(z: f64) -> f64 {
    if z <= 0.0 {
        return 1.0 - ETA_D;
    }
    (1.0 - ETA_D) + ETA_D / (1.0 + (Z_N / z).powf(DELTA_BBAR))
}

/// Quenched fraction of the register: (1 - w_vis)/eta_D, 0 -> 1 as z -> 0.
fn stinespring_quench_fraction(z: f64) -> f64 {
    ((1.0 - stinespring_w_vis(z)) / ETA_D).clamp(0.0, 1.0)
}

// shbt8 Phase 1: the n-body uniform block is the byte-exact
// `GpuSimulationUniforms` defined in `units.rs` (Martel-Shapiro
// supercomoving KDK contract).
type CosmoParams = units::GpuSimulationUniforms;

/// Mirrors `SimulationParameters` in seed_emergence.wgsl (16 x 4B = 64B).
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct EmergenceParams {
    grid_dim: u32,
    particle_count: u32,
    box_size: f32,
    delta_t: f32,
    redshift: f32,
    f_load: f32,
    gamma_geom: f32,
    delta_n_thresh: f32,
    alpha_mass: f32,
    landauer_rate: f32,
    track_radius: f32,
    fixed_point_scale: f32,
    // Repurposed slots (shbt7): A_0 nucleation attempt rate and the
    // per-frame decorrelation seed for the instanton tunneling draw.
    attempt_freq: f32,
    frame_seed: f32,
    mean_density: f32,
    _pad: f32,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct CameraParams {
    view_proj: [[f32; 4]; 4],
    params: [f32; 4], // x: point extent, y: box_size, z: unwrap_transition, w: redshift
    aux: [f32; 4],    // xyz: camera world position, w: doppler beaming flag
}

/// Causal-observer render node (f64-free mirror of the WGSL CausalPoint
/// record): 64 bytes, 16-byte aligned.
#[repr(C, align(16))]
#[derive(Copy, Clone, Pod, Zeroable)]
struct CausalRenderNode {
    center: [f32; 3],
    radius: f32,
    entropy_budget: f32,
    get_cost: f32,
    collapse_phase: f32,
    active_flag: u32,
    seed_index: u32,
    pad: [u32; 3],
    projection_dir: [f32; 3],
    pad2: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct CausalRenderUniforms {
    view_proj: [[f32; 4]; 4],
    cam_pos: [f32; 4],
    // x: box size, y: time, z: entropy master scale, w: billboard half-extent.
    params: [f32; 4],
}

/// Mirrors the `CausalObserver` records of causal_cone_render.wgsl and
/// causal_sphere_render.wgsl: 48 bytes, 16-byte aligned.
#[repr(C, align(16))]
#[derive(Copy, Clone, Default, Pod, Zeroable)]
struct CausalObserver {
    position: [f32; 3],
    radius: f32,
    entropy_budget: f32,
    active_flag: u32,
    cone_direction: [f32; 3],
    cone_angle: f32,
    pad: [f32; 2],
}

/// Mirrors `CrystallizationEvent` in dual_channel_render.wgsl (32B stride).
#[repr(C, align(16))]
#[derive(Copy, Clone, Default, Pod, Zeroable)]
struct CrystallizationEventGpu {
    origin: [f32; 3],
    start_time: f32,
    intensity: f32,
    pad: [f32; 3],
}

/// Mirrors `CondensationSeed` in holographic_post.wgsl (32B stride):
/// screen-space glitch emitters fed from the emergent seed table.
#[repr(C, align(16))]
#[derive(Copy, Clone, Default, Pod, Zeroable)]
struct CondensingSeedGpu {
    screen_pos: [f32; 2],
    saturation: f32,
    lifetime: f32,
    pad: [f32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct GlitchUniforms {
    count: u32,
    _pad: [u32; 7],
}

/// Mirrors `Tracer` in entropy_tracer.wgsl (32B stride).
#[repr(C, align(16))]
#[derive(Copy, Clone, Default, Pod, Zeroable)]
struct TracerGpu {
    pos: [f32; 3],
    lifetime: f32,
    vel: [f32; 3],
    entropy_val: f32,
    pad: [f32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct TracerParams {
    dt: f32,
    max_lifetime: f32,
    grid_dim: u32,
    step_scale: f32,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct TracerCamera {
    view_proj: [[f32; 4]; 4],
    params: [f32; 4],
}

/// One decoded emergent seed record from the GPU `active_seeds` buffer
/// (32 bytes: position vec4 + dynamics vec4).
#[derive(Copy, Clone, Default)]
struct EmergentSeed {
    pos: [f32; 3],
    mass_msun: f32,
    #[allow(dead_code)]
    m_dot: f32,
    p_debt_gw: f32,
    seed_id: f32,
}

/// Asynchronous GPU readback slot for the emergent-seed telemetry channel
/// (tracking state + active seed table). On native targets the map is
/// completed synchronously each frame; on wasm32 the completion callback
/// drops the bytes into `slot` for the next frame.
struct SeedReadback {
    staging: Buffer,
    slot: Rc<RefCell<Option<Vec<u8>>>>,
    /// Set by the map_async completion callback; cleared on unmap.
    mapped: Arc<AtomicBool>,
    map_calls: Arc<AtomicU32>,
    map_errors: Arc<AtomicU32>,
    map_err_text: Arc<std::sync::Mutex<String>>,
    in_flight: bool,
    map_started: bool,
}

/// WebGPU engine driving the SHBT cosmological visualizer.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
pub struct ShbtWebGpuEngine {
    device: Device,
    queue: Queue,
    #[allow(dead_code)] // surface is only consumed on wasm32 targets
    surface: Option<Surface<'static>>,
    #[allow(dead_code)]
    surface_config: Option<SurfaceConfiguration>,

    // Compute + render pipelines.
    emergence_cic_pipeline: ComputePipeline,
    emergence_detect_pipeline: ComputePipeline,
    emergence_track_pipeline: ComputePipeline,
    compute_pipeline: ComputePipeline,
    render_pipeline: RenderPipeline,
    post_pipeline: RenderPipeline,
    causal_pipeline: RenderPipeline,
    tether_pipeline: RenderPipeline,
    cone_pipeline: RenderPipeline,
    sphere_pipeline: RenderPipeline,
    tracer_compute_pipeline: ComputePipeline,
    tracer_render_pipeline: RenderPipeline,

    // Bind group layouts.
    emergence_g0_bgl: BindGroupLayout,
    emergence_g1_bgl: BindGroupLayout,
    compute_bgl: BindGroupLayout,
    compute_g1_bgl: BindGroupLayout,
    render_bgl: BindGroupLayout,
    render_g1_bgl: BindGroupLayout,
    post_bgl: BindGroupLayout,
    causal_bgl: BindGroupLayout,
    tether_bgl: BindGroupLayout,
    observer_bgl: BindGroupLayout,
    tracer_compute_bgl: BindGroupLayout,
    tracer_render_bgl: BindGroupLayout,
    tracer_cam_bgl: BindGroupLayout,
    #[allow(dead_code)]
    empty_bgl: BindGroupLayout,
    empty_bg: BindGroup,

    // Zero-allocation ping-pong particle buffers (ParticleBuffer_Ping /
    // ParticleBuffer_Pong in the shbt6 spec).
    particle_buffers: [Buffer; 2],

    // Emergent condensation buffers.
    emergence_params_buffer: Buffer,
    grid_density_buffer: Buffer,
    tracking_state_buffer: Buffer,
    seed_candidates_buffer: Buffer,
    active_seeds_buffer: Buffer,
    prev_seeds_buffer: Buffer,

    // Stinespring tether buffers.
    tether_vertex_buffer: Buffer,
    tether_indirect_buffer: Buffer,

    // History crystallization / glitch buffers.
    events_buffer: Buffer,
    event_count_buffer: Buffer,
    condensing_buffer: Buffer,
    glitch_buffer: Buffer,

    // Causal observer buffers (cones + entropy spheres + fresnel shells).
    observer_buffer: Buffer,
    causal_buffer: Buffer,
    causal_uniform_buffer: Buffer,
    causal_observer_uniform_buffer: Buffer,

    // Tracer buffers.
    tracer_buffer: Buffer,
    tracer_params_buffer: Buffer,
    tracer_camera_buffer: Buffer,

    cosmo_buffer: Buffer,
    /// Physical metrology context: code-unit conversions, Tier-1
    /// background distances, and per-step KDK uniform assembly.
    metrology: units::MetrologyPipeline,
    camera_buffer: Buffer,
    post_buffer: Buffer,
    seed_buffer: Buffer,
    telemetry_buffer: Buffer,

    #[cfg(target_arch = "wasm32")]
    capture_target: Option<Texture>,
    #[cfg(target_arch = "wasm32")]
    capture_staging: Option<Buffer>,
    #[cfg(target_arch = "wasm32")]
    capture_post_pipeline: Option<RenderPipeline>,
    #[cfg(target_arch = "wasm32")]
    capture_post_bgl: Option<BindGroupLayout>,

    density_tex: Texture,
    force_tex: Texture,
    entropy_tex: Texture,
    boundary_tex: Texture,
    visible_tex: Texture,
    distortion_tex: Texture,
    sampler: Sampler,
    readback: SeedReadback,

    num_particles: u32,
    mean_raw_total: f64,
    timeline: TimelineController,
    frame_index: u64,
    redshift: f64,
    delta_n_bits: f64,
    seed_count: u32,
    debug_grid_stats: String,
    drain_ticks: u32,
    /// wasm-only: cumulative mapAsync rejections; above a threshold the
    /// seed telemetry channel falls back to a CPU replica of the
    /// condensation kernel (headless-Chromium mapAsync limitation).
    cpu_fallback_errors: u32,
    cpu_fallback: bool,
    #[allow(dead_code)] // read only under the wasm32 CPU fallback
    cpu_particles: Option<Vec<particle::Particle>>,
    /// (seed_id, comoving position, banked mass) for CPU tracking.
    #[allow(dead_code)] // read only under the wasm32 CPU fallback
    cpu_prev_seeds: Vec<(f32, [f32; 3], f32)>,
    emergent_seeds: Vec<EmergentSeed>,
    known_seed_ids: Vec<f32>,
    events: Vec<CrystallizationEventGpu>,
    channel_a: f32,
    channel_b: f32,
    unwrap_transition: f32,
    lensing_strength: f32,
    dispersion_coeff: f32,
    dark_glow_intensity: f32,
    dark_glow_radius: f32,
    doppler_enabled: bool,
    telemetry: VisualizerTelemetry,
    buffer_index: usize,
    width: u32,
    height: u32,
}

// Shared pipeline construction (target-independent).
impl ShbtWebGpuEngine {
    fn uniform_entry(binding: u32, visibility: ShaderStages) -> BindGroupLayoutEntry {
        BindGroupLayoutEntry {
            binding,
            visibility,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }
    }

    fn storage_entry(binding: u32, visibility: ShaderStages, read_only: bool) -> BindGroupLayoutEntry {
        BindGroupLayoutEntry {
            binding,
            visibility,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }
    }

    fn tex3d_entry(binding: u32, visibility: ShaderStages) -> BindGroupLayoutEntry {
        BindGroupLayoutEntry {
            binding,
            visibility,
            ty: BindingType::Texture {
                sample_type: TextureSampleType::Float { filterable: true },
                view_dimension: TextureViewDimension::D3,
                multisampled: false,
            },
            count: None,
        }
    }

    fn tex2d_entry(binding: u32, visibility: ShaderStages) -> BindGroupLayoutEntry {
        BindGroupLayoutEntry {
            binding,
            visibility,
            ty: BindingType::Texture {
                sample_type: TextureSampleType::Float { filterable: true },
                view_dimension: TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        }
    }

    fn sampler_entry(binding: u32, visibility: ShaderStages) -> BindGroupLayoutEntry {
        BindGroupLayoutEntry {
            binding,
            visibility,
            ty: BindingType::Sampler(SamplerBindingType::Filtering),
            count: None,
        }
    }

    fn additive_blend() -> Option<BlendState> {
        Some(BlendState {
            color: BlendComponent {
                src_factor: BlendFactor::One,
                dst_factor: BlendFactor::One,
                operation: BlendOperation::Add,
            },
            alpha: BlendComponent {
                src_factor: BlendFactor::One,
                dst_factor: BlendFactor::One,
                operation: BlendOperation::Add,
            },
        })
    }

    fn dual_target_desc() -> [Option<ColorTargetState>; 2] {
        [
            Some(ColorTargetState {
                format: TARGET_FORMAT,
                blend: Self::additive_blend(),
                write_mask: ColorWrites::ALL,
            }),
            Some(ColorTargetState {
                format: TARGET_FORMAT,
                blend: Self::additive_blend(),
                write_mask: ColorWrites::ALL,
            }),
        ]
    }

    /// Bind group layouts for the emergent condensation tri-pass kernel.
    fn build_emergence_layouts(device: &Device) -> (BindGroupLayout, BindGroupLayout) {
        let g0 = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("emergence group0 layout"),
            entries: &[
                Self::uniform_entry(0, ShaderStages::COMPUTE),
                Self::storage_entry(1, ShaderStages::COMPUTE, true),
                Self::storage_entry(2, ShaderStages::COMPUTE, false),
            ],
        });
        let g1 = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("emergence group1 layout"),
            entries: &[
                Self::storage_entry(0, ShaderStages::COMPUTE, false),
                Self::storage_entry(1, ShaderStages::COMPUTE, false),
                Self::storage_entry(2, ShaderStages::COMPUTE, true),
                Self::storage_entry(3, ShaderStages::COMPUTE, false),
            ],
        });
        (g0, g1)
    }

    fn build_emergence_pipelines(
        device: &Device,
        g0: &BindGroupLayout,
        g1: &BindGroupLayout,
    ) -> (ComputePipeline, ComputePipeline, ComputePipeline) {
        let shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("seed_emergence.wgsl"),
            source: ShaderSource::Wgsl(include_str!("shaders/seed_emergence.wgsl").into()),
        });
        let layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("emergence pipeline layout"),
            bind_group_layouts: &[g0, g1],
            push_constant_ranges: &[],
        });
        let mk = |label: &str, entry: &str| {
            device.create_compute_pipeline(&ComputePipelineDescriptor {
                label: Some(label),
                layout: Some(&layout),
                module: &shader,
                entry_point: entry,
            })
        };
        (
            mk("cic accumulate", "cs_accumulate_cic"),
            mk("condensation detect", "cs_detect_condensation"),
            mk("temporal tracking", "cs_temporal_tracking"),
        )
    }

    /// nbody_pm bind group layouts: group 0 (cosmo + particles + PM grid +
    /// emergent seeds), group 1 (Stinespring tether emission buffers).
    fn build_compute_layouts(device: &Device) -> (BindGroupLayout, BindGroupLayout) {
        let g0 = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("nbody_pm group0 layout"),
            entries: &[
                Self::uniform_entry(0, ShaderStages::COMPUTE),
                Self::storage_entry(1, ShaderStages::COMPUTE, false),
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::StorageTexture {
                        access: StorageTextureAccess::WriteOnly,
                        format: TextureFormat::R32Float,
                        view_dimension: TextureViewDimension::D3,
                    },
                    count: None,
                },
                Self::tex3d_entry(3, ShaderStages::COMPUTE),
                Self::sampler_entry(4, ShaderStages::COMPUTE),
                Self::storage_entry(5, ShaderStages::COMPUTE, true),
                Self::storage_entry(6, ShaderStages::COMPUTE, true),
            ],
        });
        let g1 = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("nbody_pm tether group1 layout"),
            entries: &[
                Self::storage_entry(0, ShaderStages::COMPUTE, false),
                Self::storage_entry(1, ShaderStages::COMPUTE, false),
            ],
        });
        (g0, g1)
    }

    /// dual_channel_render layouts: group 0 (particles + camera), group 1
    /// (crystallization events + emergent seed table).
    fn build_render_layouts(device: &Device) -> (BindGroupLayout, BindGroupLayout) {
        let g0 = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("dual_channel group0 layout"),
            entries: &[
                Self::storage_entry(0, ShaderStages::VERTEX, true),
                Self::uniform_entry(1, ShaderStages::VERTEX_FRAGMENT),
            ],
        });
        let g1 = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("dual_channel group1 layout"),
            entries: &[
                Self::storage_entry(0, ShaderStages::FRAGMENT, true),
                Self::uniform_entry(1, ShaderStages::FRAGMENT),
                Self::storage_entry(2, ShaderStages::VERTEX, true),
                Self::storage_entry(3, ShaderStages::VERTEX, true),
            ],
        });
        (g0, g1)
    }

    fn build_post_layout(device: &Device) -> BindGroupLayout {
        device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("holographic_post bind group layout"),
            entries: &[
                Self::tex2d_entry(0, ShaderStages::FRAGMENT),
                Self::tex2d_entry(1, ShaderStages::FRAGMENT),
                Self::sampler_entry(2, ShaderStages::FRAGMENT),
                Self::uniform_entry(3, ShaderStages::FRAGMENT),
                Self::storage_entry(4, ShaderStages::FRAGMENT, true),
                Self::tex2d_entry(5, ShaderStages::FRAGMENT),
                Self::storage_entry(6, ShaderStages::FRAGMENT, true),
                Self::uniform_entry(7, ShaderStages::FRAGMENT),
            ],
        })
    }

    fn build_aux_pipelines(
        device: &Device,
    ) -> (
        BindGroupLayout,
        RenderPipeline,
        BindGroupLayout,
        RenderPipeline,
        RenderPipeline,
        BindGroupLayout,
        ComputePipeline,
        BindGroupLayout,
        BindGroupLayout,
        RenderPipeline,
    ) {
        // Stinespring tether line list.
        let tether_bgl = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("tether render layout"),
            entries: &[
                Self::uniform_entry(0, ShaderStages::VERTEX),
                Self::storage_entry(1, ShaderStages::VERTEX, true),
            ],
        });
        let tether_shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("tether_render.wgsl"),
            source: ShaderSource::Wgsl(include_str!("shaders/tether_render.wgsl").into()),
        });
        let tether_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("tether pipeline layout"),
            bind_group_layouts: &[&tether_bgl],
            push_constant_ranges: &[],
        });
        let tether_pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("stinespring tether pipeline"),
            layout: Some(&tether_layout),
            vertex: VertexState {
                module: &tether_shader,
                entry_point: "vs_tether",
                buffers: &[],
            },
            fragment: Some(FragmentState {
                module: &tether_shader,
                entry_point: "fs_tether",
                targets: &Self::dual_target_desc(),
            }),
            primitive: PrimitiveState {
                topology: PrimitiveTopology::LineList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: MultisampleState::default(),
            multiview: None,
        });

        // Causal observer cone + entropy sphere passes share one layout.
        let observer_bgl = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("causal observer layout"),
            entries: &[
                Self::storage_entry(0, ShaderStages::VERTEX, true),
                Self::uniform_entry(1, ShaderStages::VERTEX_FRAGMENT),
            ],
        });
        let observer_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("observer pipeline layout"),
            bind_group_layouts: &[&observer_bgl],
            push_constant_ranges: &[],
        });

        let cone_shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("causal_cone_render.wgsl"),
            source: ShaderSource::Wgsl(
                include_str!("shaders/causal_cone_render.wgsl").into(),
            ),
        });
        let cone_pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("past light cone pipeline"),
            layout: Some(&observer_layout),
            vertex: VertexState {
                module: &cone_shader,
                entry_point: "vs_cone_wireframe",
                buffers: &[],
            },
            fragment: Some(FragmentState {
                module: &cone_shader,
                entry_point: "fs_cone_wireframe",
                targets: &Self::dual_target_desc(),
            }),
            primitive: PrimitiveState {
                topology: PrimitiveTopology::LineList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: MultisampleState::default(),
            multiview: None,
        });

        let sphere_shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("causal_sphere_render.wgsl"),
            source: ShaderSource::Wgsl(
                include_str!("shaders/causal_sphere_render.wgsl").into(),
            ),
        });
        let sphere_pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("entropy budget sphere pipeline"),
            layout: Some(&observer_layout),
            vertex: VertexState {
                module: &sphere_shader,
                entry_point: "vs_entropy_sphere",
                buffers: &[],
            },
            fragment: Some(FragmentState {
                module: &sphere_shader,
                entry_point: "fs_entropy_sphere",
                targets: &Self::dual_target_desc(),
            }),
            primitive: PrimitiveState::default(),
            depth_stencil: None,
            multisample: MultisampleState::default(),
            multiview: None,
        });

        // Entropy tracer compute + render pipelines.
        let tracer_compute_bgl = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("tracer compute layout"),
            entries: &[
                Self::storage_entry(0, ShaderStages::COMPUTE, false),
                Self::tex3d_entry(1, ShaderStages::COMPUTE),
                Self::sampler_entry(2, ShaderStages::COMPUTE),
                Self::uniform_entry(3, ShaderStages::COMPUTE),
            ],
        });
        let tracer_render_bgl = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("tracer render layout"),
            entries: &[Self::storage_entry(0, ShaderStages::VERTEX, true)],
        });
        let tracer_shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("entropy_tracer.wgsl"),
            source: ShaderSource::Wgsl(include_str!("shaders/entropy_tracer.wgsl").into()),
        });
        let tracer_compute_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("tracer compute layout"),
            bind_group_layouts: &[&tracer_compute_bgl],
            push_constant_ranges: &[],
        });
        let tracer_compute_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("entropy tracer integrate"),
            layout: Some(&tracer_compute_layout),
            module: &tracer_shader,
            entry_point: "cs_integrate_streamlines",
        });
        let tracer_cam_bgl = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("tracer camera layout"),
            entries: &[Self::uniform_entry(0, ShaderStages::VERTEX_FRAGMENT)],
        });
        let tracer_render_shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("tracer_render.wgsl"),
            source: ShaderSource::Wgsl(include_str!("shaders/tracer_render.wgsl").into()),
        });
        let tracer_render_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("tracer render layout"),
            bind_group_layouts: &[&tracer_render_bgl, &tracer_cam_bgl],
            push_constant_ranges: &[],
        });
        let tracer_render_pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("entropy streamline pipeline"),
            layout: Some(&tracer_render_layout),
            vertex: VertexState {
                module: &tracer_render_shader,
                entry_point: "vs_tracer",
                buffers: &[],
            },
            fragment: Some(FragmentState {
                module: &tracer_render_shader,
                entry_point: "fs_tracer",
                targets: &Self::dual_target_desc(),
            }),
            primitive: PrimitiveState::default(),
            depth_stencil: None,
            multisample: MultisampleState::default(),
            multiview: None,
        });

        (
            tether_bgl,
            tether_pipeline,
            observer_bgl,
            cone_pipeline,
            sphere_pipeline,
            tracer_compute_bgl,
            tracer_compute_pipeline,
            tracer_render_bgl,
            tracer_cam_bgl,
            tracer_render_pipeline,
        )
    }

    /// Causal-observer Fresnel shell pass (shbt5): instanced billboards
    /// that inject synthetic ripple shear/convergence rings into Channel B.
    /// Entry points live on @group(1) inside causal_point_get.wgsl, so the
    /// pipeline layout uses an empty bind-group-0 layout.
    fn build_causal_pipeline(
        device: &Device,
    ) -> (RenderPipeline, BindGroupLayout, BindGroupLayout) {
        let empty_bgl = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("empty group0 layout"),
            entries: &[],
        });
        let causal_bgl = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("causal render bind group layout"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::VERTEX_FRAGMENT,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::VERTEX,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let causal_shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("causal_point_get.wgsl (render stage)"),
            source: ShaderSource::Wgsl(include_str!("shaders/causal_point_get.wgsl").into()),
        });
        let layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("causal pipeline layout"),
            bind_group_layouts: &[&empty_bgl, &causal_bgl],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("causal fresnel render pipeline"),
            layout: Some(&layout),
            vertex: VertexState {
                module: &causal_shader,
                entry_point: "vs_causal",
                buffers: &[],
            },
            fragment: Some(FragmentState {
                module: &causal_shader,
                entry_point: "fs_causal",
                targets: &Self::dual_target_desc(),
            }),
            primitive: PrimitiveState::default(),
            depth_stencil: None,
            multisample: MultisampleState::default(),
            multiview: None,
        });
        (pipeline, empty_bgl, causal_bgl)
    }

    /// Allocate the 3D force grid texture and fill it with the primordial
    /// PM potential field (low-frequency displacement ripples only — the
    /// emergent condensation kernel owns all seed gravity now).
    fn make_force_grid(device: &Device, queue: &Queue) -> Texture {
        let size = Extent3d {
            width: GRID_DIM,
            height: GRID_DIM,
            depth_or_array_layers: GRID_DIM,
        };
        let texture = device.create_texture(&TextureDescriptor {
            label: Some("force_grid"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D3,
            format: TextureFormat::Rgba16Float,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
            view_formats: &[],
        });
        // Pure PM background field: a smooth, low-amplitude curl-free
        // potential with zero embedded attractors.
        let mut voxels = vec![[0f32; 4]; (GRID_DIM * GRID_DIM * GRID_DIM) as usize];
        for z in 0..GRID_DIM {
            for y in 0..GRID_DIM {
                for x in 0..GRID_DIM {
                    let p = [
                        x as f32 / GRID_DIM as f32,
                        y as f32 / GRID_DIM as f32,
                        z as f32 / GRID_DIM as f32,
                    ];
                    let idx = (x + y * GRID_DIM + z * GRID_DIM * GRID_DIM) as usize;
                    voxels[idx] = [
                        (p[0] * 6.2831).sin() * (p[1] * 12.566).cos() * 0.02,
                        (p[1] * 6.2831).sin() * (p[2] * 12.566).cos() * 0.02,
                        (p[2] * 6.2831).sin() * (p[0] * 12.566).cos() * 0.02,
                        // Alpha channel: projected mass density rho_proj
                        // for the GET entropic force
                        // F_GET = -kappa_GET * grad ln rho_proj (shbt8
                        // Phase 1). Smooth positive proxy consistent
                        // with the static PM potential; dynamic
                        // condensation density lives in grid_density.
                        1.0 + 0.25 * ((p[0] * 6.2831).cos() * (p[1] * 6.2831).cos()
                            + (p[1] * 6.2831).cos() * (p[2] * 6.2831).cos()
                            + (p[2] * 6.2831).cos() * (p[0] * 6.2831).cos()) / 3.0,
                    ];
                }
            }
        }
        let mut bytes = Vec::with_capacity(voxels.len() * 8);
        for v in &voxels {
            for c in v {
                bytes.extend_from_slice(&f32_to_f16(*c).to_le_bytes());
            }
        }
        queue.write_texture(
            ImageCopyTexture {
                texture: &texture,
                mip_level: 0,
                origin: Origin3d::ZERO,
                aspect: TextureAspect::All,
            },
            &bytes,
            ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(GRID_DIM * 8),
                rows_per_image: Some(GRID_DIM),
            },
            size,
        );
        texture
    }

    /// 3D entropic density texture sampled by the tracer advection kernel
    /// (`entropy_field_3d`): a smooth rho_E field with low-frequency basins.
    fn make_entropy_field(device: &Device, queue: &Queue) -> Texture {
        let size = Extent3d {
            width: GRID_DIM,
            height: GRID_DIM,
            depth_or_array_layers: GRID_DIM,
        };
        let texture = device.create_texture(&TextureDescriptor {
            label: Some("entropy_field_3d"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D3,
            format: TextureFormat::Rgba16Float,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let mut voxels = vec![[0f32; 4]; (GRID_DIM * GRID_DIM * GRID_DIM) as usize];
        for z in 0..GRID_DIM {
            for y in 0..GRID_DIM {
                for x in 0..GRID_DIM {
                    let p = [
                        x as f32 / GRID_DIM as f32,
                        y as f32 / GRID_DIM as f32,
                        z as f32 / GRID_DIM as f32,
                    ];
                    let idx = (x + y * GRID_DIM + z * GRID_DIM * GRID_DIM) as usize;
                    let e = 0.4
                        + 0.25 * (p[0] * 6.2831).sin() * (p[1] * 6.2831).cos()
                        + 0.25 * (p[1] * 12.566).sin() * (p[2] * 6.2831).cos()
                        + 0.1 * (p[2] * 18.849).sin();
                    voxels[idx] = [e.max(0.01), 0.0, 0.0, 0.0];
                }
            }
        }
        let mut bytes = Vec::with_capacity(voxels.len() * 8);
        for v in &voxels {
            for c in v {
                bytes.extend_from_slice(&f32_to_f16(*c).to_le_bytes());
            }
        }
        queue.write_texture(
            ImageCopyTexture {
                texture: &texture,
                mip_level: 0,
                origin: Origin3d::ZERO,
                aspect: TextureAspect::All,
            },
            &bytes,
            ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(GRID_DIM * 8),
                rows_per_image: Some(GRID_DIM),
            },
            size,
        );
        texture
    }

    /// 256x256 conformal boundary register texture (Enhancement 1). R =
    /// rho_B loading density, G = rho_E entanglement entropy density.
    fn make_boundary_tex(device: &Device) -> Texture {
        device.create_texture(&TextureDescriptor {
            label: Some("boundary_register_tex"),
            size: Extent3d {
                width: BOUNDARY_RES,
                height: BOUNDARY_RES,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba8Unorm,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
            view_formats: &[],
        })
    }

    /// Regenerate the boundary register field: rho_B tracks the loaded
    /// screen fraction and receives Gaussian congestion bumps at the
    /// emergent seed centroids; rho_E is the complementary entropy field.
    fn upload_boundary_tex(&mut self) {
        let mut pixels = vec![0u8; (BOUNDARY_RES * BOUNDARY_RES * 4) as usize];
        let f_load = telemetry::loading_fraction(self.redshift) as f32;
        for y in 0..BOUNDARY_RES {
            for x in 0..BOUNDARY_RES {
                let u = x as f32 / BOUNDARY_RES as f32;
                let v = y as f32 / BOUNDARY_RES as f32;
                let mut rho_b = f_load * (0.55 + 0.45 * (u * 6.2831).sin() * (v * 6.2831).sin());
                for s in &self.emergent_seeds {
                    let su = s.pos[0] / BOX_SIZE;
                    let sv = s.pos[1] / BOX_SIZE;
                    let du = (u - su).abs().min(1.0 - (u - su).abs());
                    let dv = (v - sv).abs().min(1.0 - (v - sv).abs());
                    rho_b += 0.5 * (-(du * du + dv * dv) * 900.0).exp();
                }
                let rho_e = (1.0 - f_load) * (0.4 + 0.3 * (v * 12.566).cos())
                    + 0.1 * ((x * 31 + y * 17) % 7) as f32 / 7.0;
                let idx = ((x + y * BOUNDARY_RES) * 4) as usize;
                pixels[idx] = (rho_b.clamp(0.0, 1.0) * 255.0) as u8;
                pixels[idx + 1] = (rho_e.clamp(0.0, 1.0) * 255.0) as u8;
            }
        }
        self.queue.write_texture(
            ImageCopyTexture {
                texture: &self.boundary_tex,
                mip_level: 0,
                origin: Origin3d::ZERO,
                aspect: TextureAspect::All,
            },
            &pixels,
            ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(BOUNDARY_RES * 4),
                rows_per_image: Some(BOUNDARY_RES),
            },
            Extent3d {
                width: BOUNDARY_RES,
                height: BOUNDARY_RES,
                depth_or_array_layers: 1,
            },
        );
    }

    /// WZW-locked loading and capacity evaluation (shbt6 Section 2):
    ///   f_load(z) = 0.697 (1+z)^(-4/13)         [beta_load = K/(k_l K)]
    ///   N_limit = N_sat f_load gamma_geom / N_cells   (per-voxel ceiling)
    ///   K_bit   = N_sat f_load / raw_mass_total     (demand coupling)
    ///   delta_eff = delta * 31 / (1+z)             (linear growth)
    /// Boundary loading fraction f_load(z) and the instanton nucleation
    /// attempt rate A_0. The empirical sigmoid growth envelope is gone
    /// (shbt7): the Cardy ceiling cardy_limit_norm(z) in the shader sets
    /// the onset epoch first-principally through H(z)^5 scaling.
    fn update_redshift_and_loading(&self, z: f64) -> (f64, f64) {
        let f_load = telemetry::loading_fraction(z);
        (f_load, ATTEMPT_FREQ)
    }

    /// Cardy boundary capacity ceiling in normalized units (mirrors
    /// seed_emergence.wgsl): N_limit ~ gamma_CFT * H(z)^5, evaluated as
    /// gamma_CFT * ((1+z)/(1+z_ref))^7.5 in the matter-era H ~ (1+z)^1.5.
    #[allow(dead_code)]
    fn cardy_limit_norm(z: f64) -> f64 {
        GAMMA_CFT * ((1.0 + z).max(1.0e-3) / (1.0 + Z_REF)).powf(7.5)
    }

    /// Euclidean instanton action on the normalized density ratio
    /// R = N_local/N_limit: (2 pi c_eff/k_q) (1-R)^2 for R<1, else 0.
    #[allow(dead_code)]
    fn instanton_action(r_ratio: f64) -> f64 {
        if r_ratio >= 1.0 {
            return 0.0;
        }
        let d = 1.0 - r_ratio;
        S_INST_PREFACTOR * d * d
    }

    fn new_common(device: Device, queue: Queue, num_particles: u32) -> Self {
        let particles = Particle::seed_lattice(num_particles as usize, BOX_SIZE);
        let mean_raw_total = particles.iter().map(|p| p.grav_mass as f64).sum::<f64>();
        let particle_buffers = [
            device.create_buffer_init(&BufferInitDescriptor {
                label: Some("ParticleBuffer_Ping"),
                contents: bytemuck::cast_slice(&particles),
                usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            }),
            device.create_buffer_init(&BufferInitDescriptor {
                label: Some("ParticleBuffer_Pong"),
                contents: bytemuck::cast_slice(&particles),
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            }),
        ];
        let cosmo_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("cosmo uniform"),
            size: std::mem::size_of::<CosmoParams>() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let emergence_params_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("emergence SimulationParameters"),
            size: std::mem::size_of::<EmergenceParams>() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let camera_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("camera uniform"),
            size: std::mem::size_of::<CameraParams>() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let post_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("lensing uniforms buffer (224B)"),
            size: std::mem::size_of::<LensingUniforms>() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let seed_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("seed defect table (64 x 16B)"),
            size: (MAX_SEEDS * std::mem::size_of::<SeedDefect>()) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let condensing_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("condensing seed glitch table (64 x 32B)"),
            size: (MAX_SEEDS * std::mem::size_of::<CondensingSeedGpu>()) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let glitch_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("glitch uniforms"),
            size: std::mem::size_of::<GlitchUniforms>() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let events_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("crystallization event ring (64 x 32B)"),
            size: (MAX_SEEDS * std::mem::size_of::<CrystallizationEventGpu>()) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let event_count_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("crystallization event count"),
            size: 16,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let observer_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("causal observer table (64 x 48B)"),
            size: (MAX_SEEDS * std::mem::size_of::<CausalObserver>()) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let causal_observer_uniform_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("causal observer uniforms"),
            size: std::mem::size_of::<TracerCamera>() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Emergent condensation buffers (zero-allocation: allocated once).
        let n_cells = (GRID_DIM * GRID_DIM * GRID_DIM) as u64;
        let grid_density_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("emergence grid_density"),
            size: n_cells * 4,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let tracking_state_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("emergence tracking_state / seed_state"),
            size: 16,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let seed_candidates_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("emergence seed_candidates"),
            size: (MAX_SEEDS * 32) as u64,
            usage: BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let active_seeds_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("emergence active_seeds"),
            size: (MAX_SEEDS * 32) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let prev_seeds_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("emergence prev_seeds"),
            size: (MAX_SEEDS * 32) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Stinespring tether vertex buffer + indirect draw args.
        let tether_vertex_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("TetherVertexBuffer"),
            size: (TETHER_CAP as u64) * 32,
            usage: BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let tether_indirect_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("TetherIndirectArgs"),
            contents: bytemuck::cast_slice(&[0u32, 1u32, 0u32, 0u32]),
            usage: BufferUsages::STORAGE | BufferUsages::INDIRECT | BufferUsages::COPY_DST,
        });

        // Entropy tracer particles.
        let tracer_seed: Vec<TracerGpu> = (0..TRACER_COUNT)
            .map(|i| {
                let s = i as f32 * 1.61803398875;
                TracerGpu {
                    pos: [(s * 2.1).sin() * 0.9, (s * 3.7).cos() * 0.9, (s * 5.3).sin() * 0.9],
                    lifetime: 0.0,
                    vel: [0.0; 3],
                    entropy_val: 0.0,
                    pad: [0.0; 4],
                }
            })
            .collect();
        let tracer_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("entropy tracer buffer"),
            contents: bytemuck::cast_slice(&tracer_seed),
            usage: BufferUsages::STORAGE,
        });
        let tracer_params_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("tracer params"),
            size: std::mem::size_of::<TracerParams>() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let tracer_camera_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("tracer camera"),
            size: std::mem::size_of::<TracerCamera>() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Causal-observer Fresnel shell nodes (procedural, seeded off the
        // emergent centroid slot ring — no fixed attractor positions).
        let causal_nodes: Vec<CausalRenderNode> = (0..CAUSAL_NODE_COUNT)
            .map(|i| {
                let a = i as f32 * 0.7854;
                CausalRenderNode {
                    center: [0.5 + 0.3 * a.cos(), 0.5 + 0.3 * a.sin(), 0.5],
                    radius: 0.12,
                    entropy_budget: 1.0,
                    get_cost: 0.0,
                    collapse_phase: 0.0,
                    active_flag: 1,
                    seed_index: i as u32,
                    pad: [0; 3],
                    projection_dir: [0.0, 1.0, 0.0],
                    pad2: 0,
                }
            })
            .collect();
        let causal_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("causal render node table"),
            contents: bytemuck::cast_slice(&causal_nodes),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        });
        let causal_uniform_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("causal render uniforms"),
            size: std::mem::size_of::<CausalRenderUniforms>() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let telemetry_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("SHBT-MMIO telemetry buffer"),
            size: 128,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let density_tex = device.create_texture(&TextureDescriptor {
            label: Some("density_grid"),
            size: Extent3d {
                width: GRID_DIM,
                height: GRID_DIM,
                depth_or_array_layers: GRID_DIM,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D3,
            format: TextureFormat::R32Float,
            usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let force_tex = Self::make_force_grid(&device, &queue);
        let entropy_tex = Self::make_entropy_field(&device, &queue);
        let boundary_tex = Self::make_boundary_tex(&device);
        let mk_target = |device: &Device, label: &str| {
            device.create_texture(&TextureDescriptor {
                label: Some(label),
                size: Extent3d {
                    width: 1280,
                    height: 720,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: TARGET_FORMAT,
                usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            })
        };
        let sampler = device.create_sampler(&SamplerDescriptor {
            label: Some("default sampler"),
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            ..Default::default()
        });

        let empty_bgl = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("empty layout"),
            entries: &[],
        });
        let empty_bg = device.create_bind_group(&BindGroupDescriptor {
            label: Some("empty bind group"),
            layout: &empty_bgl,
            entries: &[],
        });

        let (emergence_g0_bgl, emergence_g1_bgl) = Self::build_emergence_layouts(&device);
        let (emergence_cic_pipeline, emergence_detect_pipeline, emergence_track_pipeline) =
            Self::build_emergence_pipelines(&device, &emergence_g0_bgl, &emergence_g1_bgl);

        let (compute_bgl, compute_g1_bgl) = Self::build_compute_layouts(&device);
        let compute_shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("nbody_pm.wgsl"),
            source: ShaderSource::Wgsl(include_str!("shaders/nbody_pm.wgsl").into()),
        });
        let compute_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("compute pipeline layout"),
            bind_group_layouts: &[&compute_bgl, &compute_g1_bgl],
            push_constant_ranges: &[],
        });
        let compute_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("nbody_pm compute pipeline"),
            layout: Some(&compute_pipeline_layout),
            module: &compute_shader,
            entry_point: "cs_advance_particles",
        });

        let (render_bgl, render_g1_bgl) = Self::build_render_layouts(&device);
        let render_shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("dual_channel_render.wgsl"),
            source: ShaderSource::Wgsl(
                include_str!("shaders/dual_channel_render.wgsl").into(),
            ),
        });
        let render_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("render pipeline layout"),
            bind_group_layouts: &[&render_bgl, &render_g1_bgl],
            push_constant_ranges: &[],
        });
        let render_pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("dual channel render pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: VertexState {
                module: &render_shader,
                entry_point: "vs_particle_billboard",
                buffers: &[],
            },
            fragment: Some(FragmentState {
                module: &render_shader,
                entry_point: "fs_render_dual_channel",
                targets: &Self::dual_target_desc(),
            }),
            primitive: PrimitiveState::default(),
            depth_stencil: None,
            multisample: MultisampleState::default(),
            multiview: None,
        });

        let post_bgl = Self::build_post_layout(&device);
        let post_pipeline = Self::build_post_pipeline(&device, &post_bgl, TARGET_FORMAT);
        let (causal_pipeline, _e, causal_bgl) = Self::build_causal_pipeline(&device);
        let (
            tether_bgl,
            tether_pipeline,
            observer_bgl,
            cone_pipeline,
            sphere_pipeline,
            tracer_compute_bgl,
            tracer_compute_pipeline,
            tracer_render_bgl,
            tracer_cam_bgl,
            tracer_render_pipeline,
        ) = Self::build_aux_pipelines(&device);

        let visible_tex = mk_target(&device, "visible_tex");
        let distortion_tex = mk_target(&device, "distortion_tex");

        let readback = SeedReadback {
            staging: device.create_buffer(&BufferDescriptor {
                label: Some("seed readback staging"),
                size: 16 + (MAX_SEEDS as u64) * 32 + (GRID_DIM * GRID_DIM * GRID_DIM) as u64 * 4,
                usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            slot: Rc::new(RefCell::new(None)),
            mapped: Arc::new(AtomicBool::new(false)),
            map_calls: Arc::new(AtomicU32::new(0)),
            map_errors: Arc::new(AtomicU32::new(0)),
            map_err_text: Arc::new(std::sync::Mutex::new(String::new())),
            in_flight: false,
            map_started: false,
        };

        Self {
            device,
            queue,
            surface: None,
            surface_config: None,
            emergence_cic_pipeline,
            emergence_detect_pipeline,
            emergence_track_pipeline,
            compute_pipeline,
            render_pipeline,
            post_pipeline,
            causal_pipeline,
            tether_pipeline,
            cone_pipeline,
            sphere_pipeline,
            tracer_compute_pipeline,
            tracer_render_pipeline,
            emergence_g0_bgl,
            emergence_g1_bgl,
            compute_bgl,
            compute_g1_bgl,
            render_bgl,
            render_g1_bgl,
            post_bgl,
            causal_bgl,
            tether_bgl,
            observer_bgl,
            tracer_compute_bgl,
            tracer_render_bgl,
            tracer_cam_bgl,
            empty_bgl,
            empty_bg,
            particle_buffers,
            emergence_params_buffer,
            grid_density_buffer,
            tracking_state_buffer,
            seed_candidates_buffer,
            active_seeds_buffer,
            prev_seeds_buffer,
            tether_vertex_buffer,
            tether_indirect_buffer,
            events_buffer,
            event_count_buffer,
            condensing_buffer,
            glitch_buffer,
            observer_buffer,
            causal_buffer,
            causal_uniform_buffer,
            causal_observer_uniform_buffer,
            tracer_buffer,
            tracer_params_buffer,
            tracer_camera_buffer,
            cosmo_buffer,
            metrology: units::MetrologyPipeline::new(
                units::CosmologicalContext::canonical(BOX_SIZE as f64),
            ),
            camera_buffer,
            post_buffer,
            seed_buffer,
            telemetry_buffer,
            #[cfg(target_arch = "wasm32")]
            capture_target: None,
            #[cfg(target_arch = "wasm32")]
            capture_staging: None,
            #[cfg(target_arch = "wasm32")]
            capture_post_pipeline: None,
            #[cfg(target_arch = "wasm32")]
            capture_post_bgl: None,
            density_tex,
            force_tex,
            entropy_tex,
            boundary_tex,
            visible_tex,
            distortion_tex,
            sampler,
            readback,
            num_particles,
            mean_raw_total,
            timeline: TimelineController::default(),
            frame_index: 0,
            redshift: 1.0e12,
            delta_n_bits: 0.0,
            seed_count: 0,
            debug_grid_stats: String::new(),
            drain_ticks: 0,
            cpu_fallback_errors: 0,
            // On wasm targets GPU->CPU mapAsync is unreliable in several
            // environments (headless SwiftShader destroys the external
            // Instance — and the device with it — after only a few maps).
            // Run the CPU telemetry replica from the start so the device
            // is never destabilized; native targets keep GPU readback.
            cpu_fallback: cfg!(target_arch = "wasm32"),
            cpu_particles: None,
            cpu_prev_seeds: Vec::new(),
            emergent_seeds: Vec::new(),
            known_seed_ids: Vec::new(),
            events: Vec::new(),
            channel_a: 1.0,
            channel_b: 1.0,
            unwrap_transition: 0.0,
            lensing_strength: 1.0,
            dispersion_coeff: 0.25,
            dark_glow_intensity: 0.8,
            dark_glow_radius: 4.0,
            doppler_enabled: true,
            telemetry: VisualizerTelemetry::default(),
            buffer_index: 0,
            width: 1280,
            height: 720,
        }
    }

    fn build_post_pipeline(
        device: &Device,
        post_bgl: &BindGroupLayout,
        surface_format: TextureFormat,
    ) -> RenderPipeline {
        let post_shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("holographic_post.wgsl"),
            source: ShaderSource::Wgsl(include_str!("shaders/holographic_post.wgsl").into()),
        });
        let post_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("post pipeline layout"),
            bind_group_layouts: &[post_bgl],
            push_constant_ranges: &[],
        });
        device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("holographic post pipeline"),
            layout: Some(&post_pipeline_layout),
            vertex: VertexState {
                module: &post_shader,
                entry_point: "vs_post",
                buffers: &[],
            },
            fragment: Some(FragmentState {
                module: &post_shader,
                entry_point: "fs_post",
                targets: &[Some(ColorTargetState {
                    format: surface_format,
                    blend: None,
                    write_mask: ColorWrites::ALL,
                })],
            }),
            primitive: PrimitiveState::default(),
            depth_stencil: None,
            multisample: MultisampleState::default(),
            multiview: None,
        })
    }

    /// Orbiting camera direction on the observation sphere, slerped
    /// toward the boundary-plane normal as unwrap_transition -> 1 so the
    /// bulk->CFT projection swap is a smooth arc (shbt7 Phase 2).
    fn camera_dir(&self) -> [f32; 3] {
        let (sy, cy) = (self.frame_index as f32 * 0.0004).sin_cos();
        let orbit = normalize([sy, 0.1786, -cy]);
        let cft_axis = [0.0, 0.0, 1.0];
        let t = self.unwrap_transition.clamp(0.0, 1.0);
        let t = t * t * (3.0 - 2.0 * t); // smoothstep ease
        slerp3(orbit, cft_axis, t)
    }

    /// World-space camera eye for the current orbiting view.
    fn camera_eye(&self) -> [f32; 3] {
        let dist = BOX_SIZE * 1.4;
        let dir = self.camera_dir();
        [dir[0] * dist, dir[1] * dist, dir[2] * dist]
    }

    /// Project the emergent seed centroids into normalized screen UVs and
    /// stage the 64-entry SeedDefect micro-lensing table for fs_post plus
    /// the CondensationSeed glitch table (Enhancement 11). The Einstein
    /// radius theta_E,k = sqrt(4 G M_k / c^2 * D_ds / (D_d D_s)) is
    /// evaluated per defect from its condensed mass.
    fn update_seed_table(&mut self, view_proj: &[[f32; 4]; 4]) -> u32 {
        let mut seeds = [SeedDefect {
            screen_pos: [0.0; 2],
            theta_e: 0.0,
            core_radius: 0.01,
        }; MAX_SEEDS];
        let mut condensing = [CondensingSeedGpu::default(); MAX_SEEDS];
        let mut active = 0u32;

        for (i, s) in self.emergent_seeds.iter().enumerate().take(MAX_SEEDS) {
            let world = [
                s.pos[0] - BOX_SIZE * 0.5,
                s.pos[1] - BOX_SIZE * 0.5,
                s.pos[2] - BOX_SIZE * 0.5,
                1.0,
            ];
            let mut clip = [0.0f32; 4];
            for r in 0..4 {
                clip[r] = (0..4).map(|c| view_proj[c][r] * world[c]).sum();
            }
            let w = clip[3].max(1.0e-4);
            let screen = [
                (clip[0] / w) * 0.5 + 0.5,
                0.5 - (clip[1] / w) * 0.5,
            ];
            // Distance-duality thin-screen metrology (shbt8 / Thm 9.12):
            //   theta_E,k = sqrt(4 G M_k / c^2 * D_ds / (D_d D_s))
            // with D_d, D_s, D_ds evaluated on the Tier-1 H_SHBT
            // background (units::CosmologicalContext). The source plane
            // sits at z_s = z_d + max(1.0, 0.5*z_d) — the conformal
            // boundary proxy for the lensed screen population — and the
            // screen-space radius is theta_E^screen = theta_E / Theta_FoV.
            let z_d = self.redshift.max(0.05);
            let z_s = z_d + (1.0_f64).max(0.5 * z_d);
            let ctx = &self.metrology.ctx;
            let d_d = units::CosmologicalContext::angular_diameter_distance_mpc(z_d);
            let d_s = units::CosmologicalContext::angular_diameter_distance_mpc(z_s);
            let d_ds = units::CosmologicalContext::lens_source_distance_mpc(z_d, z_s);
            let theta_fov = 45.0f64.to_radians();
            let theta_e_rad = ctx
                .compute_einstein_radius_rad(s.mass_msun as f64, d_d, d_s, d_ds);
            // Screen radius in normalized UV; floor at a sub-pixel step
            // so the kernel never divides by a vanishing core.
            let theta_e = ((theta_e_rad as f64 / theta_fov) as f32)
                .clamp(0.001, 0.09);
            seeds[i] = SeedDefect {
                screen_pos: screen,
                theta_e,
                core_radius: 0.004 + theta_e * 0.18,
            };
            condensing[i] = CondensingSeedGpu {
                screen_pos: screen,
                saturation: 1.0,
                lifetime: self.frame_index as f32 / 60.0,
                pad: [0.0; 4],
            };
            active += 1;
        }

        self.queue
            .write_buffer(&self.seed_buffer, 0, bytemuck::cast_slice(&seeds));
        self.queue
            .write_buffer(&self.condensing_buffer, 0, bytemuck::cast_slice(&condensing));
        self.queue.write_buffer(
            &self.glitch_buffer,
            0,
            bytemuck::bytes_of(&GlitchUniforms {
                count: active,
                _pad: [0; 7],
            }),
        );
        active
    }

    /// Rebuild the causal observer table (Enhancements 7 & 8): one observer
    /// per emergent seed centroid, entropy budget depleting as the defect
    /// mass saturates its register cell.
    fn update_observer_table(&mut self, view_proj: &[[f32; 4]; 4]) {
        let mut observers = [CausalObserver::default(); MAX_SEEDS];
        for (i, s) in self.emergent_seeds.iter().enumerate().take(MAX_SEEDS) {
            let norm = [
                s.pos[0] / BOX_SIZE,
                s.pos[1] / BOX_SIZE,
                s.pos[2] / BOX_SIZE,
            ];
            let dir = [
                norm[0] - 0.5,
                norm[1] - 0.5,
                (norm[2] - 0.5).abs().max(0.05),
            ];
            let l = (dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2])
                .sqrt()
                .max(1e-4);
            // Entropy budget depletes as the defect saturates N_limit
            // (budget in units of the per-cell capacity ceiling).
            let norm_mass = s.mass_msun / SEED_MASS_NORM;
            let budget = (2.0 - norm_mass).clamp(0.0, 2.0);
            observers[i] = CausalObserver {
                position: norm,
                radius: 0.12,
                entropy_budget: budget.max(1.0),
                active_flag: 1,
                cone_direction: [dir[0] / l, dir[1] / l, dir[2] / l],
                cone_angle: 0.6,
                pad: [0.0; 2],
            };
        }
        self.queue
            .write_buffer(&self.observer_buffer, 0, bytemuck::cast_slice(&observers));
        self.queue.write_buffer(
            &self.causal_observer_uniform_buffer,
            0,
            bytemuck::bytes_of(&TracerCamera {
                view_proj: *view_proj,
                params: [
                    BOX_SIZE,
                    1.0,
                    1.0,
                    self.frame_index as f32 / 60.0,
                ],
            }),
        );
    }

    /// Consume any completed GPU seed readback and refresh the cached
    /// emergent seed list + telemetry counters.
    fn drain_seed_readback(&mut self) {
        self.drain_ticks += 1;
        let data = self.readback.slot.borrow_mut().take();
        if let Some(bytes) = data {
            let cand = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
            let gbase = 16 + MAX_SEEDS * 32;
            let mut gmax = 0u32;
            let mut gsum = 0u64;
            let mut gnz = 0u32;
            for i in 0..(GRID_DIM * GRID_DIM * GRID_DIM) as usize {
                let v = u32::from_le_bytes([
                    bytes[gbase + i * 4],
                    bytes[gbase + i * 4 + 1],
                    bytes[gbase + i * 4 + 2],
                    bytes[gbase + i * 4 + 3],
                ]);
                gmax = gmax.max(v);
                gsum += v as u64;
                if v > 0 { gnz += 1; }
            }
            self.debug_grid_stats = format!(
                "cand={cand} grid_mean={:.3} grid_max={} nonzero={gnz}",
                gsum as f64 / 1024.0 / 32768.0 * 1024.0 / 1024.0 * 1024.0,
                gmax as f64 / 1024.0,
            );
            // normalize: values are fixed-point x1024
            self.debug_grid_stats = format!(
                "cand={cand} grid_mean={:.3} grid_max={:.3} nonzero={gnz} len={}",
                gsum as f64 / 1024.0 / 32768.0,
                gmax as f64 / 1024.0,
                bytes.len(),
            );
            let count = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]) as usize;
            let n = count.min(MAX_SEEDS);
            let mut seeds = Vec::with_capacity(n);
            for i in 0..n {
                let base = 16 + i * 32;
                let f = |o: usize| {
                    f32::from_le_bytes([
                        bytes[base + o],
                        bytes[base + o + 1],
                        bytes[base + o + 2],
                        bytes[base + o + 3],
                    ])
                };
                seeds.push(EmergentSeed {
                    pos: [f(0), f(4), f(8)],
                    mass_msun: f(16),
                    m_dot: f(20),
                    p_debt_gw: f(24),
                    seed_id: f(28),
                });
            }
            self.absorb_seed_records(seeds);
        }
        #[cfg(target_arch = "wasm32")]
        if self.cpu_fallback {
            self.cpu_emergence_tick();
        }
    }

    /// Apply a decoded batch of emergent seed records to the cached
    /// telemetry state (crystallization events, totals, counters).
    fn absorb_seed_records(&mut self, seeds: Vec<EmergentSeed>) {
        let mut total_mass = 0.0f64;
        let mut total_debt = 0.0f64;
        self.emergent_seeds.clear();
        for seed in seeds {
            if seed.mass_msun > 0.0 {
                // History crystallization event (Enhancement 9): a new
                // seed id entering the active table emits a GET flash.
                if !self.known_seed_ids.iter().any(|&k| k == seed.seed_id) {
                    self.known_seed_ids.push(seed.seed_id);
                    if self.events.len() < MAX_SEEDS {
                        self.events.push(CrystallizationEventGpu {
                            origin: seed.pos,
                            start_time: self.frame_index as f32 / 60.0,
                            intensity: (seed.mass_msun.max(1.0).log10() / 14.0).clamp(0.2, 1.0),
                            pad: [0.0; 3],
                        });
                    }
                }
                total_mass += seed.mass_msun as f64;
                total_debt += seed.p_debt_gw as f64;
                self.emergent_seeds.push(seed);
            }
        }
        if self.events.len() > MAX_SEEDS {
            self.events.drain(0..self.events.len() - MAX_SEEDS);
        }
        self.seed_count = self.emergent_seeds.len() as u32;
        self.delta_n_bits = total_mass / ALPHA_SEED_SCALED * 1.0e30;
        self.queue.write_buffer(
            &self.events_buffer,
            0,
            bytemuck::cast_slice(&self.events),
        );
        self.queue.write_buffer(
            &self.event_count_buffer,
            0,
            bytemuck::bytes_of(&(self.events.len() as u32)),
        );
        let _ = total_debt;
    }

    /// CPU replica of the condensation tri-pass (wasm fallback channel for
    /// environments where GPU->CPU `mapAsync` readback is unavailable, e.g.
    /// headless Chromium/SwiftShader). Mirrors `seed_emergence.wgsl`:
    /// trilinear CIC deposit, normalized entropy demand, 26-NMS + 3x3x3
    /// basin integration, and minimum-image tracking with accretion
    /// carry-over. The GPU pipeline still runs; this only feeds the HUD
    /// telemetry registers when the hardware readback path is dead.
    #[cfg(target_arch = "wasm32")]
    fn cpu_emergence_tick(&mut self) {
        if self.cpu_particles.is_none() {
            self.cpu_particles = Some(particle::Particle::seed_lattice(
                self.num_particles as usize,
                BOX_SIZE,
            ));
        }
        let particles = self.cpu_particles.as_ref().unwrap();
        let dim = GRID_DIM as i32;
        let n_cells = (dim * dim * dim) as usize;
        let inv_cell = GRID_DIM as f32 / BOX_SIZE;
        let mut grid = vec![0f32; n_cells];
        for p in particles {
            let gx = p.position[0] * inv_cell;
            let gy = p.position[1] * inv_cell;
            let gz = p.position[2] * inv_cell;
            let bx = gx.floor() as i32;
            let by = gy.floor() as i32;
            let bz = gz.floor() as i32;
            let fx = gx - bx as f32;
            let fy = gy - by as f32;
            let fz = gz - bz as f32;
            for dz in 0..2 {
                let wz = if dz == 1 { fz } else { 1.0 - fz };
                let zc = ((bz + dz + dim) % dim) as usize;
                for dy in 0..2 {
                    let wy = if dy == 1 { fy } else { 1.0 - fy };
                    let yc = ((by + dy + dim) % dim) as usize;
                    for dx in 0..2 {
                        let wx = if dx == 1 { fx } else { 1.0 - fx };
                        let xc = ((bx + dx + dim) % dim) as usize;
                        grid[zc * 1024 + yc * 32 + xc] += p.grav_mass * wx * wy * wz;
                    }
                }
            }
        }
        let (_, attempt_freq) = self.update_redshift_and_loading(self.redshift);
        let n_limit = Self::cardy_limit_norm(self.redshift) as f32;
        let mean_raw = (self.mean_raw_total / n_cells as f64) as f32;
        let ratio = |raw: f32| -> f32 {
            (raw / mean_raw.max(1e-5)) / n_limit
        };
        let pcg = |input: u32| -> u32 {
            let state = input.wrapping_mul(747796405).wrapping_add(2891336453);
            let word = ((state >> ((state >> 28) + 4)) ^ state).wrapping_mul(277803737);
            (word >> 22) ^ word
        };
        let idx = |x: i32, y: i32, z: i32| -> usize {
            (((z + dim) % dim) * 1024 + ((y + dim) % dim) * 32 + ((x + dim) % dim)) as usize
        };
        let cell_size = BOX_SIZE / GRID_DIM as f32;
        let mut candidates: Vec<([f32; 3], f32)> = Vec::new();
        let mut dbg_rmax = 0f32;
        let mut dbg_ovmax = 0f32;
        for z in 0..dim {
            for y in 0..dim {
                for x in 0..dim {
                    let r_local = ratio(grid[idx(x, y, z)]);
                    dbg_rmax = dbg_rmax.max(r_local);
                    dbg_ovmax = dbg_ovmax
                        .max((grid[idx(x, y, z)] / mean_raw.max(1e-5) - n_limit).max(0.0));
                    // Barrierless condensation on overflow; instanton-gated
                    // tunneling draw below the Cardy ceiling (shbt7 5.2).
                    let mut tunneled = false;
                    if r_local < 1.0 {
                        let s_inst = Self::instanton_action(r_local as f64);
                        let gamma_nuc = attempt_freq * (-s_inst).exp();
                        let p_nuc = 1.0 - (-gamma_nuc * (1.0 / 60.0)).exp();
                        let rng = pcg(
                            (idx(x, y, z) as u32)
                                ^ (self.frame_index as u32).wrapping_mul(2654435761),
                        ) as f32
                            / u32::MAX as f32;
                        if rng >= p_nuc as f32 || r_local <= DELTA_N_THRESH_NORM {
                            continue;
                        }
                        tunneled = true;
                    } else if r_local - 1.0 <= DELTA_N_THRESH_NORM {
                        continue;
                    }
                    let mut is_max = true;
                    'nms: for dz in -1..=1 {
                        for dy in -1..=1 {
                            for dx in -1..=1 {
                                if dx == 0 && dy == 0 && dz == 0 {
                                    continue;
                                }
                                let n_idx = idx(x + dx, y + dy, z + dz);
                                let n_neigh = ratio(grid[n_idx]);
                                if n_neigh > r_local
                                    || (n_neigh == r_local && n_idx < idx(x, y, z))
                                {
                                    is_max = false;
                                    break 'nms;
                                }
                            }
                        }
                    }
                    if !is_max {
                        continue;
                    }
                    let mut sum_overflow = 0f32;
                    let mut wp = [0f32; 3];
                    for dz in -1..=1 {
                        for dy in -1..=1 {
                            for dx in -1..=1 {
                                let n_idx = idx(x + dx, y + dy, z + dz);
                                let n_local_basin = grid[n_idx] / mean_raw.max(1e-5);
                                // Overflow in absolute normalized units:
                                // n_local - n_limit (bounded by the cell
                                // content as n_limit -> 0 for z -> -1).
                                // A tunneling nucleation has no overflow;
                                // it deposits the register content it
                                // tunnels (the full basin n_local) into the
                                // seed defect.
                                let contrib = if tunneled {
                                    n_local_basin
                                } else {
                                    (n_local_basin - n_limit).max(0.0)
                                };
                                sum_overflow += contrib;
                                wp[0] += ((x + dx) as f32 * cell_size) * contrib;
                                wp[1] += ((y + dy) as f32 * cell_size) * contrib;
                                wp[2] += ((z + dz) as f32 * cell_size) * contrib;
                            }
                        }
                    }
                    let c = [
                        (wp[0] / sum_overflow.max(1e-6) + BOX_SIZE) % BOX_SIZE,
                        (wp[1] / sum_overflow.max(1e-6) + BOX_SIZE) % BOX_SIZE,
                        (wp[2] / sum_overflow.max(1e-6) + BOX_SIZE) % BOX_SIZE,
                    ];
                    candidates.push((c, sum_overflow * SEED_MASS_NORM));
                }
            }
        }
        candidates.truncate(MAX_SEEDS);
        let dt = 1.0f32 / 60.0;
        let mut next_prev: Vec<(f32, [f32; 3], f32)> = Vec::new();
        let mut records = Vec::new();
        let mut matched_count = 0u32;
        let prev_n = self.cpu_prev_seeds.len();
        for (cand_idx, (pos, cand_mass)) in candidates.iter().enumerate() {
            let mut matched_id = -1f32;
            let mut min_dist = TRACK_RADIUS_MPC;
            let mut prev_mass = 0f32;
            for &(pid, ppos, pmass) in &self.cpu_prev_seeds {
                let mut diff = [
                    (pos[0] - ppos[0]).abs(),
                    (pos[1] - ppos[1]).abs(),
                    (pos[2] - ppos[2]).abs(),
                ];
                for d in diff.iter_mut() {
                    *d = d.min(BOX_SIZE - *d);
                }
                let dist = (diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2]).sqrt();
                if dist < min_dist {
                    min_dist = dist;
                    matched_id = pid;
                    prev_mass = pmass;
                }
            }
            if matched_id < 0.0 {
                matched_id = (cand_idx + 100) as f32;
                prev_mass = *cand_mass;
            } else {
                matched_count += 1;
            }
            let accreted = prev_mass + cand_mass * 0.02;
            let m_dot = (accreted - prev_mass) / dt;
            records.push(EmergentSeed {
                pos: *pos,
                mass_msun: accreted,
                m_dot,
                p_debt_gw: accreted * LANDAUER_RATE as f32,
                seed_id: matched_id,
            });
            next_prev.push((matched_id, *pos, accreted));
        }
        self.cpu_prev_seeds = next_prev;
        let tot: f32 = records.iter().map(|r| r.mass_msun).sum();
        self.debug_grid_stats = format!(
            "cpu cands={} matched={} prev_n={} total={:.3e} rmax={:.4} ovmax={:.4} nlim={:.4}",
            records.len(), matched_count, prev_n, tot, dbg_rmax, dbg_ovmax, n_limit,
        );
        self.absorb_seed_records(records);
    }

    /// Enqueue the GPU->CPU copy of tracking state + active seed table and
    /// arm the async map (native targets resolve it on the next poll).
    fn issue_seed_readback(&mut self, encoder: &mut CommandEncoder) {
        if self.readback.in_flight || self.cpu_fallback {
            return;
        }
        encoder.copy_buffer_to_buffer(
            &self.tracking_state_buffer,
            0,
            &self.readback.staging,
            0,
            16,
        );
        encoder.copy_buffer_to_buffer(
            &self.active_seeds_buffer,
            0,
            &self.readback.staging,
            16,
            (MAX_SEEDS * 32) as u64,
        );
        encoder.copy_buffer_to_buffer(
            &self.grid_density_buffer,
            0,
            &self.readback.staging,
            16 + (MAX_SEEDS as u64) * 32,
            (GRID_DIM * GRID_DIM * GRID_DIM) as u64 * 4,
        );
        self.readback.in_flight = true;
    }

    /// Begin the async map of the staging buffer. The completion callback
    /// stashes the decoded bytes into `readback.slot` for the next frame.
    fn map_seed_readback(&mut self) {
        if !self.readback.in_flight || self.cpu_fallback {
            return;
        }
        if !self.readback.map_started {
            let slice = self.readback.staging.slice(..);
            let mapped = self.readback.mapped.clone();
            let calls = self.readback.map_calls.clone();
            let errors = self.readback.map_errors.clone();
            let err_text = self.readback.map_err_text.clone();
            slice.map_async(MapMode::Read, move |res| {
                calls.fetch_add(1, Ordering::SeqCst);
                let ok = res.is_ok();
                if let Err(e) = res {
                    errors.fetch_add(1, Ordering::SeqCst);
                    if let Ok(mut t) = err_text.lock() {
                        *t = format!("{e:?}");
                    }
                }
                mapped.store(ok, Ordering::SeqCst);
            });
            self.readback.map_started = true;
        }
        // Native targets resolve the map on the synchronous poll; on wasm the
        // callback fires on the browser's GPU timeline and the bytes are
        // harvested on a subsequent frame instead. `Wait` cannot block on
        // wasm — request a non-blocking poll instead.
        #[cfg(target_arch = "wasm32")]
        let _ = self.device.poll(Maintain::Poll);
        #[cfg(not(target_arch = "wasm32"))]
        let _ = self.device.poll(Maintain::Wait);
        if !self.readback.mapped.load(Ordering::SeqCst) {
            // If the map request was rejected (transient Dawn/workload
            // failure), drop back to the issue state so the next frame
            // re-enqueues the copy and retries a fresh map. A rejected map
            // leaves wgpu's map_context pinned ("already mapped"), so the
            // staging buffer itself is recycled with a fresh allocation.
            if self.readback.map_errors.load(Ordering::SeqCst) > 0 {
                self.readback.map_errors.store(0, Ordering::SeqCst);
                self.cpu_fallback_errors += 1;
                #[cfg(target_arch = "wasm32")]
                if self.cpu_fallback_errors > 4 {
                    self.cpu_fallback = true;
                }
                self.readback.in_flight = false;
                self.readback.map_started = false;
                self.readback.staging = self.device.create_buffer(&BufferDescriptor {
                    label: Some("seed readback staging (retry)"),
                    size: 16 + (MAX_SEEDS as u64) * 32 + (GRID_DIM * GRID_DIM * GRID_DIM) as u64 * 4,
                    usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
            }
            return;
        }
        {
            let data = self.readback.staging.slice(..).get_mapped_range();
            *self.readback.slot.borrow_mut() = Some(data.to_vec());
            self.drain_ticks += 1000;
        }
        self.readback.staging.unmap();
        self.readback.mapped.store(false, Ordering::SeqCst);
        self.readback.in_flight = false;
        self.readback.map_started = false;
    }

    fn view_proj(&self) -> [[f32; 4]; 4] {
        // Orthographic projection onto the boundary CFT plane.
        let s = 2.0 / BOX_SIZE;
        let ortho = [
            [s, 0.0, 0.0, 0.0],
            [0.0, s, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        let f = 1.0 / (45.0f32.to_radians() / 2.0).tan();
        let aspect = self.width as f32 / self.height as f32;
        let near = 1.0f32;
        let far = 4000.0f32;
        let proj = [
            [f / aspect, 0.0, 0.0, 0.0],
            [0.0, f, 0.0, 0.0],
            [0.0, 0.0, far / (near - far), -1.0],
            [0.0, 0.0, near * far / (near - far), 0.0],
        ];
        let dist = BOX_SIZE * 1.4;
        let dir = self.camera_dir();
        let eye = [dir[0] * dist, dir[1] * dist, dir[2] * dist];
        let view = look_at(eye, [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
        let persp = mat_mul(proj, view);
        let t = self.unwrap_transition.clamp(0.0, 1.0);
        let mut out = persp;
        for r in 0..4 {
            for c in 0..4 {
                out[r][c] = persp[r][c] * (1.0 - t) + ortho[r][c] * t;
            }
        }
        out
    }

    fn emergence_bind_groups(&self) -> (BindGroup, BindGroup) {
        let g0 = self.device.create_bind_group(&BindGroupDescriptor {
            label: Some("emergence g0"),
            layout: &self.emergence_g0_bgl,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: self.emergence_params_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: self.particle_buffers[self.buffer_index].as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: self.grid_density_buffer.as_entire_binding(),
                },
            ],
        });
        let g1 = self.device.create_bind_group(&BindGroupDescriptor {
            label: Some("emergence g1"),
            layout: &self.emergence_g1_bgl,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: self.tracking_state_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: self.seed_candidates_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: self.prev_seeds_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: self.active_seeds_buffer.as_entire_binding(),
                },
            ],
        });
        (g0, g1)
    }

    fn compute_bind_group(&self) -> (BindGroup, BindGroup) {
        let g0 = self.device.create_bind_group(&BindGroupDescriptor {
            label: Some("compute bind group"),
            layout: &self.compute_bgl,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: self.cosmo_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: self.particle_buffers[self.buffer_index].as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: BindingResource::TextureView(
                        &self.density_tex.create_view(&TextureViewDescriptor::default()),
                    ),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: BindingResource::TextureView(
                        &self.force_tex.create_view(&TextureViewDescriptor::default()),
                    ),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: BindingResource::Sampler(&self.sampler),
                },
                BindGroupEntry {
                    binding: 5,
                    resource: self.active_seeds_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 6,
                    resource: self.tracking_state_buffer.as_entire_binding(),
                },
            ],
        });
        let g1 = self.device.create_bind_group(&BindGroupDescriptor {
            label: Some("compute tether group"),
            layout: &self.compute_g1_bgl,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: self.tether_vertex_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: self.tether_indirect_buffer.as_entire_binding(),
                },
            ],
        });
        (g0, g1)
    }

    fn render_bind_groups(&self) -> (BindGroup, BindGroup) {
        let g0 = self.device.create_bind_group(&BindGroupDescriptor {
            label: Some("render bind group"),
            layout: &self.render_bgl,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: self.particle_buffers[self.buffer_index].as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: self.camera_buffer.as_entire_binding(),
                },
            ],
        });
        let g1 = self.device.create_bind_group(&BindGroupDescriptor {
            label: Some("render emergent group"),
            layout: &self.render_g1_bgl,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: self.events_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: self.event_count_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: self.active_seeds_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: self.tracking_state_buffer.as_entire_binding(),
                },
            ],
        });
        (g0, g1)
    }

    fn post_bind_group(&self) -> BindGroup {
        self.device.create_bind_group(&BindGroupDescriptor {
            label: Some("post bind group"),
            layout: &self.post_bgl,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(
                        &self.visible_tex.create_view(&TextureViewDescriptor::default()),
                    ),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::TextureView(
                        &self
                            .distortion_tex
                            .create_view(&TextureViewDescriptor::default()),
                    ),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: BindingResource::Sampler(&self.sampler),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: self.post_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: self.seed_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 5,
                    resource: BindingResource::TextureView(
                        &self.boundary_tex.create_view(&TextureViewDescriptor::default()),
                    ),
                },
                BindGroupEntry {
                    binding: 6,
                    resource: self.condensing_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 7,
                    resource: self.glitch_buffer.as_entire_binding(),
                },
            ],
        })
    }

    fn causal_bind_group(&self) -> BindGroup {
        self.device.create_bind_group(&BindGroupDescriptor {
            label: Some("causal render bind group"),
            layout: &self.causal_bgl,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: self.causal_uniform_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: self.causal_buffer.as_entire_binding(),
                },
            ],
        })
    }

    fn tether_bind_group(&self) -> BindGroup {
        self.device.create_bind_group(&BindGroupDescriptor {
            label: Some("tether bind group"),
            layout: &self.tether_bgl,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: self.camera_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: self.tether_vertex_buffer.as_entire_binding(),
                },
            ],
        })
    }

    fn observer_bind_group(&self) -> BindGroup {
        self.device.create_bind_group(&BindGroupDescriptor {
            label: Some("observer bind group"),
            layout: &self.observer_bgl,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: self.observer_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: self.causal_observer_uniform_buffer.as_entire_binding(),
                },
            ],
        })
    }

    fn tracer_compute_bind_group(&self) -> BindGroup {
        self.device.create_bind_group(&BindGroupDescriptor {
            label: Some("tracer compute bind group"),
            layout: &self.tracer_compute_bgl,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: self.tracer_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::TextureView(
                        &self.entropy_tex.create_view(&TextureViewDescriptor::default()),
                    ),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: BindingResource::Sampler(&self.sampler),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: self.tracer_params_buffer.as_entire_binding(),
                },
            ],
        })
    }

    fn tracer_render_bind_group(&self) -> BindGroup {
        self.device.create_bind_group(&BindGroupDescriptor {
            label: Some("tracer render bind group"),
            layout: &self.tracer_render_bgl,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: self.tracer_buffer.as_entire_binding(),
            }],
        })
    }

    fn tracer_cam_bind_group(&self) -> BindGroup {
        self.device.create_bind_group(&BindGroupDescriptor {
            label: Some("tracer camera bind group"),
            layout: &self.tracer_cam_bgl,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: self.tracer_camera_buffer.as_entire_binding(),
            }],
        })
    }

    /// Advance the simulation timeline by `dt_seconds` and run one
    /// emergence -> nbody -> dual-channel -> composite frame.
    pub fn step(&mut self, dt_seconds: f64, target: Option<&TextureView>) {
        // Timeline: scrub z downward; seeds condense organically across the
        // cosmic-dawn window (z ~ 30 -> 7) wherever rho_E overflows N_limit.
        self.redshift = self.timeline.advance(dt_seconds);
        self.drain_seed_readback();

        let frame = encode_mmio_frame(
            self.frame_index,
            self.redshift,
            self.num_particles as u64,
            self.delta_n_bits,
            self.seed_count,
        );
        self.queue
            .write_buffer(&self.telemetry_buffer, 0, &frame);

        let (f_load, attempt_freq) =
            self.update_redshift_and_loading(self.redshift);
        let mean_density = (self.mean_raw_total
            / (GRID_DIM * GRID_DIM * GRID_DIM) as f64) as f32;

        // Supercomoving metrology step (shbt8 / Thm 9.11): the engine's
        // per-frame code-time increment doubles as the canonical
        // supercomoving tick; the effective scale-factor advance is
        // reconstructed from dtau = da / (a_half^3 * H(a_half)) so the
        // two stay consistent regardless of how fast the user scrubs
        // the timeline (Delta_tau is always the integrator's tick, not
        // the raw timeline jump).
        let a_now = (1.0 / (1.0 + self.redshift.max(-0.9999))) as f32;
        let h_h0 = units::CosmologicalContext::h_of_z(self.redshift) as f32;
        let dt_code = (dt_seconds * self.timeline.speed * 3.0e8).min(0.05) as f32;
        let dtau = dt_code;
        let a_half_probe = a_now.max(1.0e-3);
        let da = dtau * a_half_probe.powi(3) * h_h0;
        let cosmo: CosmoParams = self.metrology.prepare_step_uniforms(
            a_now,
            da,
            h_h0,
            f_load as f32,
            GRID_DIM as f32,
            0.3333, // eta_soft: adaptive Plummer kernel eta/N_grid
            self.num_particles,
            self.emergent_seeds.len() as u32,
            units::kappa_get(f_load as f32),
            BOX_SIZE,
            dt_code,
        );
        self.queue
            .write_buffer(&self.cosmo_buffer, 0, bytemuck::bytes_of(&cosmo));

        let eparams = EmergenceParams {
            grid_dim: GRID_DIM,
            particle_count: self.num_particles,
            box_size: BOX_SIZE,
            delta_t: dt_seconds.max(1e-4) as f32,
            redshift: self.redshift as f32,
            f_load: f_load as f32,
            gamma_geom: GAMMA_GEOM as f32,
            delta_n_thresh: DELTA_N_THRESH_NORM,
            alpha_mass: SEED_MASS_NORM,
            landauer_rate: LANDAUER_RATE as f32,
            track_radius: TRACK_RADIUS_MPC,
            fixed_point_scale: FIXED_POINT_SCALE as f32,
            attempt_freq: attempt_freq as f32,
            frame_seed: self.frame_index as f32,
            mean_density,
            _pad: 0.0,
        };
        self.queue
            .write_buffer(&self.emergence_params_buffer, 0, bytemuck::bytes_of(&eparams));

        let vp = self.view_proj();
        let eye = self.camera_eye();
        let camera = CameraParams {
            view_proj: vp,
            params: [
                // Halved point extent vs the hardcoded-well era: the
                // uniform pre-onset foam otherwise stacks ~11 quads/pixel
                // and additively saturates to white.
                0.003 - 0.001 * self.unwrap_transition,
                BOX_SIZE,
                self.unwrap_transition,
                self.redshift as f32,
            ],
            aux: [
                eye[0],
                eye[1],
                eye[2],
                if self.doppler_enabled { 1.0 } else { 0.0 },
            ],
        };
        self.queue
            .write_buffer(&self.camera_buffer, 0, bytemuck::bytes_of(&camera));

        // Per-frame zero-fill of the condensation registers.
        let zeros = vec![0u8; (GRID_DIM * GRID_DIM * GRID_DIM * 4) as usize];
        self.queue.write_buffer(&self.grid_density_buffer, 0, &zeros);
        self.queue
            .write_buffer(&self.tracking_state_buffer, 0, &[0u8; 16]);
        self.queue.write_buffer(
            &self.tether_indirect_buffer,
            0,
            bytemuck::cast_slice(&[0u32, 1u32, 0u32, 0u32]),
        );

        // Stage the emergent micro-lensing + observer tables (from the
        // latest completed GPU readback).
        let active_seeds = self.update_seed_table(&vp);
        self.update_observer_table(&vp);
        if self.frame_index % 4 == 0 {
            self.upload_boundary_tex();
        }

        let mut vp_flat = [0.0f32; 16];
        let mut inv_flat = [0.0f32; 16];
        let inv = mat_inv(vp);
        for r in 0..4 {
            for c in 0..4 {
                vp_flat[c * 4 + r] = vp[c][r];
                inv_flat[c * 4 + r] = inv[c][r];
            }
        }
        let lensing = LensingUniforms {
            view_proj: vp_flat,
            inv_view_proj: inv_flat,
            cam_pos: [eye[0], eye[1], eye[2], 1.0],
            screen_size: [self.width as f32, self.height as f32],
            lensing_strength: self.lensing_strength,
            dispersion_coeff: self.dispersion_coeff,
            dark_glow_intensity: self.dark_glow_intensity,
            dark_glow_radius: self.dark_glow_radius,
            doppler_enabled: self.doppler_enabled as u32,
            seed_count: active_seeds,
            time: self.frame_index as f32 / 60.0,
            _pad0: 0.0,
            _pad1: [0.0; 2],
            post0: [self.channel_a, self.channel_b, f_load as f32, 1.6],
            post1: [self.unwrap_transition, 0.0, 0.0, 0.0],
            post2: [
                45.0f32.to_radians(), // Theta_FoV (rad)
                self.dispersion_coeff * 0.04, // zeta_disp boundary dispersion
                0.0,
                0.0,
            ],
        };
        self.queue
            .write_buffer(&self.post_buffer, 0, bytemuck::bytes_of(&lensing));

        let tracer_params = TracerParams {
            dt: dt_seconds.max(1e-4) as f32,
            max_lifetime: 240.0,
            grid_dim: GRID_DIM,
            step_scale: 0.6,
        };
        self.queue
            .write_buffer(&self.tracer_params_buffer, 0, bytemuck::bytes_of(&tracer_params));
        self.queue.write_buffer(
            &self.tracer_camera_buffer,
            0,
            bytemuck::bytes_of(&TracerCamera {
                view_proj: vp,
                params: [0.004, BOX_SIZE, self.frame_index as f32 / 60.0, 1.0],
            }),
        );

        // Caustic telemetry for the HUD ledger (analytic estimates — the
        // post pass itself never read-backs G-buffer data).
        let strength = self.lensing_strength * self.channel_b;
        let peak_gamma = 0.428 * strength + 0.05 * strength * (self.frame_index as f32 * 0.1).sin();
        let peak_kappa = 1.185 * strength * (0.5 + 0.5 * f_load as f32)
            + 0.08 * strength * (self.frame_index as f32 * 0.1).cos();
        // Physical Einstein radii in arcseconds for the HUD ledger
        // (same distance-duality geometry as update_seed_table).
        let z_d = self.redshift.max(0.05);
        let z_s = z_d + (1.0_f64).max(0.5 * z_d);
        let d_d = units::CosmologicalContext::angular_diameter_distance_mpc(z_d);
        let d_s = units::CosmologicalContext::angular_diameter_distance_mpc(z_s);
        let d_ds = units::CosmologicalContext::lens_source_distance_mpc(z_d, z_s);
        let ctx = &self.metrology.ctx;
        let mut max_te = 0.0f32;
        for s in &self.emergent_seeds {
            let te_rad = ctx.compute_einstein_radius_rad(s.mass_msun as f64, d_d, d_s, d_ds);
            max_te = max_te.max(te_rad * 206_265.0);
        }
        let mut caustics = active_seeds;
        if peak_kappa >= 1.0 || (peak_gamma * peak_gamma + peak_kappa * peak_kappa) > 0.8 {
            caustics += 1;
        }
        self.telemetry = VisualizerTelemetry {
            peak_shear: peak_gamma,
            peak_convergence: peak_kappa,
            max_einstein_radius: max_te,
            active_caustics: caustics,
        };

        let entropy_scale = if self.redshift <= 7.0 && self.redshift > 0.0 {
            1.0
        } else {
            0.0
        };
        let causal_uniforms = CausalRenderUniforms {
            view_proj: vp,
            cam_pos: [eye[0], eye[1], eye[2], 1.0],
            params: [
                BOX_SIZE,
                self.frame_index as f32 / 60.0,
                entropy_scale,
                0.05,
            ],
        };
        self.queue.write_buffer(
            &self.causal_uniform_buffer,
            0,
            bytemuck::bytes_of(&causal_uniforms),
        );

        // ---- Encoder A: emergent condensation tri-pass + seed readback ----
        let mut encoder_a = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("emergence encoder"),
            });
        // Ping-pong the tracking table: last frame's active seeds become
        // this frame's prev_seeds for persistent-id matching.
        encoder_a.copy_buffer_to_buffer(
            &self.active_seeds_buffer,
            0,
            &self.prev_seeds_buffer,
            0,
            (MAX_SEEDS * 32) as u64,
        );
        {
            let (g0, g1) = self.emergence_bind_groups();
            let mut cpass = encoder_a.begin_compute_pass(&ComputePassDescriptor {
                label: Some("emergent condensation passes"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.emergence_cic_pipeline);
            cpass.set_bind_group(0, &g0, &[]);
            cpass.set_bind_group(1, &g1, &[]);
            cpass.dispatch_workgroups((self.num_particles + 255) / 256, 1, 1);
            cpass.set_pipeline(&self.emergence_detect_pipeline);
            cpass.dispatch_workgroups((GRID_DIM + 3) / 4, (GRID_DIM + 3) / 4, (GRID_DIM + 3) / 4);
            cpass.set_pipeline(&self.emergence_track_pipeline);
            cpass.dispatch_workgroups(1, 1, 1);
        }
        // Entropy streamline advection (Enhancement 3).
        {
            let tbg = self.tracer_compute_bind_group();
            let mut cpass = encoder_a.begin_compute_pass(&ComputePassDescriptor {
                label: Some("entropy tracer pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.tracer_compute_pipeline);
            cpass.set_bind_group(0, &tbg, &[]);
            cpass.dispatch_workgroups((TRACER_COUNT + 63) / 64, 1, 1);
        }
        self.issue_seed_readback(&mut encoder_a);
        self.queue.submit([encoder_a.finish()]);
        self.map_seed_readback();

        // ---- Encoder B: Fast-PM + dual-channel render + composite ----
        let mut encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("frame encoder"),
            });
        {
            let (g0, g1) = self.compute_bind_group();
            let mut cpass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("nbody_pm compute pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.compute_pipeline);
            cpass.set_bind_group(0, &g0, &[]);
            cpass.set_bind_group(1, &g1, &[]);
            cpass.dispatch_workgroups((self.num_particles + 255) / 256, 1, 1);
        }
        // Double-buffered snapshot: copy the updated buffer to the shadow.
        let next = 1 - self.buffer_index;
        let byte_len = (self.num_particles as u64) * std::mem::size_of::<Particle>() as u64;
        encoder.copy_buffer_to_buffer(
            &self.particle_buffers[self.buffer_index],
            0,
            &self.particle_buffers[next],
            0,
            byte_len,
        );
        {
            let visible_view = self
                .visible_tex
                .create_view(&TextureViewDescriptor::default());
            let distortion_view = self
                .distortion_tex
                .create_view(&TextureViewDescriptor::default());
            let (bg, bg1) = self.render_bind_groups();
            let tbg = self.tether_bind_group();
            let obg = self.observer_bind_group();
            let trbg = self.tracer_render_bind_group();
            let tcbg = self.tracer_cam_bind_group();
            let cbg = self.causal_bind_group();
            let mut rpass = encoder.begin_render_pass(&RenderPassDescriptor {
                label: Some("dual channel render pass"),
                color_attachments: &[
                    Some(RenderPassColorAttachment {
                        view: &visible_view,
                        resolve_target: None,
                        ops: Operations {
                            load: LoadOp::Clear(Color::BLACK),
                            store: StoreOp::Store,
                        },
                    }),
                    Some(RenderPassColorAttachment {
                        view: &distortion_view,
                        resolve_target: None,
                        ops: Operations {
                            load: LoadOp::Clear(Color::BLACK),
                            store: StoreOp::Store,
                        },
                    }),
                ],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            rpass.set_pipeline(&self.render_pipeline);
            rpass.set_bind_group(0, &bg, &[]);
            rpass.set_bind_group(1, &bg1, &[]);
            rpass.draw(0..6, 0..self.num_particles);

            // Entropy streamline sprites (Enhancement 3).
            rpass.set_pipeline(&self.tracer_render_pipeline);
            rpass.set_bind_group(0, &trbg, &[]);
            rpass.set_bind_group(1, &tcbg, &[]);
            rpass.draw(0..6, 0..TRACER_COUNT);

            // Past light cone wireframes (Enhancement 7).
            rpass.set_pipeline(&self.cone_pipeline);
            rpass.set_bind_group(0, &obg, &[]);
            rpass.draw(0..32, 0..MAX_SEEDS as u32);

            // Entropy budget spheres (Enhancement 8).
            rpass.set_pipeline(&self.sphere_pipeline);
            rpass.draw(0..6, 0..MAX_SEEDS as u32);

            // Stinespring transition tethers (Enhancement 4).
            rpass.set_pipeline(&self.tether_pipeline);
            rpass.set_bind_group(0, &tbg, &[]);
            rpass.draw_indirect(&self.tether_indirect_buffer, 0);

            // Causal-observer Fresnel ripple shells -> Channel B only.
            if entropy_scale > 0.0 {
                rpass.set_pipeline(&self.causal_pipeline);
                rpass.set_bind_group(0, &self.empty_bg, &[]);
                rpass.set_bind_group(1, &cbg, &[]);
                rpass.draw(0..6, 0..CAUSAL_NODE_COUNT as u32);
            }
        }
        if let Some(target_view) = target {
            let bg = self.post_bind_group();
            let mut rpass = encoder.begin_render_pass(&RenderPassDescriptor {
                label: Some("holographic composite pass"),
                color_attachments: &[Some(RenderPassColorAttachment {
                    view: target_view,
                    resolve_target: None,
                    ops: Operations {
                        load: LoadOp::Clear(Color::BLACK),
                        store: StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            rpass.set_pipeline(&self.post_pipeline);
            rpass.set_bind_group(0, &bg, &[]);
            rpass.draw(0..3, 0..1);
        }
        self.queue.submit([encoder.finish()]);
        self.frame_index += 1;
    }

    /// Gravitational optics controls (shared native/wasm surface).
    pub fn apply_lensing_enabled(&mut self, enabled: bool) {
        self.lensing_strength = if enabled { 1.0 } else { 0.0 };
    }

    pub fn apply_lensing_scale(&mut self, scale: f32) {
        self.lensing_strength = scale.clamp(0.0, 5.0);
    }

    pub fn apply_dispersion(&mut self, dispersion: f32) {
        self.dispersion_coeff = dispersion.clamp(0.0, 1.0);
    }

    pub fn apply_doppler(&mut self, enabled: bool) {
        self.doppler_enabled = enabled;
    }

    pub fn apply_dark_glow(&mut self, intensity: f32) {
        self.dark_glow_intensity = intensity.clamp(0.0, 2.0);
    }

    /// Total condensed defect mass of the emergent seed population (M_sun).
    pub fn emergent_total_mass_msun(&self) -> f64 {
        self.emergent_seeds
            .iter()
            .map(|s| s.mass_msun as f64)
            .sum()
    }

    /// Total Landauer dissipation of the emergent seed population (GW).
    pub fn emergent_landauer_debt_gw(&self) -> f64 {
        self.emergent_seeds
            .iter()
            .map(|s| s.p_debt_gw as f64)
            .sum()
    }

    /// Latest HUD metrics decoded from the telemetry frame.
    pub fn hud_metrics(&self) -> HudMetrics {
        let frame = encode_mmio_frame(
            self.frame_index,
            self.redshift,
            self.num_particles as u64,
            self.delta_n_bits,
            self.seed_count,
        );
        HudMetrics::from_frame(&frame).unwrap_or_default()
    }
}

fn f32_to_f16(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    let mant = bits & 0x7f_ffff;
    if exp <= 0 {
        return sign; // underflow to signed zero
    }
    if exp >= 0x1f {
        return sign | 0x7c00; // overflow to inf
    }
    sign | ((exp as u16) << 10) | ((mant >> 13) as u16)
}

fn look_at(eye: [f32; 3], center: [f32; 3], up: [f32; 3]) -> [[f32; 4]; 4] {
    let f = normalize([
        center[0] - eye[0],
        center[1] - eye[1],
        center[2] - eye[2],
    ]);
    let s = normalize(cross(f, up));
    let u = cross(s, f);
    [
        [s[0], u[0], -f[0], 0.0],
        [s[1], u[1], -f[1], 0.0],
        [s[2], u[2], -f[2], 0.0],
        [
            -dot(s, eye),
            -dot(u, eye),
            dot(f, eye),
            1.0,
        ],
    ]
}

/// General 4x4 inverse (Gauss-Jordan with partial pivoting). Matrices are
/// stored column-major as `m[c][r]` throughout the engine.
fn mat_inv(m: [[f32; 4]; 4]) -> [[f32; 4]; 4] {
    let mut a = [[0f32; 4]; 4];
    for c in 0..4 {
        for r in 0..4 {
            a[r][c] = m[c][r];
        }
    }
    let mut inv = [[0f32; 4]; 4];
    for (i, row) in inv.iter_mut().enumerate() {
        row[i] = 1.0;
    }
    for col in 0..4 {
        let mut pivot = col;
        for r in col + 1..4 {
            if a[r][col].abs() > a[pivot][col].abs() {
                pivot = r;
            }
        }
        if pivot != col {
            a.swap(col, pivot);
            inv.swap(col, pivot);
        }
        let d = a[col][col];
        if d.abs() < 1.0e-12 {
            continue;
        }
        for j in 0..4 {
            a[col][j] /= d;
            inv[col][j] /= d;
        }
        for r in 0..4 {
            if r == col {
                continue;
            }
            let f = a[r][col];
            for j in 0..4 {
                a[r][j] -= f * a[col][j];
                inv[r][j] -= f * inv[col][j];
            }
        }
    }
    let mut out = [[0f32; 4]; 4];
    for c in 0..4 {
        for r in 0..4 {
            out[c][r] = inv[r][c];
        }
    }
    out
}

fn mat_mul(a: [[f32; 4]; 4], b: [[f32; 4]; 4]) -> [[f32; 4]; 4] {
    let mut out = [[0f32; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            out[i][j] = (0..4).map(|k| a[k][j] * b[i][k]).sum();
        }
    }
    out
}

/// Spherical linear interpolation between two unit directions: keeps the
/// camera transition on the observation sphere instead of cutting
/// through it (shbt7 Phase 2, smooth orbit camera).
fn slerp3(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
    let d = (a[0] * b[0] + a[1] * b[1] + a[2] * b[2]).clamp(-1.0, 1.0);
    if d > 0.9995 {
        return normalize([
            a[0] + (b[0] - a[0]) * t,
            a[1] + (b[1] - a[1]) * t,
            a[2] + (b[2] - a[2]) * t,
        ]);
    }
    let theta = d.acos();
    let s = theta.sin().max(1.0e-9);
    let w0 = ((1.0 - t) * theta).sin() / s;
    let w1 = (t * theta).sin() / s;
    [
        a[0] * w0 + b[0] * w1,
        a[1] * w0 + b[1] * w1,
        a[2] * w0 + b[2] * w1,
    ]
}

fn normalize(v: [f32; 3]) -> [f32; 3] {
    let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt().max(1e-9);
    [v[0] / l, v[1] / l, v[2] / l]
}

fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

// ---- WebAssembly bindings -------------------------------------------------

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
impl ShbtWebGpuEngine {
    /// Create an engine bound to `<canvas id="canvas_id">` (async factory).
    #[wasm_bindgen]
    pub async fn create(canvas_id: &str) -> Result<ShbtWebGpuEngine, JsValue> {
        console_error_panic_hook::set_once();
        let _ = console_log::init_with_level(log::Level::Info);

        let window = web_sys::window().ok_or("Global window missing")?;
        let document = window.document().ok_or("Document missing")?;
        let canvas = document
            .get_element_by_id(canvas_id)
            .ok_or("Canvas element not found")?
            .dyn_into::<web_sys::HtmlCanvasElement>()
            .map_err(|_| "element is not a canvas")?;

        let instance = Instance::new(InstanceDescriptor {
            backends: Backends::all(),
            dx12_shader_compiler: Default::default(),
            flags: InstanceFlags::default(),
            gles_minor_version: Gles3MinorVersion::Automatic,
        });

        let surface = instance
            .create_surface(wgpu::SurfaceTarget::Canvas(canvas.clone()))
            .map_err(|e| format!("Failed to create surface: {:?}", e))?;

        let adapter = instance
            .request_adapter(&RequestAdapterOptions {
                power_preference: PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .ok_or("Failed to locate compatible WebGPU adapter")?;

        let (device, queue) = adapter
            .request_device(
                &DeviceDescriptor {
                    label: Some("SHBT-Precision WebGPU Device"),
                    required_features: Features::empty(),
                    required_limits: Limits::downlevel_defaults(),
                },
                None,
            )
            .await
            .map_err(|e| format!("Device request error: {:?}", e))?;

        let width = canvas.width().max(1);
        let height = canvas.height().max(1);
        let caps = surface.get_capabilities(&adapter);
        let format = caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(caps.formats[0]);
        let surface_config = SurfaceConfiguration {
            usage: TextureUsages::RENDER_ATTACHMENT,
            format,
            width,
            height,
            present_mode: PresentMode::Fifo,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &surface_config);

        let mut engine = Self::new_common(device, queue, 262_144);
        engine.surface = Some(surface);
        engine.surface_config = Some(surface_config);
        engine.width = width;
        engine.height = height;
        // The post pass renders to the canvas surface format, not TARGET_FORMAT.
        let post_bgl = Self::build_post_layout(&engine.device);
        engine.post_pipeline = Self::build_post_pipeline(&engine.device, &post_bgl, format);
        engine.post_bgl = post_bgl;
        Ok(engine)
    }

    /// Write one 128-byte SHBT-MMIO telemetry frame into the GPU uniform
    /// region and refresh the engine timeline.
    #[wasm_bindgen]
    pub fn update_frame_telemetry(&mut self, header_bytes: &[u8]) -> Result<(), JsValue> {
        if header_bytes.len() < 128 {
            return Err(JsValue::from_str("Invalid telemetry header size"));
        }
        self.queue
            .write_buffer(&self.telemetry_buffer, 0, &header_bytes[0..128]);
        if let Some(metrics) = HudMetrics::from_frame(header_bytes) {
            self.redshift = metrics.redshift;
            self.timeline.seek(metrics.redshift);
            self.delta_n_bits = metrics.delta_n_bits;
            self.seed_count = if metrics.seed_mass_msun > 0.0 { 1 } else { 0 };
        }
        Ok(())
    }

    /// Advance the timeline and render one frame to the canvas.
    #[wasm_bindgen]
    pub fn step_frame(&mut self, dt_seconds: f64) -> Result<(), JsValue> {
        let surface = self.surface.as_ref().ok_or("no surface")?;
        let frame = match surface.get_current_texture() {
            Ok(frame) => frame,
            Err(_) => {
                surface.configure(&self.device, self.surface_config.as_ref().unwrap());
                surface
                    .get_current_texture()
                    .map_err(|e| JsValue::from_str(&format!("surface lost: {e}")))?
            }
        };
        let view = frame.texture.create_view(&TextureViewDescriptor::default());
        self.step(dt_seconds, Some(&view));
        frame.present();
        Ok(())
    }

    #[wasm_bindgen]
    pub fn set_redshift(&mut self, z: f64) {
        // Scrubbing pins the epoch: pause the free-running timeline so the
        // reported redshift stays at the requested value until playback is
        // explicitly resumed (epoch buttons re-enable it).
        self.timeline.seek(z);
        self.redshift = z;
        self.timeline.playing = false;
    }

    #[wasm_bindgen]
    pub fn set_speed(&mut self, speed: f64) {
        self.timeline.speed = speed;
    }

    #[wasm_bindgen]
    pub fn set_playing(&mut self, playing: bool) {
        self.timeline.playing = playing;
    }

    /// projection: 0 = comoving bulk, 1 = 2D boundary CFT.
    #[wasm_bindgen]
    pub fn set_projection(&mut self, mode: u32) {
        self.unwrap_transition = if mode != 0 { 1.0 } else { 0.0 };
    }

    /// Continuous torus-unwrap transition, 0.0 = comoving bulk,
    /// 1.0 = flat boundary CFT torus [0, 2pi)^2.
    #[wasm_bindgen]
    pub fn set_unwrap_transition(&mut self, value: f32) {
        self.unwrap_transition = value.clamp(0.0, 1.0);
    }

    /// Enable/disable Channel A (visible) and Channel B (dark ghost).
    #[wasm_bindgen]
    pub fn set_channels(&mut self, channel_a: bool, channel_b: bool) {
        self.channel_a = if channel_a { 1.0 } else { 0.0 };
        self.channel_b = if channel_b { 1.0 } else { 0.0 };
    }

    /// Toggle gravitational lensing (macro + seed deflection) on/off.
    #[wasm_bindgen]
    pub fn set_lensing_enabled(&mut self, enabled: bool) {
        self.apply_lensing_enabled(enabled);
    }

    /// Lensing strength scale (lambda_lens), clamped to [0.0, 5.0].
    #[wasm_bindgen]
    pub fn set_lensing_scale(&mut self, scale: f32) {
        self.apply_lensing_scale(scale);
    }

    /// Wave-optics chromatic dispersion coefficient, clamped to [0.0, 1.0].
    #[wasm_bindgen]
    pub fn set_dispersion(&mut self, dispersion: f32) {
        self.apply_dispersion(dispersion);
    }

    /// Toggle relativistic Doppler beaming + thermal color shift.
    #[wasm_bindgen]
    pub fn set_doppler_enabled(&mut self, enabled: bool) {
        self.apply_doppler(enabled);
    }

    /// Volumetric dark-matter halo glow intensity, clamped to [0.0, 2.0].
    #[wasm_bindgen]
    pub fn set_dark_glow(&mut self, intensity: f32) {
        self.apply_dark_glow(intensity);
    }

    /// JSON-encoded HUD metrics of the latest telemetry frame, including
    /// the emergent-seed telemetry channel (seedCount, totalMass,
    /// landauerDebt) read back from the condensation kernels.
    #[wasm_bindgen]
    pub fn hud_json(&self) -> String {
        self.hud_json_impl()
    }

    /// Debug export: emergence uniform inputs + readback state as seen by
    /// the wasm host (diagnoses host-vs-GPU discrepancies in Dawn runs).
    #[wasm_bindgen]
    pub fn debug_eparams(&self) -> String {
        let (f_load, attempt_freq) = self.update_redshift_and_loading(self.redshift);
        let mean_density = (self.mean_raw_total
            / (GRID_DIM * GRID_DIM * GRID_DIM) as f64) as f32;
        format!(
            "{{\"attempt_freq\":{:.6e},\"mean_density\":{:.6e},\"f_load\":{:.6e},\"particles\":{},\"in_flight\":{},\"map_started\":{},\"map_calls\":{},\"map_errors\":{},\"map_err\":\"{}\",\"seed_count\":{},\"cpu_fallback\":{},\"grid\":\"{}\",\"drains\":{},\"redshift\":{:.4}}}",
            attempt_freq, mean_density, f_load, self.num_particles,
            self.readback.in_flight, self.readback.map_started,
            self.readback.map_calls.load(Ordering::SeqCst), self.readback.map_errors.load(Ordering::SeqCst),
            self.readback.map_err_text.lock().map(|t| t.clone()).unwrap_or_default(),
            self.seed_count, self.cpu_fallback, self.debug_grid_stats, self.drain_ticks, self.redshift,
        )
    }
}

fn clean_zero(v: f64) -> f64 {
    if v == 0.0 { 0.0 } else { v }
}

/// Keep hud_json valid JSON even when a telemetry channel overflows to
/// inf/NaN (e.g. runaway diagnostic sums): NaN -> 0, +/-inf -> +/-3.0e38.
fn jnum(v: f64) -> f64 {
    if v.is_nan() {
        0.0
    } else if v.is_infinite() {
        v.signum() * 3.0e38
    } else {
        v
    }
}

impl ShbtWebGpuEngine {
    fn hud_json_impl(&self) -> String {
        let m = self.hud_metrics();
        format!(
            "{{\"z\":{:.6e},\"a\":{:.6e},\"t_gyr\":{:.6e},\"hubble\":{:.4},\
             \"f_load\":{:.8},\"n_vis\":{:.6e},\"n_dark\":{:.6e},\
             \"delta_n_bits\":{:.6e},\"seed_mass_msun\":{:.6e},\
             \"landauer_debt_gw\":{:.6e},\"f_sigma8\":{:.6e},\
             \"delta_isw\":{:.6e},\"particles\":{},\"frame\":{},\
             \"delta_fr_zero\":{},\"e_munu_zero\":{},\"horizon_frozen\":{},\
             \"peak_shear\":{:.6},\"peak_convergence\":{:.6},\
             \"max_einstein_radius\":{:.6},\"active_caustics\":{},\
             \"seedCount\":{},\"totalMass\":{:.6e},\"landauerDebt\":{:.6e},\
             \"redshift\":{:.6e}}}",
            jnum(m.redshift),
            jnum(m.scale_factor),
            jnum(m.bulk_time_gyr),
            jnum(m.hubble),
            jnum(m.loading_frac),
            jnum(m.n_vis),
            jnum(m.n_dark),
            jnum(m.delta_n_bits),
            jnum(m.seed_mass_msun),
            jnum(m.landauer_debt_gw),
            jnum(m.f_sigma8),
            jnum(m.delta_isw),
            m.particle_count,
            m.frame_index,
            m.delta_fr_zero,
            m.e_munu_zero,
            m.horizon_frozen,
            jnum(self.telemetry.peak_shear as f64),
            jnum(self.telemetry.peak_convergence as f64),
            jnum(self.telemetry.max_einstein_radius as f64),
            self.telemetry.active_caustics,
            self.seed_count,
            jnum(clean_zero(self.emergent_total_mass_msun())),
            jnum(clean_zero(self.emergent_landauer_debt_gw())),
            jnum(m.redshift),
        )
    }
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
impl ShbtWebGpuEngine {
    #[wasm_bindgen]
    pub fn particle_count(&self) -> u32 {
        self.num_particles
    }

    /// Render one frame into an offscreen RGBA8 target and resolve the
    /// pixels to JS. Used when the WebGPU canvas cannot be composited
    /// (e.g. headless Chromium / SwiftShader): the page blits the bytes
    /// into a 2D overlay canvas for capture.
    #[wasm_bindgen]
    pub async fn capture_frame_rgba(&mut self, dt_seconds: f64) -> Result<js_sys::Uint8Array, JsValue> {
        if self.capture_target.is_none() {
            let tex = self.device.create_texture(&TextureDescriptor {
                label: Some("capture target"),
                size: Extent3d {
                    width: self.width,
                    height: self.height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: TextureFormat::Rgba8Unorm,
                usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::COPY_SRC,
                view_formats: &[],
            });
            let bytes_per_row = (self.width * 4 + 255) / 256 * 256;
            let staging = self.device.create_buffer(&BufferDescriptor {
                label: Some("capture readback"),
                size: (bytes_per_row * self.height) as u64,
                usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let post_bgl = Self::build_post_layout(&self.device);
            let post = Self::build_post_pipeline(&self.device, &post_bgl, TextureFormat::Rgba8Unorm);
            self.capture_target = Some(tex);
            self.capture_staging = Some(staging);
            self.capture_post_pipeline = Some(post);
            self.capture_post_bgl = Some(post_bgl);
        }

        // Render through the offscreen-format post pipeline, then restore
        // the surface pipeline for normal canvas frames.
        let mut cap_post = self.capture_post_pipeline.take().unwrap();
        let mut cap_bgl = self.capture_post_bgl.take().unwrap();
        std::mem::swap(&mut self.post_pipeline, &mut cap_post);
        std::mem::swap(&mut self.post_bgl, &mut cap_bgl);
        let view = self
            .capture_target
            .as_ref()
            .unwrap()
            .create_view(&TextureViewDescriptor::default());
        self.step(dt_seconds, Some(&view));
        std::mem::swap(&mut self.post_pipeline, &mut cap_post);
        std::mem::swap(&mut self.post_bgl, &mut cap_bgl);
        self.capture_post_pipeline = Some(cap_post);
        self.capture_post_bgl = Some(cap_bgl);

        let bytes_per_row = (self.width * 4 + 255) / 256 * 256;
        let staging = self.capture_staging.as_ref().unwrap();
        let mut encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor { label: None });
        encoder.copy_texture_to_buffer(
            ImageCopyTexture {
                texture: self.capture_target.as_ref().unwrap(),
                mip_level: 0,
                origin: Origin3d::ZERO,
                aspect: TextureAspect::All,
            },
            ImageCopyBuffer {
                buffer: staging,
                layout: ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(bytes_per_row),
                    rows_per_image: Some(self.height),
                },
            },
            Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
        );
        self.queue.submit([encoder.finish()]);

        let slice = staging.slice(..);
        let (tx, rx) = futures_channel::oneshot::channel();
        slice.map_async(MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.device.poll(Maintain::Wait);
        rx.await
            .map_err(|_| "capture map channel dropped")?
            .map_err(|e| format!("capture map_async failed: {e:?}"))?;

        let data = slice.get_mapped_range();
        let mut packed = Vec::with_capacity((self.width * self.height * 4) as usize);
        for row in 0..self.height {
            let start = (row * bytes_per_row) as usize;
            packed.extend_from_slice(&data[start..start + (self.width * 4) as usize]);
        }
        drop(data);
        staging.unmap();
        Ok(js_sys::Uint8Array::from(&packed[..]))
    }
}

// ---- Native (non-wasm) headless entry point --------------------------------

#[cfg(not(target_arch = "wasm32"))]
impl ShbtWebGpuEngine {
    /// Debug: read back the CIC density grid and tracking state.
    pub fn debug_emergence_stats(&self) -> String {
        let n = (GRID_DIM * GRID_DIM * GRID_DIM) as u64;
        let staging = self.device.create_buffer(&BufferDescriptor {
            label: Some("dbg grid staging"),
            size: n * 4 + 16 + (MAX_SEEDS as u64) * 64,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = self.device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("dbg"),
        });
        enc.copy_buffer_to_buffer(&self.grid_density_buffer, 0, &staging, 0, n * 4);
        enc.copy_buffer_to_buffer(&self.tracking_state_buffer, 0, &staging, n * 4, 16);
        enc.copy_buffer_to_buffer(&self.active_seeds_buffer, 0, &staging, n * 4 + 16, (MAX_SEEDS * 32) as u64);
        self.queue.submit([enc.finish()]);
        let slice = staging.slice(..);
        slice.map_async(MapMode::Read, |_| {});
        let _ = self.device.poll(Maintain::Wait);
        let bytes = slice.get_mapped_range().to_vec();
        staging.unmap();
        let data = &bytes[..];
        let cells: Vec<u32> = data[..(n * 4) as usize]
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let max_v = *cells.iter().max().unwrap_or(&0) as f64 / 1024.0;
        let mean_v = cells.iter().map(|&v| v as f64).sum::<f64>() / cells.len() as f64 / 1024.0;
        let ts = &data[(n * 4) as usize..(n * 4) as usize + 16];
        let cand = u32::from_le_bytes([ts[0], ts[1], ts[2], ts[3]]);
        let act = u32::from_le_bytes([ts[4], ts[5], ts[6], ts[7]]);
        let mut seeds = String::new();
        for i in 0..(act.min(5) as usize) {
            let b = (n * 4 + 16 + i as u64 * 32) as usize;
            let f = |o: usize| f32::from_le_bytes([data[b + o], data[b + o + 1], data[b + o + 2], data[b + o + 3]]);
            seeds += &format!(" seed{i}: pos=({:.1},{:.1},{:.1}) m={:.3e} p={:.3e} id={:.0}", f(0), f(4), f(8), f(16), f(24), f(28));
        }
        format!(
            "grid: mean={:.3} max={:.3} nonzero={} | candidates={} active={} |{}",
            mean_v,
            max_v,
            cells.iter().filter(|&&v| v > 0).count(),
            cand,
            act,
            seeds
        )
    }

    /// Headless engine for CI benchmarking and native runners (no surface).
    pub async fn headless(num_particles: u32) -> Result<Self, String> {
        let instance = Instance::new(InstanceDescriptor::default());
        let adapter = instance
            .request_adapter(&RequestAdapterOptions {
                power_preference: PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .ok_or("no WebGPU adapter available")?;
        let (device, queue) = adapter
            .request_device(
                &DeviceDescriptor {
                    label: Some("SHBT headless device"),
                    required_features: Features::empty(),
                    required_limits: Limits::downlevel_defaults(),
                },
                None,
            )
            .await
            .map_err(|e| format!("device request error: {e}"))?;
        Ok(Self::new_common(device, queue, num_particles))
    }

    /// Render one frame into the offscreen composite texture and read back
    /// RGBA8 pixels via a post pass onto a color target.
    pub fn render_to_rgba(&mut self, dt_seconds: f64) -> Vec<u8> {
        let target = self.device.create_texture(&TextureDescriptor {
            label: Some("headless target"),
            size: Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba8Unorm,
            usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        // Rebuild the post pipeline against Rgba8Unorm once.
        let post_bgl = Self::build_post_layout(&self.device);
        self.post_pipeline = Self::build_post_pipeline(&self.device, &post_bgl, TextureFormat::Rgba8Unorm);
        self.post_bgl = post_bgl;
        let view = target.create_view(&TextureViewDescriptor::default());
        self.step(dt_seconds, Some(&view));

        let bytes_per_row = (self.width * 4 + 255) / 256 * 256;
        let staging = self.device.create_buffer(&BufferDescriptor {
            label: Some("readback"),
            size: (bytes_per_row * self.height) as u64,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor { label: None });
        encoder.copy_texture_to_buffer(
            ImageCopyTexture {
                texture: &target,
                mip_level: 0,
                origin: Origin3d::ZERO,
                aspect: TextureAspect::All,
            },
            ImageCopyBuffer {
                buffer: &staging,
                layout: ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(bytes_per_row),
                    rows_per_image: Some(self.height),
                },
            },
            Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
        );
        self.queue.submit([encoder.finish()]);
        let slice = staging.slice(..);
        slice.map_async(MapMode::Read, |_| {});
        self.device.poll(Maintain::Wait);
        let data = slice.get_mapped_range().to_vec();
        staging.unmap();
        // Strip row padding.
        let mut out = Vec::with_capacity((self.width * self.height * 4) as usize);
        for row in 0..self.height as usize {
            let start = row * bytes_per_row as usize;
            out.extend_from_slice(&data[start..start + (self.width * 4) as usize]);
        }
        out
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl ShbtWebGpuEngine {
    /// Native mirror of the wasm `set_redshift`/`hud_json` bindings for the
    /// headless calibration sweep.
    pub fn set_redshift(&mut self, z: f64) {
        self.timeline.seek(z);
        self.redshift = z;
        self.timeline.playing = false;
    }

    pub fn hud_json(&self) -> String {
        self.hud_json_impl()
    }
}
