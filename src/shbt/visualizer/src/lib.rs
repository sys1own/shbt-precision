//! SHBT WebGPU visualizer engine (Tier 2 of the two-tier topology).
//!
//! Consumes 128-byte SHBT-MMIO telemetry frames produced by the Tier-1
//! `shbt_simulator` core and drives the Fast-PM compute pipeline, the
//! dual-channel render pass, and the holographic composite pass.

mod engine;
mod hud;
mod particle;
mod telemetry;

pub use engine::{CausalPointRecord, ParticleRecord, SeedDefectRecord, WasmShbtEngine};
pub use hud::{
    HorizonLedger, HudMetrics, LensingUniforms, SeedDefect, TimelineController,
    VisualizerEngine, VisualizerTelemetry, GAMMA_LOCK, N_SAT,
};
pub use particle::Particle;
pub use telemetry::encode_mmio_frame;

use bytemuck::{Pod, Zeroable};
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::*;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

const GRID_DIM: u32 = 32;
const BOX_SIZE: f32 = 200.0; // comoving Mpc/h
const TARGET_FORMAT: TextureFormat = TextureFormat::Rgba16Float;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct CosmoParams {
    a: f32,
    hubble: f32,
    dt: f32,
    d1_growth: f32,
    d2_growth: f32,
    f_load: f32,
    box_size: f32,
    grid_dim: u32,
    num_particles: u32,
    _pad: [u32; 3],
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

/// Ghost-seed attractor wells (mirrors `make_force_grid` / the WGSL `wells`
/// table) in normalized box units, with Einstein radii (screen units) and
/// softening cores for the micro-lensing seed table.
const SEED_WELLS: [([f32; 3], f32, f32); 4] = [
    ([0.25, 0.25, 0.25], 0.045, 0.008),
    ([0.75, 0.75, 0.25], 0.022, 0.006),
    ([0.25, 0.75, 0.75], 0.030, 0.007),
    ([0.75, 0.25, 0.75], 0.018, 0.005),
];
const MAX_SEEDS: usize = 64;
const CAUSAL_NODE_COUNT: usize = 8;

/// WebGPU engine driving the SHBT cosmological visualizer.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
pub struct ShbtWebGpuEngine {
    device: Device,
    queue: Queue,
    #[allow(dead_code)] // surface is only consumed on wasm32 targets
    surface: Option<Surface<'static>>,
    #[allow(dead_code)]
    surface_config: Option<SurfaceConfiguration>,
    compute_pipeline: ComputePipeline,
    render_pipeline: RenderPipeline,
    post_pipeline: RenderPipeline,
    causal_pipeline: RenderPipeline,
    compute_bgl: BindGroupLayout,
    render_bgl: BindGroupLayout,
    post_bgl: BindGroupLayout,
    causal_bgl: BindGroupLayout,
    particle_buffers: [Buffer; 2],
    cosmo_buffer: Buffer,
    camera_buffer: Buffer,
    post_buffer: Buffer,
    seed_buffer: Buffer,
    causal_buffer: Buffer,
    causal_uniform_buffer: Buffer,
    telemetry_buffer: Buffer,
    density_tex: Texture,
    force_tex: Texture,
    visible_tex: Texture,
    distortion_tex: Texture,
    sampler: Sampler,
    num_particles: u32,
    timeline: TimelineController,
    frame_index: u64,
    redshift: f64,
    delta_n_bits: f64,
    seed_count: u32,
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
    fn build_pipelines(
        device: &Device,
        surface_format: TextureFormat,
    ) -> (
        ComputePipeline,
        RenderPipeline,
        RenderPipeline,
        BindGroupLayout,
        BindGroupLayout,
        BindGroupLayout,
    ) {
        let compute_bgl = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("nbody_pm bind group layout"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
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
                BindGroupLayoutEntry {
                    binding: 3,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Float { filterable: true },
                        view_dimension: TextureViewDimension::D3,
                        multisampled: false,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 4,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Sampler(SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let compute_shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("nbody_pm.wgsl"),
            source: ShaderSource::Wgsl(include_str!("shaders/nbody_pm.wgsl").into()),
        });
        let compute_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("compute pipeline layout"),
            bind_group_layouts: &[&compute_bgl],
            push_constant_ranges: &[],
        });
        let compute_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("nbody_pm compute pipeline"),
            layout: Some(&compute_pipeline_layout),
            module: &compute_shader,
            entry_point: "cs_advance_particles",
        });

        let render_bgl = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("dual_channel bind group layout"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::VERTEX,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::VERTEX_FRAGMENT,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let render_shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("dual_channel_render.wgsl"),
            source: ShaderSource::Wgsl(
                include_str!("shaders/dual_channel_render.wgsl").into(),
            ),
        });
        let blend_add = Some(BlendState {
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
        });
        let render_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("render pipeline layout"),
            bind_group_layouts: &[&render_bgl],
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
                    targets: &[
                    Some(ColorTargetState {
                        format: TARGET_FORMAT,
                        blend: blend_add,
                        write_mask: ColorWrites::ALL,
                    }),
                    Some(ColorTargetState {
                        format: TARGET_FORMAT,
                        blend: blend_add,
                        write_mask: ColorWrites::ALL,
                    }),
                ],
            }),
            primitive: PrimitiveState::default(),
            depth_stencil: None,
            multisample: MultisampleState::default(),
            multiview: None,
        });

        let post_bgl = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("holographic_post bind group layout"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Float { filterable: true },
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Float { filterable: true },
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Sampler(SamplerBindingType::Filtering),
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 3,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 4,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let post_shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("holographic_post.wgsl"),
            source: ShaderSource::Wgsl(include_str!("shaders/holographic_post.wgsl").into()),
        });
        let post_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("post pipeline layout"),
            bind_group_layouts: &[&post_bgl],
            push_constant_ranges: &[],
        });
        let post_pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
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
        });

        (
            compute_pipeline,
            render_pipeline,
            post_pipeline,
            compute_bgl,
            render_bgl,
            post_bgl,
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
        let blend_add = Some(BlendState {
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
                targets: &[
                    None,
                    Some(ColorTargetState {
                        format: TARGET_FORMAT,
                        blend: blend_add,
                        write_mask: ColorWrites::ALL,
                    }),
                ],
            }),
            primitive: PrimitiveState::default(),
            depth_stencil: None,
            multisample: MultisampleState::default(),
            multiview: None,
        });
        (pipeline, empty_bgl, causal_bgl)
    }

    /// Allocate the 3D force grid texture and fill it with the primordial
    /// ghost-seed potential: a few supermassive attractor wells seeded at
    /// fixed comoving coordinates (the mass-congestion condensates).
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
        // Ghost-seed attractor wells at 1/4 and 3/4 box fractions.
        let wells: [[f32; 4]; 4] = [
            [0.25, 0.25, 0.25, 1.0],
            [0.75, 0.75, 0.25, 0.8],
            [0.25, 0.75, 0.75, 0.9],
            [0.75, 0.25, 0.75, 0.7],
        ];
        let mut voxels = vec![[0f32; 4]; (GRID_DIM * GRID_DIM * GRID_DIM) as usize];
        for z in 0..GRID_DIM {
            for y in 0..GRID_DIM {
                for x in 0..GRID_DIM {
                    let p = [
                        x as f32 / GRID_DIM as f32,
                        y as f32 / GRID_DIM as f32,
                        z as f32 / GRID_DIM as f32,
                    ];
                    let mut f = [0f32; 3];
                    for w in &wells {
                        let d = [w[0] - p[0], w[1] - p[1], w[2] - p[2]];
                        let r2 = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).max(1.0e-4);
                        let g = w[3] / r2.sqrt();
                        f[0] += d[0] * g;
                        f[1] += d[1] * g;
                        f[2] += d[2] * g;
                    }
                    // Scale into half-float-friendly range.
                    let idx = (x + y * GRID_DIM + z * GRID_DIM * GRID_DIM) as usize;
                    voxels[idx] = [
                        f[0] * 0.01,
                        f[1] * 0.01,
                        f[2] * 0.01,
                        0.0,
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

    fn new_common(device: Device, queue: Queue, num_particles: u32) -> Self {
        let particles = Particle::seed_lattice(num_particles as usize, BOX_SIZE);
        let particle_buffers = [
            device.create_buffer_init(&BufferInitDescriptor {
                label: Some("particle buffer A"),
                contents: bytemuck::cast_slice(&particles),
                usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            }),
            device.create_buffer_init(&BufferInitDescriptor {
                label: Some("particle buffer B"),
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
        // Causal-observer nodes: one per seed well quadrant (two shells per
        // well for the Fresnel ripple pass), zero-allocation static table.
        let causal_nodes: Vec<CausalRenderNode> = (0..CAUSAL_NODE_COUNT)
            .map(|i| {
                let well = SEED_WELLS[i % SEED_WELLS.len()];
                CausalRenderNode {
                    center: well.0,
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
        let (compute_pipeline, render_pipeline, post_pipeline, compute_bgl, render_bgl, post_bgl) =
            Self::build_pipelines(&device, TARGET_FORMAT);
        let (causal_pipeline, _empty_bgl, causal_bgl) = Self::build_causal_pipeline(&device);
        let visible_tex = mk_target(&device, "visible_tex");
        let distortion_tex = mk_target(&device, "distortion_tex");

        Self {
            device,
            queue,
            surface: None,
            surface_config: None,
            compute_pipeline,
            render_pipeline,
            post_pipeline,
            causal_pipeline,
            compute_bgl,
            render_bgl,
            post_bgl,
            causal_bgl,
            particle_buffers,
            cosmo_buffer,
            camera_buffer,
            post_buffer,
            seed_buffer,
            causal_buffer,
            causal_uniform_buffer,
            telemetry_buffer,
            density_tex,
            force_tex,
            visible_tex,
            distortion_tex,
            sampler,
            num_particles,
            timeline: TimelineController::default(),
            frame_index: 0,
            redshift: 1.0e12,
            delta_n_bits: 0.0,
            seed_count: 0,
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

    /// World-space camera eye for the current orbiting view.
    fn camera_eye(&self) -> [f32; 3] {
        let dist = BOX_SIZE * 1.4;
        let (sy, cy) = (self.frame_index as f32 * 0.0004).sin_cos();
        [sy * dist, 0.25 * BOX_SIZE, -cy * dist]
    }

    /// Project the ghost-seed wells into normalized screen UVs and stage
    /// the 64-entry SeedDefect micro-lensing table for fs_post.
    fn update_seed_table(&mut self, view_proj: &[[f32; 4]; 4]) -> u32 {
        if !(2.0..=30.0).contains(&self.redshift) {
            return 0;
        }
        let mut seeds = [SeedDefect {
            screen_pos: [0.0; 2],
            theta_e: 0.0,
            core_radius: 0.01,
        }; MAX_SEEDS];
        for (i, (well, theta_e, core)) in SEED_WELLS.iter().enumerate() {
            let world = [
                (well[0] - 0.5) * BOX_SIZE,
                (well[1] - 0.5) * BOX_SIZE,
                (well[2] - 0.5) * BOX_SIZE,
                1.0,
            ];
            let mut clip = [0.0f32; 4];
            for r in 0..4 {
                clip[r] = (0..4).map(|c| view_proj[c][r] * world[c]).sum();
            }
            let w = clip[3].max(1.0e-4);
            seeds[i] = SeedDefect {
                screen_pos: [
                    (clip[0] / w) * 0.5 + 0.5,
                    0.5 - (clip[1] / w) * 0.5,
                ],
                theta_e: *theta_e,
                core_radius: *core,
            };
        }
        self.queue
            .write_buffer(&self.seed_buffer, 0, bytemuck::cast_slice(&seeds));
        SEED_WELLS.len() as u32
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
        // Simple perspective camera orbiting the comoving box.
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
        // Camera at distance 1.4 * box behind -z, looking at origin.
        let dist = BOX_SIZE * 1.4;
        let (sy, cy) = (self.frame_index as f32 * 0.0004).sin_cos();
        let eye = [sy * dist, 0.25 * BOX_SIZE, -cy * dist];
        let view = look_at(eye, [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
        let persp = mat_mul(proj, view);
        // Continuous torus unwrapping: blend bulk perspective with the
        // flat boundary-CFT orthographic view by unwrap_transition.
        let t = self.unwrap_transition.clamp(0.0, 1.0);
        let mut out = persp;
        for r in 0..4 {
            for c in 0..4 {
                out[r][c] = persp[r][c] * (1.0 - t) + ortho[r][c] * t;
            }
        }
        out
    }

    fn compute_bind_group(&self) -> BindGroup {
        self.device.create_bind_group(&BindGroupDescriptor {
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
            ],
        })
    }

    fn render_bind_group(&self) -> BindGroup {
        self.device.create_bind_group(&BindGroupDescriptor {
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
        })
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

    /// Advance the simulation timeline by `dt_seconds` and run one
    /// compute+render+composite frame, writing to `target` (or the canvas
    /// surface when attached).
    pub fn step(&mut self, dt_seconds: f64, target: Option<&TextureView>) {
        // Timeline: scrub z downward; ghost seeds condense across z ~ 30..7.
        self.redshift = self.timeline.advance(dt_seconds);
        if (7.0..30.0).contains(&self.redshift) {
            self.seed_count = 4;
            self.delta_n_bits = 6.0e59;
        }
        let frame = encode_mmio_frame(
            self.frame_index,
            self.redshift,
            self.num_particles as u64,
            self.delta_n_bits,
            self.seed_count,
        );
        self.queue
            .write_buffer(&self.telemetry_buffer, 0, &frame);

        let f_load = telemetry::loading_fraction(self.redshift);
        let cosmo = CosmoParams {
            a: (1.0 / (1.0 + self.redshift)) as f32,
            hubble: telemetry::hubble(self.redshift) as f32,
            dt: (dt_seconds * self.timeline.speed * 3.0e8).min(0.05) as f32,
            d1_growth: 1.0,
            d2_growth: 0.0,
            f_load: f_load as f32,
            box_size: BOX_SIZE,
            grid_dim: GRID_DIM,
            num_particles: self.num_particles,
            _pad: [0; 3],
        };
        self.queue
            .write_buffer(&self.cosmo_buffer, 0, bytemuck::bytes_of(&cosmo));

        let vp = self.view_proj();
        let eye = self.camera_eye();
        let camera = CameraParams {
            view_proj: vp,
            params: [
                0.006 - 0.002 * self.unwrap_transition,
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

        // Stage the softened point-mass seed table (micro-lensing loop).
        let active_seeds = self.update_seed_table(&vp);

        // Flatten view_proj into the WGSL column-major uniform layout and
        // fill the 224-byte LensingUniforms contract.
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
        };
        self.queue
            .write_buffer(&self.post_buffer, 0, bytemuck::bytes_of(&lensing));

        // Caustic telemetry for the HUD ledger (analytic estimates — the
        // post pass itself never read-backs G-buffer data).
        let strength = self.lensing_strength * self.channel_b;
        let peak_gamma = 0.428 * strength + 0.05 * strength * (self.frame_index as f32 * 0.1).sin();
        let peak_kappa = 1.185 * strength * (0.5 + 0.5 * f_load as f32)
            + 0.08 * strength * (self.frame_index as f32 * 0.1).cos();
        let mut max_te = 0.0f32;
        if active_seeds > 0 {
            for w in &SEED_WELLS {
                max_te = max_te.max(w.1);
            }
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

        // Causal-observer Fresnel shells: active while GET clustering
        // (0 < z <= 7); entropy scale drops to zero inside the freeze.
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

        let mut encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("frame encoder"),
            });
        {
            let bg = self.compute_bind_group();
            let mut cpass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("nbody_pm compute pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.compute_pipeline);
            cpass.set_bind_group(0, &bg, &[]);
            cpass.dispatch_workgroups((self.num_particles + 255) / 256, 1, 1);
        }
        // Double-buffered snapshot: copy the updated buffer to the shadow.
        let next = 1 - self.buffer_index;
        let byte_len = (self.num_particles as u64) * 32;
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
            let bg = self.render_bind_group();
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
            rpass.draw(0..6, 0..self.num_particles);
            // Causal-observer Fresnel ripple shells -> Channel B only.
            if entropy_scale > 0.0 {
                rpass.set_pipeline(&self.causal_pipeline);
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
    // Convert to row-major working copy.
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
        // Partial pivot.
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
    // Store back column-major.
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
        let (_c, _r, post, _cb, _rb, post_bgl) = Self::build_pipelines(&engine.device, format);
        engine.post_pipeline = post;
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
        self.timeline.seek(z);
        self.redshift = z;
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

    /// JSON-encoded HUD metrics of the latest telemetry frame.
    #[wasm_bindgen]
    pub fn hud_json(&self) -> String {
        let m = self.hud_metrics();
        format!(
            "{{\"z\":{:.6e},\"a\":{:.6e},\"t_gyr\":{:.6e},\"hubble\":{:.4},\
             \"f_load\":{:.8},\"n_vis\":{:.6e},\"n_dark\":{:.6e},\
             \"delta_n_bits\":{:.6e},\"seed_mass_msun\":{:.6e},\
             \"landauer_debt_gw\":{:.6e},\"f_sigma8\":{:.6e},\
             \"delta_isw\":{:.6e},\"particles\":{},\"frame\":{},\
             \"delta_fr_zero\":{},\"e_munu_zero\":{},\"horizon_frozen\":{},\
             \"peak_shear\":{:.6},\"peak_convergence\":{:.6},\
             \"max_einstein_radius\":{:.6},\"active_caustics\":{}}}",
            m.redshift,
            m.scale_factor,
            m.bulk_time_gyr,
            m.hubble,
            m.loading_frac,
            m.n_vis,
            m.n_dark,
            m.delta_n_bits,
            m.seed_mass_msun,
            m.landauer_debt_gw,
            m.f_sigma8,
            m.delta_isw,
            m.particle_count,
            m.frame_index,
            m.delta_fr_zero,
            m.e_munu_zero,
            m.horizon_frozen,
            self.telemetry.peak_shear,
            self.telemetry.peak_convergence,
            self.telemetry.max_einstein_radius,
            self.telemetry.active_caustics
        )
    }

    #[wasm_bindgen]
    pub fn particle_count(&self) -> u32 {
        self.num_particles
    }
}

// ---- Native (non-wasm) headless entry point --------------------------------

#[cfg(not(target_arch = "wasm32"))]
impl ShbtWebGpuEngine {
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
        let (_c, _r, post, _cb, _rb, post_bgl) =
            Self::build_pipelines(&self.device, TextureFormat::Rgba8Unorm);
        self.post_pipeline = post;
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
