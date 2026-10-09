/* tslint:disable */
/* eslint-disable */
/**
* WebGPU engine driving the SHBT cosmological visualizer.
*/
export class ShbtWebGpuEngine {
  free(): void;
/**
* Create an engine bound to `<canvas id="canvas_id">` (async factory).
* @param {string} canvas_id
* @returns {Promise<ShbtWebGpuEngine>}
*/
  static create(canvas_id: string): Promise<ShbtWebGpuEngine>;
/**
* Render one frame into an offscreen RGBA8 target and resolve the
* pixels to JS. Used when the WebGPU canvas cannot be composited
* (e.g. headless Chromium / SwiftShader): the page blits the bytes
* into a 2D overlay canvas for capture.
* @param {number} dt_seconds
* @returns {Promise<Uint8Array>}
*/
  capture_frame_rgba(dt_seconds: number): Promise<Uint8Array>;
/**
* Advance the timeline and render one frame to the canvas.
* @param {number} dt_seconds
*/
  step_frame(dt_seconds: number): void;
/**
* @param {boolean} playing
*/
  set_playing(playing: boolean): void;
/**
* Enable/disable Channel A (visible) and Channel B (dark ghost).
* @param {boolean} channel_a
* @param {boolean} channel_b
*/
  set_channels(channel_a: boolean, channel_b: boolean): void;
/**
* @param {number} z
*/
  set_redshift(z: number): void;
/**
* Debug export: emergence uniform inputs + readback state as seen by
* the wasm host (diagnoses host-vs-GPU discrepancies in Dawn runs).
* @returns {string}
*/
  debug_eparams(): string;
/**
* Volumetric dark-matter halo glow intensity, clamped to [0.0, 2.0].
* @param {number} intensity
*/
  set_dark_glow(intensity: number): void;
/**
* Wave-optics chromatic dispersion coefficient, clamped to [0.0, 1.0].
* @param {number} dispersion
*/
  set_dispersion(dispersion: number): void;
/**
* projection: 0 = comoving bulk, 1 = 2D boundary CFT.
* @param {number} mode
*/
  set_projection(mode: number): void;
/**
* Lensing strength scale (lambda_lens), clamped to [0.0, 5.0].
* @param {number} scale
*/
  set_lensing_scale(scale: number): void;
/**
* Viewport projection toggle: 0 = 3D bulk, 1 = split bulk | phase-space.
* @param {number} mode
*/
  set_viewport_mode(mode: number): void;
/**
* Toggle the emergent seed-glitch post effect (Enhancement 11).
* @param {boolean} enabled
*/
  set_glitch_enabled(enabled: boolean): void;
/**
* Toggle relativistic Doppler beaming + thermal color shift.
* @param {boolean} enabled
*/
  set_doppler_enabled(enabled: boolean): void;
/**
* Toggle gravitational lensing (macro + seed deflection) on/off.
* @param {boolean} enabled
*/
  set_lensing_enabled(enabled: boolean): void;
/**
* Cinematic director hook (shbt10 record_simulation_events.py):
* atomically sets the cosmic epoch and the camera eye/look-at in
* world coordinates. Passing eye == look keeps the previous orbit.
* @param {number} z
* @param {number} px
* @param {number} py
* @param {number} pz
* @param {number} lx
* @param {number} ly
* @param {number} lz
*/
  update_cosmic_state(z: number, px: number, py: number, pz: number, lx: number, ly: number, lz: number): void;
/**
* Seed-glitch master intensity, clamped to [0.0, 1.0].
* @param {number} intensity
*/
  set_glitch_intensity(intensity: number): void;
/**
* Clear the cinematic camera override (back to the procedural orbit).
*/
  clear_camera_override(): void;
/**
* Continuous torus-unwrap transition, 0.0 = comoving bulk,
* 1.0 = flat boundary CFT torus [0, 2pi)^2.
* @param {number} value
*/
  set_unwrap_transition(value: number): void;
/**
* Write one 128-byte SHBT-MMIO telemetry frame into the GPU uniform
* region and refresh the engine timeline.
* @param {Uint8Array} header_bytes
*/
  update_frame_telemetry(header_bytes: Uint8Array): void;
/**
* shbt9 Phase 2: stream the six sandbox controls from the DOM
* (sound speed, percolation threshold, lensing strength, chromatic
* dispersion, target redshift, viewport split mode).
* @param {number} sound_speed_scale
* @param {number} percolation_threshold_scale
* @param {number} lensing_strength
* @param {number} chromatic_dispersion
* @param {number} target_redshift
* @param {number} viewport_split_mode
*/
  set_simulation_controls(sound_speed_scale: number, percolation_threshold_scale: number, lensing_strength: number, chromatic_dispersion: number, target_redshift: number, viewport_split_mode: number): void;
/**
* Count of active sandbox observers (click-dispatched records whose
* entropy budget has not depleted).
* @returns {number}
*/
  get_active_observers_count(): number;
/**
* shbt9 Phase 2 click-to-measure: unproject the canvas NDC point and
* dispatch a causal-point observer with entropy budget
* R_entropy = n_limit - c_get. Returns true when the ray-volume
* intersection landed inside the bulk box.
* @param {number} ndc_x
* @param {number} ndc_y
* @param {number} c_get
* @param {number} n_limit
* @returns {boolean}
*/
  unproject_and_dispatch_causal_point(ndc_x: number, ndc_y: number, c_get: number, n_limit: number): boolean;
/**
* JSON-encoded HUD metrics of the latest telemetry frame, including
* the emergent-seed telemetry channel (seedCount, totalMass,
* landauerDebt) read back from the condensation kernels.
* @returns {string}
*/
  hud_json(): string;
/**
* @param {number} speed
*/
  set_speed(speed: number): void;
/**
* @returns {number}
*/
  particle_count(): number;
}
/**
* Zero-allocation lensing uniform / seed-table manager. Stages the
* `LensingUniforms` block and the 64-entry `SeedDefect` table for direct
* memory-copy upload to the GPU (get_uniform_ptr / get_seeds_ptr).
*/
export class VisualizerEngine {
  free(): void;
/**
*/
  clear_seeds(): void;
/**
* @returns {number}
*/
  get_seeds_ptr(): number;
/**
* @param {number} idx
* @param {number} u
* @param {number} v
* @param {number} theta_e
* @param {number} core
*/
  register_seed(idx: number, u: number, v: number, theta_e: number, core: number): void;
/**
* @param {number} intensity
*/
  set_dark_glow(intensity: number): void;
/**
* @param {number} dispersion
*/
  set_dispersion(dispersion: number): void;
/**
* @returns {number}
*/
  get_uniform_ptr(): number;
/**
* Peak shear/convergence plus dominant Einstein radius and active
* caustic node count for the HUD ledger.
* @param {number} peak_gamma
* @param {number} peak_kappa
*/
  update_telemetry(peak_gamma: number, peak_kappa: number): void;
/**
* @param {number} scale
*/
  set_lensing_scale(scale: number): void;
/**
* @param {boolean} enabled
*/
  set_doppler_enabled(enabled: boolean): void;
/**
* @param {boolean} enabled
*/
  set_lensing_enabled(enabled: boolean): void;
/**
* @param {Float32Array} vp
* @param {Float32Array} inv_vp
* @param {Float32Array} pos
* @param {number} time
*/
  update_camera_matrices(vp: Float32Array, inv_vp: Float32Array, pos: Float32Array, time: number): void;
/**
* @param {number} width
* @param {number} height
*/
  constructor(width: number, height: number);
/**
* @returns {boolean}
*/
  is_dirty(): boolean;
/**
* @returns {VisualizerTelemetry}
*/
  telemetry(): VisualizerTelemetry;
}
/**
* Caustic telemetry readout for the HUD ledger.
*/
export class VisualizerTelemetry {
  free(): void;
/**
*/
  active_caustics: number;
/**
*/
  max_einstein_radius: number;
/**
*/
  peak_convergence: number;
/**
*/
  peak_shear: number;
}
/**
* Wasm/JS-facing engine: owns the shared simulation buffers and the
* epoch ledger; exposes raw buffer pointers for zero-allocation upload.
*/
export class WasmShbtEngine {
  free(): void;
/**
* @returns {number}
*/
  seed_count(): number;
/**
* @param {number} z
*/
  update_epoch(z: number): void;
/**
* @returns {number}
*/
  particle_count(): number;
/**
* @returns {number}
*/
  causal_point_count(): number;
/**
* @returns {number}
*/
  get_seed_buffer_ptr(): number;
/**
* @returns {string}
*/
  get_hud_telemetry_json(): string;
/**
* @returns {number}
*/
  get_particle_buffer_ptr(): number;
/**
* @returns {number}
*/
  get_seed_buffer_byte_len(): number;
/**
* @returns {number}
*/
  get_causal_point_buffer_ptr(): number;
/**
* @returns {number}
*/
  get_particle_buffer_byte_len(): number;
/**
* @returns {number}
*/
  get_causal_point_buffer_byte_len(): number;
/**
* @param {number} particle_count
* @param {number} causal_point_count
* @param {number} seed_count
*/
  constructor(particle_count: number, causal_point_count: number, seed_count: number);
}

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
  readonly memory: WebAssembly.Memory;
  readonly __wbg_shbtwebgpuengine_free: (a: number) => void;
  readonly shbtwebgpuengine_capture_frame_rgba: (a: number, b: number) => number;
  readonly shbtwebgpuengine_clear_camera_override: (a: number) => void;
  readonly shbtwebgpuengine_create: (a: number, b: number) => number;
  readonly shbtwebgpuengine_debug_eparams: (a: number, b: number) => void;
  readonly shbtwebgpuengine_get_active_observers_count: (a: number) => number;
  readonly shbtwebgpuengine_hud_json: (a: number, b: number) => void;
  readonly shbtwebgpuengine_particle_count: (a: number) => number;
  readonly shbtwebgpuengine_set_channels: (a: number, b: number, c: number) => void;
  readonly shbtwebgpuengine_set_dark_glow: (a: number, b: number) => void;
  readonly shbtwebgpuengine_set_dispersion: (a: number, b: number) => void;
  readonly shbtwebgpuengine_set_doppler_enabled: (a: number, b: number) => void;
  readonly shbtwebgpuengine_set_glitch_enabled: (a: number, b: number) => void;
  readonly shbtwebgpuengine_set_glitch_intensity: (a: number, b: number) => void;
  readonly shbtwebgpuengine_set_lensing_enabled: (a: number, b: number) => void;
  readonly shbtwebgpuengine_set_lensing_scale: (a: number, b: number) => void;
  readonly shbtwebgpuengine_set_playing: (a: number, b: number) => void;
  readonly shbtwebgpuengine_set_projection: (a: number, b: number) => void;
  readonly shbtwebgpuengine_set_redshift: (a: number, b: number) => void;
  readonly shbtwebgpuengine_set_simulation_controls: (a: number, b: number, c: number, d: number, e: number, f: number, g: number) => void;
  readonly shbtwebgpuengine_set_speed: (a: number, b: number) => void;
  readonly shbtwebgpuengine_set_unwrap_transition: (a: number, b: number) => void;
  readonly shbtwebgpuengine_set_viewport_mode: (a: number, b: number) => void;
  readonly shbtwebgpuengine_step_frame: (a: number, b: number, c: number) => void;
  readonly shbtwebgpuengine_unproject_and_dispatch_causal_point: (a: number, b: number, c: number, d: number, e: number) => number;
  readonly shbtwebgpuengine_update_cosmic_state: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number) => void;
  readonly shbtwebgpuengine_update_frame_telemetry: (a: number, b: number, c: number, d: number) => void;
  readonly __wbg_wasmshbtengine_free: (a: number) => void;
  readonly wasmshbtengine_causal_point_count: (a: number) => number;
  readonly wasmshbtengine_get_causal_point_buffer_byte_len: (a: number) => number;
  readonly wasmshbtengine_get_causal_point_buffer_ptr: (a: number) => number;
  readonly wasmshbtengine_get_hud_telemetry_json: (a: number, b: number) => void;
  readonly wasmshbtengine_get_particle_buffer_byte_len: (a: number) => number;
  readonly wasmshbtengine_get_particle_buffer_ptr: (a: number) => number;
  readonly wasmshbtengine_get_seed_buffer_byte_len: (a: number) => number;
  readonly wasmshbtengine_get_seed_buffer_ptr: (a: number) => number;
  readonly wasmshbtengine_new: (a: number, b: number, c: number) => number;
  readonly wasmshbtengine_particle_count: (a: number) => number;
  readonly wasmshbtengine_seed_count: (a: number) => number;
  readonly wasmshbtengine_update_epoch: (a: number, b: number) => void;
  readonly __wbg_get_visualizertelemetry_active_caustics: (a: number) => number;
  readonly __wbg_get_visualizertelemetry_max_einstein_radius: (a: number) => number;
  readonly __wbg_get_visualizertelemetry_peak_convergence: (a: number) => number;
  readonly __wbg_get_visualizertelemetry_peak_shear: (a: number) => number;
  readonly __wbg_set_visualizertelemetry_active_caustics: (a: number, b: number) => void;
  readonly __wbg_set_visualizertelemetry_max_einstein_radius: (a: number, b: number) => void;
  readonly __wbg_set_visualizertelemetry_peak_convergence: (a: number, b: number) => void;
  readonly __wbg_set_visualizertelemetry_peak_shear: (a: number, b: number) => void;
  readonly __wbg_visualizerengine_free: (a: number) => void;
  readonly __wbg_visualizertelemetry_free: (a: number) => void;
  readonly visualizerengine_clear_seeds: (a: number) => void;
  readonly visualizerengine_get_seeds_ptr: (a: number) => number;
  readonly visualizerengine_get_uniform_ptr: (a: number) => number;
  readonly visualizerengine_is_dirty: (a: number) => number;
  readonly visualizerengine_new: (a: number, b: number) => number;
  readonly visualizerengine_register_seed: (a: number, b: number, c: number, d: number, e: number, f: number) => void;
  readonly visualizerengine_set_dark_glow: (a: number, b: number) => void;
  readonly visualizerengine_set_dispersion: (a: number, b: number) => void;
  readonly visualizerengine_set_doppler_enabled: (a: number, b: number) => void;
  readonly visualizerengine_set_lensing_enabled: (a: number, b: number) => void;
  readonly visualizerengine_set_lensing_scale: (a: number, b: number) => void;
  readonly visualizerengine_telemetry: (a: number) => number;
  readonly visualizerengine_update_camera_matrices: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number) => void;
  readonly visualizerengine_update_telemetry: (a: number, b: number, c: number) => void;
  readonly __wbindgen_malloc: (a: number, b: number) => number;
  readonly __wbindgen_realloc: (a: number, b: number, c: number, d: number) => number;
  readonly __wbindgen_export_2: WebAssembly.Table;
  readonly _dyn_core_608f92abc48d28da___ops__function__FnMut_______Output______as_wasm_bindgen_551d08d34b350dd1___closure__WasmClosure___describe__invoke___wgpu_ee9b15b627933c05___backend__webgpu__webgpu_sys__gen_GpuUncapturedErrorEvent__GpuUncapturedErrorEvent_____: (a: number, b: number, c: number) => void;
  readonly _dyn_core_608f92abc48d28da___ops__function__FnMut_______Output______as_wasm_bindgen_551d08d34b350dd1___closure__WasmClosure___describe__invoke___wasm_bindgen_551d08d34b350dd1___JsValue_____: (a: number, b: number, c: number) => void;
  readonly __wbindgen_add_to_stack_pointer: (a: number) => number;
  readonly __wbindgen_free: (a: number, b: number, c: number) => void;
  readonly __wbindgen_exn_store: (a: number) => void;
  readonly wasm_bindgen_551d08d34b350dd1___convert__closures__invoke2_mut___wasm_bindgen_551d08d34b350dd1___JsValue__wasm_bindgen_551d08d34b350dd1___JsValue_____: (a: number, b: number, c: number, d: number) => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;
/**
* Instantiates the given `module`, which can either be bytes or
* a precompiled `WebAssembly.Module`.
*
* @param {SyncInitInput} module
*
* @returns {InitOutput}
*/
export function initSync(module: SyncInitInput): InitOutput;

/**
* If `module_or_path` is {RequestInfo} or {URL}, makes a request and
* for everything else, calls `WebAssembly.instantiate` directly.
*
* @param {InitInput | Promise<InitInput>} module_or_path
*
* @returns {Promise<InitOutput>}
*/
export default function __wbg_init (module_or_path?: InitInput | Promise<InitInput>): Promise<InitOutput>;
