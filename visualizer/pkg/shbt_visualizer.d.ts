/* tslint:disable */
/* eslint-disable */

/**
 * WebGPU engine driving the SHBT cosmological visualizer.
 */
export class ShbtWebGpuEngine {
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
    /**
     * Render one frame into an offscreen RGBA8 target and resolve the
     * pixels to JS. Used when the WebGPU canvas cannot be composited
     * (e.g. headless Chromium / SwiftShader): the page blits the bytes
     * into a 2D overlay canvas for capture.
     */
    capture_frame_rgba(dt_seconds: number): Promise<Uint8Array>;
    /**
     * Create an engine bound to `<canvas id="canvas_id">` (async factory).
     */
    static create(canvas_id: string): Promise<ShbtWebGpuEngine>;
    /**
     * JSON-encoded HUD metrics of the latest telemetry frame.
     */
    hud_json(): string;
    particle_count(): number;
    /**
     * Enable/disable Channel A (visible) and Channel B (dark ghost).
     */
    set_channels(channel_a: boolean, channel_b: boolean): void;
    /**
     * Volumetric dark-matter halo glow intensity, clamped to [0.0, 2.0].
     */
    set_dark_glow(intensity: number): void;
    /**
     * Wave-optics chromatic dispersion coefficient, clamped to [0.0, 1.0].
     */
    set_dispersion(dispersion: number): void;
    /**
     * Toggle relativistic Doppler beaming + thermal color shift.
     */
    set_doppler_enabled(enabled: boolean): void;
    /**
     * Toggle gravitational lensing (macro + seed deflection) on/off.
     */
    set_lensing_enabled(enabled: boolean): void;
    /**
     * Lensing strength scale (lambda_lens), clamped to [0.0, 5.0].
     */
    set_lensing_scale(scale: number): void;
    set_playing(playing: boolean): void;
    /**
     * projection: 0 = comoving bulk, 1 = 2D boundary CFT.
     */
    set_projection(mode: number): void;
    set_redshift(z: number): void;
    set_speed(speed: number): void;
    /**
     * Continuous torus-unwrap transition, 0.0 = comoving bulk,
     * 1.0 = flat boundary CFT torus [0, 2pi)^2.
     */
    set_unwrap_transition(value: number): void;
    /**
     * Advance the timeline and render one frame to the canvas.
     */
    step_frame(dt_seconds: number): void;
    /**
     * Write one 128-byte SHBT-MMIO telemetry frame into the GPU uniform
     * region and refresh the engine timeline.
     */
    update_frame_telemetry(header_bytes: Uint8Array): void;
}

/**
 * Zero-allocation lensing uniform / seed-table manager. Stages the
 * `LensingUniforms` block and the 64-entry `SeedDefect` table for direct
 * memory-copy upload to the GPU (get_uniform_ptr / get_seeds_ptr).
 */
export class VisualizerEngine {
    free(): void;
    [Symbol.dispose](): void;
    clear_seeds(): void;
    get_seeds_ptr(): number;
    get_uniform_ptr(): number;
    is_dirty(): boolean;
    constructor(width: number, height: number);
    register_seed(idx: number, u: number, v: number, theta_e: number, core: number): void;
    set_dark_glow(intensity: number): void;
    set_dispersion(dispersion: number): void;
    set_doppler_enabled(enabled: boolean): void;
    set_lensing_enabled(enabled: boolean): void;
    set_lensing_scale(scale: number): void;
    telemetry(): VisualizerTelemetry;
    update_camera_matrices(vp: Float32Array, inv_vp: Float32Array, pos: Float32Array, time: number): void;
    /**
     * Peak shear/convergence plus dominant Einstein radius and active
     * caustic node count for the HUD ledger.
     */
    update_telemetry(peak_gamma: number, peak_kappa: number): void;
}

/**
 * Caustic telemetry readout for the HUD ledger.
 */
export class VisualizerTelemetry {
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
    active_caustics: number;
    max_einstein_radius: number;
    peak_convergence: number;
    peak_shear: number;
}

/**
 * Wasm/JS-facing engine: owns the shared simulation buffers and the
 * epoch ledger; exposes raw buffer pointers for zero-allocation upload.
 */
export class WasmShbtEngine {
    free(): void;
    [Symbol.dispose](): void;
    causal_point_count(): number;
    get_causal_point_buffer_byte_len(): number;
    get_causal_point_buffer_ptr(): number;
    get_hud_telemetry_json(): string;
    get_particle_buffer_byte_len(): number;
    get_particle_buffer_ptr(): number;
    get_seed_buffer_byte_len(): number;
    get_seed_buffer_ptr(): number;
    constructor(particle_count: number, causal_point_count: number, seed_count: number);
    particle_count(): number;
    seed_count(): number;
    update_epoch(z: number): void;
}

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly __wbg_get_visualizertelemetry_active_caustics: (a: number) => number;
    readonly __wbg_get_visualizertelemetry_max_einstein_radius: (a: number) => number;
    readonly __wbg_get_visualizertelemetry_peak_convergence: (a: number) => number;
    readonly __wbg_get_visualizertelemetry_peak_shear: (a: number) => number;
    readonly __wbg_set_visualizertelemetry_active_caustics: (a: number, b: number) => void;
    readonly __wbg_set_visualizertelemetry_max_einstein_radius: (a: number, b: number) => void;
    readonly __wbg_set_visualizertelemetry_peak_convergence: (a: number, b: number) => void;
    readonly __wbg_set_visualizertelemetry_peak_shear: (a: number, b: number) => void;
    readonly __wbg_shbtwebgpuengine_free: (a: number, b: number) => void;
    readonly __wbg_visualizerengine_free: (a: number, b: number) => void;
    readonly __wbg_visualizertelemetry_free: (a: number, b: number) => void;
    readonly __wbg_wasmshbtengine_free: (a: number, b: number) => void;
    readonly shbtwebgpuengine_capture_frame_rgba: (a: number, b: number) => any;
    readonly shbtwebgpuengine_create: (a: number, b: number) => any;
    readonly shbtwebgpuengine_hud_json: (a: number) => [number, number];
    readonly shbtwebgpuengine_particle_count: (a: number) => number;
    readonly shbtwebgpuengine_set_channels: (a: number, b: number, c: number) => void;
    readonly shbtwebgpuengine_set_dark_glow: (a: number, b: number) => void;
    readonly shbtwebgpuengine_set_dispersion: (a: number, b: number) => void;
    readonly shbtwebgpuengine_set_doppler_enabled: (a: number, b: number) => void;
    readonly shbtwebgpuengine_set_lensing_enabled: (a: number, b: number) => void;
    readonly shbtwebgpuengine_set_lensing_scale: (a: number, b: number) => void;
    readonly shbtwebgpuengine_set_playing: (a: number, b: number) => void;
    readonly shbtwebgpuengine_set_projection: (a: number, b: number) => void;
    readonly shbtwebgpuengine_set_redshift: (a: number, b: number) => void;
    readonly shbtwebgpuengine_set_speed: (a: number, b: number) => void;
    readonly shbtwebgpuengine_set_unwrap_transition: (a: number, b: number) => void;
    readonly shbtwebgpuengine_step_frame: (a: number, b: number) => [number, number];
    readonly shbtwebgpuengine_update_frame_telemetry: (a: number, b: number, c: number) => [number, number];
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
    readonly wasmshbtengine_causal_point_count: (a: number) => number;
    readonly wasmshbtengine_get_causal_point_buffer_byte_len: (a: number) => number;
    readonly wasmshbtengine_get_causal_point_buffer_ptr: (a: number) => number;
    readonly wasmshbtengine_get_hud_telemetry_json: (a: number) => [number, number];
    readonly wasmshbtengine_get_particle_buffer_byte_len: (a: number) => number;
    readonly wasmshbtengine_get_particle_buffer_ptr: (a: number) => number;
    readonly wasmshbtengine_get_seed_buffer_byte_len: (a: number) => number;
    readonly wasmshbtengine_get_seed_buffer_ptr: (a: number) => number;
    readonly wasmshbtengine_new: (a: number, b: number, c: number) => number;
    readonly wasmshbtengine_particle_count: (a: number) => number;
    readonly wasmshbtengine_seed_count: (a: number) => number;
    readonly wasmshbtengine_update_epoch: (a: number, b: number) => void;
    readonly wasm_bindgen_740f87ab467470cf___convert__closures_____invoke___js_sys_f8d1592f528dc307___Function_fn_wasm_bindgen_740f87ab467470cf___JsValue_____wasm_bindgen_740f87ab467470cf___sys__Undefined___js_sys_f8d1592f528dc307___Function_fn_wasm_bindgen_740f87ab467470cf___JsValue_____wasm_bindgen_740f87ab467470cf___sys__Undefined_______true_: (a: number, b: number, c: any, d: any) => void;
    readonly wasm_bindgen_740f87ab467470cf___convert__closures_____invoke___wasm_bindgen_740f87ab467470cf___JsValue__core_608f92abc48d28da___result__Result_____wasm_bindgen_740f87ab467470cf___JsError___true_: (a: number, b: number, c: any) => [number, number];
    readonly wasm_bindgen_740f87ab467470cf___convert__closures_____invoke___wasm_bindgen_740f87ab467470cf___JsValue______true_: (a: number, b: number, c: any) => void;
    readonly wasm_bindgen_740f87ab467470cf___convert__closures_____invoke___wasm_bindgen_740f87ab467470cf___JsValue______true__54: (a: number, b: number, c: any) => void;
    readonly __wbindgen_malloc: (a: number, b: number) => number;
    readonly __wbindgen_realloc: (a: number, b: number, c: number, d: number) => number;
    readonly __wbindgen_exn_store: (a: number) => void;
    readonly __externref_table_alloc: () => number;
    readonly __wbindgen_externrefs: WebAssembly.Table;
    readonly __wbindgen_free: (a: number, b: number, c: number) => void;
    readonly __wbindgen_destroy_closure: (a: number, b: number) => void;
    readonly __externref_table_dealloc: (a: number) => void;
    readonly __wbindgen_start: () => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;

/**
 * Instantiates the given `module`, which can either be bytes or
 * a precompiled `WebAssembly.Module`.
 *
 * @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
 *
 * @returns {InitOutput}
 */
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
 * If `module_or_path` is {RequestInfo} or {URL}, makes a request and
 * for everything else, calls `WebAssembly.instantiate` directly.
 *
 * @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
 *
 * @returns {Promise<InitOutput>}
 */
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
