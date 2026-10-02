/* @ts-self-types="./shbt_visualizer.d.ts" */

/**
 * WebGPU engine driving the SHBT cosmological visualizer.
 */
export class ShbtWebGpuEngine {
    static __wrap(ptr) {
        const obj = Object.create(ShbtWebGpuEngine.prototype);
        obj.__wbg_ptr = ptr;
        ShbtWebGpuEngineFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        ShbtWebGpuEngineFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_shbtwebgpuengine_free(ptr, 0);
    }
    /**
     * Render one frame into an offscreen RGBA8 target and resolve the
     * pixels to JS. Used when the WebGPU canvas cannot be composited
     * (e.g. headless Chromium / SwiftShader): the page blits the bytes
     * into a 2D overlay canvas for capture.
     * @param {number} dt_seconds
     * @returns {Promise<Uint8Array>}
     */
    capture_frame_rgba(dt_seconds) {
        const ret = wasm.shbtwebgpuengine_capture_frame_rgba(this.__wbg_ptr, dt_seconds);
        return ret;
    }
    /**
     * Create an engine bound to `<canvas id="canvas_id">` (async factory).
     * @param {string} canvas_id
     * @returns {Promise<ShbtWebGpuEngine>}
     */
    static create(canvas_id) {
        const ptr0 = passStringToWasm0(canvas_id, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.shbtwebgpuengine_create(ptr0, len0);
        return ret;
    }
    /**
     * JSON-encoded HUD metrics of the latest telemetry frame.
     * @returns {string}
     */
    hud_json() {
        let deferred1_0;
        let deferred1_1;
        try {
            const ret = wasm.shbtwebgpuengine_hud_json(this.__wbg_ptr);
            deferred1_0 = ret[0];
            deferred1_1 = ret[1];
            return getStringFromWasm0(ret[0], ret[1]);
        } finally {
            wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
        }
    }
    /**
     * @returns {number}
     */
    particle_count() {
        const ret = wasm.shbtwebgpuengine_particle_count(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * Enable/disable Channel A (visible) and Channel B (dark ghost).
     * @param {boolean} channel_a
     * @param {boolean} channel_b
     */
    set_channels(channel_a, channel_b) {
        wasm.shbtwebgpuengine_set_channels(this.__wbg_ptr, channel_a, channel_b);
    }
    /**
     * Volumetric dark-matter halo glow intensity, clamped to [0.0, 2.0].
     * @param {number} intensity
     */
    set_dark_glow(intensity) {
        wasm.shbtwebgpuengine_set_dark_glow(this.__wbg_ptr, intensity);
    }
    /**
     * Wave-optics chromatic dispersion coefficient, clamped to [0.0, 1.0].
     * @param {number} dispersion
     */
    set_dispersion(dispersion) {
        wasm.shbtwebgpuengine_set_dispersion(this.__wbg_ptr, dispersion);
    }
    /**
     * Toggle relativistic Doppler beaming + thermal color shift.
     * @param {boolean} enabled
     */
    set_doppler_enabled(enabled) {
        wasm.shbtwebgpuengine_set_doppler_enabled(this.__wbg_ptr, enabled);
    }
    /**
     * Toggle gravitational lensing (macro + seed deflection) on/off.
     * @param {boolean} enabled
     */
    set_lensing_enabled(enabled) {
        wasm.shbtwebgpuengine_set_lensing_enabled(this.__wbg_ptr, enabled);
    }
    /**
     * Lensing strength scale (lambda_lens), clamped to [0.0, 5.0].
     * @param {number} scale
     */
    set_lensing_scale(scale) {
        wasm.shbtwebgpuengine_set_lensing_scale(this.__wbg_ptr, scale);
    }
    /**
     * @param {boolean} playing
     */
    set_playing(playing) {
        wasm.shbtwebgpuengine_set_playing(this.__wbg_ptr, playing);
    }
    /**
     * projection: 0 = comoving bulk, 1 = 2D boundary CFT.
     * @param {number} mode
     */
    set_projection(mode) {
        wasm.shbtwebgpuengine_set_projection(this.__wbg_ptr, mode);
    }
    /**
     * @param {number} z
     */
    set_redshift(z) {
        wasm.shbtwebgpuengine_set_redshift(this.__wbg_ptr, z);
    }
    /**
     * @param {number} speed
     */
    set_speed(speed) {
        wasm.shbtwebgpuengine_set_speed(this.__wbg_ptr, speed);
    }
    /**
     * Continuous torus-unwrap transition, 0.0 = comoving bulk,
     * 1.0 = flat boundary CFT torus [0, 2pi)^2.
     * @param {number} value
     */
    set_unwrap_transition(value) {
        wasm.shbtwebgpuengine_set_unwrap_transition(this.__wbg_ptr, value);
    }
    /**
     * Advance the timeline and render one frame to the canvas.
     * @param {number} dt_seconds
     */
    step_frame(dt_seconds) {
        const ret = wasm.shbtwebgpuengine_step_frame(this.__wbg_ptr, dt_seconds);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * Write one 128-byte SHBT-MMIO telemetry frame into the GPU uniform
     * region and refresh the engine timeline.
     * @param {Uint8Array} header_bytes
     */
    update_frame_telemetry(header_bytes) {
        const ptr0 = passArray8ToWasm0(header_bytes, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.shbtwebgpuengine_update_frame_telemetry(this.__wbg_ptr, ptr0, len0);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
}
if (Symbol.dispose) ShbtWebGpuEngine.prototype[Symbol.dispose] = ShbtWebGpuEngine.prototype.free;

/**
 * Zero-allocation lensing uniform / seed-table manager. Stages the
 * `LensingUniforms` block and the 64-entry `SeedDefect` table for direct
 * memory-copy upload to the GPU (get_uniform_ptr / get_seeds_ptr).
 */
export class VisualizerEngine {
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        VisualizerEngineFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_visualizerengine_free(ptr, 0);
    }
    clear_seeds() {
        wasm.visualizerengine_clear_seeds(this.__wbg_ptr);
    }
    /**
     * @returns {number}
     */
    get_seeds_ptr() {
        const ret = wasm.visualizerengine_get_seeds_ptr(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    get_uniform_ptr() {
        const ret = wasm.visualizerengine_get_uniform_ptr(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {boolean}
     */
    is_dirty() {
        const ret = wasm.visualizerengine_is_dirty(this.__wbg_ptr);
        return ret !== 0;
    }
    /**
     * @param {number} width
     * @param {number} height
     */
    constructor(width, height) {
        const ret = wasm.visualizerengine_new(width, height);
        this.__wbg_ptr = ret;
        VisualizerEngineFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * @param {number} idx
     * @param {number} u
     * @param {number} v
     * @param {number} theta_e
     * @param {number} core
     */
    register_seed(idx, u, v, theta_e, core) {
        wasm.visualizerengine_register_seed(this.__wbg_ptr, idx, u, v, theta_e, core);
    }
    /**
     * @param {number} intensity
     */
    set_dark_glow(intensity) {
        wasm.visualizerengine_set_dark_glow(this.__wbg_ptr, intensity);
    }
    /**
     * @param {number} dispersion
     */
    set_dispersion(dispersion) {
        wasm.visualizerengine_set_dispersion(this.__wbg_ptr, dispersion);
    }
    /**
     * @param {boolean} enabled
     */
    set_doppler_enabled(enabled) {
        wasm.visualizerengine_set_doppler_enabled(this.__wbg_ptr, enabled);
    }
    /**
     * @param {boolean} enabled
     */
    set_lensing_enabled(enabled) {
        wasm.visualizerengine_set_lensing_enabled(this.__wbg_ptr, enabled);
    }
    /**
     * @param {number} scale
     */
    set_lensing_scale(scale) {
        wasm.visualizerengine_set_lensing_scale(this.__wbg_ptr, scale);
    }
    /**
     * @returns {VisualizerTelemetry}
     */
    telemetry() {
        const ret = wasm.visualizerengine_telemetry(this.__wbg_ptr);
        return VisualizerTelemetry.__wrap(ret);
    }
    /**
     * @param {Float32Array} vp
     * @param {Float32Array} inv_vp
     * @param {Float32Array} pos
     * @param {number} time
     */
    update_camera_matrices(vp, inv_vp, pos, time) {
        const ptr0 = passArrayF32ToWasm0(vp, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passArrayF32ToWasm0(inv_vp, wasm.__wbindgen_malloc);
        const len1 = WASM_VECTOR_LEN;
        const ptr2 = passArrayF32ToWasm0(pos, wasm.__wbindgen_malloc);
        const len2 = WASM_VECTOR_LEN;
        wasm.visualizerengine_update_camera_matrices(this.__wbg_ptr, ptr0, len0, ptr1, len1, ptr2, len2, time);
    }
    /**
     * Peak shear/convergence plus dominant Einstein radius and active
     * caustic node count for the HUD ledger.
     * @param {number} peak_gamma
     * @param {number} peak_kappa
     */
    update_telemetry(peak_gamma, peak_kappa) {
        wasm.visualizerengine_update_telemetry(this.__wbg_ptr, peak_gamma, peak_kappa);
    }
}
if (Symbol.dispose) VisualizerEngine.prototype[Symbol.dispose] = VisualizerEngine.prototype.free;

/**
 * Caustic telemetry readout for the HUD ledger.
 */
export class VisualizerTelemetry {
    static __wrap(ptr) {
        const obj = Object.create(VisualizerTelemetry.prototype);
        obj.__wbg_ptr = ptr;
        VisualizerTelemetryFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        VisualizerTelemetryFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_visualizertelemetry_free(ptr, 0);
    }
    /**
     * @returns {number}
     */
    get active_caustics() {
        const ret = wasm.__wbg_get_visualizertelemetry_active_caustics(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    get max_einstein_radius() {
        const ret = wasm.__wbg_get_visualizertelemetry_max_einstein_radius(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {number}
     */
    get peak_convergence() {
        const ret = wasm.__wbg_get_visualizertelemetry_peak_convergence(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {number}
     */
    get peak_shear() {
        const ret = wasm.__wbg_get_visualizertelemetry_peak_shear(this.__wbg_ptr);
        return ret;
    }
    /**
     * @param {number} arg0
     */
    set active_caustics(arg0) {
        wasm.__wbg_set_visualizertelemetry_active_caustics(this.__wbg_ptr, arg0);
    }
    /**
     * @param {number} arg0
     */
    set max_einstein_radius(arg0) {
        wasm.__wbg_set_visualizertelemetry_max_einstein_radius(this.__wbg_ptr, arg0);
    }
    /**
     * @param {number} arg0
     */
    set peak_convergence(arg0) {
        wasm.__wbg_set_visualizertelemetry_peak_convergence(this.__wbg_ptr, arg0);
    }
    /**
     * @param {number} arg0
     */
    set peak_shear(arg0) {
        wasm.__wbg_set_visualizertelemetry_peak_shear(this.__wbg_ptr, arg0);
    }
}
if (Symbol.dispose) VisualizerTelemetry.prototype[Symbol.dispose] = VisualizerTelemetry.prototype.free;

/**
 * Wasm/JS-facing engine: owns the shared simulation buffers and the
 * epoch ledger; exposes raw buffer pointers for zero-allocation upload.
 */
export class WasmShbtEngine {
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        WasmShbtEngineFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_wasmshbtengine_free(ptr, 0);
    }
    /**
     * @returns {number}
     */
    causal_point_count() {
        const ret = wasm.wasmshbtengine_causal_point_count(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    get_causal_point_buffer_byte_len() {
        const ret = wasm.wasmshbtengine_get_causal_point_buffer_byte_len(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    get_causal_point_buffer_ptr() {
        const ret = wasm.wasmshbtengine_get_causal_point_buffer_ptr(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {string}
     */
    get_hud_telemetry_json() {
        let deferred1_0;
        let deferred1_1;
        try {
            const ret = wasm.wasmshbtengine_get_hud_telemetry_json(this.__wbg_ptr);
            deferred1_0 = ret[0];
            deferred1_1 = ret[1];
            return getStringFromWasm0(ret[0], ret[1]);
        } finally {
            wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
        }
    }
    /**
     * @returns {number}
     */
    get_particle_buffer_byte_len() {
        const ret = wasm.wasmshbtengine_get_particle_buffer_byte_len(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    get_particle_buffer_ptr() {
        const ret = wasm.wasmshbtengine_get_particle_buffer_ptr(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    get_seed_buffer_byte_len() {
        const ret = wasm.wasmshbtengine_get_seed_buffer_byte_len(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    get_seed_buffer_ptr() {
        const ret = wasm.wasmshbtengine_get_seed_buffer_ptr(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @param {number} particle_count
     * @param {number} causal_point_count
     * @param {number} seed_count
     */
    constructor(particle_count, causal_point_count, seed_count) {
        const ret = wasm.wasmshbtengine_new(particle_count, causal_point_count, seed_count);
        this.__wbg_ptr = ret;
        WasmShbtEngineFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * @returns {number}
     */
    particle_count() {
        const ret = wasm.wasmshbtengine_particle_count(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    seed_count() {
        const ret = wasm.wasmshbtengine_seed_count(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @param {number} z
     */
    update_epoch(z) {
        wasm.wasmshbtengine_update_epoch(this.__wbg_ptr, z);
    }
}
if (Symbol.dispose) WasmShbtEngine.prototype[Symbol.dispose] = WasmShbtEngine.prototype.free;
function __wbg_get_imports() {
    const import0 = {
        __proto__: null,
        __wbg_Window_84483f7af9c60d06: function(arg0) {
            const ret = arg0.Window;
            return ret;
        },
        __wbg_WorkerGlobalScope_ab3a96a72cd85de0: function(arg0) {
            const ret = arg0.WorkerGlobalScope;
            return ret;
        },
        __wbg___wbindgen_debug_string_4687d8d8c2017d52: function(arg0, arg1) {
            const ret = debugString(arg1);
            const ptr1 = passStringToWasm0(ret, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
            const len1 = WASM_VECTOR_LEN;
            getDataViewMemory0().setInt32(arg0 + 4 * 1, len1, true);
            getDataViewMemory0().setInt32(arg0 + 4 * 0, ptr1, true);
        },
        __wbg___wbindgen_is_function_1f9d30630b8b1d3d: function(arg0) {
            const ret = typeof(arg0) === 'function';
            return ret;
        },
        __wbg___wbindgen_is_object_3c45d4f2dde4e749: function(arg0) {
            const val = arg0;
            const ret = typeof(val) === 'object' && val !== null;
            return ret;
        },
        __wbg___wbindgen_is_undefined_8865fb403f8fe9d8: function(arg0) {
            const ret = arg0 === undefined;
            return ret;
        },
        __wbg___wbindgen_string_get_0380ccaa2f57f0d9: function(arg0, arg1) {
            const obj = arg1;
            const ret = typeof(obj) === 'string' ? obj : undefined;
            var ptr1 = isLikeNone(ret) ? 0 : passStringToWasm0(ret, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
            var len1 = WASM_VECTOR_LEN;
            getDataViewMemory0().setInt32(arg0 + 4 * 1, len1, true);
            getDataViewMemory0().setInt32(arg0 + 4 * 0, ptr1, true);
        },
        __wbg___wbindgen_throw_41e9ee4f547fc59a: function(arg0, arg1) {
            throw new Error(getStringFromWasm0(arg0, arg1));
        },
        __wbg__wbg_cb_unref_dcc1a90847f04c41: function(arg0) {
            arg0._wbg_cb_unref();
        },
        __wbg_beginComputePass_9375c71b502a0455: function(arg0, arg1) {
            const ret = arg0.beginComputePass(arg1);
            return ret;
        },
        __wbg_beginRenderPass_fd9b3599b40a4e9d: function(arg0, arg1) {
            const ret = arg0.beginRenderPass(arg1);
            return ret;
        },
        __wbg_buffer_56ec2905a66f58b9: function(arg0) {
            const ret = arg0.buffer;
            return ret;
        },
        __wbg_call_187d372bd5fdd4aa: function() { return handleError(function (arg0, arg1, arg2) {
            const ret = arg0.call(arg1, arg2);
            return ret;
        }, arguments); },
        __wbg_clearBuffer_b84962a8f0aa90b1: function(arg0, arg1, arg2, arg3) {
            arg0.clearBuffer(arg1, arg2, arg3);
        },
        __wbg_clearBuffer_f84175d32a71c1db: function(arg0, arg1, arg2) {
            arg0.clearBuffer(arg1, arg2);
        },
        __wbg_configure_97342a5a768c09bf: function(arg0, arg1) {
            arg0.configure(arg1);
        },
        __wbg_copyBufferToBuffer_ff5f3818f4aae39f: function(arg0, arg1, arg2, arg3, arg4, arg5) {
            arg0.copyBufferToBuffer(arg1, arg2, arg3, arg4, arg5);
        },
        __wbg_copyBufferToTexture_66b359a9bd725e76: function(arg0, arg1, arg2, arg3) {
            arg0.copyBufferToTexture(arg1, arg2, arg3);
        },
        __wbg_copyExternalImageToTexture_0244f8dd1e9d3666: function(arg0, arg1, arg2, arg3) {
            arg0.copyExternalImageToTexture(arg1, arg2, arg3);
        },
        __wbg_copyTextureToBuffer_ed31c6ed5fb0a178: function(arg0, arg1, arg2, arg3) {
            arg0.copyTextureToBuffer(arg1, arg2, arg3);
        },
        __wbg_copyTextureToTexture_fe5c187e35852e73: function(arg0, arg1, arg2, arg3) {
            arg0.copyTextureToTexture(arg1, arg2, arg3);
        },
        __wbg_createBindGroupLayout_4bafc99be90ff601: function(arg0, arg1) {
            const ret = arg0.createBindGroupLayout(arg1);
            return ret;
        },
        __wbg_createBindGroup_754d11219b67ca3a: function(arg0, arg1) {
            const ret = arg0.createBindGroup(arg1);
            return ret;
        },
        __wbg_createBuffer_585f53c980a46c80: function(arg0, arg1) {
            const ret = arg0.createBuffer(arg1);
            return ret;
        },
        __wbg_createCommandEncoder_5245feb4bc162e3b: function(arg0, arg1) {
            const ret = arg0.createCommandEncoder(arg1);
            return ret;
        },
        __wbg_createComputePipeline_c503d3663c315b5d: function(arg0, arg1) {
            const ret = arg0.createComputePipeline(arg1);
            return ret;
        },
        __wbg_createPipelineLayout_f5affbb320321657: function(arg0, arg1) {
            const ret = arg0.createPipelineLayout(arg1);
            return ret;
        },
        __wbg_createQuerySet_7ea147f670344315: function(arg0, arg1) {
            const ret = arg0.createQuerySet(arg1);
            return ret;
        },
        __wbg_createRenderBundleEncoder_09daab6411147213: function(arg0, arg1) {
            const ret = arg0.createRenderBundleEncoder(arg1);
            return ret;
        },
        __wbg_createRenderPipeline_19c47d00e98d4d60: function(arg0, arg1) {
            const ret = arg0.createRenderPipeline(arg1);
            return ret;
        },
        __wbg_createSampler_58b4fb3e4edaac2f: function(arg0, arg1) {
            const ret = arg0.createSampler(arg1);
            return ret;
        },
        __wbg_createShaderModule_ee0f8a959b0f7694: function(arg0, arg1) {
            const ret = arg0.createShaderModule(arg1);
            return ret;
        },
        __wbg_createTexture_b0f46a76611fcc11: function(arg0, arg1) {
            const ret = arg0.createTexture(arg1);
            return ret;
        },
        __wbg_createView_ea89449bf935aae8: function(arg0, arg1) {
            const ret = arg0.createView(arg1);
            return ret;
        },
        __wbg_debug_52bda05ddf50b736: function(arg0) {
            console.debug(arg0);
        },
        __wbg_destroy_4e983840e408b877: function(arg0) {
            arg0.destroy();
        },
        __wbg_destroy_850986b5679d1c9a: function(arg0) {
            arg0.destroy();
        },
        __wbg_destroy_dbabe68ae90d269e: function(arg0) {
            arg0.destroy();
        },
        __wbg_dispatchWorkgroupsIndirect_dbd5b7b0b3c254b4: function(arg0, arg1, arg2) {
            arg0.dispatchWorkgroupsIndirect(arg1, arg2);
        },
        __wbg_dispatchWorkgroups_6ee9d5a6ce45b349: function(arg0, arg1, arg2, arg3) {
            arg0.dispatchWorkgroups(arg1 >>> 0, arg2 >>> 0, arg3 >>> 0);
        },
        __wbg_document_9854e03c05fc8834: function(arg0) {
            const ret = arg0.document;
            return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
        },
        __wbg_drawIndexedIndirect_ccdd49ad96c56a1f: function(arg0, arg1, arg2) {
            arg0.drawIndexedIndirect(arg1, arg2);
        },
        __wbg_drawIndexedIndirect_d471e7025af81960: function(arg0, arg1, arg2) {
            arg0.drawIndexedIndirect(arg1, arg2);
        },
        __wbg_drawIndexed_1702492d59a47fd1: function(arg0, arg1, arg2, arg3, arg4, arg5) {
            arg0.drawIndexed(arg1 >>> 0, arg2 >>> 0, arg3 >>> 0, arg4, arg5 >>> 0);
        },
        __wbg_drawIndexed_65b316e01808838f: function(arg0, arg1, arg2, arg3, arg4, arg5) {
            arg0.drawIndexed(arg1 >>> 0, arg2 >>> 0, arg3 >>> 0, arg4, arg5 >>> 0);
        },
        __wbg_drawIndirect_21df95add1c99821: function(arg0, arg1, arg2) {
            arg0.drawIndirect(arg1, arg2);
        },
        __wbg_drawIndirect_e70b1c753d40ae86: function(arg0, arg1, arg2) {
            arg0.drawIndirect(arg1, arg2);
        },
        __wbg_draw_39f93fd169d46879: function(arg0, arg1, arg2, arg3, arg4) {
            arg0.draw(arg1 >>> 0, arg2 >>> 0, arg3 >>> 0, arg4 >>> 0);
        },
        __wbg_draw_ed75a545c7294fd4: function(arg0, arg1, arg2, arg3, arg4) {
            arg0.draw(arg1 >>> 0, arg2 >>> 0, arg3 >>> 0, arg4 >>> 0);
        },
        __wbg_end_0f83bf598d056f18: function(arg0) {
            arg0.end();
        },
        __wbg_end_83ca5b2c3a835c0f: function(arg0) {
            arg0.end();
        },
        __wbg_error_0bfcc2fa021d2d65: function(arg0) {
            const ret = arg0.error;
            return ret;
        },
        __wbg_error_757e9472f8410341: function(arg0, arg1) {
            let deferred0_0;
            let deferred0_1;
            try {
                deferred0_0 = arg0;
                deferred0_1 = arg1;
                console.error(getStringFromWasm0(arg0, arg1));
            } finally {
                wasm.__wbindgen_free(deferred0_0, deferred0_1, 1);
            }
        },
        __wbg_error_c9cf3fc2064683a9: function(arg0) {
            console.error(arg0);
        },
        __wbg_executeBundles_eca349a0aa326da6: function(arg0, arg1) {
            arg0.executeBundles(arg1);
        },
        __wbg_features_62957b709b1920ad: function(arg0) {
            const ret = arg0.features;
            return ret;
        },
        __wbg_features_a1d3722fcf227919: function(arg0) {
            const ret = arg0.features;
            return ret;
        },
        __wbg_finish_9153ab51653482f3: function(arg0) {
            const ret = arg0.finish();
            return ret;
        },
        __wbg_finish_a208817305958f64: function(arg0) {
            const ret = arg0.finish();
            return ret;
        },
        __wbg_finish_af20f6b480146535: function(arg0, arg1) {
            const ret = arg0.finish(arg1);
            return ret;
        },
        __wbg_finish_ed287af5f0b31c21: function(arg0, arg1) {
            const ret = arg0.finish(arg1);
            return ret;
        },
        __wbg_getBindGroupLayout_a355dd2e3f85e6a7: function(arg0, arg1) {
            const ret = arg0.getBindGroupLayout(arg1 >>> 0);
            return ret;
        },
        __wbg_getBindGroupLayout_c4e025f3bd535641: function(arg0, arg1) {
            const ret = arg0.getBindGroupLayout(arg1 >>> 0);
            return ret;
        },
        __wbg_getContext_635e36719cad2623: function() { return handleError(function (arg0, arg1, arg2) {
            const ret = arg0.getContext(getStringFromWasm0(arg1, arg2));
            return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
        }, arguments); },
        __wbg_getContext_e567868594d2e0c6: function() { return handleError(function (arg0, arg1, arg2) {
            const ret = arg0.getContext(getStringFromWasm0(arg1, arg2));
            return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
        }, arguments); },
        __wbg_getCurrentTexture_fe2f125c8290a060: function(arg0) {
            const ret = arg0.getCurrentTexture();
            return ret;
        },
        __wbg_getElementById_cc94972b404e4eaa: function(arg0, arg1, arg2) {
            const ret = arg0.getElementById(getStringFromWasm0(arg1, arg2));
            return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
        },
        __wbg_getMappedRange_dfe3aadc720182df: function(arg0, arg1, arg2) {
            const ret = arg0.getMappedRange(arg1, arg2);
            return ret;
        },
        __wbg_getPreferredCanvasFormat_1526aa242f263004: function(arg0) {
            const ret = arg0.getPreferredCanvasFormat();
            return (__wbindgen_enum_GpuTextureFormat.indexOf(ret) + 1 || 95) - 1;
        },
        __wbg_get_fd12a9100976c296: function(arg0, arg1) {
            const ret = arg0[arg1 >>> 0];
            return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
        },
        __wbg_gpu_a3357ac66c09a517: function(arg0) {
            const ret = arg0.gpu;
            return ret;
        },
        __wbg_has_934262a65a914df3: function(arg0, arg1, arg2) {
            const ret = arg0.has(getStringFromWasm0(arg1, arg2));
            return ret;
        },
        __wbg_height_fc97e1a0c2e7331f: function(arg0) {
            const ret = arg0.height;
            return ret;
        },
        __wbg_info_26925f73a2eee895: function(arg0) {
            console.info(arg0);
        },
        __wbg_instanceof_GpuAdapter_812393144f747a28: function(arg0) {
            let result;
            try {
                result = arg0 instanceof GPUAdapter;
            } catch (_) {
                result = false;
            }
            const ret = result;
            return ret;
        },
        __wbg_instanceof_GpuCanvasContext_7d8c2aee896960ef: function(arg0) {
            let result;
            try {
                result = arg0 instanceof GPUCanvasContext;
            } catch (_) {
                result = false;
            }
            const ret = result;
            return ret;
        },
        __wbg_instanceof_GpuOutOfMemoryError_5661073b28c982a3: function(arg0) {
            let result;
            try {
                result = arg0 instanceof GPUOutOfMemoryError;
            } catch (_) {
                result = false;
            }
            const ret = result;
            return ret;
        },
        __wbg_instanceof_GpuValidationError_b2b2abc70da536b4: function(arg0) {
            let result;
            try {
                result = arg0 instanceof GPUValidationError;
            } catch (_) {
                result = false;
            }
            const ret = result;
            return ret;
        },
        __wbg_instanceof_HtmlCanvasElement_0a30c11fbbf41841: function(arg0) {
            let result;
            try {
                result = arg0 instanceof HTMLCanvasElement;
            } catch (_) {
                result = false;
            }
            const ret = result;
            return ret;
        },
        __wbg_instanceof_Object_67a83cdc00c5d141: function(arg0) {
            let result;
            try {
                result = arg0 instanceof Object;
            } catch (_) {
                result = false;
            }
            const ret = result;
            return ret;
        },
        __wbg_instanceof_Window_82d71df4eddf88bc: function(arg0) {
            let result;
            try {
                result = arg0 instanceof Window;
            } catch (_) {
                result = false;
            }
            const ret = result;
            return ret;
        },
        __wbg_label_0ff434301c1dc29f: function(arg0, arg1) {
            const ret = arg1.label;
            const ptr1 = passStringToWasm0(ret, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
            const len1 = WASM_VECTOR_LEN;
            getDataViewMemory0().setInt32(arg0 + 4 * 1, len1, true);
            getDataViewMemory0().setInt32(arg0 + 4 * 0, ptr1, true);
        },
        __wbg_length_7f3c00c40364105e: function(arg0) {
            const ret = arg0.length;
            return ret;
        },
        __wbg_limits_00c50879fa9a15a7: function(arg0) {
            const ret = arg0.limits;
            return ret;
        },
        __wbg_limits_94f1cd0da18d4c22: function(arg0) {
            const ret = arg0.limits;
            return ret;
        },
        __wbg_log_17c30ef363c61cf4: function(arg0) {
            console.log(arg0);
        },
        __wbg_mapAsync_5dd79aaf4f56c0f0: function(arg0, arg1, arg2, arg3) {
            const ret = arg0.mapAsync(arg1 >>> 0, arg2, arg3);
            return ret;
        },
        __wbg_maxBindGroups_d7b545d814773df1: function(arg0) {
            const ret = arg0.maxBindGroups;
            return ret;
        },
        __wbg_maxBindingsPerBindGroup_b26d015df21bbdfa: function(arg0) {
            const ret = arg0.maxBindingsPerBindGroup;
            return ret;
        },
        __wbg_maxBufferSize_de4fb5ea32634d2e: function(arg0) {
            const ret = arg0.maxBufferSize;
            return ret;
        },
        __wbg_maxComputeInvocationsPerWorkgroup_b666c8d42afcfb7c: function(arg0) {
            const ret = arg0.maxComputeInvocationsPerWorkgroup;
            return ret;
        },
        __wbg_maxComputeWorkgroupSizeX_bfe6212af4533b2f: function(arg0) {
            const ret = arg0.maxComputeWorkgroupSizeX;
            return ret;
        },
        __wbg_maxComputeWorkgroupSizeY_31156375aa7b93a4: function(arg0) {
            const ret = arg0.maxComputeWorkgroupSizeY;
            return ret;
        },
        __wbg_maxComputeWorkgroupSizeZ_9c96c3749eb89a89: function(arg0) {
            const ret = arg0.maxComputeWorkgroupSizeZ;
            return ret;
        },
        __wbg_maxComputeWorkgroupStorageSize_d4d4f855ff543384: function(arg0) {
            const ret = arg0.maxComputeWorkgroupStorageSize;
            return ret;
        },
        __wbg_maxComputeWorkgroupsPerDimension_d3c6855db6a1497a: function(arg0) {
            const ret = arg0.maxComputeWorkgroupsPerDimension;
            return ret;
        },
        __wbg_maxDynamicStorageBuffersPerPipelineLayout_4a71448e3d67653e: function(arg0) {
            const ret = arg0.maxDynamicStorageBuffersPerPipelineLayout;
            return ret;
        },
        __wbg_maxDynamicUniformBuffersPerPipelineLayout_5d7e7ab3786be24d: function(arg0) {
            const ret = arg0.maxDynamicUniformBuffersPerPipelineLayout;
            return ret;
        },
        __wbg_maxInterStageShaderComponents_9d05c7ae8c47498d: function(arg0) {
            const ret = arg0.maxInterStageShaderComponents;
            return ret;
        },
        __wbg_maxSampledTexturesPerShaderStage_672c74b8d59a31cc: function(arg0) {
            const ret = arg0.maxSampledTexturesPerShaderStage;
            return ret;
        },
        __wbg_maxSamplersPerShaderStage_b417de86ba96244d: function(arg0) {
            const ret = arg0.maxSamplersPerShaderStage;
            return ret;
        },
        __wbg_maxStorageBufferBindingSize_fcff90d733f4dff6: function(arg0) {
            const ret = arg0.maxStorageBufferBindingSize;
            return ret;
        },
        __wbg_maxStorageBuffersPerShaderStage_bf5d03da6fbc393e: function(arg0) {
            const ret = arg0.maxStorageBuffersPerShaderStage;
            return ret;
        },
        __wbg_maxStorageTexturesPerShaderStage_4b8048876f1a3bfe: function(arg0) {
            const ret = arg0.maxStorageTexturesPerShaderStage;
            return ret;
        },
        __wbg_maxTextureArrayLayers_d6f3b298c8c3f211: function(arg0) {
            const ret = arg0.maxTextureArrayLayers;
            return ret;
        },
        __wbg_maxTextureDimension1D_bf6699d60b3d6d53: function(arg0) {
            const ret = arg0.maxTextureDimension1D;
            return ret;
        },
        __wbg_maxTextureDimension2D_0dffed57b5b5494c: function(arg0) {
            const ret = arg0.maxTextureDimension2D;
            return ret;
        },
        __wbg_maxTextureDimension3D_2c64863f596dfe80: function(arg0) {
            const ret = arg0.maxTextureDimension3D;
            return ret;
        },
        __wbg_maxUniformBufferBindingSize_99ce6228c6b1a159: function(arg0) {
            const ret = arg0.maxUniformBufferBindingSize;
            return ret;
        },
        __wbg_maxUniformBuffersPerShaderStage_6c2eb8354f6591e3: function(arg0) {
            const ret = arg0.maxUniformBuffersPerShaderStage;
            return ret;
        },
        __wbg_maxVertexAttributes_93cb29cb7eb03efc: function(arg0) {
            const ret = arg0.maxVertexAttributes;
            return ret;
        },
        __wbg_maxVertexBufferArrayStride_5348a5f186321487: function(arg0) {
            const ret = arg0.maxVertexBufferArrayStride;
            return ret;
        },
        __wbg_maxVertexBuffers_b853e682ab401f10: function(arg0) {
            const ret = arg0.maxVertexBuffers;
            return ret;
        },
        __wbg_message_408e560bce2d5baa: function(arg0, arg1) {
            const ret = arg1.message;
            const ptr1 = passStringToWasm0(ret, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
            const len1 = WASM_VECTOR_LEN;
            getDataViewMemory0().setInt32(arg0 + 4 * 1, len1, true);
            getDataViewMemory0().setInt32(arg0 + 4 * 0, ptr1, true);
        },
        __wbg_minStorageBufferOffsetAlignment_defeef24db62c786: function(arg0) {
            const ret = arg0.minStorageBufferOffsetAlignment;
            return ret;
        },
        __wbg_minUniformBufferOffsetAlignment_5709e95a6f039bcc: function(arg0) {
            const ret = arg0.minUniformBufferOffsetAlignment;
            return ret;
        },
        __wbg_navigator_2156486643462a87: function(arg0) {
            const ret = arg0.navigator;
            return ret;
        },
        __wbg_navigator_c9bce48d4b578c9c: function(arg0) {
            const ret = arg0.navigator;
            return ret;
        },
        __wbg_new_227d7c05414eb861: function() {
            const ret = new Error();
            return ret;
        },
        __wbg_new_617a8cdb8bb1130e: function() {
            const ret = new Object();
            return ret;
        },
        __wbg_new_ee2291f50781bf1d: function() {
            const ret = new Array();
            return ret;
        },
        __wbg_new_from_slice_9a868026ffa4208a: function(arg0, arg1) {
            const ret = new Uint8Array(getArrayU8FromWasm0(arg0, arg1));
            return ret;
        },
        __wbg_new_typed_b01cb72a8af741a3: function(arg0, arg1) {
            try {
                var state0 = {a: arg0, b: arg1};
                var cb0 = (arg0, arg1) => {
                    const a = state0.a;
                    state0.a = 0;
                    try {
                        return wasm_bindgen_740f87ab467470cf___convert__closures_____invoke___js_sys_f8d1592f528dc307___Function_fn_wasm_bindgen_740f87ab467470cf___JsValue_____wasm_bindgen_740f87ab467470cf___sys__Undefined___js_sys_f8d1592f528dc307___Function_fn_wasm_bindgen_740f87ab467470cf___JsValue_____wasm_bindgen_740f87ab467470cf___sys__Undefined_______true_(a, state0.b, arg0, arg1);
                    } finally {
                        state0.a = a;
                    }
                };
                const ret = new Promise(cb0);
                return ret;
            } finally {
                state0.a = 0;
            }
        },
        __wbg_new_with_byte_offset_and_length_2f5d7fc2a828b74d: function(arg0, arg1, arg2) {
            const ret = new Uint8Array(arg0, arg1 >>> 0, arg2 >>> 0);
            return ret;
        },
        __wbg_popErrorScope_a78bee8446b72279: function(arg0) {
            const ret = arg0.popErrorScope();
            return ret;
        },
        __wbg_prototypesetcall_bc27214492979395: function(arg0, arg1, arg2) {
            Uint8Array.prototype.set.call(getArrayU8FromWasm0(arg0, arg1), arg2);
        },
        __wbg_pushErrorScope_1e02cbedfeae6073: function(arg0, arg1) {
            arg0.pushErrorScope(__wbindgen_enum_GpuErrorFilter[arg1]);
        },
        __wbg_push_2baf45db356cf468: function(arg0, arg1) {
            const ret = arg0.push(arg1);
            return ret;
        },
        __wbg_querySelectorAll_6aebfe3df1fb2014: function() { return handleError(function (arg0, arg1, arg2) {
            const ret = arg0.querySelectorAll(getStringFromWasm0(arg1, arg2));
            return ret;
        }, arguments); },
        __wbg_queueMicrotask_9833f9a49df95a49: function(arg0) {
            const ret = arg0.queueMicrotask;
            return ret;
        },
        __wbg_queueMicrotask_a72f977e97f23c5f: function(arg0) {
            queueMicrotask(arg0);
        },
        __wbg_queue_86095a7bbfecffb7: function(arg0) {
            const ret = arg0.queue;
            return ret;
        },
        __wbg_requestAdapter_200309a5193bf1eb: function(arg0, arg1) {
            const ret = arg0.requestAdapter(arg1);
            return ret;
        },
        __wbg_requestDevice_125eeb799c66e4b6: function(arg0, arg1) {
            const ret = arg0.requestDevice(arg1);
            return ret;
        },
        __wbg_resolveQuerySet_a3e53789e95185d3: function(arg0, arg1, arg2, arg3, arg4, arg5) {
            arg0.resolveQuerySet(arg1, arg2 >>> 0, arg3 >>> 0, arg4, arg5 >>> 0);
        },
        __wbg_resolve_0076e10020304ede: function(arg0) {
            const ret = Promise.resolve(arg0);
            return ret;
        },
        __wbg_setBindGroup_1d442a48a595b3bf: function(arg0, arg1, arg2) {
            arg0.setBindGroup(arg1 >>> 0, arg2);
        },
        __wbg_setBindGroup_504394ea36cb4add: function(arg0, arg1, arg2, arg3, arg4, arg5, arg6) {
            arg0.setBindGroup(arg1 >>> 0, arg2, getArrayU32FromWasm0(arg3, arg4), arg5, arg6 >>> 0);
        },
        __wbg_setBindGroup_5d01c655b2a7befe: function(arg0, arg1, arg2) {
            arg0.setBindGroup(arg1 >>> 0, arg2);
        },
        __wbg_setBindGroup_8420d8ef9f49bbba: function(arg0, arg1, arg2, arg3, arg4, arg5, arg6) {
            arg0.setBindGroup(arg1 >>> 0, arg2, getArrayU32FromWasm0(arg3, arg4), arg5, arg6 >>> 0);
        },
        __wbg_setBindGroup_bf56e5d623ae0a95: function(arg0, arg1, arg2) {
            arg0.setBindGroup(arg1 >>> 0, arg2);
        },
        __wbg_setBindGroup_cd569ab5be10d0b3: function(arg0, arg1, arg2, arg3, arg4, arg5, arg6) {
            arg0.setBindGroup(arg1 >>> 0, arg2, getArrayU32FromWasm0(arg3, arg4), arg5, arg6 >>> 0);
        },
        __wbg_setBlendConstant_1a70f218a633617e: function(arg0, arg1) {
            arg0.setBlendConstant(arg1);
        },
        __wbg_setIndexBuffer_1d2dd076f925d2ec: function(arg0, arg1, arg2, arg3) {
            arg0.setIndexBuffer(arg1, __wbindgen_enum_GpuIndexFormat[arg2], arg3);
        },
        __wbg_setIndexBuffer_3882ded30efe8699: function(arg0, arg1, arg2, arg3, arg4) {
            arg0.setIndexBuffer(arg1, __wbindgen_enum_GpuIndexFormat[arg2], arg3, arg4);
        },
        __wbg_setIndexBuffer_452b4561a3899ce8: function(arg0, arg1, arg2, arg3, arg4) {
            arg0.setIndexBuffer(arg1, __wbindgen_enum_GpuIndexFormat[arg2], arg3, arg4);
        },
        __wbg_setIndexBuffer_d05b723ebb2004a2: function(arg0, arg1, arg2, arg3) {
            arg0.setIndexBuffer(arg1, __wbindgen_enum_GpuIndexFormat[arg2], arg3);
        },
        __wbg_setPipeline_41c08398b6fdbaf5: function(arg0, arg1) {
            arg0.setPipeline(arg1);
        },
        __wbg_setPipeline_9f00755767652963: function(arg0, arg1) {
            arg0.setPipeline(arg1);
        },
        __wbg_setPipeline_c75036e1cbd58786: function(arg0, arg1) {
            arg0.setPipeline(arg1);
        },
        __wbg_setScissorRect_103aee4136a0b8bf: function(arg0, arg1, arg2, arg3, arg4) {
            arg0.setScissorRect(arg1 >>> 0, arg2 >>> 0, arg3 >>> 0, arg4 >>> 0);
        },
        __wbg_setStencilReference_d2cd6bbf177426c7: function(arg0, arg1) {
            arg0.setStencilReference(arg1 >>> 0);
        },
        __wbg_setVertexBuffer_923bd23cd3da619c: function(arg0, arg1, arg2, arg3) {
            arg0.setVertexBuffer(arg1 >>> 0, arg2, arg3);
        },
        __wbg_setVertexBuffer_96aa126628c421ae: function(arg0, arg1, arg2, arg3, arg4) {
            arg0.setVertexBuffer(arg1 >>> 0, arg2, arg3, arg4);
        },
        __wbg_setVertexBuffer_e1e02336688a3924: function(arg0, arg1, arg2, arg3, arg4) {
            arg0.setVertexBuffer(arg1 >>> 0, arg2, arg3, arg4);
        },
        __wbg_setVertexBuffer_f2936fdbb904abbd: function(arg0, arg1, arg2, arg3) {
            arg0.setVertexBuffer(arg1 >>> 0, arg2, arg3);
        },
        __wbg_setViewport_a2868622b5237556: function(arg0, arg1, arg2, arg3, arg4, arg5, arg6) {
            arg0.setViewport(arg1, arg2, arg3, arg4, arg5, arg6);
        },
        __wbg_set_145a351398b48c65: function() { return handleError(function (arg0, arg1, arg2) {
            const ret = Reflect.set(arg0, arg1, arg2);
            return ret;
        }, arguments); },
        __wbg_set_34d08fd992d43d61: function(arg0, arg1, arg2) {
            arg0.set(arg1, arg2 >>> 0);
        },
        __wbg_set_height_c9789c1c77eaedff: function(arg0, arg1) {
            arg0.height = arg1 >>> 0;
        },
        __wbg_set_height_fde391767df1ce27: function(arg0, arg1) {
            arg0.height = arg1 >>> 0;
        },
        __wbg_set_onuncapturederror_e4269b42b6a94b67: function(arg0, arg1) {
            arg0.onuncapturederror = arg1;
        },
        __wbg_set_width_0ca908d167690992: function(arg0, arg1) {
            arg0.width = arg1 >>> 0;
        },
        __wbg_set_width_b0e1267db4b196b5: function(arg0, arg1) {
            arg0.width = arg1 >>> 0;
        },
        __wbg_shbtwebgpuengine_new: function(arg0) {
            const ret = ShbtWebGpuEngine.__wrap(arg0);
            return ret;
        },
        __wbg_size_f31e7f807537ca60: function(arg0) {
            const ret = arg0.size;
            return ret;
        },
        __wbg_stack_3b0d974bbf31e44f: function(arg0, arg1) {
            const ret = arg1.stack;
            const ptr1 = passStringToWasm0(ret, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
            const len1 = WASM_VECTOR_LEN;
            getDataViewMemory0().setInt32(arg0 + 4 * 1, len1, true);
            getDataViewMemory0().setInt32(arg0 + 4 * 0, ptr1, true);
        },
        __wbg_static_accessor_GLOBAL_266715b9d96ba635: function() {
            const ret = typeof global === 'undefined' ? null : global;
            return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
        },
        __wbg_static_accessor_GLOBAL_THIS_10fb7dc1ae063179: function() {
            const ret = typeof globalThis === 'undefined' ? null : globalThis;
            return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
        },
        __wbg_static_accessor_SELF_0b583911f537483a: function() {
            const ret = typeof self === 'undefined' ? null : self;
            return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
        },
        __wbg_static_accessor_WINDOW_d7f903d1508cbdc4: function() {
            const ret = typeof window === 'undefined' ? null : window;
            return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
        },
        __wbg_submit_fa002a371d622a03: function(arg0, arg1) {
            arg0.submit(arg1);
        },
        __wbg_then_4e48da6c0a644c56: function(arg0, arg1, arg2) {
            const ret = arg0.then(arg1, arg2);
            return ret;
        },
        __wbg_then_c949d5a25a4e78f8: function(arg0, arg1, arg2) {
            const ret = arg0.then(arg1, arg2);
            return ret;
        },
        __wbg_then_e71170d78fcf8954: function(arg0, arg1) {
            const ret = arg0.then(arg1);
            return ret;
        },
        __wbg_unmap_fa9902769025442b: function(arg0) {
            arg0.unmap();
        },
        __wbg_usage_b39863f16cf0f3d0: function(arg0) {
            const ret = arg0.usage;
            return ret;
        },
        __wbg_valueOf_02fd4d101b7553c7: function(arg0) {
            const ret = arg0.valueOf();
            return ret;
        },
        __wbg_warn_13abc63e0d4b3527: function(arg0) {
            console.warn(arg0);
        },
        __wbg_width_3d0dce3d9892e35e: function(arg0) {
            const ret = arg0.width;
            return ret;
        },
        __wbg_writeBuffer_8ff0ce799fa73af6: function(arg0, arg1, arg2, arg3, arg4, arg5) {
            arg0.writeBuffer(arg1, arg2, arg3, arg4, arg5);
        },
        __wbg_writeTexture_0ff4670b961f7196: function(arg0, arg1, arg2, arg3, arg4) {
            arg0.writeTexture(arg1, arg2, arg3, arg4);
        },
        __wbg_writeTimestamp_3189accd0719f57d: function(arg0, arg1, arg2) {
            arg0.writeTimestamp(arg1, arg2 >>> 0);
        },
        __wbindgen_generic_0000000000000001: function(arg0, arg1) {
            // Cast intrinsic for `Closure(Closure { owned: true, function: Function { arguments: [Externref], shim_idx: 262, ret: Unit, inner_ret: Some(Unit) }, mutable: true }) -> Externref`.
            const ret = makeMutClosure(arg0, arg1, wasm_bindgen_740f87ab467470cf___convert__closures_____invoke___wasm_bindgen_740f87ab467470cf___JsValue______true_);
            return ret;
        },
        __wbindgen_generic_0000000000000002: function(arg0, arg1) {
            // Cast intrinsic for `Closure(Closure { owned: true, function: Function { arguments: [Externref], shim_idx: 289, ret: Result(Unit), inner_ret: Some(Result(Unit)) }, mutable: true }) -> Externref`.
            const ret = makeMutClosure(arg0, arg1, wasm_bindgen_740f87ab467470cf___convert__closures_____invoke___wasm_bindgen_740f87ab467470cf___JsValue__core_608f92abc48d28da___result__Result_____wasm_bindgen_740f87ab467470cf___JsError___true_);
            return ret;
        },
        __wbindgen_generic_0000000000000003: function(arg0, arg1) {
            // Cast intrinsic for `Closure(Closure { owned: true, function: Function { arguments: [NamedExternref("GPUUncapturedErrorEvent")], shim_idx: 262, ret: Unit, inner_ret: Some(Unit) }, mutable: true }) -> Externref`.
            const ret = makeMutClosure(arg0, arg1, wasm_bindgen_740f87ab467470cf___convert__closures_____invoke___wasm_bindgen_740f87ab467470cf___JsValue______true__54);
            return ret;
        },
        __wbindgen_generic_0000000000000004: function(arg0) {
            // Cast intrinsic for `F64 -> Externref`.
            const ret = arg0;
            return ret;
        },
        __wbindgen_generic_0000000000000005: function(arg0, arg1) {
            // Cast intrinsic for `Ref(Slice(U8)) -> NamedExternref("Uint8Array")`.
            const ret = getArrayU8FromWasm0(arg0, arg1);
            return ret;
        },
        __wbindgen_generic_0000000000000006: function(arg0, arg1) {
            // Cast intrinsic for `Ref(String) -> Externref`.
            const ret = getStringFromWasm0(arg0, arg1);
            return ret;
        },
        __wbindgen_init_externref_table: function() {
            const table = wasm.__wbindgen_externrefs;
            const offset = table.grow(4);
            table.set(0, undefined);
            table.set(offset + 0, undefined);
            table.set(offset + 1, null);
            table.set(offset + 2, true);
            table.set(offset + 3, false);
        },
    };
    return {
        __proto__: null,
        "./shbt_visualizer_bg.js": import0,
    };
}

function wasm_bindgen_740f87ab467470cf___convert__closures_____invoke___wasm_bindgen_740f87ab467470cf___JsValue______true_(arg0, arg1, arg2) {
    wasm.wasm_bindgen_740f87ab467470cf___convert__closures_____invoke___wasm_bindgen_740f87ab467470cf___JsValue______true_(arg0, arg1, arg2);
}

function wasm_bindgen_740f87ab467470cf___convert__closures_____invoke___wasm_bindgen_740f87ab467470cf___JsValue______true__54(arg0, arg1, arg2) {
    wasm.wasm_bindgen_740f87ab467470cf___convert__closures_____invoke___wasm_bindgen_740f87ab467470cf___JsValue______true__54(arg0, arg1, arg2);
}

function wasm_bindgen_740f87ab467470cf___convert__closures_____invoke___wasm_bindgen_740f87ab467470cf___JsValue__core_608f92abc48d28da___result__Result_____wasm_bindgen_740f87ab467470cf___JsError___true_(arg0, arg1, arg2) {
    const ret = wasm.wasm_bindgen_740f87ab467470cf___convert__closures_____invoke___wasm_bindgen_740f87ab467470cf___JsValue__core_608f92abc48d28da___result__Result_____wasm_bindgen_740f87ab467470cf___JsError___true_(arg0, arg1, arg2);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
}

function wasm_bindgen_740f87ab467470cf___convert__closures_____invoke___js_sys_f8d1592f528dc307___Function_fn_wasm_bindgen_740f87ab467470cf___JsValue_____wasm_bindgen_740f87ab467470cf___sys__Undefined___js_sys_f8d1592f528dc307___Function_fn_wasm_bindgen_740f87ab467470cf___JsValue_____wasm_bindgen_740f87ab467470cf___sys__Undefined_______true_(arg0, arg1, arg2, arg3) {
    wasm.wasm_bindgen_740f87ab467470cf___convert__closures_____invoke___js_sys_f8d1592f528dc307___Function_fn_wasm_bindgen_740f87ab467470cf___JsValue_____wasm_bindgen_740f87ab467470cf___sys__Undefined___js_sys_f8d1592f528dc307___Function_fn_wasm_bindgen_740f87ab467470cf___JsValue_____wasm_bindgen_740f87ab467470cf___sys__Undefined_______true_(arg0, arg1, arg2, arg3);
}


const __wbindgen_enum_GpuErrorFilter = ["validation", "out-of-memory", "internal"];


const __wbindgen_enum_GpuIndexFormat = ["uint16", "uint32"];


const __wbindgen_enum_GpuTextureFormat = ["r8unorm", "r8snorm", "r8uint", "r8sint", "r16uint", "r16sint", "r16float", "rg8unorm", "rg8snorm", "rg8uint", "rg8sint", "r32uint", "r32sint", "r32float", "rg16uint", "rg16sint", "rg16float", "rgba8unorm", "rgba8unorm-srgb", "rgba8snorm", "rgba8uint", "rgba8sint", "bgra8unorm", "bgra8unorm-srgb", "rgb9e5ufloat", "rgb10a2unorm", "rg11b10ufloat", "rg32uint", "rg32sint", "rg32float", "rgba16uint", "rgba16sint", "rgba16float", "rgba32uint", "rgba32sint", "rgba32float", "stencil8", "depth16unorm", "depth24plus", "depth24plus-stencil8", "depth32float", "depth32float-stencil8", "bc1-rgba-unorm", "bc1-rgba-unorm-srgb", "bc2-rgba-unorm", "bc2-rgba-unorm-srgb", "bc3-rgba-unorm", "bc3-rgba-unorm-srgb", "bc4-r-unorm", "bc4-r-snorm", "bc5-rg-unorm", "bc5-rg-snorm", "bc6h-rgb-ufloat", "bc6h-rgb-float", "bc7-rgba-unorm", "bc7-rgba-unorm-srgb", "etc2-rgb8unorm", "etc2-rgb8unorm-srgb", "etc2-rgb8a1unorm", "etc2-rgb8a1unorm-srgb", "etc2-rgba8unorm", "etc2-rgba8unorm-srgb", "eac-r11unorm", "eac-r11snorm", "eac-rg11unorm", "eac-rg11snorm", "astc-4x4-unorm", "astc-4x4-unorm-srgb", "astc-5x4-unorm", "astc-5x4-unorm-srgb", "astc-5x5-unorm", "astc-5x5-unorm-srgb", "astc-6x5-unorm", "astc-6x5-unorm-srgb", "astc-6x6-unorm", "astc-6x6-unorm-srgb", "astc-8x5-unorm", "astc-8x5-unorm-srgb", "astc-8x6-unorm", "astc-8x6-unorm-srgb", "astc-8x8-unorm", "astc-8x8-unorm-srgb", "astc-10x5-unorm", "astc-10x5-unorm-srgb", "astc-10x6-unorm", "astc-10x6-unorm-srgb", "astc-10x8-unorm", "astc-10x8-unorm-srgb", "astc-10x10-unorm", "astc-10x10-unorm-srgb", "astc-12x10-unorm", "astc-12x10-unorm-srgb", "astc-12x12-unorm", "astc-12x12-unorm-srgb"];
const ShbtWebGpuEngineFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_shbtwebgpuengine_free(ptr, 1));
const VisualizerEngineFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_visualizerengine_free(ptr, 1));
const VisualizerTelemetryFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_visualizertelemetry_free(ptr, 1));
const WasmShbtEngineFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_wasmshbtengine_free(ptr, 1));

function addToExternrefTable0(obj) {
    const idx = wasm.__externref_table_alloc();
    wasm.__wbindgen_externrefs.set(idx, obj);
    return idx;
}

const CLOSURE_DTORS = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(state => wasm.__wbindgen_destroy_closure(state.a, state.b));

function debugString(val) {
    // primitive types
    const type = typeof val;
    if (type == 'number' || type == 'boolean' || val == null) {
        return  `${val}`;
    }
    if (type == 'string') {
        return `"${val}"`;
    }
    if (type == 'symbol') {
        const description = val.description;
        if (description == null) {
            return 'Symbol';
        } else {
            return `Symbol(${description})`;
        }
    }
    if (type == 'function') {
        const name = val.name;
        if (typeof name == 'string' && name.length > 0) {
            return `Function(${name})`;
        } else {
            return 'Function';
        }
    }
    // objects
    if (Array.isArray(val)) {
        const length = val.length;
        let debug = '[';
        if (length > 0) {
            debug += debugString(val[0]);
        }
        for(let i = 1; i < length; i++) {
            debug += ', ' + debugString(val[i]);
        }
        debug += ']';
        return debug;
    }
    // Test for built-in
    const builtInMatches = /\[object ([^\]]+)\]/.exec(toString.call(val));
    let className;
    if (builtInMatches && builtInMatches.length > 1) {
        className = builtInMatches[1];
    } else {
        // Failed to match the standard '[object ClassName]'
        return toString.call(val);
    }
    if (className == 'Object') {
        // we're a user defined class or Object
        // JSON.stringify avoids problems with cycles, and is generally much
        // easier than looping through ownProperties of `val`.
        try {
            return 'Object(' + JSON.stringify(val) + ')';
        } catch (_) {
            return 'Object';
        }
    }
    // errors
    if (val instanceof Error) {
        return `${val.name}: ${val.message}\n${val.stack}`;
    }
    // TODO we could test for more things here, like `Set`s and `Map`s.
    return className;
}

function getArrayU32FromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return getUint32ArrayMemory0().subarray(ptr / 4, ptr / 4 + len);
}

function getArrayU8FromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return getUint8ArrayMemory0().subarray(ptr / 1, ptr / 1 + len);
}

let cachedDataViewMemory0 = null;
function getDataViewMemory0() {
    if (cachedDataViewMemory0 === null || cachedDataViewMemory0.buffer.detached === true || (cachedDataViewMemory0.buffer.detached === undefined && cachedDataViewMemory0.buffer !== wasm.memory.buffer)) {
        cachedDataViewMemory0 = new DataView(wasm.memory.buffer);
    }
    return cachedDataViewMemory0;
}

let cachedFloat32ArrayMemory0 = null;
function getFloat32ArrayMemory0() {
    if (cachedFloat32ArrayMemory0 === null || cachedFloat32ArrayMemory0.byteLength === 0) {
        cachedFloat32ArrayMemory0 = new Float32Array(wasm.memory.buffer);
    }
    return cachedFloat32ArrayMemory0;
}

function getStringFromWasm0(ptr, len) {
    return decodeText(ptr >>> 0, len);
}

let cachedUint32ArrayMemory0 = null;
function getUint32ArrayMemory0() {
    if (cachedUint32ArrayMemory0 === null || cachedUint32ArrayMemory0.byteLength === 0) {
        cachedUint32ArrayMemory0 = new Uint32Array(wasm.memory.buffer);
    }
    return cachedUint32ArrayMemory0;
}

let cachedUint8ArrayMemory0 = null;
function getUint8ArrayMemory0() {
    if (cachedUint8ArrayMemory0 === null || cachedUint8ArrayMemory0.byteLength === 0) {
        cachedUint8ArrayMemory0 = new Uint8Array(wasm.memory.buffer);
    }
    return cachedUint8ArrayMemory0;
}

function handleError(f, args) {
    try {
        return f.apply(this, args);
    } catch (e) {
        const idx = addToExternrefTable0(e);
        wasm.__wbindgen_exn_store(idx);
    }
}

function isLikeNone(x) {
    return x === undefined || x === null;
}

function makeMutClosure(arg0, arg1, f) {
    const state = { a: arg0, b: arg1, cnt: 1 };
    const real = (...args) => {

        // First up with a closure we increment the internal reference
        // count. This ensures that the Rust closure environment won't
        // be deallocated while we're invoking it.
        state.cnt++;
        const a = state.a;
        state.a = 0;
        try {
            return f(a, state.b, ...args);
        } finally {
            state.a = a;
            real._wbg_cb_unref();
        }
    };
    real._wbg_cb_unref = () => {
        if (--state.cnt === 0) {
            wasm.__wbindgen_destroy_closure(state.a, state.b);
            state.a = 0;
            CLOSURE_DTORS.unregister(state);
        }
    };
    CLOSURE_DTORS.register(real, state, state);
    return real;
}

function passArray8ToWasm0(arg, malloc) {
    const ptr = malloc(arg.length * 1, 1) >>> 0;
    getUint8ArrayMemory0().set(arg, ptr / 1);
    WASM_VECTOR_LEN = arg.length;
    return ptr;
}

function passArrayF32ToWasm0(arg, malloc) {
    const ptr = malloc(arg.length * 4, 4) >>> 0;
    getFloat32ArrayMemory0().set(arg, ptr / 4);
    WASM_VECTOR_LEN = arg.length;
    return ptr;
}

function passStringToWasm0(arg, malloc, realloc) {
    if (realloc === undefined) {
        const buf = cachedTextEncoder.encode(arg);
        const ptr = malloc(buf.length, 1) >>> 0;
        getUint8ArrayMemory0().subarray(ptr, ptr + buf.length).set(buf);
        WASM_VECTOR_LEN = buf.length;
        return ptr;
    }

    let len = arg.length;
    let ptr = malloc(len, 1) >>> 0;

    const mem = getUint8ArrayMemory0();

    let offset = 0;

    for (; offset < len; offset++) {
        const code = arg.charCodeAt(offset);
        if (code > 0x7F) break;
        mem[ptr + offset] = code;
    }
    if (offset !== len) {
        if (offset !== 0) {
            arg = arg.slice(offset);
        }
        ptr = realloc(ptr, len, len = offset + arg.length * 3, 1) >>> 0;
        const view = getUint8ArrayMemory0().subarray(ptr + offset, ptr + len);
        const ret = cachedTextEncoder.encodeInto(arg, view);

        offset += ret.written;
        ptr = realloc(ptr, len, offset, 1) >>> 0;
    }

    WASM_VECTOR_LEN = offset;
    return ptr;
}

function takeFromExternrefTable0(idx) {
    const value = wasm.__wbindgen_externrefs.get(idx);
    wasm.__externref_table_dealloc(idx);
    return value;
}

let cachedTextDecoder = new TextDecoder('utf-8', { ignoreBOM: true, fatal: true });
cachedTextDecoder.decode();
const MAX_SAFARI_DECODE_BYTES = 2146435072;
let numBytesDecoded = 0;
function decodeText(ptr, len) {
    numBytesDecoded += len;
    if (numBytesDecoded >= MAX_SAFARI_DECODE_BYTES) {
        cachedTextDecoder = new TextDecoder('utf-8', { ignoreBOM: true, fatal: true });
        cachedTextDecoder.decode();
        numBytesDecoded = len;
    }
    return cachedTextDecoder.decode(getUint8ArrayMemory0().subarray(ptr, ptr + len));
}

const cachedTextEncoder = new TextEncoder();

if (!('encodeInto' in cachedTextEncoder)) {
    cachedTextEncoder.encodeInto = function (arg, view) {
        const buf = cachedTextEncoder.encode(arg);
        view.set(buf);
        return {
            read: arg.length,
            written: buf.length
        };
    };
}

let WASM_VECTOR_LEN = 0;

let wasmModule, wasmInstance, wasm;
function __wbg_finalize_init(instance, module) {
    wasmInstance = instance;
    wasm = instance.exports;
    wasmModule = module;
    cachedDataViewMemory0 = null;
    cachedFloat32ArrayMemory0 = null;
    cachedUint32ArrayMemory0 = null;
    cachedUint8ArrayMemory0 = null;
    wasm.__wbindgen_start();
    return wasm;
}

async function __wbg_load(module, imports) {
    if (typeof Response === 'function' && module instanceof Response) {
        if (!module.ok) {
            throw new Error(`failed to fetch Wasm: ${module.status} ${module.statusText} fetching '${module.url}'`);
        }

        if (typeof WebAssembly.instantiateStreaming === 'function') {
            try {
                return await WebAssembly.instantiateStreaming(module, imports);
            } catch (e) {
                const validResponse = expectedResponseType(module.type);

                if (validResponse && module.headers.get('Content-Type') !== 'application/wasm') {
                    console.warn("`WebAssembly.instantiateStreaming` failed because your server does not serve Wasm with `application/wasm` MIME type. Falling back to `WebAssembly.instantiate` which is slower. Original error:\n", e);

                } else { throw e; }
            }
        }

        const bytes = await module.arrayBuffer();
        return await WebAssembly.instantiate(bytes, imports);
    } else {
        const instance = await WebAssembly.instantiate(module, imports);

        if (instance instanceof WebAssembly.Instance) {
            return { instance, module };
        } else {
            return instance;
        }
    }

    function expectedResponseType(type) {
        switch (type) {
            case 'basic': case 'cors': case 'default': return true;
        }
        return false;
    }
}

function initSync(module) {
    if (wasm !== undefined) return wasm;


    if (module !== undefined) {
        if (Object.getPrototypeOf(module) === Object.prototype) {
            ({module} = module)
        } else {
            console.warn('using deprecated parameters for `initSync()`; pass a single object instead')
        }
    }

    const imports = __wbg_get_imports();
    if (!(module instanceof WebAssembly.Module)) {
        module = new WebAssembly.Module(module);
    }
    const instance = new WebAssembly.Instance(module, imports);
    return __wbg_finalize_init(instance, module);
}

async function __wbg_init(module_or_path) {
    if (wasm !== undefined) return wasm;


    if (module_or_path !== undefined) {
        if (Object.getPrototypeOf(module_or_path) === Object.prototype) {
            ({module_or_path} = module_or_path)
        } else {
            console.warn('using deprecated parameters for the initialization function; pass a single object instead')
        }
    }

    if (module_or_path === undefined) {
        module_or_path = new URL('shbt_visualizer_bg.wasm', import.meta.url);
    }
    const imports = __wbg_get_imports();

    if (typeof module_or_path === 'string' || (typeof Request === 'function' && module_or_path instanceof Request) || (typeof URL === 'function' && module_or_path instanceof URL)) {
        module_or_path = fetch(module_or_path);
    }

    const { instance, module } = await __wbg_load(await module_or_path, imports);

    return __wbg_finalize_init(instance, module);
}

export { initSync, __wbg_init as default };
