// SHBT WebGPU visualizer harness.
// Loads the wasm engine bound to #shbt-canvas, drives the timeline scrub
// (z: 1e14 -> -1), playback speeds, projection switch and channel toggles,
// and refreshes the HUD from the engine's SHBT-MMIO telemetry decode.

const $ = (id) => document.getElementById(id);
const status = (msg) => { $("status").textContent = msg; };

let engine = null;
let lastT = performance.now();

const fmt = (x, d = 3) => {
  if (!isFinite(x)) return "\u221e";
  const a = Math.abs(x);
  if (a !== 0 && (a >= 1e6 || a < 1e-3)) return x.toExponential(d);
  return x.toPrecision(d);
};

function zToSlider(z) {
  // slider x in [-12, 12] <-> z = 10^x - 1
  return Math.log10(Math.max(z + 1, 1e-12));
}
function sliderToZ(x) {
  return Math.pow(10, x) - 1;
}

function refreshHud(m) {
  window.__lastHudMetrics = m;
  refreshHudTrack(m);
  $("hud-time").textContent = `${fmt(m.t_gyr)} Gyr`;
  $("bar-time").style.width = `${Math.min(m.t_gyr / 13.8, 1) * 100}%`;
  $("hud-fload").textContent = fmt(m.f_load, 5);
  $("bar-fload").style.width = `${m.f_load * 100}%`;
  $("hud-capacity").textContent = `N_sat = 3.312e122 bits\u00b7 loaded = ${fmt(m.n_vis + m.n_dark)}`;
  $("hud-hubble").textContent = `${fmt(m.hubble)} km/s/Mpc`;
  const totalBits = m.n_vis + m.n_dark;
  const quenchPct = totalBits > 0 ? (100 * m.n_dark / totalBits).toFixed(1) : "0.0";
  $("hud-ledger").textContent =
    `\u0394N = ${fmt(m.delta_n_bits)} bits\n` +
    `N_vis = ${fmt(m.n_vis)}  N_dark = ${fmt(m.n_dark)}\n` +
    `Channel A quench = ${quenchPct}% (target 23/33 = 69.7%)\n` +
    `M_seed = ${fmt(m.seed_mass_msun)} M\u2609`;
  $("hud-debt").textContent = `${fmt(m.landauer_debt_gw)} GW`;
  // shbt10 recorder contract: machine-readable telemetry overlay
  // (asserted by record_simulation_events.py milestone keyframes).
  if ($("hud-telemetry-overlay")) {
    $("hud-redshift-val").textContent = fmt(m.z, 4);
    const qf = totalBits > 0 ? m.n_dark / totalBits : 0;
    $("hud-quench-fraction-val").textContent = (qf * 100).toFixed(2);
    const debt = m.landauerDebt ?? m.landauer_debt_gw ?? 0;
    $("hud-landauer-debt-val").textContent = fmt(debt, 4);
    $("hud-fps-val").textContent = (window.__fpsEma ?? 0).toFixed(1);
  }
  $("bar-debt").style.width =
    `${Math.min(m.landauer_debt_gw / 1e21, 1) * 100}%`;
  $("inv-fr").classList.toggle("ok", m.delta_fr_zero);
  $("inv-emu").classList.toggle("ok", m.e_munu_zero);
  $("inv-horizon").classList.toggle("ok", m.horizon_frozen);
  // Emergent defect table: live readout of the condensation kernels
  // (seedCount / totalMass / landauerDebt come from the GPU readback).
  const sc = m.seedCount ?? 0;
  $("seed-count").textContent = `${sc} seed${sc === 1 ? "" : "s"}`;
  $("seed-table").textContent = sc > 0
    ? `M_defect = ${fmt(m.totalMass)} M\u2609\n` +
      `P_Landauer = ${fmt(m.landauerDebt)} GW\n` +
      `condensation: emergent (\u03b4_eff \u2265 \u03b3 \u2212 1)`
    : `M_defect = 0\nP_Landauer = 0 GW\nregister sub-critical (N_local < N_limit)`;
  // Causal-point observer activity monitor: admissible observer set
  // cardinality R_adm and entropy-margin status.
  const rAdm = m.z <= -0.95 ? 0 : Math.round(Math.min(Math.max((1 + m.z) * 1024, 0), 1024));
  $("hud-observers").innerHTML = `R<sub>adm</sub> = ${rAdm}`;
  $("bar-observers").style.width = `${(rAdm / 1024) * 100}%`;
  $("hud-rentropy").innerHTML = rAdm > 0
    ? "R<sub>entropy</sub> = N<sub>limit</sub> &minus; C<sub>get</sub> &ge; 0 &mdash; GET active"
    : "R<sub>entropy</sub> &lt; 0 &mdash; observer set frozen (&empty;)";
  $("hud-optics").innerHTML =
    `&gamma;<sub>max</sub> = ${fmt(m.peak_shear)}  &kappa;<sub>max</sub> = ${fmt(m.peak_convergence)}\n` +
    `&theta;<sub>E</sub> = ${fmt(m.max_einstein_radius)} rad  caustics = ${m.active_caustics}`;
  $("zlabel").textContent = `z = ${fmt(m.z, 3)}  a = ${fmt(m.a, 3)}`;
  const banner = $("phase-banner");
  if (m.z <= -0.95) {
    banner.className = "ph-freeze";
    banner.innerHTML = "PHASE 5: ASYMPTOTIC DE SITTER OBSERVER FREEZE &nbsp;|&nbsp; <em>R</em><sub>adm</sub> &rarr; &empty; &nbsp;|&nbsp; <em>E</em><sub>&mu;&nu;</sub> = 0 &nbsp;|&nbsp; &Delta;<sub>fr</sub> = 0";
  } else if (m.z > 1e12) {
    banner.className = "ph-load";
    banner.innerHTML = "PHASE 1: CONFORMAL SCREEN BIT LOADING &nbsp;|&nbsp; Ṡ = <em>H</em>(<em>t</em>) &middot; <em>C</em><sub>max</sub> &nbsp;|&nbsp; <em>N</em><sub>sat</sub> = 3.312&times;10<sup>122</sup> bits";
  } else if (m.z >= 1e9) {
    banner.className = "ph-bary";
    banner.innerHTML = "PHASE 2: TOPOLOGICAL BARYOGENESIS &nbsp;|&nbsp; Stinespring De-Rendering 23/33 &nbsp;|&nbsp; &eta;<sub>B</sub> = 6.1&times;10<sup>&minus;10</sup>";
  } else if (m.z >= 7) {
    banner.className = "ph-seed";
    banner.innerHTML = "PHASE 3: TOPOLOGICAL GHOST SEED CONDENSATION &nbsp;|&nbsp; <em>M</em><sub>seed</sub> &asymp; 10<sup>9</sup> <em>M</em><sub>&#9737;</sub> &nbsp;|&nbsp; <em>K</em> = 312, &Delta;<em>N</em> = 6.0&times;10<sup>59</sup> bits";
  } else {
    banner.className = "ph-get";
    banner.innerHTML = "PHASE 4: CAUSAL POINT GET CLUSTERING &nbsp;|&nbsp; <strong>a</strong><sub>GET</sub> = &minus;&kappa;<sub>GET</sub> &nabla; ln &rho;<sub>proj</sub> &nbsp;|&nbsp; <em>R</em><sub>entropy</sub> &ge; 0";
  }
  if (document.activeElement !== $("timeline")) {
    $("timeline").value = zToSlider(m.z);
  }
}

function refreshHudTrack(m) {
  lastKnownZ = m.z;
}

async function boot() {
  if (!navigator.gpu) {
    status("WebGPU unavailable in this browser.\nUse Chrome/Edge >= 113 with WebGPU enabled.");
    return;
  }
  // Chrome >= 133 removed the legacy maxInterStageShaderComponents limit;
  // wgpu 0.19 still forwards it in requiredLimits, which aborts
  // requestDevice. Strip it before the adapter sees the descriptor.
  const origRequestDevice = GPUAdapter.prototype.requestDevice;
  GPUAdapter.prototype.requestDevice = function (descriptor) {
    if (descriptor && descriptor.requiredLimits) {
      delete descriptor.requiredLimits.maxInterStageShaderComponents;
    }
    return origRequestDevice.call(this, descriptor).then((device) => {
      device.addEventListener("uncapturederror", (e) =>
        console.error(`[webgpu] ${e.error.message}`)
      );
      return device;
    });
  };
  try {
    const wasm = await import("./pkg/shbt_visualizer.js");
    await wasm.default();
    engine = await wasm.ShbtWebGpuEngine.create("shbt-canvas");
    status(
      `engine online\nparticles: ${engine.particle_count().toLocaleString()}\n` +
      "timeline: z = 1e14 \u2192 \u22121"
    );
  } catch (e) {
    status(`engine init failed:\n${e}`);
    throw e;
  }
  // Software capture path: headless Chromium/SwiftShader cannot composite
  // the WebGPU canvas into screenshots or video, so ?capture=1 renders each
  // frame into an offscreen RGBA target, reads it back, and blits it into a
  // 2D overlay canvas that captures correctly. The WebGPU canvas stays live
  // underneath for interactive use without the flag.
  if (new URLSearchParams(location.search).get("capture") === "1") {
    // capture_frame_rgba holds &mut engine across an await; serialize every
    // mutating call through a queue so event handlers cannot re-enter it.
    const mutating = new Set([
      "step_frame", "capture_frame_rgba", "update_frame_telemetry",
      "set_redshift", "set_speed", "set_playing", "set_projection",
      "set_unwrap_transition", "set_channels", "set_lensing_enabled",
      "set_lensing_scale", "set_dispersion", "set_doppler_enabled",
      "set_dark_glow", "set_glitch_enabled", "set_glitch_intensity",
      "set_simulation_controls", "set_viewport_mode",
      "unproject_and_dispatch_causal_point",
      "update_cosmic_state", "clear_camera_override",
    ]);
    const raw = engine;
    engine = new Proxy(raw, {
      get(t, prop) {
        const v = t[prop];
        if (!mutating.has(prop)) return v;
        return (...args) => {
          const run = engineQueue.then(() => v.apply(t, args));
          engineQueue = run.catch((e) => console.error(`[engine queue] ${e}`));
          return run;
        };
      },
    });
    window.__engine = engine;
    const src = $("shbt-canvas");
    const overlay = document.createElement("canvas");
    overlay.id = "capture-canvas";
    overlay.width = src.width;
    overlay.height = src.height;
    overlay.style.cssText =
      "position:absolute;inset:0;width:100%;height:100%;z-index:0";
    src.parentElement.insertBefore(overlay, src.nextSibling);
    captureCtx = overlay.getContext("2d");
  }
  // Test hook (Phase 3 invariant gates): deterministic redshift scrub +
  // emergent-seed telemetry readout. Works in both interactive and
  // ?capture=1 (queued) modes.
  window.__SHBT_ENGINE__ = {
    setRedshift: (z) => engine.set_redshift(z),
    // In ?capture=1 mode hud_json can race capture_frame_rgba's &mut
    // borrow of the engine; fall back to the last metrics pushed into the
    // HUD in that case.
    getTelemetry: () => {
      try {
        return JSON.parse(engine.hud_json());
      } catch {
        return window.__lastHudMetrics ?? null;
      }
    },
    wasmMemory: () => engine.memory ? engine.memory.buffer.byteLength : 0,
    debugEparams: () => JSON.parse(engine.debug_eparams()),
    // Seed-glitch controls (Enhancement 11 refinement). The enabled
    // flag rides the hud_json telemetry frame (glitchEnabled) so reads
    // cannot race capture_frame_rgba's &mut borrow.
    setGlitchEnabled: (b) => engine.set_glitch_enabled(b),
    setGlitchIntensity: (v) => engine.set_glitch_intensity(v),
    setPlaying: (b) => engine.set_playing(b),
    // shbt9 sandbox controls + click-to-measure dispatch.
    setSimulationControls: (cs, pt, ls, cd, tz, split) =>
      engine.set_simulation_controls(cs, pt, ls, cd, tz, split),
    setViewportMode: (m) => engine.set_viewport_mode(m),
    dispatchCausalPoint: (nx, ny, cGet, nLimit) =>
      engine.unproject_and_dispatch_causal_point(nx, ny, cGet, nLimit),
    getActiveObservers: () => engine.get_active_observers_count(),
    // shbt10 cinematic director: atomic (z, eye, look-at) state commit.
    updateCosmicState: (z, pos, look) =>
      engine.update_cosmic_state(z, pos[0], pos[1], pos[2], look[0], look[1], look[2]),
    // Spec-verbatim alias (shbt10 record_simulation_events.py calls
    // __SHBT_ENGINE__.update_cosmic_state snake_case).
    update_cosmic_state: (z, pos, look) =>
      engine.update_cosmic_state(z, pos[0], pos[1], pos[2], look[0], look[1], look[2]),
    clearCameraOverride: () => engine.clear_camera_override(),
  };
  requestAnimationFrame(frame);
}

let captureCtx = null;
let captureBusy = false;
let engineQueue = Promise.resolve();

// Log-redshift scrub easing (shbt7 Phase 2): slider input sets a target
// epoch and the engine redshift is advanced toward it each frame in
// ln(1+z) space, so drags across decades land smoothly instead of
// snapping the camera through abrupt epoch jumps.
let zTarget = null;
let lastKnownZ = null;

function easeRedshift(dt) {
  if (zTarget === null || !engine) return;
  const cur = lastKnownZ ?? zTarget;
  const a = Math.log1p(Math.max(cur, -0.999));
  const b = Math.log1p(Math.max(zTarget, -0.999));
  if (Math.abs(b - a) < 1e-3) {
    engine.set_redshift(zTarget);
    zTarget = null;
    return;
  }
  const k = Math.min(1.0, dt * 7.0); // ~140 ms exponential approach
  engine.set_redshift(Math.expm1(a + (b - a) * k));
}

function frame(now) {
  const dt = Math.min((now - lastT) / 1000, 0.1);
  lastT = now;
  if (dt > 0) {
    const inst = 1.0 / dt;
    window.__fpsEma = window.__fpsEma === undefined ? inst : window.__fpsEma * 0.9 + inst * 0.1;
  }
  easeRedshift(dt);
  if (captureCtx) {
    if (!captureBusy) {
      captureBusy = true;
      engine
        .capture_frame_rgba(dt)
        .then((px) => {
          const c = captureCtx.canvas;
          captureCtx.putImageData(
            new ImageData(new Uint8ClampedArray(px), c.width, c.height),
            0, 0
          );
          refreshHud(JSON.parse(engine.hud_json()));
        })
        .catch((e) => status(`capture error:\n${e}`))
        .finally(() => {
          captureBusy = false;
        });
    }
    requestAnimationFrame(frame);
    return;
  }
  try {
    engine.step_frame(dt);
    refreshHud(JSON.parse(engine.hud_json()));
  } catch (e) {
    status(`frame error:\n${e}`);
    return;
  }
  requestAnimationFrame(frame);
}

$("timeline").addEventListener("input", (ev) => {
  if (engine) zTarget = sliderToZ(parseFloat(ev.target.value));
});
$("play").addEventListener("click", () => {
  if (!engine) return;
  const playing = $("play").dataset.on !== "1";
  $("play").dataset.on = playing ? "1" : "0";
  $("play").innerHTML = playing ? "&#9208;" : "&#9654;";
  engine.set_playing(playing);
});
document.querySelectorAll(".speed").forEach((b) =>
  b.addEventListener("click", () => {
    document.querySelectorAll(".speed").forEach((x) => x.classList.remove("active"));
    b.classList.add("active");
    if (engine) engine.set_speed(parseFloat(b.dataset.s));
  })
);
$("projection").addEventListener("change", (ev) => {
  if (engine) engine.set_projection(parseInt(ev.target.value, 10));
  const u = parseInt(ev.target.value, 10) === 1 ? 1 : 0;
  $("unwrap").value = u;
  $("unwrap-label").textContent = u.toFixed(2);
});
$("unwrap").addEventListener("input", (ev) => {
  const u = parseFloat(ev.target.value);
  $("unwrap-label").textContent = u.toFixed(2);
  if (engine && engine.set_unwrap_transition) engine.set_unwrap_transition(u);
});
const EPOCHS = {
  load: { z: 1e14, speed: 100, playing: true },
  bary: { z: 1e11, speed: 10, playing: true },
  seed: { z: 18.0, speed: 1, playing: true },
  get: { z: 3.0, speed: 1, playing: true },
  freeze: { z: -0.999, speed: 1, playing: false },
};
document.querySelectorAll("#epoch-bar button").forEach((b) =>
  b.addEventListener("click", () => {
    if (!engine) return;
    const e = EPOCHS[b.dataset.epoch];
    if (!e) return;
    zTarget = null; // epoch jumps are instant; cancel any pending ease
    engine.set_redshift(e.z);
    engine.set_speed(e.speed);
    engine.set_playing(e.playing);
    $("play").dataset.on = e.playing ? "1" : "0";
    $("play").innerHTML = e.playing ? "&#9208;" : "&#9654;";
    document.querySelectorAll(".speed").forEach((x) =>
      x.classList.toggle("active", parseFloat(x.dataset.s) === e.speed));
  })
);
$("ch-a").addEventListener("change", syncChannels);
$("ch-b").addEventListener("change", syncChannels);
function syncChannels() {
  if (engine) engine.set_channels($("ch-a").checked, $("ch-b").checked);
}

// Gravitational optics controls: lensing toggle/scale, wave-optics
// dispersion, Doppler beaming, and dark-matter halo glow.
function applyOptics() {
  if (!engine) return;
  if ($("toggle-lensing").checked) {
    engine.set_lensing_scale(parseFloat($("slider-strength").value));
  } else {
    engine.set_lensing_enabled(false);
  }
  engine.set_dispersion(parseFloat($("slider-dispersion").value));
  engine.set_doppler_enabled($("toggle-doppler").checked);
  engine.set_dark_glow(
    $("toggle-glow").checked ? parseFloat($("slider-dark-glow").value) : 0.0
  );
}
$("toggle-lensing").addEventListener("change", () => {
  if (engine && $("toggle-lensing").checked) engine.set_lensing_enabled(true);
  applyOptics();
});
$("slider-strength").addEventListener("input", (ev) => {
  $("strength-label").textContent = parseFloat(ev.target.value).toFixed(2);
  applyOptics();
});
$("slider-dispersion").addEventListener("input", (ev) => {
  $("dispersion-label").textContent = parseFloat(ev.target.value).toFixed(2);
  applyOptics();
});
$("toggle-doppler").addEventListener("change", applyOptics);
$("toggle-glow").addEventListener("change", applyOptics);
$("slider-dark-glow").addEventListener("input", (ev) => {
  $("glow-label").textContent = parseFloat(ev.target.value).toFixed(2);
  applyOptics();
});

// Emergent seed-glitch controls (Enhancement 11 refinement): toggle
// plus master intensity, live-wired to the Wasm engine.
function applyGlitch() {
  if (!engine) return;
  engine.set_glitch_enabled($("glitch-toggle").checked);
  engine.set_glitch_intensity(parseFloat($("glitch-slider").value));
}
$("glitch-toggle").addEventListener("change", applyGlitch);
$("glitch-slider").addEventListener("input", (ev) => {
  $("glitch-label").textContent = parseFloat(ev.target.value).toFixed(2);
  applyGlitch();
});

// shbt9 Phase 2 sandbox controls: P-PM sound-speed scale, percolation
// threshold, lensing strength, chromatic dispersion, target redshift,
// and the split-viewport diagnostic mode.
function applySandboxControls() {
  if (!engine || !engine.set_simulation_controls) return;
  engine.set_simulation_controls(
    parseFloat($("slider-cs").value),
    parseFloat($("slider-pt").value),
    parseFloat($("slider-ls").value),
    parseFloat($("slider-cd").value),
    sliderToZ(parseFloat($("slider-z").value)),
    parseInt($("select-viewport-mode").value, 10),
  );
}
[["slider-cs", "cs-label"], ["slider-pt", "pt-label"],
 ["slider-ls", "ls-label"], ["slider-cd", "cd-label"]].forEach(([sid, lid]) => {
  $(sid).addEventListener("input", (ev) => {
    $(lid).textContent = parseFloat(ev.target.value).toFixed(2);
    applySandboxControls();
  });
});
$("slider-z").addEventListener("input", applySandboxControls);
$("select-viewport-mode").addEventListener("change", (ev) => {
  if (engine && engine.set_viewport_mode) {
    engine.set_viewport_mode(parseInt(ev.target.value, 10));
  }
});

// Click-to-measure (shbt9 Phase 2): unproject the click into the bulk
// box and dispatch a causal-point observer with entropy budget
// R_entropy = N_limit - C_get.
$("shbt-canvas").addEventListener("pointerdown", (ev) => {
  if (!engine || !engine.unproject_and_dispatch_causal_point) return;
  const c = ev.target;
  const rect = c.getBoundingClientRect();
  const ndcX = ((ev.clientX - rect.left) / rect.width) * 2 - 1;
  const ndcY = 1 - ((ev.clientY - rect.top) / rect.height) * 2;
  const N_LIMIT = 8.0, C_GET = 1.0;
  const ok = engine.unproject_and_dispatch_causal_point(ndcX, ndcY, C_GET, N_LIMIT);
  if (ok && engine.get_active_observers_count) {
    $("observer-count").textContent =
      `Observers: ${engine.get_active_observers_count()}`;
  }
});

// Keyboard shortcuts: L lensing, D doppler, G dark glow,
// [ / ] lensing strength, - / = dispersion, H toggle HUD.
document.addEventListener("keydown", (ev) => {
  if (ev.target.tagName === "INPUT" || ev.target.tagName === "SELECT") return;
  const step = (id, d) => {
    const s = $(id);
    s.value = Math.min(Math.max(parseFloat(s.value) + d, parseFloat(s.min)), parseFloat(s.max));
    s.dispatchEvent(new Event("input"));
  };
  switch (ev.key.toLowerCase()) {
    case "l": $("toggle-lensing").checked = !$("toggle-lensing").checked; $("toggle-lensing").dispatchEvent(new Event("change")); break;
    case "d": $("toggle-doppler").checked = !$("toggle-doppler").checked; applyOptics(); break;
    case "g": $("toggle-glow").checked = !$("toggle-glow").checked; applyOptics(); break;
    case "[": step("slider-strength", -0.1); break;
    case "]": step("slider-strength", 0.1); break;
    case "-": step("slider-dispersion", -0.05); break;
    case "=": step("slider-dispersion", 0.05); break;
    case "h": {
      const hud = $("hud");
      hud.style.display = hud.style.display === "none" ? "" : "none";
      break;
    }
  }
});

window.addEventListener("resize", () => {
  const c = $("shbt-canvas");
  c.width = c.clientWidth;
  c.height = c.clientHeight;
});

boot();
