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
  $("bar-debt").style.width =
    `${Math.min(m.landauer_debt_gw / 1e21, 1) * 100}%`;
  $("inv-fr").classList.toggle("ok", m.delta_fr_zero);
  $("inv-emu").classList.toggle("ok", m.e_munu_zero);
  $("inv-horizon").classList.toggle("ok", m.horizon_frozen);
  // Causal-point observer activity monitor: admissible observer set
  // cardinality R_adm and entropy-margin status.
  const rAdm = m.z <= -0.95 ? 0 : Math.round(Math.min(Math.max((1 + m.z) * 1024, 0), 1024));
  $("hud-observers").innerHTML = `R<sub>adm</sub> = ${rAdm}`;
  $("bar-observers").style.width = `${(rAdm / 1024) * 100}%`;
  $("hud-rentropy").innerHTML = rAdm > 0
    ? "R<sub>entropy</sub> = N<sub>limit</sub> &minus; C<sub>get</sub> &ge; 0 &mdash; GET active"
    : "R<sub>entropy</sub> &lt; 0 &mdash; observer set frozen (&empty;)";
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

async function boot() {
  if (!navigator.gpu) {
    status("WebGPU unavailable in this browser.\nUse Chrome/Edge >= 113 with WebGPU enabled.");
    return;
  }
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
  requestAnimationFrame(frame);
}

function frame(now) {
  const dt = Math.min((now - lastT) / 1000, 0.1);
  lastT = now;
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
  if (engine) engine.set_redshift(sliderToZ(parseFloat(ev.target.value)));
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

window.addEventListener("resize", () => {
  const c = $("shbt-canvas");
  c.width = c.clientWidth;
  c.height = c.clientHeight;
});

boot();
