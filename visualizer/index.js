// SHBT WebGPU visualizer harness.
// Loads the wasm engine bound to #shbt-canvas, drives the timeline scrub
// (z: 1e12 -> -1), playback speeds, projection switch and channel toggles,
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
  $("hud-hubble").textContent = `${fmt(m.hubble)} km/s/Mpc`;
  $("hud-ledger").textContent =
    `\u0394N = ${fmt(m.delta_n_bits)} bits\n` +
    `N_vis = ${fmt(m.n_vis)}  N_dark = ${fmt(m.n_dark)}\n` +
    `M_seed = ${fmt(m.seed_mass_msun)} M\u2609`;
  $("hud-debt").textContent = `${fmt(m.landauer_debt_gw)} GW`;
  $("bar-debt").style.width =
    `${Math.min(m.landauer_debt_gw / 1e21, 1) * 100}%`;
  $("inv-fr").classList.toggle("ok", m.delta_fr_zero);
  $("inv-emu").classList.toggle("ok", m.e_munu_zero);
  $("inv-horizon").classList.toggle("ok", m.horizon_frozen);
  $("zlabel").textContent = `z = ${fmt(m.z, 3)}  a = ${fmt(m.a, 3)}`;
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
      "timeline: z = 1e12 \u2192 \u22121"
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
});
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
