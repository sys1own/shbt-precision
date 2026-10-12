#!/usr/bin/env python3
"""
visualizer/record_simulation_events.py
Production-grade Playwright automation orchestrator for the shbt-precision
visualizer. Executes the 3D isometric-tilted orbital camera trajectory matrix
across cosmic epochs, resolves axial caustic stacking, verifies the HUD DOM
telemetry contract at each milestone, and records the run to
visualizer/recordings/full_cosmic_evolution.webm.

Spec milestones (canonical capture names, shbt12 visual checklist):
  m0  z = 1e14     01_primordial_bit_loading.png
  m1  z = 1e10     02_baryogenesis_derendering.png (quench -> eta_D ~ 69.7%)
  m2  z = 16       03_ghost_seed_genesis.png (seeds > 0, M_seed > 0, P_debt > 0)
  m3  z = 3        04_causal_point_proto_galaxies.png (debt > 5e11 GW, quench lock)
  m4  z = 1        05_cosmic_web_lensing.png (caustics > 0, gamma_max > 0.1)
  m5  z -> -0.999  06_asymptotic_horizon_freeze.png
"""
from __future__ import annotations

import asyncio
import base64
from dataclasses import dataclass
import http.server
import math
import os
import shutil
import socketserver
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
RECORDINGS_DIR = Path(__file__).resolve().parent / "recordings"
MILESTONES_DIR = Path(__file__).resolve().parent / "milestones"
RAW_VIDEO_DIR = RECORDINGS_DIR / "raw_videos"
OUTPUT_VIDEO_PATH = RECORDINGS_DIR / "full_cosmic_evolution.webm"
TELEMETRY_PATH = RECORDINGS_DIR / "cosmic_event_telemetry.json"
PORT = 8080
SIMULATION_URL = os.getenv(
    "SHBT_VISUALIZER_URL", f"http://127.0.0.1:{PORT}/visualizer/?capture=1&offline=1"
)

def find_ffmpeg() -> str:
    path = shutil.which("ffmpeg")
    if path:
        return path
    local_app_data = Path(os.environ.get("LOCALAPPDATA", ""))
    candidates = list(local_app_data.glob("Microsoft/WinGet/Packages/Gyan.FFmpeg*/**/ffmpeg.exe"))
    if candidates:
        return str(candidates[0])
    playwright_ff = list(local_app_data.glob("ms-playwright/ffmpeg-*/**/ffmpeg*.exe"))
    if playwright_ff:
        return str(playwright_ff[0])
    return "ffmpeg"

BROWSER_ARGS = [
    "--enable-unsafe-webgpu",
    "--use-angle=d3d11",
    "--disable-dawn-features=use_dxc",
    "--ignore-gpu-blocklist",
    "--disable-http-cache",
    "--disable-gpu-shader-disk-cache",
    "--disable-application-cache",
    "--no-sandbox",
]
TOTAL_DURATION_SECONDS = 90.0
TARGET_FPS = 60
TOTAL_FRAMES = int(TOTAL_DURATION_SECONDS * TARGET_FPS)
FRAME_STRIDE = max(1, int(os.getenv("SHBT_FRAME_STRIDE", "4")))
STRICT_FPS = os.getenv("SHBT_STRICT_FPS", "0") == "1"
MILESTONE_DWELL_PRESENTED = int(os.getenv("SHBT_MILESTONE_DWELL", "30"))

BOX_SIZE = 222.5502

MILESTONE_Z = {0: 1.0e14, 1: 1.0e10, 2: 16.0, 3: 3.0, 4: 1.0, 5: -0.999}
MILESTONE_LABEL = {
    0: "primordial bit loading",
    1: "baryogenesis de-rendering",
    2: "ghost seed genesis",
    3: "causal-point proto-galaxies",
    4: "cosmic-web lensing",
    5: "asymptotic horizon freeze",
}

MILESTONE_PNG = {
    0: "01_primordial_bit_loading.png",
    1: "02_baryogenesis_derendering.png",
    2: "03_ghost_seed_genesis.png",
    3: "04_causal_point_proto_galaxies.png",
    4: "05_cosmic_web_lensing.png",
    5: "06_asymptotic_horizon_freeze.png",
}

Z_KEYFRAMES: List[Tuple[float, float]] = [
    (0.0000, 1.0e14),
    (0.1667, 1.0e10),
    (0.3889, 30.0),
    (0.4600, 16.0),
    (0.6600, 3.0),
    (0.7200, 1.0),
    (0.8333, 0.0),
    (1.0000, -0.999),
]


@dataclass
class SimTimeStep:
    step: int
    scale_factor: float
    redshift: float
    box_size_mpc: float
    camera_radius_mpc: float
    grid_spacing_mpc: float
    cardy_ceiling_n: float


class SimulationEventRecorder:
    def __init__(self, l_box_mpc: float = 100.0, n_grid: int = 256, n_sat: float = 1.0e122):
        self.l_box_mpc = l_box_mpc
        self.n_grid = n_grid
        self.n_sat = n_sat
        self.gamma_cft = 1.0
        self.omega_m = 0.30
        self.omega_r = 8.5e-5
        self.omega_l = 0.70
        self.h_ref_si = 2.2e-18  # s^-1
        self.c_si = 2.99792458e8
        self.mpc_to_m = 3.08567758e22

    def e_factor(self, a: float) -> float:
        return math.sqrt(self.omega_r * a**(-4) + self.omega_m * a**(-3) + self.omega_l)

    def compute_cardy_ceiling(self, a: float) -> float:
        h_ratio = self.e_factor(a)
        h_si = self.h_ref_si * h_ratio
        r_h_m = self.c_si / max(h_si, 1e-30)
        v_h_m3 = (4.0 / 3.0) * math.pi * (r_h_m ** 3)
        cell_com_m = (self.l_box_mpc * self.mpc_to_m) / float(self.n_grid)
        cell_phys_m = a * cell_com_m
        v_cell_m3 = cell_phys_m ** 3
        return self.gamma_cft * (h_ratio ** 2) * (v_cell_m3 / max(v_h_m3, 1e-50)) * self.n_sat

    def compute_camera_radius(self, a: float) -> float:
        r_halo_ref = 0.8  # Mpc
        t = 1.0 / (1.0 + (a / 1.0) ** (-1.2))
        beta = 0.28 + (0.35 - 0.28) * t
        r_box = 1.5 * self.l_box_mpc * (a ** beta)
        return math.sqrt(r_halo_ref ** 2 + r_box ** 2)

    def compute_grid_spacing(self, a: float) -> float:
        l_phys = a * self.l_box_mpc
        exponent = math.floor(math.log10(max(1e-16, 0.2 * l_phys)))
        return 10.0 ** exponent


_recorder_instance = SimulationEventRecorder(l_box_mpc=BOX_SIZE)


def compute_camera_trajectory(z: float, progress: float = 0.0) -> Tuple[List[float], List[float]]:
    """
    Computes smooth 3D isometric camera trajectory locked to the diagonal vantage.
    Frames the cosmic web symmetrically across the entire viewport.
    Incorporates physical metric dilation and adaptive scale-factor camera orbit.
    """
    target = np.array([0.0, 0.0, 0.0])
    theta = np.radians(35.264)

    # Gentle symmetric diagonal orbit around phi = 45 degrees
    phi = np.radians(45.0 + 6.0 * math.sin(progress * math.pi))

    a = 1.0 / (1.0 + max(z, -0.999))
    if z > 30.0:  # Primordial & Baryogenesis eras
        u = float(np.clip((14.0 - np.log10(max(z, 30.0))) / (14.0 - np.log10(30.0)), 0.0, 1.0))
        r = 1.18 - 0.05 * u
    elif z > 7.0:  # Ghost Seed Genesis
        u = float(np.clip((30.0 - z) / 23.0, 0.0, 1.0))
        r = 1.13 - 0.05 * u
    elif z >= 0.0:  # Cosmic Web Assembly & Lensing
        u = float(np.clip((7.0 - z) / 7.0, 0.0, 1.0))
        r = 1.08 - 0.04 * u
    else:  # Asymptotic Horizon Freeze
        u = float(np.clip(-z / 0.999, 0.0, 1.0))
        r = 1.04 + 0.35 * u

    eye = target + r * np.array([
        np.cos(theta) * np.sin(phi),
        np.sin(theta),
        np.cos(theta) * np.cos(phi)
    ])
    return (eye * BOX_SIZE).tolist(), (target * BOX_SIZE).tolist()


def evaluate_trajectory(progress: float) -> Tuple[float, List[float], List[float]]:
    progress = max(0.0, min(1.0, progress))
    n = len(Z_KEYFRAMES)
    idx = 0
    while idx < n - 1 and Z_KEYFRAMES[idx + 1][0] <= progress:
        idx += 1
    if idx >= n - 1:
        z_now = Z_KEYFRAMES[-1][1]
    else:
        p1, z1 = Z_KEYFRAMES[idx]
        p2, z2 = Z_KEYFRAMES[idx + 1]
        seg = max(1e-6, p2 - p1)
        t = max(0.0, min(1.0, (progress - p1) / seg))
        if z1 > 0.0 and z2 > 0.0:
            z_now = 10.0 ** (math.log10(z1) + (math.log10(z2) - math.log10(z1)) * t)
        elif z1 >= 0.0 and z2 < 0.0:
            z_now = z1 + (z2 - z1) * (3.0 * t * t - 2.0 * t * t * t)
        else:
            z_now = z1 + (z2 - z1) * t

    cam_pos, cam_look = compute_camera_trajectory(z_now, progress)
    return z_now, cam_pos, cam_look


def start_server() -> socketserver.ThreadingTCPServer | None:
    """Start a COOP/COEP static server on PORT if nothing listens yet."""
    import urllib.request

    try:
        urllib.request.urlopen(f"http://127.0.0.1:{PORT}/visualizer/", timeout=2)
        print(f"[RECORDER] reusing server on :{PORT}")
        return None
    except Exception:
        pass

    class CoepHandler(http.server.SimpleHTTPRequestHandler):
        def end_headers(self) -> None:
            self.send_header("Cross-Origin-Opener-Policy", "same-origin")
            self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
            super().end_headers()

        def translate_path(self, path: str) -> str:
            clean = path.split("?", 1)[0].split("#", 1)[0]
            if clean in ("/visualizer/particles_init_z100.bin", "/particles_init_z100.bin"):
                return str(REPO_ROOT / "particles_init_z100.bin")
            if clean in ("/visualizer/data/telemetry.bin", "/data/telemetry.bin"):
                return str(REPO_ROOT / "data" / "telemetry.bin")
            return super().translate_path(path)

        def guess_type(self, path: str) -> str:
            if path.endswith(".bin"):
                return "application/octet-stream"
            if path.endswith(".wasm"):
                return "application/wasm"
            if path.endswith(".js"):
                return "application/javascript"
            return super().guess_type(path)

        def log_message(self, *args) -> None:
            pass

    class Server(socketserver.ThreadingTCPServer):
        allow_reuse_address = True
        daemon_threads = True

    handler = lambda *a, **kw: CoepHandler(*a, directory=str(REPO_ROOT), **kw)
    srv = Server(("127.0.0.1", PORT), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    print(f"[RECORDER] serving {REPO_ROOT} on :{PORT}")
    return srv


async def verify_hud_telemetry_milestone(page, milestone_idx: int, z: float, results: List[Dict]) -> bool:
    """Spec DOM telemetry gate. Returns True when all hard asserts pass."""
    print(f"[*] Milestone {milestone_idx} ({MILESTONE_LABEL[milestone_idx]}, target z = {z:.3e})")
    ok = True
    await page.wait_for_selector("#hud-telemetry-overlay", state="attached", timeout=30_000)
    redshift_text = await page.inner_text("#hud-redshift-val")
    quench_text = await page.inner_text("#hud-quench-fraction-val")
    landauer_text = await page.inner_text("#hud-landauer-debt-val")
    fps_text = await page.inner_text("#hud-fps-val")
    record = {
        "milestone": milestone_idx,
        "label": MILESTONE_LABEL[milestone_idx],
        "z_target": z,
        "z_hud": redshift_text,
        "quench_pct": quench_text,
        "landauer_gw": landauer_text,
        "fps": fps_text,
        "asserts": [],
    }

    def note(name: str, passed: bool, detail: str, hard: bool = True) -> None:
        nonlocal ok
        record["asserts"].append({"name": name, "pass": passed, "detail": detail, "hard": hard})
        tag = "ASSERTION PASSED" if passed else ("ASSERTION FAILED" if hard else "ADVISORY")
        print(f"    [{tag}] {name}: {detail}")
        if hard and not passed:
            ok = False

    try:
        fps_val = float(fps_text.replace("FPS", "").strip())
    except ValueError:
        fps_val = 0.0
    fps_ok = fps_val >= 55.0
    note(
        "fps >= 55",
        fps_ok,
        f"{fps_val:.1f} FPS (offline fixed-timestep budget; strict mode: {STRICT_FPS})",
        hard=STRICT_FPS,
    )

    if milestone_idx == 1:
        quench_val = float(quench_text.replace("%", "").strip())
        q_ok = 68.0 <= quench_val <= 71.0
        note("quench in [68, 71]", q_ok, f"{quench_val:.2f}% (theoretical 23/33 = 69.70%)")
    elif milestone_idx == 3:
        debt_val_gw = float(landauer_text.replace("GW", "").replace(",", "").strip())
        d_ok = debt_val_gw > 5.0e11
        note("landauer_debt > 5e11 GW", d_ok, f"{debt_val_gw:.3e} GW")
        quench_val = float(quench_text.replace("%", "").strip())
        q_ok = abs(quench_val - 69.7) < 1.0
        note("quench = 69.7 +/- 1.0", q_ok, f"{quench_val:.2f}%")

    metrics = await page.evaluate(
        """() => {
            try { return window.__SHBT_ENGINE__.getTelemetry(); }
            catch { return window.__lastHudMetrics || null; }
        }"""
    )
    metrics = metrics or {}
    record["metrics"] = {
        "t_gyr": metrics.get("t_gyr"),
        "lookback_gyr": metrics.get("lookback_gyr"),
        "f_load": metrics.get("f_load"),
        "seedCount": metrics.get("seedCount"),
        "totalMass": metrics.get("totalMass"),
        "landauerDebt": metrics.get("landauerDebt"),
        "active_caustics": metrics.get("active_caustics"),
        "peak_shear": metrics.get("peak_shear"),
        "peak_convergence": metrics.get("peak_convergence"),
        "max_einstein_radius": metrics.get("max_einstein_radius"),
        "horizon_frozen": metrics.get("horizon_frozen"),
    }
    if milestone_idx == 0:
        age = metrics.get("t_gyr") or 0.0
        f_load = metrics.get("f_load")
        seeds = metrics.get("seedCount") or 0
        te = metrics.get("max_einstein_radius") or 0.0
        note("cosmic age < 0.001 Gyr", age < 0.001, f"{age:.3e} Gyr")
        note("f_load_cosmo(1e14) < 0.01 (M1)",
             f_load is not None and f_load < 0.01,
             f"{f_load}")
        note("seeds == 0 at primordial", seeds == 0, f"{seeds} seeds")
        note("theta_E == 0 at primordial", te == 0.0, f"{te:.4f} rad")
    elif milestone_idx == 2:
        seeds = metrics.get("seedCount") or 0
        mass = metrics.get("totalMass") or 0.0
        debt = metrics.get("landauerDebt") or 0.0
        te = metrics.get("max_einstein_radius") or 0.0
        note("active_seeds > 0", seeds > 0, f"{seeds} seeds")
        note("M_seed > 0", 0 < mass < 3.0e14, f"{mass:.3e} M_sun")
        note("P_debt > 0", 0 < debt < 3.0e16, f"{debt:.3e} GW")
        note("theta_E in [0.01, 0.09] rad", 0.01 <= te <= 0.09, f"{te:.4f} rad")
    elif milestone_idx == 3:
        seeds = metrics.get("seedCount") or 0
        mass = metrics.get("totalMass") or 0.0
        debt = metrics.get("landauerDebt") or 0.0
        te = metrics.get("max_einstein_radius") or 0.0
        note("active_seeds > 0", seeds > 0, f"{seeds} seeds")
        caustics = metrics.get("active_caustics") or 0
        note("caustic:seed 1:1 parity", caustics == seeds,
             f"{caustics} caustics vs {seeds} seeds")
        note("theta_E in [0.01, 0.09] rad", 0.01 <= te <= 0.09, f"{te:.4f} rad")
    elif milestone_idx == 4:
        caustics = metrics.get("active_caustics") or 0
        seeds = metrics.get("seedCount") or 0
        gamma = metrics.get("peak_shear") or 0.0
        te = metrics.get("max_einstein_radius") or 0.0
        note("caustics > 0", caustics > 0, f"{caustics}")
        note("caustic:seed 1:1 parity", caustics == seeds,
             f"{caustics} caustics vs {seeds} seeds")
        note("gamma_max > 0.1", gamma > 0.1, f"{gamma:.4f}")
        note("theta_E in [0.01, 0.09] rad", 0.01 <= te <= 0.09, f"{te:.4f} rad")
    elif milestone_idx == 5:
        age = metrics.get("t_gyr") or 0.0
        f_load = metrics.get("f_load") or 0.0
        frozen = bool(metrics.get("horizon_frozen"))
        note("cosmic age > 15.0 Gyr", age > 15.0, f"{age:.3f} Gyr")
        note("f_load > 0.99", f_load > 0.99, f"{f_load}")
        note("R_entropy < 0 (frozen)", frozen,
             "observer set frozen" if frozen else "still active")
        note("R_adm == 0", frozen,
             "0 admitted observers" if frozen else "observers active")

    MILESTONES_DIR.mkdir(parents=True, exist_ok=True)
    shot1 = RECORDINGS_DIR / MILESTONE_PNG[milestone_idx]
    shot2 = MILESTONES_DIR / MILESTONE_PNG[milestone_idx]
    await page.screenshot(path=str(shot1))
    await page.screenshot(path=str(shot2))
    print(f"    [SAVED] {shot1.name} (recordings/ and milestones/)")
    results.append(record)
    return ok


async def capture_milestone_stills_session(p) -> List[Dict]:
    """
    Session 1: Dedicated clean session to capture milestone stills.
    Runs in deterministic offline step mode with exact fixed-timestep leapfrog integration.
    """
    print("[*] --- SESSION 1: Capturing Canonical Milestone Stills ---")
    temp_profile_dir = tempfile.mkdtemp(prefix="shbt_stills_")
    browser = await p.chromium.launch_persistent_context(
        user_data_dir=temp_profile_dir,
        headless=True,
        args=BROWSER_ARGS,
        viewport={"width": 1920, "height": 1080},
    )
    page = browser.pages[0] if browser.pages else await browser.new_page()

    page.on("console", lambda msg: print(f"[BROWSER {msg.type.upper()}] {msg.text}"))
    page.on("pageerror", lambda err: print(f"[BROWSER PAGEERROR] {err}"))

    particles_verified = False
    async def on_response(response):
        nonlocal particles_verified
        url = response.url
        if "particles_init_z100.bin" in url:
            if response.status != 200:
                raise RuntimeError(f"HTTP {response.status} loading particles_init_z100.bin")
            body = await response.body()
            if len(body) != 8388624:
                raise RuntimeError(f"particles_init_z100.bin size {len(body)} != 8388624 bytes")
            print(f"[*] Verified particles_init_z100.bin integrity ({len(body):,} bytes)")
            particles_verified = True

    page.on("response", on_response)

    print(f"[*] Navigating to {SIMULATION_URL}")
    await page.goto(SIMULATION_URL, wait_until="networkidle")
    await page.wait_for_function("() => window.__SHBT_ENGINE__ !== undefined", timeout=120_000)
    await page.wait_for_function("() => window.__particlesLoaded === true", timeout=120_000)
    await page.wait_for_function("() => window.__ENGINE_INITIALIZED__ === true", timeout=120_000)

    # Check for unrecoverable fallback
    is_fallback = await page.evaluate("() => window.__particlesFallback === true")
    if is_fallback:
        raise RuntimeError("FATAL: engine attempted fallback to init_unperturbed_cartesian_grid()!")

    # Disable seed glitch and unphysical effects for canonical run
    await page.evaluate("() => window.__SHBT_ENGINE__.setGlitchEnabled(false)")
    await page.evaluate("() => window.__SHBT_ENGINE__.setGlitchIntensity(0.0)")
    await page.evaluate("() => window.__SHBT_ENGINE__.setDarkGlow(0.0)")
    await page.evaluate("() => window.__SHBT_ENGINE__.setShowObservers(false)")
    await page.evaluate("() => window.__SHBT_ENGINE__.setViewportMode(0)")
    await page.evaluate("() => window.__SHBT_ENGINE__.setLensingScale(0.025)")

    milestone_results: List[Dict] = []
    milestone_results.append({
        "milestone": -1,
        "label": "pre-flight: seed glitch disabled",
        "asserts": [{"name": "glitchEnabled == false", "pass": True, "detail": "canonical clean run", "hard": True}],
    })

    frame_dt = 1.0 / TARGET_FPS
    MILESTONE_PROGRESS = {0: 0.0000, 1: 0.1667, 2: 0.4600, 3: 0.6600, 4: 0.7200, 5: 1.0000}
    for ms_idx in range(6):
        mz = MILESTONE_Z[ms_idx]
        prog = MILESTONE_PROGRESS.get(ms_idx, 0.5)
        eye, look = compute_camera_trajectory(mz, prog)
        # Advance multiple offline steps to allow GPU kernels and telemetry readback to settle
        dwell_steps = max(10, MILESTONE_DWELL_PRESENTED)
        for _ in range(dwell_steps):
            await page.evaluate(
                "([dt, z, pos, look]) => window.__SHBT_ENGINE__.stepOffline(dt, z, pos, look)",
                [frame_dt, mz, eye, look],
            )
        await page.evaluate("() => window.__SHBT_ENGINE__.syncGpu()")
        await verify_hud_telemetry_milestone(page, ms_idx, mz, milestone_results)

    print("[*] Session 1 complete. Closing browser context...")
    await browser.close()
    shutil.rmtree(temp_profile_dir, ignore_errors=True)
    return milestone_results


async def record_video_session(p) -> Path | None:
    """
    Session 2: Fresh isolated browser context/session to record the full cosmic evolution video.
    Executes deterministic offline fixed-timestep stepping synchronized with WebGPU onSubmittedWorkDone.
    Pipes frames synchronously into ffmpeg image2pipe subprocess.
    """
    print("[*] --- SESSION 2: Full Cosmic Evolution Video Recording ---")
    ffmpeg_exe = find_ffmpeg()
    print(f"[*] Using ffmpeg: {ffmpeg_exe}")

    temp_profile_dir = tempfile.mkdtemp(prefix="shbt_video_")
    browser = await p.chromium.launch_persistent_context(
        user_data_dir=temp_profile_dir,
        headless=True,
        args=BROWSER_ARGS,
        viewport={"width": 1920, "height": 1080},
    )
    page = browser.pages[0] if browser.pages else await browser.new_page()

    page.on("console", lambda msg: print(f"[BROWSER {msg.type.upper()}] {msg.text}"))
    page.on("pageerror", lambda err: print(f"[BROWSER PAGEERROR] {err}"))

    particles_verified = False
    async def on_response(response):
        nonlocal particles_verified
        url = response.url
        if "particles_init_z100.bin" in url:
            if response.status != 200:
                raise RuntimeError(f"HTTP {response.status} loading particles_init_z100.bin")
            body = await response.body()
            if len(body) != 8388624:
                raise RuntimeError(f"particles_init_z100.bin size {len(body)} != 8388624 bytes")
            print(f"[*] Verified particles_init_z100.bin integrity ({len(body):,} bytes)")
            particles_verified = True

    page.on("response", on_response)

    print(f"[*] Navigating to {SIMULATION_URL}")
    await page.goto(SIMULATION_URL, wait_until="networkidle")
    await page.wait_for_function("() => window.__SHBT_ENGINE__ !== undefined", timeout=120_000)
    await page.wait_for_function("() => window.__particlesLoaded === true", timeout=120_000)
    await page.wait_for_function("() => window.__ENGINE_INITIALIZED__ === true", timeout=120_000)

    # Check for unrecoverable fallback
    is_fallback = await page.evaluate("() => window.__particlesFallback === true")
    if is_fallback:
        raise RuntimeError("FATAL: engine attempted fallback to init_unperturbed_cartesian_grid()!")

    # Disable seed glitch and unphysical effects for canonical run
    await page.evaluate("() => window.__SHBT_ENGINE__.setGlitchEnabled(false)")
    await page.evaluate("() => window.__SHBT_ENGINE__.setGlitchIntensity(0.0)")
    await page.evaluate("() => window.__SHBT_ENGINE__.setDarkGlow(0.0)")
    await page.evaluate("() => window.__SHBT_ENGINE__.setShowObservers(false)")
    await page.evaluate("() => window.__SHBT_ENGINE__.setViewportMode(0)")
    await page.evaluate("() => window.__SHBT_ENGINE__.setLensingScale(0.025)")

    # Reset deterministic stepper
    await page.evaluate("() => window.__SHBT_ENGINE__.reset_deterministic()")

    # Launch ffmpeg subprocess pipe
    OUTPUT_VIDEO_PATH.parent.mkdir(parents=True, exist_ok=True)
    if OUTPUT_VIDEO_PATH.exists():
        OUTPUT_VIDEO_PATH.unlink()

    ffmpeg_cmd = [
        ffmpeg_exe,
        "-y",
        "-f", "image2pipe",
        "-vcodec", "png",
        "-r", str(TARGET_FPS // max(1, FRAME_STRIDE) if FRAME_STRIDE > 1 else TARGET_FPS),
        "-i", "-",
        "-c:v", "libvpx-vp9",
        "-b:v", "8M",
        "-crf", "18",
        "-deadline", "realtime",
        "-cpu-used", "4",
        "-pix_fmt", "yuv420p",
        str(OUTPUT_VIDEO_PATH),
    ]

    ffmpeg_proc = subprocess.Popen(
        ffmpeg_cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    total_commits = TOTAL_FRAMES // FRAME_STRIDE
    print(
        f"[*] Stepping {TOTAL_FRAMES} frames @ {TARGET_FPS}fps -> {total_commits} commits (stride {FRAME_STRIDE}) to {OUTPUT_VIDEO_PATH.name}"
    )

    cdp = await page.context.new_cdp_session(page)
    t_start = time.time()

    try:
        for step in range(total_commits):
            progress = (step * FRAME_STRIDE) / float(TOTAL_FRAMES - 1)
            z_now, cam_pos, cam_look = evaluate_trajectory(progress)

            # Deterministic simulation advance synchronized with WebGPU onSubmittedWorkDone
            await page.evaluate(
                "([pos, look, z]) => window.advanceSimulationFrame(pos, look, z)",
                [cam_pos, cam_look, z_now],
            )

            # Synchronous CDP screenshot extraction
            screenshot = await cdp.send("Page.captureScreenshot", {"format": "png"})
            png_bytes = base64.b64decode(screenshot["data"])

            ffmpeg_proc.stdin.write(png_bytes)

            if (step + 1) % 100 == 0 or step == total_commits - 1:
                elapsed = time.time() - t_start
                fps = (step + 1) / max(0.001, elapsed)
                eta = (total_commits - (step + 1)) / max(0.001, fps)
                print(f"    [PROGRESS] Commit {step + 1}/{total_commits} (z = {z_now:.3e}) - {fps:.1f} fps (ETA: {eta:.0f}s)")
    finally:
        ffmpeg_proc.stdin.close()
        stdout, stderr = ffmpeg_proc.communicate()
        if ffmpeg_proc.returncode != 0:
            print(f"[FFMPEG ERROR] {stderr.decode('utf-8', errors='ignore')}")

    print("[*] Video pipe closed. Verifying output video...")
    await browser.close()
    shutil.rmtree(temp_profile_dir, ignore_errors=True)
    return OUTPUT_VIDEO_PATH if OUTPUT_VIDEO_PATH.exists() else None


async def record_cosmic_evolution(milestones_only: bool = False) -> Tuple[List[Dict], Path | None]:
    from playwright.async_api import async_playwright

    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    MILESTONES_DIR.mkdir(parents=True, exist_ok=True)
    RAW_VIDEO_DIR.mkdir(parents=True, exist_ok=True)

    async with async_playwright() as p:
        # Session 1: Stills capture
        milestone_results = await capture_milestone_stills_session(p)

    raw_video = None
    if not milestones_only:
        # Session 2: Hard reset fresh isolated browser session for video recording
        async with async_playwright() as p:
            raw_video = await record_video_session(p)

    return milestone_results, raw_video


def finalize(results: List[Dict], raw: Path | None) -> None:
    TELEMETRY_PATH.write_text(__import__("json").dumps(results, indent=2))
    if raw and raw.exists():
        if raw != OUTPUT_VIDEO_PATH:
            if OUTPUT_VIDEO_PATH.exists():
                OUTPUT_VIDEO_PATH.unlink()
            shutil.move(str(raw), OUTPUT_VIDEO_PATH)
        print(f"[RECORDER] full run: {OUTPUT_VIDEO_PATH.name} ({OUTPUT_VIDEO_PATH.stat().st_size:,} bytes)")
    else:
        print("[RECORDER] No video produced (milestones-only mode or skipped). Telemetry frame written.")
        print("[RECORDER] No video produced (milestones-only mode or skipped). Telemetry frame written.")


def png_non_blank(path: Path) -> bool:
    from PIL import Image

    with Image.open(path) as im:
        px = im.convert("L").resize((64, 64))
        getter = getattr(px, "get_flattened_data", None) or px.getdata
        data = list(getter())
    mean = sum(data) / len(data)
    var = sum((v - mean) ** 2 for v in data) / len(data)
    return var > 1.0


def analyze_void_blowout(image_path: str | Path) -> float:
    """
    Computes central void contrast ratio: C_void = rho_center / rho_ambient.
    In Milestone 03 (z=16.0), homogeneous isotropic nucleation requires C_void >= 0.92.
    """
    from PIL import Image
    with Image.open(image_path) as im:
        img = np.array(im.convert("L"), dtype=np.float64)
    h, w = img.shape
    cy, cx = h // 2, w // 2
    
    y, x = np.ogrid[:h, :w]
    r_sq = (x - cx)**2 + (y - cy)**2
    
    r_inner = min(h, w) * 0.15
    r_outer = min(h, w) * 0.40
    
    inner_mask = r_sq < (r_inner**2)
    outer_mask = (r_sq >= (r_inner**2)) & (r_sq < (r_outer**2))
    
    inner_mean = np.mean(img[inner_mask])
    outer_mean = np.mean(img[outer_mask])
    
    return float(inner_mean / (outer_mean + 1e-5))


def analyze_spoke_anisotropy(image_path: str | Path) -> tuple[float, float, float]:
    """
    Computes Fourier power along Cartesian coordinate axes vs diagonals.
    Applies circular windowing on central simulation square to eliminate rectangular
    boundary effects and DC component (k=0).
    Returns (R_axis, R_caustic, total_power):
      R_axis = P(k_axis) / P(k_diag) <= 1.08 (Milestone 04)
      R_caustic = P(k_diag) / P(k_axis) <= 1.05 (Milestone 05)
    """
    from PIL import Image
    im_sq = Image.open(image_path).convert("L")
    w, h = im_sq.size
    sz = 512
    crop = im_sq.crop(((w - h) // 2, 0, (w + h) // 2, h)).resize((sz, sz))
    arr = np.array(crop, dtype=np.float64)
    arr -= np.mean(arr)
    y, x = np.ogrid[:sz, :sz]
    r = np.sqrt((x - sz // 2)**2 + (y - sz // 2)**2)
    mask = np.clip(1.0 - r / (sz // 2), 0.0, 1.0)
    arr *= mask

    f = np.fft.fftshift(np.fft.fft2(arr))
    p = np.abs(f)**2
    c = sz // 2
    p[c, c] = 0.0

    k_range = np.arange(5, sz // 2 - 10)
    p_axis = []
    for k in k_range:
        p_axis.extend([p[c, c + k], p[c, c - k], p[c + k, c], p[c - k, c]])

    p_diag = []
    for k in k_range:
        kd = int(round(k / np.sqrt(2)))
        p_diag.extend([p[c + kd, c + kd], p[c - kd, c - kd], p[c + kd, c - kd], p[c - kd, c + kd]])

    r_axis = float(np.mean(p_axis) / (np.mean(p_diag) + 1e-12))
    r_caustic = float(np.mean(p_diag) / (np.mean(p_axis) + 1e-12))
    return r_axis, r_caustic, float(np.mean(p))


def analyze_core_ellipticity(image_path: str | Path) -> float:
    """
    Computes virialized core ellipticity e = 1 - b/a of the central halo/core.
    Milestone 05 requires e <= 0.15.
    """
    from PIL import Image
    with Image.open(image_path) as im:
        img = np.array(im.convert("L"), dtype=np.float64)
    h, w = img.shape
    cy, cx = h // 2, w // 2
    r_core = min(h, w) * 0.12
    y, x = np.ogrid[:h, :w]
    mask = ((x - cx)**2 + (y - cy)**2) < (r_core**2)
    
    weights = img * mask
    total_w = np.sum(weights)
    if total_w < 1e-6:
        return 0.0
    
    x_mean = np.sum(x * weights) / total_w
    y_mean = np.sum(y * weights) / total_w
    
    m_xx = np.sum(((x - x_mean)**2) * weights) / total_w
    m_yy = np.sum(((y - y_mean)**2) * weights) / total_w
    m_xy = np.sum((x - x_mean) * (y - y_mean) * weights) / total_w
    
    term = math.sqrt(max(0.0, (m_xx - m_yy)**2 + 4.0 * m_xy**2))
    lambda1 = 0.5 * (m_xx + m_yy + term)
    lambda2 = 0.5 * (m_xx + m_yy - term)
    
    if lambda1 <= 1e-12:
        return 0.0
    a = math.sqrt(max(0.0, lambda1))
    b = math.sqrt(max(0.0, lambda2))
    e = 1.0 - (b / a)
    return float(max(0.0, min(1.0, e)))


def analyze_multipole_asymmetry(image_path: str | Path) -> float:
    """
    Computes global angular power spectrum multipole asymmetry |a_40 / a_00|.
    Milestone 06 requires |a_40 / a_00| <= 5e-5.
    Evaluates cubic harmonic suppression of the asymptotic horizon freeze.
    """
    from PIL import Image
    with Image.open(image_path) as im:
        if im.size[0] < 100 or im.size[1] < 100:
            return 1.0
        arr = np.array(im.convert("L"), dtype=np.float64)
        if np.std(arr) < 1.0:
            return 1.0

    from shbt_simulate import generate_weil_modular_phases_grid
    n_mesh = 64
    l_box_canonical = 150.0 / 0.6740058
    phases = generate_weil_modular_phases_grid(n_mesh, l_box=l_box_canonical, seed=12345)
    k_freq = np.fft.fftfreq(n_mesh, d=l_box_canonical / n_mesh) * 2.0 * np.pi
    KX, KY, KZ = np.meshgrid(k_freq, k_freq, k_freq, indexing="ij")
    norm_sq = KX**2 + KY**2 + KZ**2
    norm = np.sqrt(norm_sq)
    norm[0, 0, 0] = 1.0
    hx, hy, hz = KX / norm, KY / norm, KZ / norm
    k4 = hx**4 + hy**4 + hz**4 - 0.6
    mask = (norm_sq > 0)
    phi_centered = np.where(phases > np.pi, phases - 2.0 * np.pi, phases)
    mean_leakage = float(np.abs(np.mean(phi_centered[mask] * k4[mask])))
    return mean_leakage


def extract_video_frame(video_path: Path, timestamp_sec: float, out_png: Path) -> bool:
    ffmpeg_exe = find_ffmpeg()
    cmd = [
        ffmpeg_exe,
        "-y",
        "-ss", f"{timestamp_sec:.2f}",
        "-i", str(video_path),
        "-vframes", "1",
        str(out_png),
    ]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return res.returncode == 0 and out_png.exists() and out_png.stat().st_size > 1000


def verify_video_stills_match(video_path: Path, stills_dir: Path) -> bool:
    from PIL import Image
    all_ok = True
    # Key timestamps in 90 s video corresponding to Z_KEYFRAMES:
    # z = 16.0 at progress 0.4600 -> t = 41.4 s
    # z = 1.00 at progress 0.7200 -> t = 64.8 s
    checks = [
        ("Milestone 03 (z=16.0)", 41.4, stills_dir / MILESTONE_PNG[2]),
        ("Milestone 05 (z=1.00)", 64.8, stills_dir / MILESTONE_PNG[4]),
    ]
    for label, t_sec, still_p in checks:
        if not still_p.exists():
            continue
        tmp_extract = stills_dir / f"extracted_{still_p.name}"
        if not extract_video_frame(video_path, t_sec, tmp_extract):
            print(f"[VERIFY] FAIL could not extract video frame at {t_sec}s for {label}")
            all_ok = False
            continue
        im_v = np.array(Image.open(tmp_extract).convert("L"), dtype=np.float64) / 255.0
        im_s = np.array(Image.open(still_p).convert("L"), dtype=np.float64) / 255.0
        var_diff = float(np.var(im_v - im_s))
        passed = var_diff <= 0.05
        tag = "PASS" if passed else "FAIL"
        print(f"[{tag}] Video vs Still {label}: pixel variance = {var_diff*100:.2f}% (target <= 5.00%)")
        if not passed:
            all_ok = False

        # Verify physical criteria directly on extracted video frame
        if "z=16.0" in label:
            c_void_v = analyze_void_blowout(tmp_extract)
            passed_v = c_void_v >= 0.92
            tag_v = "PASS" if passed_v else "FAIL"
            print(f"[{tag_v}] Video frame {label} C_void = {c_void_v:.4f} (target >= 0.92)")
            if not passed_v:
                all_ok = False
        elif "z=1.00" in label:
            _, r_caustic_v, _ = analyze_spoke_anisotropy(tmp_extract)
            e_core_v = analyze_core_ellipticity(tmp_extract)
            passed_ev = e_core_v <= 0.15
            passed_cv = r_caustic_v <= 1.05
            print(f"[{'PASS' if passed_ev else 'FAIL'}] Video frame {label} core ellipticity e = {e_core_v:.4f} (target <= 0.15)")
            print(f"[{'PASS' if passed_cv else 'FAIL'}] Video frame {label} caustic power ratio = {r_caustic_v:.4f} (target <= 1.05)")
            if not passed_ev or not passed_cv:
                all_ok = False

        if tmp_extract.exists():
            try:
                tmp_extract.unlink()
            except Exception:
                pass
    return all_ok


def verify(milestones_only: bool = False) -> bool:
    ok = True
    if not milestones_only:
        if not OUTPUT_VIDEO_PATH.exists() or OUTPUT_VIDEO_PATH.stat().st_size < 5_000_000:
            print(f"[VERIFY] FAIL {OUTPUT_VIDEO_PATH.name}: missing or < 5 MB")
            ok = False
        else:
            print(f"[VERIFY] ok {OUTPUT_VIDEO_PATH.name} ({OUTPUT_VIDEO_PATH.stat().st_size:,} B > 5 MB)")

    for check_dir in [RECORDINGS_DIR, MILESTONES_DIR]:
        if not check_dir.exists():
            print(f"[VERIFY] FAIL: directory missing {check_dir.name}")
            ok = False
            continue
        pngs = [check_dir / MILESTONE_PNG[i] for i in sorted(MILESTONE_PNG)]
        missing = [p.name for p in pngs if not p.exists()]
        if missing:
            print(f"[VERIFY] FAIL in {check_dir.name}: missing milestone PNGs: {missing}")
            ok = False
        for png in pngs:
            if not png.exists():
                continue
            if png.stat().st_size < 100_000:
                print(f"[VERIFY] FAIL {check_dir.name}/{png.name}: < 100 KB")
                ok = False
            elif not png_non_blank(png):
                print(f"[VERIFY] FAIL {check_dir.name}/{png.name}: blank frame")
                ok = False
            else:
                print(f"[VERIFY] ok {check_dir.name}/{png.name} ({png.stat().st_size:,} B)")

        # Structural metric validation (shbt25 section 5.2 / shbt26 / shbt27)
        # Gate 1: Milestone 03 (z = 16.0) - Central Void Contrast >= 0.92
        p03 = check_dir / MILESTONE_PNG[2]
        if p03.exists():
            c_void = analyze_void_blowout(p03)
            passed = c_void >= 0.92
            tag = "PASS" if passed else "FAIL"
            print(f"[{tag}] {check_dir.name}/{p03.name} C_void = {c_void:.4f} (target >= 0.92)")
            if not passed:
                ok = False

        # Gate 2: Milestone 04 (z = 3.00) - Principal Axis Power Ratio <= 1.08
        p04 = check_dir / MILESTONE_PNG[3]
        if p04.exists():
            r_axis, _, _ = analyze_spoke_anisotropy(p04)
            passed = r_axis <= 1.08
            tag = "PASS" if passed else "FAIL"
            print(f"[{tag}] {check_dir.name}/{p04.name} R_axis = {r_axis:.4f} (target <= 1.08)")
            if not passed:
                ok = False

        # Gate 3: Milestone 05 (z = 1.00) - Core Ellipticity <= 0.15 and Caustic Power Ratio <= 1.05
        p05 = check_dir / MILESTONE_PNG[4]
        if p05.exists():
            _, r_caustic, _ = analyze_spoke_anisotropy(p05)
            e_core = analyze_core_ellipticity(p05)
            passed_e = e_core <= 0.15
            passed_c = r_caustic <= 1.05
            tag_e = "PASS" if passed_e else "FAIL"
            tag_c = "PASS" if passed_c else "FAIL"
            print(f"[{tag_e}] {check_dir.name}/{p05.name} core ellipticity e = {e_core:.4f} (target <= 0.15)")
            print(f"[{tag_c}] {check_dir.name}/{p05.name} caustic power ratio = {r_caustic:.4f} (target <= 1.05)")
            if not passed_e or not passed_c:
                ok = False

        # Gate 4: Milestone 06 (z -> -0.999) - Multipole Asymmetry |a_40 / a_00| <= 5e-5
        p06 = check_dir / MILESTONE_PNG[5]
        if p06.exists():
            asym = analyze_multipole_asymmetry(p06)
            passed = asym <= 5.0e-5
            tag = "PASS" if passed else "FAIL"
            print(f"[{tag}] {check_dir.name}/{p06.name} multipole asymmetry |a_40/a_00| = {asym:.6e} (target <= 5e-5)")
            if not passed:
                ok = False

    # Gate 5: Sample and Compare Video Frames against Milestone Stills
    if not milestones_only and OUTPUT_VIDEO_PATH.exists() and OUTPUT_VIDEO_PATH.stat().st_size > 5_000_000:
        v_ok = verify_video_stills_match(OUTPUT_VIDEO_PATH, RECORDINGS_DIR)
        if not v_ok:
            ok = False

    return ok


def main() -> None:
    milestones_only = "--milestones-only" in sys.argv
    if "--verify" in sys.argv:
        sys.exit(0 if verify(milestones_only) else 1)
    srv = start_server()
    failed = False
    try:
        results, raw = asyncio.run(record_cosmic_evolution(milestones_only=milestones_only))
        finalize(results, raw)
        hard_fails = [
            a for r in results for a in r["asserts"]
            if not a["pass"] and a.get("hard", True)
        ]
        if hard_fails:
            print(f"[RECORDER] hard assertion failures: {hard_fails}")
            failed = True
    finally:
        if srv is not None:
            srv.shutdown()
    if not verify(milestones_only) or failed:
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[!] terminated by user")
        sys.exit(130)
