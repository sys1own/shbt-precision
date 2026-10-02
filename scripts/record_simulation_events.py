#!/usr/bin/env python3
"""SHBT WebGPU full-length event recorder.

Serves visualizer/ with COOP/COEP headers on 127.0.0.1:8080, drives a
headless chromium (SwiftShader WebGPU) through all five cosmological
epochs, records a continuous video of the whole run plus a milestone
screenshot per event, captures HUD telemetry at each milestone, and splits
the full-run video into per-event clips with ffmpeg.

Artifacts (visualizer/recordings/):
  full_simulation_run.webm          entire epoch sequence
  NN_<event>.webm                   per-event clip cut from the full run
  NN_<event>.png                    milestone screenshot with HUD
  event_telemetry.json              HUD state captured at each milestone
"""
from __future__ import annotations

import asyncio
import http.server
import json
import os
import shutil
import socketserver
import subprocess
import threading
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
VISUALIZER_DIR = REPO_ROOT / "visualizer"
RECORDINGS_DIR = VISUALIZER_DIR / "recordings"
RAW_VIDEO_DIR = RECORDINGS_DIR / "raw_videos"
PORT = 8080

# (epoch button data-epoch, clip filename, screenshot filename, dwell s)
EPOCHS = [
    ("load", "01_primordial_bit_loading.webm", "01_primordial_bit_loading.png", 10.0),
    ("bary", "02_baryogenesis_derendering.webm", "02_baryogenesis_derendering.png", 10.0),
    ("seed", "03_ghost_seed_einstein_rings.webm", "03_ghost_seed_einstein_rings.png", 10.0),
    ("get", "04_causal_point_proto_galaxies.webm", "04_causal_point_proto_galaxies.png", 10.0),
    ("freeze", "05_asymptotic_horizon_freeze.webm", "05_asymptotic_horizon_freeze.png", 8.0),
]

# Target redshift each epoch button sets (mirror of EPOCHS in index.js).
TARGET_Z = {"load": 1e14, "bary": 1e11, "seed": 18.0, "get": 3.0, "freeze": -0.999}


async def wait_for_z(page, target: float, timeout: float = 300.0, tol_dec: float = 0.35):
    """Poll #zlabel until the timeline converges near `target`.

    Capture mode serializes every engine mutation behind the in-flight
    capture_frame_rgba (~4-5 s each on SwiftShader), so a clicked epoch only
    takes effect several seconds later. Fixed dwells fire the screenshot
    before the transition lands; polling the HUD label is the only reliable
    signal.
    """
    import math
    import re

    t0 = time.monotonic()
    target_l = math.log10(max(target + 1.0, 1e-9))
    last = None
    while time.monotonic() - t0 < timeout:
        txt = await page.evaluate("document.getElementById('zlabel').textContent")
        m = re.search(r"z\s*=\s*([-+0-9.eE]+)", txt)
        if m:
            z = float(m.group(1))
            last = z
            if abs(math.log10(max(z + 1.0, 1e-9)) - target_l) < tol_dec:
                return z
        await asyncio.sleep(1.0)
    print(f"[RECORDER] WARN: z did not converge to {target} within {timeout}s; last={last}")
    return last


class CoepServer(socketserver.TCPServer):
    allow_reuse_address = True


class WebGpuHttpHandler(http.server.SimpleHTTPRequestHandler):
    extensions_map = {
        **http.server.SimpleHTTPRequestHandler.extensions_map,
        ".wasm": "application/wasm",
        ".wgsl": "text/plain",
        ".js": "application/javascript",
        ".html": "text/html",
        ".json": "application/json",
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(VISUALIZER_DIR), **kwargs)

    def end_headers(self) -> None:
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        super().end_headers()

    def log_message(self, *args) -> None:
        pass


def start_server() -> CoepServer:
    server = CoepServer(("127.0.0.1", PORT), WebGpuHttpHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


HUD_SNAPSHOT_JS = """() => ({
    status: document.getElementById('status').textContent.trim(),
    z: document.getElementById('zlabel').textContent.trim(),
    f_load: document.getElementById('hud-fload').textContent.trim(),
    observers: document.getElementById('hud-observers').textContent.trim(),
    rentropy: document.getElementById('hud-rentropy').textContent.trim(),
    optics: document.getElementById('hud-optics').textContent.trim(),
    banner: document.getElementById('phase-banner').textContent.trim(),
    ledger: document.getElementById('hud-ledger').textContent.trim(),
})"""


async def record() -> dict:
    from playwright.async_api import async_playwright

    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    RAW_VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    for old in RAW_VIDEO_DIR.glob("*.webm"):
        old.unlink()

    telemetry = {"epochs": [], "video_segments": []}

    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True,
            args=[
                "--enable-unsafe-webgpu",
                "--disable-dawn-features=disallow_unsafe_apis",
                "--use-webgpu-adapter=swiftshader",
                "--use-angle=swiftshader",
                "--enable-features=Vulkan",
                "--no-sandbox",
                "--disable-setuid-sandbox",
                "--ignore-gpu-blocklist",
                "--window-size=1280,720",
            ],
        )
        context = await browser.new_context(
            viewport={"width": 1280, "height": 720},
            record_video_dir=str(RAW_VIDEO_DIR),
            record_video_size={"width": 1280, "height": 720},
        )
        page = await context.new_page()
        page.on("console", lambda m: print(f"[BROWSER:{m.type}] {m.text}"))
        page.on("pageerror", lambda e: print(f"[BROWSER:pageerror] {e}"))

        url = f"http://127.0.0.1:{PORT}/index.html?capture=1"
        print(f"[RECORDER] Navigating to {url} ...")
        await page.goto(url, wait_until="networkidle")
        await page.wait_for_selector("#shbt-canvas", timeout=15000)
        # Wait for the engine to report online (not just the canvas element).
        for _ in range(30):
            txt = await page.evaluate(
                "document.getElementById('status').textContent"
            )
            if "engine online" in txt:
                break
            await asyncio.sleep(0.5)
        status_text = await page.evaluate(
            "document.getElementById('status').textContent"
        )
        print(f"[RECORDER] Engine status: {status_text.strip()}")
        assert "engine online" in status_text, status_text

        t0 = time.monotonic()
        for epoch_key, clip_name, shot_name, dwell in EPOCHS:
            btn = page.locator(f'#epoch-bar button[data-epoch="{epoch_key}"]')
            await btn.click()
            seg_start = time.monotonic() - t0
            print(f"[RECORDER] Event {epoch_key}: waiting for z -> {TARGET_Z[epoch_key]} ...")
            z_now = await wait_for_z(page, TARGET_Z[epoch_key])
            print(f"[RECORDER] {epoch_key}: z converged at {z_now}; dwell {dwell}s ...")
            # Pause the timeline sweep so the milestone HUD state (f_load, z,
            # optics ledger) is captured before playback drifts past it, then
            # resume for the remainder of the video segment.
            was_playing = await page.evaluate(
                "document.getElementById('play').dataset.on === '1'"
            )
            if was_playing:
                await page.click("#play")
            # Wait long enough for one full capture cycle so the paused HUD
            # and the blitted frame reflect the converged epoch state.
            await asyncio.sleep(6.0)
            snap = await page.evaluate(HUD_SNAPSHOT_JS)
            shot_path = RECORDINGS_DIR / shot_name
            await page.screenshot(path=str(shot_path), timeout=90_000)
            if was_playing:
                await page.click("#play")
            await asyncio.sleep(max(dwell - 1.2, 0.5))
            seg_end = time.monotonic() - t0

            telemetry["epochs"].append(
                {"epoch": epoch_key, "screenshot": shot_name, **snap}
            )
            telemetry["video_segments"].append(
                {"epoch": epoch_key, "clip": clip_name,
                 "start_s": round(seg_start, 2), "end_s": round(seg_end, 2)}
            )
            print(
                f"[RECORDER] {epoch_key}: {snap['z']} | {snap['f_load']} | "
                f"{snap['optics'].splitlines()[0]}"
            )

        video = page.video
        await context.close()
        await browser.close()
        raw_video_path = Path(await video.path()) if video else None
        return telemetry, raw_video_path


def finalize_videos(telemetry: dict, raw_path) -> None:
    if not raw_path or not raw_path.exists():
        print("[RECORDER] WARNING: no raw video recorded")
        return
    full = RECORDINGS_DIR / "full_simulation_run.webm"
    shutil.copyfile(raw_path, full)
    print(f"[RECORDER] Full run video: {full.name} ({full.stat().st_size:,} bytes)")

    if shutil.which("ffmpeg"):
        for seg in telemetry["video_segments"]:
            out = RECORDINGS_DIR / seg["clip"]
            subprocess.run(
                [
                    "ffmpeg", "-y", "-ss", str(seg["start_s"]), "-to",
                    str(seg["end_s"]), "-i", str(full),
                    "-c:v", "libvpx-vp9", "-b:v", "2M", "-an", str(out),
                ],
                check=True, capture_output=True,
            )
            print(f"[RECORDER] Clip {seg['clip']} ({out.stat().st_size:,} bytes)")
    else:
        print("[RECORDER] ffmpeg unavailable; keeping full-run video only")


def main() -> None:
    server = start_server()
    print(f"[SERVER] COOP/COEP daemon on http://127.0.0.1:{PORT}")
    try:
        telemetry, video = asyncio.run(record())
        finalize_videos(telemetry, video)
        (RECORDINGS_DIR / "event_telemetry.json").write_text(
            json.dumps(telemetry, indent=2)
        )
        print("[SUCCESS] Events recorded; telemetry at visualizer/recordings/event_telemetry.json")
    finally:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
