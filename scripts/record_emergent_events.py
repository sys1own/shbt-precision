#!/usr/bin/env python3
"""SHBT WebGPU emergent-condensation event recorder.

Drives a headless Chromium (SwiftShader WebGPU) through the emergent
seed-condensation window (z = 30 -> 7) and records the six emergent events
with Playwright compositor video + milestone screenshots. Unlike
`record_simulation_events.py` (which uses the ?capture=1 in-engine
readback path), this recorder only needs the presented canvas, so it works
in environments where GPU->CPU mapAsync is unavailable.

Events (shbt6 roadmap, Phase 4):
  01 pure quantum foam            z = 30  (zero seeds)
  02 first seed glitch            z = 17  (register overflow onset)
  03 first Ghost Seeds            z = 14  (condensed defects visible)
  04 Stinespring tethers          z = 12  (anti-baryon de-rendering)
  05 lensing starbursts           z = 10  (emergent-mass lensing)
  06 shrinking causal spheres     z = 7   (entropy budget depletion)

Artifacts (visualizer/recordings/):
  emergent_full_run.webm          continuous scrub video
  NN_<event>.webm                 per-event clips cut with ffmpeg
  NN_<event>.png                  milestone screenshots with HUD
  emergent_event_telemetry.json   HUD telemetry at each milestone
"""
from __future__ import annotations

import asyncio
import http.server
import json
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

# (filename slug, target z, dwell seconds)
EVENTS = [
    ("01_pure_quantum_foam", 30.0, 14.0),
    ("02_first_seed_glitch", 17.0, 14.0),
    ("03_first_ghost_seeds", 14.0, 14.0),
    ("04_stinespring_tethers", 12.0, 14.0),
    ("05_lensing_starbursts", 10.0, 14.0),
    ("06_shrinking_causal_spheres", 7.0, 14.0),
]


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


async def record() -> tuple[list[dict], Path | None]:
    from playwright.async_api import async_playwright

    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    RAW_VIDEO_DIR.mkdir(parents=True, exist_ok=True)

    segments: list[dict] = []
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(
            args=[
                "--enable-unsafe-webgpu",
                "--use-angle=vulkan",
                "--enable-features=Vulkan,DefaultANGLEVulkan",
            ]
        )
        ctx = await browser.new_context(
            viewport={"width": 1280, "height": 720},
            record_video_dir=str(RAW_VIDEO_DIR),
            record_video_size={"width": 1280, "height": 720},
        )
        page = await ctx.new_page()
        t0 = time.monotonic()
        # ?capture=1 blits each rendered frame into a 2D overlay canvas —
        # the only path whose pixels headless compositing can see.
        await page.goto(
            f"http://127.0.0.1:{PORT}/visualizer/?capture=1",
            wait_until="networkidle",
        )
        await page.wait_for_function("window.__SHBT_ENGINE__ !== undefined", timeout=120_000)
        # capture_frame_rgba takes ~4-5 s per frame on SwiftShader; wait for
        # the overlay to hold real pixels before scrubbing.
        await page.wait_for_function(
            """() => {
                const c = document.getElementById('capture-canvas');
                if (!c) return false;
                const d = c.getContext('2d').getImageData(0, 0, 16, 16).data;
                for (const v of d) if (v) return true;
                return false;
            }""",
            timeout=60_000,
        )

        for slug, z, dwell in EVENTS:
            mark = time.monotonic() - t0
            await page.evaluate(
                "(z) => window.__SHBT_ENGINE__.setRedshift(z)", z
            )
            print(f"[RECORDER] {slug}: scrubbing to z={z}, dwell {dwell}s")
            # capture_frame_rgba is ~5 s/frame; the queued setRedshift +
            # one HUD refresh lag the scrub. Wait until the displayed epoch
            # actually reaches the target before dwelling/screenshotting.
            await page.wait_for_function(
                """(z) => {
                    const t = window.__SHBT_ENGINE__.getTelemetry();
                    return t && t.z <= z + 0.3;
                }""",
                arg=z,
                timeout=60_000,
            )
            await page.wait_for_timeout(int(dwell * 1000))
            shot = RECORDINGS_DIR / f"{slug}.png"
            await page.screenshot(path=str(shot))
            tele = await page.evaluate("window.__SHBT_ENGINE__.getTelemetry()")
            segments.append(
                {
                    "event": slug,
                    "z_target": z,
                    "start_s": mark,
                    "end_s": time.monotonic() - t0,
                    "telemetry": tele,
                }
            )
            print(
                f"[RECORDER] {slug}: z={tele['z']:.3f} seeds={tele['seedCount']} "
                f"mass={tele['totalMass']:.3e} debt={tele['landauerDebt']:.3e}"
            )

        video = page.video
        await ctx.close()
        await browser.close()
        raw = Path(await video.path()) if video else None
        return segments, raw


def finalize(segments: list[dict], raw: Path | None) -> None:
    if not raw or not raw.exists():
        print("[RECORDER] WARNING: no compositor video produced")
        return
    full = RECORDINGS_DIR / "emergent_full_run.webm"
    shutil.move(str(raw), full)
    print(f"[RECORDER] full run: {full.name} ({full.stat().st_size:,} bytes)")
    if not shutil.which("ffmpeg"):
        return
    for seg in segments:
        clip = RECORDINGS_DIR / f"{seg['event']}.webm"
        subprocess.run(
            [
                "ffmpeg", "-y", "-ss", f"{seg['start_s']:.2f}",
                "-to", f"{seg['end_s']:.2f}", "-i", str(full),
                "-c", "copy", str(clip),
            ],
            check=True, capture_output=True,
        )
        print(f"[RECORDER] clip: {clip.name} ({clip.stat().st_size:,} bytes)")


def main() -> None:
    srv = start_server()
    try:
        segments, raw = asyncio.run(record())
        (RECORDINGS_DIR / "emergent_event_telemetry.json").write_text(
            json.dumps(segments, indent=2)
        )
        finalize(segments, raw)
    finally:
        if srv is not None:
            srv.shutdown()


if __name__ == "__main__":
    main()
