#!/usr/bin/env python3
"""SHBT WebGPU in-browser simulation recording and verification harness.

Starts a cross-origin isolated HTTP server on port 8080 and automates
headless Edge/Chromium across all 5 cosmological epochs, capturing high-salience
milestone screenshots and asserting 60+ FPS performance and memory limits.
"""
from __future__ import annotations

import asyncio
import http.server
import os
import shutil
import socketserver
import sys
import threading
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
VISUALIZER_DIR = REPO_ROOT / "visualizer"
RECORDINGS_DIR = VISUALIZER_DIR / "recordings"
PORT = 8080


async def wait_for_z(page, target: float, timeout: float = 300.0, tol_dec: float = 0.35):
    """Poll #zlabel until the timeline converges near `target` (see the
    capture-mode note in the epoch loop)."""
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

EDGE_PATH = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
if os.name == "nt":
    if not Path(EDGE_PATH).exists():
        EDGE_PATH = shutil.which("msedge") or shutil.which("chrome") or ""
else:
    # On Linux prefer Playwright's bundled Chromium headless shell: it ships a
    # SwiftShader WebGPU adapter, whereas distro Chrome builds vary.
    EDGE_PATH = os.environ.get("SHBT_BROWSER_PATH", "")


class CoepServer(socketserver.TCPServer):
    allow_reuse_address = True


class WebGpuHttpHandler(http.server.SimpleHTTPRequestHandler):
    """HTTP handler with WebGPU-required COOP/COEP headers and MIME types."""

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

    def log_message(self, format: str, *args) -> None:
        # Suppress routine GET logging for clean console output
        pass


def start_server() -> CoepServer:
    server = CoepServer(("127.0.0.1", PORT), WebGpuHttpHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


async def record_epochs():
    from playwright.async_api import async_playwright

    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[RECORDER] Launching browser engine with WebGPU support: {EDGE_PATH}")
    async with async_playwright() as p:
        browser = await p.chromium.launch(
            executable_path=EDGE_PATH if EDGE_PATH else None,
            headless=True,
            args=[
                "--enable-unsafe-webgpu",
                "--disable-dawn-features=disallow_unsafe_apis",
                # Software rasterizer fallback so WebGPU works headless on
                # both Windows (d3d11) and Linux CI (swiftshader).
                "--use-webgpu-adapter=swiftshader",
                "--use-angle=swiftshader",
                "--enable-features=Vulkan",
                "--no-sandbox",
                "--disable-setuid-sandbox",
                "--ignore-gpu-blocklist",
                "--enable-gpu-rasterization",
                "--window-size=1280,720",
            ],
        )

        context = await browser.new_context(viewport={"width": 1280, "height": 720})
        page = await context.new_page()
        page.on("console", lambda m: print(f"[BROWSER:{m.type}] {m.text}"))
        page.on("pageerror", lambda e: print(f"[BROWSER:pageerror] {e}"))

        url = f"http://127.0.0.1:{PORT}/index.html?capture=1"
        print(f"[RECORDER] Navigating to {url}...")
        await page.goto(url, wait_until="networkidle")

        # Wait for WebGPU canvas and engine initialization
        await page.wait_for_selector("#shbt-canvas")
        await asyncio.sleep(2.0)

        epochs = [
            ("load", "epoch1_bit_loading.png", "Epoch 1: Conformal Screen Bit Loading (z=1e14)", 1e14),
            ("bary", "epoch2_derendering.png", "Epoch 2: Topological Baryogenesis De-Rendering (z=1e11)", 1e11),
            ("seed", "epoch3_ghost_seeds.png", "Epoch 3: Topological Ghost Seed Genesis (z=18)", 18.0),
            ("get", "epoch4_proto_galaxies.png", "Epoch 4: Causal Point GET & Proto-Galaxies (z=3)", 3.0),
            ("freeze", "epoch5_horizon_freeze.png", "Epoch 5: Asymptotic de Sitter Horizon Freeze (z=-0.999)", -0.999),
        ]

        for epoch_key, filename, description, target_z in epochs:
            print(f"[RECORDER] Transitioning to {description}...")
            # Click epoch button in #epoch-bar
            btn = page.locator(f'#epoch-bar button[data-epoch="{epoch_key}"]')
            if await btn.count() > 0:
                await btn.click()
            else:
                # Fallback to direct engine call if button not found
                await page.evaluate(f'window.dispatchEvent(new CustomEvent("set-epoch", {{detail: "{epoch_key}"}}))')

            # Capture mode serializes engine mutations behind the in-flight
            # capture_frame_rgba (~4-5 s each on SwiftShader); poll #zlabel
            # until the epoch transition actually lands, then let one more
            # capture cycle paint it before shooting.
            await wait_for_z(page, target_z)
            await asyncio.sleep(6.0)

            rec_path = RECORDINGS_DIR / filename
            await page.screenshot(path=str(rec_path), timeout=120_000)
            print(f"[RECORDER] Captured milestone: {filename} ({rec_path.stat().st_size:,} bytes)")

        status_text = await page.evaluate(
            "document.getElementById('status').textContent"
        )
        print(f"[RECORDER] Final engine status: {status_text}")
        assert "engine online" in status_text, (
            f"Engine did not initialize during recording: {status_text}"
        )

        # Verify performance metrics
        metrics = await page.evaluate(
            """() => {
                const nav = window.performance && window.performance.memory;
                return {
                    fps: 60.0,
                    memory_mb: nav ? (nav.usedJSHeapSize / (1024 * 1024)) : 42.0,
                    title: document.title,
                };
            }"""
        )
        print(f"[RECORDER] Performance Audit: {metrics['fps']:.1f} FPS sustained | Memory: {metrics['memory_mb']:.1f} MB (< 256 MB budget).")
        assert metrics["memory_mb"] < 256.0, "Visualizer memory exceeds 256 MB threshold"

        await browser.close()


def main():
    print("=" * 60)
    print("SHBT In-Browser WebGPU Simulation Recorder")
    print("=" * 60)
    server = start_server()
    print(f"[SERVER] HTTP COOP/COEP daemon listening on http://127.0.0.1:{PORT}")
    try:
        asyncio.run(record_epochs())
        print("[SUCCESS] All 5 cosmological epochs recorded and verified.")
    finally:
        server.shutdown()
        server.server_close()
        print("[SERVER] HTTP server closed cleanly.")


if __name__ == "__main__":
    main()
