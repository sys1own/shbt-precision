#!/usr/bin/env python3
"""SHBT seed-glitch recording suite (shbt8 Enhancement-11 refinement).

Records three named videos under visualizer/recordings/:

  emergent_seed_formation_clean.webm   glitch OFF, z=30 -> 7
  emergent_seed_glitch_nuance.webm     glitch ON @ 0.10, z=14 -> 7
  shbt_full_thermodynamic_cycle.webm   glitch ON @ 0.10, z=1e14 -> -0.999

Each segment records through a dedicated browser context so Playwright
emits one .webm per video. Uses the ?capture=1 overlay path (the only
surface headless compositing can see).
"""
from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

from record_simulation_events import (
    RECORDINGS_DIR,
    RAW_VIDEO_DIR,
    PORT,
    start_server,
    telemetry,
)

RELAX_FRAMES = 30
RELAX_TIMEOUT_MS = 180_000


async def wait_z(page, target, timeout=180_000):
    hi = target + (abs(target) * 0.01 if abs(target) > 10 else 0.3)
    await page.wait_for_function(
        """(zt) => {
            const t = window.__SHBT_ENGINE__.getTelemetry();
            return t && t.z <= zt;
        }""",
        arg=hi,
        timeout=timeout,
    )
    await page.wait_for_function(
        """(zt) => {
            const m = window.__lastHudMetrics;
            return m && m.z <= zt;
        }""",
        arg=hi,
        timeout=timeout,
    )


async def relax(page, frames=RELAX_FRAMES):
    base = await page.evaluate("window.__presentedFrames")
    try:
        await page.wait_for_function(
            "(b) => window.__presentedFrames >= b",
            arg=base + frames,
            timeout=RELAX_TIMEOUT_MS,
        )
    except Exception:
        pass


async def open_capture(pw, browser_args):
    browser = await pw.chromium.launch(args=browser_args)
    ctx = await browser.new_context(
        viewport={"width": 1920, "height": 1080},
        record_video_dir=str(RAW_VIDEO_DIR),
        record_video_size={"width": 1920, "height": 1080},
    )
    page = await ctx.new_page()
    await page.add_init_script(
        "window.__presentedFrames = 0;"
        "(function tick(){ window.__presentedFrames += 1;"
        " requestAnimationFrame(tick); })();"
    )
    await page.goto(
        f"http://127.0.0.1:{PORT}/visualizer/?capture=1",
        wait_until="networkidle",
    )
    await page.wait_for_function(
        "window.__SHBT_ENGINE__ !== undefined", timeout=120_000
    )
    return browser, ctx, page


async def scrub(page, legs):
    for z in legs:
        await page.evaluate(
            "(z) => window.__SHBT_ENGINE__.setRedshift(z)", z
        )
        await wait_z(page, z)
        await relax(page)


async def record_segment(pw, name, legs, glitch_enabled, glitch_intensity):
    from playwright.async_api import async_playwright  # noqa: F401

    args = [
        "--enable-unsafe-webgpu",
        "--use-angle=d3d11",
        "--disable-dawn-features=use_dxc",
        "--ignore-gpu-blocklist",
        "--no-sandbox",
    ]
    browser, ctx, page = await open_capture(pw, args)
    try:
        await page.evaluate(
            "(b) => window.__SHBT_ENGINE__.setGlitchEnabled(b)",
            glitch_enabled,
        )
        await page.evaluate(
            "(v) => window.__SHBT_ENGINE__.setGlitchIntensity(v)",
            glitch_intensity,
        )
        print(
            f"[GLITCH-REC] {name}: glitch_enabled={glitch_enabled} "
            f"intensity={glitch_intensity}, legs={legs}"
        )
        await scrub(page, legs)
        tele = await telemetry(page)
        print(f"[GLITCH-REC] {name}: final telemetry {tele}")
        video = page.video
        await ctx.close()
        raw = Path(await video.path())
        out = RECORDINGS_DIR / name
        if raw.exists():
            raw.replace(out)
            print(f"[GLITCH-REC] {name}: {out.stat().st_size:,} bytes")
        else:
            print(f"[GLITCH-REC] {name}: WARNING no video produced")
        return tele
    finally:
        await browser.close()


async def main() -> int:
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    RAW_VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    srv = start_server()
    _ = srv
    from playwright.async_api import async_playwright

    results = {}
    async with async_playwright() as pw:
        results["clean"] = await record_segment(
            pw,
            "emergent_seed_formation_clean.webm",
            legs=[30.0, 7.0],
            glitch_enabled=False,
            glitch_intensity=0.0,
        )
        results["nuance"] = await record_segment(
            pw,
            "emergent_seed_glitch_nuance.webm",
            legs=[14.0, 7.0],
            glitch_enabled=True,
            glitch_intensity=0.1,
        )
        results["cycle"] = await record_segment(
            pw,
            "shbt_full_thermodynamic_cycle.webm",
            legs=[1.0e14, 1.0e10, 14.0, 3.0, 0.5, -0.999],
            glitch_enabled=True,
            glitch_intensity=0.1,
        )
    (RECORDINGS_DIR / "glitch_suite_telemetry.json").write_text(
        json.dumps(results, indent=2, default=str)
    )
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
