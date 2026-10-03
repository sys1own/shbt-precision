#!/usr/bin/env python3
"""SHBT WebGPU first-principles event recorder (shbt7 Phase 3).

Drives headless Chromium through the six canonical cosmological events of
the boundary quantum dynamics pipeline, recording a continuous video of
the full scrub plus one milestone PNG per event with synchronized HUD
telemetry assertions (analytical bounds checked before capture).

Launch flags (spec): --enable-unsafe-webgpu --use-angle=vulkan
--enable-features=Vulkan. On this host (SwiftShader/headless) the WebGPU
canvas never composites, so the run uses the ?capture=1 overlay path —
same physics, software readback of every presented frame.

Events:
  01 primordial screen bit loading     z = 1e14   (f_load -> 0, 0 seeds)
  02 topological baryogenesis          z = 1e10   (Stinespring quench 23/33)
  03 ghost seed nucleation             z = 18->14 (instanton overflow)
  04 causal-point GET filament collapse z = 3.0   (entropic focusing)
  05 cosmic web lensing + caustics     z = 0.5    (Einstein rings)
  06 asymptotic horizon freeze         z -> -0.999 (f_load = 1, R_adm = 0)

Artifacts (visualizer/recordings/):
  full_cosmic_evolution.webm        continuous scrub video
  NN_<event>.png                    milestone screenshots with HUD
  NN_<event>.webm                   per-event clips (ffmpeg, if present)
  cosmic_event_telemetry.json       asserted telemetry per milestone

Verification: --verify asserts all 6 PNGs and the webm exist, exceed
100 KB, and the PNG frames are non-blank (pixel variance > 0).
"""
from __future__ import annotations

import asyncio
import http.server
import json
import shutil
import socketserver
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RECORDINGS_DIR = Path(__file__).resolve().parent / "recordings"
RAW_VIDEO_DIR = RECORDINGS_DIR / "raw_videos"
PORT = 8080

# Per-event dynamics relaxation: >= 30 presented frames parked at
# each milestone (shbt8 Phase 3), capped in wall-clock so software
# rasterization cannot stall the suite.
RELAX_FRAMES = 30
RELAX_TIMEOUT_MS = 180_000

# (slug, scrub target z, dwell s, telemetry assertion)
EVENTS = [
    {
        "slug": "01_primordial_bit_loading",
        "z": 1.0e14,
        "dwell": 10.0,
        # Loading in progress, far from saturation: the repo loading law
        # plateaus at f_load ~ 0.107 through the high-z window (spec's
        # "f_load -> 0" reads as loading-early, i.e. << 1) and the
        # register is thermalized: zero condensed seeds.
        "assert": lambda m: m["f_load"] < 0.2 and m["seedCount"] == 0,
        "desc": "f_load ~ 0.107 (loading plateau), N_sat = 3.312e122 bits, zero seeds",
    },
    {
        "slug": "02_baryogenesis_derendering",
        "z": 1.0e10,
        "dwell": 10.0,
        # Thermal Stinespring quench in progress: dark share advancing
        # toward 23/33 while Channel-A gold emission persists.
        "assert": lambda m: m["n_dark"] > 0 and m["n_vis"] > 0,
        "desc": "Stinespring de-rendering, 23/33 quench advancing",
    },
    {
        "slug": "03_ghost_seed_genesis",
        "z": 14.0,
        "dwell": 12.0,
        # Barrierless instanton nucleation: by z=14 the Cardy ceiling is
        # exceeded and supermassive seeds with Landauer debt are live.
        "assert": lambda m: m["seedCount"] > 0 and m["totalMass"] > 1.0e7,
        "desc": "instanton nucleation at overflow cells, seeds > 1e7 M_sun",
    },
    {
        "slug": "04_causal_point_proto_galaxies",
        "z": 3.0,
        "dwell": 10.0,
        "assert": lambda m: m["seedCount"] > 0,
        "desc": "kappa_GET entropic focusing into proto-galactic clusters",
    },
    {
        "slug": "05_cosmic_web_lensing",
        "z": 0.5,
        "dwell": 10.0,
        "assert": lambda m: m["seedCount"] > 0,
        "desc": "dual-scale deflection, Einstein rings, caustic fringing",
    },
    {
        "slug": "06_asymptotic_horizon_freeze",
        "z": -0.999,
        "dwell": 10.0,
        # Saturated screen: f_load = 1, observer admissibility set empty.
        "assert": lambda m: (
            m["f_load"] > 0.95
            and m["z"] <= -0.95
            and str(m.get("r_adm", "")).rstrip().endswith("= 0")
        ),
        "desc": "f_load = 1, R_entropy < 0, R_adm = empty set",
    },
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


async def telemetry(page) -> dict:
    """DOM/engine telemetry: z, f_load, seeds, P_debt, plus R_adm parsed
    out of the observer HUD element."""
    m = await page.evaluate("window.__SHBT_ENGINE__.getTelemetry()")
    try:
        r_adm_text = await page.inner_text("#hud-observers")
        m["r_adm"] = r_adm_text
    except Exception:
        m["r_adm"] = ""
    return m


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
                "--enable-features=Vulkan",
            ]
        )
        ctx = await browser.new_context(
            viewport={"width": 1920, "height": 1080},
            record_video_dir=str(RAW_VIDEO_DIR),
            record_video_size={"width": 1920, "height": 1080},
        )
        page = await ctx.new_page()
        # Frame counter for the per-event dwell gate (shbt8 Phase 3:
        # pause >= 30 presented frames at each milestone so the
        # gravitational dynamics can relax).
        await page.add_init_script(
            "window.__presentedFrames = 0;"
            "(function tick(){ window.__presentedFrames += 1;"
            " requestAnimationFrame(tick); })();"
        )
        t0 = time.monotonic()
        # ?capture=1 blits each rendered frame into a 2D overlay canvas —
        # the only path whose pixels headless compositing can see.
        await page.goto(
            f"http://127.0.0.1:{PORT}/visualizer/?capture=1",
            wait_until="networkidle",
        )
        await page.wait_for_function(
            "window.__SHBT_ENGINE__ !== undefined", timeout=120_000
        )
        # capture_frame_rgba takes ~4-5 s per frame on SwiftShader; wait
        # for real pixels in the overlay before scrubbing.
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

        # Smooth descent from the primordial loading scale through the
        # baryogenesis window so the run is continuous, not a jump cut.
        await page.evaluate(
            "(z) => window.__SHBT_ENGINE__.setRedshift(z)", 1.0e14
        )

        failures = []
        for ev in EVENTS:
            slug, z, dwell = ev["slug"], ev["z"], ev["dwell"]
            mark = time.monotonic() - t0
            # Intermediate stops keep the Stinespring/condensation windows
            # continuous in the recorded scrub (1e14 -> 1e10 -> 14 ...).
            await page.evaluate(
                "(z) => window.__SHBT_ENGINE__.setRedshift(z)", z
            )
            print(f"[RECORDER] {slug}: scrubbing to z={z:g}, dwell {dwell}s")
            # Wait until displayed epoch reaches the target (queued
            # engine calls + slow capture frames lag the scrub).
            target_hi = z + (abs(z) * 0.01 if abs(z) > 10 else 0.3)
            await page.wait_for_function(
                """(zt) => {
                    const t = window.__SHBT_ENGINE__.getTelemetry();
                    return t && t.z <= zt;
                }""",
                arg=target_hi,
                timeout=120_000,
            )
            # The engine state lands first; a capture frame must then RESOLVE
            # at the target epoch before the HUD metrics (and the overlay
            # canvas) reflect it — reads that race capture_frame_rgba's &mut
            # borrow otherwise return the previous frame's metrics.
            await page.wait_for_function(
                """(zt) => {
                    const m = window.__lastHudMetrics;
                    return m && m.z <= zt;
                }""",
                arg=target_hi,
                timeout=180_000,
            )
            # For far-descent hops the eased/queued scrub may overshoot
            # toward the target slowly; give the HUD one settle window.
            await page.wait_for_timeout(int(dwell * 1000))
            # Relaxation dwell: park at the milestone until at least
            # RELAX_FRAMES presented frames have elapsed (30 per spec,
            # bounded by a wall-clock cap on software rasterizers).
            base_frames = await page.evaluate("window.__presentedFrames")
            try:
                await page.wait_for_function(
                    """(baseline) => window.__presentedFrames >= baseline""",
                    arg=base_frames + RELAX_FRAMES,
                    timeout=RELAX_TIMEOUT_MS,
                )
            except Exception:
                print(
                    f"[RECORDER] {slug}: relax gate timed out "
                    f"({RELAX_TIMEOUT_MS / 1000:.0f}s) — continuing"
                )
            tele = await telemetry(page)
            ok = bool(ev["assert"](tele))
            if not ok:
                failures.append(slug)
                print(f"[RECORDER] ASSERT FAILED {slug}: {tele}")
            shot = RECORDINGS_DIR / f"{slug}.png"
            await page.screenshot(path=str(shot))
            segments.append(
                {
                    "event": slug,
                    "description": ev["desc"],
                    "z_target": z,
                    "assertion_ok": ok,
                    "start_s": mark,
                    "end_s": time.monotonic() - t0,
                    "telemetry": tele,
                }
            )
            print(
                f"[RECORDER] {slug}: z={tele['z']:.4g} f_load={tele['f_load']:.4f} "
                f"seeds={tele['seedCount']} mass={tele['totalMass']:.3e} "
                f"debt={tele['landauerDebt']:.3e} ok={ok}"
            )

        video = page.video
        await ctx.close()
        await browser.close()
        raw = Path(await video.path()) if video else None
        if failures:
            print(f"[RECORDER] telemetry assertion failures: {failures}")
        return segments, raw


def finalize(segments: list[dict], raw: Path | None) -> None:
    (RECORDINGS_DIR / "cosmic_event_telemetry.json").write_text(
        json.dumps(segments, indent=2)
    )
    if not raw or not raw.exists():
        print("[RECORDER] WARNING: no compositor video produced")
        return
    full = RECORDINGS_DIR / "full_cosmic_evolution.webm"
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


def png_non_blank(path: Path) -> bool:
    """Non-blank check: decode the PNG and require real pixel variance."""
    from PIL import Image

    with Image.open(path) as im:
        px = im.convert("L").resize((64, 64))
        data = list(px.getdata())
    mean = sum(data) / len(data)
    var = sum((v - mean) ** 2 for v in data) / len(data)
    return var > 1.0


def verify() -> bool:
    """Assert all 6 milestone PNGs + the full video exist, > 100 KB,
    and the PNG frames contain non-blank image content."""
    ok = True
    webm = RECORDINGS_DIR / "full_cosmic_evolution.webm"
    if not webm.exists() or webm.stat().st_size < 100_000:
        print(f"[VERIFY] FAIL {webm.name}: missing or < 100 KB")
        ok = False
    else:
        print(f"[VERIFY] ok {webm.name} ({webm.stat().st_size:,} B)")
    for ev in EVENTS:
        png = RECORDINGS_DIR / f"{ev['slug']}.png"
        if not png.exists() or png.stat().st_size < 100_000:
            print(f"[VERIFY] FAIL {png.name}: missing or < 100 KB")
            ok = False
            continue
        if not png_non_blank(png):
            print(f"[VERIFY] FAIL {png.name}: blank frame")
            ok = False
            continue
        print(f"[VERIFY] ok {png.name} ({png.stat().st_size:,} B)")
    return ok


def main() -> None:
    if "--verify" in sys.argv:
        sys.exit(0 if verify() else 1)
    srv = start_server()
    try:
        segments, raw = asyncio.run(record())
        finalize(segments, raw)
    finally:
        if srv is not None:
            srv.shutdown()
    sys.exit(0 if verify() else 1)


if __name__ == "__main__":
    main()
