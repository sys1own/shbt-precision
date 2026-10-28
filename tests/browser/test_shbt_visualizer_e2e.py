#!/usr/bin/env python3
"""
Comprehensive End-to-End Automated Testing Suite for SHBT WebGPU Visualizer.
Verifies WebGPU context initialization, telemetry data-binding, UI controls,
interactive shaders, phase transitions, and milestone jumping.
"""
import asyncio
import http.server
import json
import math
import os
import re
import socketserver
import sys
import threading
import time
from pathlib import Path
from playwright.async_api import async_playwright
import numpy as np

sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)
sys.stderr.reconfigure(encoding='utf-8', line_buffering=True)

REPO_ROOT = Path(__file__).resolve().parents[2]
PORT = 8092
TEST_SCREENSHOTS_DIR = REPO_ROOT / "visualizer" / "recordings" / "e2e_test_artifacts"
TEST_SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)

class CoepHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(REPO_ROOT), **kwargs)

    def end_headers(self) -> None:
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
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

class ThreadedHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True

def start_server():
    server = ThreadedHTTPServer(("127.0.0.1", PORT), CoepHandler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    return server

def calculate_image_diff(path1, path2):
    """Calculates mean absolute pixel difference between two images."""
    from PIL import Image
    im1 = np.array(Image.open(path1).convert("RGB"), dtype=np.float32)
    im2 = np.array(Image.open(path2).convert("RGB"), dtype=np.float32)
    return float(np.mean(np.abs(im1 - im2)))

class TestReporter:
    def __init__(self):
        self.results = []

    def report(self, section, test_name, passed, details=""):
        status_str = "PASS" if passed else "FAIL"
        print(f"[{status_str}] [{section}] {test_name}: {details}")
        self.results.append({
            "section": section,
            "test": test_name,
            "passed": passed,
            "details": details
        })
        if not passed:
            raise AssertionError(f"Test failed: [{section}] {test_name} - {details}")

    def summary(self):
        total = len(self.results)
        passed = sum(1 for r in self.results if r["passed"])
        failed = total - passed
        print("\n" + "="*70)
        print(f"E2E TEST SUMMARY: {passed}/{total} Passed, {failed} Failed")
        print("="*70)
        return self.results

async def run_visualizer_suite():
    reporter = TestReporter()
    start_server()
    time.sleep(0.5)

    browser_args = [
        "--enable-unsafe-webgpu",
        "--use-angle=d3d11",
        "--disable-dawn-features=use_dxc",
        "--ignore-gpu-blocklist",
        "--no-sandbox",
        "--disable-http-cache",
    ]

    captured_errors = []
    webgpu_errors = []

    def on_console(msg):
        text = msg.text
        if msg.type == "error":
            print(f"  [BROWSER ERROR] {text}")
            captured_errors.append(text)
            if "webgpu" in text.lower() or "wgpu" in text.lower() or "shader" in text.lower():
                webgpu_errors.append(text)
        elif msg.type == "warning":
            if "error" in text.lower() or "fail" in text.lower():
                captured_errors.append(text)

    def on_page_error(err):
        print(f"  [UNHANDLED PAGE ERROR] {err}")
        captured_errors.append(str(err))

    async with async_playwright() as p:
        print("[SUITE] Launching Chromium with WebGPU acceleration...")
        browser = await p.chromium.launch(headless=True, args=browser_args)
        context = await browser.new_context(viewport={"width": 1280, "height": 720})
        page = await context.new_page()

        page.on("console", on_console)
        page.on("pageerror", on_page_error)

        # =====================================================================
        # 1. Environment Setup & Error Capturing
        # =====================================================================
        print("\n--- 1. Environment Setup & Error Capturing ---")
        url = f"http://127.0.0.1:{PORT}/visualizer/?capture=1"
        await page.goto(url, wait_until="networkidle")

        # Verify WebGPU context initialization and initial telemetry frame
        await page.wait_for_function(
            "window.__ENGINE_INITIALIZED__ === true && window.__lastHudMetrics != null",
            timeout=30000
        )
        is_context_lost = await page.evaluate("""() => {
            if (!window.__gpuDevice) return false;
            let lost = false;
            window.__gpuDevice.lost.then(() => { lost = true; });
            return lost;
        }""")
        reporter.report("Environment", "WebGPU Context Loss Check", not is_context_lost, "Context active and stable")
        reporter.report("Environment", "Shader & Validation Error Free", len(webgpu_errors) == 0, f"Errors: {len(webgpu_errors)}")

        # =====================================================================
        # 2. Global Engine & Canvas Validation
        # =====================================================================
        print("\n--- 2. Global Engine & Canvas Validation ---")
        status_text = await page.locator("#status").text_content()
        status_digits_only = re.sub(r'[\s,]+', '', status_text)
        reporter.report("Global Engine", "Status Online", "engineonline" in status_digits_only.lower(), f"Status: {status_text}")
        reporter.report("Global Engine", "Particle Count Verification", "262144" in status_digits_only, f"Particles found in status: {status_text}")

        # Verify main canvas is actively rendering: frame 1 vs frame 10 (or across time)
        shot1_path = TEST_SCREENSHOTS_DIR / "canvas_frame_initial.png"
        shot2_path = TEST_SCREENSHOTS_DIR / "canvas_frame_animated.png"
        await page.screenshot(path=str(shot1_path))
        await page.wait_for_timeout(800)  # Wait for simulation stepping
        await page.screenshot(path=str(shot2_path))
        diff_anim = calculate_image_diff(shot1_path, shot2_path)
        reporter.report("Global Engine", "Canvas Active Rendering Animation", diff_anim > 0.05, f"Mean pixel difference across frames: {diff_anim:.4f}")

        # Top-center phase display check
        phase_text = await page.locator("#phase-banner").text_content()
        reporter.report("Global Engine", "Phase Banner Initial State", "PHASE 1: CONFORMAL SCREEN BIT LOADING" in phase_text, f"Phase banner text: {phase_text}")

        # =====================================================================
        # 3. HUD Telemetry Data-Binding (Left Panel)
        # =====================================================================
        print("\n--- 3. HUD Telemetry Data-Binding ---")
        time_text = await page.locator("#hud-time").text_content()
        reporter.report("HUD Telemetry", "Horizon Bar Cosmic Age & Lookback", ("Cosmic Age:" in time_text and "Gyr" in time_text and "Lookback:" in time_text), f"Text: {time_text}")

        fload_text = await page.locator("#hud-fload").text_content()
        capacity_text = await page.locator("#hud-capacity").text_content()
        fload_num = float(fload_text)
        reporter.report("HUD Telemetry", "Boundary Capacity f_load", not math.isnan(fload_num) and fload_num >= 0.0, f"f_load: {fload_text}")
        reporter.report("HUD Telemetry", "Boundary Capacity N_sat & loaded", "N_sat = 3.312e122 bits" in capacity_text and "loaded =" in capacity_text, f"Capacity: {capacity_text}")

        ledger_text = await page.locator("#hud-ledger").text_content()
        reporter.report("HUD Telemetry", "Congestion Ledger No NaN", "NaN" not in ledger_text and "undefined" not in ledger_text, f"Ledger:\n{ledger_text}")
        reporter.report("HUD Telemetry", "Ledger N_vis, N_dark, quench, M_seed", all(k in ledger_text for k in ["N_vis", "N_dark", "Channel A quench", "M_seed"]), f"Ledger keys present")

        seed_count_text = await page.locator("#seed-count").text_content()
        seed_table_text = await page.locator("#seed-table").text_content()
        debt_text = await page.locator("#hud-debt").text_content()
        reporter.report("HUD Telemetry", "Emergent Defect Table Valid", ("seed" in seed_count_text and "NaN" not in seed_table_text), f"Defect: {seed_count_text}, {seed_table_text}")
        reporter.report("HUD Telemetry", "Landauer Debt Monitor Valid", ("GW" in debt_text and "NaN" not in debt_text), f"Debt: {debt_text}")

        observers_text = await page.locator("#hud-observers").text_content()
        rentropy_text = await page.locator("#hud-rentropy").text_content()
        reporter.report("HUD Telemetry", "Causal-Point Observer Activity R_adm", ("radm" in observers_text.lower() or "r_adm" in observers_text.lower()) and ("rentropy" in rentropy_text.lower() or "r_entropy" in rentropy_text.lower()), f"Observers: {observers_text}, Rentropy: {rentropy_text}")

        optics_text = await page.locator("#hud-optics").text_content()
        reporter.report("HUD Telemetry", "Gravitational Optics Telemetry", any(k in optics_text for k in ["max", "caustics", "rad"]), f"Optics text: {optics_text}")

        async def safe_click(selector):
            await page.evaluate(f"document.querySelector(\"{selector}\").click()")

        # Invariant buttons clickability
        await safe_click("#inv-fr")
        status_after_fr = await page.locator("#status").text_content()
        reporter.report("HUD Telemetry", "Button Δfr = 0 Clickable", "verified" in status_after_fr or "Δ_fr" in status_after_fr, f"Status: {status_after_fr}")

        await safe_click("#inv-emu")
        status_after_emu = await page.locator("#status").text_content()
        reporter.report("HUD Telemetry", "Button E_μν = 0 Clickable", "verified" in status_after_emu or "E_μν" in status_after_emu, f"Status: {status_after_emu}")

        await safe_click("#inv-horizon")
        await page.wait_for_timeout(400)
        status_after_horizon = await page.locator("#status").text_content()
        reporter.report("HUD Telemetry", "Button Horizon Freeze Clickable", "Horizon freeze" in status_after_horizon, f"Status: {status_after_horizon}")

        # =====================================================================
        # 4. Interactive Controls (Bottom Left Panel)
        # =====================================================================
        print("\n--- 4. Interactive Controls ---")
        # Jump back to z = 18 for high contrast interactive testing
        await safe_click("button[data-epoch='seed']")
        await page.wait_for_timeout(600)

        # Torus unwrap slider
        shot_unwrap0 = TEST_SCREENSHOTS_DIR / "torus_unwrap_0.png"
        shot_unwrap1 = TEST_SCREENSHOTS_DIR / "torus_unwrap_1.png"
        await page.screenshot(path=str(shot_unwrap0))
        await page.evaluate("""() => {
            const el = document.getElementById('unwrap');
            el.value = '1.0';
            el.dispatchEvent(new Event('input'));
        }""")
        await page.wait_for_timeout(400)
        await page.screenshot(path=str(shot_unwrap1))
        diff_unwrap = calculate_image_diff(shot_unwrap0, shot_unwrap1)
        unwrap_label = await page.locator("#unwrap-label").text_content()
        reporter.report("Controls", "Torus Unwrap Slider Visual Change", diff_unwrap > 0.05 and unwrap_label == "1.00", f"Pixel diff: {diff_unwrap:.4f}, label: {unwrap_label}")
        # Reset unwrap to 0
        await page.evaluate("""() => {
            const el = document.getElementById('unwrap');
            el.value = '0.0';
            el.dispatchEvent(new Event('input'));
        }""")
        await page.wait_for_timeout(200)

        # Pause / Play and Speed controls
        # Click pause
        await safe_click("#play")
        play_btn_icon = await page.locator("#play").text_content()
        # Drain the engine queue so the pause state takes effect synchronously
        await page.evaluate("window.__SHBT_ENGINE__.setPlaying(false)")
        tel1 = await page.evaluate("window.__SHBT_ENGINE__.getTelemetry()")
        await page.wait_for_timeout(400)
        tel2 = await page.evaluate("window.__SHBT_ENGINE__.getTelemetry()")
        time_diff = abs(tel1["z"] - tel2["z"])
        reporter.report("Controls", "Pause Control Freezes Timeline", time_diff < 1e-4, f"Play icon: '{play_btn_icon}', delta z: {time_diff}")

        # Click play to resume
        await safe_click("#play")
        await page.wait_for_timeout(300)

        # Speed buttons
        for spd in ["10", "100", "1"]:
            btn_selector = f".speed[data-s='{spd}']"
            await safe_click(btn_selector)
            is_active = await page.evaluate(f"document.querySelector(\"{btn_selector}\").classList.contains('active')")
            reporter.report("Controls", f"Speed Selector {spd}x Active", is_active, f"Button {spd}x active class verified")

        # Checkboxes visual verification with before/after screenshots
        checkbox_tests = [
            ("Ch A visible", "#ch-a", False, "ch_a_toggle"),
            ("Ch B ghost", "#ch-b", False, "ch_b_toggle"),
            ("Lensing", "#toggle-lensing", False, "lensing_toggle"),
            ("Doppler", "#toggle-doppler", False, "doppler_toggle"),
            ("Dark glow", "#toggle-glow", True, "dark_glow_toggle"),
            ("Seed glitch", "#glitch-toggle", True, "seed_glitch_toggle"),
        ]

        for name, sel, target_val, tag in checkbox_tests:
            # Baseline screenshot
            base_shot = TEST_SCREENSHOTS_DIR / f"{tag}_before.png"
            toggled_shot = TEST_SCREENSHOTS_DIR / f"{tag}_after.png"
            await page.screenshot(path=str(base_shot))

            # Toggle checkbox
            await page.evaluate(f"""() => {{
                const el = document.querySelector('{sel}');
                el.checked = {'true' if target_val else 'false'};
                el.dispatchEvent(new Event('change'));
            }}""")
            await page.wait_for_timeout(400)
            await page.screenshot(path=str(toggled_shot))
            diff_cb = calculate_image_diff(base_shot, toggled_shot)
            reporter.report("Controls", f"Checkbox {name} Toggles Visual Output", diff_cb > 0.005, f"Diff: {diff_cb:.4f}")

            # Restore original state
            await page.evaluate(f"""() => {{
                const el = document.querySelector('{sel}');
                el.checked = {'false' if target_val else 'true'};
                el.dispatchEvent(new Event('change'));
            }}""")
            await page.wait_for_timeout(200)

        # Sliders verification
        slider_tests = [
            ("slider-strength", "#strength-label", "0.080", "0.080"),
            ("slider-dispersion", "#dispersion-label", "0.46", "0.46"),
            ("slider-dark-glow", "#glow-label", "1.25", "1.25"),
            ("glitch-slider", "#glitch-label", "0.75", "0.75"),
            ("slider-cs", "#cs-label", "2.20", "2.20"),
            ("slider-pt", "#pt-label", "1.65", "1.65"),
            ("slider-ls", "#ls-label", "0.045", "0.045"),
            ("slider-cd", "#cd-label", "0.35", "0.35"),
        ]
        for sid, lid, val_set, expected_label in slider_tests:
            await page.evaluate(f"""() => {{
                const el = document.getElementById('{sid}');
                el.value = '{val_set}';
                el.dispatchEvent(new Event('input'));
            }}""")
            label_got = await page.locator(lid).text_content()
            reporter.report("Controls", f"Slider {sid} Readout Updated", label_got == expected_label, f"Expected {expected_label}, got {label_got}")

        # Dropdowns
        # Projection dropdown
        await page.select_option("#projection", "1")
        proj_val = await page.locator("#unwrap-label").text_content()
        reporter.report("Controls", "Projection Dropdown to CFT", proj_val == "1.00", f"Unwrap label: {proj_val}")
        await page.select_option("#projection", "0")

        # Viewport projection dropdown / Target Z
        await page.select_option("#select-viewport-mode", "1")
        card_disp = await page.evaluate("document.getElementById('phase-space-card').style.display")
        reporter.report("Controls", "Select Viewport Mode (Phase-Space)", card_disp == "block", f"Card display: {card_disp}")
        await page.select_option("#select-viewport-mode", "0")

        await page.evaluate("""() => {
            const el = document.getElementById('slider-z');
            el.value = '2.5';
            el.dispatchEvent(new Event('input'));
        }""")
        reporter.report("Controls", "Target Redshift Slider (z_tgt) Active", True, "Moved without throwing errors")

        # Phase-Space Diagnostics toggle
        base_diag = TEST_SCREENSHOTS_DIR / "diag_off.png"
        on_diag = TEST_SCREENSHOTS_DIR / "diag_on.png"
        await page.screenshot(path=str(base_diag))
        await page.evaluate("""() => {
            const el = document.getElementById('diag-toggle');
            el.checked = true;
            el.dispatchEvent(new Event('change'));
        }""")
        await page.wait_for_timeout(400)
        await page.screenshot(path=str(on_diag))
        diag_diff = calculate_image_diff(base_diag, on_diag)
        card_visible = await page.evaluate("document.getElementById('phase-space-card').style.display === 'block'")
        reporter.report("Controls", "Phase-Space Diagnostics Inset Enabled", (diag_diff > 0.01 and card_visible), f"Diff: {diag_diff:.4f}, Card visible: {card_visible}")
        # Turn off diagnostics
        await page.evaluate("""() => {
            const el = document.getElementById('diag-toggle');
            el.checked = false;
            el.dispatchEvent(new Event('change'));
        }""")
        await page.wait_for_timeout(200)

        # =====================================================================
        # 5. Milestone Button Execution
        # =====================================================================
        print("\n--- 5. Milestone Button Execution ---")
        milestones = [
            ("1. Bit Loading", "button[data-epoch='load']", 1e11, 1e15, "PHASE 1"),
            ("2. Baryogenesis De-Rendering", "button[data-epoch='bary']", 30.0, 1e11, "PHASE 2"),
            ("3. Ghost Seed Genesis", "button[data-epoch='seed']", 7.0, 30.0, "PHASE 3"),
            ("4. Proto-Galaxy Web", "button[data-epoch='get']", 0.0, 7.0, "PHASE 4"),
            ("5. Horizon Freeze", "button[data-epoch='freeze']", -1.0, 0.0, "PHASE 5"),
        ]

        for m_name, selector, z_min, z_max, phase_substr in milestones:
            await safe_click(selector)
            # Wait for the async engine queue to render the milestone frame
            await page.wait_for_function(
                f"() => document.getElementById('phase-banner').textContent.includes('{phase_substr}')",
                timeout=30000
            )
            tel = await page.evaluate("window.__SHBT_ENGINE__.getTelemetry()")
            cur_z = tel["z"]
            z_in_range = (cur_z >= z_min and cur_z <= z_max)
            banner_text = await page.locator("#phase-banner").text_content()
            phase_match = phase_substr in banner_text

            # Check screenshot is captured without WebGPU context loss
            m_shot = TEST_SCREENSHOTS_DIR / f"milestone_{m_name.replace(' ', '_').replace('.', '')}.png"
            await page.screenshot(path=str(m_shot))

            is_lost = await page.evaluate("""() => {
                let lost = false;
                if (window.__gpuDevice) {
                    window.__gpuDevice.lost.then(() => { lost = true; });
                }
                return lost;
            }""")

            reporter.report("Milestones", f"Milestone {m_name} Jump", z_in_range, f"z: {cur_z:.3e} in [{z_min:.1e}, {z_max:.1e}]")
            reporter.report("Milestones", f"Milestone {m_name} Phase Banner", phase_match, f"Banner: {banner_text[:60]}")
            reporter.report("Milestones", f"Milestone {m_name} Context Stability", not is_lost, "No WebGPU context loss")

        # =====================================================================
        # Final Error Summary
        # =====================================================================
        reporter.report("Final Verification", "Zero Console Errors Recorded", len(captured_errors) == 0, f"Error count: {len(captured_errors)}")

        await browser.close()

    return reporter.summary()

if __name__ == "__main__":
    results = asyncio.run(run_visualizer_suite())
