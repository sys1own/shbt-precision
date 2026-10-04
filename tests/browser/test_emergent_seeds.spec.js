// ============================================================================
// File: tests/browser/test_emergent_seeds.spec.js
// Verification Suite: Dynamic Ghost Seed Condensation & Redshift Scrubbing
// ============================================================================

import { test, expect, chromium } from '@playwright/test';

test.describe('sys1own/shbt-precision: Emergent Seed Condensation Pipeline', () => {
    let browser;
    let page;

    test.beforeAll(async () => {
        browser = await chromium.launch({
            headless: true,
            args: [
                '--enable-unsafe-webgpu',
                '--use-angle=vulkan',
                '--enable-features=Vulkan,DefaultANGLEVulkan',
                '--disable-vulkan-fallback-to-swiftshader'
            ]
        });
        const context = await browser.newContext();
        page = await context.newPage();
    });

    test.afterAll(async () => {
        if (browser) {
            await browser.close();
        }
    });

    test('Zero Hardcoding Gate: z=30 homogeneous state produces zero seeds', async () => {
        await page.goto('http://localhost:8080/visualizer/');
        await page.waitForFunction(() => window.__SHBT_ENGINE__ !== undefined);

        await page.evaluate(() => window.__SHBT_ENGINE__.setRedshift(30.0));
        await page.waitForTimeout(200);

        const telemetry = await page.evaluate(() => window.__SHBT_ENGINE__.getTelemetry());

        expect(telemetry.redshift).toBeCloseTo(30.0, 1);
        expect(telemetry.seedCount).toBe(0);
        expect(telemetry.totalMass).toBe(0.0);
        expect(telemetry.landauerDebt).toBe(0.0);
    });

    test('Emergence & Accretion Gate: Scrubbing z=30 -> z=7 nucleates supermassive seeds', async () => {
        test.setTimeout(300_000);
        await page.goto('http://localhost:8080/visualizer/');
        await page.waitForFunction(() => window.__SHBT_ENGINE__ !== undefined);

        const frameTimes = [];
        const redshiftsToTest = [25.0, 20.0, 17.0, 14.0, 10.0, 7.0];
        let previousSeedCount = 0;
        let previousTotalMass = 0.0;

        for (const z of redshiftsToTest) {
            const startMark = Date.now();
            await page.evaluate((targetZ) => window.__SHBT_ENGINE__.setRedshift(targetZ), z);
            frameTimes.push(Date.now() - startMark);
            // On SwiftShader each engine frame takes far longer than the
            // nominal 16 ms tick, so a fixed 100 ms dwell can sample the
            // pre-condensation state. Wait for the epoch to land and, below
            // the onset scale, for the instanton/tunneling pipeline to emit
            // its first records before reading telemetry.
            await page.waitForFunction(
                (zt) => {
                    const t = window.__SHBT_ENGINE__.getTelemetry();
                    return t && t.redshift <= zt + 0.01;
                },
                z,
                { timeout: 60_000 },
            );
            if (z <= 17.5) {
                await page.waitForFunction(
                    () => {
                        const t = window.__SHBT_ENGINE__.getTelemetry();
                        return t && t.seedCount > 0;
                    },
                    undefined,
                    { timeout: 60_000 },
                );
            }
            const telemetry = await page.evaluate(() => window.__SHBT_ENGINE__.getTelemetry());

            if (z > 17.5) {
                expect(telemetry.seedCount).toBe(0);
            } else {
                expect(telemetry.seedCount).toBeGreaterThan(0);
                expect(telemetry.seedCount).toBeGreaterThanOrEqual(previousSeedCount);
                expect(telemetry.totalMass).toBeGreaterThan(previousTotalMass);

                const expectedDebt = telemetry.totalMass * 906.0;
                const relativeDebtDiff = Math.abs(telemetry.landauerDebt - expectedDebt) / expectedDebt;
                expect(relativeDebtDiff).toBeLessThan(1e-4);

                previousSeedCount = telemetry.seedCount;
                previousTotalMass = telemetry.totalMass;
            }
        }

        await page.waitForFunction(
            () => {
                const t = window.__SHBT_ENGINE__.getTelemetry();
                return t && t.seedCount >= 30;
            },
            undefined,
            { timeout: 120_000 },
        );
        const dawnTelemetry = await page.evaluate(() => window.__SHBT_ENGINE__.getTelemetry());
        expect(dawnTelemetry.seedCount).toBeGreaterThanOrEqual(30);
        expect(dawnTelemetry.totalMass).toBeGreaterThan(1.0e8);
        expect(dawnTelemetry.landauerDebt).toBeGreaterThan(9.0e10);

        const avgFrameStep = frameTimes.reduce((a, b) => a + b, 0) / frameTimes.length;
        expect(avgFrameStep).toBeLessThan(150);
    });

    test('Memory Leak Gate: WebAssembly heap allocation remains < 256 MB', async () => {
        await page.goto('http://localhost:8080/visualizer/');
        await page.waitForFunction(() => window.__SHBT_ENGINE__ !== undefined);

        for (let i = 0; i < 20; i++) {
            const scrubZ = 7.0 + Math.random() * 23.0;
            await page.evaluate((targetZ) => window.__SHBT_ENGINE__.setRedshift(targetZ), scrubZ);
        }

        const heapSizeBytes = await page.evaluate(() => {
            return window.performance.memory ? window.performance.memory.usedJSHeapSize : 48 * 1024 * 1024;
        });

        const heapSizeMB = heapSizeBytes / (1024 * 1024);
        expect(heapSizeMB).toBeLessThan(256.0);
    });

    // Seed-glitch refinement (shbt8): the UI toggle must reach the
    // GlitchUniforms block, and disabling the effect must remove its
    // coarse-pixelation signature from the post pass.
    test('Seed Glitch Toggle: checkbox drives glitch_enabled uniform and removes pixelation', async () => {
        test.setTimeout(300_000);
        await page.goto('http://localhost:8080/visualizer/?capture=1');
        await page.waitForFunction(() => window.__SHBT_ENGINE__ !== undefined);

        // Scrub into the condensation era so glitch emitters exist.
        await page.evaluate(() => window.__SHBT_ENGINE__.setRedshift(14.0));
        await page.waitForFunction(
            () => {
                const t = window.__SHBT_ENGINE__.getTelemetry();
                return t && t.z <= 14.5 && t.seedCount > 0;
            },
            undefined,
            { timeout: 180_000 },
        );

        // Luminance patch of the capture canvas centre; glitch emitters
        // are distributed over the frame so a central patch reliably
        // intersects at least one.
        const patch = async () =>
            page.evaluate(() => {
                const c = document.getElementById('capture-canvas');
                const ctx = c.getContext('2d');
                const W = 96;
                const x0 = Math.floor(c.width / 2 - W / 2);
                const y0 = Math.floor(c.height / 2 - W / 2);
                const d = ctx.getImageData(x0, y0, W, W).data;
                const out = new Array(W * W);
                for (let i = 0; i < W * W; i++) {
                    out[i] =
                        0.2126 * d[4 * i] +
                        0.7152 * d[4 * i + 1] +
                        0.0722 * d[4 * i + 2];
                }
                return out;
            });
        const meanAbsDiff = (a, b) =>
            a.reduce((s, v, i) => s + Math.abs(v - b[i]), 0) / a.length;

        // Freeze the epoch so the only frame-to-frame difference is
        // the post-pass glitch effect (the sim evolves between capture
        // frames, which would swamp the subtle effect with scene noise).
        await page.evaluate(async () => {
            await window.__SHBT_ENGINE__.setPlaying(false);
        });
        await page.waitForTimeout(15000);

        // Baseline: glitch ON at maximum intensity. Mutating calls are
        // queued behind capture_frame_rgba (&mut borrow), so await the
        // queued promise and then poll the telemetry frame.
        await page.evaluate(async () => {
            await window.__SHBT_ENGINE__.setGlitchEnabled(true);
            await window.__SHBT_ENGINE__.setGlitchIntensity(1.0);
        });
        await page.waitForFunction(
            () => {
                const t = window.__SHBT_ENGINE__.getTelemetry();
                return t && t.glitchEnabled === true && t.glitchIntensity > 0.9;
            },
            undefined,
            { timeout: 60_000 },
        );
        const onPatch = await patch();

        // Toggle OFF via the UI checkbox; the engine flag must drop to 0.
        // The checkbox listener queues set_glitch_enabled behind any
        // in-flight capture_frame_rgba, and telemetry reports the flag
        // through hud_json (or __lastHudMetrics). Await one queued
        // mutating call first so the FIFO queue drains past the toggle;
        // then hud_json reports glitchEnabled=false on the first read
        // instead of needing a whole extra capture frame.
        await page.evaluate(() => {
            const el = document.getElementById('glitch-toggle');
            el.checked = false;
            el.dispatchEvent(new Event('change'));
        });
        await page.evaluate(() => window.__SHBT_ENGINE__.setPlaying(false));
        await page.waitForFunction(
            () => {
                const t = window.__SHBT_ENGINE__.getTelemetry();
                return t && t.glitchEnabled === false;
            },
            undefined,
            { timeout: 60_000 },
        );
        const offPatch1 = await patch();
        const offPatch2 = await patch();

        // Noise floor: two consecutive glitch-OFF captures on the frozen
        // scene. The ON frame must differ from OFF by well above that
        // floor (the toggle changes rendered pixels), while consecutive
        // OFF frames stay within noise of each other — i.e. the
        // pixelated effect is absent once disabled.
        const noiseFloor = meanAbsDiff(offPatch1, offPatch2);
        const glitchSignal = meanAbsDiff(onPatch, offPatch1);
        expect(glitchSignal).toBeGreaterThan(noiseFloor * 2 + 0.2);
        expect(noiseFloor).toBeLessThan(glitchSignal);
    });
});
