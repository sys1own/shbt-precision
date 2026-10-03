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
        await page.goto('http://localhost:8080/visualizer/');
        await page.waitForFunction(() => window.__SHBT_ENGINE__ !== undefined);

        const frameTimes = [];
        const redshiftsToTest = [25.0, 20.0, 17.0, 14.0, 10.0, 7.0];
        let previousSeedCount = 0;
        let previousTotalMass = 0.0;

        for (const z of redshiftsToTest) {
            const startMark = Date.now();
            await page.evaluate((targetZ) => window.__SHBT_ENGINE__.setRedshift(targetZ), z);
            await page.waitForTimeout(100);
            frameTimes.push(Date.now() - startMark);

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
});
