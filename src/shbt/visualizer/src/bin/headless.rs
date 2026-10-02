//! Native headless runner: `shbt-visualizer-headless [frames] [particles]`
//! Benchmarks the compute+render pipeline without a surface.

#[cfg(not(target_arch = "wasm32"))]
fn main() {
    use shbt_visualizer::ShbtWebGpuEngine;
    use std::time::Instant;

    let frames: u32 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(60);
    let particles: u32 = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(262_144);

    pollster::block_on(async {
        let mut engine = ShbtWebGpuEngine::headless(particles)
            .await
            .expect("WebGPU adapter required");
        let start = Instant::now();
        for _ in 0..frames {
            engine.step(1.0 / 60.0, None);
        }
        let elapsed = start.elapsed();
        println!(
            "shbt-visualizer headless: {} frames x {} particles in {:.2?} ({:.1} FPS)",
            frames,
            particles,
            elapsed,
            frames as f64 / elapsed.as_secs_f64()
        );
        // Touch the readback path once to validate the composite pass.
        let rgba = engine.render_to_rgba(1.0 / 60.0);
        let nonzero = rgba.iter().filter(|&&b| b != 0).count();
        println!(
            "readback frame: {} bytes, {} nonzero (hud: z={:.3e} f_load={:.4})",
            rgba.len(),
            nonzero,
            engine.hud_metrics().redshift,
            engine.hud_metrics().loading_frac
        );
    });
}

#[cfg(target_arch = "wasm32")]
fn main() {}
