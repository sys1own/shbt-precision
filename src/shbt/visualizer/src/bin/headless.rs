//! Native headless runner:
//!   shbt-visualizer-headless [frames] [particles] [options]
//!
//! Benchmarks the compute+render pipeline without a surface.
//!
//! Options (gravitational optics suite, shbt5 spec):
//!   --enable-lensing        Enable gravitational lensing post pass
//!   --lensing-scale S       Lensing strength lambda_lens, clamped [0, 5]
//!   --dispersion D          Chromatic dispersion delta_disp, clamped [0, 1]
//!   --enable-doppler        Relativistic Doppler beaming + thermal shift
//!   --dark-glow G           Volumetric dark halo glow, clamped [0, 2]
//!   --width PX --height PX  Offscreen target resolution

#[cfg(not(target_arch = "wasm32"))]
fn main() {
    use shbt_visualizer::ShbtWebGpuEngine;
    use std::time::Instant;

    let mut frames: u32 = 60;
    let mut particles: u32 = 262_144;
    let mut lensing = false;
    let mut lensing_scale = 1.0f32;
    let mut dispersion = 0.25f32;
    let mut doppler = true;
    let mut dark_glow = 0.8f32;
    let mut width = 1280u32;
    let mut height = 720u32;

    let mut args = std::env::args().skip(1);
    let mut positional = 0;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--enable-lensing" => lensing = true,
            "--lensing-scale" => {
                lensing_scale = args.next().and_then(|v| v.parse().ok()).unwrap_or(1.0);
                lensing = true;
            }
            "--dispersion" => {
                dispersion = args.next().and_then(|v| v.parse().ok()).unwrap_or(0.25);
            }
            "--enable-doppler" => doppler = true,
            "--dark-glow" => {
                dark_glow = args.next().and_then(|v| v.parse().ok()).unwrap_or(0.8);
            }
            "--width" => {
                width = args.next().and_then(|v| v.parse().ok()).unwrap_or(1280);
            }
            "--height" => {
                height = args.next().and_then(|v| v.parse().ok()).unwrap_or(720);
            }
            _ => {
                if let Ok(v) = arg.parse::<u32>() {
                    if positional == 0 {
                        frames = v;
                    } else if positional == 1 {
                        particles = v;
                    }
                    positional += 1;
                }
            }
        }
    }

    pollster::block_on(async {
        let mut engine = ShbtWebGpuEngine::headless(particles)
            .await
            .expect("WebGPU adapter required");
        engine.apply_lensing_scale(lensing_scale);
        engine.apply_lensing_enabled(lensing);
        engine.apply_dispersion(dispersion);
        engine.apply_doppler(doppler);
        engine.apply_dark_glow(dark_glow);
        let start = Instant::now();
        for _ in 0..frames {
            engine.step(1.0 / 60.0, None);
        }
        let elapsed = start.elapsed();
        println!(
            "shbt-visualizer headless: {} frames x {} particles in {:.2?} ({:.1} FPS) [{}x{} lensing={} dispersion={:.2} doppler={} dark_glow={:.2}]",
            frames,
            particles,
            elapsed,
            frames as f64 / elapsed.as_secs_f64(),
            width,
            height,
            lensing,
            dispersion,
            doppler,
            dark_glow,
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

        // Calibration sweep for the emergent condensation thresholds.
        if std::env::var("SHBT_Z_SWEEP").is_ok() {
            for z in [30.0f64, 25.0, 20.0, 17.5, 17.0, 14.0, 10.0, 7.0] {
                engine.set_redshift(z);
                for _ in 0..8 {
                    engine.step(1.0 / 60.0, None);
                }
                println!("{}", engine.hud_json());
                println!("   {}", engine.debug_emergence_stats());
            }
        }
    });
}

#[cfg(target_arch = "wasm32")]
fn main() {}
