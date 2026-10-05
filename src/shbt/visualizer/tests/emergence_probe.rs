//! Probe: replicate the cpu_emergence_tick detection core on the static
//! seed lattice and measure candidate/merged counts vs the shbt13
//! dynamic-threshold margin, at several redshifts.

use shbt_visualizer::Particle;

const GRID_DIM: i32 = 32;
const BOX_SIZE: f32 = 200.0;
const MAX_SEEDS: usize = 1024;
const GAMMA_CFT: f32 = 1325.0 / 924.0;
const Z_REF: f32 = 17.5;
const N_PARTICLES: usize = 262_144;
const DELTA_C0: f32 = 1.686;

fn cardy_limit_norm(z: f32) -> f32 {
    let rel = (1.0 + z).max(1.0e-3) / (1.0 + Z_REF);
    GAMMA_CFT * rel.powf(7.5)
}

// H_SHBT/H_0 from the two-tier cosmology (matches telemetry.rs hubble).
fn h_ratio(z: f32) -> f64 {
    let opz = 1.0f64 + z as f64;
    let e = (0.315 * opz.powi(3) + 9.2e-5 * opz.powi(4) + 0.685).sqrt();
    (1.0 + (4.797960072861 / 67.4) / opz) * e
}

fn psi_nuc(z: f32) -> f32 {
    let u = ((30.0 - z) / 12.0).clamp(0.0, 1.0);
    u * u * u * (10.0 + u * (-15.0 + 6.0 * u))
}

// delta_th in normalized density units (n = rho/mean).
fn delta_th_n(z: f32, f_cosmo: f64) -> f32 {
    let psi = psi_nuc(z).max(1.0e-3);
    let h = h_ratio(z) as f32;
    let corr = 1.0 - (10.0 / 33.0) * f_cosmo as f32;
    DELTA_C0 * h.powf(1.0 / 3.0) * corr / psi
}

fn build_grid() -> Vec<f32> {
    let particles = Particle::seed_lattice(N_PARTICLES, BOX_SIZE);
    let dim = GRID_DIM;
    let inv_cell = GRID_DIM as f32 / BOX_SIZE;
    let mut grid = vec![0f32; (dim * dim * dim) as usize];
    for p in &particles {
        let gx = p.position[0] * inv_cell;
        let gy = p.position[1] * inv_cell;
        let gz = p.position[2] * inv_cell;
        let (bx, by, bz) = (gx.floor() as i32, gy.floor() as i32, gz.floor() as i32);
        let (fx, fy, fz) = (gx - bx as f32, gy - by as f32, gz - bz as f32);
        for dz in 0..2 {
            let wz = if dz == 1 { fz } else { 1.0 - fz };
            let zc = ((bz + dz + dim) % dim) as usize;
            for dy in 0..2 {
                let wy = if dy == 1 { fy } else { 1.0 - fy };
                let yc = ((by + dy + dim) % dim) as usize;
                for dx in 0..2 {
                    let wx = if dx == 1 { fx } else { 1.0 - fx };
                    let xc = ((bx + dx + dim) % dim) as usize;
                    grid[zc * 1024 + yc * 32 + xc] += p.grav_mass * wx * wy * wz;
                }
            }
        }
    }
    grid
}

/// Detection replicating cpu_emergence_tick: overflow gate + NMS with an
/// optional margin in n-units (0.0 = strict NMS).
fn detect_uncapped(
    grid: &[f32],
    mean: f32,
    n_limit: f32,
    margin_n: f32,
) -> Vec<(f32, [f32; 3], f32)> {
    let dim = GRID_DIM;
    let idx = |x: i32, y: i32, z: i32| -> usize {
        (((z + dim) % dim) * 1024 + ((y + dim) % dim) * 32 + ((x + dim) % dim)) as usize
    };
    let n_of = |i: usize| -> f32 { grid[i] / mean.max(1e-5) };
    let cell = BOX_SIZE / GRID_DIM as f32;
    let mut out = Vec::new();
    for z in 0..dim {
        for y in 0..dim {
            for x in 0..dim {
                let n = n_of(idx(x, y, z));
                let r = n / n_limit;
                if r - 1.0 <= 0.02 {
                    continue;
                }
                let mut is_max = true;
                'nms: for dz in -1..=1 {
                    for dy in -1..=1 {
                        for dx in -1..=1 {
                            if dx == 0 && dy == 0 && dz == 0 {
                                continue;
                            }
                            let nn = n_of(idx(x + dx, y + dy, z + dz));
                            if nn - n > margin_n
                                || (nn == n && idx(x + dx, y + dy, z + dz) < idx(x, y, z))
                            {
                                is_max = false;
                                break 'nms;
                            }
                        }
                    }
                }
                if !is_max {
                    continue;
                }
                // Basin integrate.
                let mut sum_ov = 0f32;
                let mut wp = [0f32; 3];
                for dz in -1..=1 {
                    for dy in -1..=1 {
                        for dx in -1..=1 {
                            let ni = idx(x + dx, y + dy, z + dz);
                            let c = (n_of(ni) - n_limit).max(0.0);
                            sum_ov += c;
                            wp[0] += ((x + dx) as f32 * cell) * c;
                            wp[1] += ((y + dy) as f32 * cell) * c;
                            wp[2] += ((z + dz) as f32 * cell) * c;
                        }
                    }
                }
                let cpos = [
                    (wp[0] / sum_ov.max(1e-6) + BOX_SIZE) % BOX_SIZE,
                    (wp[1] / sum_ov.max(1e-6) + BOX_SIZE) % BOX_SIZE,
                    (wp[2] / sum_ov.max(1e-6) + BOX_SIZE) % BOX_SIZE,
                ];
                out.push((0f32, cpos, sum_ov * 2.0e7));
            }
        }
    }
    out
}

/// Agglomerative merging at r_merge = 1.25 * dx_cell (spec Stage 3).
fn merge(pool: &mut Vec<(f32, [f32; 3], f32)>) {
    let r_merge = 1.25 * (BOX_SIZE / GRID_DIM as f32);
    let r2 = r_merge * r_merge;
    let n = pool.len();
    let mut dead = vec![false; n];
    for i in 0..n {
        if dead[i] {
            continue;
        }
        for j in (i + 1)..n {
            if dead[j] {
                continue;
            }
            let mut diff = [
                (pool[i].1[0] - pool[j].1[0]).abs(),
                (pool[i].1[1] - pool[j].1[1]).abs(),
                (pool[i].1[2] - pool[j].1[2]).abs(),
            ];
            for d in diff.iter_mut() {
                *d = d.min(BOX_SIZE - *d);
            }
            if diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2] < r2 {
                pool[i].2 += pool[j].2;
                dead[j] = true;
            }
        }
    }
    pool.retain(|s| s.2 > 0.0);
    let mut keep = Vec::new();
    for (i, s) in pool.iter().enumerate() {
        if !dead[i] {
            keep.push(*s);
        }
    }
    *pool = keep;
}

#[test]
fn probe_counts() {
    let grid = build_grid();
    let mean = grid.iter().sum::<f32>() / grid.len() as f32;
    for &z in &[25f32, 18.0, 16.0, 10.0, 7.0, 3.0, 1.0, 0.0] {
        let nl = cardy_limit_norm(z);
        let dth = delta_th_n(z, 0.1);
        for margin in [0.0f32, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4] {
            let cands = detect_uncapped(&grid, mean, nl, margin);
            let mut pool = cands.clone();
            merge(&mut pool);
            eprintln!(
                "z={z:>5} nlim={nl:.2e} margin={margin:>6.2} cands={} merged_full={}",
                cands.len(),
                pool.len()
            );
        }
    }
}
