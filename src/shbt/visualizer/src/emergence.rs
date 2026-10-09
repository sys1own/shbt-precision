//! Emergence analysis and agglomerative halo merging module.
//!
//! Replaces empirical heuristics with exact Causal-Point light-cone packing
//! derived from the D_5 arithmetic skeleton.

use std::collections::HashMap;
use std::f32::consts::PI;
use bytemuck::{Pod, Zeroable};

/// Canonical Kac-Moody boundary affine level k_l (critical string dimension)
pub const K_L: f32 = 26.0;
/// Canonical transverse gauge symmetry rank k_q
pub const K_Q: f32 = 8.0;
/// Total holographic saturation capacity K
pub const K_TOTAL: f32 = 312.0;
/// Effective Virasoro central charge c_eff = k_l - k_q
pub const C_EFF: f32 = 18.0;

/// First-principles holographic diffusion coefficient: 2 / (17 * PI)
pub const KAPPA_DIFF: f32 = 2.0 / (17.0 * PI); // 0.037447463
/// 2D site percolation threshold
pub const P_C_SITE: f32 = 0.59274621;
/// Stinespring visible partition
pub const ETA_V: f32 = 10.0 / 33.0;

/// Canonical cosmological saturation entropy scale: ln(N_sat) ~ 284.470588
pub const CANONICAL_LN_N_SAT: f32 = 284.470588;

/// Arithmetic cell measure for the 5-dimensional root lattice D_5.
pub const KAPPA_STAR_D5: f64 = 0.988725345719;

/// Exact analytical agglomerative merge radius coefficient:
/// r_merge = sqrt(2) * kappa_*^D5 * dx_cell ≈ 1.398249068038 * dx_cell.
pub const D5_MERGE_COEFFICIENT: f64 = 1.398249068038;

/// Particle representation within the holographic boundary register.
#[derive(Clone, Copy, Debug)]
pub struct Particle {
    pub id: u64,
    pub position: [f64; 3],
    pub mass: f64,
    pub halo_id: Option<u64>,
}

/// Agglomerative halo merger driven by the D_5 light-cone packing radius.
pub struct HaloMerger {
    pub dx_cell: f64,
    pub r_merge: f64,
    pub r_merge_sq: f64,
}

impl HaloMerger {
    /// Constructs a new `HaloMerger` parameterized by the exact D_5 packing metric.
    pub fn new(dx_cell: f64) -> Self {
        let r_merge = D5_MERGE_COEFFICIENT * dx_cell;
        let r_merge_sq = r_merge * r_merge;
        Self {
            dx_cell,
            r_merge,
            r_merge_sq,
        }
    }

    /// Computes the spatial hash voxel key using the exact merge radius as bucket size.
    #[inline(always)]
    fn spatial_key(&self, pos: &[f64; 3]) -> (i64, i64, i64) {
        (
            (pos[0] / self.r_merge).floor() as i64,
            (pos[1] / self.r_merge).floor() as i64,
            (pos[2] / self.r_merge).floor() as i64,
        )
    }

    /// Merges particles into causally connected halos based on light-cone horizon overlap.
    pub fn merge_halos(&self, particles: &mut [Particle]) {
        if particles.is_empty() {
            return;
        }

        let mut grid: HashMap<(i64, i64, i64), Vec<usize>> = HashMap::with_capacity(particles.len());

        for (idx, p) in particles.iter().enumerate() {
            let key = self.spatial_key(&p.position);
            grid.entry(key).or_default().push(idx);
        }

        let mut parent: Vec<usize> = (0..particles.len()).collect();

        fn find(parent: &mut [usize], mut i: usize) -> usize {
            let mut root = i;
            while root != parent[root] {
                root = parent[root];
            }
            while i != root {
                let nxt = parent[i];
                parent[i] = root;
                i = nxt;
            }
            root
        }

        fn union(parent: &mut [usize], i: usize, j: usize) {
            let root_i = find(parent, i);
            let root_j = find(parent, j);
            if root_i != root_j {
                parent[root_i] = root_j;
            }
        }

        for (&(cx, cy, cz), indices) in &grid {
            for dx in -1..=1 {
                for dy in -1..=1 {
                    for dz in -1..=1 {
                        let neighbor_key = (cx + dx, cy + dy, cz + dz);
                        if let Some(neighbor_indices) = grid.get(&neighbor_key) {
                            for &i in indices {
                                for &j in neighbor_indices {
                                    if i >= j {
                                        continue;
                                    }
                                    let p1 = &particles[i];
                                    let p2 = &particles[j];
                                    let dist_sq = (p1.position[0] - p2.position[0]).powi(2)
                                        + (p1.position[1] - p2.position[1]).powi(2)
                                        + (p1.position[2] - p2.position[2]).powi(2);

                                    if dist_sq <= self.r_merge_sq {
                                        union(&mut parent, i, j);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        for i in 0..particles.len() {
            let root = find(&mut parent, i);
            particles[i].halo_id = Some(root as u64);
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct EmergenceUniforms {
    pub z_start: f32,
    pub z_end: f32,
    pub kappa_diff: f32,
    pub eta_v: f32,
    pub current_z: f32,
    pub psi_nuc: f32,
    pub _padding: [f32; 2],
}

impl EmergenceUniforms {
    pub fn new(current_z: f32) -> Self {
        let prefactor = (C_EFF * (K_L + K_Q)) / (18.0 * K_TOTAL); // 17.0 / 156.0
        let z_start = (prefactor * CANONICAL_LN_N_SAT) - 1.0; // 30.000000
        let z_end = z_start * P_C_SITE; // 17.782386

        let psi_nuc = if current_z >= z_start {
            0.0
        } else if current_z <= z_end {
            1.0
        } else {
            let u = (z_start - current_z) / (z_start - z_end);
            u * u * u * (u * (u * 6.0 - 15.0) + 10.0)
        };

        Self {
            z_start,
            z_end,
            kappa_diff: KAPPA_DIFF,
            eta_v: ETA_V,
            current_z,
            psi_nuc,
            _padding: [0.0; 2],
        }
    }
}
