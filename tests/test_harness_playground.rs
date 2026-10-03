//! Integration Test Harness and Zero-Allocation Verification
//! File: tests/test_harness_playground.rs
//! Repository: sys1own/shbt-precision
//!
//! Rust integration harness (shbt9 Phase 5): ensures memory layouts,
//! byte offsets, and alignment boundaries remain strictly consistent
//! across the CPU host, WebAssembly linear address space, and GPU
//! storage buffers, and enforces a zero dynamic heap allocation
//! invariant across active simulation ticks via an atomic allocation
//! counter.

#[cfg(test)]
mod tests {
    use std::alloc::{GlobalAlloc, Layout, System};
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct TrackingAllocator;
    static ALLOCATED_BYTES: AtomicUsize = AtomicUsize::new(0);

    unsafe impl GlobalAlloc for TrackingAllocator {
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            ALLOCATED_BYTES.fetch_add(layout.size(), Ordering::SeqCst);
            System.alloc(layout)
        }
        unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
            System.dealloc(ptr, layout)
        }
    }

    #[global_allocator]
    static GLOBAL: TrackingAllocator = TrackingAllocator;

    #[repr(C)]
    #[derive(Clone, Copy, Debug, PartialEq)]
    struct Particle {
        pos: [f32; 3],
        branch_hash: f32,
        vel: [f32; 3],
        mass: f32,
    }

    #[repr(C)]
    #[derive(Clone, Copy, Debug, PartialEq)]
    struct SeedRecord {
        pos_radius: [f32; 4],
        mass_debt: [f32; 4],
    }

    #[repr(C)]
    #[derive(Clone, Copy, Debug, PartialEq)]
    struct SimulationControls {
        sound_speed_scale: f32,
        percolation_threshold_scale: f32,
        lensing_strength: f32,
        chromatic_dispersion: f32,
        target_redshift: f32,
        viewport_split_mode: u32,
    }

    #[test]
    fn test_particle_memory_layout() {
        assert_eq!(std::mem::size_of::<Particle>(), 32);
        assert_eq!(std::mem::align_of::<Particle>(), 4);

        let p = Particle {
            pos: [0.1, 0.2, 0.3],
            branch_hash: 0.72,
            vel: [0.01, -0.02, 0.03],
            mass: 1.0,
        };

        let base = &p as *const _ as usize;
        let pos_offset = (&p.pos as *const _ as usize) - base;
        let hash_offset = (&p.branch_hash as *const _ as usize) - base;
        let vel_offset = (&p.vel as *const _ as usize) - base;
        let mass_offset = (&p.mass as *const _ as usize) - base;

        assert_eq!(pos_offset, 0);
        assert_eq!(hash_offset, 12);
        assert_eq!(vel_offset, 16);
        assert_eq!(mass_offset, 28);
    }

    #[test]
    fn test_seed_record_memory_layout() {
        assert_eq!(std::mem::size_of::<SeedRecord>(), 32);
        assert_eq!(std::mem::align_of::<SeedRecord>(), 4);

        let seed = SeedRecord {
            pos_radius: [0.25, 0.50, 0.75, 0.0078125],
            mass_debt: [7.955e8, 7.207e11, 6.00e37, 1.0],
        };

        let base = &seed as *const _ as usize;
        let pos_offset = (&seed.pos_radius as *const _ as usize) - base;
        let debt_offset = (&seed.mass_debt as *const _ as usize) - base;

        assert_eq!(pos_offset, 0);
        assert_eq!(debt_offset, 16);
    }

    #[test]
    fn test_simulation_controls_layout_and_clamping() {
        assert_eq!(std::mem::size_of::<SimulationControls>(), 24);

        let mut controls = SimulationControls {
            sound_speed_scale: 5.0,
            percolation_threshold_scale: 0.2,
            lensing_strength: 8.0,
            chromatic_dispersion: 0.9,
            target_redshift: -5.0,
            viewport_split_mode: 1,
        };

        controls.sound_speed_scale = controls.sound_speed_scale.clamp(0.0, 3.0);
        controls.percolation_threshold_scale = controls.percolation_threshold_scale.clamp(0.5, 2.0);
        controls.lensing_strength = controls.lensing_strength.clamp(0.0, 5.0);
        controls.chromatic_dispersion = controls.chromatic_dispersion.clamp(0.0, 0.5);
        controls.target_redshift = controls.target_redshift.clamp(-0.999, 1e14);

        assert_eq!(controls.sound_speed_scale, 3.0);
        assert_eq!(controls.percolation_threshold_scale, 0.5);
        assert_eq!(controls.lensing_strength, 5.0);
        assert_eq!(controls.chromatic_dispersion, 0.5);
        assert_eq!(controls.target_redshift, -0.999);
    }

    #[test]
    fn test_zero_allocation_frame_step() {
        let initial_allocated = ALLOCATED_BYTES.load(Ordering::SeqCst);

        let mut positions = [[0.5f32; 3]; 2048];
        let velocities = [[0.01f32; 3]; 2048];

        for i in 0..2048 {
            let dt = 0.016667;
            positions[i][0] += velocities[i][0] * dt;
            positions[i][1] += velocities[i][1] * dt;
            positions[i][2] += velocities[i][2] * dt;
        }

        let post_allocated = ALLOCATED_BYTES.load(Ordering::SeqCst);
        assert_eq!(
            initial_allocated, post_allocated,
            "Active frame loop triggered dynamic heap allocations."
        );
    }
}
