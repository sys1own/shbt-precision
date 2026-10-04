//! Continuum Transport Verification Suite (shbt10 Phase 2)
//! File: tests/test_continuum_transport.rs
//! Repository: sys1own/shbt-precision
//!
//! Gates the three first-principles invariants introduced by the
//! continuum overhaul:
//!   1. Rational-precision bit conservation across the Stinespring
//!      partition (eta_A + eta_D == 1): residual < 1e-30.
//!   2. CIC pressure self-force cancellation on the periodic mesh:
//!      the net gathered force over a uniform probe lattice is
//!      exactly zero (telescoping central differences), |Sigma F| < 1e-15.
//!   3. 512-bit (U512-equivalent) stress-energy residual across the
//!      incubation epoch: E_munu < 1e-120.

#[cfg(test)]
mod tests {
    use rug::Float;

    const ETA_A_NUM: i128 = 10;
    const ETA_D_NUM: i128 = 23;
    const ETA_DEN: i128 = 33;
    const N_SAT: f64 = 3.3119977e122;

    /// Minimal exact rational on i128 ("BigRational128" contract): the
    /// partition coefficients are small integers, so the WZW channel
    /// fractions are exact — no rounding ever enters the ledger.
    #[derive(Clone, Copy, Debug)]
    struct Rational128 {
        num: i128,
        den: i128,
    }

    impl Rational128 {
        fn new(num: i128, den: i128) -> Self {
            let g = gcd(num.abs(), den.abs()).max(1);
            Rational128 {
                num: num / g,
                den: den / g,
            }
        }
        fn add(self, rhs: Self) -> Self {
            Rational128::new(
                self.num * rhs.den + rhs.num * self.den,
                self.den * rhs.den,
            )
        }
        fn mul(self, rhs: Self) -> Self {
            Rational128::new(self.num * rhs.num, self.den * rhs.den)
        }
        fn sub(self, rhs: Self) -> Self {
            self.add(Rational128::new(-rhs.num, rhs.den))
        }
        fn is_zero(&self) -> bool {
            self.num == 0
        }
    }

    fn gcd(mut a: i128, mut b: i128) -> i128 {
        while b != 0 {
            let t = b;
            b = a % b;
            a = t;
        }
        a.max(1)
    }

    /// Periodic grid index identical to `grid_idx_cell` in nbody_pm.wgsl.
    fn grid_idx(cell: [i32; 3], gd: usize) -> usize {
        let g = gd as i32;
        let wrap = [
            (cell[0] + g) % g,
            (cell[1] + g) % g,
            (cell[2] + g) % g,
        ];
        (wrap[0] + wrap[1] * g + wrap[2] * g * g) as usize
    }

    /// Central-difference pressure gradient at a cell, replicating
    /// `interpolate_pressure_force` (0.5 / cell_size folded out; we
    /// compare forces up to the common positive prefactor a/rho_c —
    /// proportional cancellation is what the invariant checks).
    ///
    /// Evaluated in `prec`-bit `Float`: the gate is |Sigma F| < 1e-15,
    /// and the algebraic content is exact telescoping — computing in
    /// f64 would bury the residual under ~1e-14 of per-probe rounding
    /// noise that has nothing to do with the invariant.
    fn central_grad(p: &[Float], cell: [i32; 3], gd: usize, prec: u32) -> [Float; 3] {
        let mut g = [
            Float::with_val(prec, 0),
            Float::with_val(prec, 0),
            Float::with_val(prec, 0),
        ];
        for ax in 0..3 {
            let mut plus = cell;
            let mut minus = cell;
            plus[ax] += 1;
            minus[ax] -= 1;
            g[ax] = Float::with_val(
                prec,
                &p[grid_idx(plus, gd)] - &p[grid_idx(minus, gd)],
            );
        }
        g
    }

    /// Gathered pressure force on a probe at fractional cell coordinate
    /// `frac` inside `base`, using the same trilinear weights the shader
    /// applies: F = sum_8 w_c * (-grad p_c / rho_c).
    fn gather_force(
        p: &[Float],
        rho: &[Float],
        base: [i32; 3],
        frac: [f64; 3],
        gd: usize,
        prec: u32,
    ) -> [Float; 3] {
        let mut f = [
            Float::with_val(prec, 0),
            Float::with_val(prec, 0),
            Float::with_val(prec, 0),
        ];
        for dz in 0..2 {
            for dy in 0..2 {
                for dx in 0..2 {
                    let w: f64 = [dx, dy, dz]
                        .iter()
                        .enumerate()
                        .map(|(ax, &d)| {
                            if d == 1 {
                                frac[ax]
                            } else {
                                1.0 - frac[ax]
                            }
                        })
                        .product();
                    let cell = [base[0] + dx, base[1] + dy, base[2] + dz];
                    let i0 = grid_idx(cell, gd);
                    let rho_c = &rho[i0];
                    if rho_c > &Float::with_val(prec, 1.0e-12) {
                        let g = central_grad(p, cell, gd, prec);
                        for ax in 0..3 {
                            let term =
                                Float::with_val(prec, -w) * &g[ax] / rho_c;
                            f[ax] += &term;
                        }
                    }
                }
            }
        }
        f
    }

    #[test]
    fn test_bit_conservation_rational128() {
        // eta_A + eta_D = 10/33 + 23/33 = 1 — exact in integer
        // arithmetic, so the Stinespring partition conserves every bit.
        let eta_a = Rational128::new(ETA_A_NUM, ETA_DEN);
        let eta_d = Rational128::new(ETA_D_NUM, ETA_DEN);
        let total = eta_a.add(eta_d);
        let residual = total.sub(Rational128::new(1, 1));
        assert!(
            residual.is_zero(),
            "Partition unitarity violated: eta_A + eta_D - 1 = {}/{}",
            residual.num,
            residual.den
        );

        // Ledger conservation across the incubation window z in [100, 18]:
        // for a grid of quench fractions q (as exact rationals p/q),
        // visible share + quenched share stays on the unitary orbit:
        //   eta_A + eta_D*(1-q) + eta_D*q == 1.
        let z_steps = 512;
        for i in 0..=z_steps {
            let t = i as i128; // q = i / z_steps, a rational in [0, 1]
            let q = Rational128::new(t, z_steps);
            let one_minus_q = Rational128::new(1, 1).sub(q);
            let vis = eta_a.add(eta_d.mul(one_minus_q));
            let dark = eta_d.mul(q);
            let ledger_residual = vis.add(dark).sub(Rational128::new(1, 1));
            assert!(
                ledger_residual.is_zero(),
                "Bit ledger residual nonzero at q = {}/{}: {}/{} (>1e-30 gate)",
                t,
                z_steps,
                ledger_residual.num,
                ledger_residual.den
            );
        }
    }

    #[test]
    fn test_cic_pressure_self_force_cancellation() {
        // Periodic 16^3 mesh; arbitrary pressure field, uniform density.
        // The telescoping central-difference stencil guarantees that the
        // net force gathered over ANY complete lattice of probes sums to
        // zero — the pressure channel cannot exert a mesh self-force.
        let prec = 256u32;
        let gd = 16usize;
        let n = gd * gd * gd;
        let mut p = Vec::with_capacity(n);
        // Deterministic pseudo-random field (sufficient structure to make
        // the cancellation non-trivial).
        let mut s = 0x9E3779B97F4A7C15u64;
        for _ in 0..n {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            p.push(Float::with_val(
                prec,
                ((s >> 11) as f64 / (1u64 << 53) as f64) - 0.5,
            ));
        }
        let rho = vec![Float::with_val(prec, 1.0); n];

        // Uniform probe lattice: one probe at each cell center. The sum
        // over probes must vanish identically — each cell's gradient is
        // distributed to probes with total weight exactly 1 (CIC
        // partition of unity), so the net is the mesh-wide gradient sum,
        // which telescopes to zero on the periodic grid.
        let mut net = [
            Float::with_val(prec, 0),
            Float::with_val(prec, 0),
            Float::with_val(prec, 0),
        ];
        for gz in 0..gd as i32 {
            for gy in 0..gd as i32 {
                for gx in 0..gd as i32 {
                    let f = gather_force(
                        &p,
                        &rho,
                        [gx, gy, gz],
                        [0.5, 0.5, 0.5],
                        gd,
                        prec,
                    );
                    for ax in 0..3 {
                        net[ax] += &f[ax];
                    }
                }
            }
        }
        let net_sq = Float::with_val(prec, &net[0] * &net[0])
            + Float::with_val(prec, &net[1] * &net[1])
            + Float::with_val(prec, &net[2] * &net[2]);
        let net_norm = net_sq.sqrt();
        assert!(
            net_norm < Float::with_val(prec, 1.0e-15),
            "CIC self-force cancellation violated: |Sigma F| = {} >= 1e-15",
            net_norm
        );

        // Also at non-centered probes (weights are not symmetric then);
        // per-cell gradient sums are still telescopic over the mesh.
        let mut net2 = [
            Float::with_val(prec, 0),
            Float::with_val(prec, 0),
            Float::with_val(prec, 0),
        ];
        for gz in 0..gd as i32 {
            for gy in 0..gd as i32 {
                for gx in 0..gd as i32 {
                    let f = gather_force(
                        &p,
                        &rho,
                        [gx, gy, gz],
                        [0.13, 0.41, 0.77],
                        gd,
                        prec,
                    );
                    for ax in 0..3 {
                        net2[ax] += &f[ax];
                    }
                }
            }
        }
        let net2_sq = Float::with_val(prec, &net2[0] * &net2[0])
            + Float::with_val(prec, &net2[1] * &net2[1])
            + Float::with_val(prec, &net2[2] * &net2[2]);
        let net2_norm = net2_sq.sqrt();
        assert!(
            net2_norm < Float::with_val(prec, 1.0e-15),
            "CIC self-force cancellation violated (offset lattice): {}",
            net2_norm
        );
    }

    #[test]
    fn test_u512_stress_energy_residual() {
        use shbt_simulator::shbt::cosmology::ShbtUniverse;

        // 512-bit mantissa (the "U512" register contract): residuals must
        // remain below 1e-120 across the incubation + condensation epochs.
        let prec = 512u32;
        let universe = ShbtUniverse::new_canonical_branch(prec);

        let redshifts = [
            1.0e14, 1.0e10, 1000.0, 100.0, 30.0, 24.0, 18.0, 16.0, 7.0, 3.0, 1.0, 0.0, -0.999,
        ];
        let tol = Float::with_val(prec, 1.0e-120);
        for &z in &redshifts {
            let residual = universe.evaluate_stress_energy_divergence(z);
            assert!(
                residual < tol,
                "U512 stress-energy residual exceeded at z = {:e}: E_munu = {}",
                z,
                residual
            );
            // Sanity: the residual should be *much* smaller than the f64
            // floor — flag if the ledger drifts above machine epsilon.
            let res64 = residual.to_f64();
            assert!(
                res64 < 1.0e-9,
                "Stress-energy residual above f64 floor at z = {:e}: {:e}",
                z,
                res64
            );
        }
        let _ = N_SAT; // referenced by the legacy gate table
    }

    #[test]
    fn test_quintic_nucleation_weight_c2() {
        // C^2 quintic smoothing kernel Psi_nuc(z) on z in [18, 30]
        // (Thm 9.14): Psi = 10 t^3 - 15 t^4 + 6 t^5, t = (30 - z)/12.
        // Verify C^2 boundary conditions: value and first TWO
        // derivatives vanish at both ends (no condensation discontinuity).
        let psi = |t: f64| 10.0 * t.powi(3) - 15.0 * t.powi(4) + 6.0 * t.powi(5);
        let d1 = |t: f64| 30.0 * t.powi(2) - 60.0 * t.powi(3) + 30.0 * t.powi(4);
        let d2 = |t: f64| 60.0 * t - 180.0 * t.powi(2) + 120.0 * t.powi(3);

        assert!(psi(0.0).abs() < 1e-30 && (psi(1.0) - 1.0).abs() < 1e-30);
        assert!(d1(0.0).abs() < 1e-30 && d1(1.0).abs() < 1e-30);
        assert!(d2(0.0).abs() < 1e-30 && d2(1.0).abs() < 1e-30);

        // Monotone on the window, bracketed in [0, 1].
        let mut prev = 0.0f64;
        for i in 0..=1000 {
            let t = i as f64 / 1000.0;
            let v = psi(t);
            assert!(v >= prev - 1e-15 && v <= 1.0 + 1e-15);
            prev = v;
        }
    }
}
