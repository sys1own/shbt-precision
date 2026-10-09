//! src/shbt/inflation.rs
//!
//! Kinematic boundary indexing implementation replacing potential-driven inflaton dynamics.

use crate::shbt::cosmology::{
    compute_first_principles_as, compute_first_principles_ns,
    solve_first_principles_h0_cmb, CosmologicalInvariants,
    C_DARK_COMP_CANONICAL, C_VIS_CANONICAL, ETA_V_CANONICAL,
    GAMMA_T_CANONICAL, N_SAT_CANONICAL,
};

pub struct KinematicInflationRegister {
    pub invariants: CosmologicalInvariants,
}

impl KinematicInflationRegister {
    pub fn initialize_from_boundary() -> Self {
        let delta_ln_p = 3.0_f64.ln() - 2.0_f64.ln();
        let a_s = compute_first_principles_as(
            C_VIS_CANONICAL,
            C_DARK_COMP_CANONICAL,
            N_SAT_CANONICAL,
        );
        let n_s = compute_first_principles_ns(GAMMA_T_CANONICAL, delta_ln_p);
        let h0_cmb = solve_first_principles_h0_cmb(
            ETA_V_CANONICAL,
            0.02237,
            0.1200,
            1e-8,
            120,
        ).expect("Dynamic boundary capacity solver failed to converge");

        Self {
            invariants: CosmologicalInvariants {
                h0_cmb,
                a_s,
                n_s,
                r: 0.0032,
                n_t: -0.0004,
                tau_nl: 0.000324,
                eta_v: ETA_V_CANONICAL,
            },
        }
    }
}
