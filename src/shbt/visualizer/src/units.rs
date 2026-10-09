//! Dimensional metrology engine (shbt8 Section 4 / Thm 9.11-9.12).
//!
//! Zero-overhead conversions between SI/cosmological units and the
//! dimensionless code-unit frame used by the supercomoving KDK
//! integrator, plus byte-exact std430 uniform layouts for the WGSL
//! pipelines.
//!
//! Code-unit scaling matrix (L_box = BOX_SIZE Mpc/h):
//!   [L] = L_box,  [T] = H_0^-1,  [M] = M_box = rho_crit,0 * Omega_m0 * L_box^3
//!   V_0 = H_0 * L_box (velocity scale),  G_code = (3/2) * Omega_m0
//!
//! The background expansion rate mirrors Tier 1 (`telemetry::hubble`,
//! `boltzmann_shbt._hubble`): the SHBT shifted intercept
//! H(z) = (H_0,CMB + A_H/(1+z)) * E(z) with
//! E(z) = sqrt(Omega_m (1+z)^3 + Omega_r (1+z)^4 + 1 - Omega_m - Omega_r).
//! Angular-diameter distances follow Etherington's distance duality
//! D_A(z) = chi(z)/(1+z) evaluated over the same H_SHBT kernel.

use bytemuck::{Pod, Zeroable};

use crate::telemetry;

pub const C_SI: f64 = 2.997_924_58e8; // Speed of light (m/s)
pub const G_SI: f64 = 6.674_30e-11; // Gravitational constant (m^3/kg/s^2)
pub const MPC_TO_METER: f64 = 3.085_677_581_49e22; // Meters per Megaparsec
pub const MSUN_TO_KG: f64 = 1.988_47e30; // Kilograms per Solar Mass
pub const C_KM_S: f64 = 299_792.458; // Speed of light (km/s)
pub const OMEGA_B0: f64 = 0.049; // Baryon density fraction (Planck-like)
pub const Z_DEC: f64 = 1089.0; // Photon decoupling redshift
pub const T_DEC_K: f64 = 2970.0; // Decoupling gas temperature (K)
pub const K_B_SI: f64 = 1.380_649e-23; // Boltzmann constant (J/K)
pub const M_H_KG: f64 = 1.673_532_84e-27; // Proton mass (kg)
pub const MU_NEUTRAL: f64 = 1.22; // Mean molecular weight, neutral gas

/// Cosmological context converting physical scales to dimensionless
/// code units. The canonical Tier-1 parameters are
/// (h, Omega_m0, Omega_r0) = (0.674, 0.315, 9.2e-5) with the SHBT
/// intercept A_H = 4.797960072861; Omega_de0 closes the flat model.
#[derive(Debug, Clone, Copy)]
pub struct CosmologicalContext {
    #[allow(dead_code)] // published metrology context (spec API surface)
    pub h: f64,
    #[allow(dead_code)] // published metrology context (spec API surface)
    pub h0_si: f64,          // H_0 in s^-1
    pub omega_m0: f64,
    #[allow(dead_code)] // published metrology context (spec API surface)
    pub omega_de0: f64,
    #[allow(dead_code)] // published metrology context (spec API surface)
    pub l_box_mpc_h: f64,    // Box size in Mpc/h
    #[allow(dead_code)] // published metrology context (spec API surface)
    pub l_box_meters: f64,   // Box size in meters
    pub v0_si: f64,          // Velocity scale V_0 = H_0 * L_box (m/s)
    pub m_box_kg: f64,       // Box mass scale M_box (kg)
    #[allow(dead_code)] // published for metrology consumers (H0^-1 lookback budgets)
    pub rho_crit_0: f64,     // Critical density today (kg/m^3)
}

impl CosmologicalContext {
    pub fn new(h: f64, omega_m0: f64, omega_de0: f64, l_box_mpc_h: f64) -> Self {
        let h0_si = (100.0 * h * 1_000.0) / MPC_TO_METER;
        let l_box_meters = (l_box_mpc_h / h) * MPC_TO_METER;
        let v0_si = h0_si * l_box_meters;
        let rho_crit_0 = (3.0 * h0_si * h0_si) / (8.0 * std::f64::consts::PI * G_SI);
        let m_box_kg = rho_crit_0 * omega_m0 * l_box_meters.powi(3);

        Self {
            h,
            h0_si,
            omega_m0,
            omega_de0,
            l_box_mpc_h,
            l_box_meters,
            v0_si,
            m_box_kg,
            rho_crit_0,
        }
    }

    /// Canonical context matching the Tier-1 background
    /// (`telemetry::{H0_CMB, OMEGA_M, OMEGA_R0}`) and the engine's
    /// BOX_SIZE comoving patch interpreted as Mpc/h.
    pub fn canonical(l_box_mpc_h: f64) -> Self {
        Self::new(
            telemetry::H0_CMB / 100.0,
            telemetry::OMEGA_M,
            1.0 - telemetry::OMEGA_M - telemetry::OMEGA_R0,
            l_box_mpc_h,
        )
    }

    /// Dimensionless expansion rate E(z) (Tier-1 kernel, flat).
    #[inline]
    pub fn e_of_z(z: f64) -> f64 {
        let opz = 1.0 + z;
        (telemetry::OMEGA_M * opz.powi(3)
            + telemetry::OMEGA_R0 * opz.powi(4)
            + (1.0 - telemetry::OMEGA_M - telemetry::OMEGA_R0))
            .sqrt()
    }

    /// SHBT Hubble rate H(z) / H_0 including the shifted intercept
    /// A_H/(1+z) (telemetry::hubble normalized to H0_CMB).
    #[inline]
    pub fn h_of_z(z: f64) -> f64 {
        telemetry::hubble(z) / telemetry::H0_CMB
    }

    /// Comoving line-of-sight distance chi(z) = c int_0^z dz'/H_SHBT(z')
    /// in Mpc (Simpson quadrature over the Tier-1 kernel; the same
    /// background law `boltzmann_shbt._hubble` encodes for telemetry).
    pub fn comoving_distance_mpc(z: f64) -> f64 {
        if z <= 0.0 {
            return 0.0;
        }
        let n = 1024usize;
        let dz = z / n as f64;
        let c_h0 = C_KM_S / telemetry::H0_CMB; // Mpc
        let mut acc = 0.0;
        for i in 0..=n {
            let zi = dz * i as f64;
            let h_ratio = (1.0 + telemetry::A_H / telemetry::H0_CMB / (1.0 + zi))
                * Self::e_of_z(zi);
            let w = match i {
                0 => 1.0,
                _ if i == n => 1.0,
                _ if i % 2 == 1 => 4.0,
                _ => 2.0,
            };
            acc += w / h_ratio;
        }
        c_h0 * dz / 3.0 * acc
    }

    /// Angular diameter distance D_A(z) = chi(z)/(1+z) in Mpc
    /// (Etherington distance duality, flat geometry).
    pub fn angular_diameter_distance_mpc(z: f64) -> f64 {
        Self::comoving_distance_mpc(z) / (1.0 + z)
    }

    /// Baryon sound speed c_s(z) in m/s (shbt9 P-PM hydrodynamics).
    /// Pre-decoupling (z > z_dec): photon-baryon tight coupling,
    ///   c_s = c / sqrt(3 (1 + 3 rho_b / (4 rho_gamma)))
    /// with rho_b/rho_gamma = (Omega_b0/Omega_r0) / (1+z).
    /// Post-decoupling: monatomic adiabatic cooling under the SHBT
    /// loaded conformal factor (1 + 1/(52 (1+z)))^-1:
    ///   c_s^2 = (5 k_B T_dec / (3 mu m_H)) ((1+z)/(1+z_dec))^2 (1 + 1/(52(1+z)))^-1
    pub fn sound_speed_ms(z: f64) -> f64 {
        if z > Z_DEC {
            let opz = 1.0 + z;
            let baryon_photon = (3.0 * OMEGA_B0 / (4.0 * telemetry::OMEGA_R0)) / opz;
            C_SI / (3.0 * (1.0 + baryon_photon)).sqrt()
        } else {
            let opz = 1.0 + z.max(-0.999);
            let thermal = 5.0 * K_B_SI * T_DEC_K / (3.0 * MU_NEUTRAL * M_H_KG);
            let cooling = (opz / (1.0 + Z_DEC)).powi(2);
            let conformal = 1.0 / (1.0 + 1.0 / (52.0 * opz));
            (thermal * cooling * conformal).sqrt()
        }
    }

    /// Dimensionless squared baryon sound speed for the supercomoving
    /// force kick: (c_s / V_0)^2, scaled by the sandbox
    /// sound_speed_scale control. The shader applies
    ///   F_hydro = -cs_sq * a * grad ln rho_b   (visible particles only)
    /// matching the A(a) conformal weighting of the PM force.
    pub fn cs_sq_code(&self, z: f64, scale: f32) -> f32 {
        let ratio = Self::sound_speed_ms(z) / self.v0_si;
        (ratio * ratio * scale as f64) as f32
    }

    /// Exact baryon temperature history (Thm 9.15): evaluated inside the
    /// WGSL kernel; the Rust mirror is kept for metrology reference.
    ///   T_b(z) = T_CMB,0 * (1+z) * z / (z + z_dec)
    /// with T_CMB,0 = 2.7255 K and z_dec = 137. Tightly coupled for
    /// z >> z_dec (T_b ~ T_CMB), adiabatic (1+z)^2 cooling for z << z_dec.
    #[allow(dead_code)]
    pub fn t_baryon_k(z: f64) -> f64 {
        const T_CMB_0: f64 = 2.7255;
        const Z_DECOUPLING: f64 = 137.0;
        let zz = z.max(1.0e-4);
        T_CMB_0 * (1.0 + z.max(-0.999)) * (zz / (zz + Z_DECOUPLING))
    }

    /// Supercomoving pressure normalizer k_B / (mu m_p) in code units
    /// (divided by V_0^2), folded with the sandbox sound_speed_scale
    /// control. The compute_thermodynamic_pressure kernel evaluates
    ///   p_gas = sound_speed_norm * T_b(z) * rho_b
    /// so the pressure force -(a/rho_b) grad P_b telescopes with the CIC
    /// deposition weights (exact self-force cancellation, Thm 9.15).
    pub fn pressure_norm_code(&self, scale: f32) -> f32 {
        let coeff = K_B_SI / (MU_NEUTRAL * M_H_KG);
        (coeff / (self.v0_si * self.v0_si) * scale as f64) as f32
    }

    /// Topological percolation correlation radius (shbt9 Eq. R_filter):
    ///   R_filter(z) = c / (k_l * a * H(z)) = d_H(z) / k_l
    /// in comoving Mpc (the comoving horizon over the leptonic level
    /// k_l = 26, since h^v/K = 12/312 = 1/k_l).
    pub fn r_filter_mpc(z: f64) -> f64 {
        const K_L: f64 = 26.0;
        let a = (1.0 / (1.0 + z)).max(1.0e-6);
        let h = telemetry::hubble(z); // km/s/Mpc
        C_KM_S / (K_L * a * h.max(1.0e-6))
    }

    /// Lens-to-source angular diameter distance
    /// D_ds = (chi(z_s) - chi(z_d)) / (1 + z_s) in Mpc.
    pub fn lens_source_distance_mpc(z_d: f64, z_s: f64) -> f64 {
        let dchi = (Self::comoving_distance_mpc(z_s) - Self::comoving_distance_mpc(z_d))
            .max(0.0);
        dchi / (1.0 + z_s)
    }

    #[inline(always)]
    pub fn physical_to_code_mass(&self, m_solar: f64) -> f32 {
        let m_kg = m_solar * MSUN_TO_KG;
        (m_kg / self.m_box_kg) as f32
    }

        #[allow(dead_code)] // reference converter; the WGSL kernel inlines v = p_tilde * V_0 / a
    #[inline(always)]
    pub fn code_to_physical_velocity(&self, p_tilde: [f32; 3], a: f32) -> [f32; 3] {
        let factor = (self.v0_si as f32) / a;
        [p_tilde[0] * factor, p_tilde[1] * factor, p_tilde[2] * factor]
    }

    /// Physical Einstein radius of a point seed in radians (distance
    /// duality thin-screen metrology, Thm 9.12):
    ///   theta_E = sqrt(4 G M / c^2 * D_ds / (D_d D_s)).
    pub fn compute_einstein_radius_rad(
        &self,
        m_solar: f64,
        d_d_mpc: f64,
        d_s_mpc: f64,
        d_ds_mpc: f64,
    ) -> f32 {
        if m_solar <= 0.0 || d_d_mpc <= 0.0 || d_s_mpc <= 0.0 || d_ds_mpc <= 0.0 {
            return 0.0;
        }
        let m_kg = m_solar * MSUN_TO_KG;
        let d_d_m = d_d_mpc * MPC_TO_METER;
        let d_s_m = d_s_mpc * MPC_TO_METER;
        let d_ds_m = d_ds_mpc * MPC_TO_METER;

        let numerator = 4.0 * G_SI * m_kg * d_ds_m;
        let denominator = (C_SI * C_SI) * (d_d_m * d_s_m);
        if denominator <= 0.0 {
            return 0.0;
        }
        let val = (numerator / denominator).sqrt();
        if val.is_nan() || val.is_infinite() {
            0.0
        } else {
            val as f32
        }
    }
}

/// GET entropic transport coupling (shbt7 Thm 9.8, single Rust source
/// mirroring `compute_kappa_get` in causal_point_get.wgsl):
///   kappa_GET(f_load) = (1/(d_1 - h_dual_su3)) * (1 + (c_eff/d_1) f_load).
#[inline(always)]
pub fn kappa_get(f_load: f32) -> f32 {
    const D_EFF_0: f32 = 26.0 - 3.0; // d_1 - h^v_SU(3) = 23
    const C_EFF_OVER_D1: f32 = (1325.0 / 154.0) / 26.0;
    (1.0 / D_EFF_0) * (1.0 + C_EFF_OVER_D1 * f_load)
}

/// WebGPU StepUniforms layout matching the holographic softening contract.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct StepUniforms {
    pub scale_factor: f32,
    pub dt: f32,
    pub epsilon_holo: f32,
    pub epsilon_holo_sq: f32,
    pub box_size: f32,
    pub lambda_holo: f32,
    pub n_sat: f32,
    pub grid_dim: u32,
}

/// Simulation uniforms layout matching std430 alignment in WGSL
/// (SimulationUniforms in nbody_pm.wgsl).
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct GpuSimulationUniforms {
    pub a: f32,
    pub a_next: f32,
    pub dtau: f32,
    pub half_dtau: f32,
    pub h_h0: f32,
    pub omega_m0: f32,
    pub f_load: f32,
    pub eta_soft: f32,
    pub grid_size: f32,
    pub eps_soft_sq: f32,
    pub kappa_get: f32,
    pub num_seeds: u32,
    pub num_particles: u32,
    pub g_code: f32,
    pub a_coupling: f32,
    // Engine extras beyond the shbt8 contract: box size (comoving Mpc/h),
    // the legacy code-time increment for the de-rendering accruals, and
    // the inverse box mass in M_sun for seed-mass code conversion.
    pub box_size: f32,
    pub dt_legacy: f32,
    pub inv_m_box_msun: f32,
    /// Wall-clock step in seconds (tether alpha decay clock); set by the
    /// engine after prepare_step_uniforms, not part of the metrology contract.
    pub wall_dt: f32,
    /// Baryon sound speed squared in code units, folded with the sandbox
    /// sound_speed_scale control (P-PM hydrodynamics, shbt9 Phase 1).
    pub cs_sq_scaled: f32,
    /// Primordial CMB normalization T_CMB,0 = 2.7255 K (Thm 9.15 thermal
    /// history evaluated in-kernel).
    pub t_cmb_0: f32,
    /// Thermal decoupling redshift z_dec = 137.0.
    pub z_dec: f32,
    /// Dimensionless pressure normalizer k_B/(mu m_p)/V_0^2 folded with
    /// the sandbox sound_speed_scale control.
    pub pressure_norm: f32,
    /// Mean fixed-point CIC deposit per grid cell (mean_cell =
    /// sum(mass) * FIXED_POINT_SCALE / N_cells) for the normalized
    /// incubation density ratio written into particle.pad.x.
    pub mean_cell_fixed: f32,
}

/// Sandbox observer uniform tail piggy-backed on the emergence-side
/// metadata: num_sandbox observers is carried in the W-group metadata
/// buffer, not the 80-byte contract above.

/// Lens post-process uniform layout matching std430 alignment in WGSL
/// (thin-screen block appended to LensingUniforms).
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct GpuLensUniforms {
    pub view_proj: [f32; 16],
    pub cam_pos: [f32; 4],
    pub viewport_dim: [f32; 2],
    pub theta_fov: f32,
    pub d_d: f32,
    pub d_s: f32,
    pub d_ds: f32,
    pub zeta_disp: f32,
    pub theta_core: f32,
    pub num_seeds: u32,
    pub _pad0: u32,
    pub _pad1: [f32; 2],
}

/// Seed lens entry matching the SeedLensBuffer std430 record in WGSL.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct GpuSeedLensBuffer {
    pub screen_pos: [f32; 4],      // xy = UV, z = depth, w = mass (M_sun)
    pub einstein_radius: [f32; 4], // x = theta_E (UV), y = theta_E (rad)
}

/// Metrology pipeline: turns per-frame cosmological state into
/// byte-aligned simulation uniforms for the KDK step.
pub struct MetrologyPipeline {
    pub ctx: CosmologicalContext,
}

impl MetrologyPipeline {
    pub fn new(ctx: CosmologicalContext) -> Self {
        Self { ctx }
    }

    /// Evaluates epsilon_holo(a) = (3*pi / (2 * Lambda_holo * N_sat))^(1/4) * a
    #[inline]
    pub fn calculate_epsilon_holo(&self, scale_factor: f64) -> f64 {
        const DEFAULT_LAMBDA_HOLO: f64 = 1.08913883e-52; // m^-2
        const DEFAULT_N_SAT: f64 = 3.312593327986e122;
        let base = (3.0 * std::f64::consts::PI) / (2.0 * DEFAULT_LAMBDA_HOLO * DEFAULT_N_SAT);
        base.powf(0.25) * scale_factor
    }

    /// Assemble the KDK uniform block. `a`/`a_next` bracket the current
    /// scale-factor step; `dtau` is the supercomoving increment
    /// Delta_tau = Delta a / (a_{1/2}^3 * H(a_{1/2})) evaluated at the
    /// mid-point, and A(a) = (3/2) Omega_m0 a (1 - (10/33) f_load)
    /// is the conformal Hamiltonian coupling.
    pub fn prepare_step_uniforms(
        &self,
        a: f32,
        da: f32,
        h_h0: f32,
        f_load: f32,
        grid_size: f32,
        eta_soft: f32,
        num_particles: u32,
        num_seeds: u32,
        kappa_get: f32,
        box_size: f32,
        dt_legacy: f32,
    ) -> GpuSimulationUniforms {
        let a_next = a + da;
        let a_half = a + 0.5 * da;

        let dtau = da / (a_half.powi(3) * h_h0);
        let half_dtau = 0.5 * dtau;
        let eps_soft = eta_soft / grid_size;
        let eps_soft_sq = eps_soft * eps_soft;
        let g_code = 1.5 * (self.ctx.omega_m0 as f32);

        // A(a) = (3/2) * Omega_m0 * a * (1 - (10/33)*f_load)
        let load_drag = 1.0 - (10.0 / 33.0) * f_load;
        let a_coupling = g_code * a * load_drag;
        // m_tilde for one solar mass — the shader multiplies per seed.
        let inv_m_box_msun = self.ctx.physical_to_code_mass(1.0);

        GpuSimulationUniforms {
            a,
            a_next,
            dtau,
            half_dtau,
            h_h0,
            omega_m0: self.ctx.omega_m0 as f32,
            f_load,
            eta_soft,
            grid_size,
            eps_soft_sq,
            kappa_get,
            num_seeds,
            num_particles,
            g_code,
            a_coupling,
            box_size,
            dt_legacy,
            inv_m_box_msun,
            wall_dt: 0.0,
            cs_sq_scaled: 0.0,
            t_cmb_0: 2.7255,
            z_dec: 137.0,
            pressure_norm: 0.0,
            mean_cell_fixed: 0.0,
        }
    }
}
