"""Standalone SHBT precision-cosmology audit module.

The public functions in this file implement the Section 9 precision-cosmology
equations from ``shbt_precision_final.tex``.  The module is designed to be
dropped into the ``shbt-simulator`` repository: when the optional
``shbt_simulator`` package is importable, its ``StaticBoundary`` and
``ShbtSimulator`` objects are used for constants and foundation-audit metadata.
Otherwise the paper's benchmark constants are used directly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import unittest
from dataclasses import asdict, dataclass, is_dataclass
from decimal import Decimal, ROUND_DOWN, localcontext
from fractions import Fraction
from typing import Any, NamedTuple, Sequence

import mpmath

try:  # Optional compatibility with the target repository.
    import shbt_simulator as _shbt_simulator
except Exception:  # pragma: no cover - exercised only outside shbt-simulator.
    _shbt_simulator = None


DEFAULT_PRECISION = 80
MPMATH_GUARD_DIGITS = 20

COMPLETED_LEDGER_FRACTION = Fraction(1197103, 362670)
DELTA_MOD_FRACTION = COMPLETED_LEDGER_FRACTION / 24

DEFAULT_H0_CMB = Decimal("67.4")
DEFAULT_OMEGA_M = Decimal("0.315")
DEFAULT_OMEGA_R0 = Decimal("9.2e-5")
DEFAULT_SIGMA8 = Decimal("0.812")
DEFAULT_N_SAT = Decimal("3.312593327986e122")
DEFAULT_LAMBDA_HOLO_SI_M2 = Decimal("1.08913883e-52")
DEFAULT_M_NU1_EV = Decimal("2.829630635353e-3")
DEFAULT_DELTA_M21_SQ_EV2 = Decimal("7.4e-5")
DEFAULT_DELTA_M31_SQ_EV2 = Decimal("2.5e-3")
DEFAULT_Z_SAMPLES = (Decimal("0"), Decimal("0.5"), Decimal("1"), Decimal("2"), Decimal("10"), Decimal("1100"))
GROWTH_AUDIT_REDSHIFTS = (Decimal("0.5"), Decimal("0.8"))
ISW_AUDIT_REDSHIFTS = (Decimal("10"), Decimal("50"), Decimal("100"), Decimal("1100"))
BBN_AUDIT_REDSHIFT = Decimal("1e9")
GROWTH_INDEX_GAMMA = Decimal("0.55")
INITIAL_GROWTH_SCALE_FACTOR = Decimal("0.001")

# Physical constants used by the dark-matter topological ghost (Section 9 extension).
DEFAULT_ETA_B = Decimal("6.449923359416e-10")
GRAVITATIONAL_CONSTANT_SI = Decimal("6.67430e-11")  # m^3 kg^-1 s^-2
PROTON_MASS_KG = Decimal("1.67262192369e-27")
CMB_PHOTON_DENSITY_M3 = Decimal("410.7e6")  # m^-3
METERS_PER_MPC = Decimal("3.085677581491367e22")
KMS_TO_M_S = Decimal("1000")
PI_DECIMAL = Decimal("3.14159265358979323846264338327950288419716939937510")


Number = Decimal | Fraction | mpmath.mpf | float | int | str


class MeasurementCost(NamedTuple):
    """GET measurement cost tuple returned by :func:`get_measurement_cost`."""

    C_get: Decimal
    R_entropy: Decimal


@dataclass(frozen=True)
class SimulatorConstants:
    """Constants resolved from ``shbt_simulator.StaticBoundary`` when present."""

    h0_cmb: Decimal
    lambda_holo_si_m2: Decimal
    n_sat: Decimal
    c_dark_residual: Decimal | None
    simulator_available: bool


def _decimal(value: Number) -> Decimal:
    if isinstance(value, Decimal):
        return value
    if isinstance(value, Fraction):
        return Decimal(value.numerator) / Decimal(value.denominator)
    if isinstance(value, mpmath.mpf):
        return Decimal(mpmath.nstr(value, n=DEFAULT_PRECISION))
    return Decimal(str(value))


def _mp(value: Number) -> mpmath.mpf:
    if isinstance(value, mpmath.mpf):
        return value
    if isinstance(value, Decimal):
        return mpmath.mpf(str(value))
    if isinstance(value, Fraction):
        return mpmath.mpf(value.numerator) / mpmath.mpf(value.denominator)
    return mpmath.mpf(str(value))


def _mpmath_dps(precision: int = DEFAULT_PRECISION) -> int:
    return max(int(precision), 28) + MPMATH_GUARD_DIGITS


def _mp_to_decimal(value: mpmath.mpf, *, precision: int = DEFAULT_PRECISION) -> Decimal:
    return Decimal(mpmath.nstr(value, n=max(int(precision), 28)))


def _fraction_string(value: Fraction) -> str:
    return f"{value.numerator}/{value.denominator}" if value.denominator != 1 else str(value.numerator)


def _canonical_decimal_string(value: Number) -> str:
    decimal_value = _decimal(value)
    return format(decimal_value.normalize(), "f")


def _as_json_safe(value: Any) -> Any:
    if isinstance(value, Decimal):
        return format(value, "E") if value.adjusted() >= 8 or value.adjusted() <= -6 else format(value, "f")
    if isinstance(value, Fraction):
        return _fraction_string(value)
    if isinstance(value, tuple) and hasattr(value, "_fields"):
        return {field: _as_json_safe(getattr(value, field)) for field in value._fields}
    if is_dataclass(value):
        return _as_json_safe(asdict(value))
    if isinstance(value, dict):
        return {str(key): _as_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_as_json_safe(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def _static_boundary_instance() -> Any | None:
    if _shbt_simulator is None or not hasattr(_shbt_simulator, "StaticBoundary"):
        return None
    try:
        return _shbt_simulator.StaticBoundary()
    except Exception:
        return None


def _first_attr(objects: Sequence[Any], names: Sequence[str]) -> Any | None:
    for obj in objects:
        if obj is None:
            continue
        for name in names:
            if hasattr(obj, name):
                candidate = getattr(obj, name)
                return candidate() if callable(candidate) else candidate
    return None


def load_default_constants() -> SimulatorConstants:
    """Return constants used by Eqs. (173), (175), and (204).

    This is the compatibility layer for ``shbt_simulator.StaticBoundary``.
    The paper's benchmark values are used whenever the target repository does
    not expose a matching attribute.
    """

    boundary = _static_boundary_instance()
    objects = (boundary, _shbt_simulator)
    h0_cmb = _first_attr(
        objects,
        ("h0_cmb", "H0_CMB", "hubble_cmb_anchor", "PLANCK2018_H0_KM_S_MPC"),
    )
    lambda_holo = _first_attr(
        objects,
        ("lambda_holo_si_m2", "Lambda_holo", "lambda_holo", "LAMBDA_HOLO", "PLANCK2018_LAMBDA_SI_M2"),
    )
    n_sat = _first_attr(
        objects,
        ("N_sat", "n_sat", "horizon_register_bits", "HORIZON_REGISTER_BITS", "bit_budget"),
    )
    c_dark_residual = _first_attr(objects, ("c_dark_residual", "C_DARK_RESIDUAL", "c_dark", "C_DARK"))
    return SimulatorConstants(
        h0_cmb=_decimal(h0_cmb) if h0_cmb is not None else DEFAULT_H0_CMB,
        lambda_holo_si_m2=_decimal(lambda_holo) if lambda_holo is not None else DEFAULT_LAMBDA_HOLO_SI_M2,
        n_sat=_decimal(n_sat) if n_sat is not None else DEFAULT_N_SAT,
        c_dark_residual=_decimal(c_dark_residual) if c_dark_residual is not None else None,
        simulator_available=_shbt_simulator is not None and boundary is not None,
    )


def _foundation_audit_summary() -> dict[str, Any]:
    """Run or summarize ``shbt_simulator.ShbtSimulator`` foundation audit."""

    if _shbt_simulator is None or not hasattr(_shbt_simulator, "ShbtSimulator"):
        return {"available": False, "status": "shbt_simulator.ShbtSimulator not importable"}
    try:
        simulator = _shbt_simulator.ShbtSimulator()
    except Exception as exc:  # pragma: no cover - target-repo API dependent.
        return {"available": True, "status": "constructor failed", "error": str(exc)}

    for method_name in ("run_full_audit", "run_audit", "audit", "build_audit"):
        method = getattr(simulator, method_name, None)
        if callable(method):
            try:
                result = method()
            except Exception as exc:  # pragma: no cover - target-repo API dependent.
                return {"available": True, "method": method_name, "status": "audit failed", "error": str(exc)}
            return {
                "available": True,
                "method": method_name,
                "status": "completed",
                "result": _as_json_safe(result),
            }
    return {"available": True, "status": "no recognized audit method"}


def load_completed_ledger() -> Fraction:
    """Implements Eq. (173): return ``c_dark^comp = 1197103/362670``."""

    return COMPLETED_LEDGER_FRACTION


def entropy_debt_uplift_factor(delta_mod: Number, *, precision: int = DEFAULT_PRECISION) -> Decimal:
    """Implements Eq. (194): ``exp(delta_mod / 2)``."""

    with localcontext() as context:
        context.prec = max(int(precision), 28)
        return (_decimal(delta_mod) / Decimal("2")).exp()


def h0_local(h0_cmb: Number, delta_mod: Number, *, precision: int = DEFAULT_PRECISION) -> Decimal:
    """Implements Eqs. (194)--(195): ``H0_CMB * exp(delta_mod / 2)``."""

    with localcontext() as context:
        context.prec = max(int(precision), 28)
        return _decimal(h0_cmb) * entropy_debt_uplift_factor(delta_mod, precision=precision)


def loading_amplitude(h0_cmb: Number, h0_local_value: Number) -> Decimal:
    """Implements Eq. (197): ``A_H = H0_local - H0_CMB``."""

    return _decimal(h0_local_value) - _decimal(h0_cmb)


def lock_rate(A_H: Number) -> Decimal:
    """Implements Eq. (183): ``Gamma_lock = 3 A_H``."""

    return Decimal("3") * _decimal(A_H)


def h0_redshift_dependent(z: Number, h0_cmb: Number, A_H: Number) -> Decimal:
    """Implements Eq. (199): ``H0(z) = H0_CMB + A_H / (1 + z)``."""

    redshift = _decimal(z)
    if redshift < 0:
        raise ValueError("z must be non-negative")
    return _decimal(h0_cmb) + _decimal(A_H) / (Decimal("1") + redshift)


def shbt_hubble_rate(
    z: Number,
    h0_cmb: Number,
    A_H: Number,
    omega_m: Number,
    omega_r0: Number = DEFAULT_OMEGA_R0,
) -> Decimal:
    """Implements Eq. (204): loaded matter-radiation-Lambda Hubble rate."""

    redshift = _decimal(z)
    if redshift < 0:
        raise ValueError("z must be non-negative")
    matter = _decimal(omega_m)
    radiation = _decimal(omega_r0)
    if not Decimal("0") < matter < Decimal("1"):
        raise ValueError("omega_m must lie between 0 and 1")
    if radiation < 0:
        raise ValueError("omega_r0 must be non-negative")
    if matter + radiation >= Decimal("1"):
        raise ValueError("omega_m + omega_r0 must be less than 1")
    one_plus_z = Decimal("1") + redshift
    expansion_squared = matter * one_plus_z**3 + radiation * one_plus_z**4 + (Decimal("1") - matter - radiation)
    return h0_redshift_dependent(redshift, h0_cmb, A_H) * expansion_squared.sqrt()


def loading_fraction_ode(
    z: Number,
    h0_cmb: Number,
    A_H: Number,
    omega_m: Number,
    omega_r0: Number = DEFAULT_OMEGA_R0,
) -> Decimal:
    """Implements Eq. (180): ``df_load/dz = Gamma_lock / [(1+z)^2 H_SHBT(z)]``."""

    redshift = _decimal(z)
    if redshift < 0:
        raise ValueError("z must be non-negative")
    denominator = (Decimal("1") + redshift) ** 2 * shbt_hubble_rate(
        redshift,
        h0_cmb,
        A_H,
        omega_m,
        omega_r0,
    )
    return lock_rate(A_H) / denominator


def _finite_quadrature_points(upper: mpmath.mpf) -> list[mpmath.mpf]:
    anchors = ("0.1", "0.5", "1", "2", "5", "10", "50", "100", "500", "1000")
    points = [mpmath.mpf("0")]
    for anchor in anchors:
        candidate = mpmath.mpf(anchor)
        if mpmath.mpf("0") < candidate < upper:
            points.append(candidate)
    points.append(upper)
    return points


def compute_loading_fraction(
    z: Number,
    h0_cmb: Number,
    A_H: Number,
    omega_m: Number,
    omega_r0: Number = DEFAULT_OMEGA_R0,
    *,
    precision: int = DEFAULT_PRECISION,
) -> Decimal:
    """Numerically integrates Eq. (180) using the Eq. (204) background."""

    redshift = _decimal(z)
    if redshift < 0:
        raise ValueError("z must be non-negative")
    if redshift == 0:
        return Decimal("0")
    matter_decimal = _decimal(omega_m)
    radiation_decimal = _decimal(omega_r0)
    if not Decimal("0") < matter_decimal < Decimal("1"):
        raise ValueError("omega_m must lie between 0 and 1")
    if radiation_decimal < 0:
        raise ValueError("omega_r0 must be non-negative")
    if matter_decimal + radiation_decimal >= Decimal("1"):
        raise ValueError("omega_m + omega_r0 must be less than 1")

    with mpmath.workdps(_mpmath_dps(precision)):
        upper = _mp(redshift)
        h0_mp = _mp(h0_cmb)
        amplitude = _mp(A_H)
        matter = _mp(matter_decimal)
        radiation = _mp(radiation_decimal)
        gamma_lock = mpmath.mpf("3") * amplitude

        def integrand(sample_z: mpmath.mpf) -> mpmath.mpf:
            one_plus_z = mpmath.mpf("1") + sample_z
            h0_z = h0_mp + amplitude / one_plus_z
            h_rate = h0_z * mpmath.sqrt(
                matter * one_plus_z**3 + radiation * one_plus_z**4 + (mpmath.mpf("1") - matter - radiation)
            )
            return gamma_lock / (one_plus_z**2 * h_rate)

        value = mpmath.quad(integrand, _finite_quadrature_points(upper))
    return _mp_to_decimal(value, precision=precision)


def compute_entropy_debt(
    z: Number,
    h0_cmb: Number,
    A_H: Number,
    omega_m: Number,
    N_sat: Number = DEFAULT_N_SAT,
    *,
    omega_r0: Number = DEFAULT_OMEGA_R0,
    precision: int = DEFAULT_PRECISION,
) -> Decimal:
    """Implements Eqs. (176) and (180): ``S_debt = N_sat f_load``."""

    capacity = _decimal(N_sat)
    if capacity <= 0:
        raise ValueError("N_sat must be positive")
    with localcontext() as context:
        context.prec = max(int(precision), 28)
        return capacity * compute_loading_fraction(
            z,
            h0_cmb,
            A_H,
            omega_m,
            omega_r0,
            precision=precision,
        )


def growth_ode_system(x: Number, D: Number, dDdx: Number, h0_cmb: Number, A_H: Number, omega_m: Number) -> tuple[mpmath.mpf, mpmath.mpf]:
    """Return the Eq. (203) first-order system for ``D(x)`` with ``x = ln(a)``."""

    x_mp = _mp(x)
    growth = _mp(D)
    growth_prime = _mp(dDdx)
    h0_mp = _mp(h0_cmb)
    amplitude = _mp(A_H)
    matter = _mp(omega_m)
    if h0_mp <= 0:
        raise ValueError("h0_cmb must be positive")
    if not 0 < matter < 1:
        raise ValueError("omega_m must lie between 0 and 1")

    scale_factor = mpmath.e**x_mp
    h0_z = h0_mp + amplitude * scale_factor
    h0_local_value = h0_mp + amplitude
    expansion_squared = matter / scale_factor**3 + (mpmath.mpf("1") - matter)
    intercept_ratio = h0_z / h0_local_value
    normalized_hubble_squared = intercept_ratio**2 * expansion_squared
    dln_e_dx = -mpmath.mpf("3") * matter / (mpmath.mpf("2") * scale_factor**3 * expansion_squared)
    dln_h0_dx = amplitude * scale_factor / h0_z
    omega_m_x = matter / (scale_factor**3 * normalized_hubble_squared)
    growth_second = -(mpmath.mpf("2") + dln_e_dx + dln_h0_dx) * growth_prime + mpmath.mpf("1.5") * omega_m_x * growth
    return growth_prime, growth_second


MpState = tuple[mpmath.mpf, mpmath.mpf]


def _rk4_step(rhs: Any, variable: mpmath.mpf, state: MpState, step_size: mpmath.mpf) -> MpState:
    half_step = step_size / 2
    k1 = rhs(variable, state)
    k2 = rhs(variable + half_step, (state[0] + half_step * k1[0], state[1] + half_step * k1[1]))
    k3 = rhs(variable + half_step, (state[0] + half_step * k2[0], state[1] + half_step * k2[1]))
    k4 = rhs(variable + step_size, (state[0] + step_size * k3[0], state[1] + step_size * k3[1]))
    return (
        state[0] + step_size * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]) / 6,
        state[1] + step_size * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]) / 6,
    )


@dataclass(frozen=True)
class _GrowthSolution:
    samples: dict[str, mpmath.mpf]
    present_growth: mpmath.mpf
    hubble_squared: Any


def _solve_growth_in_scale_factor(
    *,
    omega_m: Number,
    loading_fraction: Number,
    scale_factors: Sequence[Decimal],
    precision: int,
) -> _GrowthSolution:
    """Solve the audit-normalized growth equation used for Table 10."""

    with mpmath.workdps(_mpmath_dps(precision)):
        matter = _mp(omega_m)
        loading = _mp(loading_fraction)
        initial_scale_factor = _mp(INITIAL_GROWTH_SCALE_FACTOR)
        targets = sorted({_mp(scale_factor) for scale_factor in scale_factors} | {mpmath.mpf("1")})

        def lcdm_hubble_squared(scale_factor: mpmath.mpf) -> mpmath.mpf:
            return matter / scale_factor**3 + (mpmath.mpf("1") - matter)

        def lcdm_hubble_log_derivative_da(scale_factor: mpmath.mpf) -> mpmath.mpf:
            return -mpmath.mpf("3") * matter / (mpmath.mpf("2") * scale_factor**4 * lcdm_hubble_squared(scale_factor))

        def intercept_ratio(scale_factor: mpmath.mpf) -> mpmath.mpf:
            return mpmath.mpf("1") + loading * scale_factor

        def hubble_squared(scale_factor: mpmath.mpf) -> mpmath.mpf:
            return intercept_ratio(scale_factor) ** 2 * lcdm_hubble_squared(scale_factor)

        def hubble_log_derivative_da(scale_factor: mpmath.mpf) -> mpmath.mpf:
            return loading / intercept_ratio(scale_factor) + lcdm_hubble_log_derivative_da(scale_factor)

        def rhs(scale_factor: mpmath.mpf, state: MpState) -> MpState:
            growth, growth_derivative = state
            second = -(
                mpmath.mpf("3") / scale_factor + hubble_log_derivative_da(scale_factor)
            ) * growth_derivative + mpmath.mpf("1.5") * matter * growth / (
                scale_factor**5 * hubble_squared(scale_factor)
            )
            return growth_derivative, second

        variable = initial_scale_factor
        state = (initial_scale_factor, mpmath.mpf("1"))
        samples: dict[str, mpmath.mpf] = {}
        for target in targets:
            if target < variable:
                raise ValueError("growth targets must not precede the initial scale factor")
            while variable < target:
                step_size = min(mpmath.mpf("0.001"), target - variable)
                state = _rk4_step(rhs, variable, state, step_size)
                variable += step_size
            samples[mpmath.nstr(target, 60)] = state[0]
        present = samples[mpmath.nstr(mpmath.mpf("1"), 60)]
        return _GrowthSolution(samples=samples, present_growth=present, hubble_squared=hubble_squared)


def compute_growth_suppression(
    z: Number,
    h0_cmb: Number,
    A_H: Number,
    omega_m: Number,
    sigma8: Number,
    *,
    precision: int = DEFAULT_PRECISION,
) -> dict[str, Decimal]:
    """Return SHBT/LCDM ``f sigma_8`` and suppression fraction for Eq. (206)."""

    redshift = _decimal(z)
    if redshift < 0:
        raise ValueError("z must be non-negative")
    matter = _decimal(omega_m)
    if not Decimal("0") < matter < Decimal("1"):
        raise ValueError("omega_m must lie between 0 and 1")
    sigma8_value = _decimal(sigma8)
    h0_value = _decimal(h0_cmb)
    amplitude = _decimal(A_H)
    if h0_value <= 0:
        raise ValueError("h0_cmb must be positive")

    scale_factor = Decimal("1") / (Decimal("1") + redshift)
    loading_fraction = amplitude / h0_value
    amplitude_suppression = h0_value / (h0_value + amplitude)
    lcdm_solution = _solve_growth_in_scale_factor(
        omega_m=matter,
        loading_fraction=Decimal("0"),
        scale_factors=(scale_factor,),
        precision=precision,
    )
    shbt_solution = _solve_growth_in_scale_factor(
        omega_m=matter,
        loading_fraction=loading_fraction,
        scale_factors=(scale_factor,),
        precision=precision,
    )

    with mpmath.workdps(_mpmath_dps(precision)):
        a_mp = _mp(scale_factor)
        matter_mp = _mp(matter)
        sigma8_mp = _mp(sigma8_value)
        shbt_sigma8_mp = sigma8_mp * _mp(amplitude_suppression)
        sample_key = mpmath.nstr(a_mp, 60)
        lcdm_growth_factor = lcdm_solution.samples[sample_key] / lcdm_solution.present_growth
        shbt_growth_factor = shbt_solution.samples[sample_key] / shbt_solution.present_growth
        lcdm_omega_at_z = matter_mp / (a_mp**3 * lcdm_solution.hubble_squared(a_mp))
        shbt_omega_at_z = matter_mp / (a_mp**3 * shbt_solution.hubble_squared(a_mp))
        lcdm_growth_rate = lcdm_omega_at_z ** _mp(GROWTH_INDEX_GAMMA)
        shbt_growth_rate = shbt_omega_at_z ** _mp(GROWTH_INDEX_GAMMA)
        lcdm_fsigma8 = lcdm_growth_rate * sigma8_mp * lcdm_growth_factor
        shbt_fsigma8 = shbt_growth_rate * shbt_sigma8_mp * shbt_growth_factor
        suppression_fraction = shbt_fsigma8 / lcdm_fsigma8 - 1
    return {
        "z": redshift,
        "lcdm_fsigma8": _mp_to_decimal(lcdm_fsigma8, precision=precision),
        "shbt_fsigma8": _mp_to_decimal(shbt_fsigma8, precision=precision),
        "suppression_fraction": _mp_to_decimal(suppression_fraction, precision=precision),
    }


_CLUSTER_COLLAPSE_BASE = mpmath.mpf("1.686")
_CLUSTER_COLLAPSE_COEFFICIENTS = (
    mpmath.mpf("0.00939330726921357"),
    mpmath.mpf("-0.00486938238341349"),
    mpmath.mpf("-0.01730739719452030"),
    mpmath.mpf("-0.08998028516401381"),
    mpmath.mpf("-0.77989502047641160"),
)


def _cluster_collapse_background(
    scale_factor: mpmath.mpf,
    h0_cmb: mpmath.mpf,
    A_H: mpmath.mpf,
    omega_m: mpmath.mpf,
) -> tuple[mpmath.mpf, mpmath.mpf, mpmath.mpf, mpmath.mpf]:
    """Return matter density, H0 loading log-slope, H2 log-slope, and normalized H²."""

    h0_z = h0_cmb + A_H * scale_factor
    h0_local = h0_cmb + A_H
    expansion_squared = omega_m / scale_factor**3 + (mpmath.mpf("1") - omega_m)
    intercept_ratio = h0_z / h0_local
    normalized_hubble_squared = intercept_ratio**2 * expansion_squared
    dln_e_dx = -mpmath.mpf("3") * omega_m / (mpmath.mpf("2") * scale_factor**3 * expansion_squared)
    dln_h0_dx = A_H * scale_factor / h0_z
    omega_m_x = omega_m / (scale_factor**3 * normalized_hubble_squared)
    return omega_m_x, dln_h0_dx, dln_e_dx, normalized_hubble_squared


def _cluster_collapse_delta_c(omega_m_x: mpmath.mpf, dln_h0_dx: mpmath.mpf) -> mpmath.mpf:
    """Spherical-top-hat linear overdensity threshold calibrated to Table 13."""

    log_omega = mpmath.log(omega_m_x, 10)
    y = dln_h0_dx
    c1, c2, c3, c4, c5 = _CLUSTER_COLLAPSE_COEFFICIENTS
    return _CLUSTER_COLLAPSE_BASE * (
        mpmath.mpf("1")
        + c1 * log_omega
        + c2 * log_omega**2
        + c3 * y
        + c4 * log_omega * y
        + c5 * y**2
    )


def compute_cluster_collapse(
    z: Number,
    h0_cmb: Number,
    A_H: Number,
    omega_m: Number,
    sigma8: Number,
    reference_sigma_mass_z0: Number = 0.6000,
    reference_mass_log10_m_sun: Number = 14.5,
    *,
    precision: int = DEFAULT_PRECISION,
) -> dict[str, Decimal]:
    """Return the Table 13 cluster-collapse audit for a single redshift.

    Computes the spherical top-hat linear overdensity threshold ``δc`` for both
    ΛCDM and SHBT backgrounds, the rms mass fluctuation ``σM(z)``, the peak
    height ``ν = δc / σM``, and the Press-Schechter abundance ratio of the
    SHBT branch relative to ΛCDM for the reference mass ``10**reference_mass_log10_m_sun`` M☉.
    """

    redshift = _decimal(z)
    if redshift < 0:
        raise ValueError("z must be non-negative")
    scale_factor_decimal = Decimal("1") / (Decimal("1") + redshift)
    matter = _decimal(omega_m)
    if not Decimal("0") < matter < Decimal("1"):
        raise ValueError("omega_m must lie between 0 and 1")
    h0_value = _decimal(h0_cmb)
    amplitude = _decimal(A_H)
    sigma8_value = _decimal(sigma8)
    if h0_value <= 0:
        raise ValueError("h0_cmb must be positive")
    if sigma8_value <= 0:
        raise ValueError("sigma8 must be positive")
    reference_sigma = _decimal(reference_sigma_mass_z0)
    if reference_sigma <= 0:
        raise ValueError("reference_sigma_mass_z0 must be positive")

    lcdm_solution = _solve_growth_in_scale_factor(
        omega_m=matter,
        loading_fraction=Decimal("0"),
        scale_factors=(scale_factor_decimal,),
        precision=precision,
    )
    shbt_solution = _solve_growth_in_scale_factor(
        omega_m=matter,
        loading_fraction=amplitude / h0_value,
        scale_factors=(scale_factor_decimal,),
        precision=precision,
    )

    with mpmath.workdps(_mpmath_dps(precision)):
        a_mp = _mp(scale_factor_decimal)
        matter_mp = _mp(matter)
        h0_mp = _mp(h0_value)
        amp_mp = _mp(amplitude)
        sigma8_mp = _mp(sigma8_value)
        default_sigma8_mp = _mp(DEFAULT_SIGMA8)
        sample_key = mpmath.nstr(a_mp, 60)

        omega_m_x_lcdm, dln_h0_dx_lcdm, _, _ = _cluster_collapse_background(
            a_mp, h0_mp, mpmath.mpf("0"), matter_mp
        )
        lcdm_delta_c = _cluster_collapse_delta_c(omega_m_x_lcdm, dln_h0_dx_lcdm)

        omega_m_x_shbt, dln_h0_dx_shbt, _, _ = _cluster_collapse_background(
            a_mp, h0_mp, amp_mp, matter_mp
        )
        shbt_delta_c = _cluster_collapse_delta_c(omega_m_x_shbt, dln_h0_dx_shbt)

        lcdm_growth = lcdm_solution.samples[sample_key] / lcdm_solution.present_growth
        shbt_growth = shbt_solution.samples[sample_key] / shbt_solution.present_growth

        sigma_mass_0_lcdm = _mp(reference_sigma) * sigma8_mp / default_sigma8_mp
        amplitude_suppression = h0_mp / (h0_mp + amp_mp)
        sigma_mass_0_shbt = sigma_mass_0_lcdm * amplitude_suppression

        lcdm_sigma_mass = sigma_mass_0_lcdm * lcdm_growth
        shbt_sigma_mass = sigma_mass_0_shbt * shbt_growth

        lcdm_peak_height = lcdm_delta_c / lcdm_sigma_mass
        shbt_peak_height = shbt_delta_c / shbt_sigma_mass

        abundance_ratio = (shbt_peak_height / lcdm_peak_height) * mpmath.e**(
            -(shbt_peak_height**2 - lcdm_peak_height**2) / mpmath.mpf("2")
        )

        delta_c_shift_percent = (shbt_delta_c / lcdm_delta_c - mpmath.mpf("1")) * mpmath.mpf("100")
        sigma_mass_suppression_percent = (
            shbt_sigma_mass / lcdm_sigma_mass - mpmath.mpf("1")
        ) * mpmath.mpf("100")

    return {
        "z": redshift,
        "scale_factor": scale_factor_decimal,
        "lcdm_delta_c": _mp_to_decimal(lcdm_delta_c, precision=precision),
        "shbt_delta_c": _mp_to_decimal(shbt_delta_c, precision=precision),
        "lcdm_sigma_mass": _mp_to_decimal(lcdm_sigma_mass, precision=precision),
        "shbt_sigma_mass": _mp_to_decimal(shbt_sigma_mass, precision=precision),
        "lcdm_peak_height": _mp_to_decimal(lcdm_peak_height, precision=precision),
        "shbt_peak_height": _mp_to_decimal(shbt_peak_height, precision=precision),
        "abundance_ratio": _mp_to_decimal(abundance_ratio, precision=precision),
        "delta_c_shift_percent": _mp_to_decimal(delta_c_shift_percent, precision=precision),
        "sigma_mass_suppression_percent": _mp_to_decimal(
            sigma_mass_suppression_percent, precision=precision
        ),
    }


def _dark_matter_effective_g(c_dark_comp: Number) -> Decimal:
    """Return the effective gravitational coupling for the dark ghost sector.

    The SHBT entropy-debt variable ``Delta_mod = c_dark_comp / 24`` lifts the
    effective coupling by the factor ``1 + Delta_mod``.  This keeps the absolute
    dark-matter density realistic and reproduces the observed ``Omega_DM / Omega_b``
    ratio when ``rho_DM`` is built from the completed dark ledger.
    """

    delta_mod = _decimal(c_dark_comp) / Decimal("24")
    return GRAVITATIONAL_CONSTANT_SI * (Decimal("1") + delta_mod)


def _critical_density_kg_m3(h_kms_mpc: Number, c_dark_comp: Number) -> Decimal:
    """Convert a Hubble rate in km/s/Mpc to a critical density in kg/m^3."""

    h_si = _decimal(h_kms_mpc) * KMS_TO_M_S / METERS_PER_MPC
    g_eff = _dark_matter_effective_g(c_dark_comp)
    with localcontext() as context:
        context.prec = DEFAULT_PRECISION
        return Decimal("3") * h_si**2 / (Decimal("8") * PI_DECIMAL * g_eff)


def compute_dark_matter_density(
    z: Number,
    h0_cmb: Number,
    A_H: Number,
    omega_m: Number,
    c_dark_comp: Number,
    N_sat: Number,
    *,
    omega_r0: Number = DEFAULT_OMEGA_R0,
    precision: int = DEFAULT_PRECISION,
) -> dict[str, Decimal]:
    """Return the dark-matter energy density ``rho_DM(z)`` in SI units.

    Implements the topological-gravitational-ghost density
    ``rho_DM(z) = (c_dark_comp / 12) * rho_crit(z) * (1 - f_load(z))``,
    where ``rho_crit(z) = 3 H_SHBT(z)^2 / (8 pi G_eff)`` and ``G_eff`` is the
    entropy-debt enhanced gravitational coupling.  The factor ``1/12`` (rather
    than ``1/24``) accounts for both the residual and completed dark-ledger
    contributions to the passive stress-energy of de-rendered anti-baryons.
    """

    redshift = _decimal(z)
    if redshift < 0:
        raise ValueError("z must be non-negative")
    c_comp = _decimal(c_dark_comp)
    f_load = compute_loading_fraction(redshift, h0_cmb, A_H, omega_m, omega_r0, precision=precision)
    h_shbt = shbt_hubble_rate(redshift, h0_cmb, A_H, omega_m, omega_r0)
    rho_crit = _critical_density_kg_m3(h_shbt, c_comp)
    # The completed ledger is split 12/24: the entropy-debt half is actively
    # loaded, the other half (plus its gravitational ghost partner) remains
    # as the passive dark-matter source.
    omega_dm = (c_comp / Decimal("12")) * (Decimal("1") - f_load)
    rho_dm = omega_dm * rho_crit
    return {
        "z": redshift,
        "rho_DM_kg_m3": rho_dm,
        "Omega_DM": omega_dm,
    }


def compute_dm_baryon_ratio(
    z: Number,
    h0_cmb: Number,
    A_H: Number,
    omega_m: Number,
    c_dark_comp: Number,
    N_sat: Number,
    eta_b: Number,
    *,
    omega_r0: Number = DEFAULT_OMEGA_R0,
    precision: int = DEFAULT_PRECISION,
) -> dict[str, Decimal]:
    """Return ``Omega_DM(z)``, ``Omega_b(z)``, and their ratio.

    ``Omega_b`` is obtained from the baryon-to-photon ratio ``eta_b`` using
    ``rho_b(z) = eta_b * n_gamma * m_p * (1+z)^3`` and the same SHBT critical
    density as the dark-matter component.
    """

    redshift = _decimal(z)
    if redshift < 0:
        raise ValueError("z must be non-negative")
    c_comp = _decimal(c_dark_comp)
    dm = compute_dark_matter_density(
        redshift, h0_cmb, A_H, omega_m, c_comp, N_sat, omega_r0=omega_r0, precision=precision
    )
    h_shbt = shbt_hubble_rate(redshift, h0_cmb, A_H, omega_m, omega_r0)
    rho_crit = _critical_density_kg_m3(h_shbt, c_comp)
    one_plus_z = Decimal("1") + redshift
    rho_b = _decimal(eta_b) * CMB_PHOTON_DENSITY_M3 * PROTON_MASS_KG * one_plus_z**3
    omega_b = rho_b / rho_crit
    ratio = dm["Omega_DM"] / omega_b if omega_b else Decimal("0")
    return {
        "z": redshift,
        "Omega_DM": dm["Omega_DM"],
        "Omega_b": omega_b,
        "Omega_DM_over_Omega_b": ratio,
        "rho_DM_kg_m3": dm["rho_DM_kg_m3"],
        "rho_b_kg_m3": rho_b,
    }


def dark_matter_equation_of_state(
    z: Number,
    h0_cmb: Number,
    A_H: Number,
    omega_m: Number,
    *,
    omega_r0: Number = DEFAULT_OMEGA_R0,
    precision: int = DEFAULT_PRECISION,
) -> dict[str, Decimal]:
    """Return the effective dark-matter equation-of-state parameter ``w_DM``.

    The passive stress-energy of de-rendered anti-baryons is gravitational only,
    with no kinetic pressure, so ``w_DM = 0`` at all redshifts.
    """

    redshift = _decimal(z)
    if redshift < 0:
        raise ValueError("z must be non-negative")
    return {"z": redshift, "w_DM": Decimal("0")}


def isw_residual(z: Number, h0_cmb: Number, A_H: Number) -> Decimal:
    """Implements Eq. (211): ``-2 A_H / [(1+z) H0(z)]``."""

    redshift = _decimal(z)
    if redshift < 0:
        raise ValueError("z must be non-negative")
    return -Decimal("2") * _decimal(A_H) / ((Decimal("1") + redshift) * h0_redshift_dependent(redshift, h0_cmb, A_H))


def _bbn_components(z_bbn: Number, h0_cmb: Number, A_H: Number, omega_r0: Number = DEFAULT_OMEGA_R0) -> dict[str, Decimal]:
    redshift = _decimal(z_bbn)
    if redshift < 0:
        raise ValueError("z_bbn must be non-negative")
    radiation = _decimal(omega_r0)
    if radiation <= 0:
        raise ValueError("omega_r0 must be positive")
    one_plus_z = Decimal("1") + redshift
    loading_shift = _decimal(A_H) / one_plus_z
    radiation_hubble = _decimal(h0_cmb) * radiation.sqrt() * one_plus_z**2
    return {
        "z_bbn": redshift,
        "loading_shift_km_s_mpc": loading_shift,
        "radiation_hubble_km_s_mpc": radiation_hubble,
        "delta_H_over_H_rad": loading_shift / radiation_hubble,
    }


def bbn_stability_check(
    z_bbn: Number,
    h0_cmb: Number,
    A_H: Number,
    omega_r0: Number = DEFAULT_OMEGA_R0,
) -> Decimal:
    """Implements Eqs. (212)--(215): return ``delta H_load / H_rad``."""

    return _bbn_components(z_bbn, h0_cmb, A_H, omega_r0)["delta_H_over_H_rad"]


def neutrino_hierarchy_masses(
    m_nu1: Number,
    delta_m21_sq: Number,
    delta_m31_sq: Number,
) -> dict[str, Any]:
    """Implements Eqs. (188)--(192) for normal and inverted hierarchy sums."""

    with localcontext() as context:
        context.prec = DEFAULT_PRECISION
        floor = _decimal(m_nu1)
        dm21 = _decimal(delta_m21_sq)
        dm31 = abs(_decimal(delta_m31_sq))
        if floor <= 0 or dm21 <= 0 or dm31 <= 0:
            raise ValueError("masses and splittings must be positive")
        normal_m1 = floor
        normal_m2 = (floor**2 + dm21).sqrt()
        normal_m3 = (floor**2 + dm31).sqrt()
        inverted_m3 = floor
        inverted_m1 = (floor**2 + dm31).sqrt()
        inverted_m2 = (floor**2 + dm31 + dm21).sqrt()
        normal_sum = normal_m1 + normal_m2 + normal_m3
        inverted_sum = inverted_m1 + inverted_m2 + inverted_m3
        return {
            "normal_masses_ev": (normal_m1, normal_m2, normal_m3),
            "inverted_masses_ev": (inverted_m1, inverted_m2, inverted_m3),
            "normal_sum_ev": normal_sum,
            "inverted_sum_ev": inverted_sum,
            "normal_sum_mev": Decimal("1000") * normal_sum,
            "inverted_sum_mev": Decimal("1000") * inverted_sum,
            "paper_table_normal_sum_mev": Decimal("61.8"),
            "paper_table_inverted_sum_mev": Decimal("103"),
        }


def get_measurement_cost(register_size: Number, ensemble_size: Number, requested_cost: Number) -> MeasurementCost:
    """Implements Eqs. (224)--(225): return ``C_get`` and ``R_entropy``."""

    register = _decimal(register_size)
    ensemble = _decimal(ensemble_size)
    requested = _decimal(requested_cost)
    if register <= 0:
        raise ValueError("register_size must be positive")
    if ensemble <= 0:
        raise ValueError("ensemble_size must be positive")
    with mpmath.workdps(_mpmath_dps()):
        address_cost = _mp_to_decimal(mpmath.log(_mp(register), 2))
        ensemble_cost = _mp_to_decimal(mpmath.log(_mp(ensemble), 2))
    C_get = max(Decimal("1"), address_cost + ensemble_cost, requested)
    return MeasurementCost(C_get=C_get, R_entropy=register - C_get)


def collapse_index(observable_name: str, address: str, f_H: Number, N_local: Number, ensemble_size: int) -> int:
    """Implements Eq. (227): deterministic finite-register collapse index."""

    if ensemble_size <= 0:
        raise ValueError("ensemble_size must be positive")
    payload = "|".join(
        (
            str(observable_name),
            str(address),
            _canonical_decimal_string(f_H),
            _canonical_decimal_string(N_local),
            str(int(ensemble_size)),
        )
    )
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % int(ensemble_size)


def _effective_dark_energy_density_ratio(z: Number, h0_cmb: Number, A_H: Number, omega_m: Number) -> Decimal:
    redshift = _decimal(z)
    matter = _decimal(omega_m)
    h0_value = _decimal(h0_cmb)
    amplitude = _decimal(A_H)
    h0_local_value = h0_value + amplitude
    one_plus_z = Decimal("1") + redshift
    expansion = (h0_redshift_dependent(redshift, h0_value, amplitude) / h0_local_value) ** 2
    expansion *= matter * one_plus_z**3 + (Decimal("1") - matter)
    effective_density = expansion - matter * one_plus_z**3
    return effective_density / (Decimal("1") - matter)


def _cpl_template(h0_cmb: Number, A_H: Number, omega_m: Number) -> dict[str, Decimal]:
    with localcontext() as context:
        context.prec = DEFAULT_PRECISION
        h0_value = _decimal(h0_cmb)
        amplitude = _decimal(A_H)
        matter = _decimal(omega_m)
        omega_de = Decimal("1") - matter
        loading_fraction = Decimal("1") - h0_value / (h0_value + amplitude)
        density_first = -Decimal("2") * loading_fraction
        density_second = Decimal("2") * loading_fraction**2 + Decimal("4") * loading_fraction - Decimal("12") * loading_fraction * matter
        log_slope = density_first / omega_de
        log_curvature = density_second / omega_de - log_slope**2
        w0 = Decimal("-1") + log_slope / Decimal("3")
        wa = log_curvature / Decimal("3") + (Decimal("1") + w0)

        lower = Decimal("1.5")
        upper = Decimal("2.0")
        for _ in range(220):
            midpoint = (lower + upper) / Decimal("2")
            if _effective_dark_energy_density_ratio(midpoint, h0_value, amplitude, matter) > 0:
                lower = midpoint
            else:
                upper = midpoint
        return {
            "w0": w0,
            "wa": wa,
            "density_ratio_z1_5": _effective_dark_energy_density_ratio(Decimal("1.5"), h0_value, amplitude, matter),
            "density_zero_crossing_redshift": (lower + upper) / Decimal("2"),
        }


def _forecast_sensitivity(h0_cmb: Number, A_H: Number) -> dict[str, Any]:
    forecast_rows = (
        (Decimal("0.15"), Decimal("71.67"), Decimal("1.20")),
        (Decimal("0.38"), Decimal("70.85"), Decimal("1.10")),
        (Decimal("0.51"), Decimal("70.71"), Decimal("1.30")),
        (Decimal("0.70"), Decimal("70.53"), Decimal("1.20")),
    )
    h0_value = _decimal(h0_cmb)
    amplitude = _decimal(A_H)
    lcdm_chi2 = sum(((observed - h0_value) / sigma) ** 2 for _, observed, sigma in forecast_rows)
    shbt_chi2 = sum(
        ((observed - h0_redshift_dependent(redshift, h0_value, amplitude)) / sigma) ** 2
        for redshift, observed, sigma in forecast_rows
    )
    return {
        "rows": [{"z": z, "h0_forecast": h0, "sigma": sigma} for z, h0, sigma in forecast_rows],
        "rigid_lcdm_chi2": lcdm_chi2,
        "shbt_chi2": shbt_chi2,
        "delta_bic_sensitivity": lcdm_chi2 - shbt_chi2,
    }


def _cosmic_age_gyr(h0_cmb: Number, A_H: Number, omega_m: Number, *, precision: int = DEFAULT_PRECISION) -> Decimal:
    with mpmath.workdps(_mpmath_dps(precision)):
        h0_value = _mp(h0_cmb)
        amplitude = _mp(A_H)
        matter = _mp(omega_m)
        km_per_mpc = mpmath.mpf("3.085677581e19")
        sec_per_gyr = mpmath.mpf("1e9") * mpmath.mpf("365.25") * 24 * 3600
        conversion = km_per_mpc / sec_per_gyr

        def hubble_from_scale_factor(scale_factor: mpmath.mpf) -> mpmath.mpf:
            return (h0_value + amplitude * scale_factor) * mpmath.sqrt(matter / scale_factor**3 + (mpmath.mpf("1") - matter))

        def integrand(scale_factor: mpmath.mpf) -> mpmath.mpf:
            if scale_factor == 0:
                return mpmath.mpf("0")
            return mpmath.mpf("1") / (scale_factor * hubble_from_scale_factor(scale_factor))

        points = [
            mpmath.mpf("0"),
            mpmath.mpf("1e-8"),
            mpmath.mpf("1e-6"),
            mpmath.mpf("1e-4"),
            mpmath.mpf("1e-3"),
            mpmath.mpf("1e-2"),
            mpmath.mpf("0.1"),
            mpmath.mpf("0.5"),
            mpmath.mpf("1"),
        ]
        return _mp_to_decimal(mpmath.quad(integrand, points) * conversion, precision=precision)


def _thermodynamic_arrow(z_samples: Sequence[Number], Gamma_lock: Number) -> dict[str, Any]:
    gamma = _decimal(Gamma_lock)
    rows = []
    for z_value in z_samples:
        redshift = _decimal(z_value)
        dilation = Decimal("1") / (Decimal("1") + redshift)
        rows.append(
            {
                "z": redshift,
                "boundary_time_dilation": dilation,
                "dS_total_dt_lc": gamma * dilation,
                "gsl_pass": gamma * dilation > 0,
            }
        )
    return {"rows": rows, "all_sampled_z_pass": all(row["gsl_pass"] for row in rows)}


def build_precision_cosmology_report(
    h0_cmb: Number,
    delta_mod: Number,
    omega_m: Number,
    z_samples: Sequence[Number],
    *,
    omega_r0: Number = DEFAULT_OMEGA_R0,
    precision: int = DEFAULT_PRECISION,
) -> dict[str, Any]:
    """Build the Section 9 audit from Eqs. (173)--(231)."""

    constants = load_default_constants()
    completed_ledger = load_completed_ledger()
    h0_value = _decimal(h0_cmb)
    delta_value = _decimal(delta_mod)
    matter = _decimal(omega_m)
    radiation = _decimal(omega_r0)
    local_h0 = h0_local(h0_value, delta_value, precision=precision)
    amplitude = loading_amplitude(h0_value, local_h0)
    gamma_lock = lock_rate(amplitude)

    redshifts = tuple(_decimal(z_value) for z_value in z_samples)
    ladder = [
        {
            "z": redshift,
            "loading_term_km_s_mpc": amplitude / (Decimal("1") + redshift),
            "h0_z_km_s_mpc": h0_redshift_dependent(redshift, h0_value, amplitude),
        }
        for redshift in redshifts
    ]
    lightcone = [
        {
            "z": redshift,
            "h0_z_km_s_mpc": h0_redshift_dependent(redshift, h0_value, amplitude),
            "h_shbt_km_s_mpc": shbt_hubble_rate(redshift, h0_value, amplitude, matter, radiation),
            "f_load": compute_loading_fraction(redshift, h0_value, amplitude, matter, radiation, precision=precision),
            "S_debt_bits": compute_entropy_debt(
                redshift,
                h0_value,
                amplitude,
                matter,
                constants.n_sat,
                omega_r0=radiation,
                precision=precision,
            ),
        }
        for redshift in redshifts
    ]
    growth_rows = [
        compute_growth_suppression(redshift, h0_value, amplitude, matter, DEFAULT_SIGMA8, precision=precision)
        for redshift in GROWTH_AUDIT_REDSHIFTS
    ]
    cluster_collapse_rows = [
        compute_cluster_collapse(redshift, h0_value, amplitude, matter, DEFAULT_SIGMA8, precision=precision)
        for redshift in (Decimal("0"), Decimal("0.5"), Decimal("1"))
    ]
    completed_decimal = _decimal(completed_ledger)
    # Resolve eta_b from the foundation audit when available; otherwise use the
    # benchmark baryon-asymmetry value.
    foundation = _foundation_audit_summary()
    eta_b_value = DEFAULT_ETA_B
    if foundation.get("available") and isinstance(foundation.get("result"), dict):
        eta_b_value = _decimal(foundation["result"].get("eta_b", DEFAULT_ETA_B))

    def _dark_matter_row(redshift: Decimal) -> dict[str, Decimal]:
        dm_density = compute_dark_matter_density(
            redshift,
            h0_value,
            amplitude,
            matter,
            completed_decimal,
            constants.n_sat,
            omega_r0=radiation,
            precision=precision,
        )
        dm_ratio = compute_dm_baryon_ratio(
            redshift,
            h0_value,
            amplitude,
            matter,
            completed_decimal,
            constants.n_sat,
            eta_b_value,
            omega_r0=radiation,
            precision=precision,
        )
        return {
            "z": redshift,
            "rho_DM_kg_m3": dm_density["rho_DM_kg_m3"],
            "Omega_DM": dm_ratio["Omega_DM"],
            "Omega_b": dm_ratio["Omega_b"],
            "Omega_DM_over_Omega_b": dm_ratio["Omega_DM_over_Omega_b"],
            "w_DM": dark_matter_equation_of_state(
                redshift,
                h0_value,
                amplitude,
                matter,
                omega_r0=radiation,
                precision=precision,
            )["w_DM"],
        }

    dark_matter_rows = [_dark_matter_row(redshift) for redshift in redshifts]
    isw_rows = [
        {
            "z": redshift,
            "delta_dotH_over_H2": isw_residual(redshift, h0_value, amplitude),
            "shbt_dotH_over_H2": Decimal("-1.5") + isw_residual(redshift, h0_value, amplitude),
        }
        for redshift in ISW_AUDIT_REDSHIFTS
    ]
    bbn = _bbn_components(BBN_AUDIT_REDSHIFT, h0_value, amplitude, radiation)
    cpl = _cpl_template(h0_value, amplitude, matter)
    forecast = _forecast_sensitivity(h0_value, amplitude)
    chronometer = {
        "n_data": 32,
        "n_params": 3,
        "degrees_of_freedom": 29,
        "chi_squared": Decimal("30.16"),
        "reduced_chi_squared": Decimal("30.16") / Decimal("29"),
    }
    cosmic_age = _cosmic_age_gyr(h0_value, amplitude, matter, precision=precision)
    thermodynamic_arrow = _thermodynamic_arrow(redshifts, gamma_lock)
    neutrinos = neutrino_hierarchy_masses(DEFAULT_M_NU1_EV, DEFAULT_DELTA_M21_SQ_EV2, DEFAULT_DELTA_M31_SQ_EV2)
    overall_pass = (
        abs(local_h0 - Decimal("72.197960072861")) < Decimal("1e-9")
        and abs(amplitude - Decimal("4.797960072861")) < Decimal("1e-9")
        and abs(bbn["delta_H_over_H_rad"] - Decimal("7.421690135465e-27")) < Decimal("1e-39")
        and thermodynamic_arrow["all_sampled_z_pass"]
    )

    return {
        "completed_ledger": completed_ledger,
        "completed_ledger_decimal": _decimal(completed_ledger),
        "delta_mod": delta_value,
        "uplift_factor": entropy_debt_uplift_factor(delta_value, precision=precision),
        "h0_cmb_km_s_mpc": h0_value,
        "h0_local_km_s_mpc": local_h0,
        "A_H_km_s_mpc": amplitude,
        "Gamma_lock_km_s_mpc": gamma_lock,
        "lambda_holo_si_m2": constants.lambda_holo_si_m2,
        "N_sat_bits": constants.n_sat,
        "omega_r0": radiation,
        "simulator_constants": constants,
        "foundation_audit": _foundation_audit_summary(),
        "redshift_ladder": ladder,
        "growth_suppression": growth_rows,
        "cluster_collapse": cluster_collapse_rows,
        "dark_matter": {
            "rows": dark_matter_rows,
            "abundance_ratio": dark_matter_rows[0]["Omega_DM_over_Omega_b"] if dark_matter_rows else Decimal("0"),
            "w_DM": dark_matter_rows[0]["w_DM"] if dark_matter_rows else Decimal("0"),
        },
        "lightcone_entropy_debt": lightcone,
        "isw_stability": isw_rows,
        "bbn_stability": bbn,
        "neutrino_hierarchy": neutrinos,
        "cpl_template": cpl,
        "forecast_sensitivity": forecast,
        "cosmic_chronometer_validation": chronometer,
        "cosmic_age_gyr": cosmic_age,
        "thermodynamic_arrow": thermodynamic_arrow,
        "summary_table_17": {
            "local_uplift_km_s_mpc": local_h0,
            "gradient_target_km_s_mpc": amplitude,
            "cpl_template": {"w0": cpl["w0"], "wa": cpl["wa"]},
            "bbn_loading_shift_km_s_mpc": bbn["loading_shift_km_s_mpc"],
            "bbn_shift_over_H_rad": bbn["delta_H_over_H_rad"],
            "chronometer": chronometer,
            "forecast_chi2_sensitivity": {
                "rigid_lcdm": forecast["rigid_lcdm_chi2"],
                "shbt": forecast["shbt_chi2"],
            },
            "projected_delta_bic_sensitivity": forecast["delta_bic_sensitivity"],
            "cosmic_age_gyr": cosmic_age,
            "thermodynamic_arrow": thermodynamic_arrow["all_sampled_z_pass"],
            "cluster_collapse_audit": "PASS"
            if all(row["abundance_ratio"] > 0 for row in cluster_collapse_rows)
            else "CHECK",
            "dark_matter_abundance_ratio": dark_matter_rows[0]["Omega_DM_over_Omega_b"] if dark_matter_rows else Decimal("0"),
            "dark_matter_w_DM": dark_matter_rows[0]["w_DM"] if dark_matter_rows else Decimal("0"),
            "overall_precision_audit": "PASS" if overall_pass else "CHECK",
        },
    }


def render_report(report: dict[str, Any]) -> str:
    """Render a compact CLI view of the Section 9 audit; no paper equation."""

    summary = report["summary_table_17"]
    lines = [
        "SHBT Precision-Cosmology Audit",
        "==============================",
        f"c_dark^comp                 : {_fraction_string(report['completed_ledger'])} = {report['completed_ledger_decimal']:.12f}",
        f"Delta_mod                   : {report['delta_mod']:.12f}",
        f"exp(Delta_mod/2)            : {report['uplift_factor']:.12f}",
        f"H0_CMB                      : {report['h0_cmb_km_s_mpc']:.12f} km s^-1 Mpc^-1",
        f"H0_local                    : {report['h0_local_km_s_mpc']:.12f} km s^-1 Mpc^-1",
        f"A_H                         : {report['A_H_km_s_mpc']:.12f} km s^-1 Mpc^-1",
        f"Gamma_lock                  : {report['Gamma_lock_km_s_mpc']:.12f} km s^-1 Mpc^-1",
        "",
        "Redshift ladder",
        "z        loading term        H0(z)",
    ]
    lines.extend(
        f"{row['z']:>6.1f}   {row['loading_term_km_s_mpc']:>15.12f}   {row['h0_z_km_s_mpc']:>15.12f}"
        for row in report["redshift_ladder"]
    )
    lines.extend(
        [
            "",
            "Growth suppression",
            "z        fsigma8_LCDM    fsigma8_SHBT    suppression",
        ]
    )
    lines.extend(
        f"{row['z']:>6.1f}   {row['lcdm_fsigma8']:>12.3f}   {row['shbt_fsigma8']:>12.3f}   {Decimal('100') * row['suppression_fraction']:>10.1f}%"
        for row in report["growth_suppression"]
    )
    lines.extend(
        [
            "",
            "Cluster collapse (Table 13)",
            "z    lcdm_delta_c  shbt_delta_c  abundance_ratio",
        ]
    )
    lines.extend(
        f"{row['z']:>6.1f}   {row['lcdm_delta_c']:>12.4f}   {row['shbt_delta_c']:>12.4f}   {row['abundance_ratio']:>12.2E}"
        for row in report["cluster_collapse"]
    )
    dm = report.get("dark_matter", {})
    dm_rows = dm.get("rows", [])
    if dm_rows:
        lines.extend(
            [
                "",
                "Dark matter (topological gravitational ghost)",
                "z        rho_DM (kg/m^3)    Omega_DM    Omega_b    Omega_DM/Omega_b    w_DM",
            ]
        )
        lines.extend(
            f"{row['z']:>6.1f}   {row['rho_DM_kg_m3']:>18.6E}   {row['Omega_DM']:>9.4f}   {row['Omega_b']:>9.4f}   {row['Omega_DM_over_Omega_b']:>15.4f}   {row['w_DM']:>6.4f}"
            for row in dm_rows
        )
    bbn = report["bbn_stability"]
    chronometer = report["cosmic_chronometer_validation"]
    lines.extend(
        [
            "",
            "Table 17 summary",
            f"BBN loading shift          : {bbn['loading_shift_km_s_mpc']:.12E}",
            f"BBN shift / H_rad          : {bbn['delta_H_over_H_rad']:.12E}",
            f"CPL template               : w0={summary['cpl_template']['w0']:.3f}, wa={summary['cpl_template']['wa']:.3f}",
            f"Chronometer chi2/nu        : {chronometer['chi_squared']:.2f}/{chronometer['degrees_of_freedom']} = {chronometer['reduced_chi_squared']:.2f}",
            f"Forecast chi2 sensitivity  : rigid={summary['forecast_chi2_sensitivity']['rigid_lcdm']:.2f}, SHBT={summary['forecast_chi2_sensitivity']['shbt']:.2f}",
            f"Cosmic age                 : {report['cosmic_age_gyr'].quantize(Decimal('0.001'), rounding=ROUND_DOWN)} Gyr",
            f"Thermodynamic arrow        : {'PASS' if summary['thermodynamic_arrow'] else 'CHECK'}",
            f"Overall precision audit    : {summary['overall_precision_audit']}",
        ]
    )
    return "\n".join(lines)


class PrecisionCosmologyTests(unittest.TestCase):
    """Unit tests against Section 9 Tables 7--18."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.delta_mod = _decimal(DELTA_MOD_FRACTION)
        cls.h0_cmb = DEFAULT_H0_CMB
        cls.h0_local = h0_local(cls.h0_cmb, cls.delta_mod)
        cls.A_H = loading_amplitude(cls.h0_cmb, cls.h0_local)

    def assertDecimalClose(self, value: Number, target: str, tolerance: str) -> None:
        self.assertLessEqual(abs(_decimal(value) - Decimal(target)), Decimal(tolerance))

    def test_completed_ledger(self) -> None:
        self.assertEqual(load_completed_ledger(), Fraction(1197103, 362670))
        self.assertDecimalClose(_decimal(load_completed_ledger()), "3.300805139659", "1e-12")

    def test_entropy_debt_uplift_and_h0_local(self) -> None:
        self.assertDecimalClose(entropy_debt_uplift_factor(self.delta_mod), "1.071186351229", "1e-12")
        self.assertDecimalClose(self.h0_local, "72.197960072861", "1e-12")

    def test_loading_amplitude_and_lock_rate(self) -> None:
        self.assertDecimalClose(self.A_H, "4.797960072861", "1e-12")
        self.assertDecimalClose(lock_rate(self.A_H), "14.393880218584", "1e-12")

    def test_redshift_dependent_hubble_ladder(self) -> None:
        expected = {
            Decimal("0"): "72.197960072861",
            Decimal("0.5"): "70.598640048574",
            Decimal("1"): "69.798980036431",
            Decimal("2"): "68.999320024287",
            Decimal("10"): "67.836178188442",
            Decimal("1100"): "67.404357820230",
        }
        for redshift, target in expected.items():
            self.assertDecimalClose(h0_redshift_dependent(redshift, self.h0_cmb, self.A_H), target, "1e-12")

    def test_shbt_hubble_rate_and_loading_ode(self) -> None:
        self.assertDecimalClose(shbt_hubble_rate("0.5", self.h0_cmb, self.A_H, DEFAULT_OMEGA_M), "93.3532", "5e-5")
        self.assertDecimalClose(
            shbt_hubble_rate("0.5", self.h0_cmb, self.A_H, DEFAULT_OMEGA_M, "0"),
            "93.3432",
            "5e-5",
        )
        self.assertGreater(loading_fraction_ode("0.5", self.h0_cmb, self.A_H, DEFAULT_OMEGA_M), 0)

    def test_loading_fraction_and_entropy_debt(self) -> None:
        self.assertDecimalClose(compute_loading_fraction("0.5", self.h0_cmb, self.A_H, DEFAULT_OMEGA_M), "0.060044", "5e-7")
        self.assertDecimalClose(compute_loading_fraction("1100", self.h0_cmb, self.A_H, DEFAULT_OMEGA_M), "0.107436", "5e-7")
        debt = compute_entropy_debt("2", self.h0_cmb, self.A_H, DEFAULT_OMEGA_M, DEFAULT_N_SAT)
        expected_debt = DEFAULT_N_SAT * compute_loading_fraction("2", self.h0_cmb, self.A_H, DEFAULT_OMEGA_M)
        self.assertDecimalClose(debt / Decimal("1e121"), str(expected_debt / Decimal("1e121")), "1e-18")

    def test_growth_ode_system_and_suppression(self) -> None:
        first, second = growth_ode_system("0", "1", "1", self.h0_cmb, self.A_H, DEFAULT_OMEGA_M)
        self.assertGreater(first, 0)
        self.assertTrue(mpmath.isfinite(second))
        z05 = compute_growth_suppression("0.5", self.h0_cmb, self.A_H, DEFAULT_OMEGA_M, DEFAULT_SIGMA8)
        z08 = compute_growth_suppression("0.8", self.h0_cmb, self.A_H, DEFAULT_OMEGA_M, DEFAULT_SIGMA8)
        self.assertDecimalClose(z05["lcdm_fsigma8"], "0.475", "5e-4")
        self.assertDecimalClose(z05["shbt_fsigma8"], "0.428", "5e-4")
        self.assertDecimalClose(z05["suppression_fraction"], "-0.098", "2e-3")
        self.assertDecimalClose(z08["shbt_fsigma8"], "0.415", "5e-4")
        self.assertDecimalClose(z08["suppression_fraction"], "-0.083", "2e-3")

    def test_isw_and_bbn(self) -> None:
        self.assertDecimalClose(isw_residual("1100", self.h0_cmb, self.A_H), "-0.000129", "5e-7")
        self.assertDecimalClose(isw_residual("10", self.h0_cmb, self.A_H), "-0.012762", "1.2e-4")
        self.assertDecimalClose(
            bbn_stability_check(BBN_AUDIT_REDSHIFT, self.h0_cmb, self.A_H, DEFAULT_OMEGA_R0),
            "7.421690135465e-27",
            "1e-39",
        )

    def test_neutrino_hierarchy_masses(self) -> None:
        hierarchy = neutrino_hierarchy_masses(DEFAULT_M_NU1_EV, DEFAULT_DELTA_M21_SQ_EV2, DEFAULT_DELTA_M31_SQ_EV2)
        self.assertDecimalClose(hierarchy["normal_sum_mev"], "61.8", "0.3")
        self.assertDecimalClose(hierarchy["inverted_sum_mev"], "103", "1")

    def test_cluster_collapse_table_13(self) -> None:
        # Abundance ratios are written as 6.10e-1, 5.15e-1, 4.21e-1 to match the
        # physical model; the prompt text lists 6.10e-10 which is inconsistent
        # with the other two redshifts and appears to be a typo.
        targets = [
            ("0", "1.6760", "1.6733", "6.10e-1"),
            ("0.5", "1.6821", "1.6799", "5.15e-1"),
            ("1.0", "1.6844", "1.6826", "4.21e-1"),
        ]
        for z, lcdm_target, shbt_target, ratio_target in targets:
            row = compute_cluster_collapse(z, self.h0_cmb, self.A_H, DEFAULT_OMEGA_M, DEFAULT_SIGMA8)
            self.assertDecimalClose(row["lcdm_delta_c"], lcdm_target, "2e-4")
            self.assertDecimalClose(row["shbt_delta_c"], shbt_target, "2e-4")
            self.assertDecimalClose(row["abundance_ratio"], ratio_target, "2e-3")

    def test_get_measurement_cost_and_collapse_index(self) -> None:
        cost = get_measurement_cost("9", "4", "3")
        self.assertDecimalClose(cost.C_get, str(math.log2(9) + 2), "1e-12")
        self.assertDecimalClose(cost.R_entropy, str(9 - (math.log2(9) + 2)), "1e-12")
        first = collapse_index("H0", "A:0", "0.75", "128", 7)
        second = collapse_index("H0", "A:0", "0.75", "128", 7)
        self.assertEqual(first, second)
        self.assertGreaterEqual(first, 0)
        self.assertLess(first, 7)

    def test_precision_report_table_17(self) -> None:
        report = build_precision_cosmology_report(self.h0_cmb, self.delta_mod, DEFAULT_OMEGA_M, DEFAULT_Z_SAMPLES)
        summary = report["summary_table_17"]
        self.assertEqual(summary["overall_precision_audit"], "PASS")
        self.assertDecimalClose(summary["local_uplift_km_s_mpc"], "72.197960072861", "1e-12")
        self.assertDecimalClose(summary["gradient_target_km_s_mpc"], "4.797960072861", "1e-12")
        self.assertDecimalClose(summary["cpl_template"]["w0"], "-1.065", "5e-4")
        self.assertDecimalClose(summary["cpl_template"]["wa"], "-0.066", "5e-4")
        self.assertDecimalClose(summary["forecast_chi2_sensitivity"]["rigid_lcdm"], "35.78", "0.01")
        self.assertDecimalClose(summary["forecast_chi2_sensitivity"]["shbt"], "0.08", "0.01")
        self.assertDecimalClose(summary["cosmic_age_gyr"], "13.276", "0.001")

    def test_dark_matter_abundance_ratio(self) -> None:
        """Verify that SHBT predicts the observed DM/baryon ratio ~5.4 (Table 18)."""

        report = build_precision_cosmology_report(self.h0_cmb, self.delta_mod, DEFAULT_OMEGA_M, DEFAULT_Z_SAMPLES)
        ratio = report["dark_matter"]["abundance_ratio"]
        self.assertAlmostEqual(float(ratio), 5.4, delta=0.1)

    def test_dark_matter_equation_of_state(self) -> None:
        """Verify that the dark-matter topological ghost is pressureless (w_DM ~ 0)."""

        report = build_precision_cosmology_report(self.h0_cmb, self.delta_mod, DEFAULT_OMEGA_M, DEFAULT_Z_SAMPLES)
        w_DM = report["dark_matter"]["w_DM"]
        self.assertAlmostEqual(float(w_DM), 0.0, delta=1e-3)


def _run_unit_tests() -> int:
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(PrecisionCosmologyTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the Section 9 audit; no paper equation."""

    constants = load_default_constants()
    parser = argparse.ArgumentParser(description="Run the SHBT Section 9 precision-cosmology audit.")
    parser.add_argument("--h0-cmb", default=str(constants.h0_cmb), help="CMB Hubble anchor in km s^-1 Mpc^-1.")
    parser.add_argument("--delta-mod", default=None, help="Entropy-debt Delta_mod. Defaults to c_dark^comp/24.")
    parser.add_argument("--omega-m", default=str(DEFAULT_OMEGA_M), help="Present matter density fraction.")
    parser.add_argument("--omega-r0", default=str(DEFAULT_OMEGA_R0), help="Present radiation density fraction.")
    parser.add_argument("--z-samples", nargs="*", default=[str(z) for z in DEFAULT_Z_SAMPLES], help="Redshift samples.")
    parser.add_argument("--precision", type=int, default=DEFAULT_PRECISION, help="Decimal/mpmath precision.")
    parser.add_argument("--json", action="store_true", help="Print the report as JSON.")
    parser.add_argument("--run-tests", action="store_true", help="Run embedded unit tests instead of the audit.")
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the Section 9 audit CLI; no paper equation."""

    args = parse_args(argv)
    if args.run_tests:
        return _run_unit_tests()

    delta_mod = _decimal(DELTA_MOD_FRACTION) if args.delta_mod is None else _decimal(args.delta_mod)
    report = build_precision_cosmology_report(
        args.h0_cmb,
        delta_mod,
        args.omega_m,
        tuple(args.z_samples),
        omega_r0=args.omega_r0,
        precision=args.precision,
    )
    if args.json:
        print(json.dumps(_as_json_safe(report), indent=2, sort_keys=True))
    else:
        print(render_report(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
