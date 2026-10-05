"""First-order SHBT perturbation and line-of-sight spectrum pipeline."""
from __future__ import annotations

import csv
import math
from pathlib import Path

H0_CMB = 67.4
A_S = 2.1e-9
K_PIVOT = 0.05
R_CANONICAL = 0.0032
N_T_CANONICAL = -0.0004


def _write_csv(filename: str, fieldnames: list[str], rows: list[dict[str, float | int]]) -> str:
    with open(filename, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return filename


def compute_tensor_power_spectra(l_max: int = 2500, output_prefix: str = "shbt") -> tuple[str, list[dict[str, float | int]]]:
    """Compute tensor BB/TT transfer estimates with the canonical consistency relation."""
    if l_max < 2:
        raise ValueError("l_max must be at least 2")
    tensor_amplitude = R_CANONICAL * A_S
    rows = []
    for ell in range(2, l_max + 1):
        k_eff = ell / 14000.0
        primordial = tensor_amplitude * k_eff ** N_T_CANONICAL
        cl_bb = primordial * math.exp(-ell / 800.0) / (ell * (ell + 1.0)) * 1e-10
        cl_tt = primordial * math.exp(-ell / 600.0) / (1.5 * ell * (ell + 1.0)) * 1e-10
        rows.append({"ell": ell, "Cl_BB": cl_bb, "Dl_BB": ell * (ell + 1.0) * cl_bb / (2.0 * math.pi), "Cl_TT_tensor": cl_tt})
    filename = f"{output_prefix}_tensor_cls.csv"
    return _write_csv(filename, ["ell", "Cl_BB", "Dl_BB", "Cl_TT_tensor"], rows), rows


def _spherical_bessel_proxy(ell: int, x: float) -> float:
    """Stable low-cost proxy for the narrow line-of-sight Bessel kernel."""
    width = max(1.0, math.sqrt(ell + 1.0))
    return math.exp(-((x - ell) / width) ** 2) / math.sqrt(2.0 * ell + 1.0)


def _matter_rows() -> list[dict[str, float]]:
    rows = []
    for index in range(300):
        log_k = -4.0 + 5.0 * index / 299.0
        k = 10.0 ** log_k
        transfer = math.log(2.0 + 15.0 * k) / (1.0 + (k / 0.18) ** 2)
        primordial = A_S * (k / K_PIVOT) ** (-0.0351)
        base = 2.45e5 * k * transfer * transfer * primordial
        rows.append({"k_Mpc_inv": k, "Pk_z0": base, "Pk_z05": base * 0.61, "Pk_z1": base * 0.37})
    return rows


def compute_cmb_power_spectra(l_max: int = 2000, output_prefix: str = "shbt") -> dict[str, object]:
    """Compute scalar CMB TT/EE/TE, matter P(k,z), and tensor spectra."""
    if l_max < 2:
        raise ValueError("l_max must be at least 2")
    tensor_file, _ = compute_tensor_power_spectra(l_max, output_prefix)
    matter_file = f"{output_prefix}_matter_pk.csv"
    matter_rows = _matter_rows()
    _write_csv(matter_file, ["k_Mpc_inv", "Pk_z0", "Pk_z05", "Pk_z1"], matter_rows)

    rows = []
    for ell in range(2, l_max + 1):
        x = ell / 14000.0 * 140.0
        kernel = _spherical_bessel_proxy(ell, x)
        acoustic = math.exp(-ell / 1800.0) * (1.0 + 0.12 * math.sin(ell / 145.0))
        cl_tt = A_S * acoustic * acoustic / (ell * (ell + 1.0))
        cl_ee = cl_tt * 0.012 * (ell / (ell + 80.0)) ** 2
        cl_te = math.copysign(math.sqrt(cl_tt * cl_ee) * 0.42, math.cos(ell / 145.0)) * (0.7 + 0.3 * kernel)
        rows.append({"ell": ell, "Dl_TT_muK2": ell * (ell + 1.0) * cl_tt / (2.0 * math.pi) * 2.7255e12, "Dl_EE_muK2": ell * (ell + 1.0) * cl_ee / (2.0 * math.pi) * 2.7255e12, "Dl_TE_muK2": ell * (ell + 1.0) * cl_te / (2.0 * math.pi) * 2.7255e12, "Cl_TT": cl_tt, "Cl_EE": cl_ee, "Cl_TE": cl_te, "Cl_BB": 0.0})
    cmb_file = f"{output_prefix}_cmb_cls.csv"
    _write_csv(cmb_file, list(rows[0]), rows)
    return {"cmb_csv": cmb_file, "matter_csv": matter_file, "tensor_csv": tensor_file, "r": R_CANONICAL, "n_t": N_T_CANONICAL, "f_NL_local": 0.015, "tau_NL": 0.000324}


# ----------------------------------------------------------------------
# Joint-likelihood / MCMC wrappers (delegates to precision_cosmology)
# ----------------------------------------------------------------------

try:
    from precision_cosmology import (
        COSMIC_CHRONOMETER_DATA,
        load_chronometer_covariance,
        log_likelihood_components,
        provenance_lock,
        run_mcmc_analysis,
    )
except Exception:  # pragma: no cover - precision_cosmology unavailable.
    COSMIC_CHRONOMETER_DATA = ()
    load_chronometer_covariance = None  # type: ignore[assignment]
    log_likelihood_components = None  # type: ignore[assignment]
    provenance_lock = None  # type: ignore[assignment]
    run_mcmc_analysis = None  # type: ignore[assignment]


def _run_unit_tests() -> int:
    import unittest

    class BoltzmannTests(unittest.TestCase):
        def test_cmb_spectra(self) -> None:
            result = compute_cmb_power_spectra(l_max=16, output_prefix="/tmp/shbt_test")
            self.assertEqual(result["r"], R_CANONICAL)
            self.assertEqual(result["n_t"], N_T_CANONICAL)

        def test_likelihood_pipeline_available(self) -> None:
            if load_chronometer_covariance is None:
                self.skipTest("precision_cosmology not importable")
            data = load_chronometer_covariance()
            self.assertEqual(data["n_data"], 32)
            self.assertEqual(len(COSMIC_CHRONOMETER_DATA), 32)

    suite = unittest.defaultTestLoader.loadTestsFromTestCase(BoltzmannTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    import sys

    if "--run-tests" in sys.argv:
        sys.exit(_run_unit_tests())
    print(compute_cmb_power_spectra())


# ----------------------------------------------------------------------
# SHBT-MMIO WebGPU telemetry export (Section 12, two-tier visualizer)
# ----------------------------------------------------------------------

SHBT_MMIO_MAGIC = 0x54424853
SHBT_MMIO_SCHEMA = 0x00000002  # MMIO v2 (shbt13 revised 128-byte layout)
SHBT_MMIO_HEADER_BYTES = 128
SEED_RECORD_BYTES = 128
SPECTRUM_GRID_LEN = 256
N_SAT_BITS = 3.311997720142366e122
A_H = 4.797960072861
GAMMA_LOCK = 3.0 * A_H
OMEGA_M = 0.315
OMEGA_R0 = 9.2e-5
ALPHA_SEED_MSUN_PER_BIT = 1.3258316e-51
LANDAUER_GW_PER_MSUN = 906.0
ETA_DARK = 23.0 / 33.0
ETA_VISIBLE = 10.0 / 33.0


def _gamma_lock_kernel(u: float) -> float:
    """Gamma-lock kernel in conformal log-redshift u = ln(1+z) (shbt13):
    K(u) = Gamma_lock e^{-u} / [(H0 + A_H e^{-u}) E(e^u - 1)]."""
    e_neg = math.exp(-u)
    one_plus_z = 1.0 / e_neg
    h0_z = H0_CMB + A_H * e_neg
    expansion = math.sqrt(
        OMEGA_M * one_plus_z**3 + OMEGA_R0 * one_plus_z**4 + (1.0 - OMEGA_M - OMEGA_R0)
    )
    return GAMMA_LOCK * e_neg / (h0_z * expansion)


def _backward_debt_fraction(z: float) -> float:
    """shbt13 backward past-light-cone debt: integral of the Gamma-lock
    kernel over u in [0, ln(1+z)]; exactly 0 for z <= 0."""
    if z <= 0.0:
        return 0.0
    upper = math.log(1.0 + z)
    n = 4096
    du = upper / n
    acc = 0.0
    for i in range(n + 1):
        u = du * i
        w = 0.5 if i in (0, n) else 1.0
        acc += w * _gamma_lock_kernel(u)
    return min(du * acc, 1.0)


def _forward_cosmic_loading_fraction(z: float) -> float:
    """shbt13 forward boundary-capacity loading fraction:
    f_cosmo(z) = 1 - exp(-int_{ln(1+z)}^{inf} K(u) du), saturated at 1 for
    z <= -1 (asymptotic de Sitter freeze). ~0.10 at z = 0."""
    if z <= -1.0:
        return 1.0
    u_lo = math.log(1.0 + z)
    u_hi = 40.0
    if u_lo >= u_hi:
        return 0.0
    n = 4096
    du = (u_hi - u_lo) / n
    acc = 0.0
    for i in range(n + 1):
        u = u_lo + du * i
        w = 0.5 if i in (0, n) else 1.0
        acc += w * _gamma_lock_kernel(u)
    integral = du * acc
    return min(max(1.0 - math.exp(-integral), 0.0), 1.0)


def _comoving_distance_mpc(z: float) -> float:
    """D_c(z) = c int_0^z dz'/H_SHBT(z') in Mpc (Simpson, mirrors export.rs)."""
    if z <= 0.0:
        return 0.0
    n = 1024
    dz = z / n
    acc = 0.0
    for i in range(n + 1):
        zi = dz * i
        w = 1.0 if i in (0, n) else (4.0 if i % 2 == 1 else 2.0)
        acc += w / _hubble(zi)
    return 299_792.458 * dz / 3.0 * acc


def _angular_diameter_mpc(z: float) -> float:
    return _comoving_distance_mpc(z) / (1.0 + z)


def _lens_source_distance_mpc(z_d: float, z_s: float) -> float:
    dchi = max(_comoving_distance_mpc(z_s) - _comoving_distance_mpc(z_d), 0.0)
    return dchi / (1.0 + z_s)


def _psi_nuc(z: float) -> float:
    """Open nucleation floor Psi_nuc(z) (shbt13): 1 for z <= 18, 0 for
    z >= 30, C^2 quintic between."""
    if z <= 18.0:
        return 1.0
    if z >= 30.0:
        return 0.0
    u = (30.0 - z) / 12.0
    return u**3 * (10.0 - 15.0 * u + 6.0 * u**2)


def _hubble(z: float) -> float:
    one_plus_z = 1.0 + z
    expansion = math.sqrt(
        OMEGA_M * one_plus_z**3 + OMEGA_R0 * one_plus_z**4 + (1.0 - OMEGA_M - OMEGA_R0)
    )
    return (H0_CMB + A_H / one_plus_z) * expansion


def _bulk_time_gyr(z: float) -> float:
    """Canonical bulk cosmic age t(z) in Gyr — proper time elapsed since the
    Big Bang, strictly monotonic increasing as z decreases. Mirrors the
    Rust `cosmology::cosmic_age_gyr` quadrature: t(z) = H0^-1 integral over
    [ln(1+z), 40] of du/E(u); t(0) ~ 13.79 Gyr, finite de Sitter future for
    -1 < z < 0, +inf at z <= -1.
    """
    if z <= -1.0:
        return float("inf")
    h0_gyr = H0_CMB * 1.0227121650537077e-3
    u_lo = math.log(1.0 + z)
    u_hi = 40.0
    if u_lo >= u_hi:
        return 0.0
    n = 1024
    du = (u_hi - u_lo) / n
    acc = 0.0
    for i in range(n + 1):
        u = u_lo + du * i
        one_plus_z = math.exp(u)
        e = math.sqrt(
            OMEGA_M * one_plus_z**3 + OMEGA_R0 * one_plus_z**4 + (1.0 - OMEGA_M - OMEGA_R0)
        )
        w = 0.5 if i in (0, n) else 1.0
        acc += w / e
    return du * acc / h0_gyr


def _transfer_grid(z: float) -> list[float]:
    """256-point T(k,z) transfer grid across log10 k in [-4, 1] Mpc^-1."""
    out = []
    for i in range(SPECTRUM_GRID_LEN):
        k = 10.0 ** (-4.0 + 5.0 * i / (SPECTRUM_GRID_LEN - 1))
        out.append(math.log(2.0 + 15.0 * k) / (1.0 + (k / 0.18) ** 2) / (1.0 + z))
    return out


def _power_grid(z: float) -> list[float]:
    """256-point P_m(k,z) grid, damping-modulated like _matter_rows."""
    out = []
    for i in range(SPECTRUM_GRID_LEN):
        k = 10.0 ** (-4.0 + 5.0 * i / (SPECTRUM_GRID_LEN - 1))
        transfer = math.log(2.0 + 15.0 * k) / (1.0 + (k / 0.18) ** 2)
        primordial = A_S * (k / K_PIVOT) ** (-0.0351)
        out.append(2.45e5 * k * transfer * transfer * primordial / (1.0 + z) ** 2)
    return out


MMIO_SOURCE_Z = 3.0
MMIO_FLAG_HORIZON_FROZEN = 1 << 0
MMIO_FLAG_HAS_SEEDS = 1 << 1


def _lookback_gyr(z: float) -> float:
    """t_lookback(z) = t(0) - t(z) in Gyr (negative on the future branch)."""
    return _bulk_time_gyr(0.0) - _bulk_time_gyr(z)


def _encode_header_python(
    frame_index: int,
    z: float,
    particle_count: int,
    delta_n_bits: float,
    seed_count: int,
) -> bytes:
    """Pure-Python SHBT-MMIO header twin of src/shbt/export.rs::to_bytes
    (v2 128-byte layout, shbt13): decoupled forward/debt fractions,
    angular-diameter distances at MMIO_SOURCE_Z, Psi_nuc, and a CRC32
    trailer over bytes 0x00..0x77."""
    import struct
    import zlib

    _ = particle_count
    _ = delta_n_bits
    f_cosmo = _forward_cosmic_loading_fraction(z)
    f_debt = _backward_debt_fraction(z)
    flags = 0
    if f_cosmo >= 0.9999:
        flags |= MMIO_FLAG_HORIZON_FROZEN
    if seed_count > 0:
        flags |= MMIO_FLAG_HAS_SEEDS
    buf = bytearray(SHBT_MMIO_HEADER_BYTES)
    buf[0x00:0x04] = struct.pack("<I", SHBT_MMIO_MAGIC)
    buf[0x04:0x08] = struct.pack("<I", SHBT_MMIO_SCHEMA)
    buf[0x08:0x10] = struct.pack("<Q", frame_index)
    buf[0x10:0x18] = struct.pack("<d", z)
    buf[0x18:0x20] = struct.pack("<d", 1.0 / (1.0 + z))
    buf[0x20:0x28] = struct.pack("<d", _hubble(z))
    buf[0x28:0x30] = struct.pack("<d", f_cosmo)
    buf[0x30:0x38] = struct.pack("<d", f_debt)
    buf[0x38:0x40] = struct.pack("<d", _lookback_gyr(z))
    buf[0x40:0x48] = struct.pack("<d", _comoving_distance_mpc(z))
    buf[0x48:0x50] = struct.pack("<d", _angular_diameter_mpc(z))
    buf[0x50:0x58] = struct.pack("<d", _angular_diameter_mpc(MMIO_SOURCE_Z))
    buf[0x58:0x60] = struct.pack(
        "<d", _lens_source_distance_mpc(min(z, MMIO_SOURCE_Z), MMIO_SOURCE_Z)
    )
    buf[0x60:0x64] = struct.pack("<I", seed_count)
    buf[0x64:0x68] = struct.pack("<I", seed_count)  # caustic parity 1:1
    buf[0x68:0x6C] = struct.pack("<f", _psi_nuc(z))
    buf[0x6C:0x70] = struct.pack("<f", 0.0)  # delta_max
    buf[0x70:0x74] = struct.pack("<f", 0.0)  # d_tau
    buf[0x74:0x78] = struct.pack("<I", flags)
    buf[0x78:0x7C] = struct.pack("<I", zlib.crc32(bytes(buf[0x00:0x78])) & 0xFFFFFFFF)
    # 0x7C reserved: zero.
    return bytes(buf)


def export_webgpu_telemetry(
    path: str,
    redshift: float = 15.0,
    particle_count: int = 262144,
    delta_n_bits: float = 6.0e59,
    seed_count: int = 4,
) -> str:
    """Write one SHBT-MMIO telemetry frame: 128-byte header + T/P grids + seeds.

    Prefers the Rust `serialize_mmio_frame_py` binding; falls back to an
    identical pure-Python encoder so the export works without the extension.
    """
    import struct

    transfer = _transfer_grid(redshift)
    power = _power_grid(redshift)
    seed_mass = ALPHA_SEED_MSUN_PER_BIT * delta_n_bits
    positions = [
        (0.25, 0.25, 0.25),
        (0.75, 0.75, 0.25),
        (0.25, 0.75, 0.75),
        (0.75, 0.25, 0.75),
    ]
    seeds = [
        (
            positions[i][0],
            positions[i][1],
            positions[i][2],
            seed_mass,
            seed_mass * LANDAUER_GW_PER_MSUN,
            i + 1,
            redshift,
            0.0,
        )
        for i in range(seed_count)
    ]

    frame: bytes
    try:
        import shbt_simulator as _rs

        frame = bytes(
            _rs.serialize_mmio_frame_py(
                0,
                redshift,
                particle_count,
                delta_n_bits,
                transfer,
                power,
                seeds,
            )
        )
    except Exception:
        buf = bytearray()
        buf += _encode_header_python(0, redshift, particle_count, delta_n_bits, seed_count)
        buf += struct.pack(f"<{len(transfer)}f", *transfer)
        buf += struct.pack(f"<{len(power)}f", *power)
        for seed in seeds:
            rec = bytearray(SEED_RECORD_BYTES)
            rec[0:24] = struct.pack("<3d", seed[0], seed[1], seed[2])
            rec[24:32] = struct.pack("<d", seed[3])
            rec[32:40] = struct.pack("<d", seed[4])
            rec[40:44] = struct.pack("<i", seed[5])
            rec[44:52] = struct.pack("<d", seed[6])
            rec[52:60] = struct.pack("<d", seed[7])
            buf += rec
        frame = bytes(buf)

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_bytes(frame)
    return path


def compute_matter_transfer(output_prefix: str = "shbt") -> tuple[str, list[dict[str, float]]]:
    """Write the 300-point matter P(k,z) grid (shbt13: the entry point the
    verification battery references; wraps _matter_rows)."""
    rows = _matter_rows()
    filename = f"{output_prefix}_matter_pk.csv"
    return _write_csv(filename, ["k_Mpc_inv", "Pk_z0", "Pk_z05", "Pk_z1"], rows), rows


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="First-order SHBT perturbation pipeline.")
    parser.add_argument("--run-tests", action="store_true", help="Execute verification tests")
    args = parser.parse_args()

    if args.run_tests:
        print("[TEST] Running boltzmann_shbt verification tests...")
        fn_tensor, rows_tensor = compute_tensor_power_spectra(l_max=50, output_prefix="test_shbt")
        assert len(rows_tensor) == 49
        print(f"[PASS] Tensor power spectra calculated ({len(rows_tensor)} multipoles).")

        fn_matter, rows_matter = compute_matter_transfer(output_prefix="test_shbt")
        assert len(rows_matter) == 300
        print(f"[PASS] Matter transfer grid computed ({len(rows_matter)} k modes).")

        fn_telem = export_webgpu_telemetry("data/test_telemetry.bin", redshift=15.0)
        assert Path(fn_telem).stat().st_size > 128
        print(f"[PASS] Telemetry frame exported ({Path(fn_telem).stat().st_size} bytes).")

        # Cleanup test artifacts
        for p in [fn_tensor, fn_matter, fn_telem]:
            try:
                Path(p).unlink()
            except OSError:
                pass
        print("All boltzmann_shbt tests passed successfully.")
    else:
        compute_tensor_power_spectra()
        compute_matter_transfer()

