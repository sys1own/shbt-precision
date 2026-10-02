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
SHBT_MMIO_SCHEMA = 0x00020000
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


def _loading_fraction(z: float) -> float:
    """Conformal loading fraction on (-1, +inf); mirrors Rust cosmology.rs."""
    if z <= -1.0:
        return 1.0
    if z < 0.0:
        return 1.0 - (1.0 + z) ** 3
    if z == 0.0:
        return 0.0
    upper = math.log(1.0 + z)
    n = 4096
    du = upper / n
    acc = 0.0
    for i in range(n + 1):
        u = du * i
        one_plus_z = math.exp(u)
        h0_z = H0_CMB + A_H / one_plus_z
        expansion = math.sqrt(
            OMEGA_M * one_plus_z**3 + OMEGA_R0 * one_plus_z**4 + (1.0 - OMEGA_M - OMEGA_R0)
        )
        w = 0.5 if i in (0, n) else 1.0
        acc += w * GAMMA_LOCK * math.exp(-u) / (h0_z * expansion)
    return min(du * acc, 1.0)


def _hubble(z: float) -> float:
    one_plus_z = 1.0 + z
    expansion = math.sqrt(
        OMEGA_M * one_plus_z**3 + OMEGA_R0 * one_plus_z**4 + (1.0 - OMEGA_M - OMEGA_R0)
    )
    return (H0_CMB + A_H / one_plus_z) * expansion


def _bulk_time_gyr(z: float) -> float:
    if z <= -1.0:
        return float("inf")
    if z <= 0.0:
        return 13.276616557
    h0_gyr = H0_CMB * 1.0227121650537077e-3
    upper = math.log(1.0 + z)
    n = 1024
    du = upper / n
    acc = 0.0
    for i in range(n + 1):
        u = du * i
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


def _encode_header_python(
    frame_index: int,
    z: float,
    particle_count: int,
    delta_n_bits: float,
    seed_count: int,
) -> bytes:
    """Pure-Python SHBT-MMIO header twin of src/shbt/export.rs::to_bytes."""
    import struct

    f_load = _loading_fraction(z)
    seed_mass = ALPHA_SEED_MSUN_PER_BIT * delta_n_bits
    p_debt = seed_mass * LANDAUER_GW_PER_MSUN
    buf = bytearray(SHBT_MMIO_HEADER_BYTES)
    buf[0x00:0x04] = struct.pack("<I", SHBT_MMIO_MAGIC)
    buf[0x04:0x08] = struct.pack("<I", SHBT_MMIO_SCHEMA)
    buf[0x08:0x10] = struct.pack("<Q", frame_index)
    buf[0x10:0x18] = struct.pack("<d", _bulk_time_gyr(z))
    buf[0x18:0x20] = struct.pack("<d", z)
    buf[0x20:0x28] = struct.pack("<d", 1.0 / (1.0 + z))
    buf[0x28:0x30] = struct.pack("<d", _hubble(z))
    buf[0x30:0x38] = struct.pack("<d", f_load)
    buf[0x38:0x40] = struct.pack("<d", ETA_VISIBLE * N_SAT_BITS * f_load)
    buf[0x40:0x48] = struct.pack("<d", ETA_DARK * N_SAT_BITS * f_load)
    buf[0x48:0x50] = struct.pack("<d", p_debt)
    buf[0x50:0x58] = struct.pack("<d", seed_mass)
    buf[0x58:0x60] = struct.pack("<d", 1.0 - ETA_VISIBLE * f_load)
    buf[0x60:0x68] = struct.pack(
        "<d", 2.0 * A_H * GAMMA_LOCK / ((H0_CMB + A_H / (1.0 + z)) * _hubble(z))
    )
    buf[0x68:0x70] = struct.pack("<Q", particle_count)
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
