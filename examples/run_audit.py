"""Run the unified SHBT audit from Python.

This is a thin wrapper around `shbt_simulate.simulate` in audit mode.  It also
exercises the new supplementary-material verification functions and writes
simulation-generated LaTeX macros to `sim_results.tex`.
"""
import json
import sys
from pathlib import Path

# Make the repo-root `shbt_simulate.py` and compiled extension importable
# when running from examples/.
ROOT = Path(__file__).resolve().parent.parent
TARGET = ROOT / "target" / "release"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(TARGET) not in sys.path:
    sys.path.insert(0, str(TARGET))

import shbt_simulate
import shbt_simulator


def _to_float(num: int, den: int, decimals: int = 12) -> str:
    """Return a fixed-point decimal string for the exact fraction."""
    return f"{num / den:.{decimals}f}"


def _write_sim_results(path: Path, macros: dict[str, str]) -> None:
    """Write LaTeX \providecommand macros to a .tex file."""
    lines = ["% Simulation-generated macros for SHBT supplementary material\n"]
    for name, value in macros.items():
        lines.append(f"\\providecommand{{\\{name}}}{{{value}}}\n")
    path.write_text("".join(lines), encoding="utf-8")


def main() -> int:
    result = shbt_simulate.simulate({"mode": "audit"})
    audit = result["audit"]

    # ------------------------------------------------------------------
    # Boundary quantum dimension checks
    # ------------------------------------------------------------------
    d_su2_1 = shbt_simulator.get_su2_quantum_dimension(1, 26)
    d_su3_10 = shbt_simulator.get_su3_quantum_dimension(1, 0, 8)
    d_su3_11 = shbt_simulator.get_su3_quantum_dimension(1, 1, 8)

    assert abs(d_su2_1 - 1.98742442) < 1e-7, f"SU(2) d_1 = {d_su2_1}"
    assert abs(d_su3_10 - 2.68250707) < 1e-7, f"SU(3) d_(1,0) = {d_su3_10}"
    assert abs(d_su3_11 - 6.19584416) < 1e-7, f"SU(3) d_(1,1) = {d_su3_11}"

    # ------------------------------------------------------------------
    # Exact central-charge ledger checks
    # ------------------------------------------------------------------
    ledger = shbt_simulator.get_exact_ledger_dict()
    fractions = {
        "c_vis": (1325, 154),
        "c_dark_res": (834433, 362670),
        "c_dark_comp": (1197103, 362670),
        "c_tot_res": (179764, 16485),
        "c_tot_comp": (196249, 16485),
    }
    for key, (expected_num, expected_den) in fractions.items():
        entry = ledger[key]
        assert entry["numerator"] == expected_num, f"{key} numerator"
        assert entry["denominator"] == expected_den, f"{key} denominator"

    assert (
        ledger["c_dark_comp"]["numerator"] - ledger["c_dark_res"]["numerator"]
        == ledger["c_dark_comp"]["denominator"]
    ), "unit dark-ledger shift"

    # ------------------------------------------------------------------
    # Framing-defect check on the canonical branch
    # ------------------------------------------------------------------
    framing_defect = shbt_simulator.get_framing_defect(26, 8, 312)
    assert framing_defect == 0.0, f"framing defect = {framing_defect}"

    # ------------------------------------------------------------------
    # Denominator prime-factorization check
    # ------------------------------------------------------------------
    factorization = shbt_simulator.get_denominator_prime_factorization()
    assert factorization["valid"] is True
    factors_362670 = {int(k): v for k, v in factorization["362670"].items()}
    assert factors_362670 == {2: 1, 3: 1, 5: 1, 7: 1, 11: 1, 157: 1}
    prime_conductor = max(factors_362670.keys())
    assert prime_conductor == 157

    # ------------------------------------------------------------------
    # Export LaTeX macros for the main document
    # ------------------------------------------------------------------
    macros = {
        "SimOutputQuantumDimSUTwoPrimaryOne": f"{d_su2_1:.8f}",
        "SimOutputQuantumDimSUThreeFundamental": f"{d_su3_10:.8f}",
        "SimOutputQuantumDimSUThreeAdjoint": f"{d_su3_11:.8f}",
        "SimOutputCentralChargeVis": _to_float(
            ledger["c_vis"]["numerator"], ledger["c_vis"]["denominator"], 12
        ),
        "SimOutputCentralChargeDarkRes": _to_float(
            ledger["c_dark_res"]["numerator"], ledger["c_dark_res"]["denominator"], 12
        ),
        "SimOutputCentralChargeDarkComp": _to_float(
            ledger["c_dark_comp"]["numerator"], ledger["c_dark_comp"]["denominator"], 12
        ),
        "SimOutputCentralChargeTotRes": _to_float(
            ledger["c_tot_res"]["numerator"], ledger["c_tot_res"]["denominator"], 12
        ),
        "SimOutputCentralChargeTotComp": _to_float(
            ledger["c_tot_comp"]["numerator"], ledger["c_tot_comp"]["denominator"], 12
        ),
        "SimOutputDenominatorPrimeConductor": str(prime_conductor),
    }
    _write_sim_results(ROOT / "sim_results.tex", macros)

    summary = {
        "branch": audit["branch"],
        "framing_defect (delta_fr)": audit["boundary_report"]["framing_defect"],
        "modular_invariant": audit["boundary_report"]["modular_invariant"],
        "zero_energy_locked": audit["boundary_report"]["zero_energy_locked"],
        "projection_dimension_26_to_4": audit["boundary_report"]["projection_dimension_26_to_4"],
        "eta_b": audit["eta_b"],
        "stress_energy_preserved": audit["stress_energy_preserved"],
        "projection_all_passed": audit["projection_report"]["all_passed"],
        "memory_all_passed": audit["memory_report"]["all_passed"],
        "metric_slices": len(audit["metric_slices"]),
        "history_entries": len(audit["history_entries"]),
        "quantum_dimensions": {
            "su2_d1": round(d_su2_1, 8),
            "su3_d10": round(d_su3_10, 8),
            "su3_d11": round(d_su3_11, 8),
        },
        "central_charge_ledger": ledger,
        "framing_defect_supplementary": framing_defect,
        "denominator_prime_conductor": prime_conductor,
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
