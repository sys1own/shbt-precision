"""Run the unified SHBT audit from Python.

This is a thin wrapper around `shbt_simulate.simulate` in audit mode.
"""
import json
import sys
from pathlib import Path

# Make the repo-root `shbt_simulate.py` importable when running from examples/.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import shbt_simulate


def main() -> int:
    result = shbt_simulate.simulate({"mode": "audit"})
    audit = result["audit"]
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
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
