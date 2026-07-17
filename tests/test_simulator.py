"""Python unit tests for the SHBT simulator CLI/API, config parsing, and exports.

These tests require the compiled `shbt_simulator` Rust extension to be available
at `target/release/shbt_simulator.so` (or `target/release/libshbt_simulator.so`).
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
TARGET_RELEASE = REPO_ROOT / "target" / "release"
SO_NAME = TARGET_RELEASE / "shbt_simulator.so"
LIBSO_NAME = TARGET_RELEASE / "libshbt_simulator.so"


def _ensure_extension() -> None:
    """Copy the compiled extension to the import name if needed and add to path."""
    if not SO_NAME.exists() and LIBSO_NAME.exists():
        shutil.copy(LIBSO_NAME, SO_NAME)
    if not SO_NAME.exists():
        pytest.skip("compiled shbt_simulator extension not found; run `cargo build --release`")
    if str(TARGET_RELEASE) not in sys.path:
        sys.path.insert(0, str(TARGET_RELEASE))


@pytest.fixture(scope="module", autouse=True)
def _rust_extension() -> None:
    _ensure_extension()


def _import_simulate():
    import shbt_simulate
    return shbt_simulate


def _run_cli(args: list[str]) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(TARGET_RELEASE)
    return subprocess.run(
        [sys.executable, "shbt_simulate.py", *args],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )


# ---------------------------------------------------------------------------
# API tests
# ---------------------------------------------------------------------------


def test_simulate_default_audit() -> None:
    shbt_simulate = _import_simulate()
    result = shbt_simulate.simulate({"mode": "audit"})
    assert result["config"]["branch"] == (26, 8, 312)
    assert result["config"]["seed"] == 0
    assert "metadata" in result
    assert result["metadata"]["version"] == shbt_simulate.__version__
    audit = result["audit"]
    assert audit["boundary_report"]["framing_defect"] == 0.0
    assert audit["boundary_report"]["modular_invariant"] is True
    assert audit["boundary_report"]["zero_energy_locked"] is True
    assert audit["boundary_report"]["projection_dimension_26_to_4"] is True
    assert audit["stress_energy_preserved"] is True
    assert len(audit["metric_slices"]) == 9
    assert len(audit["history_entries"]) == 9


def test_simulate_cosmology_mode() -> None:
    shbt_simulate = _import_simulate()
    result = shbt_simulate.simulate({"mode": "cosmology", "redshift_samples": 5})
    assert result["config"]["mode"] == "cosmology"
    assert "cosmology" in result
    assert len(result["cosmology"]) == 5
    assert all("tau" in s for s in result["cosmology"])


def test_simulate_baryogenesis_mode() -> None:
    shbt_simulate = _import_simulate()
    result = shbt_simulate.simulate({"mode": "baryogenesis", "particles": 256})
    assert result["config"]["mode"] == "baryogenesis"
    assert result["config"]["particles"] == 256
    # Canonical top-level keys
    assert "baryogenesis_identity" in result
    assert "benchmark_delta" in result
    assert isinstance(result["eta_b"], float)
    assert abs(result["eta_b"] - 6.449923359416e-10) < 1e-20
    assert abs(result["baryogenesis_identity"]["eta_b"] - 6.449923359416e-10) < 1e-20
    assert result["benchmark_delta"]["stress_energy_preserved"] is True
    # Backward-compatible nested alias
    assert result["baryogenesis"]["identity"]["eta_b"] == result["baryogenesis_identity"]["eta_b"]


def test_simulate_history_seed_reproducibility() -> None:
    shbt_simulate = _import_simulate()
    r1 = shbt_simulate.simulate({"mode": "history", "seed": 123, "redshift_samples": 5})
    r2 = shbt_simulate.simulate({"mode": "history", "seed": 123, "redshift_samples": 5})
    c1 = [e["selected_coordinate"] for e in r1["history"]]
    c2 = [e["selected_coordinate"] for e in r2["history"]]
    assert c1 == c2


def test_simulate_history_seed_changes_output() -> None:
    shbt_simulate = _import_simulate()
    r1 = shbt_simulate.simulate({"mode": "history", "seed": 0, "redshift_samples": 5})
    r2 = shbt_simulate.simulate({"mode": "history", "seed": 1, "redshift_samples": 5})
    c1 = [e["selected_coordinate"] for e in r1["history"]]
    c2 = [e["selected_coordinate"] for e in r2["history"]]
    assert c1 != c2


def test_invalid_branch_raises() -> None:
    shbt_simulate = _import_simulate()
    with pytest.raises(ValueError):
        shbt_simulate.simulate({"branch": [1, 2]})


def test_invalid_mode_raises() -> None:
    shbt_simulate = _import_simulate()
    with pytest.raises(ValueError):
        shbt_simulate.simulate({"mode": "unknown"})


# ---------------------------------------------------------------------------
# Config parsing tests
# ---------------------------------------------------------------------------


def test_load_config_file_yaml(tmp_path: Path) -> None:
    shbt_simulate = _import_simulate()
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text("mode: audit\nseed: 7\nparticles: 256\n")
    cfg = shbt_simulate._load_config_file(cfg_path)
    assert cfg["mode"] == "audit"
    assert cfg["seed"] == 7
    assert cfg["particles"] == 256


def test_load_config_file_json(tmp_path: Path) -> None:
    shbt_simulate = _import_simulate()
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text('{"mode": "cosmology", "redshift_samples": 11}')
    cfg = shbt_simulate._load_config_file(cfg_path)
    assert cfg["mode"] == "cosmology"
    assert cfg["redshift_samples"] == 11


def test_cli_overrides_config_file(tmp_path: Path) -> None:
    shbt_simulate = _import_simulate()
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text("mode: audit\nseed: 7\n")
    parser = shbt_simulate._build_arg_parser()
    args = parser.parse_args(["--config", str(cfg_path), "--mode", "baryogenesis"])
    config = shbt_simulate._merge_with_cli(args)
    assert config["mode"] == "baryogenesis"
    assert config["seed"] == 7


def test_invalid_config_file_raises(tmp_path: Path) -> None:
    shbt_simulate = _import_simulate()
    cfg_path = tmp_path / "bad.yaml"
    cfg_path.write_text("mode: notamode\n")
    parser = shbt_simulate._build_arg_parser()
    args = parser.parse_args(["--config", str(cfg_path)])
    with pytest.raises((ValueError, Exception)):
        shbt_simulate._merge_with_cli(args)


# ---------------------------------------------------------------------------
# Export tests
# ---------------------------------------------------------------------------


def test_export_json(tmp_path: Path) -> None:
    shbt_simulate = _import_simulate()
    result = shbt_simulate.simulate({"mode": "cosmology", "redshift_samples": 3})
    out = tmp_path / "out.json"
    paths = shbt_simulate.export_result(result, out, fmt="json")
    assert paths == [out]
    data = json.loads(out.read_text())
    assert "cosmology" in data
    assert len(data["cosmology"]) == 3


def test_export_csv(tmp_path: Path) -> None:
    shbt_simulate = _import_simulate()
    result = shbt_simulate.simulate({"mode": "cosmology", "redshift_samples": 3})
    base = tmp_path / "slices"
    paths = shbt_simulate.export_result(result, base, fmt="csv")
    assert any(p.name.endswith("_metric_slices.csv") for p in paths)
    csv_path = next(p for p in paths if "metric_slices" in p.name)
    text = csv_path.read_text()
    lines = text.strip().splitlines()
    assert len(lines) == 4  # header + 3 slices
    assert "tau" in lines[0] or "eigenvalues" in lines[0]


def _normalize(obj: Any) -> Any:
    if isinstance(obj, tuple):
        return list(obj)
    if isinstance(obj, list):
        return [_normalize(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _normalize(v) for k, v in obj.items()}
    return obj


def test_export_roundtrip_json(tmp_path: Path) -> None:
    shbt_simulate = _import_simulate()
    result = shbt_simulate.simulate({"mode": "baryogenesis"})
    out = tmp_path / "rt.json"
    shbt_simulate.export_result(result, out, fmt="json")
    loaded = json.loads(out.read_text())
    assert _normalize(result["config"]) == loaded["config"]
    assert loaded["baryogenesis"]["identity"]["eta_b"] == pytest.approx(6.449923359416e-10, rel=1e-12)


# ---------------------------------------------------------------------------
# CLI tests
# ---------------------------------------------------------------------------


def test_cli_default_audit(tmp_path: Path) -> None:
    out = tmp_path / "audit.json"
    proc = _run_cli(["--mode", "audit", "--output", str(out)])
    assert proc.returncode == 0, proc.stderr
    assert out.exists()
    data = json.loads(out.read_text())
    assert data["config"]["mode"] == "audit"
    assert data["audit"]["stress_energy_preserved"] is True


def test_cli_cosmology_csv(tmp_path: Path) -> None:
    base = tmp_path / "cosmo"
    proc = _run_cli(["--mode", "cosmology", "--format", "csv", "--output", str(base)])
    assert proc.returncode == 0, proc.stderr
    csv_path = tmp_path / "cosmo_metric_slices.csv"
    assert csv_path.exists()
    text = csv_path.read_text()
    assert text.strip()


def test_cli_baryogenesis(tmp_path: Path) -> None:
    out = tmp_path / "baryo.json"
    proc = _run_cli(["--mode", "baryogenesis", "--particles", "128", "--output", str(out)])
    assert proc.returncode == 0, proc.stderr
    data = json.loads(out.read_text())
    assert data["config"]["particles"] == 128
    assert abs(data["baryogenesis"]["identity"]["eta_b"] - 6.449923359416e-10) < 1e-20


def test_cli_config_file(tmp_path: Path) -> None:
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("mode: audit\nseed: 99\n")
    outdir = tmp_path / "out"
    proc = _run_cli(["--config", str(cfg), "--output-dir", str(outdir)])
    assert proc.returncode == 0, proc.stderr
    dirs = [d for d in outdir.iterdir() if d.is_dir()]
    assert len(dirs) == 1
    assert (dirs[0] / "result.json").exists()
    assert (dirs[0] / "result.log").exists()
    assert (dirs[0] / "result_run_info.json").exists()


def test_cli_seed_reproducibility(tmp_path: Path) -> None:
    out1 = tmp_path / "seed1.json"
    out2 = tmp_path / "seed2.json"
    proc1 = _run_cli(["--mode", "history", "--seed", "123", "--redshift-samples", "5", "--output", str(out1)])
    proc2 = _run_cli(["--mode", "history", "--seed", "123", "--redshift-samples", "5", "--output", str(out2)])
    assert proc1.returncode == 0 and proc2.returncode == 0
    c1 = [e["selected_coordinate"] for e in json.loads(out1.read_text())["history"]]
    c2 = [e["selected_coordinate"] for e in json.loads(out2.read_text())["history"]]
    assert c1 == c2


def test_cli_invalid_branch() -> None:
    proc = _run_cli(["--branch", "1", "2"])
    assert proc.returncode != 0


def test_cli_summary_printed(tmp_path: Path) -> None:
    out = tmp_path / "audit.json"
    proc = _run_cli(["--mode", "audit", "--output", str(out)])
    assert proc.returncode == 0, proc.stderr
    assert "SHBT Audit Summary" in proc.stdout
    assert "eta_b" in proc.stdout


def test_cli_json_logging(tmp_path: Path) -> None:
    out = tmp_path / "audit.json"
    proc = _run_cli(["--mode", "audit", "--log-format", "json", "--output", str(out)])
    assert proc.returncode == 0, proc.stderr
    assert '"event": "audit_complete"' in proc.stdout


def test_cli_quiet(tmp_path: Path) -> None:
    out = tmp_path / "audit.json"
    proc = _run_cli(["--mode", "audit", "--quiet", "--output", str(out)])
    assert proc.returncode == 0, proc.stderr
    assert "Wrote SHBT output" not in proc.stdout
    assert "SHBT Audit Summary" in proc.stdout
