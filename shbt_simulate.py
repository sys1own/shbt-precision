"""SHBT research simulation CLI and programmable API.

This module wraps the compiled `shbt_simulator` Rust extension and adds
data export (JSON/CSV/HDF5), optional plotting, parameter sweeps, and
configuration-file support (YAML or JSON).

Example CLI::

    python shbt_simulate.py --config config.default.yaml
    python shbt_simulate.py --mode all --output result.json
    python shbt_simulate.py --mode cosmology --format csv --output slices
    python shbt_simulate.py --sweep sweep.json --output sweep.json

Programmatic usage::

    import shbt_simulate
    result = shbt_simulate.simulate({"mode": "all"})
    shbt_simulate.export_result(result, "result.json", fmt="json")
"""
from __future__ import annotations

import argparse
import csv
import datetime
import itertools
import json
import logging
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable

try:
    import shbt_simulator as _rs  # type: ignore[import]
    _HAS_RUST = True
except Exception as exc:  # pragma: no cover
    _rs = None  # type: ignore[assignment]
    _HAS_RUST = False
    _IMPORT_ERROR = exc

try:
    import matplotlib  # type: ignore[import]
    matplotlib.use("Agg")
    import matplotlib.pyplot as _plt  # type: ignore[import]
    _HAS_MPL = True
except Exception:  # pragma: no cover
    _HAS_MPL = False

try:
    import h5py  # type: ignore[import]
    _HAS_H5 = True
except Exception:  # pragma: no cover
    _HAS_H5 = False

try:
    import yaml  # type: ignore[import]
    _HAS_YAML = True
except Exception:  # pragma: no cover
    _HAS_YAML = False

try:
    import jsonschema  # type: ignore[import]
    _HAS_JSONSCHEMA = True
except Exception:  # pragma: no cover
    _HAS_JSONSCHEMA = False


def _ensure_rust() -> None:
    if not _HAS_RUST:
        raise RuntimeError(
            "The compiled `shbt_simulator` Rust extension is not available. "
            "Build it first with `cargo build --release` and make sure the shared "
            "object is on your PYTHONPATH (e.g. copy target/release/libshbt_simulator.so "
            "to target/release/shbt_simulator.so)."
        ) from _IMPORT_ERROR


_LOGGER = logging.getLogger("shbt")


# Standard logging attributes we do not want to leak into JSON event payloads.
_STD_LOG_ATTRS = {
    "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
    "module", "lineno", "funcName", "created", "msecs", "relativeCreated",
    "thread", "threadName", "process", "processName", "exc_info", "exc_text",
    "stack_info", "message", "asctime", "event",
}


class _JsonFormatter(logging.Formatter):
    """Emit log records as JSON lines.

    When a record carries an ``event`` extra, the event name and any extra
    keyword fields are serialised directly (e.g. ``{"event": "audit_complete",
    "eta_b": 6.45e-10}``). Otherwise a plain ``message`` field is emitted.
    """

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
        }
        event = getattr(record, "event", None)
        if event:
            payload["event"] = event
            for key, value in record.__dict__.items():
                if key in _STD_LOG_ATTRS or key.startswith("_") or callable(value):
                    continue
                payload[key] = value
        else:
            payload["message"] = record.getMessage()
        return json.dumps(payload, default=str)


def _setup_logging(
    level_name: str = "INFO",
    fmt: str = "text",
    log_file: str | Path | None = None,
    quiet: bool = False,
) -> None:
    """Configure the SHBT logger."""
    level = getattr(logging, level_name.upper(), logging.INFO)
    _LOGGER.setLevel(level)
    _LOGGER.handlers.clear()
    _LOGGER.propagate = False

    formatter: logging.Formatter
    if fmt == "json":
        formatter = _JsonFormatter()
    else:
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    if log_file:
        fh = logging.FileHandler(log_file, mode="a")
        fh.setLevel(level)
        fh.setFormatter(formatter)
        _LOGGER.addHandler(fh)

    if not quiet:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(level)
        ch.setFormatter(formatter)
        _LOGGER.addHandler(ch)
    else:
        # Even in quiet mode, route errors to stderr.
        ch = logging.StreamHandler(sys.stderr)
        ch.setLevel(max(level, logging.ERROR))
        ch.setFormatter(formatter)
        _LOGGER.addHandler(ch)


def _log_event(event: str, **kwargs: Any) -> None:
    """Emit a structured event through the logging system."""
    text = ", ".join(f"{k}={v}" for k, v in kwargs.items())
    message = f"{event}: {text}" if text else event
    _LOGGER.info("%s", message, extra={"event": event, **kwargs})


def get_version() -> str:
    """Return the simulator version from Cargo.toml, or 'unknown'."""
    cargo_toml = Path(__file__).with_name("Cargo.toml")
    if not cargo_toml.exists():
        return "unknown"
    text = cargo_toml.read_text(encoding="utf-8")
    m = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
    return m.group(1) if m else "unknown"


def get_git_info() -> dict[str, str] | None:
    """Return git commit hash and branch if available."""
    repo_root = Path(__file__).resolve().parent
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        branch = subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "--abbrev-ref", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return {"commit": commit, "branch": branch}
    except Exception:
        return None


__version__ = get_version()


# ---------------------------------------------------------------------------
# Configuration schema and defaults
# ---------------------------------------------------------------------------

DEFAULT_CONFIG: dict[str, Any] = {
    "mode": "all",
    "branch": (26, 8, 312),
    "observer_radius_fraction": 0.125,
    "redshift_max": 3.0,
    "redshift_samples": 9,
    "particles": 512,
    "seed": 0,
    "output_dir": "./simulation_results",
    "output_name": "result",
    "export_formats": ["json"],
    "plot": False,
    "verbose": False,
    "log_level": "INFO",
    "log_format": "text",
    "log_file": None,
    "quiet": False,
}

SHBT_CONFIG_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "mode": {"type": "string", "enum": ["audit", "cosmology", "baryogenesis", "history", "all"]},
        "branch": {
            "type": "array",
            "minItems": 3,
            "maxItems": 3,
            "items": {"type": "integer"},
        },
        "observer_radius_fraction": {"type": "number"},
        "redshift_max": {"type": "number"},
        "redshift_samples": {"type": "integer"},
        "particles": {"type": "integer"},
        "seed": {"type": "integer"},
        "output_dir": {"type": "string"},
        "output_name": {"type": "string"},
        "export_formats": {
            "type": "array",
            "items": {"type": "string", "enum": ["json", "csv", "hdf5", "h5"]},
        },
        "plot": {"type": "boolean"},
        "verbose": {"type": "boolean"},
        "log_level": {"type": "string", "enum": ["DEBUG", "INFO", "WARNING", "ERROR"]},
        "log_format": {"type": "string", "enum": ["text", "json"]},
        "log_file": {"type": ["string", "null"]},
        "quiet": {"type": "boolean"},
    },
}


def _load_config_file(path: str | Path) -> dict[str, Any]:
    """Load a YAML or JSON configuration file."""
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()
    if suffix in (".yaml", ".yml"):
        if not _HAS_YAML:
            raise RuntimeError(
                "YAML config files require PyYAML. Install it with `pip install pyyaml` "
                "or use a JSON config file instead."
            )
        data = yaml.safe_load(text)
    elif suffix == ".json":
        data = json.loads(text)
    else:
        # Try YAML first, then JSON
        if _HAS_YAML:
            try:
                data = yaml.safe_load(text)
            except Exception:
                data = json.loads(text)
        else:
            data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError(f"configuration file must contain a top-level object, got {type(data)}")
    return data


def _validate_config(config: dict[str, Any]) -> None:
    """Validate a loaded configuration dictionary."""
    if _HAS_JSONSCHEMA:
        jsonschema.validate(config, SHBT_CONFIG_SCHEMA)
    else:
        allowed = set(SHBT_CONFIG_SCHEMA["properties"].keys())
        for key in config:
            if key not in allowed:
                raise ValueError(f"unknown configuration key: {key!r}")
        if "mode" in config and config["mode"] not in ("audit", "cosmology", "baryogenesis", "history", "all"):
            raise ValueError(f"invalid mode: {config['mode']!r}")
        if "branch" in config:
            b = config["branch"]
            if not isinstance(b, (list, tuple)) or len(b) != 3 or not all(isinstance(x, int) for x in b):
                raise ValueError(f"branch must be a list of three integers, got {b!r}")
        for key in ("observer_radius_fraction", "redshift_max"):
            if key in config and not isinstance(config[key], (int, float)):
                raise ValueError(f"{key} must be a number")
        for key in ("redshift_samples", "particles", "seed"):
            if key in config and not isinstance(config[key], int):
                raise ValueError(f"{key} must be an integer")
        if "export_formats" in config:
            fmts = config["export_formats"]
            if not isinstance(fmts, list):
                raise ValueError("export_formats must be a list")
            for f in fmts:
                if f not in ("json", "csv", "hdf5", "h5"):
                    raise ValueError(f"unknown export format: {f!r}")


def _merge_with_cli(args: argparse.Namespace) -> dict[str, Any]:
    """Build the final runtime config from defaults, file, and CLI overrides."""
    config: dict[str, Any] = dict(DEFAULT_CONFIG)

    if args.config:
        file_config = _load_config_file(args.config)
        _validate_config(file_config)
        config.update(file_config)
        # Normalise branch to tuple for downstream use
        if "branch" in config:
            config["branch"] = tuple(int(x) for x in config["branch"])

    # CLI overrides take precedence. Booleans and sweep are handled specially.
    if args.mode != "all":
        config["mode"] = args.mode
    branch_cli = tuple(int(x) for x in args.branch) if args.branch else None
    if branch_cli and branch_cli != DEFAULT_CONFIG["branch"]:
        config["branch"] = branch_cli
    if args.observer_radius_fraction != DEFAULT_CONFIG["observer_radius_fraction"]:
        config["observer_radius_fraction"] = args.observer_radius_fraction
    if args.redshift_max != DEFAULT_CONFIG["redshift_max"]:
        config["redshift_max"] = args.redshift_max
    if args.redshift_samples != DEFAULT_CONFIG["redshift_samples"]:
        config["redshift_samples"] = args.redshift_samples
    if args.particles != DEFAULT_CONFIG["particles"]:
        config["particles"] = args.particles
    if args.seed != DEFAULT_CONFIG["seed"]:
        config["seed"] = args.seed
    if args.format:
        config["export_formats"] = [args.format]
    if args.plot:
        config["plot"] = True

    # Logging and output control
    if args.log_level:
        config["log_level"] = args.log_level.upper()
    if args.verbose:
        config["log_level"] = "DEBUG"
        config["verbose"] = True
    if args.quiet:
        config["quiet"] = True
        if not args.log_level and not args.verbose:
            config["log_level"] = "WARNING"
    if args.log_format:
        config["log_format"] = args.log_format
    if args.log_file:
        config["log_file"] = args.log_file

    return config


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------

def _flatten_dict(prefix: str, obj: Any) -> dict[str, Any]:
    """Flatten a nested dict/list into dotted scalar columns."""
    out: dict[str, Any] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            out.update(_flatten_dict(key, v))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            key = f"{prefix}[{i}]" if prefix else f"[{i}]"
            out.update(_flatten_dict(key, v))
    else:
        out[prefix] = obj
    return out


def _rows_from_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return a list of flat dicts, one per record, normalised to the union of keys."""
    flat = [_flatten_dict("", r) for r in records]
    if not flat:
        return []
    keys = sorted({k for row in flat for k in row.keys()})
    return [{k: row.get(k, "") for k in keys} for row in flat]


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        path.write_text("")
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _write_records_csv(records: list[dict[str, Any]], path: Path) -> None:
    _write_csv(_rows_from_records(records), path)


def _write_hdf5_group(group: Any, key: str, value: Any) -> None:
    if isinstance(value, dict):
        g = group.create_group(key)
        for k, v in value.items():
            _write_hdf5_group(g, str(k), v)
    elif isinstance(value, (list, tuple)):
        if not value:
            group.create_dataset(key, data=[])
        elif all(isinstance(x, (int, float, bool)) for x in value):
            group.create_dataset(key, data=value)
        elif all(isinstance(x, str) for x in value):
            dt = h5py.string_dtype()
            arr = [str(x).encode("utf-8") for x in value]
            group.create_dataset(key, data=arr, dtype=dt)
        elif all(isinstance(x, dict) for x in value):
            g = group.create_group(key)
            for i, x in enumerate(value):
                _write_hdf5_group(g, f"row_{i}", x)
        else:
            group.create_dataset(key, data=[str(x) for x in value])
    elif isinstance(value, str):
        group.create_dataset(key, data=value.encode("utf-8"))
    elif isinstance(value, (int, float, bool)):
        group.create_dataset(key, data=value)
    elif value is None:
        group.create_dataset(key, data="")
    else:
        group.create_dataset(key, data=str(value))


def _metric_slices_for_export(result: dict[str, Any]) -> list[dict[str, Any]]:
    if "cosmology" in result and result["cosmology"]:
        return result["cosmology"]
    if "audit" in result and result["audit"].get("metric_slices"):
        return result["audit"]["metric_slices"]
    return []


def _history_entries_for_export(result: dict[str, Any]) -> list[dict[str, Any]]:
    if "history" in result and result["history"]:
        return result["history"]
    if "audit" in result and result["audit"].get("history_entries"):
        return result["audit"]["history_entries"]
    return []


def export_result(
    result: dict[str, Any],
    output_path: str | Path,
    fmt: str = "json",
) -> list[Path]:
    """Export a simulation result to the requested format.

    Supported formats:
      - ``json``: single JSON file with the full result tree.
      - ``csv``: two CSV files, ``{prefix}_metric_slices.csv`` and
        ``{prefix}_history.csv``.
      - ``hdf5``: single HDF5 file with groups ``/audit``, ``/cosmology``,
        ``/history``, and ``/baryogenesis``.
    """
    output_path = Path(output_path)
    fmt = fmt.lower()

    if fmt == "json":
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, sort_keys=True)
        return [output_path]

    if fmt == "csv":
        base = output_path.with_suffix("") if output_path.suffix else output_path
        paths: list[Path] = []
        slices = _metric_slices_for_export(result)
        if slices:
            p = Path(str(base) + "_metric_slices.csv")
            _write_records_csv(slices, p)
            paths.append(p)
        history = _history_entries_for_export(result)
        if history:
            p = Path(str(base) + "_history.csv")
            _write_records_csv(history, p)
            paths.append(p)
        if not paths:
            p = Path(str(base) + "_summary.csv")
            _write_csv([_flatten_dict("", result)], p)
            paths.append(p)
        return paths

    if fmt in ("hdf5", "h5"):
        if not _HAS_H5:
            raise RuntimeError("HDF5 export requires `h5py`. Install it from requirements.txt.")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(output_path, "w") as f:
            _write_hdf5_group(f, "shbt", result)
        return [output_path]

    raise ValueError(f"Unknown export format: {fmt!r}")


def export_results(
    result: dict[str, Any],
    output_dir: str | Path,
    basename: str = "result",
    formats: list[str] | None = None,
) -> list[Path]:
    """Export a result into *output_dir* in all requested formats.

    Returns the list of written paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    formats = formats or ["json"]
    paths: list[Path] = []
    for fmt in formats:
        fmt = fmt.lower()
        if fmt == "json":
            p = output_dir / f"{basename}.json"
        elif fmt == "csv":
            p = output_dir / basename  # export_result adds suffixes
        elif fmt in ("hdf5", "h5"):
            p = output_dir / f"{basename}.h5"
        else:
            raise ValueError(f"Unknown export format: {fmt!r}")
        paths.extend(export_result(result, p, fmt=fmt))
    return paths


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _sweep_combinations(sweep_config: dict[str, Any]) -> Iterable[dict[str, Any]]:
    """Yield dicts for every Cartesian combination of parameter lists."""
    keys = list(sweep_config.keys())
    values = [sweep_config[k] if isinstance(sweep_config[k], list) else [sweep_config[k]] for k in keys]
    for combo in itertools.product(*values):
        yield dict(zip(keys, combo))


def _numeric_or_none(x: Any) -> float | None:
    if isinstance(x, (int, float)):
        return float(x)
    return None


def _plot_result(result: dict[str, Any], prefix: Path, *, sweep: bool = False) -> list[Path]:
    """Generate optional plots and return the written file paths."""
    if not _HAS_MPL:
        logging.warning("Plotting requested but matplotlib is not installed; skipping.")
        return []

    files: list[Path] = []
    slices = _metric_slices_for_export(result)

    # 1. Metric eigenvalues vs tau/redshift
    if slices:
        fig, ax = _plt.subplots(figsize=(6, 4))
        taus = [s.get("tau", i) for i, s in enumerate(slices)]
        for dim in range(4):
            vals = [s.get("eigenvalues", [None] * 4)[dim] for s in slices]
            if all(v is not None for v in vals):
                ax.plot(taus, vals, marker="o", label=f"λ{dim}")
        ax.set_xlabel("τ")
        ax.set_ylabel("Metric eigenvalue")
        ax.set_title("Metric eigenvalues across bulk slices")
        ax.legend()
        fig.tight_layout()
        p = prefix.parent / (prefix.name + "_eigenvalues.png")
        fig.savefig(p)
        files.append(p)
        _plt.close(fig)

    # 2. Heatmap of spatial metric components from the first slice
    if slices and slices[0].get("spatial_metric"):
        fig, ax = _plt.subplots(figsize=(5, 4))
        sm = slices[0]["spatial_metric"]
        im = ax.imshow(sm, cmap="viridis", aspect="auto")
        ax.set_title("Spatial metric components (first slice)")
        ax.set_xlabel("j")
        ax.set_ylabel("i")
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        p = prefix.parent / (prefix.name + "_spatial_metric.png")
        fig.savefig(p)
        files.append(p)
        _plt.close(fig)

    # 3. eta_b sweep / bar chart
    if sweep and "sweep_results" in result:
        configs = result.get("sweep_configs", [])
        results = result["sweep_results"]
        eta_values = [r.get("audit", {}).get("eta_b") or r.get("baryogenesis", {}).get("identity", {}).get("eta_b") for r in results]
        # Try to find a numeric varying parameter for the x-axis
        varying_numeric = None
        if configs:
            for key in configs[0].keys():
                vals = [_numeric_or_none(c.get(key)) for c in configs]
                if all(v is not None for v in vals):
                    varying_numeric = key
                    xvals = vals
                    break
        fig, ax = _plt.subplots(figsize=(7, 4))
        if varying_numeric:
            ax.plot(xvals, eta_values, marker="o")
            ax.set_xlabel(varying_numeric)
        else:
            labels = [str(c.get("branch", i)) for i, c in enumerate(configs)]
            ax.bar(range(len(labels)), eta_values)
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=45, ha="right")
            ax.set_xlabel("sweep index")
        ax.set_ylabel("η_b")
        ax.set_title("Baryon asymmetry across sweep configurations")
        fig.tight_layout()
        p = prefix.parent / (prefix.name + "_eta_b.png")
        fig.savefig(p)
        files.append(p)
        _plt.close(fig)

    return files


# ---------------------------------------------------------------------------
# Simulation core
# ---------------------------------------------------------------------------

def simulate(config: dict[str, Any]) -> dict[str, Any]:
    """Run an SHBT simulation according to *config* and return a JSON-serialisable dict.

    Config keys:
      - mode: ``"audit"`` | ``"cosmology"`` | ``"baryogenesis"`` | ``"history"`` | ``"all"``
      - branch: tuple/list of three ints, default ``(26, 8, 312)``
      - observer_radius_fraction: float, default ``0.125``
      - redshift_max: float, default ``3.0``
      - redshift_samples: int, default ``9``
      - particles: int, default ``512``
      - seed: int, default ``0``
    """
    _ensure_rust()
    assert _rs is not None

    mode = config.get("mode", "all")
    branch_raw = config.get("branch", (26, 8, 312))
    branch = tuple(int(x) for x in branch_raw)
    if len(branch) != 3:
        raise ValueError(f"branch must be a tuple/list of three integers, got {branch!r}")
    observer_radius_fraction = float(config.get("observer_radius_fraction", 0.125))
    redshift_max = float(config.get("redshift_max", 3.0))
    redshift_samples = int(config.get("redshift_samples", 9))
    particles = int(config.get("particles", 512))
    seed = int(config.get("seed", 0))

    sim = _rs.ShbtSimulator.with_config(
        branch,
        observer_radius_fraction,
        redshift_max,
        redshift_samples,
        particles,
        seed,
    )

    result: dict[str, Any] = {
        "config": {
            "mode": mode,
            "branch": branch,
            "observer_radius_fraction": observer_radius_fraction,
            "redshift_max": redshift_max,
            "redshift_samples": redshift_samples,
            "particles": particles,
            "seed": seed,
        },
    }

    if mode in ("audit", "all"):
        report = sim.run_full_audit()
        result["audit"] = report.to_dict()

    if mode in ("cosmology", "all"):
        slices = sim.simulate_cosmology(redshift_max, redshift_samples)
        result["cosmology"] = [s.to_dict() for s in slices]

    if mode in ("baryogenesis", "all"):
        identity = sim.baryogenesis_identity()
        benchmark = sim.baryogenesis_benchmark()
        identity_dict = identity.to_dict()
        benchmark_dict = benchmark.to_dict()
        # Canonical top-level keys (mirrors the ShbtReport structure from Rust).
        result["baryogenesis_identity"] = identity_dict
        result["benchmark_delta"] = benchmark_dict
        result["eta_b"] = identity_dict["eta_b"]
        # Backward-compatible nested alias for older consumers.
        result["baryogenesis"] = {
            "identity": identity_dict,
            "benchmark": benchmark_dict,
        }

    if mode in ("history", "all"):
        entries = sim.crystallize_history()
        result["history"] = [e.to_dict() for e in entries]

    if mode in ("audit", "all"):
        # Ensure a top-level numeric eta_b is available for every mode that
        # computes baryogenesis (audit already contains it under `audit.eta_b`).
        result.setdefault("eta_b", result["audit"]["eta_b"])

    if mode not in ("audit", "cosmology", "baryogenesis", "history", "all"):
        raise ValueError(f"unknown simulation mode: {mode}")

    _add_repro_metadata(result)
    return result


def _add_repro_metadata(result: dict[str, Any]) -> None:
    """Attach version, git, and timestamp metadata to a simulation result."""
    result.setdefault("metadata", {}).update({
        "version": get_version(),
        "git_info": get_git_info(),
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    })


def run_sweep(sweep_config: dict[str, Any]) -> dict[str, Any]:
    """Run ``simulate`` for every Cartesian product of parameter lists in *sweep_config*.

    Returns a dict with ``sweep_configs`` (the list of concrete configs) and
    ``sweep_results`` (their outputs).
    """
    configs = list(_sweep_combinations(sweep_config))
    results: list[dict[str, Any]] = []
    total = len(configs)
    for i, cfg in enumerate(configs, 1):
        _LOGGER.info("Sweep run %d/%d with config %s", i, total, cfg)
        results.append(simulate(cfg))
    sweep_result: dict[str, Any] = {"sweep_configs": configs, "sweep_results": results}
    _add_repro_metadata(sweep_result)
    _log_event("sweep_complete", sweep_runs=total)
    return sweep_result


def _summarise(result: dict[str, Any]) -> dict[str, Any]:
    """Return a small human-readable summary of the result."""
    summary: dict[str, Any] = {"branch": result["config"]["branch"]}

    if "audit" in result:
        audit = result["audit"]
        summary["framing_defect"] = audit.get("boundary_report", {}).get("framing_defect")
        summary["modular_invariant"] = audit.get("boundary_report", {}).get("modular_invariant")
        summary["zero_energy_locked"] = audit.get("boundary_report", {}).get("zero_energy_locked")
        summary["projection_dimension_26_to_4"] = audit.get("boundary_report", {}).get("projection_dimension_26_to_4")
        summary["eta_b"] = audit.get("eta_b")
        summary["stress_energy_preserved"] = audit.get("stress_energy_preserved")
        summary["metric_slices"] = len(audit.get("metric_slices", []))
        summary["history_entries"] = len(audit.get("history_entries", []))

    if "cosmology" in result:
        summary["cosmology_slices"] = len(result["cosmology"])

    if "baryogenesis_identity" in result:
        summary["eta_b"] = result["baryogenesis_identity"].get("eta_b")
    elif "baryogenesis" in result:
        summary["eta_b"] = result["baryogenesis"]["identity"].get("eta_b")
    if "benchmark_delta" in result:
        summary["stress_energy_preserved"] = result["benchmark_delta"].get("stress_energy_preserved")
    elif "baryogenesis" in result:
        summary["stress_energy_preserved"] = result["baryogenesis"]["benchmark"].get("stress_energy_preserved")

    if "history" in result:
        summary["history_entries"] = len(result["history"])

    if "sweep_results" in result:
        summary["sweep_runs"] = len(result["sweep_results"])

    return summary


def print_summary(result: dict[str, Any]) -> None:
    """Print a clean, aligned ASCII table of the audit summary."""
    if "audit" not in result:
        return
    audit = result["audit"]
    rows = [
        ("Branch", str(result["config"]["branch"])),
        ("Framing defect (delta_fr)", str(audit.get("boundary_report", {}).get("framing_defect"))),
        ("Modular invariant", str(audit.get("boundary_report", {}).get("modular_invariant"))),
        ("Zero energy locked", str(audit.get("boundary_report", {}).get("zero_energy_locked"))),
        ("Projection dimension 26 -> 4", str(audit.get("boundary_report", {}).get("projection_dimension_26_to_4"))),
        ("eta_b", str(audit.get("eta_b"))),
        ("Stress energy preserved", str(audit.get("stress_energy_preserved"))),
        ("Metric slices", str(len(audit.get("metric_slices", [])))),
        ("History entries", str(len(audit.get("history_entries", [])))),
    ]
    label_width = max(len(label) for label, _ in rows) + 2
    value_width = max(len(value) for _, value in rows) + 2
    total_width = label_width + value_width + 1
    sep = "+" + "-" * total_width + "+"
    print(sep)
    print(f"| {'SHBT Audit Summary':<{total_width-2}} |")
    print(sep)
    for label, value in rows:
        print(f"| {label:<{label_width-1}}| {value:<{value_width}}|")
    print(sep)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="shbt_simulate.py",
        description="Run custom SHBT simulations using the Rust/PyO3 simulator.",
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default=None,
        help="path to a YAML or JSON configuration file (values can be overridden by CLI flags)",
    )
    parser.add_argument(
        "--mode",
        "-m",
        choices=["audit", "cosmology", "baryogenesis", "history", "all"],
        default=DEFAULT_CONFIG["mode"],
        help="simulation mode (default: all)",
    )
    parser.add_argument(
        "--branch",
        "-b",
        nargs=3,
        type=int,
        default=list(DEFAULT_CONFIG["branch"]),
        metavar=("K_L", "K_Q", "K"),
        help="boundary branch as three integers (default: 26 8 312)",
    )
    parser.add_argument(
        "--observer-radius-fraction",
        "-r",
        type=float,
        default=DEFAULT_CONFIG["observer_radius_fraction"],
        help="observer radius as a fraction of the global horizon (default: 0.125)",
    )
    parser.add_argument(
        "--redshift-max",
        "-z",
        type=float,
        default=DEFAULT_CONFIG["redshift_max"],
        help="maximum redshift for the past light cone (default: 3.0)",
    )
    parser.add_argument(
        "--redshift-samples",
        "-n",
        type=int,
        default=DEFAULT_CONFIG["redshift_samples"],
        help="number of redshift / causal samples (default: 9)",
    )
    parser.add_argument(
        "--particles",
        "-p",
        type=int,
        default=DEFAULT_CONFIG["particles"],
        help="particle count for the baryogenesis benchmark (default: 512)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_CONFIG["seed"],
        help="random seed for causal-point collapse selection (default: 0)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="optional explicit output path; overrides output_dir from config",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="directory for timestamped results (overrides config output_dir)",
    )
    parser.add_argument(
        "--format",
        "-f",
        choices=["json", "csv", "hdf5", "h5"],
        default=None,
        help="single output format; overrides export_formats from config",
    )
    parser.add_argument(
        "--sweep",
        "-s",
        type=str,
        default=None,
        help="path to a JSON file with parameter lists to sweep over",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="generate optional matplotlib plots alongside the data export",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="alias for --log-level DEBUG",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default=None,
        help="set logging level (default: INFO)",
    )
    parser.add_argument(
        "--log-format",
        choices=["text", "json"],
        default=None,
        help="log output format (default: text)",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="write log messages to this file in addition to the console",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="suppress non-essential console output",
    )
    return parser


def _timestamp_dir() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def _run_from_config(config: dict[str, Any]) -> dict[str, Any]:
    """Run simulate or a sweep based on the final config."""
    if "sweep" in config and config["sweep"]:
        sweep_path = config["sweep"]
        sweep_config = json.loads(Path(sweep_path).read_text(encoding="utf-8"))
        return run_sweep(sweep_config)
    return simulate(config)


def _write_run_info(result: dict[str, Any], output_dir: Path, basename: str) -> Path:
    """Write a reproducibility metadata file (JSON) and a plain-text log."""
    info = {
        "version": get_version(),
        "git_info": get_git_info(),
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "config": result.get("config"),
        "summary": _summarise(result),
    }
    info_path = output_dir / f"{basename}_run_info.json"
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, sort_keys=True)

    log_path = output_dir / f"{basename}.log"
    lines = [
        f"SHBT simulation run log",
        f"Version: {info['version']}",
        f"Timestamp: {info['timestamp']}",
        f"Git: {info['git_info']}",
        f"Config: {json.dumps(info['config'], sort_keys=True)}",
        f"Summary: {json.dumps(info['summary'], sort_keys=True)}",
    ]
    log_path.write_text("\n".join(lines), encoding="utf-8")
    return info_path


def _shbt_print(message: str, quiet: bool = False) -> None:
    """Print a message unless quiet mode is enabled."""
    if not quiet:
        print(message)


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    # Sweep mode is triggered by the dedicated CLI flag for now.
    if args.sweep:
        sweep_config = json.loads(Path(args.sweep).read_text(encoding="utf-8"))
        result = run_sweep(sweep_config)
        export_formats = [args.format] if args.format else ["json"]
        output_path = args.output or f"sweep_{_timestamp_dir()}.json"
        paths = export_result(result, output_path, fmt=export_formats[0])
        for p in paths:
            _shbt_print(f"Wrote SHBT sweep output to {p}", quiet=args.quiet)
        if args.plot:
            prefix = Path(output_path).with_name(Path(output_path).stem)
            plot_paths = _plot_result(result, prefix, sweep=True)
            for p in plot_paths:
                _shbt_print(f"Wrote plot to {p}", quiet=args.quiet)
        return 0

    config = _merge_with_cli(args)

    _setup_logging(
        level_name=config.get("log_level", "INFO"),
        fmt=config.get("log_format", "text"),
        log_file=config.get("log_file"),
        quiet=config.get("quiet", False),
    )

    _LOGGER.info("Starting SHBT simulation")
    _log_event("simulation_start", **{k: v for k, v in config.items() if k not in ("log_file",)})
    start = time.time()
    result = simulate(config)
    duration = time.time() - start
    result["metadata"]["duration_s"] = duration
    _log_event("simulation_complete", mode=config.get("mode"), duration_s=duration)

    # Determine where to write results
    explicit_output = args.output
    output_dir = args.output_dir or config.get("output_dir", "./simulation_results")
    output_name = config.get("output_name", "result")
    export_formats = config.get("export_formats", ["json"])
    if args.format:
        export_formats = [args.format]

    written: list[Path] = []
    out_dir: Path
    if explicit_output:
        explicit_path = Path(explicit_output)
        out_dir = explicit_path.parent or Path(".")
        output_name = explicit_path.stem
        fmt = args.format or _infer_format_from_path(explicit_output)
        written.extend(export_result(result, explicit_output, fmt=fmt))
    else:
        out_dir = Path(output_dir) / _timestamp_dir()
        out_dir.mkdir(parents=True, exist_ok=True)
        written.extend(export_results(result, out_dir, basename=output_name, formats=export_formats))

    for p in written:
        _shbt_print(f"Wrote SHBT output to {p}", quiet=config.get("quiet", False))

    _write_run_info(result, out_dir, output_name)
    _shbt_print(f"Wrote run log to {out_dir / (output_name + '.log')}", quiet=config.get("quiet", False))

    if config.get("plot") or args.plot:
        if explicit_output:
            prefix = Path(explicit_output).with_name(Path(explicit_output).stem)
        else:
            prefix = out_dir / output_name
        plot_paths = _plot_result(result, prefix, sweep=False)
        for p in plot_paths:
            _shbt_print(f"Wrote plot to {p}", quiet=config.get("quiet", False))

    if config.get("mode") in ("audit", "all"):
        _log_event(
            "audit_complete",
            branch=config.get("branch"),
            eta_b=result["audit"].get("eta_b"),
            framing_defect=result["audit"].get("boundary_report", {}).get("framing_defect"),
            stress_energy_preserved=result["audit"].get("stress_energy_preserved"),
            metric_slices=len(result["audit"].get("metric_slices", [])),
            history_entries=len(result["audit"].get("history_entries", [])),
        )
        print_summary(result)

    return 0


def _infer_format_from_path(path: str) -> str:
    ext = Path(path).suffix.lower()
    if ext == ".csv":
        return "csv"
    if ext in (".h5", ".hdf5"):
        return "hdf5"
    return "json"


if __name__ == "__main__":
    sys.exit(main())
