---
name: Test the shbt-precision build pipeline
description: |
  How to set up and run the end-to-end Rust/Python/LaTeX build, test, audit,
  and PDF pipeline for the shbt-precision repository, including common path
  and interpreter pitfalls.
---

# Testing the shbt-precision build pipeline

## Devin Secrets Needed

None.

## Environment

- Target repo: `sys1own/shbt-precision`
- Primary branch under test: `devin/1785603332-latex-build-pipeline` (or whatever PR branch)
- Required system packages: `m4`, `build-essential`, `texlive-latex-base`, `texlive-latex-recommended`, `texlive-fonts-recommended`, `texlive-latex-extra`
- Required toolchain: latest stable Rust via `rustup update stable`, `cargo` in PATH
- Python: `python3` and `pytest`, with `requirements.txt` + `mpmath` installed
- TeX: `pdflatex` available

## One-liner

```bash
cd /path/to/shbt-precision
make clean && make
```

`make` runs `build` → `test` → `audit` → `pdf`. The `build` target compiles the Rust extension and copies `target/release/libshbt_simulator.so` to `target/release/shbt_simulator.so`.

## Important gotchas

1. **`pytest tests/` and `python examples/run_audit.py` are now self-contained.**
   `tests/test_simulator.py` inserts the repo root and `target/release` into `sys.path`, and `examples/run_audit.py` inserts `target/release`. If you still see `ModuleNotFoundError`, use the `Makefile` targets or set `PYTHONPATH=$(pwd):$(pwd)/target/release`.

2. **Watch out for pyenv shims.**
   If the environment has a pyenv `python`/`python3` shim, `python` may point to a Python that does not have `pytest` installed, while `pytest` may be a console script installed for the system Python. This can make `python3 -m pytest` fail even though `pytest` works.
   - Workaround: use `make test`, or explicitly call the system Python with `/usr/bin/python3 -m pytest tests/`.

3. **Cargo target directory is gitignored and not cleaned by `make clean`.**
   `make clean` removes auxiliary LaTeX files (`*.aux`, `*.log`, `*.out`, `*.toc`, `*.synctex.gz`, `sim_results.tex`) but not `target/`. If you need a truly fresh Rust build, run `cargo clean` first.

4. **LaTeX build has expected undefined references from the placeholder supplementary file.**
   `supplementary.tex` is intentionally blank, so `main.tex` may log warnings like `Reference `eq:supplement-lightest-neutrino' undefined`. These are warnings, not fatal errors, and `pdflatex` still exits `0`.

5. **`python shbt_simulate.py --build` calls `os._exit(0)` after a successful build.**
   This is a workaround to avoid an intermittent PyO3 interpreter-shutdown segfault. It allows `make` to continue to the next targets, but it bypasses normal Python cleanup (`atexit`, `__del__`, buffered log flush). Treat this as a decision item, not a routine success.

## Verification checklist

1. `make` completes with exit code `0`.
2. `target/release/libshbt_simulator.so` and `target/release/shbt_simulator.so` exist.
3. `pytest` reports all tests passed (23/23 in the current suite).
4. `python examples/run_audit.py` prints JSON containing `branch: [26, 8, 312]` and `eta_b` close to `6.449923359416e-10`.
5. `main.pdf` exists, is non-empty, and is newer than `main.tex`.
6. `main.log` contains no `Fatal error`, `Emergency stop`, or `!` error lines.
7. `make clean` removes auxiliary files while preserving `main.pdf` and the compiled extension.
