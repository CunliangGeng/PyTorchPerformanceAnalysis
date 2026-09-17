#!/usr/bin/env python
"""Build the notebooks from notebooks/src/*.py.

    uv run python build.py                      # every notebook
    uv run python build.py ep1                  # one notebook
    uv run python build.py --execute --verify   # also run them into references/, then check

For each notebook (every notebooks/src/*.py and notebooks/*.ipynb found, or the
names given):

  1. writes notebooks/<name>.ipynb from notebooks/src/<name>.py,
  2. with --execute, runs that notebook into references/<name>.ipynb (needs a GPU),
  3. with --verify, runs verify.py at the end.

--execute copies the notebook into references/ and runs it there, so the relative
"traces" directory the notebooks write to resolves to references/traces/ and the
run leaves its trace files beside the executed notebook.

Cell ids are derived from cell content, so an unchanged source rebuilds byte-identically.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
NB_DIR = os.path.join(HERE, "notebooks")            # the notebooks students open
SRC_DIR = os.path.join(NB_DIR, "src")               # the py:percent source of truth
REF_DIR = os.path.join(HERE, "references")          # executed copies, with outputs
TRACE_DIR = os.path.join(REF_DIR, "traces")         # the traces those runs leave behind


def discover():
    """Notebook names: every notebooks/src/*.py and every notebooks/*.ipynb, sorted.

    Dot-prefixed files are skipped, so checkpoints and other hidden files never
    become notebook names.
    """
    names = set()
    for folder, ext in ((SRC_DIR, ".py"), (NB_DIR, ".ipynb")):
        for entry in os.listdir(folder) if os.path.isdir(folder) else []:
            stem, suffix = os.path.splitext(entry)
            if suffix == ext and stem and not stem.startswith("."):
                names.add(stem)
    return sorted(names)


NOTEBOOKS = discover()

# Generous: ep2 times the training loop ten epochs at a time, several times over.
TIMEOUT_S = 1800


def run(cmd, **kw):
    print("  $", " ".join(cmd), flush=True)  # flush, or a log shows nothing until the end
    return subprocess.run(cmd, check=True, **kw)


def stable_cell_ids(nb):
    """Give every cell an id derived from its type and text. Duplicates get a counter."""
    seen = {}
    for cell in nb.cells:
        key = (cell.cell_type, cell.source)
        seen[key] = seen.get(key, 0) + 1
        digest = hashlib.sha1(f"{cell.cell_type}\n{seen[key]}\n{cell.source}".encode())
        cell.id = digest.hexdigest()[:8]
    return nb


def convert(src, dst):
    """Write the py:percent file `src` as the notebook `dst`, with stable cell ids."""
    import jupytext  # lazy: verify.py imports this module and tolerates a missing jupytext

    jupytext.write(stable_cell_ids(jupytext.read(src)), dst)


def generate(names=NOTEBOOKS):
    """Regenerate notebooks/<name>.ipynb from notebooks/src/<name>.py for each name."""
    for name in names:
        src = os.path.join(SRC_DIR, f"{name}.py")
        dst = os.path.join(NB_DIR, f"{name}.ipynb")
        if not os.path.exists(src):
            sys.exit(f"missing source: {src}\n"
                     f"`sync.py {name}` writes one from notebooks/{name}.ipynb")
        convert(src, dst)
        print(f"  -> {os.path.relpath(dst, HERE)}")


def list_traces():
    """Print every trace file in references/traces/ with its size."""
    files = sorted(f for f in (os.listdir(TRACE_DIR) if os.path.isdir(TRACE_DIR) else [])
                   if f.endswith(".json"))
    if not files:
        return
    print(f"\ntraces in {os.path.relpath(TRACE_DIR, HERE)}:")
    for f in files:
        mb = os.path.getsize(os.path.join(TRACE_DIR, f)) / 1e6
        print(f"  {f:34s} {mb:6.2f} MB")


def execute(names=NOTEBOOKS):
    """Run each notebook into references/<name>.ipynb, keeping its outputs and traces.

    The notebook is copied into references/ and executed in place. nbconvert starts
    the kernel in the notebook's own directory, so the notebooks' relative "traces"
    path becomes references/traces/ and every trace of the run lands there.
    """
    os.makedirs(REF_DIR, exist_ok=True)
    failures = []
    for name in names:
        nb = os.path.join(NB_DIR, f"{name}.ipynb")
        out = os.path.join(REF_DIR, f"{name}.ipynb")
        if not os.path.exists(nb):
            sys.exit(f"missing notebook: {nb}; run `build.py {name}` first")
        print(f"\n--- executing {name} -> {os.path.relpath(out, HERE)} ---", flush=True)
        t0 = time.perf_counter()
        shutil.copyfile(nb, out)
        try:
            run([
                sys.executable, "-m", "jupyter", "nbconvert",
                "--to", "notebook", "--execute", "--inplace",
                f"--ExecutePreprocessor.timeout={TIMEOUT_S}", out,
            ])
            print(f"  ok  {time.perf_counter() - t0:.0f} s")
        except subprocess.CalledProcessError:
            # --inplace only writes on success, so `out` is still the unexecuted copy.
            print(f"  FAILED after {time.perf_counter() - t0:.0f} s")
            failures.append(name)
    list_traces()
    if failures:
        sys.exit(f"\nexecution failed for: {', '.join(failures)}")
    print("\nall notebooks executed")


def verify():
    """Run verify.py and return its exit status."""
    return subprocess.run([sys.executable, os.path.join(HERE, "verify.py")],
                          check=False).returncode


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("names", nargs="*",
                    help="notebook names, e.g. ep1 (default: every notebook)")
    ap.add_argument("--all", action="store_true", help="build every notebook")
    ap.add_argument("--execute", action="store_true",
                    help="also run the built notebooks and write references/")
    ap.add_argument("--verify", action="store_true", help="run verify.py afterwards")
    args = ap.parse_args()

    names = NOTEBOOKS if args.all or not args.names else args.names
    unknown = [n for n in names if n not in NOTEBOOKS]
    if unknown:
        ap.error(f"unknown notebook(s): {unknown}; choose from {NOTEBOOKS}")

    if args.execute and shutil.which("nvidia-smi") is None:
        print("warning: no nvidia-smi found; execution will fail without a GPU")

    print(f"generating {', '.join(names)} from {os.path.relpath(SRC_DIR, HERE)}/")
    generate(names)
    if args.execute:
        execute(names)
    if args.verify:
        sys.exit(verify())


if __name__ == "__main__":
    main()
