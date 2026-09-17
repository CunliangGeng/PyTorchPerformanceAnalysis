#!/usr/bin/env python
"""Copy hand edits from a notebook back into notebooks/src/<name>.py.

    uv run python sync.py ep1                    # sync and rebuild
    uv run python sync.py --execute --verify     # every notebook, then references/, then check

Save the notebook in Jupyter, then run this. For each notebook named it:

  1. writes notebooks/src/<name>.py from notebooks/<name>.ipynb, with the standard header,
  2. rebuilds the notebook from that source, as build.py does,
  3. fails if any cell differs between the edited and the rebuilt notebook,
  4. with --execute and --verify, does what build.py does with those flags.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import jupytext

import build

HERE = os.path.dirname(os.path.abspath(__file__))

HEADER = """# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---
"""


def cells_of(nb):
    return [(c["cell_type"], "".join(c["source"])) for c in nb["cells"]]


def sync(name):
    nb_path = os.path.join(build.NB_DIR, f"{name}.ipynb")
    src_path = os.path.join(build.SRC_DIR, f"{name}.py")
    if not os.path.exists(nb_path):
        print(f"FAIL {name}: no such notebook: {os.path.relpath(nb_path, HERE)}")
        return False
    with open(nb_path) as fh:
        edited = json.load(fh)

    text = jupytext.writes(jupytext.read(nb_path), fmt="py:percent")
    # Swap jupytext's header (kernel name, jupytext version) for the one src/ uses.
    end = text.index("# ---\n", 5) + len("# ---\n")
    os.makedirs(build.SRC_DIR, exist_ok=True)
    with open(src_path, "w") as fh:
        fh.write(HEADER + text[end:])
    print(f"wrote {os.path.relpath(src_path, HERE)}")

    build.generate([name])

    with open(nb_path) as fh:
        rebuilt = json.load(fh)
    before, after = cells_of(edited), cells_of(rebuilt)
    if before == after:
        print(f"ok  {name}: {len(after)} cells identical before and after the rebuild")
        return True
    print(f"FAIL {name}: the rebuilt notebook differs from the edited one "
          f"({len(before)} cells before, {len(after)} after)")
    for i, (b, a) in enumerate(zip(before, after)):
        if b != a:
            print(f"  cell {i} ({b[0]}) differs")
    return False


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("names", nargs="*", help="notebook names, e.g. ep1")
    ap.add_argument("--all", action="store_true", help="sync every notebook")
    ap.add_argument("--execute", action="store_true",
                    help="re-run the synced notebooks into references/")
    ap.add_argument("--verify", action="store_true", help="run verify.py afterwards")
    args = ap.parse_args()

    names = build.NOTEBOOKS if args.all or not args.names else args.names
    if not names:
        ap.error("give one or more notebook names, or --all")
    unknown = [n for n in names if n not in build.NOTEBOOKS]
    if unknown:
        ap.error(f"unknown notebook(s): {unknown}; choose from {build.NOTEBOOKS}")

    if not all([sync(n) for n in names]):
        sys.exit(1)
    if args.execute:
        build.execute(names)
    if args.verify:
        subprocess.run([sys.executable, os.path.join(HERE, "verify.py")], check=False)


if __name__ == "__main__":
    main()
