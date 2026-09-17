#!/usr/bin/env python
"""Integrity checks on the built notebooks. Prints PASS/FAIL per check, exits 1 on any FAIL.

    uv run python verify.py        # or build.py --verify, sync.py --verify

  * each notebook has a source, a built .ipynb and an executed copy in references/
  * each notebooks/*.ipynb is exactly jupytext(notebooks/src/<name>.py), and the
    executed copy in references/ has the same cells
  * notebooks/ ships with no saved outputs; references/ ran fully, raised nothing,
    and printed no out-of-range line
  * every notebook declares the same EXPECTED_MATMUL_MS, so both scale their
    expected ranges from the same anchor
  * every notebook guards on CUDA with a RuntimeError and downloads nothing
  * any DataLoader iterator class a notebook names exists on the installed torch
  * references/traces/ holds complete Chrome traces under 20 MB

Nothing about wording or content is checked: the notebooks are revised by hand.
"""

from __future__ import annotations

import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

sys.path.insert(0, HERE)
import build  # noqa: E402 - the notebook list and the directory layout

NB_DIR = build.NB_DIR
SRC_DIR = build.SRC_DIR
REF_DIR = build.REF_DIR
TRACE_DIR = build.TRACE_DIR
NOTEBOOKS = build.NOTEBOOKS

TRACE_LIMIT_MB = 20

results: list[tuple[str, str, str]] = []


def check(item, ok, detail=""):
    """Record one PASS/FAIL row. `detail` says what to do about it, so only a FAIL shows it."""
    results.append(("PASS" if ok else "FAIL", item, "" if ok else detail))


def note(item, detail):
    results.append(("NOTE", item, detail))


def load(path):
    """The notebook at `path`, or None when the file does not exist."""
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return json.load(fh)


def cells(nb, kind=None):
    return [c for c in nb["cells"] if kind is None or c["cell_type"] == kind]


def src(cell):
    return "".join(cell["source"])


def text(nb, kind=None):
    return "\n".join(src(c) for c in cells(nb, kind))


def all_outputs(nb):
    """Everything the code cells of `nb` printed or returned, as one string."""
    parts = []
    for c in cells(nb, "code"):
        for o in c.get("outputs", []):
            if o.get("output_type") == "stream":
                parts.append("".join(o["text"]))
            elif o.get("output_type") == "execute_result":
                parts.append("".join(o["data"].get("text/plain", "")))
    return "\n".join(parts)


def cell_pairs(nb):
    """(cell_type, source) for every cell, which is all that a rebuild must preserve."""
    return [(c["cell_type"], src(c)) for c in nb["cells"]]


if not NOTEBOOKS:
    sys.exit(f"no notebooks found in {os.path.relpath(NB_DIR, HERE)}/")

print(f"checking {', '.join(NOTEBOOKS)} ...")

# ---------------------------------------------------------------- files present
nbs, refs, sources = {}, {}, {}
for n in NOTEBOOKS:
    src_path = os.path.join(SRC_DIR, f"{n}.py")
    nb_path = os.path.join(NB_DIR, f"{n}.ipynb")
    ref_path = os.path.join(REF_DIR, f"{n}.ipynb")
    missing = [
        f"{os.path.relpath(p, HERE)} ({how})"
        for p, how in ((src_path, f"`sync.py {n}` writes one from the notebook"),
                       (nb_path, f"run `build.py {n}`"),
                       (ref_path, f"run `build.py {n} --execute`"))
        if not os.path.exists(p)
    ]
    check(f"files {n} has a source, a built notebook and an executed copy", not missing,
          "missing " + "; ".join(missing))
    if os.path.exists(src_path):
        sources[n] = open(src_path).read()
    if (nb := load(nb_path)) is not None:
        nbs[n] = nb
    if (nb := load(ref_path)) is not None:
        refs[n] = nb

# ---------------------------------------------------------------- reference traces
# Executing a notebook writes its traces next to the executed copy. Each one must be
# complete and small enough for Perfetto to open in class.
traces = sorted(f for f in (os.listdir(TRACE_DIR) if os.path.isdir(TRACE_DIR) else [])
                if f.endswith(".json"))
check("traces references/traces/ holds at least one trace", bool(traces),
      "run `build.py --execute`")
for f in traces:
    p = os.path.join(TRACE_DIR, f)
    mb = os.path.getsize(p) / 1e6
    try:
        with open(p) as fh:
            events = json.load(fh).get("traceEvents")
        complete = isinstance(events, list) and bool(events)
    except (json.JSONDecodeError, AttributeError, UnicodeDecodeError):
        complete = False
    problems = []
    if not complete:
        problems.append("truncated or not a Chrome trace; rerun `build.py --execute`")
    if mb >= TRACE_LIMIT_MB:
        problems.append("Perfetto gets slow; profile fewer steps or drop with_stack")
    check(f"traces references/traces/{f} is a complete Chrome trace "
          f"under {TRACE_LIMIT_MB} MB ({mb:.2f} MB)", not problems, "; ".join(problems))

# ---------------------------------------------------------------- build artifacts
# The core rule: every .ipynb is generated from notebooks/src/*.py, never edited by hand.
try:
    import jupytext
except ImportError:  # pragma: no cover - jupytext is a hard dependency of build.py
    jupytext = None

if jupytext is None:
    note("build .ipynb matches notebooks/src/*.py", "jupytext not importable; check skipped")
else:
    for n in NOTEBOOKS:
        if n not in sources:
            continue
        fresh = cell_pairs(jupytext.reads(sources[n], fmt="py:percent"))
        if n in nbs:
            built = cell_pairs(nbs[n])
            check(f"build notebooks/{n}.ipynb is exactly jupytext(notebooks/src/{n}.py)",
                  built == fresh,
                  f"{len(built)} cells built vs {len(fresh)} from source; "
                  f"edited by hand? run `sync.py {n}` or `build.py {n}`")
        if n in refs:
            got = cell_pairs(refs[n])
            check(f"build references/{n}.ipynb has the same cells as notebooks/src/{n}.py",
                  got == fresh,
                  f"{len(got)} cells vs {len(fresh)} expected; the executed copy is stale, "
                  f"run `build.py {n} --execute`")

# ---------------------------------------------------------------- outputs
for n, nb in nbs.items():
    with_out = [i for i, c in enumerate(cells(nb, "code")) if c.get("outputs")]
    check(f"outputs notebooks/{n}.ipynb ships with no saved outputs", not with_out,
          f"{len(with_out)} code cells have outputs; students would open a solved notebook")

for n, nb in refs.items():
    code = cells(nb, "code")
    # execution_count, not outputs: definition-only cells run but print nothing.
    unexecuted = [i for i, c in enumerate(code) if c.get("execution_count") is None]
    check(f"outputs references/{n}.ipynb is fully executed", not unexecuted,
          f"{len(unexecuted)}/{len(code)} code cells were never executed")
    raised = [i for i, c in enumerate(code)
              if any(o.get("output_type") == "error" for o in c.get("outputs", []))]
    check(f"outputs references/{n}.ipynb ran without raising", not raised,
          f"code cells {raised} raised")
    # expect() starts the line with a warning sign when a value is outside its range.
    bad = [ln for ln in all_outputs(nb).splitlines() if ln.startswith("⚠")]
    check(f"outputs references/{n}.ipynb has every expectation in range", not bad,
          "; ".join(bad)[:200])

# ---------------------------------------------------------------- calibration anchor
# Both notebooks scale every expected range from this one measured constant, so a
# recalibration that touches only one of them would leave the two disagreeing.
ANCHOR = re.compile(r"^EXPECTED_MATMUL_MS\s*=\s*([\d.]+)\s*(?:#.*)?$", re.M)
anchors = {n: (m.group(1) if (m := ANCHOR.search(s)) else None) for n, s in sources.items()}
check("anchor every notebook declares the same EXPECTED_MATMUL_MS",
      len(set(anchors.values())) == 1 and None not in anchors.values(),
      f"{anchors}; the SETUP cells scale their expected ranges from it, so every "
      f"notebooks/src/*.py needs the same measured value")

# ---------------------------------------------------------------- runs on its own
# SystemExit does not stop a Jupyter kernel and aborts nbconvert, hence RuntimeError.
for n, nb in nbs.items():
    body = text(nb, "code")
    check(f"standalone {n} stops with a RuntimeError when CUDA is missing",
          "torch.cuda.is_available()" in body and "raise RuntimeError" in body)
    check(f"standalone {n} downloads nothing at runtime",
          not re.search(r"urllib|requests\.|wget|hf_hub|torchvision\.datasets", body),
          "the classroom wifi is not a dependency; generate the data instead")

# ---------------------------------------------------------------- torch internals
# The DataLoader iterator a notebook tells students to find in a trace is a private class,
# and show_stages() renames its row, so a torch rename would break both silently.
named = set()
for nb in nbs.values():
    named |= set(re.findall(r"\b(_\w*DataLoaderIter\w*)\b", text(nb)))
if named:
    import torch.utils.data.dataloader as _dl  # noqa: E402 - only to confirm the names

    for cls in sorted(named):
        check(f"torch the quoted DataLoader class {cls} exists on the installed torch",
              hasattr(_dl, cls))

# ---------------------------------------------------------------- report
print()
w = max(len(i) for _, i, _ in results)
fails = 0
for status, item, detail in results:
    if status == "FAIL":
        fails += 1
    tail = f"  {detail}" if detail else ""
    print(f"[{status}] {item:<{w}}{tail}")

n_pass = sum(1 for s, _, _ in results if s == "PASS")
n_note = sum(1 for s, _, _ in results if s == "NOTE")
print(f"\n{n_pass} passed, {fails} failed, {n_note} notes")
sys.exit(1 if fails else 0)
