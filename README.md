# PyTorch Performance Analysis: from Profiling to Optimization

Hands-on notebooks to learn how to measure, profile, diagnose, and optimize AI models using PyTorch.

| Notebook | What it covers |
|---|---|
| `ep1` | Timing one GPU operation correctly, and reading its profile as a timeline and as a table |
| `ep2` | Analyse a whole training loop: measure, profile, diagnose, optimise, verify, then next iteration |


## For students

### Requirements

- Linux system
- A GPU
- Multiple CPUs
- [uv](https://docs.astral.sh/uv/)

### Setup

```bash
uv sync
uv run jupyter lab   # then open notebooks/ep1.ipynb
```

If a cell fails on your machine, `references/` has both notebooks already executed with
their outputs and traces kept, so you can keep following along.


## For instructors

| Path | What it is |
|---|---|
| `notebooks/src/*.py` | **Source of truth.** |
| `notebooks/*.ipynb` | What students open. Generated, and shipped with no outputs |
| `references/` | Both notebooks executed on the reference GPU, with outputs and traces kept |
| `build.py`, `sync.py`, `verify.py` | Build, sync back, check. Run each with `--help` |

### Before class

```bash
uv run python build.py --execute --verify   # rebuild, re-run references/, must report 0 failed
```

Needs a GPU and takes a few minutes. Teaching on a different GPU? Recalibrate first: run
either SETUP cell there, take the anchor it prints (`4096x4096 matmul = ... ms/iter`), put
that number in `EXPECTED_MATMUL_MS` in **both** `notebooks/src/*.py`, then rebuild.

### Editing

`notebooks/src/*.py` is the source of truth; the notebooks are rebuilt from it.

```bash
uv run python build.py ep1   # source -> notebook
uv run python sync.py ep1    # notebook -> source, after editing in Jupyter
```

Never edit a notebook and its source at the same time: `sync.py` overwrites the source with
the notebook. Sync before committing, because `verify.py` fails on a notebook that does not
match its source.


## Troubleshooting

**`torch.cuda.is_available()` is False.** Check `nvidia-smi` and that Jupyter runs in the
environment `uv sync` created.

**`num_workers > 0` hangs in `ep2`.** The `Dataset` is defined inside the notebook, which
needs Linux's fork. On Windows or macOS, stay at `num_workers=0` and read the reference
traces.

**Perfetto will not load a student's file.** Use the matching file from
`references/traces/`.
