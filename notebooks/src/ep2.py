# ---
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

# %% [markdown]
# # Episode 2: Profile and optimise a PyTorch training loop
#
# In Episode 1 you timed and profiled one operation. Here you profile a whole training
# loop: loading a batch, forward, loss, backward, optimizer and a metric. You will learn
# the workflow for profiling and optimising a PyTorch training loop.
#
# ## The performance optimisation loop
#
# Making PyTorch training faster is not a one-time effort but a loop, repeated until it is fast enough:
#
# ```
#   ⏱️ MEASURE  ->  📊 PROFILE  ->  🔍 DIAGNOSE  ->  🔧 OPTIMISE
#       ^                                                   |
#       └────────────────  ✅ VERIFY ───────────────────────┘
# ```
#
# 1. **MEASURE.** One stopwatch number for the whole training loop. Only this number says whether anything got faster.
# 2. **PROFILE.** Record a few steps with `torch.profiler`. Read the result as a timeline in Perfetto.
# 3. **DIAGNOSE.** Find the stage that owns the time, and why. Work out the most a fix
#    to it could buy. Decide on *one* change.
# 4. **OPTIMISE.** Make that one change. Nothing else.
# 5. **VERIFY.** Measure again: faster, and by how much compared with the diagnosis?
#    Profile again: where is the bottleneck *now*? That profile starts the next round.

# %% [markdown]
# ## SETUP — run this first
#
# The same cell as Episode 1: the GPU check, the 4096x4096 matmul benchmark that scales
# every "expected" range to your hardware, and the helpers `timeit()` and `expect()`.

# %%
import os
import time

import torch
import torch.nn as nn

# Stop here if there is no GPU.
if not torch.cuda.is_available():
    raise RuntimeError(
        "STOP. This notebook needs a CUDA GPU and torch.cuda.is_available() is False.\n"
        "Nothing below this cell will work. Ask for help before continuing."
    )

DEVICE = torch.device("cuda")

print("Environment Check:")
print(f"torch      {torch.__version__}")
print(f"CUDA       {torch.version.cuda}")
print(f"GPU        {torch.cuda.get_device_name(0)}")
vram_gib = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
print(f"VRAM       {vram_gib:.1f} GiB")
print(f"capability {torch.cuda.get_device_capability(0)}")
print()


def timeit(fn, warmup=3, iters=10):
    """Time a GPU function correctly. Returns milliseconds per call.

    Warm up first, then wait for the GPU before starting and before stopping
    the clock.
    """
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / iters * 1e3

def expect(label, value, low, high, unit="ms"):
    """Print a measured value next to the range we expect on this GPU."""
    ok = "✅ " if low <= value <= high else "⚠️ "
    print(f"{ok}{label}: {value:10.4f}{unit} (expected {low:g}-{high:g}{unit})")


# The GPU used to prepare these notebooks: NVIDIA GeForce RTX 2080 Ti, torch 2.13.0+cu130, CUDA 13.0.
# Every expected range below is scaled from this number.
EXPECTED_MATMUL_MS = 12.19

_a = torch.randn(4096, 4096, device=DEVICE)
_b = torch.randn(4096, 4096, device=DEVICE)
ANCHOR_MS = timeit(lambda: _a @ _b, warmup=3, iters=10)
SCALE = ANCHOR_MS / EXPECTED_MATMUL_MS
print(f"Anchor benchmark: 4096x4096 matmul = {ANCHOR_MS:.2f} ms/iter "
      f"(reference {EXPECTED_MATMUL_MS} ms)")
if SCALE > 3 or SCALE < 1 / 3:
    print()
    print("*" * 78)
    print(f"WARNING: this GPU is {SCALE:.1f}x the reference speed.")
    print("Every 'expected' range printed below is scaled from that reference and will")
    print("look wrong. The lesson still holds; the numbers will not match.")
    print("*" * 78)
else:
    print(f"this GPU is {SCALE:.2f}x the reference speed - expected ranges below apply.")

# %% [markdown]
# ---
# ## The training loop

# %% [markdown]
# ### The model
#
# The model predicts a binding affinity, one number, from an encoded peptide-MHC pair:
# a vector of 860 numbers. A `Linear` layer widens it to 2048 features, eight
# `AffinityBlock`s follow, and a final `Linear` layer gives the output. About 35M
# parameters. Each block is one `Linear` layer and a chain of cheap elementwise
# operations, the two kinds of kernel you measured in Episode 1.
#
# The cell also makes one random batch on the GPU, `xb` and `yb`. Round 1's MEASURE step
# uses it to time the step on its own, without any data loading.

# %%
D_IN = 860       # size of one input vector: 43 residues x 20 amino acids
HIDDEN = 2048    # number of hidden features
DEPTH = 8        # number of AffinityBlocks
BATCH = 256      # samples per batch
torch.manual_seed(0)   # the same weights and the same random batch on every run


class AffinityBlock(nn.Module):
    """Linear -> ReLU -> scale -> shift -> Dropout, then add the block's input back."""

    def __init__(self, hidden):
        super().__init__()
        self.linear = nn.Linear(hidden, hidden)
        self.scale = nn.Parameter(torch.ones(hidden))
        self.shift = nn.Parameter(torch.zeros(hidden))
        self.drop = nn.Dropout(0.1)

    def forward(self, x):
        h = self.linear(x)
        h = h.relu()
        h = h * self.scale
        h = h + self.shift
        h = self.drop(h)
        return x + h


class AffinityNet(nn.Module):
    """A Linear layer in, `depth` blocks, and a Linear layer out to one number."""

    def __init__(self, d_in=D_IN, hidden=HIDDEN, depth=DEPTH):
        super().__init__()
        self.stem = nn.Linear(d_in, hidden)
        self.blocks = nn.Sequential(*[AffinityBlock(hidden) for _ in range(depth)])
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        return self.head(self.blocks(self.stem(x)))


model = AffinityNet().to(DEVICE)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

# One random batch that already lives on the GPU, to time the step without data loading.
xb = torch.randn(BATCH, D_IN, device=DEVICE)
yb = torch.rand(BATCH, 1, device=DEVICE)

n_params = sum(p.numel() for p in model.parameters())
print(f"The model has {n_params / 1e6:.1f}M parameters")
print(f"Batch size is {BATCH}, input dimension is {D_IN}, hidden dimension is {HIDDEN}, output dimension is 1")

# %% [markdown]
# ### The data
#
# A peptide is a short chain of amino acids. The peptide and the MHC binding groove are
# encoded one residue per row, each row a one-hot vector over the 20 amino acids: 9 + 34
# residues, 43 rows of 20, 860 numbers. That is the model's input dimension `D_IN`.
#
# The cell defines the encoder, plain readable Python, and a `Dataset` around it. One
# epoch is `N_STEPS` batches of `BATCH` samples.

# %%
import random

from torch.utils.data import DataLoader, Dataset

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
MHC_PSEUDOSEQUENCE = "YFAMYQENMAHTDANTLYIIYRDYTWVARVYRGY"  # 34 residues


def encode_complex(peptide):
    """9-mer peptide + MHC pseudo-sequence -> one flat one-hot vector (43 x 20)."""
    seq = peptide + MHC_PSEUDOSEQUENCE
    rows = []
    for ch in seq:
        rows.append([1.0 if aa == ch else 0.0 for aa in AMINO_ACIDS])
    return torch.tensor(rows).reshape(-1)


class PeptideMHCDataset(Dataset):
    """Random 9-mer peptides with a random target affinity each."""

    def __init__(self, n_samples, seed=1):
        rng = random.Random(seed)
        self.peptides = [
            "".join(rng.choice(AMINO_ACIDS) for _ in range(9)) for _ in range(n_samples)
        ]
        self.targets = torch.rand(n_samples, 1, generator=torch.Generator().manual_seed(seed))

    def __len__(self):
        return len(self.peptides)

    def __getitem__(self, idx):
        return encode_complex(self.peptides[idx]), self.targets[idx]


N_STEPS = 10           # one epoch is N_STEPS batches
dataset = PeptideMHCDataset(BATCH * N_STEPS)

x0, y0 = dataset[0]
print(f"The dataset has {len(dataset)} samples, each encoded to a {x0.numel()}-dim vector")

# %% [markdown]
# ### One training step
#
# One training step consists of the following stages: forward, loss, backward, optimizer, and a metric.
# The metric is: a per-sample error and the loss pulled back to Python with `.item()`.
#
# Each stage is wrapped in a `record_function` label, as in Episode 1.

# %%
import torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile, record_function, schedule


def train_step(xb, yb):
    """One training step on the batch (xb, yb). Returns the loss and a metric."""
    with record_function("## 1-forward ##"):
        pred = model(xb)
    with record_function("## 2-loss ##"):
        loss = F.mse_loss(pred, yb)
    with record_function("## 3-backward ##"):
        loss.backward()
    with record_function("## 4-optimizer ##"):
        opt.step()
        opt.zero_grad(set_to_none=True)
    with record_function("## 5-metrics ##"):
        # Calculate the RMSE for each sample in the batch
        per_sample = []
        for i in range(len(pred)):
            per_sample.append(F.mse_loss(pred[i], yb[i]))
        batch_rmse = torch.stack(per_sample).mean().sqrt().item()
        # Get the loss value as a Python float (not a tensor)
        loss_value = loss.item()
    return loss_value, batch_rmse

# %% [markdown]
# ### The training loop for one epoch
#
# `train_epoch()` is the loop itself: for each batch, copy it to the GPU and call
# `train_step()`. PyTorch labels the fetching of a batch itself, and the copy gets
# our label `## 0-data-to-gpu ##`. The copy uses pinned memory and `non_blocking=True`, so
# that it does not make the CPU wait for the GPU: otherwise the GPU's time on the
# *previous* step would be charged to it.
#
#
# `time_step()` measures the time per training step (ms/step). It is built the way `timeit()` is:
# warm up, wait for the GPU, start the clock, run, wait for the GPU, stop the clock.

# %%
def train_epoch(loader, prof=None, step_fn=None):
    """Training on one epoch.

    Args:
        loader: a DataLoader, or an iterator over one.
        prof: a profiler to step after every batch, or None.
        step_fn: the training step function to run. None means `train_step`, the
            starting point. No cell below ever rebinds that name, so a round's own
            step is always passed in explicitly.
    """
    step = step_fn if step_fn is not None else train_step
    for xb, yb in loader:
        with record_function("## 0-data-to-gpu ##"):
            xb = xb.to(DEVICE, non_blocking=True)   # background copy: the CPU does not
            yb = yb.to(DEVICE, non_blocking=True)   # wait for it, or for the GPU
        step(xb, yb)
        if prof is not None:
            prof.step()


def make_loader(num_workers=0):
    """A DataLoader over the dataset.

    Args:
        num_workers: number of CPU processes to build and load the data batches.
            Default is 0, which means the main process does it.
    """
    return DataLoader(dataset, batch_size=BATCH, drop_last=True,
                      num_workers=num_workers, pin_memory=True)


def time_step(num_workers=0, step_fn=None):
    """Time the training step in ms for the whole training loop.

    Use this function to evaluate whether an optimization change helps, and to
    measure what the loop would cost with a stage taken out.

    Runs N_EPOCHS epochs and times each one, not counting its first batch,
    which waits for the workers to start. The first epoch is a warm-up and is thrown
    away; the median of the rest is returned. Enough epochs that a couple of noisy
    ones cannot move the median: this is the number every decision here rests on.

    Args:
        num_workers: passed on to make_loader().
        step_fn: the training step function to run, passed on to train_epoch().
            None means `train_step`, the starting point.
    """
    N_EPOCHS = 10                         # 1 warm-up + 9 timed
    loader = make_loader(num_workers)
    ms_per_step = []
    for _ in range(N_EPOCHS):
        batches = iter(loader)
        next(batches)                     # skip the first batch: with workers, it waits for them to start
        torch.cuda.synchronize()
        t0 = time.time()
        train_epoch(batches, step_fn=step_fn)
        torch.cuda.synchronize()
        ms_per_step.append((time.time() - t0) / (len(loader) - 1) * 1e3)
    timed = sorted(ms_per_step[1:])       # drop the warm-up epoch
    return timed[len(timed) // 2]         # median of the timed epochs

# %% [markdown]
# ---
# ## Round 1: is the data pipeline the bottleneck?
#
# The first round of the performance optimisation workflow is to check whether the data pipeline is the bottleneck.

# %% [markdown]
# ### ⏱️ MEASURE: the performance baseline and its ceiling
#
# 🔻 Question: how much time does the data pipeline cost?
#
# To answer this, we could measure the training loop with the data pipeline, and compare it with the time of a training loop without the data pipeline.

# %%
# The baseline time per step of the training loop, including the data pipeline plus the training step.
step_time_baseline = time_step()

# The ceiling: the training step without the data pipeline, timed on a batch that already
# lives on the GPU. The loop cannot beat this, whatever is done to the data pipeline.
step_time_ideal_r1 = timeit(lambda: train_step(xb, yb), warmup=5, iters=30)



print("The baseline time with data pipeline (ms/step)", step_time_baseline)
print("The ceiling: the training step alone (ms/step)", step_time_ideal_r1)
print()
print(f"The baseline is {step_time_baseline / step_time_ideal_r1:.1f}x slower than the step alone: "
      f"{step_time_baseline - step_time_ideal_r1:.0f} ms of every step is the data pipeline.")

# %% [markdown]
# ### 📊 PROFILE: record the trace
#
# Two helpers, used in every PROFILE and VERIFY step of this notebook:
#
# * `show_stages()` displays the CPU time of each stage in a table, and returns them as a dict. The total CPU time is also returned.
# * `save_trace()` writes the profile trace file, as in Episode 1.

# %%
N_ACTIVE = 2   # every profile below records this many steps; the helpers divide by it


def show_stages(prof, title):
    """Print a table with one row per '## n-name ##' stage: ms per step and share.

    What is measured: for each label, the time the main Python thread spent
    between entering and leaving it, averaged over the recorded steps. That is
    CPU time, not GPU time. A stage that only launches kernels (forward,
    backward, optimizer) shows the time spent *sending* work to the GPU. A stage
    that waits for the GPU (.item(), a blocking copy) shows the wait as well, so
    it is charged for all the GPU work queued before it. The total is the sum
    of the labels, not a wall clock.

    When to use it: to see which stage owns the step, and to compare the share
    column between two profiles of the same loop. To decide whether a change
    helped, use time_step() instead: that is a real stopwatch. To see gaps and
    overlap between the CPU and the GPU, open the trace in Perfetto: a table
    adds things up, and a gap is not a thing you can add up.

    Return:
        The time for each stage in ms (a dict), and the total time of all stages in ms.
    """
    stages = {}
    for e in prof.key_averages():
        if e.device_type == torch.autograd.DeviceType.CUDA:
            continue
        name = e.key
        if name.startswith("enumerate(DataLoader)#"):   # the loader fetching a batch
            name = "## 0-data-fetch ##"
        if not (name.startswith("## ") and name[3:4].isdigit()):   # '## <number>-name ##'
            continue
        stages[name] = stages.get(name, 0.0) + e.cpu_time_total / 1e3 / N_ACTIVE
    stages = dict(sorted(stages.items()))
    total = sum(stages.values())

    print(f"=== {title} ===")
    print(f"{'stage':<26s} {'ms/step':>9s}  {'% of step':>9s}")
    print("-" * 48)
    for name, ms in stages.items():
        print(f"{name:<26s} {ms:9.2f}  {100 * ms / total:8.1f}%")
    print("-" * 48)
    print(f"{'whole step (profiled)':<26s} {total:9.2f}  {100.0:8.1f}%")
    return stages, total


TRACE_DIR = "traces"
os.makedirs(TRACE_DIR, exist_ok=True)

def save_trace(prof, name):
    """Write the profile as a Chrome/Perfetto trace file and offer it for download."""
    path = os.path.join(TRACE_DIR, name)
    prof.export_chrome_trace(path, use_python_export=True)

    try:  # are we on Colab?
        from google.colab import files  # type: ignore
    except ImportError:
        files = None
    if files is not None:
        # Kept outside the try above: if the download itself fails we want to see
        # that error, not mistake it for "this is not Colab".
        files.download(path)
    else:
        try:  # Jupyter / JupyterLab
            from IPython.display import FileLink, display
            print("Download the trace and view it in https://ui.perfetto.dev:")
            print("- VSCode Jupyter: Explorer sidebar -> Right-click the file -> Download")
            print("- Browser Jupyter: Right-click the link below -> Save link as")
            display(FileLink(path))
        except ImportError:
            print(f"Open this file manually: {os.path.abspath(path)}")

# %% [markdown]
# The profiler call is similar to what you have done in Episode 1, with the same schedule: `train_epoch()` runs one
# epoch, the schedule skips the first step, warms up on the next two and records the two
# after that, averaging them. Recording several steps matters twice over: one slow step
# cannot distort the table, and the timeline shows the rhythm from step to step, which a
# single step cannot. The cell writes the trace;
# the stage table comes a few cells below, under DIAGNOSE.

# %%
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=schedule(wait=1, warmup=2, active=N_ACTIVE, repeat=1),
) as prof_baseline:
    train_epoch(make_loader(num_workers=0), prof=prof_baseline)

save_trace(prof_baseline, "ep2_baseline.json")


# %% [markdown]
# ### 🔍 DIAGNOSE: which stage owns the time, and why?
#
# #### 🛠 Task: diagnose it yourself
#
# Open `ep2_baseline.json` in **<https://ui.perfetto.dev>**. In Episode 1 you have read the trace of one matrix multiplication operation. Here you read the trace of a whole training loop which has hundreds or thousands of operations.
#
# You could first look the whole trace and find the different stages:
# - `## 0-data-to-gpu ##`
# - `## 1-forward ##`
# - `## 2-loss ##`
# - `## 3-backward ##`
# - `## 4-optimizer ##`
# - `## 5-metrics ##`
#
# then zoom in to one of the stages and look at the CPU row and GPU row to get a sense of what is happening.
#
# ⚠️ Two of those stages need a warning before you do. A `##` label is recorded on the main
# thread, and the GPU band drawn under it holds only the kernels launched from inside the
# label *on that thread*. Backward does not launch its own: it hands the work to the
# autograd engine, which runs on a separate CPU thread, so the band under
# `## 3-backward ##` is about a microsecond wide even though backward is roughly a third
# of the step's GPU time. `## 4-optimizer ##` gets no band at all — its GPU work sits one
# level down, under PyTorch's own `Optimizer.step#Adam.step`. For those two, read the GPU
# track itself rather than the band under the label, and look for backward's launches on
# that second CPU thread.
#
#
# The goal is to answer the following questions:
# 1. Is the GPU busy all the time, or does it wait for the CPU? If it waits, which stage is the CPU waiting for?
# 2. When GPU is idle, what is the CPU doing? Is it busy or idle?
# 3. How much of the CPU time is spent on each stage? Which stage owns the most time?

# %% [markdown]
# <details>
# <summary>💡 <b>Solution</b> </summary>
#
# **The GPU is idle most of the time, and during those gaps the CPU is busy.** The widest CPU block is
# `enumerate(DataLoader)#_SingleProcessDataLoaderIter.__next__`: the loader building a batch in the main process, one sample at a time.
#
# **Why.** `encode_complex` is a Python loop over 43 residues, about 150 microseconds
# per sample, times 256 samples: roughly 40 ms of CPU work per batch, none of it
# overlapping with the GPU. The GPU finishes a step, waits 40 ms for the next batch,
# works for a few milliseconds, waits again.
#
# **Why it matters.** The model's rows are small, so the model is not the problem.
# Training is a pipeline of stages on different hardware, and the slowest stage sets
# the pace.
# </details>

# %% [markdown]
# #### Display the stage time in a table

# %%
stage_time_baseline, stage_total_time_baseline = show_stages(prof_baseline, "the baseline")

# %% [markdown]
# Note that `## 0-data-fetch ##` is a name `show_stages()` gives to the
# `enumerate(DataLoader)#_SingleProcessDataLoaderIter.__next__` block. The trace itself only
# has the long name, so search Perfetto for that one.
#
# 🔻 Question: Are these numbers the same as what you found in the Perfetto profile?

# %% [markdown]
# #### The most a fix can buy
#
# A diagnosis ends with a number: the most a perfect fix to this stage could buy. The
# table says *which* stage to aim at, but it cannot give that number. Its shares are CPU
# time inside labels, and the labels do not add up to the wall clock: a label that waits
# for the GPU is charged for all the GPU work queued before it, which is why the column
# sums to more than the step really costs.
#
# So measure it rather than compute it. Run the loop again with the stage gone, and the
# stopwatch gives you the **ceiling**: the fastest this loop could ever run, in ms/step,
# if a fix to that stage were perfect. Divide the baseline by the ceiling and you have the
# **speed-up cap**: the same fact written as a multiple, which is the form VERIFY checks
# against.
#
# For the data pipeline we already have the ceiling, from MEASURE. `step_time_ideal_r1` timed the
# step on a batch that was already on the GPU, which is exactly this loop with no data
# pipeline in it at all.

# %%
data_fetch_share = stage_time_baseline["## 0-data-fetch ##"] / stage_total_time_baseline
speedup_cap_r1 = step_time_baseline / step_time_ideal_r1

print(f"## 0-data-fetch ## owns {100 * data_fetch_share:.0f}% of the step's CPU labels,")
print("which is why the data pipeline is the stage to aim at.")
print()
print(f"With no data pipeline at all the loop would be {step_time_ideal_r1:.2f} ms/step, against")
print(f"{step_time_baseline:.2f} ms/step now. So no fix to the data pipeline can beat "
      f"{speedup_cap_r1:.2f}x,")
print("and that is what VERIFY checks against.")

# %% [markdown]
# ### 🔧 OPTIMISE: worker processes
#
# The time goes into building batches in the main process while the GPU waits. Aim
# there, and change one thing: `num_workers=4`. Four extra processes then build the
# batches in the background while the GPU works on the previous step. Building a batch
# does not get cheaper; the work moves off the main process.
#
# More workers than CPU cores will not help. On a two-core machine use 2 and expect a
# smaller gain.
#

# %%
NUM_WORKERS = 4
cores_avail = os.cpu_count()
print(f"CPU cores available: {cores_avail}, workers: {NUM_WORKERS}")

# %% [markdown]
# ### ✅ VERIFY (1/2): measure again
#
# The stopwatch, next to the speed-up cap from the diagnosis. Close to the cap: the fix did what
# the diagnosis said. Far below: the stage is still there, or the fix cost something
# elsewhere.
#
# The speed-up cap is itself a stopwatch reading, with the same run-to-run spread as this
# one, so a result can land a little above it. That is not a fix beating its own cap; it is
# two measurements agreeing to within their noise, which means the fix took essentially
# all there was to take.

# %%
step_time_r1 = time_step(num_workers=NUM_WORKERS)
speedup_r1 = step_time_baseline / step_time_r1

print(f"with num_workers={NUM_WORKERS}, the training loop takes: {step_time_r1:.2f} ms/step")
print(f"the baseline was {step_time_baseline:.2f} ms/step")
print(f"so the speed-up is {speedup_r1:.2f}x, the diagnosis said at most {speedup_cap_r1:.2f}x")

# %% [markdown]
# ### ✅ VERIFY (2/2): profile again
#
# The stopwatch says it helped. The profile says where the time goes *now*.

# %%
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=schedule(wait=1, warmup=2, active=N_ACTIVE, repeat=1),
) as prof_r1:
    train_epoch(make_loader(num_workers=NUM_WORKERS), prof=prof_r1)

save_trace(prof_r1, "ep2_r1.json")

print()
stage_time_r1, stage_total_time_r1 = show_stages(prof_r1, "after fixing the data loading")

# %% [markdown]
# #### 🛠 Task: compare the two traces side by side
#
# Open `ep2_r1.json` in a **second browser tab**, next to the starting point.
#
# 1. Look for `enumerate(DataLoader)#_MultiProcessingDataLoaderIter.__next__`. Can you still find it?
#    Is it still the widest block?
# 2. Does GPU still wait for loading a batch? If so, which stage is the CPU waiting for? How much time does it take?
# 3. Which block or stage is the widest now? How much time does it take?

# %% [markdown]
# ---
# ## Round 2: has the bottleneck moved?
#
# Fixing the slowest stage hands the title to whatever was second. Round 2 starts from
# Round 1's VERIFY: `step_time_r1` and the table "after fixing the data loading" are its MEASURE
# and PROFILE, so it begins at DIAGNOSE.
#
# The metric stage turns out to have two defects. The rule is one change at a time, so
# OPTIMISE and VERIFY run twice: one change, one measurement, then the next change and
# another measurement.
#

# %% [markdown]
# ### 🔍 DIAGNOSE: the same stage, before and after
#
# 🔻**Question**: `## 5-metrics ##` has not changed, so it does the same work as at the
# starting point. What is its *percentage* of the step now?

# %%
share_before = stage_time_baseline["## 5-metrics ##"] / stage_total_time_baseline
share_after = stage_time_r1["## 5-metrics ##"] / stage_total_time_r1

print(f"## 5-metrics ## before the loader fix: {stage_time_baseline['## 5-metrics ##']:6.2f} ms, "
      f"{100 * share_before:5.1f}% of the step's CPU labels")
print(f"## 5-metrics ## after the loader fix:  {stage_time_r1['## 5-metrics ##']:6.2f} ms, "
      f"{100 * share_after:5.1f}% of the step's CPU labels")

# %% [markdown]
# #### The most a fix to the metric can buy
#
# The ceiling again, measured the same way: the same loop, with the metric stage taken out.

# %%
def train_step_no_metric(xb, yb):
    """The training step above, with the ## 5-metrics ## stage removed."""
    with record_function("## 1-forward ##"):
        pred = model(xb)
    with record_function("## 2-loss ##"):
        loss = F.mse_loss(pred, yb)
    with record_function("## 3-backward ##"):
        loss.backward()
    with record_function("## 4-optimizer ##"):
        opt.step()
        opt.zero_grad(set_to_none=True)
    return None, None


# The ceiling for Round 2: the same loop with the metric stage gone altogether.
step_time_ideal_r2 = time_step(num_workers=NUM_WORKERS, step_fn=train_step_no_metric)
speedup_cap_r2 = step_time_r1 / step_time_ideal_r2

print(f"With no metric stage at all the loop would be {step_time_ideal_r2:.2f} ms/step, against")
print(f"{step_time_r1:.2f} ms/step now. So no fix to the metric can beat {speedup_cap_r2:.2f}x,")
print("and that is what VERIFY checks against.")

# %% [markdown]
# ### 🔧 OPTIMISE (1/2): vectorise the metric
#
# The metric stage has two defects, not one. The first one is a Python loop over 256 samples, launching a
# couple of tiny kernels per sample, and the second one will be left to you to discover and fix later.
#
# The cell defines a *new* function, `train_step_vectorised()`, and leaves `train_step()`
# alone. The first four stages are identical. The metric is computed for the whole batch
# at once instead of sample by sample.

# %%
def train_step_vectorised(xb, yb):
    """train_step() with the per-sample metric loop replaced by one batch computation."""
    with record_function("## 1-forward ##"):
        pred = model(xb)
    with record_function("## 2-loss ##"):
        loss = F.mse_loss(pred, yb)
    with record_function("## 3-backward ##"):
        loss.backward()
    with record_function("## 4-optimizer ##"):
        opt.step()
        opt.zero_grad(set_to_none=True)
    with record_function("## 5-metrics ##"):
        # The batch RMSE for all 256 samples at once, in a handful of kernels instead of
        # a couple per sample.
        batch_rmse = ((pred - yb) ** 2).mean().sqrt().item()
        loss_value = loss.item()
    return loss_value, batch_rmse

# %% [markdown]
# ### ✅ VERIFY (1/2): measure and profile
#
# The stopwatch, then the stage table, the kernel count and the trace. The kernel count
# is the number that should move: the Python loop was launching a couple of kernels per
# sample.

# %%
# Measure to verify
step_time_r2a = time_step(num_workers=NUM_WORKERS, step_fn=train_step_vectorised)
speedup_r2a = step_time_r1 / step_time_r2a

print(f"with the vectorised metric, the training loop takes: {step_time_r2a:.2f} ms/step")
print(f"after Round 1 it was {step_time_r1:.2f} ms/step")
print(f"so this half-round bought {speedup_r2a:.2f}x, of the {speedup_cap_r2:.2f}x available")
print("for the metric stage as a whole.")

# %%
# helper function to count the number of kernel launches in a profile
def count_kernels(prof):
    """CUDA kernel launches per recorded step: GPU-side rows that ran a kernel,
    leaving out our '##' labels, PyTorch's own annotations, and memory copies.

    Read it as a round number, not an exact one. A recorded step ends when the CPU
    calls prof.step(), so once the CPU runs ahead of the GPU that boundary stops
    lining up with GPU execution and kernels from neighbouring steps drift into the
    count. It is steady to the kernel while something in the step makes the CPU wait
    for the GPU (an .item(), say), and wobbles by a few percent once nothing does.
    """
    return sum(
        e.count
        for e in prof.key_averages()
        if e.device_type == torch.autograd.DeviceType.CUDA
        and not e.is_user_annotation
        and e.self_device_time_total > 0
        and not e.key.startswith(("Memset", "Memcpy"))
    ) // N_ACTIVE


# profile to verify
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=schedule(wait=1, warmup=2, active=N_ACTIVE, repeat=1),
) as prof_r2a:
    train_epoch(make_loader(num_workers=NUM_WORKERS), prof=prof_r2a,
                step_fn=train_step_vectorised)

save_trace(prof_r2a, "ep2_r2a.json")

stage_time_r2a, stage_total_time_r2a = show_stages(prof_r2a, "after vectorising the metric")
print()
print(f"CUDA kernels per step: {count_kernels(prof_r1)} before, "
      f"{count_kernels(prof_r2a)} after")

# %% [markdown]
# #### 🛠 Task: what changed in the metric stage?
#
# Open `ep2_r2a.json` next to Round 1's `ep2_r1.json`, in two browser tabs,
# and zoom into `## 5-metrics ##` in each.
#
# 1. In the Round 1 trace that block is a dense picket fence of tiny kernels. How many
#    are there now, and how much narrower is the block?
# 2. The block did not disappear. Look at the CPU row inside it: is the CPU running, or
#    is it stopped? If it is stopped, what is it stopped on?
# 3. Look at the GPU row across a whole step. Is it busy from end to end, or does it go
#    idle and start again? Where does the idle time sit?

# %% [markdown]
# ### 🛠 Your turn: the second half-round
#
# One defect is left in the metric stage, and this half-round is yours to run, code and
# all. Round 2 began at DIAGNOSE because Round 1's VERIFY had already measured and
# profiled for it. The same is true here: VERIFY (1/2) just handed you a stopwatch
# number, a stage table and a trace. Every VERIFY is the next round's MEASURE and
# PROFILE, so you start at DIAGNOSE.
#
# Work the steps in order and write the code yourself, in the empty cell below. The first
# half-round is your template: every kind of cell you need has been run once already, a
# few cells up. Read them, do not copy them blindly.
#
# **1. DIAGNOSE — read the evidence.**
#
# * a. In the table above, `## 5-metrics ##` launches only a handful of kernels now. What
#      does it still cost in ms/step, and what share of the step is that?
# * b. Zoom into `## 5-metrics ##` in `ep2_r2a.json`. Is the CPU running inside that
#      block, or stopped? If it is stopped, what is it stopped on, and what is it waiting
#      for?
# * c. Which two lines of `train_step_vectorised()` put it there? Name them before you go
#      on.
#
# **2. DIAGNOSE — end with a number.** `speedup_cap_r2` was measured for the metric stage as a
# whole, and the first half-round has already taken part of it. How much is left for this
# one? Write that number down now, before you measure, so that VERIFY can contradict you.
#
# **3. OPTIMISE — one change, and only one.** Decide from 1c what to change, and write a
# new step function. It must:
#
# * have a **new name**, and leave `train_step` and `train_step_vectorised` untouched, so
#   that every number measured so far stays reproducible;
# * change nothing except what your diagnosis pointed at.
#
# **4. VERIFY — measure.** Time the loop with your step, at the same `num_workers` as the
# cells above, and print three numbers: your ms/step, the speed-up over `step_time_r2a` for
# this half-round alone, and the speed-up over `step_time_r1` for both halves together. Put
# the last one next to `speedup_cap_r2` and next to your own prediction from step 2.
#
# **5. VERIFY — profile.** Profile one epoch with your step, on the same schedule as the
# cells above. Save the trace as `ep2_r2b.json`, print the stage table with
# `show_stages()`, and print your kernel count next to `count_kernels(prof_r2a)`. Before
# you run it: predict whether the kernel count will go up, down, or stay where it is, and
# why. Then run the cell a second time and see whether the count is even the same as
# itself.
#
# **6. VERIFY — look at the timeline.** Open your trace next to `ep2_r2a.json` and
# answer the questions in the task below. Then add your row to the overview table further
# down, so the whole notebook is in one place.

# %%
# Write your code here

# %% [markdown]
# #### 🛠 Task: what did your change do to the timeline?
#
# Open `ep2_r2b.json` next to `ep2_r2a.json` from the first half-round. Your kernel count
# probably came out a little *higher* than Round 2a's, and a second run probably gave a
# different number again. Nothing extra is being computed: `.detach()` launches no kernel
# at all, and the arithmetic above it is untouched. What moved is the bookkeeping, and the
# solution below explains it. The work is the same; only *when* it happens has changed,
# which is what the rest of this task is about.
#
# 1. Inside `## 5-metrics ##` the CPU was stopped, waiting. Is it still?
# 2. Follow one step into the next. In the first trace the CPU cannot start step k+1
#    until the GPU has finished step k. Is that still true in the second?
# 3. Look at the GPU row across the boundary between two steps. Does it stay busy now,
#    and if so, where did the work filling that gap come from?
#
# Then open `ep2_baseline.json` from the very beginning in a third tab and look at all
# three. That is the whole notebook in one picture.

# %% [markdown]
# <details>
# <summary>💡 <b>Solution</b> — the code and why it works. Open it after you have measured your own.</summary>
#
# **The diagnosis.** `## 5-metrics ##` still cost milliseconds while launching almost no
# kernels, and inside it the CPU was stopped. It was stopped on the two `.item()` calls.
# `.item()` copies one number from the GPU to Python, and the CPU cannot have that number
# until the GPU has computed it, which means waiting for everything queued ahead of it:
# this step's forward, backward and optimizer. The label is charged for all of that
# waiting.
#
# **The change.** Replace the two `.item()` calls with `.detach()`. The arithmetic is
# identical and so are the numbers; they simply stay on the GPU as tensors. A real training
# loop keeps a running total that way and calls `.item()` on it once an epoch, not once a
# step.
#
# ```python
# def train_step_no_sync(xb, yb):
#     """train_step_vectorised(), with the two .item() calls replaced by .detach()."""
#     with record_function("## 1-forward ##"):
#         pred = model(xb)
#     with record_function("## 2-loss ##"):
#         loss = F.mse_loss(pred, yb)
#     with record_function("## 3-backward ##"):
#         loss.backward()
#     with record_function("## 4-optimizer ##"):
#         opt.step()
#         opt.zero_grad(set_to_none=True)
#     with record_function("## 5-metrics ##"):
#         # The same arithmetic, left on the GPU as tensors. Nothing here waits.
#         batch_rmse = ((pred - yb) ** 2).mean().sqrt().detach()
#         loss_value = loss.detach()
#     return loss_value, batch_rmse
#
#
# # VERIFY: measure
# step_time_r2b = time_step(num_workers=NUM_WORKERS, step_fn=train_step_no_sync)
#
# print(f"with no .item() in the metric, the loop takes: {step_time_r2b:.2f} ms/step")
# print(f"after vectorising it was {step_time_r2a:.2f} ms/step, after Round 1 {step_time_r1:.2f} ms/step")
# print()
# print(f"this half-round on its own:  {step_time_r2a / step_time_r2b:.2f}x")
# print(f"both halves against Round 1: {step_time_r1 / step_time_r2b:.2f}x, "
#       f"the diagnosis said at most {speedup_cap_r2:.2f}x")
#
#
# # VERIFY: profile
# with profile(
#     activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
#     schedule=schedule(wait=1, warmup=2, active=N_ACTIVE, repeat=1),
# ) as prof_r2b:
#     train_epoch(make_loader(num_workers=NUM_WORKERS), prof=prof_r2b,
#                 step_fn=train_step_no_sync)
#
# save_trace(prof_r2b, "ep2_r2b.json")
#
# stage_time_r2b, stage_total_time_r2b = show_stages(prof_r2b, "after the metric fix")
# print()
# print(f"CUDA kernels per step: {count_kernels(prof_r2a)} before this half-round, "
#       f"{count_kernels(prof_r2b)} after")
# ```
#
# **Why it works.** `.item()` makes the CPU wait for this step's GPU work before it can
# queue the next step's, so the two take turns instead of overlapping. Remove it and the
# CPU runs ahead: the launch work for step k+1 happens while the GPU is still busy with
# step k. Nothing got cheaper; things stopped waiting for each other.
#
# **Why the table could not tell you.** The kernel count counts work, and this was never
# about the amount of work. The stage table adds up CPU time inside labels and has no
# column for an overlap. For that you open the trace.
#
# </details>

# %% [markdown]
# ---
# ## Overview of the two rounds
#
# Let's see how much speedup we have gained in the two rounds of profiling and optimising the training loop.

# %%
# The ceiling as Round 2a left it: the step alone, with the vectorised metric.
step_time_ideal_r2a = timeit(lambda: train_step_vectorised(xb, yb), warmup=5, iters=30)

print(f"{'configuration':<40s} {'ms/step':>9s} {'us/sample':>11s} {'speed-up':>10s}")
print("-" * 74)
for label, ms in [
    ("starting point", step_time_baseline),
    ("+ num_workers=4", step_time_r1),
    ("+ vectorised metric", step_time_r2a),
    ("ceiling at the start: the step alone", step_time_ideal_r1),
    ("ceiling now: the step alone", step_time_ideal_r2a),
]:
    print(f"{label:<40s} {ms:9.2f} {ms / BATCH * 1000:11.1f} {step_time_baseline / ms:9.2f}x")
print("-" * 74)
print()
print(f"The loop is now {step_time_r2a / step_time_ideal_r2a:.2f}x the step alone.")
print(f"On a 2,000,000-sample dataset, one epoch went from "
      f"{step_time_baseline / BATCH * 2e6 / 1000 / 60:.1f} min to "
      f"{step_time_r2a / BATCH * 2e6 / 1000 / 60:.1f} min.")

# %% [markdown]
# ---
# ## Round 3: the model itself (optional, after class)
#
# One more turn of the loop, and this one is entirely yours: the diagnosis, the change and
# the verification.
#
# Everything you need is something you produced yourself: the loop time from Round 2b, the
# stage table printed beside it, and `ep2_r2b.json` in Perfetto.

# %% [markdown]
# **1. DIAGNOSE.** Which rows are left in your Round 2b stage table, and what share do
# they hold? Add them up: the total no longer matches the loop time, so where is the rest of
# the time going? Then put the five questions at the end of this notebook to `ep2_r2b.json`.
#
# **2. Set your expectations.** Rounds 1 and 2 measured a ceiling by timing the loop with
# the offending stage removed. What would that mean here, for the forward and backward
# passes? Say what size of win to expect *before* you go looking for one.
#
# **3. Take your baseline.** Before changing anything, time your Round 2b step on `xb, yb`
# with `timeit(lambda: ..., warmup=5, iters=30)`. Not `step_time_ideal_r2a`: that times
# `train_step_vectorised`, which still calls `.item()`, so it would mix in your Round 2b
# change.
#
# **4. OPTIMISE — one change.** It must leave the model's arithmetic alone (the same maths,
# run more cheaply), leave your Round 2b step function alone, and if it costs something once
# rather than every step, you measure that too. Stuck? Open the hint.
#
# **5. VERIFY — measure.** Print the one-off cost, the loop time next to Round 2b's, and the
# step alone next to your step 3 baseline.
#
# **6. VERIFY — profile.** Same schedule as the cells above. Save `ep2_r3.json`, print the
# stage table, and print the kernel count next to the one from your own Round 2b profile.
#
# **7. VERIFY — decide.** The loop, the step alone and the kernel count can disagree. Say
# which you believe, and whether the change is worth keeping. No cap to check against and no
# `expect()` line here: the judgement is yours.

# %% [markdown]
# <details>
# <summary>💡 <b>Hint</b> — open this if you got stuck</summary>
#
# Your diagnosis should have landed on the forward and backward passes, and those are two
# kinds of work: a few big matmuls, and long chains of cheap elementwise operations — the
# same two kinds you measured in Episode 1. You cannot delete either and still be training.
# But you can change how many *kernels* they take.
#
# `torch.compile(model)` reads the model once and fuses each block's chain of elementwise
# operations into a single generated kernel. The matmuls are left alone, which already tells
# you roughly how big the win can be.
#
# </details>

# %%
# Write your code here

# %% [markdown]
# <details>
# <summary>💡 <b>Solution</b> — the code and what it measures. Open it after your own attempt.</summary>
#
# This code carries straight on from the Round 2b solution, so it uses that solution's
# names: `train_step_no_sync` for the step function, `step_time_r2b` for its loop time and
# `prof_r2b` for its profile. Round 2b asked you to pick your own name, so yours are
# probably different — put yours in wherever these three appear below, or the cell will
# stop at a `NameError`.
#
# ```python
# # The step 3 baseline: the step alone, before compiling.
# step_time_ideal_r2b = timeit(lambda: train_step_no_sync(xb, yb), warmup=5, iters=30)
# print(f"the step alone, eager: {step_time_ideal_r2b:.2f} ms")
#
# # The change. train_step_no_sync looks `model` up by name, so it picks this up.
# model = torch.compile(model)
#
# # The one-off cost: the first call pays for the compile.
# torch.cuda.synchronize()
# t0 = time.time()
# train_step_no_sync(xb, yb)
# torch.cuda.synchronize()
# compile_s = time.time() - t0
# print(f"the first compiled step took {compile_s:.1f} s")
#
# # The loop, against Round 2b.
# step_time_r3 = time_step(num_workers=NUM_WORKERS, step_fn=train_step_no_sync)
# print()
# print(f"the compiled loop takes {step_time_r3:.2f} ms/step, against {step_time_r2b:.2f} before")
# gain_ms = step_time_r2b - step_time_r3
# if gain_ms > 0:
#     print(f"that is {gain_ms:.2f} ms/step, which would pay the compile back after "
#           f"{compile_s * 1e3 / gain_ms:,.0f} steps -- but read the note below first")
# else:
#     print("the loop did not get faster at all, so the compile never pays for itself here")
#
# # The profile: stage table, kernel count, trace.
# with profile(
#     activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
#     schedule=schedule(wait=1, warmup=2, active=N_ACTIVE, repeat=1),
# ) as prof_r3:
#     train_epoch(make_loader(num_workers=NUM_WORKERS), prof=prof_r3,
#                 step_fn=train_step_no_sync)
#
# save_trace(prof_r3, "ep2_r3.json")
# print()
# stage_time_r3, stage_total_time_r3 = show_stages(prof_r3, "after torch.compile")
# print()
# print(f"CUDA kernels per step: {count_kernels(prof_r2b)} before, "
#       f"{count_kernels(prof_r3)} after")
#
# # The step alone again, measured exactly as above: the clean before and after.
# step_time_ideal_r3 = timeit(lambda: train_step_no_sync(xb, yb), warmup=5, iters=30)
# print()
# print(f"the step alone: {step_time_ideal_r2b:.2f} ms eager -> {step_time_ideal_r3:.2f} ms compiled "
#       f"({step_time_ideal_r2b / step_time_ideal_r3:.2f}x)")
# print(f"the loop:       {step_time_r2b:.2f} ms eager -> {step_time_r3:.2f} ms compiled "
#       f"({step_time_r2b / step_time_r3:.2f}x)")
# ```
#
# **Why the speedup is so little.** The compiler fused the elementwise chains, not the matmuls, and the
# matmuls are where the forward and backward time is. Step 2 should have told you that
# before you measured anything.
#
# Note that torch.compile() does not compile the optimizer, you could try to compile it too to see if it helps.
#
# </details>

# %% [markdown]
# ---
# ## Summary
# ```
#   ⏱️ MEASURE  ->  📊 PROFILE  ->  🔍 DIAGNOSE  ->  🔧 OPTIMISE
#       ^                                                   |
#       └────────────────  ✅ VERIFY ───────────────────────┘
# ```
#
# Round 1 moved data loading to worker processes. Round 2 found the bottleneck had moved
# to the metric and took its two defects one at a time: the per-sample loop, then the
# `.item()` that stopped the CPU every step. Round 3 was yours.
#
# Three rules hold the method together:
#
# * **A diagnosis ends with a number** — measured by timing the loop with the stage taken
#   out, not computed from a share of CPU labels.
# * **A fix counts only if the stopwatch says so.** Round 2b bought time without removing a
#   single kernel; Round 3 removed many of them and bought little.
# * **Every fix moves the bottleneck**, so VERIFY profiles as well as measures, and that
#   profile starts the next round.
#
# ### A useful checklist
#
# 1. **Is the GPU row solid, or does it have gaps?** Solid: compute-bound, work on the
#    model. Gaps: something else sets the pace.
# 2. **If there are gaps, is the CPU busy at that moment?** Busy: the work before the GPU
#    is too slow, usually data loading or Python. Idle too: a synchronisation, usually
#    `.item()`, `.cpu()` or printing a tensor.
# 3. **Are the kernels many and small, or few and large?** Many and small: launch overhead
#    instead of arithmetic. Look for Python loops over samples.
# 4. **Does one stage own most of the time?** Then it is the only stage worth touching, and
#    its share caps what a fix can buy.
# 5. **Did ms/step actually change after the fix?**  Measure again, or you do not know.
