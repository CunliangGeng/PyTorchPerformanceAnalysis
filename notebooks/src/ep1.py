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
# # Episode 1: Measure and profile PyTorch GPU work
#
# In this notebook you know what the code does. The goal is to learn what the 
# timing and profiling *tools* say about it. Later, when you do not know what a piece of code
# does, you will still be able to read the answer from the same tools.

# %% [markdown]
# ## SETUP — run this first
#
# This cell checks that you have a GPU and runs one fixed benchmark: a 4096x4096 matrix
# multiplication (matmul). The notebook uses that result to scale every "expected" range below to
# your hardware.

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
# ## Section 1. How to correctly time GPU work?
#
# In this section we will take a matrix multiplication (matmul) as an example and learn how to correctly measure its execution time on the GPU.

# %% [markdown]
# ### The naive timer
#
# The next cell multiplies two 4096x4096 matrices on the GPU. It times the multiplication with `time.time()`, the way you
# would time any Python code:
#
# ```python
# t0 = time.time()
# c = a @ b
# t1 = time.time()
# ```
#
# 🔻**Question**: what is the problem with this timing?

# %%
a = torch.randn(4096, 4096, device=DEVICE)
b = torch.randn(4096, 4096, device=DEVICE)

naive_runs = []
for _ in range(5):
    t0 = time.time()
    _ = a @ b
    t1 = time.time()
    naive_runs.append((t1 - t0) * 1e3)

naive_ms = sum(naive_runs) / len(naive_runs)
expect("Naive timer says the matmul took", naive_ms, 0.01, 0.5)

# %% [markdown]
# ### Timing the same matmul, timed correctly
#
# This cell warmups the GPU by running the matmul a few times before timing it. This is important because the first time you run a GPU operation, there may be additional overhead for initializing the GPU context and loading kernels.
#
# More importantly, the naive timer does not account for the fact that GPU operations are asynchronous. When you call `a @ b`, it launches the operation on the GPU and returns immediately, without waiting for the operation to finish. Therefore, when you read the time right after the matmul, it may not have completed yet. To get an accurate measurement, you need to synchronize the CPU and GPU before reading the end time using `torch.cuda.synchronize()`. This ensures that the CPU waits for the GPU to finish all its work before measuring the elapsed time.

# %%
# warm up first
for _ in range(5):
    _ = a @ b
torch.cuda.synchronize()

synced_runs = []
for _ in range(5):
    t0 = time.time()
    _ = a @ b
    torch.cuda.synchronize()    # wait for the GPU to finish before reading the clock
    synced_runs.append((time.time() - t0) * 1e3)

synced_ms = sum(synced_runs) / len(synced_runs)
expect("With warm up and `torch.cuda.synchronize()`, the matmul took", synced_ms, 4 * SCALE, 40 * SCALE)
print()

print(f"The naive timer understated the cost by {synced_ms / naive_ms:.0f}x.")


# %% [markdown]
# #### 🛠 Task: time other fundamental operations
#
# Use the following helper function `timeit` to time other fundamental operations of Pytorch, such as:
# - `a + b` (element-wise addition)
# - `a * b` (element-wise multiplication)
# - `a.relu()` (ReLU activation)
#
#
# Compare the time of these operations with matrix multiplication. You can also try timing other operations that you are interested in.

# %%
# helper function to time GPU work correctly

def timeit(fn, warmup=3, iters=10):
    """Time a GPU function correctly. Returns milliseconds per call.
    """
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    t0 = time.time()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()

    return (time.time() - t0) / iters * 1e3


# %%
# Write your code here

# %% [markdown]
# ---
# ## Section 2. How to profile GPU work?
#
# Timing tells you *that* something is slow. The profiler tells you *what* is slow, and
# *when*. 
#
# This section profiles the matmul and reads the result in two ways: first as a timeline in Perfetto (https://ui.perfetto.dev/), then as the `key_averages()` table.

# %% [markdown]
# ### Profile the matmul with `torch.profiler`
#
# `torch.profiler.profile` records every PyTorch operator and every CUDA kernel while it is
# active. Its argument `activities` says what to record: CPU-side operators, GPU-side kernels, or both.

# %%
from torch.profiler import ProfilerActivity, profile

a = torch.randn(4096, 4096, device=DEVICE)
b = torch.randn(4096, 4096, device=DEVICE)

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
) as prof:
    _ = a @ b
    prof.step()

# %%
# helper function to save the trace and offer it for download

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


# %%
save_trace(prof, "ep1_matmul.json")

# %% [markdown]
# ### View the trace in Perfetto
#
# Go to **<https://ui.perfetto.dev>** in your browser and drag the file you just downloaded onto the page.
# Nothing is uploaded anywhere: Perfetto runs entirely inside your browser.
#
# Shortcuts to navigate the trace: 
# - **W / S** zoom in and out, 
# - **A / D** move left and right, 
# - **click** a block to see its name and duration in the panel at the bottom,
# - **Q** to show and hide the info panel at the bottom, 
# - **drag** across the timeline to measure a distance
#
#
#
# What you see in Perfetto is a timeline. Time runs from left to right, and the rows are grouped by
# process:
#
# * **CPU**: one row for the Python thread. Its blocks are the profiler steps, the PyTorch operators (`aten::...`) and the CUDA runtime calls
#   (`cudaLaunchKernel`). A block drawn under another block was called by it.
# * **GPU**: one row per CUDA stream. Its blocks are the kernels that actually ran.
#
#
# 💡 Two things to notice:
#
# 1. **Nesting.** On the CPU row, `aten::matmul` contains `aten::mm`, which contains
#    `cudaLaunchKernel`. The `cudaLaunchKernel` launches the CUDA kernel `volta_sgemm_128x64_nn` on the GPU.
#
#     The call path is roughly:
#     ```bash
#     Python: a @ b
#     ↓
#     aten::matmul           # generic matrix-multiplication PyTorch ATen operator
#     ↓
#     aten::mm               # specialized 2-D matrix-by-matrix ATen operator
#     ↓
#     cudaLaunchKernel       # CPU-side CUDA runtime call: enqueue work, then return
#     ↓
#     volta_sgemm_128x64_nn  # GPU run this CUDA kernel to perform the multiplication
#     ```
#
# 2. **The long block at the end of the CPU row** is `cudaDeviceSynchronize`: the profiler
#    waiting for the GPU to finish everything it recorded. That is the same wait as
#    `torch.cuda.synchronize()` in INVESTIGATE 1, drawn as a block.
#
#

# %% [markdown]
# #### 🛠 Task: compare traces with and without stack
#
# What happens if you turn on `with_stack=True`? The next cell does that.
#
# Compare the two traces in Perfetto. Answer these questions:
#
# - How did the file size change?
# - What extra information can you find for an operator?
# - Did the GPU timeline change, or did the profiler mostly add metadata and CPU overhead?
#
# Note: You need to open a new tab for Perfetto https://ui.perfetto.dev/ and drag the new trace file onto it.

# %%
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    with_stack=True, # Turn on recording of stack
) as perf:
    _ = a @ b
    perf.step()

save_trace(perf, "ep1_matmul_with_stack.json")

# %% [markdown]
# ***
# ### Profile with schedule and labelling
#
# `schedule` says *which loop iterations* to record:
# - `wait=1` skips the first iteration which is slow because it warms up the GPU,
# - `warmup=1` runs the profiler on the second but throws that data away, warming up the profiler itself,
# - `active=2` keeps the third and fourth,
# - `repeat=1` runs the whole schedule once
#
# `record_function` is a context manager that labels a block of code in the trace.

# %%
from torch.profiler import schedule, record_function
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=schedule(wait=1, warmup=1, active=2, repeat=1),
) as prof:
    for _ in range(4):
        with record_function("## 0-matmul ##"): # Label the matmul in the trace
            y = a @ b
        prof.step()

save_trace(prof, "ep1_matmul_scheduled.json")

# %% [markdown]
# #### 🛠 Tasks: Profile a matmul and a ReLU
#
# The matmul is `y = a @ b` and the ReLU is `_ = y.relu()`.
#
# - Subtask 1. Write code in the following cell to profile a matmul and a ReLU, labeling each with `record_function()`, save the trace as `ep1_matmul_relu.json`, and view it in Perfetto.
# - Subtask 2. Based on the subtask 1, add `torch.cuda.synchronize()` after matmul and ReLU, profile the code, and save the trace as `ep1_matmul_relu_sync.json`. Compare the two traces in Perfetto. What is different? Why?
#

# %%
# Subtask 1: Profile a matmul and a ReLU, labeling each with `record_function()`, save the trace as `ep1_matmul_relu.json`


# %%
# Subtask 2: add `torch.cuda.synchronize()` after matmul and ReLU, and save the trace as `ep1_matmul_relu_sync.json`


# %% [markdown]
# ---
# ### Read the `key_averages()` table (Optional)
#
# The timeline shows *when* things ran. The table shows *how much* each kind of event cost
# in total. `prof.key_averages()` groups every recorded event by name and adds up its times;
# `.table(sort_by=..., row_limit=...)` prints the result. The columns you need:
#
# | Column | Meaning |
# |---|---|
# | *Self CPU* | time the CPU spent in this event itself, not in the events it called |
# | *CPU total* | the same, including the events it called |
# | *Self CUDA*, *CUDA total* | the same two numbers for the GPU |
# | *# of Calls* | how many times the event was recorded |

# %%
# profile matmul with schedule and labelling
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=schedule(wait=1, warmup=1, active=2, repeat=1),
) as prof:
    for _ in range(4):
        with record_function("## 0-matmul ##"):
            y = a @ b
        prof.step()

# %%
# Sort by `self_cuda_time_total` to see what the GPU spent time on
print("=" * 26, "sorted by SELF CUDA  (what the GPU spent time on)", "=" * 26)
print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=20))


# %%
# Sort by `cpu_time_total` to see what the CPU spent time on
print("=" * 26, "sorted by CPU TOTAL", "=" * 25)
print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))

# %% [markdown]
# 💡 Things to notice:
#
# Two tables, the same recording. The first is sorted by a *self CUDA* column, the second by a
# *CPU total* column. 
#
# **1. There are two time families, and they disagree.** In the first table, find the
# `aten::mm` row at the top. Compare its *Self CPU* with its *Self CUDA*. The CPU time is
# about a hundred microseconds; the CUDA time is tens of milliseconds. That is Section 1
# again, now as two numbers in one row: the CPU spent microseconds *asking*, the GPU spent milliseconds *doing*.
#
# **2. Self time and total time.** *Self* is the time spent in that event alone. *Total*
# also includes the events it called. Find `aten::matmul` in the **second** table. Its
# *Self CUDA* is `0.000us` and its *CUDA total* is tens of milliseconds, because
# `aten::matmul` runs nothing on the GPU itself: all the work is in its child `aten::mm`,
# which has the same CUDA total and a large self time. That is why `aten::matmul` is nowhere near the 
# top of the first table: sorted by self CUDA time it ranks zero. 
#
# **3. The `# of Calls` column.** Most rows say 2: one per recorded step. The matmul kernel
# row says 3 (check yours). The extra one is the warm-up step's matmul: it was launched
# before recording started, but it ran on the GPU after recording started, because the GPU
# was running behind the CPU. 
#
#
#

# %% [markdown]
# #### 🛠 Task: try other sort keys 
#
# Other available sort keys are:
# - `cuda_time_total`
# - `self_cpu_time_total`
# - `count`
#
# Try sorting by each of these keys and explore what questions they can answer.

# %% [markdown]
# ---
# ## Summary
#
# You have learned how to time and profile GPU work in PyTorch. You have seen that timing is not as simple as it seems, and that profiling can give you a lot of information about what is happening on the CPU and GPU. You have also learned how to read the profiler's timeline and table outputs, and how to use them to understand the performance of your code.
