# AI NVMe Accelerator 8X

Experimental Windows project for studying storage-backed GGUF inference. The project does not promise an 8X speedup or 24 tokens/s. Disk and inference results must be measured on the target workload.

## Stage 1 status

Implemented a read-only hardware probe and a bounded file-read benchmark. The probe collects CPU, Windows version, Windows-visible memory, disks, volumes, pagefile usage, and display adapters. The disk benchmark never creates or modifies files; it reads up to 1 GiB sequentially and performs at most 4096 random 4 KiB reads from a file the user explicitly selects.

Hardware and storage results vary by machine; run the diagnostic commands below to collect measurements for your system.

Stage 2 adds `python main.py monitor --samples 5 --interval 1`: read-only samples of physical RAM, system commit, current pagefile usage, and free space on the system volume. One authoritative threshold set from `config/settings.json` is used throughout the project: WARNING below 2 GiB available or at 85% RAM/commit; CRITICAL below 512 MiB available or at 95% RAM/commit. The Guard never kills processes or modifies the pagefile.

Model weights are not included in this repository. Supply a local GGUF path when running model commands.

## Run in PowerShell

```powershell
cd C:\path\to\AI-NVME-ACCELERATOR-8X
python main.py diagnose
python main.py monitor --samples 5 --interval 1 --budget-gib 3 --reserve-mib 512
python main.py disk-read "$env:USERPROFILE\Downloads\model.gguf" --max-mib 256
python -m benchmark.disk_benchmark "$env:USERPROFILE\Downloads\model.gguf" --max-mib 256
python -m unittest discover -s tests -v
```

The disk command requires a real existing file. Prefer a GGUF on the target NVMe. Its output is also appended to `logs/performance.json`. A sequential block-size sweep is available with `python main.py disk-sweep "$env:USERPROFILE\Downloads\model.gguf" --mib-per-block 32 --uncached`; it uses Windows aligned `FILE_FLAG_NO_BUFFERING` reads and is capped at 320 MiB total. The ordinary `disk-sweep` mode leaves Windows caching on and is explicitly exploratory.

Stage 4 adds an experimental bounded LFU/LRU byte-block cache (`engine/cache_manager.py`). Its capacity is capped at 64 MiB and 2048 entries, accounts 512 bytes per entry for metadata, and turns off entirely while the memory guard reports warning or critical pressure. Run `python -m benchmark.cache_benchmark` for a small synthetic behavior check; it does not touch the SSD or predict inference speed.

Stage 5 adds a one-worker, one-block asynchronous sequential prefetcher (`engine/prefetch_engine.py`). It requires two matching positive strides, is limited to aligned blocks up to 4 MiB, and refuses to queue reads unless the memory guard says `normal` with the reserve still available. Its `take()` call is nonblocking by default. Do not judge prefetch benefit until a normal-pressure run records useful versus wasted prefetched bytes and end-to-end inference latency.

Stage 6 adds a guarded `llama-cpp-python` inference wrapper with `use_mmap=True`, `use_mlock=False`, CPU-only `n_gpu_layers=0`, configurable context/batch, and up to 4 threads. New model loads and generations now require the Memory Guard to report NORMAL and enough RAM headroom after the configured reserve. It enforces one loaded model and serializes generation calls. Memory-mapped weights still consume RAM as Windows pages them in; inference may be slow or refused when the working set exceeds available memory. `llama-cpp-python`'s official docs describe `use_mmap` and Windows' compiler/wheel options: [API](https://github.com/abetlen/llama-cpp-python/blob/main/llama_cpp/llama.py), [installation](https://pypi.org/project/llama-cpp-python/).

The inference wrapper now has a load-only diagnostic path. It records system RAM, commit, process working set/private bytes and Guard transitions every 250 ms, while a separate Windows PSAPI monitor counts resident pages belonging to the GGUF mapping about once per second. llama.cpp native load output is captured separately so its model, KV-cache and compute-buffer sizes can be compared with the OS residency counts. These diagnostics do not generate tokens; the load profiler still obeys the shared Memory Guard and closes the model after a pressured load completes.

To explicitly run the load-only profile (no inference), use `python main.py accelerator profile-load --run`. Samples append to `logs/model_load_profiles.jsonl`; native buffer/allocation messages append to `logs/llama_native_load_*.log`. It may refuse to begin unless the shared Guard is NORMAL.

Create a project-local virtual environment and install the CPU package (uses a prebuilt CPU wheel when one matches the interpreter; source builds on Windows need Visual Studio or MinGW):

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install --only-binary=:all: -r requirements-inference.txt --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
```

Run the bounded inference measurement after installation:

```powershell
.\.venv\Scripts\python.exe -m benchmark.inference_benchmark --model "$env:USERPROFILE\Downloads\model.gguf" --context 512 --batch 16 --threads 4 --max-tokens 32
```

Performance depends on the model, hardware, available memory, and workload. Use the guarded benchmark on your target system; a single run is not a stable performance estimate.

## Safety and measurement limits

- No settings, pagefile, services, BIOS, RAID/AHCI, or personal files are changed.
- Existing-file disk benchmarks are read-only; the opt-in NVMe I/O benchmark creates only bounded temporary files inside the project to measure reads, never write performance.
- Windows' file cache is not flushed or bypassed. Results can reflect cached data.
- No synthetic write benchmark is included.
- RAM available at one instant is not the same as physical capacity or a safe model budget.
- Model runs are gated by the live memory monitor and may be refused when memory pressure is high.

## Memory Guard for Ollama

The `memory_guard/` package adds a conservative admission gate around Ollama chat requests. It reads Windows RAM and system commit through `GlobalMemoryStatusEx` (no `Get-Counter` or third-party package), checks `ollama ps`, and refuses a new request unless memory is `NORMAL` and no model is already resident. One shared threshold set is configured in `config/settings.json`: warning below 2 GiB available / 85% RAM or commit; critical below 512 MiB / 95%. WARNING also blocks new requests by default to avoid starting discretionary work under pressure.

Start the low-frequency terminal dashboard (2-second refresh):

```powershell
python main.py memory-guard
```

Run one guarded local Ollama chat request; it is rejected if memory is WARNING/CRITICAL, Ollama is offline, or any model is already loaded:

```powershell
python main.py ollama-chat --model qwen3:4b --prompt "Explain in three sentences how an NVMe SSD works."
```

Thresholds are configured centrally in `config/settings.json`. A monitoring thread samples while a request is in progress. If pressure becomes critical, it records that event and does not terminate the request or kill any process. After the call returns, check resident models with `ollama ps`; an explicit unload request is available only with a confirmation that no inference is active:

```powershell
ollama ps
python main.py ollama-stop --model qwen3:4b --confirm-idle
```

Run `python -m unittest discover -s tests -v` for simulated memory/Ollama cases. Completed, blocked, and failed `ollama-chat` attempts are recorded in `reports/memory_report.json` and `reports/memory_history.csv`; Memory Guard does not modify the existing `logs/performance.json`.

Process Monitor extends the same dashboard with read-only per-process metrics for executable names containing `ollama`: PID, working set, private working set when supported, private commit, virtual address-space size when the native query is available, thread count, and uptime. It collects no command lines. Every dashboard sample is appended to `logs/process_monitor.json` (JSON array) and `logs/process_monitor.csv`; history is appended without loading the existing JSON history into memory. A high private-memory total with Ollama online and no model is labeled `MEMORY_RESIDUAL_SUSPECTED` only as a diagnostic hypothesis; it does not establish a leak. Windows' `PROCESS_MEMORY_COUNTERS_EX2` supplies private working set where supported; virtual size is queried separately and may be unavailable. See [PROCESS_MEMORY_COUNTERS_EX2](https://learn.microsoft.com/en-us/windows/win32/api/psapi/ns-psapi-process_memory_counters_ex2) and [Toolhelp process walking](https://learn.microsoft.com/en-us/windows/win32/toolhelp/process-walking).

## Adaptive Cache Engine 1.0

The new `adaptive_cache/` package reuses the existing bounded byte cache and single-worker prefetcher, with the existing Memory Guard as the only memory-state authority. Configuration is in `config/adaptive_cache.json`. Defaults cap RAM cache at 64 MiB and 5% of physical memory, apply a 1 GiB Windows reserve and an extra conservative process-working-set adjustment, reduce the WARNING cache to at most 1 MiB, and set CRITICAL capacity to zero. Missing/failed guard or process readings fail closed.

The persistent cache stores only caller-provided immutable byte values inside `cache/nvme/`; it never accesses, moves, or replaces model files. SQLite maps a hashed key to a SHA-256 content blob; duplicate content shares one blob and reads verify its hash. The configured NVMe limit is 2 GiB including the index, items are limited to 8 MiB, entries expire after seven days, and writes are refused if they would leave less than 10 GiB free. LRU, LFU, and HYBRID are configurable; HYBRID is the default. Eviction and expiration remove only validated hash blobs in the cache directory.

Prefetch is disabled by default. When enabled, it reuses the existing single-worker predictor, runs only in NORMAL, is limited to 64 KiB at a time, and rejects paths outside the cache directory or any source other than a `.blob` entry. It never prefetches GGUF/model files. WARNING disables SSD reads/writes and all prefetch; CRITICAL also clears/reduces owned RAM cache entries and blocks new cache data. Cache hit rates and I/O counters are observational; no speedup is assumed.

The existing dashboard now shows cache RAM/SSD use, hit/miss rate, free disk space, and prefetch state:

```powershell
python main.py memory-guard
```

Adaptive Cache logs append to `logs/adaptive_cache.json` and `logs/adaptive_cache.csv`; all previous logs remain separate. Synthetic tests are run with `python -m unittest tests.test_adaptive_cache -v`; the complete suite uses `python -m unittest discover -s tests -v`.

## NVMe I/O Engine 1.0

`nvme_io/` performs storage-only measurements, separate from inference. Device information is read-only. The CLI defaults to device information and never benchmarks unless `--benchmark` is explicitly passed:

```powershell
python main.py nvme-io --info
python main.py nvme-io --drive C:
```

The explicitly requested benchmark creates one temporary file at a time under `benchmark/nvme_io_runs`, with configured sizes 16, 64, and 256 MiB, a 336 MiB total-write ceiling, a 1 GiB free-space reserve, and an inter-process lock to prevent simultaneous runs. It performs chunked sequential reads, an immediate repeated pass, and seeded random reads in 4 KiB and 64 KiB blocks. WARNING reduces the largest file to 16 MiB and caps random reads at 16 operations; CRITICAL or unavailable memory state blocks the run. A memory-state change during a run stops additional I/O. Files are exclusively created, marked as owned, and removed only after path/marker validation; reports remain on failure.

Windows cache flushing is never attempted. The first pass is labeled cache-unknown, and the repeated pass may be served by Windows' file cache. Device bus type is reported as Windows provides it; a RAID label is explicitly inconclusive about an underlying NVMe device. Results append to `logs/nvme_io.json` and `logs/nvme_io.csv`; `reports/nvme_io_report.txt` describes the latest run. The existing Memory Guard panel shows the latest logged I/O measurement. No inference is started and no token/s metric is derived.

To run the physical SSD benchmark, use `python main.py nvme-io --benchmark` only after explicitly deciding to perform that measurement. The implementation/test phase does not invoke it. `python main.py nvme-io --info` queries device metadata only.

## AI NVMe Accelerator Integration 1.0

The `accelerator/` package coordinates the current model path. `accelerator status` and `accelerator plan` provide read-only status/planning; the `memory-guard` dashboard also shows its admission decision. Configuration is in `config/accelerator.json`.

```powershell
python main.py accelerator status
python main.py accelerator plan
```

The model streaming implementation delegates file-backed weight access to `llama.cpp` through `use_mmap=True` and keeps `use_mlock=False`. This avoids an eager Python copy of the full GGUF; it is not a custom chunk loader. Windows remains responsible for page faults and working-set residency. `config/accelerator.json` sets `use_extra_bufts=false`, disabling llama.cpp's optional CPU weight-repacking buffers through the binding's native model-parameter default. This may reduce CPU inference throughput and does not stop mapped GGUF pages from becoming resident as generation touches the weights. The scheduler consults Memory Guard, blocks new model work in WARNING/CRITICAL or when the memory probe is unavailable, requires the configurable 2 GiB minimum available-RAM startup threshold, preserves a configurable 1 GiB reserve, and selects bounded context, batch, and CPU thread settings from available headroom. Below 4 GiB available, it uses context 128, batch 4, one CPU thread, and no prefetch. If pressure rises to WARNING/CRITICAL while loading completes, the loaded object is closed before generation starts. A running generation is observed but not forcibly terminated; later requests are blocked by the admission gate. File-backed mmap is not a guarantee that the model will fit or run successfully under current system pressure.

The memory observer reports system RAM minima and process working set/private bytes, page-fault count delta, and process read-I/O counters during load/generation. Process-wide I/O and page faults are not model-file-only measurements. `Adaptive Cache` remains separate and does not duplicate weight blocks. `Predictive Prefetch 2.0` reports whether headroom would permit prefetch, but does not issue speculative GGUF reads: the installed Python binding does not expose stable mapped tensor addresses for safe, targeted prefetch. llama.cpp/Windows internal mmap read-ahead behavior is not controlled by this Python layer.

Generation also samples Windows PDH `PhysicalDisk` counters every 250 ms for device read throughput, read operations, and average read latency. These whole-device counters include unrelated background traffic; byte/operation totals are integrated estimates and latency percentiles are unavailable. They do not attribute physical reads to the GGUF or measure time blocked on storage. An explicit single-inference action can capture WPR DiskIO/FileIO ETW and export it with `tracerpt`: from an elevated PowerShell, run `python main.py accelerator infer --run --capture-etw --max-tokens 16 --prompt "Responda somente: teste do modelo."`. The trace is whole-system and writes an ETL/CSV under `logs`; it may include unrelated file paths and its own trace-write I/O. Current capture is not enabled by default. Current instrumentation details are in `reports/ssd_io_measurement_instrumentation.txt`.

The comparison engine is sequential and logs to `logs/accelerator_runs.jsonl`; it compares a configured mmap baseline with the adaptive context/batch/thread profile. It has one repetition, so the result is exploratory, not a validated benchmark. It is never launched by `status` or `plan`; actual model loading and generation require the explicit command:

```powershell
python main.py accelerator compare --run --prompt "Responda somente: teste concluído."
```

Each profile rechecks Memory Guard before loading and generating. No 8× speedup is assumed. The reported comparison does not claim that SSD-backed mmap itself accelerates token generation.

## Planned stages

1. Hardware diagnostics and safe storage baseline — completed.
2. RAM, virtual memory, and storage monitor — implemented and logged.
3. Read benchmark refinements — implemented. Physical read measurements require explicit invocation.
4. Adaptive Cache Engine 1.0 — implemented with Memory Guard policy, bounded content-addressed NVMe cache, three eviction strategies and optional constrained prefetch; not connected to inference workloads or benchmarked.
5. Predictive prefetch — implemented; live model test deferred while memory pressure is warning.
6. llama.cpp integration — implemented with guarded CPU/mmap loading, plus bounded memory observation during model loading and generation; actual Q4_K_M CPU/mmap load and short generation measured once. Peak memory pressure reached critical.
7. Qwen2.5-Coder-7B-Instruct Q4_K_M GGUF — inference uses the guarded benchmark; results depend on the target system.
8. Memory Guard for Ollama admission, dashboard and JSON/CSV logging — implemented; simulated tests pass. Repeated live inference and cache/prefetch comparisons remain gated on safe memory headroom.
9. Compare cache/prefetch configurations and inference tokens/s.
10. NVMe I/O Engine 1.0 — implemented and simulated tests pass; physical SSD measurement remains pending explicit authorization.
11. Accelerator Integration 1.0 — mmap streaming delegation, Windows-managed paging observations, RAM-aware runtime scheduler, guarded llama.cpp integration, prefetch policy reporting, optimizer planning, explicit sequential comparison command, and dashboard status implemented. Targeted tensor paging/prefetch remains unavailable through the Python binding; physical/model comparison has not been run in this implementation step.

The model is never silently replaced with a smaller model. SSD-backed mmap can reduce eager loading and allow the OS to page file-backed weights, but it does not make SSD latency or bandwidth equivalent to RAM; inference may become extremely slow under page faults and Windows memory pressure.
