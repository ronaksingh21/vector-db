# vector-db

An HNSW vector search implementation in C++ with Python bindings, built and benchmarked with a specific focus other libraries skip: real, measured performance on constrained edge hardware, not just server-class machines.

Most ANN libraries (faiss, hnswlib, annoy) are tuned and benchmarked assuming AVX2, large caches, and server memory budgets. If you're building on-device search for robotics, IoT, or a local-first app running on something like a Raspberry Pi, none of them tell you what to expect. This project does.

## Why this exists

- **int8 quantization** cuts memory 4x with a measured, documented recall cost, not a guess.
- **Verified on real Raspberry Pi 5 hardware.** Every number below was measured on-device, not extrapolated from x86.
- **`target_device` presets are backed by an actual parameter sweep**, not arbitrary defaults with a device name attached. See the sweep results below for the reasoning.

## Quickstart

```bash
pip install vector-db
```

```python
from vector_db import HNSWIndex

# Presets tuned from a real M / ef_construction sweep on Pi 5 hardware
index = HNSWIndex(target="raspberry-pi")  # or target="server"

index.insert(1, [0.1, 0.2, 0.3, 0.4])
index.insert(2, [0.9, 0.8, 0.7, 0.6])

results = index.search([0.1, 0.2, 0.3, 0.4], k=2)
```

## Benchmarks

All numbers below use a 100,000-vector subset of SIFT1M (128 dimensions).

### Three-way comparison

| | x86 float32 | x86 int8 | Pi 5 int8 |
|---|---|---|---|
| Recall@10 | 96.7% | 89-93% | ~92% (90.7-93.0% across 3 runs) |
| Query latency | 4.4-4.9ms | ~4.0ms | ~7.0ms (6.6-7.22ms across 3 runs) |
| Storage | 48.8 MB | 12.2 MB | 12.2 MB |

The float32 code path was removed from `main` when quantization landed. It remains fully reproducible at tag [`v1.0-float32`](../../releases/tag/v1.0-float32) if you want to verify or build on the unquantized version.

Build time on the Pi varied more than query performance across repeated runs (170-216s for the default config), most likely due to thermal conditions rather than any property of the algorithm. Query latency and recall were stable across the same runs.

One self-query consistency check occasionally reports a single-sample mismatch on both x86 and the Pi. A 100-sample aggregate consistently shows a 94-100% hit rate, consistent with expected variance at this scale rather than a real correctness bug.

### Honest comparison against pgvector

Benchmarked against a tuned pgvector installation on the same x86 machine: pgvector wins on both build time (11s vs. our 114s) and query latency (1-3ms vs. our 4.4-4.9ms). This implementation wins on recall (96.7% vs. pgvector's baseline). pgvector is a mature, SIMD-optimized C implementation with a flat memory layout; this project uses a `std::map`-based storage layer, which is the primary reason for the gap. This comparison is not softened or hidden because the point of publishing it is to be a real reference, not a marketing page.

## The M / ef_construction sweep

A full sweep of `M` (max connections per node) and `ef_construction` (search width during index build) was run on the Pi 5, holding int8 quantization on and `max_layers` at 16:

| M | ef_construction | Build time | Query latency | Recall@10 |
|---|---|---|---|---|
| 4 | 100 | 77.5s | 3.69ms | 64.7% |
| 4 | 200 | 143.9s | 6.03ms | 89.7% |
| 4 | 400 | 294.3s | 11.76ms | 93.3% |
| 8 | 100 | 136.3s | 4.08ms | 89.4% |
| 8 | 200 | 244.0s | 11.71ms | 87.6%* |
| 8 | 400 | 481.0s | 20.65ms | 94.7% |
| 16 | 100 | 239.3s | 6.50ms | 95.1% |
| 16 | 200 | 418.5s | 11.44ms | 95.4% |
| 16 | 400 | 780.2s | 32.05ms | 95.5% |
| 32 | 100 | 479.3s | 9.94ms | 95.5% |
| 32 | 200 | 785.8s | 17.88ms | 95.5% |
| 32 | 400 | 1333.5s | 45.22ms | 95.5% |

*This point is lower than the M=8/ef=100 result immediately above it. Given the 100-query sample size and the self-query variance already documented above, this is treated as noise rather than a real regression.

**The finding that matters:** recall plateaus at approximately 95.5% once `M` reaches 16, regardless of how much higher `ef_construction` goes. `M=32, ef_construction=400` takes 1,333 seconds to build and 45.2ms per query, for the same recall `M=16, ef_construction=100` reaches in 239 seconds and 6.5ms per query. Past that plateau, additional `M` or `ef_construction` buys nothing but slower builds and slower queries.

## `target_device` presets

Built directly from the sweep above, not from guessed or default values:

| Preset | M | ef_construction | Rationale |
|---|---|---|---|
| `raspberry-pi` | 8 | 100 | Prioritizes fast build and low query latency on constrained hardware. Accepts ~89% recall as the tradeoff. |
| `server` | 16 | 100 | Recall plateaus at M=16, so there is no measured benefit to M=32 on this workload. Gets the same recall ceiling with more headroom than the Pi preset. |

Both presets currently use int8 quantization, since `main` does not have a float32 code path (see the tag note above). `max_layers` was not varied in the sweep and is left at the library default (16) for both presets.

## Building from source

Requires CMake 3.10+, a C++17 compiler, and pybind11.

```bash
git clone https://github.com/ronaksingh21/vector-db.git
cd vector-db
pip install .
```

To build the standalone C++ executables directly instead of the Python package:

```bash
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make
```

## Running the benchmark yourself

```bash
./build/benchmark data/sift/sift_base.fvecs data/sift/sift_query.fvecs [max_layers] [M] [ef_construction]
```

Arguments are optional and fall back to environment variables (`HNSW_MAX_LAYERS`, `HNSW_M`, `HNSW_EF_CONSTRUCTION`, `SIFT_BASE_PATH`, `SIFT_QUERY_PATH`), then to library defaults (16, 5, 200), in that order.

## Known limitations

- `main` is int8-only. There is no runtime toggle back to float32; use the `v1.0-float32` tag if you need the unquantized version.
- The `server` preset is not yet a fully separate hardware profile. It currently differs from `raspberry-pi` only in `M` and `ef_construction`, both derived from the same single sweep. No separate tuning has been done for concurrent query load or larger memory budgets.
- The sweep was run once per configuration on a single Pi 5 unit. Recall and latency showed some run-to-run variance in earlier repeated tests (see benchmarks section); the sweep table above reflects single runs per configuration, not averages.
- Storage figures reflect raw vector storage only, not graph edge overhead.

## License

MIT
