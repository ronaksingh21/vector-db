# vector-db

An HNSW vector search implementation in C++ with Python bindings, built and benchmarked with a specific focus other libraries skip: real, measured performance on constrained edge hardware, not just server-class machines.

Most ANN libraries (faiss, hnswlib, annoy) are tuned and benchmarked assuming AVX2, large caches, and server memory budgets. If you're building on-device search for robotics, IoT, or a local-first app running on something like a Raspberry Pi, none of them tell you what to expect. This project does.

## Why this exists

- **int8 quantization** cuts memory 4x with a measured, documented recall cost, not a guess.
- **A NEON-accelerated distance calculation** on ARM, not just a scalar loop that happens to compile.
- **Flat, cache-friendly storage** (two-level indexing: hash lookup by external ID, dense vector storage internally) instead of a tree-based map, which measurably improved both build time and query latency.
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

All numbers below use a 100,000-vector subset of SIFT1M (128 dimensions), default config (M=5, ef_construction=200) unless otherwise noted.

### Three-way comparison

| | x86 float32 | x86 int8 | Pi 5 int8 |
|---|---|---|---|
| Recall@10 | 96.7% | 89-93% | 91.5-93.5% (5 runs) |
| Query latency | 4.4-4.9ms | ~4.0ms | ~3.0-3.8ms (5 runs) |
| Build time | ~114s | not separately measured | ~83-85s (5 runs) |
| Storage | 48.8 MB | 12.2 MB | 12.2 MB |

The float32 code path was removed from `main` when quantization landed. It remains fully reproducible at tag [`v1.0-float32`](../../releases/tag/v1.0-float32) if you want to verify or build on the unquantized version.

These Pi numbers reflect the current implementation, after two optimization passes described below (flat storage layout and NEON-accelerated distance calculation) and a correctness fix to the graph's random layer assignment. Earlier benchmark runs, before these changes, showed a wider recall range (63-93%) with occasional low outliers; that instability is discussed below and has been resolved.

## What changed since the first Pi benchmark, and why the numbers moved

The initial edge-deployment benchmark used `std::map<int, Node>` for graph storage, which meant every node lookup during search involved a tree traversal rather than a direct memory access, and stored external IDs directly in neighbor lists rather than compact internal indices. Two changes fixed this:

1. **Storage**: replaced with `std::unordered_map<int, size_t>` (mapping external ID to a dense internal index) plus a flat `std::vector<Node>` for actual storage. Neighbor lists now store internal indices, not external IDs, so graph traversal during search never needs a hash lookup, only direct vector indexing. This is a pure storage-layer change: the public API (`insert(id, vector)`, `search(query, k)`) is unchanged, arbitrary/sparse IDs are still fully supported.

2. **Distance calculation**: `l2_distance` gained a NEON-accelerated path for ARM, processing 16 int8 lanes per iteration with widening subtract/multiply to avoid overflow, and a manual horizontal reduction for broad ARM compatibility. The original scalar loop is kept as the exact fallback on non-ARM builds, verified byte-identical in behavior there.

Together these roughly halved both build time and query latency on the Pi, with no measurable change to recall (as expected, since neither change touches the algorithm itself).

**A real correctness bug surfaced and was fixed during this work, and is documented here rather than glossed over.** The graph's random layer assignment (`get_random_layer`) was reseeding its random number generator on every single call, roughly once per inserted vector. This produced occasional badly-shaped graphs, which showed up as isolated benchmark runs with recall as low as 63-78%, against an otherwise stable ~90-93% baseline. The fix was seeding the RNG once per index instance instead. After the fix, recall is stable and tightly clustered across repeated runs, with no further low-recall outliers observed in over a dozen runs during verification.

**A second, non-code lesson from this process, also worth documenting honestly**: immediately after making the storage and NEON changes, a round of Pi benchmarks showed wildly unstable recall (as low as 63% and 73% in some runs) that initially looked like it could be a NEON correctness bug. Isolating NEON against a forced-scalar build on the same hardware showed both paths were actually fine; the real cause was a stale incremental build directory that had accumulated small inconsistencies across many `make` invocations following header and struct-layout changes. A full clean rebuild resolved it immediately. If you are modifying this codebase and see inexplicable, non-reproducible correctness issues after a structural change (changed struct layout, changed header), rule out a stale build directory before assuming a logic bug.

One self-query consistency check occasionally reports a single-sample mismatch even in the current, stable version. A 100-sample aggregate consistently shows a 94-100% hit rate, consistent with expected variance at this scale on a single specific query rather than a real correctness bug.

### Honest comparison against pgvector

Benchmarked against a tuned pgvector installation on the same x86 machine: pgvector wins on both build time (11s vs. our 114s) and query latency (1-3ms vs. our 4.4-4.9ms, pre-optimization numbers). This implementation wins on recall (96.7% vs. pgvector's baseline). pgvector is a mature, SIMD-optimized C implementation with a flat memory layout; this project's earlier `std::map`-based storage was the primary reason for the latency gap. The storage optimization described above narrows that gap somewhat but does not close it; pgvector remains faster on raw build and query time. This comparison is not softened or hidden because the point of publishing it is to be a real reference, not a marketing page.

## The M / ef_construction sweep

A full sweep of `M` (max connections per node) and `ef_construction` (search width during index build) was run on the Pi 5, holding int8 quantization on and `max_layers` at 16, using the current optimized implementation (flat storage, NEON, fixed RNG):

| M | ef_construction | Build time | Query latency | Recall@10 |
|---|---|---|---|---|
| 4 | 100 | 37.3s | 1.88ms | 70.9% |
| 4 | 200 | 76.1s | 2.65ms | 85.7% |
| 4 | 400 | 150.5s | 5.45ms | 91.7% |
| 8 | 100 | 61.7s | 2.08ms | 91.5% |
| 8 | 200 | 120.1s | 4.04ms | 94.7% |
| 8 | 400 | 241.3s | 9.47ms | 95.2% |
| 16 | 100 | 110.2s | 3.53ms | 94.9% |
| 16 | 200 | 215.6s | 7.35ms | 95.4% |
| 16 | 400 | 415.7s | 16.45ms | 95.5% |
| 32 | 100 | 214.8s | 5.89ms | 95.5% |
| 32 | 200 | 409.7s | 14.54ms | 95.5% |
| 32 | 400 | 743.8s | 25.65ms | 95.5% |

**The finding that matters:** recall plateaus at approximately 95.5% once `M` reaches 16, regardless of how much higher `ef_construction` goes. `M=32, ef_construction=400` takes 743.8 seconds to build and 25.65ms per query, for the same recall `M=16, ef_construction=100` reaches in 110.2 seconds and 3.53ms per query. Past that plateau, additional `M` or `ef_construction` buys nothing but slower builds and slower queries. This plateau held both before and after the storage/NEON optimizations; the optimizations made every configuration faster, but did not change where the plateau sits, which is expected since they are implementation changes, not algorithmic ones.

## `target_device` presets

Built directly from the sweep above, not from guessed or default values:

| Preset | M | ef_construction | Recall@10 | Build | Query |
|---|---|---|---|---|---|
| `raspberry-pi` | 8 | 100 | 91.5% | 61.7s | 2.08ms |
| `server` | 16 | 100 | 94.9% | 110.2s | 3.53ms |

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

After any structural change (modified struct layout, modified header), prefer a clean rebuild (`rm -rf build`) over an incremental one before trusting benchmark output. See the note above on why.

## Running the benchmark yourself

```bash
./build/benchmark data/sift/sift_base.fvecs data/sift/sift_query.fvecs [max_layers] [M] [ef_construction]
```

Arguments are optional and fall back to environment variables (`HNSW_MAX_LAYERS`, `HNSW_M`, `HNSW_EF_CONSTRUCTION`, `SIFT_BASE_PATH`, `SIFT_QUERY_PATH`), then to library defaults (16, 5, 200), in that order.

The benchmark prints which distance calculation path is active (`NEON` or `SCALAR`) at the start of each run, so there is no ambiguity about which code executed for a given result.

## Known limitations

- `main` is int8-only. There is no runtime toggle back to float32; use the `v1.0-float32` tag if you need the unquantized version.
- The `server` preset is not yet a fully separate hardware profile. It currently differs from `raspberry-pi` only in `M` and `ef_construction`, both derived from the same single sweep. No separate tuning has been done for concurrent query load or larger memory budgets.
- The sweep was run once per configuration on a single Pi 5 unit. The three-way comparison table above uses 5-run averages for the default config; the sweep table uses single runs per configuration, not averages.
- Storage figures reflect raw vector storage only, not graph edge overhead or the `unordered_map` index's own memory footprint.
- The NEON path has been verified for correctness (matches the scalar path exactly on a 128-dimension test vector, and produces stable recall across repeated full benchmark runs) but has not been profiled at the instruction level; further NEON-specific tuning may be possible.

## License

MIT
