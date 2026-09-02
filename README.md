# Vector Database (HNSW) — C++

A from-scratch implementation of Hierarchical Navigable Small World (HNSW) graphs for approximate nearest neighbor search, built to understand and benchmark against production vector databases.

## What it does

Indexes high-dimensional vectors and finds the k-nearest neighbors to a query vector in sub-linear time, using a multi-layer graph structure for fast approximate search.

## Benchmarks (SIFT1M dataset, 100K vectors, 128 dimensions)

| Metric | This implementation | pgvector (tuned) |
|---|---|---|
| Build time | 114s | 10.96s |
| Query latency (avg) | 4.89ms | ~1-3ms |
| Recall@10 | 93.7% | ~100% |

Benchmarked against [pgvector](https://github.com/pgvector/pgvector) v0.8.6 with `maintenance_work_mem` tuned to 1GB.

## Architecture

- **Graph structure:** Multi-layer HNSW graph with exponentially decreasing layer probability
- **Search:** Priority-queue-based beam search (min-heap for candidates, max-heap for results)
- **Insert:** Greedy descent to entry layer, then connects at every layer down to 0 with neighbor pruning (M=5, 2M at layer 0)
- **Distance metric:** L2 (Euclidean)

## Build & Run

### Windows
\`\`\`powershell
mkdir build
cd build
cmake .. -G "Visual Studio 17 2022"
cmake --build . --config Release
.\Release\benchmark.exe
\`\`\`

### Linux/Mac
\`\`\`bash
mkdir build
cd build
cmake ..
make
./benchmark
\`\`\`

## Known limitations / future work

- Uses `std::map` for node storage instead of flat contiguous vectors (cache-unfriendly)
- No SIMD vectorization for distance calculations
- Single-threaded (no concurrent insert/query support)
- Not yet tested at 1M+ scale

## Dataset

Benchmarked using [SIFT1M](https://huggingface.co/datasets/qbo-odp/sift1m) (128-dimensional SIFT descriptors).



## Python Bindings & REST API

This project includes Python bindings (via pybind11) and a Flask REST API for interacting with the HNSW index over HTTP.

### Setup

Install Python dependencies:
```bash
pip install pybind11 flask requests
```

Build the Python module (from repo root):
```powershell
cmake -S . -B build -Dpybind11_DIR="<path_to_your_pybind11_cmake_dir>"
cmake --build build --config Release
```

Find your pybind11 CMake dir with:
```bash
python -c "import pybind11; print(pybind11.get_cmake_dir())"
```

### Running the API

The compiled module (`hnsw_module.*.pyd` or `.so`) will be in `build/Release/` (Windows) or `build/` (Linux/Mac).

```bash
cd python
python api.py
```

Server runs on `http://127.0.0.1:5000`.

### API Endpoints

**POST /insert**
```json
{
  "id": 1,
  "vector": [0.1, 0.2, ...]
}
```

**POST /search**
```json
{
  "vector": [0.1, 0.2, ...],
  "k": 10
}
```

Returns:
```json
{
  "results": [id1, id2, ...]
}
```

### Example

```python
import requests

requests.post('http://127.0.0.1:5000/insert', json={'id': 1, 'vector': [1.0, 0.0, 0.0]})
response = requests.post('http://127.0.0.1:5000/search', json={'vector': [0.95, 0.05, 0.0], 'k': 5})
print(response.json())
```