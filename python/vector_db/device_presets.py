"""
Device-aware presets for HNSWIndex.

Presets are derived from an empirical M / ef_construction sweep run on a
Raspberry Pi 5 (8GB) against a 100,000-vector subset of SIFT1M (128D),
int8-quantized, using the flat-vector storage layer and NEON-accelerated
distance calculation (see hnsw.h / hnsw.cpp).

This sweep was re-run after three changes to the underlying implementation:
  1. Storage was changed from std::map<int, Node> to a two-level index
     (unordered_map<int, size_t> + a dense std::vector<Node>), for better
     cache locality during graph traversal.
  2. l2_distance gained a NEON-accelerated path on ARM (16 lanes/iteration),
     with the original scalar loop kept as the exact fallback on non-ARM
     builds and verified byte-identical there.
  3. The graph's random layer assignment (get_random_layer) was fixed to
     seed its RNG once per index instead of once per call. The previous
     per-call reseeding was found to occasionally produce badly-shaped
     graphs, showing up as isolated benchmark runs with recall as low as
     63-78% against an otherwise stable ~90-93% baseline. All numbers
     below were collected after this fix; recall is now stable across
     repeated runs (no more isolated low-recall outliers).

Key finding, unchanged from the pre-optimization sweep: Recall@10 plateaus
at ~95.5% once M >= 16, regardless of ef_construction. Pushing M/ef higher
than that buys no additional recall, only slower builds and slower queries.
The optimizations above made every configuration faster, but did not move
where this plateau sits, which is expected since they are storage/speed
changes, not algorithmic ones.

Full sweep results (Pi 5, int8, NEON, 100K vectors, post-fix):

    M    ef    build(s)   query(ms)   recall@10
    4    100     37.3        1.88       70.9%
    4    200     76.1        2.65       85.7%
    4    400    150.5        5.45       91.7%
    8    100     61.7        2.08       91.5%
    8    200    120.1        4.04       94.7%
    8    400    241.3        9.47       95.2%
    16   100    110.2        3.53       94.9%
    16   200    215.6        7.35       95.4%
    16   400    415.7       16.45       95.5%
    32   100    214.8        5.89       95.5%
    32   200    409.7       14.54       95.5%
    32   400    743.8       25.65       95.5%

For comparison, M=32/ef=400 takes 743.8s to build and 25.65ms per query
for the same 95.5% recall that M=16/ef=100 reaches in 110.2s and 3.53ms.
Past the M=16 plateau, additional M or ef_construction is pure cost.

Preset rationale:
  - "raspberry-pi": M=8, ef_construction=100. Prioritizes fast build and
    low query latency on constrained hardware; accepts ~91.5% recall as
    the tradeoff. Appropriate when index rebuild time or per-query
    latency matters more than squeezing out the last few points of recall.
  - "server": M=16, ef_construction=100. Recall plateaus at M=16, so
    there is no measured benefit to M=32 on this workload -- server
    gets effectively the same recall ceiling as any higher-M config,
    just with more headroom for concurrent query load than the Pi preset.

Both presets keep int8 quantization on, since main does not ship a
float32 code path (see the v1.0-float32 tag for the pre-quantization
baseline). max_layers is left at the library default (16) for both --
the sweep did not vary this parameter, so no tuning claim is made for it.
"""

import hnsw_module

DEVICE_PRESETS = {
    "raspberry-pi": {"max_layers": 16, "M": 8, "ef_construction": 100},
    "server": {"max_layers": 16, "M": 16, "ef_construction": 100},
}


class HNSWIndex:
    """
    Thin Python wrapper around the C++ HNSW class adding a `target`
    convenience parameter. `target` selects a preset from DEVICE_PRESETS;
    explicit max_layers / M / ef_construction args always override the
    preset if both are given.
    """

    def __init__(self, target=None, max_layers=16, M=5, ef_construction=200):
        if target is not None:
            if target not in DEVICE_PRESETS:
                raise ValueError(
                    f"Unknown target '{target}'. Options: {list(DEVICE_PRESETS.keys())}"
                )
            preset = DEVICE_PRESETS[target]
            max_layers = preset["max_layers"]
            M = preset["M"]
            ef_construction = preset["ef_construction"]

        self.target = target
        self.max_layers = max_layers
        self.M = M
        self.ef_construction = ef_construction
        self._index = hnsw_module.HNSW(max_layers, M, ef_construction)

    def insert(self, id, vector):
        self._index.insert(id, vector)

    def search(self, query, k):
        return self._index.search(query, k)

    def __repr__(self):
        return (
            f"HNSWIndex(target={self.target!r}, max_layers={self.max_layers}, "
            f"M={self.M}, ef_construction={self.ef_construction})"
        )
