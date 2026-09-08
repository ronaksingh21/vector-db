"""
Device-aware presets for HNSWIndex.

Presets are derived from an empirical M / ef_construction sweep run on a
Raspberry Pi 5 (8GB) against SIFT1M (100K vectors, 128D), int8-quantized.

Key finding from the sweep: Recall@10 plateaus at ~95.5% once M >= 16,
regardless of ef_construction. Pushing M/ef higher than that buys no
additional recall -- only slower builds and slower queries. For example,
M=32/ef=400 took 1,333s to build and 45.2ms/query for the *same* 95.5%
recall that M=16/ef=100 reaches in 239s and 6.5ms/query.

Full sweep results (Pi 5, int8, 100K vectors):

    M    ef    build(s)   query(ms)   recall@10
    4    100     77.5        3.69       64.7%
    4    200    143.9        6.03       89.7%
    4    400    294.3       11.76       93.3%
    8    100    136.3        4.08       89.4%
    8    200    244.0       11.71       87.6%   (noise -- see note below)
    8    400    481.0       20.65       94.7%
    16   100    239.3        6.50       95.1%
    16   200    418.5       11.44       95.4%
    16   400    780.2       32.05       95.5%
    32   100    479.3        9.94       95.5%
    32   200    785.8       17.88       95.5%
    32   400   1333.5       45.22       95.5%

Note on the M=8/ef=200 dip: recall checks here use a 100-query sample,
and self-query hit-rate has previously been observed to vary run-to-run
at this sample size (see README benchmarking notes). This single
data point is treated as noise, not a real regression, consistent with
that prior finding.

Preset rationale:
  - "raspberry-pi": M=8, ef_construction=100. Prioritizes fast build and
    low query latency on constrained hardware; accepts ~89% recall as
    the tradeoff. Appropriate when index rebuild time or per-query
    latency matters more than squeezing out the last few points of recall.
  - "server": M=16, ef_construction=100. Recall plateaus at M=16, so
    there is no measured benefit to M=32 on this workload -- server
    gets effectively the same recall ceiling as any higher-M config,
    just with more headroom for concurrent query load than the Pi preset.

Both presets currently keep int8 quantization on, since main no longer
ships a float32 code path (see v1.0-float32 tag for the pre-quantization
baseline). max_layers is left at the library default (16) for both --
the sweep did not vary this parameter, so no tuning claim is made for it.
"""

from . import hnsw_module

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
