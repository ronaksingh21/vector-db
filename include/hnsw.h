#pragma once
#include <vector>
#include <map>
#include <unordered_map>
#include <cmath>
#include <cstdint>
#include <random>

struct Node {
    int id;  // external id -- kept only so search() can translate results back at the boundary
    std::vector<int8_t> vec;  // quantized, was std::vector<float>
    std::map<int, std::vector<size_t>> neighbors_by_layer;  // layer -> internal dense indices (not external ids)
    int current_layer;
};

class HNSW{
    public:
        HNSW(int max_layers = 16, int M = 5, int ef_construction = 200);

        void insert(int id, const std::vector<float>& vec);
        std::vector<int> search(const std::vector<float>& query, int k);

        void set_scale_factor(float sf) { scale_factor = sf; }
    private:
        float scale_factor = 1.0f;
        int max_layers;
        int M;
        int ef_construction;
        float ml;

        std::unordered_map<int, size_t> id_to_index;  // external id -> internal dense index
        std::vector<Node> nodes;                       // dense storage, indexed by internal index
        // internal dense index of the entry point; -1 = empty index. Valid indices are
        // always >= 0 (they're positions in `nodes`), so -1 remains a safe sentinel.
        int entry_point = -1;

        // seeded once in the constructor, not reseeded from std::random_device per call --
        // reseeding every call was correlating layer draws across inserts and skewing recall
        std::mt19937 rng;

        //helpers
        float l2_distance(const std::vector<int8_t>& a, const std::vector<int8_t>& b);

        int get_random_layer();
        // entry_points/return value are internal dense indices, not external ids --
        // traversal stays entirely in index space for cache locality.
        std::vector<size_t> search_layer(const std::vector<int8_t>& query, const std::vector<size_t>& entry_points, int layer, int ef);
        void prune_neighbors(size_t node_idx, int layer);

        std::vector<int8_t> quantize(const std::vector<float>& vec);

};
