#pragma once
#include <vector>
#include <map>
#include <cmath>
#include <cstdint>

struct Node {
    int id;
    std::vector<int8_t> vec;  // quantized, was std::vector<float>
    std::map<int, std::vector<int>> neighbors_by_layer;
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

        std::map<int,Node> data;
        int entry_point = -1;

        //helpers
        float l2_distance(const std::vector<int8_t>& a, const std::vector<int8_t>& b);

        int get_random_layer();
        std::vector<int> search_layer(const std::vector<int8_t>& query, const std::vector<int>& entry_points, int layer, int ef);
        void prune_neighbors(int node_id,int layer);

        std::vector<int8_t> quantize(const std::vector<float>& vec);

};
