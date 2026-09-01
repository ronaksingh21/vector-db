#pragma once
#include <vector>
#include <map>
#include <cmath>

struct Node{
    int id;
    std::vector<float> vec;
    std:: map<int, std:: vector<int>> neighbors_by_layer;
    int current_layer;
};

class HNSW{
    public:
        HNSW(int max_layers = 16, int M = 5, int ef_construction = 200);
        
        void insert(int id, const std::vector<float>& vec);
        std::vector<int> search(const std::vector<float>& query, int k);
    private:
        int max_layers;
        int M;
        int ef_construction;
        float ml;

        std::map<int,Node> data;
        int entry_point = -1;

        //helpers
        float l2_distance(const std::vector<float>& a, const std::vector<float>& b);

        int get_random_layer();
        std::vector<int> search_layer(const std::vector<float>& query, const std::vector<int>& entry_points, int layer, int ef);

};