#include "hnsw.h"
#include <vector>
#include <cmath>
#include<random>
#include <algorithm>

HNSW::HNSW(int max_layers, int M, int ef_construction)
    : max_layers(max_layers), M(M), ef_construction(ef_construction){
    ml = 1.0/ std::log(2.0);
    

}
float HNSW::l2_distance(const std::vector<float>& a, const std::vector<float>& b){
        float sum = 0;
        for(int i =0; i<a.size();i++){
            float d =a[i] - b[i];
            sum+= d*d;
        }
        return std::sqrt(sum);
    }
int HNSW::get_random_layer(){
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0,1.0);
    int layer = (int)(-std::log(dis(gen))*ml);
    return std::min(layer, max_layers-1);
}

void HNSW::insert(int id, const std::vector<float>& vec) {
    Node new_node = {id, vec, {}, get_random_layer()};
    if(entry_point == -1){
        entry_point = id;
        data[id]= new_node;
        return;
    }
    Node& entry = data[entry_point];
    for( int layer = entry.current_layer; layer>= new_node.current_layer; layer--){
        std::vector <std::pair<float,int>> distances;
        for (auto& [node_id,node] : data){
            if(node.current_layer>= layer){
                float dist = l2_distance(new_node.vec, node.vec);
                distances.push_back({dist,node_id});
            }
        }
        std::sort(distances.begin(), distances.end());
    
        // connect to m closest neighbors bidirectionally
        for (int i = 0; i < std::min(M, (int)distances.size()); i++) {
            int neighbor_id = distances[i].second;
            new_node.neighbors_by_layer[layer].push_back(neighbor_id);
            data[neighbor_id].neighbors_by_layer[layer].push_back(id);
        }
    }
    if (new_node.current_layer > entry.current_layer){
        entry_point = id;
    }
    data[id] = new_node;
}

std::vector<int> HNSW::search(const std::vector<float>& query, int k) {
    if(data.empty() || entry_point == -1){
        return {};
    }

    std::vector<int> candidates;
    candidates.push_back(entry_point);

    Node& entry = data[entry_point];

    for (int layer = entry.current_layer; layer>=0; layer--){
        std::vector<std::pair<float, int>> distances;

        for(int candidate_id : candidates){
            float dist = l2_distance(query, data[candidate_id].vec);
            distances.push_back({dist, candidate_id});

        }
        std::sort(distances.begin(), distances.end());

        candidates.clear();
        if(!distances.empty()){
            candidates.push_back(distances[0].second);
        }
    }
    std::vector<std::pair<float, int>> final_distances;
    for(auto& [node_id, node] : data) {
        float dist = l2_distance(query, node.vec);
        final_distances.push_back({dist, node_id});
    }

    std::sort(final_distances.begin(), final_distances.end());

    std::vector<int> result;
    for (int i = 0; i < std::min(k, (int)final_distances.size()); i++) {
        result.push_back(final_distances[i].second);
    }
    
    return result;

}