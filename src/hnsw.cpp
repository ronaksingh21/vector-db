#include "hnsw.h"
#include <vector>
#include <cmath>
#include<random>
#include <algorithm>
#include <set>
#include <queue>

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
//compared to prior version, it is a bit slower, but improves correctness
void HNSW::insert(int id, const std::vector<float>& vec) {
    Node new_node = {id, vec, {}, get_random_layer()};
    if(entry_point == -1){
        entry_point = id;
        data[id] = new_node;
        return;
    }

    Node& entry = data[entry_point];
    std::vector<int> candidates = {entry_point};

    // navigate from entry layer down to newnodes layer (ef=1 just find entry point)
    for (int layer = entry.current_layer; layer > new_node.current_layer; layer--) {
        candidates = search_layer(new_node.vec, candidates, layer, 1);
    }

    // from newnodes layer down to 0 connect at each layer
    for (int layer = std::min(new_node.current_layer, entry.current_layer); layer >= 0; layer--) {
        candidates = search_layer(new_node.vec, candidates, layer, ef_construction);
        
        for (int i = 0; i < std::min(M, (int)candidates.size()); i++) {
            int neighbor_id = candidates[i];
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

    std::vector<int> candidates = {entry_point};
    Node& entry = data[entry_point];

    for (int layer = entry.current_layer; layer >= 0; layer--){
        candidates = search_layer(query, candidates, layer, std::max(k, ef_construction));
    }

    // Sort final candidates by distance, return top k
    std::vector<std::pair<float, int>> final_distances;
    for (int candidate_id : candidates) {
        float dist = l2_distance(query, data[candidate_id].vec);
        final_distances.push_back({dist, candidate_id});
    }
    std::sort(final_distances.begin(), final_distances.end());

    std::vector<int> result;
    for (int i = 0; i < std::min(k, (int)final_distances.size()); i++) {
        result.push_back(final_distances[i].second);
    }
    
    return result;
}

//update to use priority queue
std::vector<int> HNSW::search_layer(const std::vector<float>& query, const std::vector<int>& entry_points, int layer, int ef) {
    std::set<int> visited(entry_points.begin(), entry_points.end());
    
    // Min-heap
    std::priority_queue<std::pair<float,int>, std::vector<std::pair<float,int>>, std::greater<>> candidates;
    
    // Max-heap
    std::priority_queue<std::pair<float,int>> results;
    
    for (int ep : entry_points) {
        float dist = l2_distance(query, data[ep].vec);
        candidates.push({dist, ep});
        results.push({dist, ep});
    }
    
    while (!candidates.empty()) {
        auto [dist, current] = candidates.top();
        candidates.pop();
        
        // Stop if current is farther than our worst result (and we have enough)
        if (results.size() >= ef && dist > results.top().first) {
            break;
        }
        
        if (data[current].neighbors_by_layer.count(layer)) {
            for (int neighbor : data[current].neighbors_by_layer[layer]) {
                if (visited.count(neighbor)) continue;
                visited.insert(neighbor);
                
                float d = l2_distance(query, data[neighbor].vec);
                
                if (results.size() < ef || d < results.top().first) {
                    candidates.push({d, neighbor});
                    results.push({d, neighbor});
                    if (results.size() > ef) {
                        results.pop();  // Remove worst
                    }
                }
            }
        }
    }
    
    std::vector<int> result_ids;
    while (!results.empty()) {
        result_ids.push_back(results.top().second);
        results.pop();
    }
    return result_ids;
}