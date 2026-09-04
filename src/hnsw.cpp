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
float HNSW::l2_distance(const std::vector<int8_t>& a, const std::vector<int8_t>& b) {
    int sum = 0;
    for (size_t i = 0; i < a.size(); i++) {
        int d = (int)a[i] - (int)b[i];
        sum += d * d;
    }
    return std::sqrt((float)sum);
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
    std::vector<int8_t> qvec = quantize(vec);  // quantize immediately
    int node_layer = get_random_layer();

    if (entry_point == -1) {
        data[id] = Node{id, qvec, {}, node_layer};  // store quantized
        entry_point = id;
        return;
    }

    int entry_layer = data[entry_point].current_layer;

    data[id] = Node{id, qvec, {}, node_layer};  // store quantized

    std::vector<int> candidates = {entry_point};

    for (int layer = entry_layer; layer > node_layer; layer--) {
        candidates = search_layer(qvec, candidates, layer, 1);  // pass quantized
    }
    for (int layer = std::min(node_layer, entry_layer); layer >= 0; layer--) {
        candidates = search_layer(qvec, candidates, layer, ef_construction);

        for (int i = 0; i < std::min(M, (int)candidates.size()); i++) {
            int neighbor_id = candidates[i];
            if (neighbor_id == id) continue;          // no self-loops
            data[id].neighbors_by_layer[layer].push_back(neighbor_id);
            data[neighbor_id].neighbors_by_layer[layer].push_back(id);
            prune_neighbors(neighbor_id, layer);
        }
    }

    if (node_layer > entry_layer) entry_point = id;
}


std::vector<int> HNSW::search(const std::vector<float>& query, int k) {
    if(data.empty() || entry_point == -1){
        return {};
    }
    std::vector<int8_t> qquery = quantize(query);  // quantize immediately

    std::vector<int> candidates = {entry_point};
    Node& entry = data[entry_point];

    for (int layer = entry.current_layer; layer >= 0; layer--){
        candidates = search_layer(qquery, candidates, layer, std::max(k, ef_construction));  // pass quantized
    }

    std::vector<std::pair<float, int>> final_distances;
    for (int candidate_id : candidates) {
        float dist = l2_distance(qquery, data[candidate_id].vec);  // both int8_t now
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
std::vector<int> HNSW::search_layer(const std::vector<int8_t>& query, const std::vector<int>& entry_points, int layer, int ef) {
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
        if ((int)results.size() >= ef && dist > results.top().first) {
            break;
        }
        
        if (data[current].neighbors_by_layer.count(layer)) {
            for (int neighbor : data[current].neighbors_by_layer[layer]) {
                if (visited.count(neighbor)) continue;
                visited.insert(neighbor);
                
                float d = l2_distance(query, data[neighbor].vec);
                
                if ((int)results.size() < ef || d < results.top().first) {
                    candidates.push({d, neighbor});
                    results.push({d, neighbor});
                    if ((int)results.size() > ef) {
                        results.pop();  // Remove worst
                    }
                }
            }
        }
    }
    
    // max-heap pops farthest-first, so flip to nearest-first:
    // insert() takes the leading M entries as neighbors and needs the closest ones(claudius helped me debug :])
    std::vector<int> result_ids;
    while (!results.empty()) {
        result_ids.push_back(results.top().second);
        results.pop();
    }
    std::reverse(result_ids.begin(), result_ids.end());
    return result_ids;
}
void HNSW::prune_neighbors(int node_id, int layer){
    auto& neighbors = data[node_id].neighbors_by_layer[layer];

    //layer 0 decides final recall, so it gets a looser cap
    int max_conn = (layer == 0) ? 2*M : M;

    if ((int) neighbors.size()<=max_conn){
        return;
    }
    std::vector<std::pair<float, int>> distances;
    for (int neighbor_id : neighbors) {
        float dist = l2_distance(data[node_id].vec, data[neighbor_id].vec);
        distances.push_back({dist, neighbor_id});
    }
    //we wanna keep the closest nodes
    std::sort(distances.begin(), distances.end());
    neighbors.clear();
    for(int i =0; i<max_conn; i++){
        neighbors.push_back(distances[i].second);
    }
}


std::vector<int8_t> HNSW::quantize(const std::vector<float>& vec) {
    std::vector<int8_t> result(vec.size());
    for (size_t i = 0; i < vec.size(); i++) {
        float q = std::round(vec[i] * scale_factor);
        // clamp: queries can fall outside the range the scale factor was fit to
        result[i] = (int8_t)std::min(127.0f, std::max(-127.0f, q));
    }
    return result;
}
