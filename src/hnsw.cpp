#include "hnsw.h"
#include <vector>
#include <cmath>
#include<random>
#include <algorithm>
#include <set>
#include <queue>

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

HNSW::HNSW(int max_layers, int M, int ef_construction)
    : max_layers(max_layers), M(M), ef_construction(ef_construction), rng(std::random_device{}()){
    ml = 1.0/ std::log(2.0);


}
float HNSW::l2_distance(const std::vector<int8_t>& a, const std::vector<int8_t>& b) {
#ifdef __ARM_NEON
    size_t n = a.size();
    size_t i = 0;
    int32x4_t acc = vdupq_n_s32(0);

    for (; i + 16 <= n; i += 16) {
        int8x16_t va = vld1q_s8(reinterpret_cast<const int8_t*>(&a[i]));
        int8x16_t vb = vld1q_s8(reinterpret_cast<const int8_t*>(&b[i]));

        // widen-subtract low/high 8-lane halves separately -- max diff is 254, safe in int16
        int16x8_t diff_lo = vsubl_s8(vget_low_s8(va), vget_low_s8(vb));
        int16x8_t diff_hi = vsubl_s8(vget_high_s8(va), vget_high_s8(vb));

        // widen-multiply (square) each diff -- max is 254*254 = 64,516, safe in int32
        int32x4_t sq0 = vmull_s16(vget_low_s16(diff_lo), vget_low_s16(diff_lo));
        int32x4_t sq1 = vmull_s16(vget_high_s16(diff_lo), vget_high_s16(diff_lo));
        int32x4_t sq2 = vmull_s16(vget_low_s16(diff_hi), vget_low_s16(diff_hi));
        int32x4_t sq3 = vmull_s16(vget_high_s16(diff_hi), vget_high_s16(diff_hi));

        acc = vaddq_s32(acc, sq0);
        acc = vaddq_s32(acc, sq1);
        acc = vaddq_s32(acc, sq2);
        acc = vaddq_s32(acc, sq3);
    }

    // horizontal reduce via vadd_s32/vpadd_s32 (not vaddvq_s32 -- that's aarch64-only)
    int32x2_t sum_pair = vadd_s32(vget_low_s32(acc), vget_high_s32(acc));
    sum_pair = vpadd_s32(sum_pair, sum_pair);
    int sum = vget_lane_s32(sum_pair, 0);

    // remainder (length not divisible by 16): original scalar loop
    for (; i < n; i++) {
        int d = (int)a[i] - (int)b[i];
        sum += d * d;
    }

    return std::sqrt((float)sum);
#else
    int sum = 0;
    for (size_t i = 0; i < a.size(); i++) {
        int d = (int)a[i] - (int)b[i];
        sum += d * d;
    }
    return std::sqrt((float)sum);
#endif
}
int HNSW::get_random_layer(){
    std::uniform_real_distribution<> dis(0.0,1.0);
    int layer = (int)(-std::log(dis(rng))*ml);
    return std::min(layer, max_layers-1);
}
//compared to prior version, it is a bit slower, but improves correctness
void HNSW::insert(int id, const std::vector<float>& vec) {
    std::vector<int8_t> qvec = quantize(vec);  // quantize immediately
    int node_layer = get_random_layer();

    // boundary: translate external id -> internal dense index here, once
    size_t idx = nodes.size();
    id_to_index[id] = idx;
    nodes.push_back(Node{id, qvec, {}, node_layer});  // store quantized

    if (entry_point == -1) {
        entry_point = (int)idx;
        return;
    }

    int entry_layer = nodes[entry_point].current_layer;

    std::vector<size_t> candidates = {(size_t)entry_point};

    for (int layer = entry_layer; layer > node_layer; layer--) {
        candidates = search_layer(qvec, candidates, layer, 1);  // pass quantized
    }
    for (int layer = std::min(node_layer, entry_layer); layer >= 0; layer--) {
        candidates = search_layer(qvec, candidates, layer, ef_construction);

        for (int i = 0; i < std::min(M, (int)candidates.size()); i++) {
            size_t neighbor_idx = candidates[i];
            if (neighbor_idx == idx) continue;          // no self-loops
            nodes[idx].neighbors_by_layer[layer].push_back(neighbor_idx);
            nodes[neighbor_idx].neighbors_by_layer[layer].push_back(idx);
            prune_neighbors(neighbor_idx, layer);
        }
    }

    if (node_layer > entry_layer) entry_point = (int)idx;
}


std::vector<int> HNSW::search(const std::vector<float>& query, int k) {
    if(nodes.empty() || entry_point == -1){
        return {};
    }
    std::vector<int8_t> qquery = quantize(query);  // quantize immediately

    std::vector<size_t> candidates = {(size_t)entry_point};
    Node& entry = nodes[entry_point];

    for (int layer = entry.current_layer; layer >= 0; layer--){
        candidates = search_layer(qquery, candidates, layer, std::max(k, ef_construction));  // pass quantized
    }

    std::vector<std::pair<float, size_t>> final_distances;
    for (size_t candidate_idx : candidates) {
        float dist = l2_distance(qquery, nodes[candidate_idx].vec);  // both int8_t now
        final_distances.push_back({dist, candidate_idx});
    }
    std::sort(final_distances.begin(), final_distances.end());

    // boundary: translate internal dense index -> external id here, once
    std::vector<int> result;
    for (int i = 0; i < std::min(k, (int)final_distances.size()); i++) {
        result.push_back(nodes[final_distances[i].second].id);
    }

    return result;
}

//update to use priority queue
//operates entirely in internal-index space -- no id_to_index lookups on the hot path
std::vector<size_t> HNSW::search_layer(const std::vector<int8_t>& query, const std::vector<size_t>& entry_points, int layer, int ef) {
    std::set<size_t> visited(entry_points.begin(), entry_points.end());

    // Min-heap
    std::priority_queue<std::pair<float,size_t>, std::vector<std::pair<float,size_t>>, std::greater<>> candidates;

    // Max-heap
    std::priority_queue<std::pair<float,size_t>> results;

    for (size_t ep : entry_points) {
        float dist = l2_distance(query, nodes[ep].vec);
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

        if (nodes[current].neighbors_by_layer.count(layer)) {
            for (size_t neighbor : nodes[current].neighbors_by_layer[layer]) {
                if (visited.count(neighbor)) continue;
                visited.insert(neighbor);

                float d = l2_distance(query, nodes[neighbor].vec);

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
    std::vector<size_t> result_ids;
    while (!results.empty()) {
        result_ids.push_back(results.top().second);
        results.pop();
    }
    std::reverse(result_ids.begin(), result_ids.end());
    return result_ids;
}
void HNSW::prune_neighbors(size_t node_idx, int layer){
    auto& neighbors = nodes[node_idx].neighbors_by_layer[layer];

    //layer 0 decides final recall, so it gets a looser cap
    int max_conn = (layer == 0) ? 2*M : M;

    if ((int) neighbors.size()<=max_conn){
        return;
    }
    std::vector<std::pair<float, size_t>> distances;
    for (size_t neighbor_idx : neighbors) {
        float dist = l2_distance(nodes[node_idx].vec, nodes[neighbor_idx].vec);
        distances.push_back({dist, neighbor_idx});
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
