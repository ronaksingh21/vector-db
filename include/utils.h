//claudius built cpp loader for data

#pragma once
#include <fstream>
#include <vector>
#include <string>
#include <queue>
#include <algorithm>

//max_count = -1 reads the whole file; otherwise stop early so we don't
//pull all 1M base vectors into RAM just to index a slice of them
std::vector<std::vector<float>> load_fvecs(const std::string& filename, int max_count = -1) {
    std::ifstream file(filename, std::ios::binary);
    std::vector<std::vector<float>> vectors;

    int d;
    while (file.read((char*)&d, sizeof(int))) {
        std::vector<float> vec(d);
        file.read((char*)vec.data(), d * sizeof(float));
        vectors.push_back(vec);
        if (max_count > 0 && (int)vectors.size() >= max_count) break;
    }

    return vectors;
}

//load truth file to benchmark
std::vector<std::vector<int>> load_ivecs(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    std::vector<std::vector<int>> vectors;
    
    int d;
    while (file.read((char*)&d, sizeof(int))) {
        std::vector<int> vec(d);
        file.read((char*)vec.data(), d * sizeof(int));
        vectors.push_back(vec);
    }

    return vectors;
}

//exact top-k by brute force, ids are indices into `corpus`.
//the shipped sift_groundtruth.ivecs indexes the full 1M base set, so it only
//applies if you index all of it; this builds an answer key for whatever
//subset you actually inserted, so recall@k can reach 100%.
std::vector<std::vector<int>> compute_ground_truth(
        const std::vector<std::vector<float>>& corpus,
        const std::vector<std::vector<float>>& queries,
        int k) {
    std::vector<std::vector<int>> gt;
    gt.reserve(queries.size());

    for (const auto& query : queries) {
        //max-heap capped at k, so we keep the k smallest distances seen
        std::priority_queue<std::pair<float, int>> heap;

        for (size_t i = 0; i < corpus.size(); i++) {
            //squared distance, no sqrt: monotonic, and we only compare
            float sum = 0.0f;
            for (size_t d = 0; d < query.size(); d++) {
                float diff = query[d] - corpus[i][d];
                sum += diff * diff;
            }

            if ((int)heap.size() < k) {
                heap.push({sum, (int)i});
            } else if (sum < heap.top().first) {
                heap.pop();
                heap.push({sum, (int)i});
            }
        }

        std::vector<int> ids;
        while (!heap.empty()) {
            ids.push_back(heap.top().second);
            heap.pop();
        }
        std::reverse(ids.begin(), ids.end());  //heap pops farthest-first
        gt.push_back(ids);
    }

    return gt;
}


float compute_scale_factor(const std::vector<std::vector<float>>& vectors) {
    float max_abs = 0.0f;
    for (const auto& vec : vectors) {
        for (float val : vec) {
            max_abs = std::max(max_abs, std::abs(val));
        }
    }
    if (max_abs == 0.0f) return 1.0f;
    return 127.0f / max_abs;
}
