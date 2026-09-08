#include "hnsw.h"
#include "utils.h"
#include <chrono>
#include <iostream>
#include <set>
#include <cstdlib>

// compile-time visibility: shows up in the build log so we know the compiler's
// target/flags actually define __ARM_NEON, before we ever get to runtime
#ifdef __ARM_NEON
#pragma message("__ARM_NEON is defined for this build")
#else
#pragma message("__ARM_NEON is NOT defined for this build")
#endif

//claudus run test for me pls
int main(int argc, char** argv) {
    const int N_BASE  = 100000;  // corpus size to index
    const int N_QUERY = 100;     // queries to benchmark (brute-force GT costs N_BASE each)
    const int k = 10;

    // runtime visibility: printed into the actual run log, so we know for certain
    // which l2_distance code path this specific run executed -- not just what the
    // compiler was capable of, but whether FORCE_SCALAR_DISTANCE overrode it too
#if defined(__ARM_NEON) && !defined(FORCE_SCALAR_DISTANCE)
    std::cout << "l2_distance path: NEON (__ARM_NEON defined, FORCE_SCALAR_DISTANCE not set)\n";
#elif defined(__ARM_NEON) && defined(FORCE_SCALAR_DISTANCE)
    std::cout << "l2_distance path: SCALAR (__ARM_NEON defined, but FORCE_SCALAR_DISTANCE forces scalar)\n";
#else
    std::cout << "l2_distance path: SCALAR (__ARM_NEON not defined)\n";
#endif

    // Path resolution order: CLI args > env vars > default relative "data/" dir.
    // Usage: benchmark [base_vectors.fvecs] [query_vectors.fvecs] [max_layers] [M] [ef_construction]
    std::string base_path, query_path;
    if (argc >= 3) {
        base_path = argv[1];
        query_path = argv[2];
    } else {
        const char* env_base = std::getenv("SIFT_BASE_PATH");
        const char* env_query = std::getenv("SIFT_QUERY_PATH");
        base_path = env_base ? env_base : "data/sift_base.fvecs";
        query_path = env_query ? env_query : "data/sift_query.fvecs";
    }

    // HNSW build params: CLI args (3rd/4th/5th positional) > env vars > defaults.
    // Exposed so a sweep script can vary M/ef_construction across runs without recompiling.
    auto int_param = [](const char* cli_val, const char* env_name, int default_val) {
        if (cli_val) return std::atoi(cli_val);
        if (const char* env_val = std::getenv(env_name)) return std::atoi(env_val);
        return default_val;
    };
    int max_layers      = int_param(argc >= 4 ? argv[3] : nullptr, "HNSW_MAX_LAYERS", 16);
    int M                = int_param(argc >= 5 ? argv[4] : nullptr, "HNSW_M", 5);
    int ef_construction  = int_param(argc >= 6 ? argv[5] : nullptr, "HNSW_EF_CONSTRUCTION", 200);

    std::cout << "Config: max_layers=" << max_layers << " M=" << M
               << " ef_construction=" << ef_construction << "\n";

    auto base_vectors  = load_fvecs(base_path, N_BASE);
    auto query_vectors = load_fvecs(query_path, N_QUERY);

    std::cout << "Loaded " << base_vectors.size() << " base vectors\n";
    std::cout << "Loaded " << query_vectors.size() << " query vectors\n";
    std::cout << "Dimension: " << base_vectors[0].size() << "\n\n";

    // Ground truth over exactly the vectors we index, so recall@k can reach 100%.
    // (sift_groundtruth.ivecs indexes the full 1M base set - only valid if N_BASE == 1000000)
    std::cout << "Computing exact ground truth (brute force)...\n";
    auto gt_start = std::chrono::high_resolution_clock::now();
    auto ground_truth = compute_ground_truth(base_vectors, query_vectors, k);
    auto gt_end = std::chrono::high_resolution_clock::now();
    std::cout << "  done in "
              << std::chrono::duration_cast<std::chrono::milliseconds>(gt_end - gt_start).count()
              << "ms\n\n";

    // Build index
    HNSW index(max_layers, M, ef_construction);
    std::cout << "Building index...\n";
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < (int)base_vectors.size(); i++) {
        index.insert(i, base_vectors[i]);
        if ((i + 1) % 10000 == 0) {
            std::cout << "  Inserted " << (i + 1) << " vectors" << std::endl;
        }
    }
    // Calculate actual memory used by vector storage
    size_t total_vector_bytes = 0;
    for (int i = 0; i < (int)base_vectors.size(); i++) {
        total_vector_bytes += sizeof(int8_t) * 128;  // quantized: 128 bytes per vector
    }
    std::cout << "\nTotal vector storage: " << total_vector_bytes << " bytes ("
            << (total_vector_bytes / 1024.0 / 1024.0) << " MB)\n";

    size_t float_equivalent_bytes = base_vectors.size() * sizeof(float) * 128;
    std::cout << "Float32 equivalent would be: " << float_equivalent_bytes << " bytes ("
            << (float_equivalent_bytes / 1024.0 / 1024.0) << " MB)\n";
    auto end = std::chrono::high_resolution_clock::now();
    auto build_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "Build time: " << build_time << "ms\n\n";

    // Sanity check: a vector that IS in the index must come back at rank 1 (distance 0).
    // If this fails the graph is broken and the recall number below is meaningless.
    auto self = index.search(base_vectors[42], k);
    std::cout << "Self-query check: got " << (self.empty() ? -1 : self[0])
              << ", want 42 -> " << ((!self.empty() && self[0] == 42) ? "PASS" : "FAIL")
              << "\n\n";

    // Query benchmark + recall calculation
    std::cout << "Querying " << query_vectors.size() << " vectors...\n";
    start = std::chrono::high_resolution_clock::now();
    
    double total_recall = 0.0;

    for (int i = 0; i < (int)query_vectors.size(); i++) {
        auto results = index.search(query_vectors[i], k);
        
        // Calculate recall: how many of our results are in ground truth top-k
        std::set<int> gt_set(ground_truth[i].begin(), ground_truth[i].begin() + k);
        
        int matches = 0;
        for (int id : results) {
            if (gt_set.count(id)) {
                matches++;
            }
        }
        
        double recall = (double)matches / k;
        total_recall += recall;
    }
    
    end = std::chrono::high_resolution_clock::now();
    auto query_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    
    double avg_recall = total_recall / query_vectors.size();
    std::cout << "\nSelf-query recall check...\n";
    int self_query_tests = 100;
    int self_query_hits = 0;

    for (int i = 0; i < self_query_tests; i++) {
        int test_id = i * (base_vectors.size() / self_query_tests);  // spread across the dataset
        auto results = index.search(base_vectors[test_id], 1);
        
        if (!results.empty() && results[0] == test_id) {
            self_query_hits++;
        }
    }

    std::cout << "Self-query hit rate: " << self_query_hits << "/" << self_query_tests 
            << " (" << (100.0 * self_query_hits / self_query_tests) << "%)\n";
    std::cout << "Total query time: " << query_time << "ms\n";
    std::cout << "Avg per query: " << (query_time / (double)query_vectors.size()) << "ms\n";
    std::cout << "Average Recall@" << k << ": " << (avg_recall * 100) << "%\n";
    
    return 0;
}