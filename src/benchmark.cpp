#include "hnsw.h"
#include "utils.h"
#include <chrono>
#include <iostream>
#include <set>
//claudus run test for me pls
int main() {
    const int N_BASE  = 100000;  // corpus size to index
    const int N_QUERY = 100;     // queries to benchmark (brute-force GT costs N_BASE each)
    const int k = 10;

    auto base_vectors  = load_fvecs("C:\\Users\\jatin\\Downloads\\sift_base.fvecs", N_BASE);
    auto query_vectors = load_fvecs("C:\\Users\\jatin\\Downloads\\sift_query.fvecs", N_QUERY);

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
    HNSW index(16, 5, 200);
    std::cout << "Building index...\n";
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < (int)base_vectors.size(); i++) {
        index.insert(i, base_vectors[i]);
        if ((i + 1) % 10000 == 0) {
            std::cout << "  Inserted " << (i + 1) << " vectors" << std::endl;
        }
    }

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