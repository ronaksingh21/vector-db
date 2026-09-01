#include "hnsw.h"
#include "utils.h"
#include <chrono>
#include <iostream>

int main() {
    auto learn_vectors = load_fvecs("C:\\Users\\jatin\\Downloads\\sift_learn.fvecs");
    auto query_vectors = load_fvecs("C:\\Users\\jatin\\Downloads\\sift_query.fvecs");
    
    std::cout << "Loaded " << learn_vectors.size() << " learn vectors\n";
    std::cout << "Loaded " << query_vectors.size() << " query vectors\n";
    std::cout << "Dimension: " << learn_vectors[0].size() << "\n\n";
    
    // Build index
    HNSW index(16, 5, 200);
    std::cout << "Building index...\n";
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < std::min(10000, (int)learn_vectors.size()); i++) {
        index.insert(i, learn_vectors[i]);
        if ((i + 1) % 10000 == 0) {
        std::cout << "  Inserted " << (i + 1) << " vectors\n";
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto build_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "Build time: " << build_time << "ms\n\n";
    
    // Query benchmark
    std::cout << "Querying " << query_vectors.size() << " vectors...\n";
    start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < query_vectors.size(); i++) {
        auto results = index.search(query_vectors[i], 10);
    }
    
    end = std::chrono::high_resolution_clock::now();
    auto query_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    
    std::cout << "Total query time: " << query_time << "ms\n";
    std::cout << "Avg per query: " << (query_time / (double)query_vectors.size()) << "ms\n";
    
    return 0;
}