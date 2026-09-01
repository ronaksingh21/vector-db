#include "hnsw.h"
#include <iostream>
//test written by claudius
int main() {
    
    HNSW index(16, 5, 200);
    
    // Insert 3 vectors
    std::vector<float> v1 = {1.0, 0.0, 0.0};
    std::vector<float> v2 = {0.9, 0.1, 0.0};
    std::vector<float> v3 = {0.0, 1.0, 0.0};
    
    index.insert(1, v1);
    index.insert(2, v2);
    index.insert(3, v3);
    
    // Query
    std::vector<float> query = {0.95, 0.05, 0.0};
    auto results = index.search(query, 2);
    
    std::cout << "Results: ";
    for (int id : results) {
        std::cout << id << " ";
    }
    std::cout << std::endl;
    
    return 0;
}