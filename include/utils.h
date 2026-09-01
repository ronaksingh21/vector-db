//claudius built cpp loader for data

#pragma once
#include <fstream>
#include <vector>

std::vector<std::vector<float>> load_fvecs(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    std::vector<std::vector<float>> vectors;
    
    int d;
    while (file.read((char*)&d, sizeof(int))) {
        std::vector<float> vec(d);
        file.read((char*)vec.data(), d * sizeof(float));
        vectors.push_back(vec);
    }
    
    return vectors;
}