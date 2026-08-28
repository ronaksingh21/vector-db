#pragma once
#include <vector>
#include <map>
#include <cmath>

struct Node{
    int id;
    std::vector<float> vec;
    std:: map<int, std:: vector<int>> neighbors_by_layer
    int current_layer;
};

class HNSW{

};