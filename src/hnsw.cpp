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
    int layer = int()(-std::log(dis(gen))*ml);
    return std::min(layer, max_layers-1);
}