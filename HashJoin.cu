#include "/csproject/yike/intern/ronak/cuCollections/include/cuco/dynamic_map.cuh"

__global__ void hash_join(cuco::dynamic_map<int, int> map, int* keys, int* values, int* output, int size) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < size) {
        auto search = map.find(keys[tid]);
        if (search != map.end()) {
            output[tid] = search->second;
        } else {
            output[tid] = -1;
        }
    }
}

int main() {
    int size = 10000;

    // Initialize keys and values
    int* keys = new int[size];
    int* values = new int[size];
    for (int i = 0; i < size; i++) {
        keys[i] = i;
        values[i] = i * 2;
    }

    // Create a cuco::dynamic_map
    cuco::dynamic_map<int, int> map(size);

    // Insert keys and values into the map
    for (int i = 0; i < size; i++) {
        map.insert({keys[i], values[i]});
    }

    // Allocate output array
    int* output = new int[size];

    // Perform hash join
    hash_join<<<size/256, 256>>>(map, keys, values, output, size);

    // Cleanup
    delete[] keys;
    delete[] values;
    delete[] output;

    return 0;
}
