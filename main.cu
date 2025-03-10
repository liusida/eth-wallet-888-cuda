#include <iostream>
#include <cuda_runtime.h>

__global__ void search_for_vanity_address(uint32_t *result_key, uint64_t target_prefix, int seed, int verbose);


int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <prefix> <suffix (TODO)>" << std::endl;
        return 1;
    }

    // Convert prefix to uint64_t
    uint64_t target_prefix = std::stoull(argv[1], nullptr, 16);
    int seed = 12345;  // Example seed, can be randomized
    int verbose = 0;    // Enable verbose logging

    // Allocate space for result key on device
    uint32_t *d_result_key;
    cudaMalloc((void**)&d_result_key, 8 * sizeof(uint32_t));

    // Launch kernel
    int threadsPerBlock = 256;
    int numBlocks = 8;
    if (verbose) {
        threadsPerBlock = 1;
        numBlocks = 1;
    }
    search_for_vanity_address<<<numBlocks, threadsPerBlock>>>(d_result_key, target_prefix, seed, verbose);

    // Cleanup
    cudaFree(d_result_key);
    cudaDeviceSynchronize();
    return 0;
}
