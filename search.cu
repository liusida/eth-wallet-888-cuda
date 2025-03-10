#include <curand_kernel.h>
#include <stdio.h>
#include <stdint.h>
#include "./third-party/common.h"

__device__ volatile int stop_flag = 0; // Shared stopping flag

// little endian to big endian
DECLSPEC u32 l2be(u32 x)
{
    return (x & 0xff) << 24 | (x & 0xff00) << 8 | (x & 0xff0000) >> 8 | (x & 0xff000000) >> 24;
}

__device__ uint64_t generate_64bit_random(curandState *state)
{
    uint64_t high = (uint64_t)curand(state); // First 32 bits
    uint64_t low = (uint64_t)curand(state);  // Next 32 bits
    return (high << 32) | low;               // Combine them into a 64-bit number
}

// ✅ Compute `prefix_length` from `target_prefix`
__device__ int compute_prefix_length(uint64_t target_prefix)
{
    int length = 0;
    while (target_prefix)
    { // Count hex digits
        length++;
        target_prefix >>= 4; // Shift right by 4 bits (1 hex digit)
    }
    return length;
}

// ✅ Compute the PREFIX_MASK dynamically for 64-bit target prefix
__device__ uint64_t compute_prefix_mask(int prefix_length)
{
    return ((1ULL << (4 * prefix_length)) - 1) << (64 - 4 * prefix_length);
}

// ✅ Kernel function to search for an Ethereum address with a given prefix
__global__ void search_for_vanity_address(uint32_t *result_key, uint64_t _target_prefix, int _seed, int _verbose)
{
    uint64_t target_prefix = _target_prefix;
    int seed = _seed;
    int verbose = _verbose;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0)
    {
        stop_flag = 0;
    }

    if (false)
    {
        printf("Result key address: %p\n", result_key);
        printf("Target: 0x%llx\n", target_prefix);
        printf("seed: %d\n", seed);
        printf("Verbose Flag: %d\n", verbose);
    }
    if (verbose)
    {
        printf("Idx: %d\n", idx);
    }

    int prefix_length = compute_prefix_length(target_prefix);
    uint64_t PREFIX_MASK = compute_prefix_mask(prefix_length);
    int shift_amount = 64 - (prefix_length * 4);                     // Compute shift dynamically
    uint64_t adjusted_target_prefix = target_prefix << shift_amount; // ✅ Shift dynamically

    // Initialize random number generator
    curandState state;
    curand_init(seed, idx, 0, &state);

    // Allocate space for SECP256K1 and Keccak hashing
    secp256k1_t g_xy_local;
    uint32_t private_key[8];
    uint32_t public_x[8], public_y[8];
    uint64_t keccak_state[KECCAK256_STATE_LEN] = {};
    uint32_t w[16];

    // Define compressed SECP256K1 generator point (G)
    uint32_t g_local[PUBLIC_KEY_LENGTH_WITH_PARITY];

    uint64_t c = 0;
    while (!stop_flag)
    {
        // ✅ Generate a random private key
        for (int i = 0; i < 8; i++)
        {
            private_key[i] = curand(&state); // 256-bit key
        }
        if (verbose)
        {
            printf("Generate a random private key: 0x");
            for (int i = 0; i < 8; i++)
            {
                printf("%08x", private_key[7 - i]);
            }
            printf("\n");
        }
        g_local[0] = SECP256K1_G_STRING0;
        g_local[1] = SECP256K1_G_STRING1;
        g_local[2] = SECP256K1_G_STRING2;
        g_local[3] = SECP256K1_G_STRING3;
        g_local[4] = SECP256K1_G_STRING4;
        g_local[5] = SECP256K1_G_STRING5;
        g_local[6] = SECP256K1_G_STRING6;
        g_local[7] = SECP256K1_G_STRING7;
        g_local[8] = SECP256K1_G_STRING8;

        // Convert the compressed public key to full (x, y) format
        if (parse_public(&g_xy_local, g_local) != 0)
        {
            printf("Error: Failed to parse SECP256K1 generator point\n");
            return;
        }

        point_mul_xy(public_x, public_y, private_key, &g_xy_local);

        // ✅ Prepare data for Keccak hashing
        w[7] = l2be(public_x[0]);
        w[6] = l2be(public_x[1]);
        w[5] = l2be(public_x[2]);
        w[4] = l2be(public_x[3]);
        w[3] = l2be(public_x[4]);
        w[2] = l2be(public_x[5]);
        w[1] = l2be(public_x[6]);
        w[0] = l2be(public_x[7]);
        w[15] = l2be(public_y[0]);
        w[14] = l2be(public_y[1]);
        w[13] = l2be(public_y[2]);
        w[12] = l2be(public_y[3]);
        w[11] = l2be(public_y[4]);
        w[10] = l2be(public_y[5]);
        w[9] = l2be(public_y[6]);
        w[8] = l2be(public_y[7]);

        // ✅ Reset Keccak state every iteration to prevent accumulation of old data
        for (int i = 0; i < KECCAK256_STATE_LEN; i++)
        {
            keccak_state[i] = 0;
        }

        keccak256_update_state(keccak_state, (uint8_t *)w, 64);

        uint32_t eth_address[5]; // 5 x 32-bit values (total 160 bits)

        // Extract the last 160 bits of Keccak-256 hash (Ethereum address)
        eth_address[0] = l2be((uint32_t)(keccak_state[1] >> 32));
        eth_address[1] = l2be((uint32_t)(keccak_state[2]));
        eth_address[2] = l2be((uint32_t)(keccak_state[2] >> 32));
        eth_address[3] = l2be((uint32_t)(keccak_state[3]));
        eth_address[4] = l2be((uint32_t)(keccak_state[3] >> 32));

        if (verbose)
        {
            printf("Ethereum Address: 0x");
            for (int i = 0; i < 5; i++)
            {
                printf("%08x", eth_address[i]); // Print each part in hex
            }
            printf("\n");
        }

        // ✅ Convert eth_address array into a single uint64_t for easy comparison
        uint64_t eth_address_high = ((uint64_t)eth_address[0] << 32) | eth_address[1]; // First 64 bits

        if (verbose)
        {
            printf("Prefix Mask: %016llx\n", PREFIX_MASK);
            printf("Target Prefix: %016llx\n", adjusted_target_prefix);
            printf("eth_address_high: %016llx\n", eth_address_high);
        }

        // ✅ Apply PREFIX_MASK to the first `prefix_length` bits
        if ((eth_address_high & PREFIX_MASK) == adjusted_target_prefix)
        {
            stop_flag = 1; // Stop all threads
            for (int i = 0; i < 8; i++)
            {
                result_key[i] = private_key[i]; // Save private key
            }
            printf("Thread %d found a match after %lld tries!\n", idx, c);
            printf("Private Key: 0x");
            for (int i = 0; i < 8; i++)
            {
                printf("%08x", private_key[7 - i]);
            }
            printf("\n");
            printf("Ethereum Address: 0x");
            for (int i = 0; i < 5; i++)
            {
                printf("%08x", eth_address[i]); // Print each part in hex
            }
            printf("\n");
        }
        c++;
        // if (verbose && c>3) {
        //     break;
        // }
        if (idx == (blockDim.x - 1) && c % 1000 == 0)
        {
            printf("Count: %lld\n", c * blockDim.x);
        }
    }
}
