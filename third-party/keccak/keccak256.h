#ifndef KECCAK256_H
#define KECCAK256_H

#include "stdint.h"
#include "../common.h"

#define KECCAK256_HASH_LEN		32
#define KECCAK256_BLOCKSIZE		(200-KECCAK256_HASH_LEN*2)
#define KECCAK256_STATE_LEN     25

DECLSPEC void keccak256_update_state(PRIVATE_AS u64* state, PRIVATE_AS const u8* msg, PRIVATE_AS const u32 len);

#endif