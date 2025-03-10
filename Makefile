NVCC = /usr/local/cuda/bin/nvcc
ARCH = -arch=sm_61
CFLAGS = $(ARCH) -rdc=true

# Source files
SRC_MAIN = main.cu
SRC_SEARCH = search.cu
SRC_SECP256K1 = third-party/secp256k1/inc_ecc_secp256k1.cu
SRC_KECCAK256 = third-party/keccak/keccak256.cu

# Object files
OBJ_MAIN = main.o
OBJ_SEARCH = search.o
OBJ_SECP256K1 = secp256k1.o
OBJ_KECCAK256 = keccak256.o

# Targets
all: search.out

# Compile object files
$(OBJ_MAIN): $(SRC_MAIN)
	$(NVCC) $(CFLAGS) -c $< -o $@

$(OBJ_SEARCH): $(SRC_SEARCH)
	$(NVCC) $(CFLAGS) -c $< -o $@

$(OBJ_SECP256K1): $(SRC_SECP256K1)
	$(NVCC) $(CFLAGS) -c $< -o $@

$(OBJ_KECCAK256): $(SRC_KECCAK256)
	$(NVCC) $(CFLAGS) -c $< -o $@

# Link final executable
search.out: $(OBJ_MAIN) $(OBJ_SEARCH) $(OBJ_SECP256K1) $(OBJ_KECCAK256)
	$(NVCC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# Clean up
clean:
	rm -f search.out $(OBJ_SEARCH) $(OBJ_SECP256K1) $(OBJ_KECCAK256)