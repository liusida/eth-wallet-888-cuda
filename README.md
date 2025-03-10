# eth-wallet-888-cuda

Fast Ethereum Vanity Wallet Address Generator using CUDA for GPU Acceleration.

## Features
- High-performance Ethereum vanity address generation.
- Optimized for CUDA to leverage GPU acceleration.
- Supports customizable prefix searching.

## Installation
### Prerequisites
- NVIDIA GPU with CUDA support.
- CUDA Toolkit installed.
- GNU Make and a C++ compiler installed.

### Build Instructions
```sh
make clean
make
```

## Usage
To run the program and search for an Ethereum address with a specific prefix:
```sh
./search.out <prefix>
```
Example:
```sh
./search.out 1234
```
This will search for an Ethereum address that starts with `0x1234`.

## Contributing
Feel free to contribute by submitting issues and pull requests.

## My Test Environment
- Windows 10 WSL 2
- NVIDIA GTX 1070
- nvcc: V12.2

If you find this project useful and want to support further development, send ETH to:
**0x8888888827d3099Ba8A1Fe5C7B47cDfca8C4c1B5**
(or even SepoliaETH is also welcome)

Thank you for your support! 🚀

