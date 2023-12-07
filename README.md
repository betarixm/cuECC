# cuECC: CUDA-accelerated Elliptic Curve Cryptography Library

[![CSED490C@POSTECH](https://img.shields.io/badge/CSED490C-POSTECH-c80150)](https://www.postech.ac.kr/eng)
[![CSED490C@POSTECH](https://img.shields.io/badge/Fall-2023-775E64)](https://www.postech.ac.kr/eng)

**cuECC** is a CUDA-accelerated Elliptic Curve Cryptography library designed for secp256k1. The primary goal of this library is to enhance the throughput of ECC by utilizing the parallelism of GPU. This library includes not only the basic operations of ECC but also 256-bit unsigned integer arithmetic operations, operations over finite fields, and curves.

It can be easily integrated into other Python projects using the included Python bindings and is zero-dependency.

## Features

- 256-bit unsigned integer arithmetic operations
- Finite field arithmetic operations
- Elliptic curve point operations
- Parallel public key generation using CUDA

## Usage

To use cuECC, follow these steps:

1. Clone this repository.
1. Navigate to the cloned directory.
1. Run `make all` to build the library.
1. Use the built `libcuecc.so` in your code.

To use cuECC in Python, follow these steps:

1. Build the library as described above.
1. Import the `Ecc` class from [`ecc.py`](src/bindings/ecc.py).
1. Initialize the class with the path to the built library, e.g., `ecc = Ecc('./libcuecc.so')`

## Benchmarks

To compare the performance of cuECC with a pure-python implementation, follow these steps:

1. Clone this repository.
1. Navigate to the cloned directory.
1. Run `make all` to build the library.
1. Run `poetry install` to install the dependencies.
1. Run `poetry run benchmark-public-key-generation` to execute the benchmark.

Note that the Python implementation is from [`cryptography-python`](https://github.com/mohanson/cryptography-python/blob/master/secp256k1.py).

## Disclaimer

Note that this library is not intended for production use. It is unoptimized and may be insecure. It is provided solely for educational purposes, specifically for the CSED490C course at POSTECH. Any usage is at your own risk, and the contributors are not responsible for any potential issues.
