#include "curve/secp256k1.cuh"
#include "ecc.cuh"
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <stdio.h>

#define cudaRunOrAbort(ans)                                                    \
  { cudaAssert((ans), __FILE__, __LINE__); }
inline void cudaAssert(cudaError_t code, const char *file, int line,
                       bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "[CUDA] Error: %s (%s:%d)\n", cudaGetErrorString(code),
            file, line);
    if (abort)
      exit(code);
  }
}

__global__ void getPublicKeyByPrivateKeyKernel(Point *output, u64 *privateKey,
                                               int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < n) {
    secp256k1PublicKey(output + i, privateKey + i * 4);
  }
}

extern "C" void getPublicKeyByPrivateKey(Point output[],
                                         u64 flattenedPrivateKeys[][4], int n) {
  dim3 blockDim(256);
  dim3 gridDim((n + blockDim.x - 1) / blockDim.x);

  int numberOfStreams = gridDim.x;

  cudaStream_t streams[numberOfStreams];

  Point *pinnedPoints;
  Point *devicePoints;
  u64 *pinnedPrivateKeys;
  u64 *devicePrivateKeys;

  cudaRunOrAbort(cudaMallocHost(&pinnedPoints, n * sizeof(Point)));
  cudaRunOrAbort(cudaMallocHost(&pinnedPrivateKeys, n * 4 * sizeof(u64)));
  cudaRunOrAbort(cudaMalloc(&devicePoints, n * sizeof(Point)));
  cudaRunOrAbort(cudaMalloc(&devicePrivateKeys, n * 4 * sizeof(u64)));

  for (int i = 0; i < n; i++) {
    for (int j = 0; j < 4; j++) {
      pinnedPrivateKeys[i * 4 + j] = flattenedPrivateKeys[i][j];
    }
  }

  for (int i = 0; i < numberOfStreams; i++) {
    cudaRunOrAbort(cudaStreamCreate(&streams[i]));

    int dimension = (i == numberOfStreams - 1) ? dimension = n - i * blockDim.x
                                               : blockDim.x;

    cudaRunOrAbort(cudaMemcpyAsync(devicePrivateKeys + i * blockDim.x * 4,
                                   pinnedPrivateKeys + i * blockDim.x * 4,
                                   dimension * 4 * sizeof(u64),
                                   cudaMemcpyHostToDevice, streams[i]));

    getPublicKeyByPrivateKeyKernel<<<1, blockDim, 0, streams[i]>>>(
        devicePoints + i * blockDim.x, devicePrivateKeys + i * blockDim.x * 4,
        dimension);

    cudaRunOrAbort(cudaMemcpyAsync(
        pinnedPoints + i * blockDim.x, devicePoints + i * blockDim.x,
        dimension * sizeof(Point), cudaMemcpyDeviceToHost, streams[i]));
  }

  for (int i = 0; i < numberOfStreams; i++) {
    cudaRunOrAbort(cudaStreamSynchronize(streams[i]));
    cudaRunOrAbort(cudaStreamDestroy(streams[i]));
  }

  for (int i = 0; i < n; i++) {
    output[i] = pinnedPoints[i];
  }

  cudaRunOrAbort(cudaFreeHost(pinnedPoints));
  cudaRunOrAbort(cudaFreeHost(pinnedPrivateKeys));
  cudaRunOrAbort(cudaFree(devicePoints));
  cudaRunOrAbort(cudaFree(devicePrivateKeys));
}
