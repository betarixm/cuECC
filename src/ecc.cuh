#ifndef __LIBSECP256K1_CUH
#define __LIBSECP256K1_CUH

#include "curve/point.cuh"

__global__ void getPublicKeyByPrivateKeyKernel(Point *output,
                                               const u64 privateKey[4]);

extern "C" void getPublicKeyByPrivateKey(Point output[], u64 privateKeys[][4],
                                         int n);

#endif
