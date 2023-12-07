#ifndef __SECP256K1_H
#define __SECP256K1_H

#include "fp.cuh"
#include "point.cuh"

__constant__ u64 A[4] = {0, 0, 0, 0};

__constant__ u64 B[4] = {7, 0, 0, 0};

__constant__ u64 P[4] = {0xfffffffefffffc2f, 0xffffffffffffffff,
                         0xffffffffffffffff, 0xffffffffffffffff};

__constant__ Point G = Point{{0x59f2815b16f81798, 0x029bfcdb2dce28d9,
                              0x55a06295ce870b07, 0x79be667ef9dcbbac},
                             {0x9c47d08ffb10d4b8, 0xfd17b448a6855419,
                              0x5da4fbfc0e1108a8, 0x483ada7726a3c465}};

__forceinline__ __device__ void secp256k1Add(Point *output, const Point *p,
                                             const Point *q) {
  pointAdd(output, p, q, P, A, B);
}

__forceinline__ __device__ void secp256k1Neg(Point *output, const Point *p) {
  pointNeg(output, p, P);
}

__forceinline__ __device__ void secp256k1Mul(Point *output, const Point *p,
                                             const u64 *scalar) {
  pointMul(output, p, scalar, P, A, B);
}

__forceinline__ __device__ void secp256k1Sub(Point *output, const Point *p,
                                             const Point *q) {
  pointSub(output, p, q, P, A, B);
}

__forceinline__ __device__ void secp256k1PublicKey(Point *output,
                                                   const u64 privateKey[4]) {
  pointMul(output, &G, privateKey, P, A, B);
}

#endif
