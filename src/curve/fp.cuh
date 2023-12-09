#ifndef _FP_CUF
#define _FP_CUF

#include "../uint/u256.cuh"

__forceinline__ __device__ void fpAdd(u64 output[4], const u64 a[4],
                                      const u64 b[4], const u64 p[4]) {
  u64 result[4];

  u256AddModP(result, a, b, p);

  u256Copy(output, result);
}

__forceinline__ __device__ void fpSub(u64 output[4], const u64 a[4],
                                      const u64 b[4], const u64 prime[4]) {
  u64 result[4];

  u256SubModP(result, a, b, prime);

  u256Copy(output, result);
}

__forceinline__ __device__ void fpMul(u64 output[4], const u64 a[4],
                                      const u64 b[4], const u64 p[4]) {

  u64 result[4];

  u256MulModP(result, a, b, p);

  u256Copy(output, result);
}

__forceinline__ __device__ void fpPow(u64 output[4], const u64 a[4],
                                      const u64 b[4], const u64 p[4]) {
  u64 result[4];

  u256PowModP(result, a, b, p);

  u256Copy(output, result);
}

__forceinline__ __device__ void fpInv(u64 output[4], const u64 a[4],
                                      const u64 p[4]) {
  u64 result[4];

  u256InvModP(result, a, p);

  u256Copy(output, result);
}

__forceinline__ __device__ void fpDiv(u64 output[4], const u64 a[4],
                                      const u64 b[4], const u64 p[4]) {

  u64 inversed[4];
  fpInv(inversed, b, p);

  u64 multiplied[4];
  fpMul(multiplied, inversed, a, p);

  u256Copy(output, multiplied);
}

__forceinline__ __device__ void fpNeg(u64 output[4], const u64 a[4],
                                      const u64 p[4]) {
  u64 result[4];

  fpSub(result, p, a, p);

  u256Copy(output, result);
}

__forceinline__ __device__ bool fpIsZero(const u64 a[4]) {
  return u256IsZero(a);
}

#endif
