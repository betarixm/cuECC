#ifndef _FP_CUF
#define _FP_CUF

#include "../config.h"
#include "../uint/u256.cuh"

#ifdef DEBUG_FP
#include <stdio.h>
#endif

__forceinline__ __device__ void fpAdd(u64 output[4], const u64 a[4],
                                      const u64 b[4], const u64 p[4]) {
#ifdef DEBUG_FP
  printf("# [fpAdd: start]\n");
#endif

  u64 result[4];

  u256AddModP(result, a, b, p);

  u256Copy(output, result);

#ifdef DEBUG_FP
  printf("assert Fq(0x%016llx%016llx%016llx%016llx) + "
         "Fq(0x%016llx%016llx%016llx%016llx) == "
         "Fq(0x%016llx%016llx%016llx%016llx), "
         "Fq(0x%016llx%016llx%016llx%016llx) + "
         "Fq(0x%016llx%016llx%016llx%016llx)\n",
         a[3], a[2], a[1], a[0], b[3], b[2], b[1], b[0], output[3], output[2],
         output[1], output[0], a[3], a[2], a[1], a[0], b[3], b[2], b[1], b[0]);

  printf("# [fpAdd: end]\n");
#endif
}

__forceinline__ __device__ void fpSub(u64 output[4], const u64 a[4],
                                      const u64 b[4], const u64 prime[4]) {
#ifdef DEBUG_FP
  printf("# [fpSub: start]\n");
#endif

  u64 result[4];

  u256SubModP(result, a, b, prime);

  u256Copy(output, result);

#ifdef DEBUG_FP
  printf("assert Fq(0x%016llx%016llx%016llx%016llx) - "
         "Fq(0x%016llx%016llx%016llx%016llx) == "
         "Fq(0x%016llx%016llx%016llx%016llx), "
         "Fq(0x%016llx%016llx%016llx%016llx) - "
         "Fq(0x%016llx%016llx%016llx%016llx)\n",
         a[3], a[2], a[1], a[0], b[3], b[2], b[1], b[0], output[3], output[2],
         output[1], output[0], a[3], a[2], a[1], a[0], b[3], b[2], b[1], b[0]);
  printf("# [fpSub: end]\n");
#endif
}

__forceinline__ __device__ void fpMul(u64 output[4], const u64 a[4],
                                      const u64 b[4], const u64 p[4]) {
#ifdef DEBUG_FP
  printf("# [fpMul: start]\n");
#endif

  u64 result[4];

  u256MulModP(result, a, b, p);

  u256Copy(output, result);

#ifdef DEBUG_FP
  printf("assert Fq(0x%016llx%016llx%016llx%016llx) * "
         "Fq(0x%016llx%016llx%016llx%016llx) == "
         "Fq(0x%016llx%016llx%016llx%016llx), "
         "Fq(0x%016llx%016llx%016llx%016llx) * "
         "Fq(0x%016llx%016llx%016llx%016llx)\n",
         a[3], a[2], a[1], a[0], b[3], b[2], b[1], b[0], output[3], output[2],
         output[1], output[0], a[3], a[2], a[1], a[0], b[3], b[2], b[1], b[0]);

  printf("# [fpMul: end]\n");
#endif
}

__forceinline__ __device__ void fpPow(u64 output[4], const u64 a[4],
                                      const u64 b[4], const u64 p[4]) {
#ifdef DEBUG_FP
  printf("# [fpPow: start]\n");
#endif

  u64 result[4];

  u256PowModP(result, a, b, p);

  u256Copy(output, result);

#ifdef DEBUG_FP
  printf("assert pow(Fq(0x%016llx%016llx%016llx%016llx), "
         "Fq(0x%016llx%016llx%016llx%016llx)) == "
         "Fq(0x%016llx%016llx%016llx%016llx), "
         "pow(Fq(0x%016llx%016llx%016llx%016llx), "
         "Fq(0x%016llx%016llx%016llx%016llx))\n",
         a[3], a[2], a[1], a[0], b[3], b[2], b[1], b[0], output[3], output[2],
         output[1], output[0], a[3], a[2], a[1], a[0], b[3], b[2], b[1], b[0]);
  printf("# [fpPow: end]\n");
#endif
}

__forceinline__ __device__ void fpInv(u64 output[4], const u64 a[4],
                                      const u64 p[4]) {
#ifdef DEBUG_FP
  printf("# [fpInv: start]\n");
#endif

  u64 result[4];

  u256InvModP(result, a, p);

  u256Copy(output, result);

#ifdef DEBUG_FP
  printf("assert pow(Fq(0x%016llx%016llx%016llx%016llx), -1) == "
         "Fq(0x%016llx%016llx%016llx%016llx), "
         "pow(Fq(0x%016llx%016llx%016llx%016llx), -1)\n",
         a[3], a[2], a[1], a[0], output[3], output[2], output[1], output[0],
         a[3], a[2], a[1], a[0]);

  printf("# [fpInv: end]\n");
#endif
}

__forceinline__ __device__ void fpDiv(u64 output[4], const u64 a[4],
                                      const u64 b[4], const u64 p[4]) {
#ifdef DEBUG_FP
  printf("# [fpDiv: start]\n");
#endif

  u64 inversed[4];
  fpInv(inversed, b, p);

  u64 multiplied[4];
  fpMul(multiplied, inversed, a, p);

  u256Copy(output, multiplied);

#ifdef DEBUG_FP
  printf("assert Fq(0x%016llx%016llx%016llx%016llx) / "
         "Fq(0x%016llx%016llx%016llx%016llx) == "
         "Fq(0x%016llx%016llx%016llx%016llx)\n",
         a[3], a[2], a[1], a[0], b[3], b[2], b[1], b[0], output[3], output[2],
         output[1], output[0]);

  printf("# [fpDiv: end]\n");
#endif
}

__forceinline__ __device__ void fpNeg(u64 output[4], const u64 a[4],
                                      const u64 p[4]) {
#ifdef DEBUG_FP
  printf("# [fpNeg: start]\n");
#endif

  u64 result[4];

  fpSub(result, p, a, p);

  u256Copy(output, result);

#ifdef DEBUG_FP
  printf("assert -Fq(0x%016llx%016llx%016llx%016llx) == "
         "Fq(0x%016llx%016llx%016llx%016llx), "
         "-Fq(0x%016llx%016llx%016llx%016llx)\n",
         a[3], a[2], a[1], a[0], output[3], output[2], output[1], output[0],
         a[3], a[2], a[1], a[0]);
#endif
}

__forceinline__ __device__ bool fpIsZero(const u64 a[4]) {
  return u256IsZero(a);
}

#endif
