#ifndef _U64_CUH
#define _U64_CUH

#include "../config.h"

#ifdef DEBUG_U64
#include <stdio.h>
#endif

typedef unsigned long long u64;

__forceinline__ __device__ bool u64Add(u64 *output, const u64 a, const u64 b,
                                       const u64 carry) {
  *output = a + b + carry;
  return (*output < a) || (*output < b);
}

__forceinline__ __device__ bool u64Sub(u64 *output, const u64 a, const u64 b,
                                       const u64 borrow) {
  *output = a - b - borrow;
  return a < b;
}

__forceinline__ __device__ u64 u64Mul(u64 *output, const u64 a, const u64 b,
                                      const u64 carry) {

#ifdef DEBUG_U64
  printf("[u64Mul: start] a: %016llx, b: %016llx, carry: %016llx\n", a, b,
         carry);
#endif

  u64 low = a * b;
  u64 high = __umul64hi(a, b);
  u64 oldLow = low;
  low += carry;

  if (low < oldLow) {
    high++;
  }

  *output = low;

#ifdef DEBUG_U64
  printf("[u64Mul: end] %016llx%016llx\n", high, low);
#endif

  return high;
}

#endif
