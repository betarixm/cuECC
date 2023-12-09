#ifndef _U64_CUH
#define _U64_CUH

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
  u64 low = a * b;
  u64 high = __umul64hi(a, b);
  u64 oldLow = low;
  low += carry;

  if (low < oldLow) {
    high++;
  }

  *output = low;

  return high;
}

#endif
