#ifndef _U256_CUH
#define _U256_CUH

#include "../config.h"
#include "u64.cuh"

#ifdef DEBUG_U256
#include <stdio.h>
#endif

#ifdef DEBUG_U512
#include <stdio.h>
#endif

__forceinline__ __device__ void u256Copy(u64 output[4], const u64 a[4]) {
  for (int i = 0; i < 4; ++i) {
    output[i] = a[i];
  }
}

__forceinline__ __device__ void u512Copy(u64 output[8], const u64 a[8]) {
  for (int i = 0; i < 8; ++i) {
    output[i] = a[i];
  }
}

__forceinline__ __device__ void u256SetZero(u64 a[4]) {
  for (int i = 0; i < 4; ++i) {
    a[i] = 0;
  }
}

__forceinline__ __device__ void u512SetZero(u64 a[8]) {
  for (int i = 0; i < 8; ++i) {
    a[i] = 0;
  }
}

__forceinline__ __device__ int u256Extend(u64 output[8], const u64 a[4]) {
  u64 result[8];

  u256SetZero(result + 4);
  u256Copy(result, a);

  u512Copy(output, result);
}

__forceinline__ __device__ bool u256GetBit(const u64 a[4], const int index) {
  return (a[index / 64] >> (index % 64)) & 1;
}

__forceinline__ __device__ bool u512GetBit(const u64 a[8], const int index) {
  return (a[index / 64] >> (index % 64)) & 1;
}

__forceinline__ __device__ void u256SetBit(u64 output[4], const int index,
                                           const bool value) {
  if (value) {
    output[index / 64] |= 1ULL << (index % 64);
  } else {
    output[index / 64] &= ~(1ULL << (index % 64));
  }
}

__forceinline__ __device__ void u512SetBit(u64 a[8], const int index,
                                           const bool value) {
  if (value) {
    a[index / 64] |= 1ULL << (index % 64);
  } else {
    a[index / 64] &= ~(1ULL << (index % 64));
  }
}

__forceinline__ __device__ void u256LShift1(u64 a[4]) {
  for (int i = 2; i >= 0; --i) {
    a[i + 1] = (a[i + 1] << 1) | (a[i] >> 31);
  }
  a[0] <<= 1;
}

__forceinline__ __device__ void u512LShift1(u64 a[8]) {
  for (int i = 6; i >= 0; --i) {
    a[i + 1] = (a[i + 1] << 1) | (a[i] >> 63);
  }
  a[0] <<= 1;
}

__forceinline__ __device__ void u256RShift1(u64 a[4]) {
  for (int i = 0; i < 4; ++i) {
    a[i] = (a[i] >> 1) | (a[i + 1] << 31);
  }
  a[4] >>= 1;
}

__forceinline__ __device__ bool u256IsOdd(const u64 a[4]) { return a[0] & 1; }

__forceinline__ __device__ bool u256IsZero(const u64 a[4]) {
  for (int i = 0; i < 4; ++i) {
    if (a[i] != 0) {
      return false;
    }
  }
  return true;
}

__forceinline__ __device__ int u256Compare(const u64 a[4], const u64 b[4]) {
  for (int i = 3; i >= 0; --i) {
    if (a[i] > b[i]) {
      return 1;
    } else if (a[i] < b[i]) {
      return -1;
    }
  }

  return 0;
}

__forceinline__ __device__ int u512Compare(const u64 a[8], const u64 b[8]) {
  for (int i = 7; i >= 0; --i) {
    if (a[i] > b[i]) {
      return 1;
    } else if (a[i] < b[i]) {
      return -1;
    }
  }

  return 0;
}

__forceinline__ __device__ bool u256Add(u64 output[4], const u64 a[4],
                                        const u64 b[4]) {
#ifdef DEBUG_U256
  printf("# [u256Add: start]\n");
#endif

  u64 result[4] = {0};

  u64 carry = 0;

  for (int i = 0; i < 4; ++i) {
    carry = u64Add(&result[i], a[i], b[i], carry);
  }

  u256Copy(output, result);

#ifdef DEBUG_U256
  printf("assert 0x%016llx%016llx%016llx%016llx + "
         "0x%016llx%016llx%016llx%016llx == "
         "0x%016llx%016llx%016llx%016llx%016llx\n",
         a[3], a[2], a[1], a[0], b[3], b[2], b[1], b[0], carry ? 1ll : 0ll,
         output[3], output[2], output[1], output[0]);

  printf("# [u256Add: end]\n");
#endif

  return carry;
}

__forceinline__ __device__ bool u256Sub(u64 output[4], const u64 a[4],
                                        const u64 b[4]) {
  u64 result[4] = {0};

  u64 borrow = 0;

  for (int i = 0; i < 4; ++i) {
    borrow = u64Sub(&result[i], a[i], b[i], borrow);
  }

  u256Copy(output, result);

  return borrow;
}

__forceinline__ __device__ bool u512Sub(u64 output[8], const u64 a[8],
                                        const u64 b[8]) {
  u64 borrow = 0;

  for (int i = 0; i < 8; ++i) {
    borrow = u64Sub(&output[i], a[i], b[i], borrow);
  }

  return borrow;
}

__forceinline__ __device__ u64 u256MulWithU64(u64 output[4], const u64 a[4],
                                              const u64 b, u64 carry) {

  u64 result[4] = {0};

  for (int i = 0; i < 4; ++i) {
    carry = u64Mul(&result[i], a[i], b, carry);
  }

  u256Copy(output, result);

  return carry;
}

__forceinline__ __device__ void u256Mul(u64 output[8], const u64 a[4],
                                        const u64 b[4]) {
  u64 result[8] = {0};

  for (int i = 0; i < 4; ++i) {
    u64 t0[4] = {0};
    u64 carry = 0;

    carry += u256MulWithU64(t0, a, b[i], carry);

    u64 t1[4];
    u256Copy(t1, result + i);

    carry += u256Add(result + i, t1, t0);

    result[i + 4] = carry;
  }

  u512Copy(output, result);
}

__forceinline__ __device__ void u256Div(u64 quotient[4], u64 remainder[4],
                                        const u64 dividend[4],
                                        const u64 divisor[4]) {
#ifdef DEBUG_U256
  printf("# [u256Div] start\n");
#endif

  u64 quotientAndRemainder[8] = {0};

  u256SetZero(quotientAndRemainder + 4);
  u256Copy(quotientAndRemainder, dividend);

  for (int i = 255; i >= 0; --i) {
    u512LShift1(quotientAndRemainder);

    u64 *q = quotientAndRemainder;
    u64 *r = quotientAndRemainder + 4;

    if (u256Compare(r, divisor) >= 0) {
      u64 temp[4];
      u256Sub(temp, r, divisor);
      u256SetBit(q, 0, 1);
      u256Copy(r, temp);
    }
  }

  u256Copy(quotient, quotientAndRemainder);
  u256Copy(remainder, quotientAndRemainder + 4);

#ifdef DEBUG_U256
  printf("assert 0x%016llx%016llx%016llx%016llx "
         "== 0x%016llx%016llx%016llx%016llx // "
         "0x%016llx%016llx%016llx%016llx\n",
         quotient[3], quotient[2], quotient[1], quotient[0], dividend[3],
         dividend[2], dividend[1], dividend[0], divisor[3], divisor[2],
         divisor[1], divisor[0]);
  printf("assert 0x%016llx%016llx%016llx%016llx "
         "== 0x%016llx%016llx%016llx%016llx %% "
         "0x%016llx%016llx%016llx%016llx\n",
         remainder[3], remainder[2], remainder[1], remainder[0], dividend[3],
         dividend[2], dividend[1], dividend[0], divisor[3], divisor[2],
         divisor[1], divisor[0]);

  printf("# [u256Div: end]\n");
#endif
}

__forceinline__ __device__ void u512Div(u64 quotient[8], u64 remainder[8],
                                        const u64 dividend[8],
                                        const u64 divisor[8]) {
#ifdef DEBUG_U512
  printf("# [u512Div] start\n");
#endif

  u512SetZero(quotient);
  u512SetZero(remainder);

  for (int i = 511; i >= 0; --i) {
    u512LShift1(remainder);
    u512SetBit(remainder, 0, u512GetBit(dividend, i));

    if (u512Compare(remainder, divisor) >= 0) {
      u512Sub(remainder, remainder, divisor);
      u512SetBit(quotient, i, 1);
    }
  }
#ifdef DEBUG_U512
  printf("assert 0x%016llx%016llx%016llx%016llx%016llx%016llx%016llx%016llx "
         "== 0x%016llx%016llx%016llx%016llx%016llx%016llx%016llx%016llx // "
         "0x%016llx%016llx%016llx%016llx%016llx%016llx%016llx%016llx\n",
         quotient[7], quotient[6], quotient[5], quotient[4], quotient[3],
         quotient[2], quotient[1], quotient[0], dividend[7], dividend[6],
         dividend[5], dividend[4], dividend[3], dividend[2], dividend[1],
         dividend[0], divisor[7], divisor[6], divisor[5], divisor[4],
         divisor[3], divisor[2], divisor[1], divisor[0]);
  printf("assert 0x%016llx%016llx%016llx%016llx%016llx%016llx%016llx%016llx "
         "== 0x%016llx%016llx%016llx%016llx%016llx%016llx%016llx%016llx %% "
         "0x%016llx%016llx%016llx%016llx%016llx%016llx%016llx%016llx\n",
         remainder[7], remainder[6], remainder[5], remainder[4], remainder[3],
         remainder[2], remainder[1], remainder[0], dividend[7], dividend[6],
         dividend[5], dividend[4], dividend[3], dividend[2], dividend[1],
         dividend[0], divisor[7], divisor[6], divisor[5], divisor[4],
         divisor[3], divisor[2], divisor[1], divisor[0]);

  printf("# [u512Div] end\n");
#endif
}

__forceinline__ __device__ void u256ModP(u64 output[4], const u64 a[4],
                                         const u64 p[4]) {
  u64 quotient[4] = {0};
  u64 remainder[4] = {0};

  u256Div(quotient, remainder, a, p);

  u256Copy(output, remainder);
}

__forceinline__ __device__ void u512ModU256P(u64 output[4], const u64 a[8],
                                             const u64 p[4]) {
#ifdef DEBUG_U512
  printf("# [u512ModU256P] start\n");
#endif

  u64 quotient[8] = {0};
  u64 outputExtended[8] = {0};
  u64 pExtended[8] = {0};

  u256Extend(pExtended, p);

  u512Div(quotient, outputExtended, a, pExtended);

  u256Copy(output, outputExtended);

#ifdef DEBUG_U512
  printf("assert 0x%016llx%016llx%016llx%016llx == "
         "0x%016llx%016llx%016llx%016llx%016llx%016llx%016llx%016llx %% "
         "0x%016llx%016llx%016llx%016llx\n",
         output[3], output[2], output[1], output[0], a[7], a[6], a[5], a[4],
         a[3], a[2], a[1], a[0], p[3], p[2], p[1], p[0]);

  printf("# [u512ModU256P] end\n");
#endif
}

__forceinline__ __device__ void u256AddModP(u64 output[4], const u64 a[4],
                                            const u64 b[4], const u64 p[4]) {
#ifdef DEBUG_U256
  printf("# [u256AddModP: start]\n");
#endif
  u64 added[4];
  int carry = u256Add(added, a, b);

  u64 extended[8];
  u256Extend(extended, added);

  if (carry) {
    extended[4] = 1;
  }

  u512ModU256P(output, extended, p);
#ifdef DEBUG_U256
  printf("assert (0x%016llx%016llx%016llx%016llx + "
         "0x%016llx%016llx%016llx%016llx) %% 0x%016llx%016llx%016llx%016llx == "
         "0x%016llx%016llx%016llx%016llx\n",
         a[3], a[2], a[1], a[0], b[3], b[2], b[1], b[0], p[3], p[2], p[1], p[0],
         output[3], output[2], output[1], output[0]);
  printf("# [u256AddModP: end]\n");
#endif
}

__forceinline__ __device__ void u256SubModP(u64 output[4], const u64 a[4],
                                            const u64 b[4], const u64 p[4]) {
  u64 result[4] = {0};

  u64 borrow = 0;

  for (int i = 0; i < 4; ++i) {
    borrow = u64Sub(&result[i], a[i], b[i], borrow);
  }

  if (borrow) {
    u64 t0[4];
    u256Copy(t0, result);
    u256Add(result, t0, p);
  }

  u256ModP(result, result, p);

  u256Copy(output, result);
}

__forceinline__ __device__ void u256MulModP(u64 output[4], const u64 a[4],
                                            const u64 b[4], const u64 p[4]) {

  u64 multiplied[8];
  u256Mul(multiplied, a, b);

  u64 result[4];
  u512ModU256P(result, multiplied, p);

  u256Copy(output, result);
}

__forceinline__ __device__ void u256PowModP(u64 output[4], const u64 a[4],
                                            const u64 b[4], const u64 p[4]) {
#ifdef DEBUG_U256
  printf("# [u256PowModP: start]\n");
#endif

  u64 result[4] = {1};

  u64 aModP[4];
  u256ModP(aModP, a, p);

  if (u256IsZero(aModP)) {

    u256SetZero(result);
  } else {
    u64 bCopy[4];
    u256Copy(bCopy, b);

    for (int i = 0; i < 256; ++i) {
      if (u256GetBit(bCopy, i)) {
        u256MulModP(result, result, aModP, p);
      }

      u256MulModP(aModP, aModP, aModP, p);
    }
  }

  u256Copy(output, result);

#ifdef DEBUG_U256
  printf("assert pow(0x%016llx%016llx%016llx%016llx, "
         "0x%016llx%016llx%016llx%016llx, 0x%016llx%016llx%016llx%016llx) == "
         "0x%016llx%016llx%016llx%016llx\n",
         a[3], a[2], a[1], a[0], b[3], b[2], b[1], b[0], p[3], p[2], p[1], p[0],
         output[3], output[2], output[1], output[0]);

  printf("# [u256PowModP: end]\n");
#endif
}

__forceinline__ __device__ void u256InvModP(u64 output[4], const u64 a[4],
                                            const u64 p[4]) {
#ifdef DEBUG_U256
  printf("# [u256InvModP: start]\n");
#endif

  u64 result[4];
  u64 pMinus2[4] = {0};

  u256Sub(pMinus2, p, (u64[4]){2});

  u256PowModP(result, a, pMinus2, p);

  u256Copy(output, result);

#ifdef DEBUG_U256
  printf("assert pow(0x%016llx%016llx%016llx%016llx, -1, "
         "0x%016llx%016llx%016llx%016llx) == 0x%016llx%016llx%016llx%016llx\n",
         a[3], a[2], a[1], a[0], p[3], p[2], p[1], p[0], output[3], output[2],
         output[1], output[0]);

  printf("# [u256InvModP: end]\n");
#endif
}

#endif
