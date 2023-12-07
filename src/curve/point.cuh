#ifndef _POINT_CUH
#define _POINT_CUH

#include "../config.h"
#include "../uint/u256.cuh"
#include "fp.cuh"

#ifdef DEBUG_POINT
#include <stdio.h>
#endif

typedef struct {
  u64 x[4];
  u64 y[4];
} Point;

__forceinline__ __device__ void pointSetZero(Point *output) {
  u256SetZero(output->x);
  u256SetZero(output->y);
}

__forceinline__ __device__ bool pointIsZero(const Point *output) {
  return fpIsZero(output->x) && fpIsZero(output->y);
}

__forceinline__ __device__ void pointCopy(Point *output, const Point *p) {
  u256Copy(output->x, p->x);
  u256Copy(output->y, p->y);
}

__forceinline__ __device__ void pointAdd(Point *output, const Point *p,
                                         const Point *q, const u64 prime[4],
                                         const u64 a[4], const u64 b[4]) {
#ifdef DEBUG_POINT
  printf("# [pointAdd: start]\n");
#endif

  if (pointIsZero(p)) {
    pointCopy(output, q);

    return;
  } else if (pointIsZero(q)) {
    pointCopy(output, p);

    return;
  } else if (u256Compare(p->x, q->x) == 0) {
    u64 negQY[4];

    fpNeg(negQY, q->y, prime);

    if (u256Compare(p->y, negQY) == 0) {
      pointSetZero(output);

      return;
    }
  }

  Point temp;

  u64 s[4];

  if (u256Compare(p->y, q->y) == 0) {
    u64 squaredPX[4];
    fpMul(squaredPX, p->x, p->x, prime);

    u64 doubledPY[4];
    fpAdd(doubledPY, p->y, p->y, prime);

    u64 s0[4];
    fpAdd(s0, squaredPX, squaredPX, prime);

    u64 s1[4];
    fpAdd(s1, s0, squaredPX, prime);

    u64 s2[4];
    fpAdd(s2, s1, a, prime);

    fpDiv(s, s2, doubledPY, prime);

  } else {
    u64 diffX[4];
    u64 diffY[4];

    fpSub(diffX, q->x, p->x, prime);
    fpSub(diffY, q->y, p->y, prime);

    fpDiv(s, diffY, diffX, prime);
  }

  u64 x0[4];
  fpMul(x0, s, s, prime);

  u64 x1[4];
  fpSub(x1, x0, p->x, prime);

  fpSub(temp.x, x1, q->x, prime);

  u64 y0[4];
  fpSub(y0, p->x, temp.x, prime);

  u64 y1[4];
  fpMul(y1, y0, s, prime);
  fpSub(temp.y, y1, p->y, prime);

  pointCopy(output, &temp);

#ifdef DEBUG_POINT
  printf("assert Pt(Fq(0x%016llx%016llx%016llx%016llx), "
         "Fq(0x%016llx%016llx%016llx%016llx)) + "
         "Pt(Fq(0x%016llx%016llx%016llx%016llx), "
         "Fq(0x%016llx%016llx%016llx%016llx)) == "
         "Pt(Fq(0x%016llx%016llx%016llx%016llx), "
         "Fq(0x%016llx%016llx%016llx%016llx))\n",
         p->x[3], p->x[2], p->x[1], p->x[0], p->y[3], p->y[2], p->y[1], p->y[0],
         q->x[3], q->x[2], q->x[1], q->x[0], q->y[3], q->y[2], q->y[1], q->y[0],
         output->x[3], output->x[2], output->x[1], output->x[0], output->y[3],
         output->y[2], output->y[1], output->y[0]);

  printf("# [pointAdd: end]\n");
#endif
}

__forceinline__ __device__ void pointNeg(Point *output, const Point *p,
                                         const u64 prime[4]) {
#ifdef DEBUG_POINT
  printf("# [pointNeg: start]\n");
#endif

  u256Copy(output->x, p->x);
  fpNeg(output->y, p->y, prime);

#ifdef DEBUG_POINT
  printf("assert -Pt(Fq(0x%016llx%016llx%016llx%016llx), "
         "Fq(0x%016llx%016llx%016llx%016llx)) == "
         "Pt(Fq(0x%016llx%016llx%016llx%016llx), "
         "Fq(0x%016llx%016llx%016llx%016llx))\n",
         p->x[3], p->x[2], p->x[1], p->x[0], p->y[3], p->y[2], p->y[1], p->y[0],
         output->x[3], output->x[2], output->x[1], output->x[0], output->y[3],
         output->y[2], output->y[1], output->y[0]);

  printf("# [pointNeg: end]\n");
#endif
}

__forceinline__ __device__ void pointSub(Point *output, const Point *p,
                                         const Point *q, const u64 prime[4],
                                         const u64 a[4], const u64 b[4]) {
#ifdef DEBUG_POINT
  printf("# [pointSub: start]\n");
#endif

  Point qNeg;
  pointNeg(&qNeg, q, prime);
  pointAdd(output, p, &qNeg, prime, a, b);

#ifdef DEBUG_POINT
  printf("assert Pt(Fq(0x%016llx%016llx%016llx%016llx), "
         "Fq(0x%016llx%016llx%016llx%016llx)) - "
         "Pt(Fq(0x%016llx%016llx%016llx%016llx), "
         "Fq(0x%016llx%016llx%016llx%016llx)) == "
         "Pt(Fq(0x%016llx%016llx%016llx%016llx), "
         "Fq(0x%016llx%016llx%016llx%016llx))\n",
         p->x[3], p->x[2], p->x[1], p->x[0], p->y[3], p->y[2], p->y[1], p->y[0],
         q->x[3], q->x[2], q->x[1], q->x[0], q->y[3], q->y[2], q->y[1], q->y[0],
         output->x[3], output->x[2], output->x[1], output->x[0], output->y[3],
         output->y[2], output->y[1], output->y[0]);

  printf("# [pointSub: end]\n");
#endif
}

__forceinline__ __device__ void pointMul(Point *output, const Point *p,
                                         const u64 k[4], const u64 prime[4],
                                         const u64 a[4], const u64 b[4]) {
#ifdef DEBUG_POINT
  printf("# [pointMul: start]\n");
#endif

  pointSetZero(output);

  if (pointIsZero(p) || fpIsZero(k)) {
    return;
  }

  Point q;

  pointCopy(&q, p);

  for (int i = 0; i < 256; i++) {
    if (u256GetBit(k, i)) {
      Point temp;
      pointCopy(&temp, output);
      pointAdd(output, &temp, &q, prime, a, b);
    }
    Point temp;
    pointCopy(&temp, &q);
    pointAdd(&q, &temp, &temp, prime, a, b);
  }

#ifdef DEBUG_POINT
  printf("assert Pt(Fq(0x%016llx%016llx%016llx%016llx), "
         "Fq(0x%016llx%016llx%016llx%016llx)) * "
         "Fq(0x%016llx%016llx%016llx%016llx) == "
         "Pt(Fq(0x%016llx%016llx%016llx%016llx), "
         "Fq(0x%016llx%016llx%016llx%016llx))\n",
         p->x[3], p->x[2], p->x[1], p->x[0], p->y[3], p->y[2], p->y[1], p->y[0],
         k[3], k[2], k[1], k[0], output->x[3], output->x[2], output->x[1],
         output->x[0], output->y[3], output->y[2], output->y[1], output->y[0]);

  printf("# [pointMul: end]\n");
#endif
}

__forceinline__ __device__ bool pointEqual(const Point *p, const Point *q) {
#ifdef DEBUG_POINT
  printf("# [pointEqual: start]\n");
#endif

  bool result = u256Compare(p->x, q->x) == 0 && u256Compare(p->y, q->y) == 0;

#ifdef DEBUG_POINT
  printf("assert Pt(Fq(0x%016llx%016llx%016llx%016llx), "
         "Fq(0x%016llx%016llx%016llx%016llx)) == "
         "Pt(Fq(0x%016llx%016llx%016llx%016llx), "
         "Fq(0x%016llx%016llx%016llx%016llx))\n",
         p->x[3], p->x[2], p->x[1], p->x[0], p->y[3], p->y[2], p->y[1], p->y[0],
         q->x[3], q->x[2], q->x[1], q->x[0], q->y[3], q->y[2], q->y[1],
         q->y[0]);
#endif

  return result;
}

#endif
