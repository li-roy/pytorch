#ifndef THC_HALF_AUTO_NUMERICS_INC
#define THC_HALF_AUTO_NUMERICS_INC

#include "THCHalf.h"
#include "THCNumerics.cuh"

// WARNING: THCNumerics is being deprecated. Read the comments and function usage
//          in THCNumerics to learn about the deprecation
//
// Half numerics functions defined as free functions, so cunn code can be
//written generically, i.e. without excessive calling of THCNumerics<half> functions.

// these functions should move to THCNumerics

inline __host__ __device__ at::Half fmaxType(at::Half x, at::Half y) {
  return THCNumerics<at::Half>::ge(x, y) ? x : y;
}

inline __host__ __device__ float fmaxType(float x, at::Half y) {
  return fmaxf(x, ScalarConvert<at::Half, float>::to(y));
}

inline __host__ __device__ float fmaxType(float x, float y) {
  return fmaxf(x, y);
}

inline __host__ __device__ double fmaxType(double x, double y) {
  return fmax(x, y);
}

#endif
