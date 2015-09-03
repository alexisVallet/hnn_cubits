#include "hnn_cubits.h"

__global__ void mul_kernel(float *a1, float *a2) {
  int i = threadIdx.x;

  a2[i] = a1[i] * a2[i];
}

__global__ void mulDouble_kernel(double *a1, double *a2) {
  int i = threadIdx.x;

  a2[i] = a1[i] * a2[i];
}

extern "C"
void mul(float *a1, float *a2, size_t size) {
  mul_kernel<<<1, size>>>(a1, a2);
}

extern "C"
void mulDouble(double *a1, double *a2, size_t size) {
  mulDouble_kernel<<<1, size>>>(a1, a2);
}

__global__ void add_kernel(float *a1, float *a2) {
  int i = threadIdx.x;

  a2[i] = a1[i] + a2[i];
}

__global__ void addDouble_kernel(double *a1, double *a2) {
  int i = threadIdx.x;

  a2[i] = a1[i] + a2[i];
}

extern "C"
void add(float *a1, float *a2, size_t size) {
  add_kernel<<<1, size>>>(a1, a2);
}

extern "C"
void addDouble(double *a1, double *a2, size_t size) {
  addDouble_kernel<<<1, size>>>(a1, a2);
}

__global__ void abs_kernel(float *a) {
  int i = threadIdx.x;

  a[i] = fabsf(a[i]);
}

__global__ void absDouble_kernel(double *a) {
  int i = threadIdx.x;

  a[i] = fabs(a[i]);
}

extern "C"
void tabs(float *a, size_t size) {
  abs_kernel<<<1, size>>>(a);
}

extern "C"
void tabsDouble(double *a, size_t size) {
  absDouble_kernel<<<1, size>>>(a);
}

__global__ void signum_kernel(float *a) {
  int i = threadIdx.x;

  a[i] = signbit(a[i]) ? -1 : 1;
}

__global__ void signumDouble_kernel(double *a) {
  int i = threadIdx.x;

  a[i] = signbit(a[i]) ? -1 : 1;
}

extern "C"
void signum(float *a, size_t size) {
  signum_kernel<<<1, size>>>(a);
}

extern "C"
void signumDouble(double *a, size_t size) {
  signumDouble_kernel<<<1, size>>>(a);
}

__global__ void subtract_kernel(float *a1, float *a2) {
  int i = threadIdx.x;

  a2[i] = a1[i] - a2[i];
}

__global__ void subtractDouble_kernel(double *a1, double *a2) {
  int i = threadIdx.x;

  a2[i] = a1[i] - a2[i];
}

extern "C"
void subtract(float *a1, float *a2, size_t size) {
  subtract_kernel<<<1, size>>>(a1, a2);
}

extern "C"
void subtractDouble(double *a1, double *a2, size_t size) {
  subtractDouble_kernel<<<1, size>>>(a1, a2);
}

__global__ void negate_kernel(float *a) {
  int i = threadIdx.x;

  a[i] = -a[i];
}

__global__ void negateDouble_kernel(double *a) {
  int i = threadIdx.x;

  a[i] = -a[i];
}

extern "C"
void negate(float *a, size_t size) {
  negate_kernel<<<1, size>>>(a);
}

extern "C"
void negateDouble(double *a, size_t size) {
  negateDouble_kernel<<<1, size>>>(a);
}

__global__ void scale_kernel(float s, float *a) {
  int i = threadIdx.x;

  a[i] *= s;
}

__global__ void scaleDouble_kernel(double s, double *a) {
  int i = threadIdx.x;

  a[i] *= s;
}

extern "C"
void scale(float s, float *a, size_t size) {
  scale_kernel<<<1, size>>>(s, a);
}

extern "C"
void scaleDouble(double s, double *a, size_t size) {
  scaleDouble_kernel<<<1, size>>>(s, a);
}
