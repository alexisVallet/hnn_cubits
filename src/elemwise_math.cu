#include "hnn_cubits.h"

#define BLOCK_SIZE 256

__global__ void mul_kernel(float *a1, float *a2, size_t size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < size)
    a2[i] = a1[i] * a2[i];
}

__global__ void mulDouble_kernel(double *a1, double *a2, size_t size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < size)
    a2[i] = a1[i] * a2[i];
}

extern "C"
void mul(float *a1, float *a2, size_t size) {
  mul_kernel<<< (size + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(a1, a2, size);
}

extern "C"
void mulDouble(double *a1, double *a2, size_t size) {
  mulDouble_kernel<<< (size + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(a1, a2, size);
}

__global__ void add_kernel(float *a1, float *a2, size_t size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < size)
    a2[i] = a1[i] + a2[i];
}

__global__ void addDouble_kernel(double *a1, double *a2, size_t size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < size)
    a2[i] = a1[i] + a2[i];
}

extern "C"
void add(float *a1, float *a2, size_t size) {
  add_kernel<<< (size + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(a1, a2, size);
}

extern "C"
void addDouble(double *a1, double *a2, size_t size) {
  addDouble_kernel<<< (size + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(a1, a2, size);
}

__global__ void abs_kernel(float *a, size_t size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < size)
    a[i] = fabsf(a[i]);
}

__global__ void absDouble_kernel(double *a, size_t size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < size)
    a[i] = fabs(a[i]);
}

extern "C"
void tabs(float *a, size_t size) {
  abs_kernel<<< (size + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(a, size);
}

extern "C"
void tabsDouble(double *a, size_t size) {
  absDouble_kernel<<< (size + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(a, size);
}

__global__ void signum_kernel(float *a, size_t size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < size)
    a[i] = signbit(a[i]) ? -1 : 1;
}

__global__ void signumDouble_kernel(double *a, size_t size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < size)
    a[i] = signbit(a[i]) ? -1 : 1;
}

extern "C"
void signum(float *a, size_t size) {
  signum_kernel<<< (size + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(a, size);
}

extern "C"
void signumDouble(double *a, size_t size) {
  signumDouble_kernel<<< (size + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(a, size);
}

__global__ void subtract_kernel(float *a1, float *a2, size_t size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < size)
    a2[i] = a1[i] - a2[i];
}

__global__ void subtractDouble_kernel(double *a1, double *a2, size_t size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < size)
    a2[i] = a1[i] - a2[i];
}

extern "C"
void subtract(float *a1, float *a2, size_t size) {
  subtract_kernel<<< (size + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(a1, a2, size);
}

extern "C"
void subtractDouble(double *a1, double *a2, size_t size) {
  subtractDouble_kernel<<< (size + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(a1, a2, size);
}

__global__ void negate_kernel(float *a, size_t size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < size)
    a[i] = -a[i];
}

__global__ void negateDouble_kernel(double *a, size_t size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < size)
    a[i] = -a[i];
}

extern "C"
void negate(float *a, size_t size) {
  negate_kernel<<< (size + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(a, size);
}

extern "C"
void negateDouble(double *a, size_t size) {
  negateDouble_kernel<<< (size + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(a, size);
}

__global__ void scale_kernel(float s, float *a, size_t size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < size)
    a[i] *= s;
}

__global__ void scaleDouble_kernel(double s, double *a, size_t size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < size)
    a[i] *= s;
}

extern "C"
void scale(float s, float *a, size_t size) {
  scale_kernel<<< (size + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(s, a, size);
}

extern "C"
void scaleDouble(double s, double *a, size_t size) {
  scaleDouble_kernel<<< (size + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(s, a, size);
}

__global__ void log_kernel(float *a, size_t size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < size)
    a[i] = log(a[i]);
}

__global__ void logDouble_kernel(double *a, size_t size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < size)
    a[i] = log(a[i]);
}

extern "C"
void logFloat(float *a, size_t size) {
  log_kernel<<< (size + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(a, size);
}

extern "C"
void logDouble(double *a, size_t size) {
  logDouble_kernel<<< (size + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(a, size);
}

__global__ void inv_kernel(float *a, size_t size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < size)
    a[i] = 1. / a[i];
}

__global__ void invDouble_kernel(double *a, size_t size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < size)
    a[i] = 1. / a[i];
}

extern "C"
void inv(float *a, size_t size) {
  inv_kernel<<< (size + BLOCK_SIZE - 1) / BLOCK_SIZE,BLOCK_SIZE>>>(a, size);
}

extern "C"
void invDouble(double *a, size_t size) {
  invDouble_kernel<<< (size + BLOCK_SIZE - 1) / BLOCK_SIZE,BLOCK_SIZE>>>(a, size);
}

__global__ void exp_kernel(float *a, size_t size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < size)
    a[i] = exp(a[i]);
}

__global__ void expDouble_kernel(double *a, size_t size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < size)
    a[i] = exp(a[i]);
}

extern "C"
void texp(float *a, size_t size) {
  exp_kernel<<<(size + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(a,size);
}

extern "C"
void texpDouble(double *a, size_t size) {
  expDouble_kernel<<<(size + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(a, size);
}
