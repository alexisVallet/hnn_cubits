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

__global__ void sqrt_kernel(float *a, size_t size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < size)
    a[i] = sqrt(a[i]);
}

__global__ void sqrtDouble_kernel(double *a, size_t size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < size)
    a[i] = sqrt(a[i]);
}

extern "C"
void tsqrt(float *a, size_t size) {
  sqrt_kernel<<<(size + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(a,size);
}

extern "C"
void tsqrtDouble(double *a, size_t size) {
  sqrtDouble_kernel<<<(size + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(a, size);
}

__global__ void cos_kernel(float *a, size_t size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < size)
    a[i] = cos(a[i]);
}

__global__ void cosDouble_kernel(double *a, size_t size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < size)
    a[i] = cos(a[i]);
}

extern "C"
void tcos(float *a, size_t size) {
  cos_kernel<<<(size + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(a,size);
}

extern "C"
void tcosDouble(double *a, size_t size) {
  cosDouble_kernel<<<(size + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(a, size);
}

__global__ void sin_kernel(float *a, size_t size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < size)
    a[i] = sin(a[i]);
}

__global__ void sinDouble_kernel(double *a, size_t size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < size)
    a[i] = sin(a[i]);
}

extern "C"
void tsin(float *a, size_t size) {
  sin_kernel<<<(size + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(a,size);
}

extern "C"
void tsinDouble(double *a, size_t size) {
  sinDouble_kernel<<<(size + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(a, size);
}

__global__ void tan_kernel(float *a, size_t size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < size)
    a[i] = tan(a[i]);
}

__global__ void tanDouble_kernel(double *a, size_t size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < size)
    a[i] = tan(a[i]);
}

extern "C"
void ttan(float *a, size_t size) {
  tan_kernel<<<(size + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(a,size);
}

extern "C"
void ttanDouble(double *a, size_t size) {
  tanDouble_kernel<<<(size + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(a, size);
}

__global__ void asin_kernel(float *a, size_t size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < size)
    a[i] = asin(a[i]);
}

__global__ void asinDouble_kernel(double *a, size_t size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < size)
    a[i] = asin(a[i]);
}

extern "C"
void tasin(float *a, size_t size) {
  asin_kernel<<<(size + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(a,size);
}

extern "C"
void tasinDouble(double *a, size_t size) {
  asinDouble_kernel<<<(size + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(a, size);
}

__global__ void acos_kernel(float *a, size_t size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < size)
    a[i] = acos(a[i]);
}

__global__ void acosDouble_kernel(double *a, size_t size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < size)
    a[i] = acos(a[i]);
}

extern "C"
void tacos(float *a, size_t size) {
  acos_kernel<<<(size + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(a,size);
}

extern "C"
void tacosDouble(double *a, size_t size) {
  acosDouble_kernel<<<(size + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(a, size);
}

__global__ void atan_kernel(float *a, size_t size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < size)
    a[i] = atan(a[i]);
}

__global__ void atanDouble_kernel(double *a, size_t size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < size)
    a[i] = atan(a[i]);
}

extern "C"
void tatan(float *a, size_t size) {
  atan_kernel<<<(size + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(a,size);
}

extern "C"
void tatanDouble(double *a, size_t size) {
  atanDouble_kernel<<<(size + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(a, size);
}

__global__ void sinh_kernel(float *a, size_t size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < size)
    a[i] = sinh(a[i]);
}

__global__ void sinhDouble_kernel(double *a, size_t size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < size)
    a[i] = sinh(a[i]);
}

extern "C"
void tsinh(float *a, size_t size) {
  sinh_kernel<<<(size + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(a,size);
}

extern "C"
void tsinhDouble(double *a, size_t size) {
  sinhDouble_kernel<<<(size + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(a, size);
}

__global__ void cosh_kernel(float *a, size_t size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < size)
    a[i] = cosh(a[i]);
}

__global__ void coshDouble_kernel(double *a, size_t size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < size)
    a[i] = cosh(a[i]);
}

extern "C"
void tcosh(float *a, size_t size) {
  cosh_kernel<<<(size + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(a,size);
}

extern "C"
void tcoshDouble(double *a, size_t size) {
  coshDouble_kernel<<<(size + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(a, size);
}

__global__ void tanh_kernel(float *a, size_t size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < size)
    a[i] = tanh(a[i]);
}

__global__ void tanhDouble_kernel(double *a, size_t size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < size)
    a[i] = tanh(a[i]);
}

extern "C"
void ttanh(float *a, size_t size) {
  tanh_kernel<<<(size + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(a,size);
}

extern "C"
void ttanhDouble(double *a, size_t size) {
  tanhDouble_kernel<<<(size + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(a, size);
}

__global__ void asinh_kernel(float *a, size_t size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < size)
    a[i] = asinh(a[i]);
}

__global__ void asinhDouble_kernel(double *a, size_t size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < size)
    a[i] = asinh(a[i]);
}

extern "C"
void tasinh(float *a, size_t size) {
  asinh_kernel<<<(size + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(a,size);
}

extern "C"
void tasinhDouble(double *a, size_t size) {
  asinhDouble_kernel<<<(size + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(a, size);
}

__global__ void acosh_kernel(float *a, size_t size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < size)
    a[i] = acosh(a[i]);
}

__global__ void acoshDouble_kernel(double *a, size_t size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < size)
    a[i] = acosh(a[i]);
}

extern "C"
void tacosh(float *a, size_t size) {
  acosh_kernel<<<(size + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(a,size);
}

extern "C"
void tacoshDouble(double *a, size_t size) {
  acoshDouble_kernel<<<(size + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(a, size);
}

__global__ void atanh_kernel(float *a, size_t size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < size)
    a[i] = atanh(a[i]);
}

__global__ void atanhDouble_kernel(double *a, size_t size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < size)
    a[i] = atanh(a[i]);
}

extern "C"
void tatanh(float *a, size_t size) {
  atanh_kernel<<<(size + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(a,size);
}

extern "C"
void tatanhDouble(double *a, size_t size) {
  atanhDouble_kernel<<<(size + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(a, size);
}

__global__ void pow_kernel(float *a1, float *a2, size_t size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < size)
    a2[i] = pow(a1[i], a2[i]);
}

__global__ void powDouble_kernel(double *a1, double *a2, size_t size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < size)
    a2[i] = pow(a1[i], a2[i]);
}

extern "C"
void tpow(float *a1, float *a2, size_t size) {
  pow_kernel<<<(size + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(a1,a2,size);
}

extern "C"
void tpowDouble(double *a1, double *a2, size_t size) {
  powDouble_kernel<<<(size + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(a1,a2,size);
}

