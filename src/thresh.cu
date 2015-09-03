#include "hnn_cubits.h"

__global__ void thresh_kernel(float *array, float thresh, float *out) {
  int i = threadIdx.x;
  
  if (array[i] < thresh) {
    out[i] = 0;
  } else {
    out[i] = 1;
  }
}

__global__ void threshDouble_kernel(double *array, double thresh, double *out) {
  int i = threadIdx.x;
  
  if (array[i] < thresh) {
    out[i] = 0;
  } else {
    out[i] = 1;
  }
}

extern "C"
void thresh(float *array, size_t size, float thresh, float *out) {
  thresh_kernel<<<1, size>>>(array, thresh, out);
}

extern "C"
void threshDouble(double *array, size_t size, double thresh, double *out) {
  threshDouble_kernel<<<1, size>>>(array, thresh, out);
}
