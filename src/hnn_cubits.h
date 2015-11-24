
#ifndef __HNN_CUBITS_H__
#define __HNN_CUBITS_H__

#ifdef __cplusplus
extern "C"
#endif
void thresh(float *array, size_t size, float thresh, float *out);

#ifdef __cplusplus
extern "C"
#endif
void threshDouble(double *array, size_t size, double thresh, double *out);

#ifdef __cplusplus
extern "C"
#endif
void mul(float *a1, float *a2, size_t size);

#ifdef __cplusplus
extern "C"
#endif
void mulDouble(double *a1, double *a2, size_t size);

#ifdef __cplusplus 
extern "C" 
#endif
void add(float *a1, float *a2, size_t size);

#ifdef __cplusplus 
extern "C" 
#endif
void addDouble(double *a1, double *a2, size_t size);

#ifdef __cplusplus 
extern "C" 
#endif
void tabs(float *a, size_t size);

#ifdef __cplusplus 
extern "C" 
#endif
void tabsDouble(double *a, size_t size);

#ifdef __cplusplus 
extern "C" 
#endif
void signum(float *a, size_t size);

#ifdef __cplusplus 
extern "C" 
#endif
void signumDouble(double *a, size_t size);

#ifdef __cplusplus 
extern "C" 
#endif
void subtract(float *a1, float *a2, size_t size);

#ifdef __cplusplus 
extern "C" 
#endif
void subtractDouble(double *a1, double *a2, size_t size);

#ifdef __cplusplus
extern "C"
#endif
void negate(float *a, size_t size);

#ifdef __cplusplus
extern "C"
#endif
void negateDouble(double *a, size_t size);

#ifdef __cplusplus
extern "C"
#endif
void scale(float s, float *a, size_t size);

#ifdef __cplusplus
extern "C"
#endif
void scaleDouble(double s, double *a, size_t size);

#ifdef __cplusplus
extern "C"
#endif
void logFloat(float *a, size_t size);

#ifdef __cplusplus
extern "C"
#endif
void logDouble(double *a, size_t size);

#ifdef __cplusplus
extern "C"
#endif
void inv(float *a, size_t size);

#ifdef __cplusplus
extern "C"
#endif
void invDouble(double *a, size_t size);

#ifdef __cplusplus
extern "C"
#endif
void texp(float *a, size_t size);

#ifdef __cplusplus
extern "C"
#endif
void texpDouble(double *a, size_t size);

#ifdef __cplusplus
extern "C"
#endif
void tsqrt(float *a, size_t size);

#ifdef __cplusplus
extern "C"
#endif
void tsqrtDouble(double *a, size_t size);

#ifdef __cplusplus
extern "C"
#endif
void tsin(float *a, size_t size);

#ifdef __cplusplus
extern "C"
#endif
void tsinDouble(double *a, size_t size);

#ifdef __cplusplus
extern "C"
#endif
void tcos(float *a, size_t size);

#ifdef __cplusplus
extern "C"
#endif
void tcosDouble(double *a, size_t size);

#ifdef __cplusplus
extern "C"
#endif
void ttan(float *a, size_t size);

#ifdef __cplusplus
extern "C"
#endif
void ttanDouble(double *a, size_t size);

#ifdef __cplusplus
extern "C"
#endif
void tasin(float *a, size_t size);

#ifdef __cplusplus
extern "C"
#endif
void tasinDouble(double *a, size_t size);

#ifdef __cplusplus
extern "C"
#endif
void tacos(float *a, size_t size);

#ifdef __cplusplus
extern "C"
#endif
void tacosDouble(double *a, size_t size);

#ifdef __cplusplus
extern "C"
#endif
void tatan(float *a, size_t size);

#ifdef __cplusplus
extern "C"
#endif
void tatanDouble(double *a, size_t size);

#ifdef __cplusplus
extern "C"
#endif
void tsinh(float *a, size_t size);

#ifdef __cplusplus
extern "C"
#endif
void tsinhDouble(double *a, size_t size);

#ifdef __cplusplus
extern "C"
#endif
void tcosh(float *a, size_t size);

#ifdef __cplusplus
extern "C"
#endif
void tcoshDouble(double *a, size_t size);

#ifdef __cplusplus
extern "C"
#endif
void ttanh(float *a, size_t size);

#ifdef __cplusplus
extern "C"
#endif
void ttanhDouble(double *a, size_t size);

#ifdef __cplusplus
extern "C"
#endif
void tasinh(float *a, size_t size);

#ifdef __cplusplus
extern "C"
#endif
void tasinhDouble(double *a, size_t size);

#ifdef __cplusplus
extern "C"
#endif
void tacosh(float *a, size_t size);

#ifdef __cplusplus
extern "C"
#endif
void tacoshDouble(double *a, size_t size);

#ifdef __cplusplus
extern "C"
#endif
void tatanh(float *a, size_t size);

#ifdef __cplusplus
extern "C"
#endif
void tatanhDouble(double *a, size_t size);

#ifdef __cplusplus
extern "C"
#endif
void tpow(float *a1, float *a2, size_t size);

#ifdef __cplusplus
extern "C"
#endif
void tpowDouble(double *a1, double *a2, size_t size);

#ifdef __cplusplus
extern "C"
#endif
void tmax(float *a1, float *a2, size_t size);

#ifdef __cplusplus
extern "C"
#endif
void tmaxDouble(double *a1, double *a2, size_t size);

#endif
