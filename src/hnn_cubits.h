
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

#endif
