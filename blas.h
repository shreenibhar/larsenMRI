#ifndef BLAS_H
#define BLAS_H

#include <bits/stdc++.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "utilities.h"

void gemm(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, float *alpha, float *A, int lda, float *B, int ldb, float *beta, float *C, int ldc);

void gemm(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, double *alpha, double *A, int lda, double *B, int ldb, double *beta, double *C, int ldc);

void gemmBatched(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, float *alpha, float *Aarray[], int lda, float *Barray[], int ldb, float *beta, float *Carray[], int ldc, int batchCount);

void gemmBatched(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, double *alpha, double *Aarray[], int lda, double *Barray[], int ldb, double *beta, double *Carray[], int ldc, int batchCount);

void gemv(cublasHandle_t handle, cublasOperation_t trans, int m, int n, float *alpha, float *A, int lda, float *x, int incx, float *beta, float *y, int incy);

void gemv(cublasHandle_t handle, cublasOperation_t trans, int m, int n, double *alpha, double *A, int lda, double *x, int incx, double *beta, double *y, int incy);

void getrfBatched(cublasHandle_t handle, int n, float *Aarray[], int lda, int *PivotArray, int *infoArray, int batchSize);

void getrfBatched(cublasHandle_t handle, int n, double *Aarray[], int lda, int *PivotArray, int *infoArray, int batchSize);

void getriBatched(cublasHandle_t handle, int n, float *Aarray[], int lda, int *PivotArray, float *Carray[], int ldc, int *infoArray, int batchSize);

void getriBatched(cublasHandle_t handle, int n, double *Aarray[], int lda, int *PivotArray, double *Carray[], int ldc, int *infoArray, int batchSize);

void amax(cublasHandle_t handle, int n, const float *x, int incx, int *result);

void amax(cublasHandle_t handle, int n, const double *x, int incx, int *result);

void amin(cublasHandle_t handle, int n, const float *x, int incx, int *result);

void amin(cublasHandle_t handle, int n, const double *x, int incx, int *result);

template<typename T>
void XAyBatched(T **XA, T *y, T *r, int *nVars, int M, int numModels);

template<typename T>
void IrBatched(T **I, T *r, T *betaOls, int *nVars, int M, int numModels, int maxVar);

template<typename T>
void XAbetaOlsBatched(T **XA, T *betaOls, T *d, int *nVars, int M, int numModels, int maxVar);

template<typename T>
void fabsMaxReduce(T *mat, T *res, T *buf, int *ind, int *intBuf, int rowSize, int colSize);

template<typename T>
void cdMinReduce(T *c, T *cd, T *cmax, T *res, T *buf, int rowSize, int colSize);

#endif
