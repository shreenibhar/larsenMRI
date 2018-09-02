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

void XAyBatched(precision **XA, precision *y, precision *r, int *nVars, int M, int numModels);

void IrBatched(precision **I, precision *r, precision *betaOls, int *nVars, int M, int numModels, int maxVar);

void XAbetaOlsBatched(precision **XA, precision *betaOls, precision *d, int *nVars, int M, int numModels, int maxVar);

void fabsMaxReduce(precision *mat, precision *res, precision *buf, int *ind, int *intBuf, int rowSize, int colSize);

void cdMinReduce(precision *c, precision *cd, precision *cmax, precision *res, precision *buf, int rowSize, int colSize);

#endif
