#ifndef BLAS_H
#define BLAS_H

#include "headers.h"


void gemm(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, float *alpha, float *A, int lda, float *B, int ldb, float *beta, float *C, int ldc);

void gemm(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, double *alpha, double *A, int lda, double *B, int ldb, double *beta, double *C, int ldc);

// void gemv(cublasHandle_t handle, cublasOperation_t trans, int m, int n, float *alpha, float *A, int lda, float *x, int incx, float *beta, float *y, int incy);

// void gemv(cublasHandle_t handle, cublasOperation_t trans, int m, int n, double *alpha, double *A, int lda, double *x, int incx, double *beta, double *y, int incy);

void getrfBatched(cublasHandle_t handle, int n, float *Aarray[], int lda, int *PivotArray, int *infoArray, int batchSize);

void getrfBatched(cublasHandle_t handle, int n, double *Aarray[], int lda, int *PivotArray, int *infoArray, int batchSize);

void getriBatched(cublasHandle_t handle, int n, float *Aarray[], int lda, int *PivotArray, float *Carray[], int ldc, int *infoArray, int batchSize);

void getriBatched(cublasHandle_t handle, int n, double *Aarray[], int lda, int *PivotArray, double *Carray[], int ldc, int *infoArray, int batchSize);


#endif
