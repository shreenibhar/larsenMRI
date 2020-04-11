#ifndef BLAS_CPP
#define BLAS_CPP

#include "blas.h"


void gemm(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, float *alpha, float *A, int lda, float *B, int ldb, float *beta, float *C, int ldc) {
  cublasSgemm(handle, transa, transb, m, n, k, (const float *)alpha, (const float *)A, lda, (const float *)B, ldb, (const float *)beta, C, ldc);
}

void gemm(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, double *alpha, double *A, int lda, double *B, int ldb, double *beta, double *C, int ldc) {
  cublasDgemm(handle, transa, transb, m, n, k, (const double *)alpha, (const double *)A, lda, (const double *)B, ldb, (const double *)beta, C, ldc);
}

// void gemv(cublasHandle_t handle, cublasOperation_t trans, int m, int n, float *alpha, float *A, int lda, float *x, int incx, float *beta, float *y, int incy) {
// 	cublasSgemv(handle, trans, m, n, (const float *)alpha, (const float *)A, lda, (const float *)x, incx, (const float *)beta, y, incy);
// }

// void gemv(cublasHandle_t handle, cublasOperation_t trans, int m, int n, double *alpha, double *A, int lda, double *x, int incx, double *beta, double *y, int incy) {
// 	cublasDgemv(handle, trans, m, n, (const double *)alpha, (const double *)A, lda, (const double *)x, incx, (const double *)beta, y, incy);
// }

void getrfBatched(cublasHandle_t handle, int n, float *Aarray[], int lda, int *PivotArray, int *infoArray, int batchSize) {
  cublasSgetrfBatched(handle, n, Aarray, lda, PivotArray, infoArray, batchSize);
}

void getrfBatched(cublasHandle_t handle, int n, double *Aarray[], int lda, int *PivotArray, int *infoArray, int batchSize) {
  cublasDgetrfBatched(handle, n, Aarray, lda, PivotArray, infoArray, batchSize);
}

void getriBatched(cublasHandle_t handle, int n, float *Aarray[], int lda, int *PivotArray, float *Carray[], int ldc, int *infoArray, int batchSize) {
  cublasSgetriBatched(handle, n, (const float **)Aarray, lda, PivotArray, Carray, ldc, infoArray, batchSize);
}

void getriBatched(cublasHandle_t handle, int n, double *Aarray[], int lda, int *PivotArray, double *Carray[], int ldc, int *infoArray, int batchSize) {
  cublasDgetriBatched(handle, n, (const double **)Aarray, lda, PivotArray, Carray, ldc, infoArray, batchSize);
}


#endif
