#ifndef BLAS_H
#define BLAS_H

#include <bits/stdc++.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

void gemm(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
          int m, int n, int k,
          const float *alpha, const float *A, int lda,
          const float *B, int ldb, const float *beta,
          float *C, int ldc);

void gemm(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
          int m, int n, int k,
          const double *alpha, const double *A, int lda,
          const double *B, int ldb, const double *beta,
          double *C, int ldc);

void gemv(cublasHandle_t handle, cublasOperation_t trans,
          int m, int n,
          const float *alpha, const float *A, int lda,
          const float *x, int incx, const float *beta,
          float *y, int incy);

void gemv(cublasHandle_t handle, cublasOperation_t trans,
          int m, int n,
          const double *alpha, const double *A, int lda,
          const double *x, int incx, const double *beta,
          double *y, int incy);

void getrfBatched(cublasHandle_t handle, int n, float *Aarray[],
				  int lda, int *PivotArray, int *infoArray,
				  int batchSize);

void getrfBatched(cublasHandle_t handle, int n, double *Aarray[],
				  int lda, int *PivotArray, int *infoArray,
				  int batchSize);

void getriBatched(cublasHandle_t handle, int n, float *Aarray[],
				  int lda, int *PivotArray, float *Carray[],
				  int ldc, int *infoArray, int batchSize);

void getriBatched(cublasHandle_t handle, int n, double *Aarray[],
				  int lda, int *PivotArray, double *Carray[],
				  int ldc, int *infoArray, int batchSize);

void amax(cublasHandle_t handle, int n, const float *x,
		  int incx, int *result);

void amax(cublasHandle_t handle, int n, const double *x,
		  int incx, int *result);

void amin(cublasHandle_t handle, int n, const float *x,
		  int incx, int *result);

void amin(cublasHandle_t handle, int n, const double *x,
		  int incx, int *result);

void amaxFabs(float *array, float *cmax, float *buf, int elements, cudaStream_t stream, dim3 blockDim);

void amaxFabs(double *array, double *cmax, double *buf, int elements, cudaStream_t stream, dim3 blockDim);

//------------------------------------

void minCd(float *c, float *cd, float *cmax, float *r, float *buf, int N, cudaStream_t stream, dim3 blockDim);

void minCd(double *c, double *cd, double *cmax, double *r, double *buf, int N, cudaStream_t stream, dim3 blockDim);

void norm2(float *y, float *mu, float *a2, float *buf, int M, cudaStream_t stream, dim3 blockDim);

void norm2(double *y, double *mu, double *a2, double *buf, int M, cudaStream_t stream, dim3 blockDim);

void norm1(float *beta, float *a1, float *buf, int N, cudaStream_t stream, dim3 blockDim);

void norm1(double *beta, double *a1, double *buf, int N, cudaStream_t stream, dim3 blockDim);

#endif