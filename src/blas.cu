#ifndef BLAS_CPP
#define BLAS_CPP

#include "blas.h"

void gemm(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, float *alpha, float *A, int lda, float *B, int ldb, float *beta, float *C, int ldc) {
	cublasSgemm(handle, transa, transb, m, n, k, (const float *)alpha, (const float *)A, lda, (const float *)B, ldb, (const float *)beta, C, ldc);
}

void gemm(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, double *alpha, double *A, int lda, double *B, int ldb, double *beta, double *C, int ldc) {
	cublasDgemm(handle, transa, transb, m, n, k, (const double *)alpha, (const double *)A, lda, (const double *)B, ldb, (const double *)beta, C, ldc);
}

void gemv(cublasHandle_t handle, cublasOperation_t trans, int m, int n, float *alpha, float *A, int lda, float *x, int incx, float *beta, float *y, int incy) {
	cublasSgemv(handle, trans, m, n, (const float *)alpha, (const float *)A, lda, (const float *)x, incx, (const float *)beta, y, incy);
}

void gemv(cublasHandle_t handle, cublasOperation_t trans, int m, int n, double *alpha, double *A, int lda, double *x, int incx, double *beta, double *y, int incy) {
	cublasDgemv(handle, trans, m, n, (const double *)alpha, (const double *)A, lda, (const double *)x, incx, (const double *)beta, y, incy);
}

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

//------------------------------------

template<class T>
struct SharedMemory {
	__device__ inline operator T *() {
		extern __shared__ int __smem[];
		return (T *)__smem;
	}

	__device__ inline operator const T *() const {
		extern __shared__ int __smem[];
		return (T *)__smem;
	}
};

template<>
struct SharedMemory<double> {
	__device__ inline operator double *() {
		extern __shared__ double __smem_d[];
		return (double *)__smem_d;
	}

	__device__ inline operator const double *() const {
		extern __shared__ double __smem_d[];
		return (double *)__smem_d;
	}
};

__global__
void XAyBatched_kernel(precision **XA, precision *y, precision *r, int *nVars, int M, int numModels) {
	int mod = threadIdx.x + blockIdx.x * blockDim.x;
	int ind = threadIdx.y + blockIdx.y * blockDim.y;
	if (mod < numModels) {
		int ni = nVars[mod];
		precision *smem = SharedMemory<precision>();
		smem[ind] = (ind < M)? y[mod * M + ind]: 0;
		__syncthreads();
		if (ind < ni) {
			precision val = 0;
			for (int i = 0; i < M; i++) {
				val += XA[mod][ind * M + i] * smem[i];
			}
			r[mod * M + ind] = val;
		}
	}
}

void XAyBatched(precision **XA, precision *y, precision *r, int *nVars, int M, int numModels) {
	dim3 blockDim(1, M);
	dim3 gridDim(numModels, 1);
	XAyBatched_kernel<<<gridDim, blockDim, M * sizeof(precision)>>>(XA, y, r, nVars, M, numModels);
}

__global__
void IrBatched_kernel(precision **I, precision *r, precision *betaOls, int *nVars, int M, int numModels) {
	int mod = threadIdx.x + blockIdx.x * blockDim.x;
	int ind = threadIdx.y + blockIdx.y * blockDim.y;
	if (mod < numModels) {
		int ni = nVars[mod];
		precision *smem = SharedMemory<precision>();
		smem[ind] = (ind < ni)? r[mod * M + ind]: 0;
		__syncthreads();
		if (ind < ni) {
			precision val = 0;
			for (int i = 0; i < ni; i++) {
				val += I[mod][ind * ni + i] * smem[i];
			}
			betaOls[mod * M + ind] = val;
		}
	}
}

void IrBatched(precision **I, precision *r, precision *betaOls, int *nVars, int M, int numModels, int maxVar) {
	dim3 blockDim(1, maxVar);
	dim3 gridDim(numModels, 1);
	IrBatched_kernel<<<gridDim, blockDim, maxVar * sizeof(precision)>>>(I, r, betaOls, nVars, M, numModels);
}

__global__
void XAbetaOlsBatched_kernel(precision **XA, precision *betaOls, precision *d, int *nVars, int M, int numModels) {
	int mod = threadIdx.x + blockIdx.x * blockDim.x;
	int ind = threadIdx.y + blockIdx.y * blockDim.y;
	if (mod < numModels) {
		int ni = nVars[mod];
		precision *smem = SharedMemory<precision>();
		if (ind < ni) smem[ind] = betaOls[mod * M + ind];
		__syncthreads();
		if (ind < M) {
			precision val = 0;
			for (int i = 0; i < ni; i++) {
				val += XA[mod][i * M + ind] * smem[i];
			}
			d[mod * M + ind] = val;
		}
	}
}

void XAbetaOlsBatched(precision **XA, precision *betaOls, precision *d, int *nVars, int M, int numModels, int maxVar) {
	dim3 blockDim(1, M);
	dim3 gridDim(numModels, 1);
	XAbetaOlsBatched_kernel<<<gridDim, blockDim, maxVar * sizeof(precision)>>>(XA, betaOls, d, nVars, M, numModels);
}

#endif
