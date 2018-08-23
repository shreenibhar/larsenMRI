#ifndef BLAS_CPP
#define BLAS_CPP

#include "blas.h"

int next_pow2(int num) {
	num--;
	num |= num >> 1;
	num |= num >> 2;
	num |= num >> 4;
	num |= num >> 8;
	num |= num >> 16;
	num++;
	return num;
}

void gemm(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, float *alpha, float *A, int lda, float *B, int ldb, float *beta, float *C, int ldc) {
	cublasSgemm(handle, transa, transb, m, n, k, (const float *)alpha, (const float *)A, lda, (const float *)B, ldb, (const float *)beta, C, ldc);
}

void gemm(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, double *alpha, double *A, int lda, double *B, int ldb, double *beta, double *C, int ldc) {
	cublasDgemm(handle, transa, transb, m, n, k, (const double *)alpha, (const double *)A, lda, (const double *)B, ldb, (const double *)beta, C, ldc);
}

void gemmBatched(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, float *alpha, float *Aarray[], int lda, float *Barray[], int ldb, float *beta, float *Carray[], int ldc, int batchCount) {
	cublasSgemmBatched(handle, transa, transb, m, n, k, (const float *)alpha, (const float **)Aarray, lda, (const float **)Barray, ldb, (const float *)beta, Carray, ldc, batchCount);
}

void gemmBatched(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, double *alpha, double *Aarray[], int lda, double *Barray[], int ldb, double *beta, double *Carray[], int ldc, int batchCount) {
	cublasDgemmBatched(handle, transa, transb, m, n, k, (const double *)alpha, (const double **)Aarray, lda, (const double **)Barray, ldb, (const double *)beta, Carray, ldc, batchCount);
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

void amax(cublasHandle_t handle, int n, const float *x, int incx, int *result) {
	cublasIsamax(handle, n, x, incx, result);
}

void amax(cublasHandle_t handle, int n, const double *x, int incx, int *result) {
	cublasIdamax(handle, n, x, incx, result);
}

void amin(cublasHandle_t handle, int n, const float *x, int incx, int *result) {
	cublasIsamin(handle, n, x, incx, result);
}

void amin(cublasHandle_t handle, int n, const double *x, int incx, int *result) {
	cublasIdamin(handle, n, x, incx, result);
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

template<typename T>
__global__
void XAyBatched_kernel(T **XA, T *y, T *r, int *nVars, int M, int numModels) {
	int mod = threadIdx.x + blockIdx.x * blockDim.x;
	int ind = threadIdx.y + blockIdx.y * blockDim.y;
	if (mod < numModels) {
		int ni = nVars[mod];
		T *smem = SharedMemory<T>();
		smem[ind] = (ind < M)? y[mod * M + ind]: 0;
		__syncthreads();
		if (ind < ni) {
			T val = 0;
			for (int i = 0; i < M; i++) {
				val += XA[mod][ind * M + i] * smem[i];
			}
			r[mod * M + ind] = val;
		}
	}
}

template<typename T>
void XAyBatched(T **XA, T *y, T *r, int *nVars, int M, int numModels) {
	dim3 blockDim(1, M);
	dim3 gridDim(numModels, 1);
	XAyBatched_kernel<T><<<gridDim, blockDim, M * sizeof(T)>>>(XA, y, r, nVars, M, numModels);
}

template void XAyBatched<float>(float **XA, float *y, float *r, int *nVars, int M, int numModels);
template void XAyBatched<double>(double **XA, double *y, double *r, int *nVars, int M, int numModels);

template<typename T>
__global__
void IrBatched_kernel(T **I, T *r, T *betaOls, int *nVars, int M, int numModels) {
	int mod = threadIdx.x + blockIdx.x * blockDim.x;
	int ind = threadIdx.y + blockIdx.y * blockDim.y;
	if (mod < numModels) {
		int ni = nVars[mod];
		T *smem = SharedMemory<T>();
		smem[ind] = (ind < ni)? r[mod * M + ind]: 0;
		__syncthreads();
		if (ind < ni) {
			T val = 0;
			for (int i = 0; i < ni; i++) {
				val += I[mod][ind * ni + i] * smem[i];
			}
			betaOls[mod * M + ind] = val;
		}
	}
}

template<typename T>
void IrBatched(T **I, T *r, T *betaOls, int *nVars, int M, int numModels, int maxVar) {
	dim3 blockDim(1, maxVar);
	dim3 gridDim(numModels, 1);
	IrBatched_kernel<T><<<gridDim, blockDim, maxVar * sizeof(T)>>>(I, r, betaOls, nVars, M, numModels);
}

template void IrBatched<float>(float **I, float *r, float *betaOls, int *nVars, int M, int numModels, int maxVar);
template void IrBatched<double>(double **I, double *r, double *betaOls, int *nVars, int M, int numModels, int maxVar);

template<typename T>
__global__
void XAbetaOlsBatched_kernel(T **XA, T *betaOls, T *d, int *nVars, int M, int numModels) {
	int mod = threadIdx.x + blockIdx.x * blockDim.x;
	int ind = threadIdx.y + blockIdx.y * blockDim.y;
	if (mod < numModels) {
		int ni = nVars[mod];
		T *smem = SharedMemory<T>();
		if (ind < ni) smem[ind] = betaOls[mod * M + ind];
		__syncthreads();
		if (ind < M) {
			T val = 0;
			for (int i = 0; i < ni; i++) {
				val += XA[mod][i * M + ind] * smem[i];
			}
			d[mod * M + ind] = val;
		}
	}
}

template<typename T>
void XAbetaOlsBatched(T **XA, T *betaOls, T *d, int *nVars, int M, int numModels, int maxVar) {
	dim3 blockDim(1, M);
	dim3 gridDim(numModels, 1);
	XAbetaOlsBatched_kernel<T><<<gridDim, blockDim, maxVar * sizeof(T)>>>(XA, betaOls, d, nVars, M, numModels);
}

template void XAbetaOlsBatched<float>(float **XA, float *betaOls, float *d, int *nVars, int M, int numModels, int maxVar);
template void XAbetaOlsBatched<double>(double **XA, double *betaOls, double *d, int *nVars, int M, int numModels, int maxVar);

template<typename T>
__global__
void fabsMaxReduce_kernel(T *mat, T *buf, int *ind, int *intBuf, int rowSize, int colSize) {
	T *smem = SharedMemory<T>();
	int row = threadIdx.x + blockIdx.x * blockDim.x;
	int tid = threadIdx.y;
	int col = threadIdx.y + blockIdx.y * blockDim.y;

	smem[tid] = (row < rowSize && col < colSize)? fabs(mat[row * colSize + col]): 0;
	if (ind == NULL) smem[tid + blockDim.y] = (row < rowSize && col < colSize)? col: 0;
	else smem[tid + blockDim.y] = (row < rowSize && col < colSize)? ind[row * colSize + col]: 0;
	__syncthreads();

	for (unsigned int s = blockDim.y / 2; s > 0; s >>= 1) {
		if (tid < s && smem[tid + s] > smem[tid]) {
			smem[tid] = smem[tid + s];
			smem[tid + blockDim.y] = smem[tid + s + blockDim.y];
		}
		__syncthreads();
	}
	if (tid == 0) {
		buf[row * gridDim.y + blockIdx.y] = smem[0];
		intBuf[row * gridDim.y + blockIdx.y] = smem[blockDim.y];
	}
}

template<typename T>
void fabsMaxReduce(T *mat, T *res, T *buf, int *ind, int *intBuf, int rowSize, int colSize) {
	dim3 blockDim(1, 512);
	dim3 gridDim(rowSize, (colSize + blockDim.y - 1) / blockDim.y);
	fabsMaxReduce_kernel<T><<<gridDim, blockDim, 2 * blockDim.y * sizeof(T)>>>(mat, buf, NULL, intBuf, rowSize, colSize);
	colSize = gridDim.y;
	blockDim = *new dim3(1, next_pow2(colSize));
	gridDim = *new dim3(rowSize, 1);
	fabsMaxReduce_kernel<T><<<gridDim, blockDim, 2 * blockDim.y * sizeof(T)>>>(buf, res, intBuf, ind, rowSize, colSize);
}

template void fabsMaxReduce<float>(float *mat, float *res, float *buf, int *ind, int *intBuf, int rowSize, int colSize);
template void fabsMaxReduce<double>(double *mat, double *res, double *buf, int *ind, int *intBuf, int rowSize, int colSize);

template<typename T>
__global__
void cdMinReduce_kernel(T *c, T *cd, T *cmax, T *buf, int rowSize, int colSize, int opt) {
	T *smem = SharedMemory<T>();
	int row = threadIdx.x + blockIdx.x * blockDim.x;
	int tid = threadIdx.y;
	int col = threadIdx.y + blockIdx.y * blockDim.y;

	smem[tid] = (row < rowSize && col < colSize)? c[row * colSize + col]: 50000;
	if (row < rowSize && col < colSize && opt) {
		if (smem[tid] == cmax[row]) smem[tid] = 0;
		if (smem[tid] != 0) {
			T a = (smem[tid] - cmax[row]) / (cd[row * colSize + col] - cmax[row]);
			T b = (smem[tid] + cmax[row]) / (cd[row * colSize + col] + cmax[row]);
			a = (a <= 0)? 50000: a;
			b = (b <= 0)? 50000: b;
			smem[tid] = min(a, b);
		}
		else {
			smem[tid] = 50000;
		}
	}
	__syncthreads();

	for (unsigned int s = blockDim.y / 2; s > 0; s >>= 1) {
		if (tid < s && smem[tid + s] < smem[tid]) smem[tid] = smem[tid + s];
		__syncthreads();
	}
	if (tid == 0) {
		buf[row * gridDim.y + blockIdx.y] = smem[0];
	}
}

template<typename T>
void cdMinReduce(T *c, T *cd, T *cmax, T *res, T *buf, int rowSize, int colSize) {
	dim3 blockDim(1, 512);
	dim3 gridDim(rowSize, (colSize + blockDim.y - 1) / blockDim.y);
	cdMinReduce_kernel<T><<<gridDim, blockDim, blockDim.y * sizeof(T)>>>(c, cd, cmax, buf, rowSize, colSize, 1);
	colSize = gridDim.y;
	blockDim = *new dim3(1, next_pow2(colSize));
	gridDim = *new dim3(rowSize, 1);
	cdMinReduce_kernel<T><<<gridDim, blockDim, blockDim.y * sizeof(T)>>>(buf, NULL, NULL, res, rowSize, colSize, 0);
}

template void cdMinReduce<float>(float *c, float *cd, float *cmax, float *res, float *buf, int rowSize, int colSize);
template void cdMinReduce<double>(double *c, double *cd, double *cmax, double *res, double *buf, int rowSize, int colSize);

#endif
