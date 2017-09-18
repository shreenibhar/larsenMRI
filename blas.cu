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

void gemm(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const float *alpha, const float *A, int lda, const float *B, int ldb, const float *beta, float *C, int ldc) {
	cublasSgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

void gemm(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const double *alpha, const double *A, int lda, const double *B, int ldb, const double *beta, double *C, int ldc) {
	cublasDgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

void gemv(cublasHandle_t handle, cublasOperation_t trans, int m, int n, const float *alpha, const float *A, int lda, const float *x, int incx, const float *beta, float *y, int incy) {
	cublasSgemv(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
}

void gemv(cublasHandle_t handle, cublasOperation_t trans, int m, int n, const double *alpha, const double *A, int lda, const double *x, int incx, const double *beta, double *y, int incy) {
	cublasDgemv(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
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

	if (opt && row < rowSize && col < colSize && c[row * colSize + col] == cmax[row]) {
		smem[tid] = 0;
	}
	else smem[tid] = (row < rowSize && col < colSize)? c[row * colSize + col]: 50000;
	if (row < rowSize && col < colSize && opt) {
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
		if (tid < s) smem[tid] = min(smem[tid], smem[tid + s]);
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

template<typename T>
__global__
void fabsAddReduce_kernel(T *mat, T *buf, int rowSize, int colSize) {
	T *smem = SharedMemory<T>();
	int row = threadIdx.x + blockIdx.x * blockDim.x;
	int tid = threadIdx.y;
	int col = threadIdx.y + blockIdx.y * blockDim.y;

	smem[tid] = (row < rowSize && col < colSize)? fabs(mat[row * colSize + col]): 0;
	__syncthreads();

	for (unsigned int s = blockDim.y / 2; s > 0; s >>= 1) {
		if (tid < s) smem[tid] += smem[tid + s];
		__syncthreads();
	}
	if (tid == 0) {
		buf[row * gridDim.y + blockIdx.y] = smem[0];
	}
}

template<typename T>
void fabsAddReduce(T *mat, T *res, T *buf, int rowSize, int colSize) {
	dim3 blockDim(1, 512);
	dim3 gridDim(rowSize, (colSize + blockDim.y - 1) / blockDim.y);
	fabsAddReduce_kernel<T><<<gridDim, blockDim, blockDim.y * sizeof(T)>>>(mat, buf, rowSize, colSize);
	colSize = gridDim.y;
	blockDim = *new dim3(1, next_pow2(colSize));
	gridDim = *new dim3(rowSize, 1);
	fabsAddReduce_kernel<T><<<gridDim, blockDim, blockDim.y * sizeof(T)>>>(buf, res, rowSize, colSize);
}

template void fabsAddReduce<float>(float *mat, float *res, float *buf, int rowSize, int colSize);
template void fabsAddReduce<double>(double *mat, double *res, double *buf, int rowSize, int colSize);

template<typename T>
__global__
void sqrAddReduce_kernel(T *y, T *mu, T *buf, int rowSize, int colSize, int opt) {
	T *smem = SharedMemory<T>();
	int row = threadIdx.x + blockIdx.x * blockDim.x;
	int tid = threadIdx.y;
	int col = threadIdx.y + blockIdx.y * blockDim.y;

	smem[tid] = (row < rowSize && col < colSize)? y[row * colSize + col]: 0;
	if (row < rowSize && col < colSize && opt) {
		smem[tid] -= mu[row * colSize + col];
		smem[tid] *= smem[tid];
	}
	__syncthreads();

	for (unsigned int s = blockDim.y / 2; s > 0; s >>= 1) {
		if (tid < s) smem[tid] += smem[tid + s];
		__syncthreads();
	}
	if (tid == 0) {
		buf[row * gridDim.y + blockIdx.y] = smem[0];
	}
}

template<typename T>
void sqrAddReduce(T *y, T *mu, T *res, T *buf, int rowSize, int colSize) {
	dim3 blockDim(1, 512);
	dim3 gridDim(rowSize, (colSize + blockDim.y - 1) / blockDim.y);
	sqrAddReduce_kernel<T><<<gridDim, blockDim, blockDim.y * sizeof(T)>>>(y, mu, buf, rowSize, colSize, 1);
	colSize = gridDim.y;
	blockDim = *new dim3(1, next_pow2(colSize));
	gridDim = *new dim3(rowSize, 1);
	sqrAddReduce_kernel<T><<<gridDim, blockDim, blockDim.y * sizeof(T)>>>(buf, NULL, res, rowSize, colSize, 0);
}

template void sqrAddReduce<float>(float *y, float *mu, float *res, float *buf, int rowSize, int colSize);
template void sqrAddReduce<double>(double *y, double *mu, double *res, double *buf, int rowSize, int colSize);

template<typename T>
__global__
void minGamma_kernel(T *gamma_tilde, int *dropidx, int *nVars, int numModels, int M) {
	int mod = threadIdx.x + blockIdx.x * blockDim.x;
	if (mod < numModels) {
		int ni = nVars[mod];
		T miner = gamma_tilde[mod * M + 0];
		int drop = 1;
		if (ni > 1) {
			for (int i = 1; i < ni - 1; i++) {
				T val = gamma_tilde[mod * M  + i];
				if (val < miner) {
					miner = val;
					drop = i + 1;
				}
			}
		}
		dropidx[mod] = drop;
	}
}

template<typename T>
void minGamma(T *gamma_tilde, int *dropidx, int *nVars, int numModels, int M, dim3 blockDim) {
	dim3 gridDim((numModels + blockDim.x - 1) / blockDim.x);
	minGamma_kernel<T><<<gridDim, blockDim>>>(gamma_tilde, dropidx, nVars, numModels, M);
}

template void minGamma<float>(float *gamma_tilde, int *dropidx, int *nVars, int numModels, int M, dim3 blockDim);
template void minGamma<double>(double *gamma_tilde, int *dropidx, int *nVars, int numModels, int M, dim3 blockDim);

#endif
