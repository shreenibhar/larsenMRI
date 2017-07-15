#ifndef BLAS_CPP
#define BLAS_CPP

#include "blas.h"

void gemm(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
          int m, int n, int k,
          const float *alpha, const float *A, int lda,
          const float *B, int ldb, const float *beta,
          float *C, int ldc) {
    cublasSgemm(handle, transa, transb,
                m, n, k,
                alpha, A, lda,
                B, ldb, beta,
                C, ldc);
}

void gemm(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
          int m, int n, int k,
          const double *alpha, const double *A, int lda,
          const double *B, int ldb, const double *beta,
          double *C, int ldc) {
    cublasDgemm(handle, transa, transb,
                m, n, k,
                alpha, A, lda,
                B, ldb, beta,
                C, ldc);
}

void gemv(cublasHandle_t handle, cublasOperation_t trans,
          int m, int n,
          const float *alpha, const float *A, int lda,
          const float *x, int incx, const float *beta,
          float *y, int incy) {
    cublasSgemv(handle, trans,
                m, n,
                alpha, A, lda,
                x, incx, beta,
                y, incy);
}

void gemv(cublasHandle_t handle, cublasOperation_t trans,
          int m, int n,
          const double *alpha, const double *A, int lda,
          const double *x, int incx, const double *beta,
          double *y, int incy) {
    cublasDgemv(handle, trans,
                m, n,
                alpha, A, lda,
                x, incx, beta,
                y, incy);
}

void getrfBatched(cublasHandle_t handle, int n, float *Aarray[],
                  int lda, int *PivotArray, int *infoArray,
                  int batchSize) {
    cublasSgetrfBatched(handle, n, Aarray,
                        lda, PivotArray, infoArray,
                        batchSize);
}

void getrfBatched(cublasHandle_t handle, int n, double *Aarray[],
                  int lda, int *PivotArray, int *infoArray,
                  int batchSize) {
    cublasDgetrfBatched(handle, n, Aarray,
                        lda, PivotArray, infoArray,
                        batchSize);
}

void getriBatched(cublasHandle_t handle, int n, float *Aarray[],
                  int lda, int *PivotArray, float *Carray[],
                  int ldc, int *infoArray, int batchSize) {
    cublasSgetriBatched(handle, n, (const float **)Aarray,
                       lda, PivotArray, Carray,
                       ldc, infoArray, batchSize);
}

void getriBatched(cublasHandle_t handle, int n, double *Aarray[],
                  int lda, int *PivotArray, double *Carray[],
                  int ldc, int *infoArray, int batchSize) {
    cublasDgetriBatched(handle, n, (const double **)Aarray,
                       lda, PivotArray, Carray,
                       ldc, infoArray, batchSize);
}

void amax(cublasHandle_t handle, int n, const float *x,
          int incx, int *result) {
    cublasIsamax(handle, n, x,
                 incx, result);
}

void amax(cublasHandle_t handle, int n, const double *x,
          int incx, int *result) {
    cublasIdamax(handle, n, x,
                 incx, result);
}

void amin(cublasHandle_t handle, int n, const float *x,
          int incx, int *result) {
    cublasIsamin(handle, n, x,
                 incx, result);
}

void amin(cublasHandle_t handle, int n, const double *x,
          int incx, int *result) {
    cublasIdamin(handle, n, x,
                 incx, result);
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

__device__
void SatomicMax(float *address, float val) {
    unsigned int* address_as_uint = (unsigned int*)address;
    unsigned int old = *address_as_uint, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_uint, assumed, __float_as_int(fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return;
}

__device__
void DatomicMax(double *address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed, __double_as_longlong(fmax(val, __longlong_as_double(assumed))));
    } while (assumed != old);
    return;
}

__device__
void SatomicMin(float *address, float val) {
    unsigned int* address_as_uint = (unsigned int*)address;
    unsigned int old = *address_as_uint, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_uint, assumed, __float_as_int(fminf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return;
}

__device__
void DatomicMin(double *address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed, __double_as_longlong(fmin(val, __longlong_as_double(assumed))));
    } while (assumed != old);
    return;
}

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
#else
__device__
void atomicAdd(double *address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return;
}
#endif

__global__
void SamaxFabs_kernel(float *array, float *cmax, int elements) {
    float *smem = SharedMemory<float>();
    int tid = threadIdx.x;
    int gid = threadIdx.x + blockIdx.x * blockDim.x;

    smem[tid] = (gid < elements)? fabsf(array[gid]): 0;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && gid < elements)
            smem[tid] = fmaxf(smem[tid], smem[tid + s]);
        __syncthreads();
    }
    if (tid == 0) {
        SatomicMax(cmax, smem[0]);
    }
}

__global__
void DamaxFabs_kernel(double *array, double *cmax, int elements) {
    double *smem = SharedMemory<double>();
    int tid = threadIdx.x;
    int gid = threadIdx.x + blockIdx.x * blockDim.x;

    smem[tid] = (gid < elements)? fabs(array[gid]): 0;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && gid < elements)
            smem[tid] = fmax(smem[tid], smem[tid + s]);
        __syncthreads();
    }
    if (tid == 0) {
        DatomicMax(cmax, smem[0]);
    }
}

void amaxFabs(float *array, float *cmax, int elements, cudaStream_t stream, dim3 blockDim) {
    dim3 gridDim((elements + blockDim.x - 1) / blockDim.x);
    SamaxFabs_kernel<<<gridDim, blockDim, blockDim.x * sizeof(float), stream>>>(array, cmax, elements);
}

void amaxFabs(double *array, double *cmax, int elements, cudaStream_t stream, dim3 blockDim) {
    dim3 gridDim((elements + blockDim.x - 1) / blockDim.x);
    DamaxFabs_kernel<<<gridDim, blockDim, blockDim.x * sizeof(double), stream>>>(array, cmax, elements);
}

//------------------------------------

__global__
void SminCd_kernel(float *c, float *cd, float *cmax, float *r, int N) {
    float *smem = SharedMemory<float>();
    int tid = threadIdx.x;
    int gid = threadIdx.x + blockIdx.x * blockDim.x;

    float val = 50000;
    if (gid < N) {
        val = c[gid];
        if (val != 0) {
            float a = (val - *cmax) / (cd[gid] - *cmax);
            float b = (val + *cmax) / (cd[gid] + *cmax);
            a = (a <= 0)? 50000: a;
            b = (b <= 0)? 50000: b;
            val = min(a, b);
        }
        else {
            val = 50000;
        }
    }
    smem[tid] = val;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && gid < N)
            smem[tid] = fminf(smem[tid], smem[tid + s]);
        __syncthreads();
    }
    if (tid == 0) {
        SatomicMin(r, smem[0]);
    }
}

__global__
void DminCd_kernel(double *c, double *cd, double *cmax, double *r, int N) {
    double *smem = SharedMemory<double>();
    int tid = threadIdx.x;
    int gid = threadIdx.x + blockIdx.x * blockDim.x;

    double val = 50000;
    if (gid < N) {
        val = c[gid];
        if (val != 0) {
            double a = (val - *cmax) / (cd[gid] - *cmax);
            double b = (val + *cmax) / (cd[gid] + *cmax);
            a = (a <= 0)? 50000: a;
            b = (b <= 0)? 50000: b;
            val = min(a, b);
        }
        else {
            val = 50000;
        }
    }
    smem[tid] = val;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && gid < N)
            smem[tid] = fmin(smem[tid], smem[tid + s]);
        __syncthreads();
    }
    if (tid == 0) {
        DatomicMin(r, smem[0]);
    }
}

void minCd(float *c, float *cd, float *cmax, float *r, int N, cudaStream_t stream, dim3 blockDim) {
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x);
    SminCd_kernel<<<gridDim, blockDim, blockDim.x * sizeof(float), stream>>>(c, cd, cmax, r, N);
}

void minCd(double *c, double *cd, double *cmax, double *r, int N, cudaStream_t stream, dim3 blockDim) {
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x);
    DminCd_kernel<<<gridDim, blockDim, blockDim.x * sizeof(double), stream>>>(c, cd, cmax, r, N);
}

__global__
void Snorm2_kernel(float *y, float *mu, float *a2, int M) {
    float *smem = SharedMemory<float>();
    int tid = threadIdx.x;
    int gid = threadIdx.x + blockIdx.x * blockDim.x;

    smem[tid] = (gid < M)? (y[gid] - mu[gid]) * (y[gid] - mu[gid]): 0;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && gid < M)
            smem[tid] += smem[tid + s];
        __syncthreads();
    }
    if (tid == 0) {
        atomicAdd(a2, smem[0]);
    }
}

__global__
void Dnorm2_kernel(double *y, double *mu, double *a2, int M) {
    double *smem = SharedMemory<double>();
    int tid = threadIdx.x;
    int gid = threadIdx.x + blockIdx.x * blockDim.x;

    smem[tid] = (gid < M)? (y[gid] - mu[gid]) * (y[gid] - mu[gid]): 0;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && gid < M)
            smem[tid] += smem[tid + s];
        __syncthreads();
    }
    if (tid == 0) {
        atomicAdd(a2, smem[0]);
    }
}

void norm2(float *y, float *mu, float *a2, int M, cudaStream_t stream, dim3 blockDim) {
    dim3 gridDim((M + blockDim.x - 1) / blockDim.x);
    Snorm2_kernel<<<gridDim, blockDim, blockDim.x * sizeof(float), stream>>>(y, mu, a2, M);
}

void norm2(double *y, double *mu, double *a2, int M, cudaStream_t stream, dim3 blockDim) {
    dim3 gridDim((M + blockDim.x - 1) / blockDim.x);
    Dnorm2_kernel<<<gridDim, blockDim, blockDim.x * sizeof(double), stream>>>(y, mu, a2, M);
}

__global__
void Snorm1_kernel(float *beta, float *a1, int N) {
    float *smem = SharedMemory<float>();
    int tid = threadIdx.x;
    int gid = threadIdx.x + blockIdx.x * blockDim.x;

    smem[tid] = (gid < N)? fabsf(beta[gid]): 0;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && gid < N)
            smem[tid] += smem[tid + s];
        __syncthreads();
    }
    if (tid == 0) {
        atomicAdd(a1, smem[0]);
    }
}

__global__
void Dnorm1_kernel(double *beta, double *a1, int N) {
    double *smem = SharedMemory<double>();
    int tid = threadIdx.x;
    int gid = threadIdx.x + blockIdx.x * blockDim.x;

    smem[tid] = (gid < N)? fabs(beta[gid]): 0;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && gid < N)
            smem[tid] += smem[tid + s];
        __syncthreads();
    }
    if (tid == 0) {
        atomicAdd(a1, smem[0]);
    }
}

void norm1(float *beta, float *a1, int N, cudaStream_t stream, dim3 blockDim) {
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x);
    Snorm1_kernel<<<gridDim, blockDim, blockDim.x * sizeof(float), stream>>>(beta, a1, N);
}

void norm1(double *beta, double *a1, int N, cudaStream_t stream, dim3 blockDim) {
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x);
    Dnorm1_kernel<<<gridDim, blockDim, blockDim.x * sizeof(double), stream>>>(beta, a1, N);
}

#endif