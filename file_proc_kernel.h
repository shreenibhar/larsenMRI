#ifndef FILE_PROC_KERNEL
#define FILE_PROC_KERNEL

#include <bits/stdc++.h>
#include <cuda_runtime.h>

using namespace std;

// Remove last row for new1 and first row for new from d_number.
template<class T>
__global__ void d_trim(T *d_number, T *d_X, T *d_Y, int M, int Z) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (y < Z) {
        if (x < M - 1)
            d_X[x * Z + y] = d_number[x * Z + y];
        if(x > 0 && x < M)
            d_Y[(x - 1) * Z + y] = d_number[x * Z + y];
    }
}

// Normalize data.
template<class T>
__global__ void d_proc(T *d_X, T *d_Y, int M, int Z) {
    int ind = threadIdx.x + blockIdx.x * blockDim.x;
    if (ind < Z) {
        float tot1 = 0, tot = 0;
        for (int i = 0; i < M - 1; i++) {
            tot1 += d_X[i * Z + ind];
            tot += d_Y[i * Z + ind];
        }
        tot1 /= M - 1;
        tot /= M - 1;
        
        float sum1 = 0, sum = 0;
        for (int i = 0; i < M - 1; i++) {
            d_X[i * Z + ind] -= tot1;
            d_Y[i * Z + ind] -= tot;
            sum1 += pow(d_X[i * Z + ind], 2);
            sum += pow(d_Y[i * Z + ind], 2);
        }
        sum1 = sqrt(sum1 / (M - 2));
        sum = sqrt(sum / (M - 2));
        
        tot1 = tot = 0;
        for (int i = 0; i < M - 1; i++) {
            d_X[i * Z + ind] /= sum1;
            d_Y[i * Z + ind] /= sum;
        }
    }
}

#endif