#ifndef FILE_PROC_KERNEL
#define FILE_PROC_KERNEL

#include <bits/stdc++.h>
#include <cuda_runtime.h>

#include "utilities.h"

using namespace std;

// Remove last row for X and first row for Y from d_number.
template<class T>
__global__ void d_trim(dmatrix<T> number, dmatrix<T> X, dmatrix<T> Y) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (y < number.N) {
        if (x < number.M - 1)
            X.set(x, y, number.get(x, y));
        if(x > 0 && x < number.M)
            Y.set(x - 1, y, number.get(x, y));
    }
}

// Normalize data.
template<class T>
__global__ void d_zscore(dmatrix<T> X, dmatrix<T> Y) {
    int ind = threadIdx.x + blockIdx.x * blockDim.x;
    if (ind < X.N) {
        T totX = 0, totY = 0;
        for (int i = 0; i < X.M; i++) {
            totX += X.get(i, ind);
            totY += Y.get(i, ind);
        }
        totX /= X.M;
        totY /= X.M;
        
        T sumX = 0, sumY = 0;
        for (int i = 0; i < X.M; i++) {
            T valX = X.get(i, ind) - totX;
            T valY = Y.get(i, ind) - totY;
            X.set(i, ind, valX);
            Y.set(i, ind, valY);
            sumX += valX * valX;
            sumY += valY * valY;
        }
        sumX = sqrt(sumX / (X.M - 1));
        sumY = sqrt(sumY / (X.M - 1));
        
        for (int i = 0; i < X.M; i++) {
            T valX = X.get(i, ind) / sumX;
            T valY = Y.get(i, ind) / sumY;
            X.set(i, ind, valX);
            Y.set(i, ind, valY);
        }
    }
}

#endif