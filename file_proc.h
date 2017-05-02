#ifndef FILE_PROC
#define FILE_PROC

#include <bits/stdc++.h>
#include <cuda_runtime.h>

#include "file_proc_kernel.h"
#include "utilities.h"

using namespace std;

// Remove used files.
void remove_used_files() {
    ofstream f, o, r;
    f.open("QBeta.csv", ios::out);
    o.open("QStep.csv", ios::out);
    r.open("QRess.csv", ios::out);
        
    f.close();
    o.close();
    r.close();
}

// Returns device pointers.
// Function to read flattened 2d FMRI image.
template<class T>
dmatrix<T> read_flat_mri(char *argv) {
    fstream fp(argv);
    int M, Z;
    string  str;
    fp >> M >> Z;
    
    T *h_number = new T[M * Z], *d_number;
    for (int i = 0; i < M; i++)
        for (int j = 0; j < Z; j++)
            fp >> h_number[i * Z + j];
    fp.close();
    
    cudaMallocManaged(&d_number, M * Z * sizeof(T));
    cudaMemcpy(d_number, h_number, M * Z * sizeof(T), cudaMemcpyHostToDevice);
    delete[] h_number;
    
    dmatrix<T> number;
    number.M = M;
    number.N = Z;
    number.d_mat = d_number;
    return number;
}

// Remove first and last row for new1 and new respectively and normalize.
template<class T>
void proc_flat_mri(dmatrix<T> number, dmatrix<T> X, dmatrix<T> Y) {
    dim3 blockDim(32, 32);
    dim3 gridDim((number.M + blockDim.x - 1) / blockDim.x,
        (number.N + blockDim.y - 1) / blockDim.y);
    d_trim<T><<<gridDim, blockDim>>>(number, X, Y);
    
    cudaFree(number.d_mat);

    dim3 bz(1024);
    dim3 gz((X.N + bz.x - 1) / bz.x);
    d_zscore<T><<<gz, bz>>>(X, Y);

    cudaDeviceSynchronize();
    for (int j = 0; j < Y.N; j++) {
        T tot = 0;
        for (int i = 0; i < Y.M; i++) {
            T val = Y.d_mat[i * Y.N + j];
            tot += val * val;
        }
        tot = sqrt(tot);
        for (int i = 0; i < Y.M; i++) {
            T val = Y.d_mat[i * Y.N + j] / tot;
            Y.d_mat[i * Y.N + j] = val;
        }
    }
    for (int j = 0; j < X.N; j++) {
        T tot = 0;
        for (int i = 0; i < X.M; i++) {
            T val = X.d_mat[i * Y.N + j];
            tot += val * val;
        }
        tot = sqrt(tot);
        for (int i = 0; i < X.M; i++) {
            T val = X.d_mat[i * X.N + j] / tot;
            X.d_mat[i * X.N + j] = val;
        }
    }
}

#endif