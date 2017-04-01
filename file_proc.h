#ifndef FILE_PROC
#define FILE_PROC

#include <bits/stdc++.h>
#include <cuda_runtime.h>

#include "file_proc_kernel.h"

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
T ** read_flat_mri(char *argv) {
    fstream fp(argv);
    int M, Z, i, j;
    string  str;
    fp >> M >> Z;
    
    T *number = new T[M * Z], *d_number;
    for (i = 0; i < M; i++)
        for (j = 0; j < Z; j++)
            fp >> number[i * Z + j];
    fp.close();
    
    cudaMallocManaged(&d_number, M * Z * sizeof(T));
    cudaMemcpy(d_number, number, M * Z * sizeof(T), cudaMemcpyHostToDevice);
    delete[] number;
    
    T **ret = new T*[1];
    ret[0] = d_number;
    return ret;
}

// Remove first and last row for new1 and new respectively and normalize.
template<class T>
T ** proc_flat_mri(T *d_number, int M, int Z) {
    T *d_X, *d_Y;
    cudaMallocManaged(&d_X, (M - 1) * Z * sizeof(T));
    cudaMallocManaged(&d_Y, (M - 1) * Z * sizeof(T));
    
    dim3 bmz(31, 31);
    dim3 gmz((M + bmz.x - 1) / bmz.x, (Z + bmz.y - 1) / bmz.y);
    d_trim<T><<<gmz, bmz>>>(d_number, d_X, d_Y, M, Z);
    
    cudaFree(d_number);

    dim3 bz(1000);
    dim3 gz((Z + bz.x - 1) / bz.x);
    d_proc<T><<<gz, bz>>>(d_X, d_Y, M, Z);

    T **ret = new T*[2];
    ret[0] = d_X;
    ret[1] = d_Y;
    return ret;
}

#endif