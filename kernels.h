#ifndef KERNELS_H
#define KERNELS_H

#include <bits/stdc++.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "utilities.h"

template<typename T>
void set_model(T *Y, T *y, T *mu, T *beta, T *a1, T *a2, T *lambda, int *nVars, int *lasso, int *step, int *done, int *act, int M, int N, int mod, int hact, cudaStream_t &stream, dim3 blockDim);

template<typename T>
void check(int *nVars, int *step, T *a1, T *a2, T * lambda, int maxVariables, int maxSteps, T l1, T l2, T g, int *done, int numModels);

template<typename T>
void mat_sub(T *a, T *b, T *c, int size, dim3 blockDim);

template<typename T>
void exclude(T *absC, int *lVars, int *nVars, int *act, int M, int N, int numModels, T def, dim3 blockDim);

void lasso_add(int *lasso, int *lVars, int *nVars, int *cidx, int M, int N, int numModels, dim3 blockDim);

template<typename T>
void gather(T *XA, T *XA1, T *X, int *lVars, int ni, int lassoCond, int drop, int M, int N, int mod, cudaStream_t &stream);

template<typename T>
void gammat(T *gamma_tilde, T *beta, T *betaOls, int *dropidx, int *lVars, int *nVars, int *lasso, int M, int N, int numModels, dim3 blockDim);

template<typename T>
void set_gamma(T *gamma, T *gamma_tilde, T *r, int *lasso, int *nVars, int maxVariables, int M, int numModels, dim3 blockDim);

template<typename T>
void update(T *beta, T *beta_prev, T *sb, T *mu, T *d, T *betaOls, T *gamma, int *lVars, int *nVars, int M, int N, int numModels, dim3 blockDim);

void drop(int *lVars, int *dropidx, int *nVars, int *lasso, int M, int numModels);

template<typename T>
void final(T **dXA, T *y, T *mu, T *beta, T *a1, T *a2, T *lambda, int *lVars, int *nVars, int *step, int numModels, int M, int N, dim3 blockDim);

template<typename T>
void compress(T *beta, T *r, int *lVars, int ni, int mod, int M, int N, cudaStream_t &stream);

#endif
