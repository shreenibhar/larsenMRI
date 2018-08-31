#ifndef KERNELS_H
#define KERNELS_H

#include <bits/stdc++.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "utilities.h"

template<typename T>
void set_model(T *Y, T *y, T *mu, T *beta, T *a1, T *a2, T *lambda, int *nVars, int *lasso, int *step, int *done, int *act, int M, int N, int hact, cudaStream_t &stream);

template<typename T>
void check(int *nVars, int *step, T *a1, T *a2, T * lambda, int maxVariables, int maxSteps, T l1, T l2, T g, int *done, int numModels);

void drop(int *lVars, int *dropidx, int *nVars, int *lasso, int M, int numModels);

template<typename T>
void mat_sub(T *a, T *b, T *c, int size);

template<typename T>
void exclude(T *absC, int *lVars, int *nVars, int *act, int M, int N, int numModels, T def);

void lasso_add(int *lasso, int *lVars, int *nVars, int *cidx, int M, int N, int numModels);

template<typename T>
void gather(T *XA, T *XA1, T *X, int *lVars, int ni, int lassoCond, int drop, int M, int N, int mod, cudaStream_t &stream);

template<typename T>
void gammat(T *gamma_tilde, T *beta, T *betaOls, int *dropidx, int *lVars, int *nVars, int *lasso, int M, int N, int numModels);

template<typename T>
void set_gamma(T *gamma, T *gamma_tilde, T *r, int *lasso, int *nVars, int maxVariables, int M, int numModels);

template<typename T>
void update(T *beta, T *beta_prev, T *mu, T *d, T *betaOls, T *gamma, T **dXA, T *y, T *a1, T *a2, T *lambda, int *lVars, int *nVars, int *step, int M, int N, int numModels, T max_l1);

template<typename T>
void copyUp(double *varUp, T *var, int size, cudaStream_t &stream);

template<typename T>
void computeSign(double *sb, T *beta, T *beta_prev, int *lVars, int ni, cudaStream_t &stream);

template<typename T>
void correct(double *beta, double *betaols, double *sb, double *y, double *yh, double *z, T *a1, T *a2, T *lambda, double min_l2, double g, int ni, int M, cudaStream_t &stream);

#endif
