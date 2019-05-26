#ifndef KERNELS_H
#define KERNELS_H

#include <bits/stdc++.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "utilities.h"

void set_model(precision *Y, precision *y, precision *mu, precision *beta, precision *a1, precision *a2, precision *lambda, precision *randnrm, int *nVars, int *eVars, int *lasso, int *step, int *done, int *act, int M, int N, int hact, cudaStream_t &stream);

void check(int *nVars, int *step, precision *a1, precision *a2, precision * lambda, int maxVariables, int maxSteps, precision l1, precision l2, precision g, int *done, int numModels);

void drop(int *lVars, int *dropidx, int *nVars, int *lasso, int M, int numModels);

void mat_sub(precision *a, precision *b, precision *c, int size);

void exclude(precision *absC, int *lVars, int *nVars, int *eVars, int *act, int M, int N, int numModels, precision def);

void lasso_add(precision *c, int *lasso, int *lVars, int *nVars, int *cidx, int M, int N, int numModels);

void gather(precision *XA, precision *XA1, precision *X, int *lVars, int ni, int lassoCond, int drop, int M, int N, int mod, cudaStream_t &stream);

void checkNan(int *nVars, int *eVars, int *lVars, int *info, int *infomapper, precision *r, precision *d, precision *randnrm, int M, int numModels);

void gammat(precision *gamma_tilde, precision *beta, precision *betaOls, int *dropidx, int *lVars, int *nVars, int *lasso, int M, int N, int numModels);

void set_gamma(precision *gamma, precision *gamma_tilde, int *lasso, int *nVars, int maxVariables, int M, int numModels);

void update(precision *beta, precision *beta_prev, precision *mu, precision *d, precision *betaOls, precision *gamma, precision **dXA, precision *y, precision *a1, precision *a2, precision *lambda, int *lVars, int *nVars, int *step, int *info, int M, int N, int numModels, precision max_l1);

void gatherAll(corr_precision *XA, corr_precision *y, corr_precision *X, int *lVars, int ni, int M, int N, int act, cudaStream_t &stream);

void computeSign(corr_precision *sb, precision *beta, precision *beta_prev, int *lVars, int *dropidx, int *lasso, int ni, cudaStream_t &stream);

void correct(corr_precision *beta, corr_precision *betaols, corr_precision *sb, corr_precision *y, corr_precision *yh, corr_precision *z, precision *a1, precision *a2, precision *lambda, corr_precision min_l2, corr_precision g, int ni, int M, cudaStream_t &stream);

#endif
