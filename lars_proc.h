#ifndef LARS_PROC
#define LARS_PROC

#include <bits/stdc++.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "lars_proc_kernel.h"
#include "utilities.h"

// X input(M - 1, Z).
// Y output(M - 1, Z).
// act is the list of current evaluating models and num_models is the buffer size.
// g is the lars pruning variable.
template<class T>
void lars(dmatrix<T> X, dmatrix<T> Xt, dmatrix<T> Y, dmatrix<T> Yt, dmatrix<T> y, dmatrix<T> mu, dmatrix<T> c, dmatrix<T> _, dmatrix<T> __, dmatrix<T> G, dmatrix<T> I, dmatrix<T> beta, dmatrix<T> betaOls, dmatrix<T> d, dmatrix<T> gamma, dmatrix<T> cmax, dmatrix<T> upper1, dmatrix<T> normb,
    dmatrix<int> lVars, dmatrix<int> nVars, dmatrix<bool> maskVars, dmatrix<int> ind, dmatrix<int> step, dmatrix<int> lasso, dmatrix<int> done, dmatrix<int> act, dmatrix<int> ctrl,
    T g)
{
    int top = c.M;
    int num_models = c.M;
    int max_vars = y.N;

    h_transpose<T>(Xt, X);
    h_transpose<T>(Yt, Y);

    h_init_full<T>(mu, 0);
    h_init_full<T>(beta, 0);
    h_init_full<bool>(maskVars, false);

    h_init_full<int>(nVars, 0);
    h_init_full<int>(lasso, 0);
    h_init_full<int>(done, 0);
    h_init_full<int>(step, 0);

    for (int i = 0; i < num_models; i++) {
        cudaDeviceSynchronize();
        act.d_mat[i] = i;
        h_init_y<T>(y, Yt, i, i);
    }

    // Cublas setup.
    cudaDeviceSynchronize();
    int *d_pivot, *d_info, batch = 1;
    cudaMallocManaged(&d_pivot, 3 * sizeof(int));
    cudaMallocManaged(&d_info, sizeof(int));
    d_info[0] = 0;
    T **dd_G, **dd_I;
    cudaMallocManaged(&dd_G, sizeof(T *));
    cudaMallocManaged(&dd_I, sizeof(T *));
    dd_G[0] = G.d_mat;
    dd_I[0] = I.d_mat;
    cublasHandle_t handle;
    cublasCreate(&handle);
        
    gpuTimer gtimer;

    while (1) {
        cudaDeviceSynchronize();
        ctrl.d_mat[0] = ctrl.d_mat[1] = 0;
        h_check(ctrl, step, nVars, done, max_vars);
        cudaDeviceSynchronize();
        if (ctrl.d_mat[0] == 0)
            break;

        h_corr<T>(Xt, y, mu, c, act, done);        
        h_max_corr<T>(c, cmax, ind, maskVars, done);
        
        h_lasso_add(ind, lVars, nVars, maskVars, lasso, done, max_vars);
        
        h_xincty<T>(Xt, y, __, lVars, nVars, act, done);

        for (int j = 0; j < y.M; j++) {
            cudaDeviceSynchronize();
            if (done.d_mat[j])
                continue;
            int ni = nVars.d_mat[j];
            h_set_gram<T>(Xt, G, lVars, act, ni, j);
            
            cublasSgetrfBatched(handle, ni, dd_G, ni, d_pivot, d_info, batch);
            cublasSgetriBatched(handle, ni, (const T **)dd_G, ni, d_pivot, dd_I, ni, d_info, batch);
            
            h_betaols<T>(I, __, betaOls, ni, j);
        }

        h_d<T>(X, betaOls, mu, d, lVars, nVars, act, done);
        
        h_gammat<T>(__, beta, betaOls, lVars, nVars, done);
        h_min_gammat<T>(gamma, __, ind, nVars, done);
        
        h_xtd<T>(Xt, d, _, c, cmax, act, done);
        h_min_tmp<T>(_, __, maskVars, done);

        h_lasso_dev<T>(__, gamma, nVars, lasso, done);
        
        h_update<T>(gamma, mu, beta, betaOls, d, lVars, nVars, done);
        
        h_lasso_drop(ind, lVars, nVars, maskVars, lasso, done);
        
        h_final<T>(mu, Yt, beta, upper1, normb, nVars, step, act,
            done, ctrl, g);

        cudaDeviceSynchronize();
        if (ctrl.d_mat[1] == 0)
            continue;
        // Remove completed models and write them to a file and replace them with new models.
        for (int j = 0; j < c.M && top < X.N; j++) {
            cudaDeviceSynchronize();
            if (done.d_mat[j]) {
                // Replace the completed model index in the buffer with model top which is the next model in the stack to be added.
                act.d_mat[j] = top++;
                nVars.d_mat[j] = lasso.d_mat[j] = step.d_mat[j] = done.d_mat[j] = 0;
                h_init_part<T>(beta, 0, j);
                h_init_part<T>(mu, 0, j);
                h_init_part<bool>(maskVars, false, j);
                h_init_y<T>(y, Yt, j, top);
                printf("\rModel stack at %d", top);
            }
        }
    }    
    cudaFree(d_pivot);
    cudaFree(d_info);
    cudaFree(dd_G);
    cudaFree(dd_I);
    cublasDestroy(handle);
}

#endif