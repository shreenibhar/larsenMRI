#ifndef LARS_PROC
#define LARS_PROC

#include <bits/stdc++.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "lars_proc_kernel.h"

#define INF 50000

using namespace std;

// X input(M - 1, Z).
// Y output(M - 1, Z).
// act is the list of current evaluating models and num_models is the buffer size.
// g is the lars pruning variable.
template<class T>
void lars(T *d_X, T *d_Y, T *d_mu, T *d_c, T *d_, T *d_G, T *d_I, T *d_beta, T *d_betaOLS, T *d_d,
    T *d_gamma, T *d_cmax, T *d_upper1, T *d_normb, int *d_lVars, int *d_nVars, int *d_ind,
    int *d_step, int *d_lasso,
    int *d_done, int *d_act, int *d_ctrl,
    int M, int Z, int num_models, T g)
{
    int max_vars = min(M - 1, Z - 1);
    int top = num_models;

    h_init_with<T>(d_mu, 0, num_models, M - 1);
    h_init_with<T>(d_beta, 0, num_models, Z - 1);

    h_init_with<int>(d_nVars, 0, num_models, 1);
    h_init_with<int>(d_lasso, 0, num_models, 1);
    h_init_with<int>(d_done, 0, num_models, 1);
    h_init_with<int>(d_step, 0, num_models, 1);

    // Cublas setup.
    cudaDeviceSynchronize();
    int *d_pivot, *d_info, batch = 1;
    cudaMallocManaged(&d_pivot, 3 * sizeof(int));
    cudaMallocManaged(&d_info, sizeof(int));
    d_info[0] = 0;
    T **dd_G, **dd_I;
    cudaMallocManaged(&dd_G, sizeof(T *));
    cudaMallocManaged(&dd_I, sizeof(T *));
    dd_G[0] = d_G;
    dd_I[0] = d_I;
    cublasHandle_t handle;
    cublasCreate(&handle);
        
    time_t timer;
    time(&timer);
    int *h_done = new int[num_models], *h_nVars = new int[num_models];
    cout << "Larsen code stated at " << ctime(&timer);

    while (1) {
        d_ctrl[0] = d_ctrl[1] = 0;
        h_check(d_ctrl, d_step, d_nVars, d_done, M, Z, num_models);
        cudaDeviceSynchronize();
        if (d_ctrl[0] == 0)
            break;

        h_corr<T>(d_X, d_Y, d_mu, d_c, d_, d_act, d_done, M, Z, num_models);
        
        h_exc_corr<T>(d_, d_lVars, d_nVars, d_done, M, Z, num_models);
        h_max_corr<T>(d_, d_cmax, d_ind, d_done, Z, num_models);
        
        h_lasso_add(d_ind, d_lVars, d_nVars, d_lasso, d_done, M, Z, num_models);
        
        h_xincty<T>(d_X, d_Y, d_, d_lVars, d_nVars, d_act, d_done, M, Z, num_models);
        
        cudaMemcpy(h_done, d_done, num_models * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_nVars, d_nVars, num_models * sizeof(int), cudaMemcpyDeviceToHost);
        for (int j = 0; j < num_models; j++) {
            if (h_done[j])
                continue;
            int ni = h_nVars[j];
            h_set_gram<T>(d_X, d_G, d_I, d_lVars, d_act, ni, M, Z, j, num_models);
            
            cublasSgetrfBatched(handle, ni, dd_G, ni, d_pivot, d_info, batch);
            cublasSgetriBatched(handle, ni, (const T **)dd_G, ni, d_pivot, dd_I, ni, d_info, batch);
            
            h_betaols<T>(d_I, d_, d_betaOLS, ni, j, M, Z, num_models);
        }
        h_dgamma<T>(d_X, d_betaOLS, d_mu, d_d, d_beta, d_gamma, d_lVars, d_nVars, d_act, d_done, M, Z, num_models);
        h_min_gamma<T>(d_gamma, d_ind, d_nVars, d_done, M, Z, num_models);
        
        h_xtd<T>(d_X, d_d, d_, d_c, d_cmax, d_act, d_done, M, Z, num_models);
        
        h_exc_tmp<T>(d_, d_lVars, d_nVars, d_done, M, Z, num_models);
        h_min_tmp<T>(d_, d_done, Z, num_models);
        
        h_lasso_dev<T>(d_, d_gamma, d_nVars, d_lasso, d_done, M, Z, num_models);
        
        h_update<T>(d_gamma, d_mu, d_beta, d_betaOLS, d_d,
            d_lVars, d_nVars, d_done, M, Z, num_models);
        
        h_lasso_drop(d_ind, d_lVars, d_nVars, d_lasso, d_done, M, Z, num_models);
        
        h_ress<T>(d_Y, d_mu, d_, d_act, d_done, M, Z, num_models);
        
        h_final<T>(d_, d_beta, d_upper1, d_normb, d_nVars, d_step,
            d_done, d_ctrl, M, Z, num_models, g);

        cudaDeviceSynchronize();
        if (d_ctrl[1] == 0)
            continue;
        // Remove completed models and write them to a file and replace them with new models.
        for (int j = 0; j < num_models && top < Z; j++) {
            if (d_done[j]) {
                // Replace the completed model index in the buffer with model top which is the next model in the stack to be added.
                d_act[j] = top++;
                d_nVars[j] = d_lasso[j] = d_step[j] = 0;
                printf("Model stack at %d\n", top);
            }
        }
        h_clear<T>(d_beta, d_mu, d_done, M, Z, num_models);
        cudaDeviceSynchronize();
        for (int j = 0; j < num_models; j++)
            if (d_done[j])
                d_done[j] = 0;
    }
    cout << "Larsen code finished at " << ctime(&timer);
    cudaFree(d_pivot);
    cudaFree(d_info);
    cudaFree(dd_G);
    cudaFree(dd_I);
    cublasDestroy(handle);
}

#endif