#include <bits/stdc++.h>
#include <cuda_runtime.h>

#include "file_proc.h"
#include "lars_proc.h"

#define INF 50000

using namespace std;

// Initialize all cuda vars.
template<class T>
void init_all(T *&d_mu, T *&d_c, T *&d_, T *&d_G, T *&d_I, T *&d_beta, T *&d_betaOLS,
    T *&d_d, T *&d_gamma, T *&d_cmax, T *&d_upper1, T *&d_normb, int *&d_lVars,
    int *&d_nVars, int *&d_ind, int *&d_step, int *&d_lasso,
    int *&d_done, int *&d_act, int *&d_ctrl,
    int M, int Z, int num_models, T g)
{
    int max_vars = min(M - 1, Z - 1);
        
    cudaMallocManaged(&d_mu, num_models * (M - 1) * sizeof(T));
    cudaMallocManaged(&d_c, num_models * (Z - 1) * sizeof(T));
    cudaMallocManaged(&d_, num_models * (Z - 1) * sizeof(T));
    cudaMallocManaged(&d_G, max_vars * max_vars * sizeof(T));
    cudaMallocManaged(&d_I, max_vars * max_vars * sizeof(T));
    cudaMallocManaged(&d_beta, num_models * (Z - 1) * sizeof(T));
    cudaMallocManaged(&d_betaOLS, num_models * max_vars * sizeof(T));
    cudaMallocManaged(&d_d, num_models * (M - 1) * sizeof(T));
    cudaMallocManaged(&d_gamma, num_models * max_vars * sizeof(T));
    cudaMallocManaged(&d_cmax, num_models * sizeof(T));
    cudaMallocManaged(&d_upper1, num_models * sizeof(T));
    cudaMallocManaged(&d_normb, num_models * sizeof(T));
        
    cudaMallocManaged(&d_lVars, num_models * max_vars * sizeof(int));
    cudaMallocManaged(&d_nVars, num_models * sizeof(int));
    cudaMallocManaged(&d_ind, num_models * sizeof(int));
    cudaMallocManaged(&d_step, num_models * sizeof(int));
    cudaMallocManaged(&d_lasso, num_models * sizeof(int));

    cudaMallocManaged(&d_done, num_models * sizeof(int));
    cudaMallocManaged(&d_act, num_models * sizeof(int));
    cudaMallocManaged(&d_ctrl, 2 * sizeof(int));
}

// Delete all cuda vars.
template<class T>
void free_all(T *d_X, T *d_Y, T *d_mu, T *d_c, T *d_, T *d_G, T *d_I, T *d_beta, T *d_betaOLS,
    T *d_d, T *d_gamma, T *d_cmax, T *d_upper1, T *d_normb, int *d_lVars, int *d_nVars,
    int *d_ind, int *d_step, int *d_lasso,
    int *d_done, int *d_act, int *d_ctrl,
    int M, int Z, int num_models, T g)
{
    cudaFree(d_X);
    cudaFree(d_Y);
    cudaFree(d_mu);
    cudaFree(d_c);
    cudaFree(d_);
    cudaFree(d_G);
    cudaFree(d_beta);
    cudaFree(d_betaOLS);
    cudaFree(d_d);
    cudaFree(d_gamma);
    cudaFree(d_cmax);
    cudaFree(d_upper1);
    cudaFree(d_normb);

    cudaFree(d_lVars);
    cudaFree(d_nVars);
    cudaFree(d_ind);
    cudaFree(d_step);
    cudaFree(d_lasso);
        
    cudaFree(d_done);
    cudaFree(d_act);
    cudaFree(d_ctrl);
}

int str_to_int(char *argv) {
    int i = 0, num_models = 0;
    while(argv[i] != '\0') {
        num_models = num_models * 10 + argv[i] - '0';
        i++;
    }
    return num_models;
}

int main(int argc, char *argv[]) {
    // Delete files used in the code if already existing.
    remove_used_files();
        
    // Reading flattened MRI image.
    float **ret = read_flat_mri<float>(argv[argc - 1]);
    int M = 295, Z = 39658;
    cout << "Read FMRI Data of shape: " << M << ' ' << Z << endl;
        
    // Remove first and last row for new1 and new respectively and normalize.
    ret = proc_flat_mri<float>(ret[0], M, Z);
    float *d_X = ret[0], *d_Y = ret[1];

    // Initialize all Lars variables.
    float *d_mu, *d_c, *d_, *d_G, *d_I, *d_beta, *d_betaOLS, *d_d, *d_gamma, *d_cmax,
        *d_upper1, *d_normb;
    int *d_lVars, *d_nVars, *d_ind, *d_step, *d_lasso, *d_done, *d_act, *d_ctrl;
    
    // Number of models to solve in parallel.
    int num_models = 1000;
    cout << "Number of models in ||l: " << num_models << endl;

    //Pruning larsen condition.
    float g = 0.43;

    init_all<float>(d_mu, d_c, d_, d_G, d_I, d_beta, d_betaOLS, d_d, d_gamma, d_cmax,
        d_upper1, d_normb, d_lVars, d_nVars, d_ind, d_step, d_lasso,
        d_done, d_act, d_ctrl,
        M, Z, num_models, g);

    // Setting initial executing models.
    cudaDeviceSynchronize();
    for (int i = 0; i < num_models; i++)
        d_act[i] = i;

    // Execute lars.
    lars<float>(d_X, d_Y, d_mu, d_c, d_, d_G, d_I, d_beta, d_betaOLS, d_d, d_gamma, d_cmax,
        d_upper1, d_normb, d_lVars, d_nVars, d_ind, d_step, d_lasso,
        d_done, d_act, d_ctrl,
        M, Z, num_models, g);

    // Clearing all Lars variables.
    free_all<float>(d_X, d_Y, d_mu, d_c, d_, d_G, d_I, d_beta, d_betaOLS, d_d, d_gamma,
        d_cmax, d_upper1, d_normb, d_lVars, d_nVars, d_ind, d_step, d_lasso,
        d_done, d_act, d_ctrl,
        M, Z, num_models, g);
    return 0;
}