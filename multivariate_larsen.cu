#include <iostream>
#include <fstream>
#include <cstring>
#include <cmath>

#include "file_proc.h"
#include "lars_proc.h"

#define INF 50000

using namespace std;

// N is the number of models to solve in parallel.
int N = 500;

// Initialize all cuda vars.
int initAll(float *&d_G, float *&d_I, float *&d_mu, float *&d_d, float *&d_c, float *&d_, float *&d_beta, float *&d_betaOLS, float *&d_gamma, float *&d_cmax, float *&d_upper1, float *&d_normb, int *&d_lVars, int *&d_nVars, int *&d_step, int *&d_ind, int *&d_done, int *&d_lasso, int *&d_act, int M, int Z){
        int n = min(M - 1, Z - 1);
        
        // Gram matrix and its inverse.        
        cudaMalloc((void **)&d_G, n * n * sizeof(float));
        cudaMalloc((void **)&d_I, n * n * sizeof(float));

        // Residual and d variables.
        cudaMalloc((void **)&d_mu, (M - 1) * N * sizeof(float));
        cudaMalloc((void **)&d_d, (M - 1) * N * sizeof(float));

        // Correlation, temp and beta.
        cudaMalloc((void **)&d_c, (Z - 1) * N * sizeof(float));
        cudaMalloc((void **)&d_, (Z - 1) * N * sizeof(float));
        cudaMalloc((void **)&d_beta, (Z - 1) * N *sizeof(float));

        // betaOLS, gamma variables.
        cudaMalloc((void **)&d_betaOLS, n * N * sizeof(float));
        cudaMalloc((void **)&d_gamma, n * N * sizeof(float));

        // max correlation cmax, upper1 and normb.
        cudaMalloc((void **)&d_cmax, N * sizeof(float));
        cudaMalloc((void **)&d_upper1, N * sizeof(float));
        cudaMalloc((void **)&d_normb, N * sizeof(float));

        // list of included variables lvars.
        cudaMalloc((void **)&d_lVars, n * N * sizeof(int));

        // number of included variables nvars, step, store indexes variable, is model done variable, lasso condition variable, act is the list of current models executing.
        cudaMalloc((void **)&d_nVars, N * sizeof(int));
        cudaMalloc((void **)&d_step, N * sizeof(int));
        cudaMalloc((void **)&d_ind, N * sizeof(int));
        cudaMalloc((void **)&d_done, N * sizeof(int));
        cudaMalloc((void **)&d_lasso, N * sizeof(int));
        cudaMalloc((void **)&d_act, N * sizeof(int));	
	return 0;
}

// Delete all cuda vars.
int freeAll(float *&d_new1, float *&d_new, float *&d_G,
        float *&d_I, float *&d_mu, float *&d_d,
        float *&d_c, float *&d_, float *&d_beta,
        float *&d_betaOLS, float *&d_gamma,
        float *&d_cmax, float *&d_upper1,
        float *&d_normb, int *&d_lVars, int *&d_nVars,
        int *&d_step, int *&d_ind, int *&d_done,
        int *&d_lasso, int *&d_act)
{
        cudaFree(d_new1);
        cudaFree(d_new);
        cudaFree(d_G);
        cudaFree(d_I);
        cudaFree(d_mu);
        cudaFree(d_d);
        cudaFree(d_c);
        cudaFree(d_);
        cudaFree(d_beta);
        cudaFree(d_betaOLS);
        cudaFree(d_gamma);
        cudaFree(d_cmax);
        cudaFree(d_upper1);
        cudaFree(d_normb);
        cudaFree(d_lVars);
        cudaFree(d_nVars);
        cudaFree(d_step);
        cudaFree(d_ind);
        cudaFree(d_done);
        cudaFree(d_lasso);
        cudaFree(d_act);
	return 0;
}

int main(int argc, char *argv[]) {
        // Delete files used in the code if already existing.
	removeUsedFiles();
        
        // Reading flattened MRI image.
        dMatrix d_mat = readFlatMRI(argv[argc - 1]);
        int M = d_mat.M, Z = d_mat.N, i;
	cout << "Read FMRI Data of shape:" << M << ' ' << Z << endl;
        
        // Remove first and last row for new1 and new respectively and normalize.
        float *d_new1, *d_new;
        dMatrix	*list = procFlatMRI(d_mat);
	d_new1 = list[0].dmatrix;
	d_new = list[1].dmatrix;

        // Initialize all Lars variables.
        float *d_G, *d_I, *d_mu, *d_d, *d_c, *d_, *d_beta, *d_betaOLS,*d_gamma;
        float *d_cmax, *d_upper1, *d_normb;
        int *d_lVars, *d_nVars, *d_step, *d_ind, *d_done, *d_lasso, *d_act;
        int n = min(M - 1, Z - 1);
	initAll(d_G, d_I, d_mu, d_d, d_c, d_, d_beta, d_betaOLS, d_gamma, d_cmax, d_upper1, d_normb, d_lVars, d_nVars, d_step, d_ind, d_done, d_lasso, d_act, M, Z);

        // Setting initial executing models.
        int *h_act = new int[N];
	for (i = 0; i < N; i++)
                h_act[i] = i;
	cudaMemcpy(d_act, h_act, N * sizeof(int),cudaMemcpyHostToDevice);
	delete[] h_act;

        // Execute lars model.
	lars(d_new1, d_new, d_mu, d_c, d_, d_G, d_I, d_beta, d_betaOLS, d_d, d_gamma, d_cmax, d_upper1, d_normb, d_lVars, d_nVars, d_ind, d_step, d_done, d_lasso, M, Z, d_act, N, 0.43);

        // Clearing all Lars variables.
	freeAll(d_new1, d_new, d_G, d_I, d_mu, d_d, d_c, d_, d_beta, d_betaOLS, d_gamma, d_cmax, d_upper1, d_normb, d_lVars, d_nVars, d_step, d_ind, d_done, d_lasso, d_act);
	return 0;
}
