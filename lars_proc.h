#include <iostream>
#include <fstream>
#include <cstring>
#include <cmath>
#include <cuda.h>

#include "lars_proc_kernel.h"

#define INF 50000

using namespace std;

extern int N;

// X input(M - 1, Z).
// Y output(M - 1, Z).
// act is the list of current evaluating models and l is the buffer size.
// g is the lars pruning variable.
int lars(float *&d_X, float *&d_Y, float *&d_mu,
    float *&d_c, float *&d_, float *&d_G, float *&d_I,
    float *&d_beta, float *&d_betaOLS, float *&d_d,
    float *&d_gamma, float *&d_cmax, float *&d_upper1,
    float *&d_normb, int *&d_lVars, int *&d_nVars,
    int *&d_ind, int *&d_step, int *&d_done, int *&d_lasso,
    int M, int Z, int *d_act, int l, float g)
{
    time_t timer;

    int n = min(M - 1, Z - 1), i, j, top = l;

    dim3 bl(1024);
    dim3 gl((l + bl.x - 1) / bl.x);

    dim3 bZl(32, 32);
    dim3 gZl((Z + bZl.x - 1) / bZl.x, (l + bZl.y - 1) / bZl.y);

    dim3 bMl(32, 32);
    dim3 gMl((M + bMl.x - 1) / bMl.x, (l + bMl.y - 1) / bMl.y);

    dim3 bnl(32, 32);
    dim3 gnl((n + bnl.x - 1) / bnl.x, (l + bnl.y - 1) / bnl.y);

    int *d_ctrl, *d_towrite;
    float *d_corr_flop, *d_ols_flop, *d_add_flop, *d_drop_flop, *d_other_flop;

	cudaMalloc((void **)&d_corr_flop, N * sizeof(float));
	cudaMalloc((void **)&d_ols_flop, N * sizeof(float));
	cudaMalloc((void **)&d_add_flop, N * sizeof(float));
	cudaMalloc((void **)&d_drop_flop, N * sizeof(float));
	cudaMalloc((void **)&d_other_flop, N * sizeof(float));
	
    cudaMalloc((void **)&d_towrite, sizeof(int));
	cudaMalloc((void **)&d_ctrl, sizeof(int));	
    
    cudaMemset(d_corr_flop, 0, N * sizeof(float));
    cudaMemset(d_ols_flop, 0, N * sizeof(float));
    cudaMemset(d_add_flop, 0, N * sizeof(float));
    cudaMemset(d_drop_flop, 0, N * sizeof(float));
    cudaMemset(d_other_flop, 0, N * sizeof(float));
	cudaMemset(d_mu, 0, (M - 1) * l * sizeof(float));
    cudaMemset(d_beta, 0, (Z - 1) * l * sizeof(float));
    
    d_init<<<gl, bl>>>(d_nVars, d_lasso, d_done, d_step, l);
        
	time(&timer);	
	cout << "Larsen code stated at " << ctime(&timer) << endl;

    int *h_done = new int[l], *h_nVars = new int[l], *h_step = new int[l], *h_lasso = new int[l];
    int *h_act = new int[l];
    for (int i = 0; i < l; i++)
        h_act[i] = i;
    int *h_ctrl = new int, *h_towrite = new int;

	while (1) {
        cudaMemset(d_ctrl, 0, sizeof(int));
        cudaMemset(d_towrite, 0, sizeof(int));
        d_check<<<gl, bl>>>(d_ctrl, d_step, d_nVars, M, Z, l, d_done, d_towrite);
        cudaMemcpy(h_ctrl, d_ctrl, sizeof(int), cudaMemcpyDeviceToHost);
        if (*h_ctrl == 0)
            break;
        d_corr<<<gZl, bZl>>>(d_X, d_Y, d_mu, d_c, d_, M, Z, d_act, l, d_done, d_corr_flop);
        d_exc_corr<<<gnl, bnl>>>(d_, d_lVars, d_nVars, M, Z, l, d_done);
        d_max_corr<<<gl, bl>>>(d_, d_cmax, d_ind, Z, l, d_done);
        d_lasso_add<<<gl, bl>>>(d_ind, d_lVars, d_nVars, d_lasso, M, Z, l, d_done, d_add_flop);
        d_xincty<<<gnl, bnl>>>(d_X, d_Y, d_, d_lVars, d_nVars, M, Z, d_act, l, d_done, d_other_flop); 

        cudaMemcpy(h_done, d_done, sizeof(int) * l, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_nVars, d_nVars, sizeof(int) * l, cudaMemcpyDeviceToHost);
        for (j = 0; j < l; j++) {
            if (h_done[j])
                continue;
            int nt = h_nVars[j];
            d_set_gram<<<nt, nt>>>(d_X, d_G, d_I, d_lVars, nt, M, Z, j, d_act, l, d_other_flop);
            for (i = 0; i < nt; i++) {
                nodiag_normalize<<<1, nt>>>(d_G, d_I, nt, i, d_other_flop);
                diag_normalize<<<1, 1>>>(d_G, d_I, nt, i);
                gauss_jordan<<<nt, nt>>>(d_G, d_I, nt, i, d_other_flop);
                set_zero<<<1, nt>>>(d_G, d_I, nt, i);
            }
            d_betaols<<<1, nt>>>(d_I, d_, d_betaOLS, nt, j, M, Z, l, d_ols_flop);
        }

        d_d_gamma<<<gMl, bMl>>>(d_X, d_mu, d_beta, d_betaOLS, d_gamma, d_d, d_lVars, d_nVars, M, Z, d_act, l, d_done, d_other_flop);
        d_min_gamma<<<gl, bl>>>(d_gamma, d_ind, d_nVars, M, Z, l, d_done);
        d_xtd<<<gZl, bZl>>>(d_X, d_c, d_, d_d, d_cmax, M, Z, d_act, l, d_done, d_other_flop);
        d_exc_tmp<<<gnl, bnl>>>(d_, d_lVars, d_nVars, M, Z, l, d_done);
        d_min_tmp<<<gl, bl>>>(d_, Z, l, d_done);
        d_lasso_dev<<<gl, bl>>>(d_, d_gamma, d_nVars, d_lasso, n, l, d_done);
        d_update<<<gMl, bMl>>>(d_gamma, d_mu, d_beta, d_betaOLS, d_d, d_lVars, d_nVars, M, Z, l, d_done, d_other_flop);
        d_lasso_drop<<<l, n>>>(d_ind, d_lVars, d_nVars, d_lasso, M, Z, l, d_done, d_drop_flop);
        d_ress<<<gMl, bMl>>>(d_Y, d_mu, d_, M, Z, d_act, l, d_done, d_other_flop);
        d_final<<<gl, bl>>>(d_, d_beta, d_upper1, d_normb, d_nVars, d_step, g, M, Z, l, d_done, d_other_flop, d_towrite);

        cudaMemcpy(h_towrite, d_towrite, sizeof(int), cudaMemcpyDeviceToHost);
        if (*h_towrite == 0)
            continue;
        cudaMemcpy(h_done, d_done, sizeof(int) * l, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_nVars, d_nVars, sizeof(int) * l, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_lasso, d_lasso, sizeof(int) * l, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_step, d_step, sizeof(int) * l, cudaMemcpyDeviceToHost);
        // // Remove completed models and write them to a file and replace them with new models.
        for (j = 0; j < l && top < Z; j++) {
            if (h_done[j]) {
                // writeModel(d_beta, d_step, d_upper1, M, Z, d_act[j], j).
                // Replace the completed model index in the buffer with model top which is the next model in the stack to be added.
                h_act[j] = top++;
                h_nVars[j] = h_done[j] = h_lasso[j] = 0;
                h_step[j] = 1;
                cudaMemset(d_mu + j * (M - 1), 0, (M - 1) * sizeof(float));
                cudaMemset(d_beta + j * (Z - 1), 0, (Z - 1) * sizeof(float));
                cout << "\rModel stack top at " << top;
            }
        }
        cudaMemcpy(d_act, h_act, sizeof(int) * l, cudaMemcpyHostToDevice);
        cudaMemcpy(d_done, h_done, sizeof(int) * l, cudaMemcpyHostToDevice);
        cudaMemcpy(d_nVars, h_nVars, sizeof(int) * l, cudaMemcpyHostToDevice);
        cudaMemcpy(d_lasso, h_lasso, sizeof(int) * l, cudaMemcpyHostToDevice);
        cudaMemcpy(d_step, h_step, sizeof(int) * l, cudaMemcpyHostToDevice);
    }
	// cout << " Clearing the buffer.\n";
    // Write the remaining models in the buffer to clear the buffer.
	// for (j = 0; j < l; j++){
	// 	writeModel(d_beta, d_step, d_upper1, M, Z, d_act[j], j);
	// }
	time(&timer);
	cout << "Larsen code finished at " << ctime(&timer) << endl;	

 //    float sum_flop = 0, sum_corr_flop = 0, sum_ols_flop = 0, sum_add_flop = 0, sum_drop_flop = 0, sum_other_flop;
	// cudaDeviceSynchronize();
	// for (int i = 0; i < N; i++) {
	// 	sum_corr_flop += d_corr_flop[i];
	// 	sum_ols_flop += d_ols_flop[i];
	// 	sum_add_flop += d_add_flop[i];
	// 	sum_drop_flop += d_drop_flop[i];
	// 	sum_other_flop += d_other_flop[i];
	// }
	// sum_flop = sum_corr_flop + sum_ols_flop + sum_add_flop + sum_drop_flop + sum_other_flop;
    // cout << sum_corr_flop << ' ' << sum_ols_flop << ' ' << sum_add_flop << ' ' << sum_drop_flop << ' ' << sum_other_flop << ' ' << sum_flop << endl;
	
    cudaFree(d_corr_flop);
	cudaFree(d_ols_flop);
	cudaFree(d_add_flop);
	cudaFree(d_drop_flop);
	cudaFree(d_other_flop);
    cudaFree(d_ctrl);
	cudaFree(d_towrite);
    return 0;
}
