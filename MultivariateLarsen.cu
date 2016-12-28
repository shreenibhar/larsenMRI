#include <iostream>
#include <fstream>
#include <cstring>
#include <cmath>
#include <ctime>
#include "fileProc.h"
#include "larsProc.h"
#define INF 50000

using namespace std;
//N is the number of models to solve in parallel.
int N = 1700;
//Calculate N based on available memory.
int calcN(int M, int Z){
int	n = min(M - 1, Z - 1);
size_t	avail, total;
	cudaMemGetInfo(&avail, &total);
//Calculating Max N possible.
	N = (avail - 2 * n * n * sizeof(double)) / (2 * (M - 1) * sizeof(double) + 3 * (Z - 1) * sizeof(double) + 2 * n * sizeof(double) + 3 * sizeof(double) + n * sizeof(int) + 6 * sizeof(int));
//Taking 95% of that.
	N = 0.90 * N;
	cout << "Setting buffer size N = " << N << endl;
	return 0;
}
//Initialize all cuda vars.
int initAll(double *&d_G, double *&d_I, double *&d_mu, double *&d_d, double *&d_c, double *&d_, double *&d_beta, double *&d_betaOLS, double *&d_gamma, double *&d_cmax, double *&d_upper1, double *&d_normb, int *&d_lVars, int *&d_nVars, int *&d_step, int *&d_ind, int *&d_done, int *&d_lasso, int *&d_act, int M, int Z){
int	n = min(M - 1, Z - 1);
//Gram matrix and its inverse.        
        cudaMalloc((void **)&d_G, n * n * sizeof(double));
        cudaMalloc((void **)&d_I, n * n * sizeof(double));
//Residual and d variables.
        cudaMalloc((void **)&d_mu, (M - 1) * N * sizeof(double));
        cudaMalloc((void **)&d_d, (M - 1) * N * sizeof(double));
//Correlation, temp and beta.
        cudaMalloc((void **)&d_c, (Z - 1) * N * sizeof(double));
        cudaMalloc((void **)&d_, (Z - 1) * N * sizeof(double));
        cudaMallocManaged((void **)&d_beta, (Z - 1) * N *sizeof(double));
//betaOLS, gamma variables.
        cudaMalloc((void **)&d_betaOLS, n * N * sizeof(double));
        cudaMalloc((void **)&d_gamma, n * N * sizeof(double));
//max correlation cmax, upper1 and normb.
        cudaMalloc((void **)&d_cmax, N * sizeof(double));
        cudaMallocManaged((void **)&d_upper1, N * sizeof(double));
        cudaMalloc((void **)&d_normb, N * sizeof(double));
//list of included variables lvars.
        cudaMalloc((void **)&d_lVars, n * N * sizeof(int));
//number of included variables nvars, step, store indexes variable, is model done variable, lasso condition variable, act is the list of current models executing.
        cudaMallocManaged((void **)&d_nVars, N * sizeof(int));
        cudaMallocManaged((void **)&d_step, N * sizeof(int));
        cudaMalloc((void **)&d_ind, N * sizeof(int));
        cudaMallocManaged((void **)&d_done, N * sizeof(int));
        cudaMallocManaged((void **)&d_lasso, N * sizeof(int));
        cudaMallocManaged((void **)&d_act, N * sizeof(int));	
	return 0;
}
//Delete all cuda vars.
int freeAll(double *&d_new1, double *&d_new, double *&d_G, double *&d_I, double *&d_mu, double *&d_d, double *&d_c, double *&d_, double *&d_beta, double *&d_betaOLS, double *&d_gamma, double *&d_cmax, double *&d_upper1, double *&d_normb, int *&d_lVars, int *&d_nVars, int *&d_step, int *&d_ind, int *&d_done, int *&d_lasso, int *&d_act){
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

int main(int argc, char *argv[]){
//Delete files used in the code if already existing.
	removeUsedFiles();
//Reading flattened MRI image.
dMatrix d_mat = readFlatMRI(argv[argc - 1]);
int     M = d_mat.M, Z = d_mat.N, i;
	cout << "Read FMRI Data of shape:" << M << ' ' << Z << endl;
//Computing max N possible.
	calcN(M, Z);
//Remove first and last row for new1 and new respectively and normalize.
double	*d_new1, *d_new;
dMatrix	*list = procFlatMRI(d_mat);
	d_new1 = list[0].dmatrix;
	d_new = list[1].dmatrix;
//Initialize all Lars variables.
double	*d_G, *d_I, *d_mu, *d_d, *d_c, *d_, *d_beta, *d_betaOLS,*d_gamma;
double  *d_cmax, *d_upper1, *d_normb;
int	*d_lVars, *d_nVars, *d_step, *d_ind, *d_done, *d_lasso, *d_act;
int 	n = min(M - 1, Z - 1);
	initAll(d_G, d_I, d_mu, d_d, d_c, d_, d_beta, d_betaOLS, d_gamma, d_cmax, d_upper1, d_normb, d_lVars, d_nVars, d_step, d_ind, d_done, d_lasso, d_act, M, Z);
//Setting initial executing models.
int	*h_act = new int[N];
	for(i = 0;i < N;i++)h_act[i] = i;
	cudaMemcpy(d_act, h_act, N * sizeof(int),cudaMemcpyHostToDevice);
	delete[] h_act;
//Execute lars model.
	lars(d_new1, d_new, d_mu, d_c, d_, d_G, d_I, d_beta, d_betaOLS, d_d, d_gamma, d_cmax, d_upper1, d_normb, d_lVars, d_nVars, d_ind, d_step, d_done, d_lasso, M, Z, d_act, N, 0.43);
//Clearing all Lars variables.
	freeAll(d_new1, d_new, d_G, d_I, d_mu, d_d, d_c, d_, d_beta, d_betaOLS, d_gamma, d_cmax, d_upper1, d_normb, d_lVars, d_nVars, d_step, d_ind, d_done, d_lasso, d_act);
//Time stop.
	return 0;
}
