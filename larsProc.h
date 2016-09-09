#include <iostream>
#include <fstream>
#include <cstring>
#include <cmath>
#include "larsProcKernel.h"
#define INF 50000

using namespace std;

extern int N;
/*
X input(M - 1, Z)
Y output(M - 1, Z)
act is the list of current evaluating models and l is the buffer size.
*/
int lars(double *&d_X, double *&d_Y, double *&d_mu, double *&d_c, double *&d_, double *&d_G, double *&d_I, double *&d_beta, double *&d_betaOLS, double *&d_d, double *&d_gamma, double *&d_cmax, double *&d_upper1, double *&d_normb, int *&d_lVars, int *&d_nVars, int *&d_ind, int *&d_step, int *&d_done, int *&d_lasso, int M, int Z, int *d_act, int l, double g){
int     n = min(M - 1, Z - 1), i, j, top = l;

dim3    bl(1024);
dim3    gl((l + bl.x - 1) / bl.x);

dim3    bZl(32, 32);
dim3    gZl((Z + bZl.x - 1) / bZl.x, (l + bZl.y - 1) / bZl.y);

dim3    bMl(32, 32);
dim3    gMl((M + bMl.x - 1) / bMl.x, (l + bMl.y - 1) / bMl.y);

dim3    bnl(32, 32);
dim3    gnl((n + bnl.x - 1) / bnl.x, (l + bnl.y - 1) / bnl.y);

int     *d_ctrl;
        cudaMallocManaged((void **)&d_ctrl, sizeof(int));
        cudaMemset(d_mu, 0, (M - 1) * l * sizeof(double));
        cudaMemset(d_beta, 0, (Z - 1) * l * sizeof(double));
        dInit<<<gl, bl>>>(d_nVars, d_lasso, d_done, d_step, l);
        while(1){
                cudaDeviceSynchronize();
                *d_ctrl = 0;
                cudaDeviceSynchronize();
                dCheck<<<gl, bl>>>(d_ctrl, d_step, d_nVars, M, Z, l, d_done);
                cudaDeviceSynchronize();
                if(*d_ctrl == 0)break;
                dCorr<<<gZl, bZl>>>(d_X, d_Y, d_mu, d_c, d_, M, Z, d_act, l, d_done);
                dExcCorr<<<gnl, bnl>>>(d_, d_lVars, d_nVars, M, Z, l, d_done);
                dMaxcorr<<<gl, bl>>>(d_, d_cmax, d_ind, Z, l, d_done);
                dLassoAdd<<<gl, bl>>>(d_ind, d_lVars, d_nVars, d_lasso, M, Z, l, d_done);
                dXincTY<<<gnl, bnl>>>(d_X, d_Y, d_, d_lVars, d_nVars, M, Z, d_act, l, d_done);
                for(j = 0;j < l;j++){
                        cudaDeviceSynchronize();
                        if(d_done[j])continue;
                int     nt = d_nVars[j];
                        dSetGram<<<nt, nt>>>(d_X, d_G, d_I, d_lVars, nt, M, Z, j, d_act, l);
                        for(i = 0;i < nt;i++){
                                nodiag_normalize<<<1, nt>>>(d_G, d_I, nt, i);
                                diag_normalize<<<1, 1>>>(d_G, d_I, nt, i);
                                gaussjordan<<<nt, nt>>>(d_G, d_I, nt, i);
                                set_zero<<<1, nt>>>(d_G, d_I, nt, i);
                        }
                        dBetaols<<<1, nt>>>(d_I, d_, d_betaOLS, nt, j, M, Z, l);
                }
                ddgamma<<<gMl, bMl>>>(d_X, d_mu, d_beta, d_betaOLS, d_gamma, d_d, d_lVars, d_nVars, M, Z, d_act, l, d_done);
                dGammamin<<<gl, bl>>>(d_gamma, d_ind, d_nVars, M, Z, l, d_done);
                dXTd<<<gZl, bZl>>>(d_X, d_c, d_, d_d, d_cmax, M, Z, d_act, l, d_done);
                dExctmp<<<gnl, bnl>>>(d_, d_lVars, d_nVars, M, Z, l, d_done);
                dTmpmin<<<gl, bl>>>(d_, Z, l, d_done);
                dLassodev<<<gl, bl>>>(d_, d_gamma, d_nVars, d_lasso, n, l, d_done);
                dUpdate<<<gMl, bMl>>>(d_gamma, d_mu, d_beta, d_betaOLS, d_d, d_lVars, d_nVars, M, Z, l, d_done);
                dLassodrop<<<l, n>>>(d_ind, d_lVars, d_nVars, d_lasso, M, Z, l, d_done);
                dRess<<<gMl, bMl>>>(d_X, d_Y, d_, d_beta, M, Z, d_act, l, d_done);
                dFinal<<<gl, bl>>>(d_, d_beta, d_upper1, d_normb, d_nVars, d_step, g, M, Z, l, d_done);
//Remove completed models and write them to a file and replace them with new models.
		if(top < Z){
                        for(j = 0;j < l;j++){
                                cudaDeviceSynchronize();
                                if(d_done[j]){
                                        writeModel(d_beta, d_step, d_upper1, M, Z, d_act[j], j);
//Replace the completed model index in the buffer with model top which is the next model in the stack to be added.
                                        d_act[j] = top++;
                                        d_nVars[j] = d_done[j] = d_lasso[j] = 0;
                                        d_step[j] = 1;
                                        cudaMemset(d_mu + j * (M - 1), 0, (M - 1) * sizeof(double));
                                        cudaMemset(d_beta + j * (Z - 1), 0, (Z - 1) * sizeof(double));
                                        cout << "\rModel stack top at " << top;
                                        if(top >= Z)break;
                                }
                        }
                }
        }
        cudaFree(d_ctrl);
        return 0;
}
