#include <bits/stdc++.h>
#include <cuda_runtime.h>

#include "file_proc.h"
#include "lars_proc.h"
#include "utilities.h"

typedef float precision;

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
    dmatrix<precision> number = read_flat_mri<precision>(argv[1]);
    printf("Read FMRI Data of shape:(%d,%d)\n", number.M, number.N);
        
    // Remove first and last row for new1 and new respectively and normalize.
    dmatrix<precision> X, Y;
    X.M = number.M - 1;
    X.N = number.N;
    cudaMallocManaged(&X.d_mat, X.M * X.N * sizeof(precision));
    Y.M = number.M - 1;
    Y.N = number.N;
    cudaMallocManaged(&Y.d_mat, Y.M * Y.N * sizeof(precision));
    proc_flat_mri<precision>(number, X, Y);

    // // Declare all Lars variables.
    dmatrix<precision> Xt, Yt, y, mu, c, _, __, G, I, beta, betaOls, d, gamma, cmax, upper1, normb;
    dmatrix<int> lVars, nVars, ind, step, lasso, done, act, ctrl;
    dmatrix<bool> maskVars;

    // Number of models to solve in parallel.
    int num_models = str_to_int(argv[2]);
    printf("Number of models in ||l:%d\n", num_models);

    //Pruning larsen condition.
    precision g = 0.43;

    // Initialize all Larsen variables.
    Xt.M = X.N;
    Xt.N = X.M;
    cudaMallocManaged(&Xt.d_mat, Xt.M * Xt.N * sizeof(precision));
    Yt.M = Y.N;
    Yt.N = Y.M;
    cudaMallocManaged(&Yt.d_mat, Yt.M * Yt.N * sizeof(precision));
    y.M = num_models;
    y.N = Y.M;
    cudaMallocManaged(&y.d_mat, y.M * y.N * sizeof(precision));
    mu.M = num_models;
    mu.N = y.N;
    cudaMallocManaged(&mu.d_mat, mu.M * mu.N * sizeof(precision));
    c.M = num_models;
    c.N = X.N - 1;
    cudaMallocManaged(&c.d_mat, c.M * c.N * sizeof(precision));
    _.M = c.M;
    _.N = c.N;
    cudaMallocManaged(&_.d_mat, _.M * _.N * sizeof(precision));
    __.M = y.M;
    __.N = y.N;
    cudaMallocManaged(&__.d_mat, __.M * __.N * sizeof(precision));
    G.M = y.N;
    G.N = y.N;
    cudaMallocManaged(&G.d_mat, G.M * G.N * sizeof(precision));
    I.M = y.N;
    I.N = y.N;
    cudaMallocManaged(&I.d_mat, I.M * I.N * sizeof(precision));
    beta.M = num_models;
    beta.N = c.N;
    cudaMallocManaged(&beta.d_mat, beta.M * beta.N * sizeof(precision));
    betaOls.M = num_models;
    betaOls.N = y.N;
    cudaMallocManaged(&betaOls.d_mat, betaOls.M * betaOls.N * sizeof(precision));
    d.M = num_models;
    d.N = y.N;
    cudaMallocManaged(&d.d_mat, d.M * d.N * sizeof(precision));
    gamma.M = num_models;
    gamma.N = y.N;
    cudaMallocManaged(&gamma.d_mat, gamma.M * gamma.N * sizeof(precision));
    cmax.M = num_models;
    cmax.N = 1;
    cudaMallocManaged(&cmax.d_mat, cmax.M * cmax.N * sizeof(precision));
    upper1.M = num_models;
    upper1.N = 1;
    cudaMallocManaged(&upper1.d_mat, upper1.M * upper1.N * sizeof(precision));
    normb.M = num_models;
    normb.N = 1;
    cudaMallocManaged(&normb.d_mat, normb.M * normb.N * sizeof(precision));

    lVars.M = num_models;
    lVars.N = y.N;
    cudaMallocManaged(&lVars.d_mat, lVars.M * lVars.N * sizeof(int));
    nVars.M = num_models;
    nVars.N = 1;
    cudaMallocManaged(&nVars.d_mat, nVars.M * nVars.N * sizeof(int));
    ind.M = num_models;
    ind.N = 1;
    cudaMallocManaged(&ind.d_mat, ind.M * ind.N * sizeof(int));
    step.M = num_models;
    step.N = 1;
    cudaMallocManaged(&step.d_mat, step.M * step.N * sizeof(int));
    lasso.M = num_models;
    lasso.N = 1;
    cudaMallocManaged(&lasso.d_mat, lasso.M * lasso.N * sizeof(int));
    done.M = num_models;
    done.N = 1;
    cudaMallocManaged(&done.d_mat, done.M * done.N * sizeof(int));
    act.M = num_models;
    act.N = 1;
    cudaMallocManaged(&act.d_mat, act.M * act.N * sizeof(int));
    ctrl.M = 2;
    ctrl.N = 1;
    cudaMallocManaged(&ctrl.d_mat, ctrl.M * ctrl.N * sizeof(int));

    maskVars.M = num_models;
    maskVars.N = c.N;
    cudaMallocManaged(&maskVars.d_mat, maskVars.M * maskVars.N * sizeof(bool));

    // Execute lars.
    lars<precision>(X, Xt, Y, Yt, y, mu, c, _, __, G, I, beta, betaOls, d, gamma, cmax, upper1, normb,
        lVars, nVars, maskVars, ind, step, lasso, done, act, ctrl,
        g);

    cudaFree(X.d_mat);
    cudaFree(Y.d_mat);
    cudaFree(Xt.d_mat);
    cudaFree(Yt.d_mat);
    cudaFree(y.d_mat);
    cudaFree(mu.d_mat);
    cudaFree(c.d_mat);
    cudaFree(_.d_mat);
    cudaFree(__.d_mat);
    cudaFree(G.d_mat);
    cudaFree(I.d_mat);
    cudaFree(beta.d_mat);
    cudaFree(betaOls.d_mat);
    cudaFree(d.d_mat);
    cudaFree(gamma.d_mat);
    cudaFree(cmax.d_mat);
    cudaFree(upper1.d_mat);
    cudaFree(normb.d_mat);

    cudaFree(lVars.d_mat);
    cudaFree(nVars.d_mat);
    cudaFree(ind.d_mat);
    cudaFree(step.d_mat);
    cudaFree(lasso.d_mat);
    cudaFree(done.d_mat);
    cudaFree(act.d_mat);
    cudaFree(ctrl.d_mat);

    cudaFree(maskVars.d_mat);

    return 0;
}