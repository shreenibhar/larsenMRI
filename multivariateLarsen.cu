#include <bits/stdc++.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "utilities.h"
#include "blas.h"
#include "kernels.h"

typedef float precision;

template<typename T>
void printDeviceVar(T *var, int size, int *ind, int numInd) {
    T *hVar = new T[size];
    cudaMemcpy(hVar, var, size * sizeof(T), cudaMemcpyDeviceToHost);
    for (int i = 0; i < numInd; i++) {
        printf("%f:%d\n", hVar[ind[i]], hVar[ind[i]]);
    }
}

int optimalBlock1D(int problemSize) {
    int blockSize, minR = inf;
    for (int i = 1024; i >= 256; i -= 32) {
        int ans = problemSize % i;
        if (ans < minR) {
            minR = ans;
            blockSize = i;
        }
    }
    return blockSize;
}

int main(int argc, char *argv[]) {
    if (argc < 4) {
        printf("Insufficient parameters, required 3! (flatMriPath, numModels, numStreams)\n");
        return 0;
    }

    // Reading flattened mri image.
    precision *X, *Y;
    IntegerTuple tuple = read_flat_mri<precision>(argv[1], X, Y);
    int M = tuple.M, N = tuple.N;
    printf("Read FMRI Data X of shape: (%d,%d)\n", M, N);
    printf("Read FMRI Data Y of shape: (%d,%d)\n", M, N);

    // Number of models to solve in ||l.
    int numModels = str_to_int(argv[2]);
    int totalModels = N;
    printf("Total number of models: %d\n", totalModels);
    printf("Number of models in ||l: %d\n", numModels);

    int numStreams = str_to_int(argv[3]);
    numStreams = pow(2, int(log(numStreams) / log(2)));
    printf("Number of streams: %d\n", numStreams);
    
    // Declare all lars variables.
    int *nVars, *step, *lasso, *done, *lVars, *cidx, *act, *dropidx;
    int *pivot, *info, *ctrl, *hNVars, *hctrl, *hdone, *hact;
    precision *alp[numModels], *bet[numModels], *ha1, *ha2;
    precision *y, *mu, *r, *beta, *c, *absC, *cmax, *betaOls, *d, *gamma, *cd, *a1, *a2;
    precision *XA[numModels], *G[numModels], *I[numModels], **dG, **dI;

    // Initialize all lars variables.
    init_var<int>(nVars, numModels);
    init_var<int>(step, numModels);
    init_var<int>(lasso, numModels);
    init_var<int>(done, numModels);
    init_var<int>(lVars, numModels * M);
    init_var<int>(cidx, numModels);
    init_var<int>(dropidx, numModels);
    init_var<int>(act, numModels);

    init_var<int>(pivot, numModels * M);
    init_var<int>(info, numModels);
    init_var<int>(ctrl, 2);

    init_var<precision>(y, numModels * M);
    init_var<precision>(mu, numModels * M);
    init_var<precision>(r, numModels * M);
    init_var<precision>(beta, numModels * N);
    init_var<precision>(c, numModels * N);
    init_var<precision>(absC, numModels * N);
    init_var<precision>(cmax, numModels);
    init_var<precision>(betaOls, numModels * M);
    init_var<precision>(d, numModels * M);
    init_var<precision>(gamma, numModels * M);
    init_var<precision>(cd, numModels * N);
    init_var<precision>(a1, numModels);
    init_var<precision>(a2, numModels);

    int maxVariables = min(M, N - 1);
    int maxSteps = 8 * maxVariables;

    int top = numModels;
    double totalFlop = 0;
    double totalTime = 0;

    int bN = optimalBlock1D(N);
    int bM = optimalBlock1D(M);
    int bModM = optimalBlock1D(numModels * M);
    int bModN = optimalBlock1D(numModels * N);
    int bMod = optimalBlock1D(numModels);

    cublasHandle_t hnd;
    cublasCreate(&hnd);
    cudaStream_t streams[numStreams];
    cublasSetPointerMode(hnd, CUBLAS_POINTER_MODE_DEVICE);

    hNVars = new int[numModels];
    hctrl = new int[2];
    hact = new int[numModels];
    hdone = new int[numModels];
    ha1 = new precision[numModels];
    ha2 = new precision[numModels];
    for (int i = 0; i < numModels; i++) {
        init_var<precision>(XA[i], M * M);
        init_var<precision>(G[i], M * M);
        init_var<precision>(I[i], M * M);
        init_var<precision>(alp[i], 1);
        init_var<precision>(bet[i], 1);
    }
    cudaMalloc(&dG, numModels * sizeof(precision *));
    cudaMemcpy(dG, G, numModels * sizeof(precision *), cudaMemcpyHostToDevice);
    cudaMalloc(&dI, numModels * sizeof(precision *));
    cudaMemcpy(dI, I, numModels * sizeof(precision *), cudaMemcpyHostToDevice);

    for (int i = 0; i < numStreams; i++)
        cudaStreamCreate(&streams[i]);

    for (int i = 0; i < numModels; i++) {
        set_model<precision>(Y, y, mu,
                             beta, alp[i], bet[i],
                             nVars, lasso, step,
                             done, act, M, N,
                             i, i, streams[i & (numStreams - 1)],
                             *(new dim3(bN)));
    }
    cudaDeviceSynchronize();

    Debug<precision> debug[totalModels];
    GpuTimer timer;

    printf("\rStack top at %d", top);
    while (true) {
        cudaMemset(ctrl, 0, 2 * sizeof(int));
        check(nVars, step, maxVariables,
              maxSteps, done, ctrl,
              numModels);
        cudaMemcpy(hctrl, ctrl, 2 * sizeof(int), cudaMemcpyDeviceToHost);
        if (hctrl[1] == 1) {
            cudaMemcpy(ha1, a1, numModels * sizeof(precision), cudaMemcpyDeviceToHost);
            cudaMemcpy(ha2, a2, numModels * sizeof(precision), cudaMemcpyDeviceToHost);
            cudaMemcpy(hdone, done, numModels * sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(hNVars, nVars, numModels * sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(hact, act, numModels * sizeof(int), cudaMemcpyDeviceToHost);
            for (int i = 0; i < numModels; i++) {
                if (hdone[i]) {
                    if (top < totalModels) {
                        set_model<precision>(Y, y, mu,
                                             beta, alp[i], bet[i],
                                             nVars, lasso, step,
                                             done, act, M, N,
                                             i, top++, streams[i & (numStreams - 1)],
                                             *(new dim3(bN)));
                        printf("\rStack top at %d", top);
                    }
                    if (debug[hact[i]].nVars == -1) {
                        debug[hact[i]].nVars = hNVars[i];
                        debug[hact[i]].a1 = ha1[i];
                        debug[hact[i]].a2 = ha2[i];
                    }
                }
            }
        }
        if (hctrl[0] == 0) {
            break;
        }
        timer.start();
        mat_sub<precision>(y, mu, r,
                           numModels * M, *(new dim3(bModM)));
        cudaDeviceSynchronize();
        cublasSetStream(hnd, NULL);
        gemm(hnd, CUBLAS_OP_N, CUBLAS_OP_N,
                 N, numModels, M,
                 alp[0], X, N,
                 r, M, bet[0],
                 c, N);
        cudaDeviceSynchronize();
        exclude<precision>(c, cmax, lVars, nVars,
                           act, M, N,
                           numModels, 0, *(new dim3(bModM)));
        cudaDeviceSynchronize();
        for (int i = 0; i < numModels; i++) {
            amaxFabs(c + i * N, cmax + i, N, streams[i & (numStreams - 1)], *(new dim3(1024)));
        }
        cudaDeviceSynchronize();
        set_cidx<precision>(cmax, cidx, c,
                            N, numModels, *(new dim3(bModN)));
        cudaDeviceSynchronize();
        lasso_add(lasso, lVars, nVars,
                  cidx, M, N,
                  numModels, *(new dim3(bMod)));
        cudaDeviceSynchronize();
        cudaMemcpy(hNVars, nVars, numModels * sizeof(int), cudaMemcpyDeviceToHost);
        for (int i = 0; i < numModels; i++) {
            gather<precision>(XA[i], X, lVars,
                              hNVars[i], M, N,
                              i, streams[(i & (numStreams - 1))], *(new dim3(optimalBlock1D(hNVars[i] * M))));
        }
        cudaDeviceSynchronize();
        for (int i = 0; i < numModels; i++) {
            cublasSetStream(hnd, streams[i & (numStreams - 1)]);
            gemm(hnd, CUBLAS_OP_N, CUBLAS_OP_T,
                 hNVars[i], hNVars[i], M,
                 alp[i], XA[i], hNVars[i],
                 XA[i], hNVars[i], bet[i],
                 G[i], hNVars[i]);
        }
        cudaDeviceSynchronize();
        for (int i = 0; i < numModels; i++) {
            cublasSetStream(hnd, streams[i & (numStreams - 1)]);
            gemv(hnd, CUBLAS_OP_N,
                 hNVars[i], M,
                 alp[i], XA[i], hNVars[i],
                 y + i * M, 1, bet[i],
                 r + i * M, 1);
        }
        cudaDeviceSynchronize();
        for (int i = 0; i < numModels; i++) {
            cublasSetStream(hnd, streams[i & (numStreams - 1)]);
            getrfBatched(hnd, hNVars[i], dG + i,
                         hNVars[i], pivot + i * M, info + i, 1);
        }
        cudaDeviceSynchronize();
        for (int i = 0; i < numModels; i++) {
            cublasSetStream(hnd, streams[i & (numStreams - 1)]);
            getriBatched(hnd, hNVars[i], dG + i,
                         hNVars[i], pivot + i * M, dI + i,
                         hNVars[i], info + i, 1);
        }
        cudaDeviceSynchronize();
        for (int i = 0; i < numModels; i++) {
            cublasSetStream(hnd, streams[i & (numStreams - 1)]);
            gemv(hnd, CUBLAS_OP_N,
                 hNVars[i], hNVars[i],
                 alp[i], I[i], hNVars[i],
                 r + i * M, 1, bet[i],
                 betaOls + i * M, 1);
        }
        cudaDeviceSynchronize();
        for (int i = 0; i < numModels; i++) {
            cublasSetStream(hnd, streams[i & (numStreams - 1)]);    
            gemv(hnd, CUBLAS_OP_T,
                 hNVars[i], M,
                 alp[i], XA[i], hNVars[i],
                 betaOls + i * M, 1, bet[i],
                 d + i * M, 1);
        }
        cudaDeviceSynchronize();
        mat_sub<precision>(d, mu, d,
                           numModels * M, *(new dim3(bModM)));
        cudaDeviceSynchronize();
        gammat<precision>(gamma, beta, betaOls, r,
                          lVars, nVars, M,
                          N, numModels, *(new dim3(bModM)));
        cudaDeviceSynchronize();
        for (int i = 0; i < numModels; i++) {
            cublasSetStream(hnd, streams[i & (numStreams - 1)]);
            int length = (hNVars[i] > 1)? hNVars[i] - 1: 1;
            amin(hnd,
                 length, gamma + i * M,
                 1, dropidx + i);
        }
        cudaDeviceSynchronize();
        cublasSetStream(hnd, NULL);
        gemm(hnd, CUBLAS_OP_N, CUBLAS_OP_N,
             N, numModels, M,
             alp[0], X, N,
             d, M, bet[0],
             cd, N);
        cudaDeviceSynchronize();
        for (int i = 0; i < numModels; i++) {
            minCd(c + i * N, cd + i * N, cmax + i, r + i, N, streams[i & (numStreams - 1)], *(new dim3(1024)));
        }
        cudaDeviceSynchronize();
        set_gamma<precision>(gamma, r, dropidx,
                             lasso, nVars, maxVariables,
                             M, numModels, *(new dim3(bMod)));
        cudaDeviceSynchronize();
        update<precision>(beta, mu, d, cmax, r,
                          betaOls, gamma, lVars,
                          nVars, M, N,
                          numModels, *(new dim3(bModM)));
        cudaDeviceSynchronize();
        drop(lVars, dropidx, nVars,
             lasso, M, numModels);
        cudaDeviceSynchronize();
        for (int i = 0; i < numModels; i++) {
            norm2(y + i * M, mu + i * M, r + i,
                  M, streams[i & (numStreams - 1)], *(new dim3(512)));
        }
        cudaDeviceSynchronize();
        for (int i = 0; i < numModels; i++) {
            norm1(beta + i * N, cmax + i, N,
                  streams[i & (numStreams - 1)], *(new dim3(1024)));
        }
        cudaDeviceSynchronize();
        final<precision>(a1, a2, cmax, r, step, done, numModels, 0.43, *(new dim3(bMod)));
        cudaDeviceSynchronize();
        timer.stop();
        totalTime += timer.elapsed();
        totalFlop += 4.0 * (double) numModels * (double) N * (double) M;
        totalFlop += 7.0 * (double) numModels * (double) N;
        totalFlop += 6.0 * (double) numModels * (double) M;
        for (int i = 0; i < numModels; i++) {
            totalFlop += 2.0 * (double) hNVars[i] * (double) hNVars[i] * (double) M;
            totalFlop += (2.0 / 3.0) * (double) hNVars[i] * (double) hNVars[i] * (double) hNVars[i];
            totalFlop += 4.0 * (double) hNVars[i] * (double) M + 2.0 * (double) hNVars[i] * (double) hNVars[i];
            totalFlop += 5.0 * (double) hNVars[i];
        }
    }
    printf("\n");

    cublasDestroy(hnd);
    cudaFree(nVars);
    cudaFree(step);
    cudaFree(lasso);
    cudaFree(done);
    cudaFree(lVars);
    cudaFree(cidx);
    cudaFree(act);
    cudaFree(dropidx);
    cudaFree(pivot);
    cudaFree(info);
    cudaFree(ctrl);
    cudaFree(hNVars);
    for (int i = 0; i < numModels; i++) {
        cudaFree(alp[i]);
        cudaFree(bet[i]);
        cudaFree(XA[i]);
        cudaFree(G[i]);
        cudaFree(I[i]);
    }
    cudaFree(y);
    cudaFree(mu);
    cudaFree(r);
    cudaFree(beta);
    cudaFree(c);
    cudaFree(absC);
    cudaFree(cmax);
    cudaFree(betaOls);
    cudaFree(d);
    cudaFree(gamma);
    cudaFree(cd);
    cudaFree(a1);
    cudaFree(a2);
    cudaFree(dG);
    cudaFree(dI);

    // for (int i = 0; i < totalModels; i++) {
    //     printf("Model = %d: a1 = %f: a2 = %f: nVars = %d\n", i, debug[i].a1, debug[i].a2, debug[i].nVars);
    // }
    printf("Gflops = %f\n", (totalFlop * 1.0e-9) / (totalTime * 1.0e-3));
    return 0;
}