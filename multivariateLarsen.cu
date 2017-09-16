#include <bits/stdc++.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "utilities.h"
#include "blas.h"
#include "kernels.h"

typedef float precision;

template<typename T>
void printDeviceVar(T *var, int size, int *ind, int numInd, int breaker = 1) {
	T *hVar = new T[size];
	cudaMemcpy(hVar, var, size * sizeof(T), cudaMemcpyDeviceToHost);
	for (int i = 0; i < numInd; i++) {
		if (i % breaker == 0) printf("\n");
		if (typeid(T).name()[0] == 'i') printf("%6d ", hVar[ind[i]]);
		else printf("%6.3f ", hVar[ind[i]]);
	}
	printf("\n");
}

void range(int *&ind, int st, int ed) {
	ind = new int[ed - st];
	for (int i = st; i < ed; i++) ind[i - st] = i;
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

double flopCounter(int M, int N, int numModels, int *hNVars) {
	double flop = 0;
	// r = y - mu
	flop += (double) M * (double) numModels;
	// c = X' * r
	flop += 2.0 * (double) M * (double) N * (double) numModels;
	for (int i = 0; i < numModels; i++) {
		// G = X(:, A)' * X(:, A)
		flop += 2.0 * (double) hNVars[i] * (double) M * (double) hNVars[i];
		// b_OLS = G\(X(:,A)'*y)
		flop += 2.0 * (double) M * (double) hNVars[i] + 2.0 * (double) hNVars[i] * (double) hNVars[i];
		// Inverse ops for G
		flop += (2.0 / 3.0) * (double) hNVars[i] * (double) hNVars[i] * (double) hNVars[i];
		// d = X(: , A) * b_OLS - mu
		flop += 2.0 * (double) M * (double) hNVars[i] + (double) M;
		// gamma_tilde
		flop += 2.0 * (double) hNVars[i];
		// b update
		flop += 3.0 * (double) hNVars[i];
	}
	// cd = X'*d
	flop += 2.0 * (double) M * (double) N * (double) numModels;
	// gamma
	flop += 6.0 * (double) N * (double) numModels;
	// mu update
	flop += 2.0 * (double) M * (double) numModels;
	// norm1 and norm2
	flop += ((double) N + 3.0 * (double) M) * (double) numModels;
	return flop;
}

int main(int argc, char *argv[]) {
	if (argc < 4) {
		printf("Insufficient parameters, required 3! (flatMriPath, numModels, numStreams)\n");
		return 0;
	}

	// Reading flattened mri image
	precision *X, *Y;
	IntegerTuple tuple = read_flat_mri<precision>(argv[1], X, Y);
	int M = tuple.M, N = tuple.N;
	printf("Read FMRI Data X of shape: (%d,%d)\n", M, N);
	printf("Read FMRI Data Y of shape: (%d,%d)\n", M, N);

	// Number of models to solve in ||l
	int numModels = str_to_int(argv[2]);
	int totalModels = N;
	printf("Total number of models: %d\n", totalModels);
	printf("Number of models in ||l: %d\n", numModels);

	int numStreams = str_to_int(argv[3]);
	numStreams = pow(2, int(log(numStreams) / log(2)));
	printf("Number of streams: %d\n", numStreams);

	// Computimal optimal block sizes
	int bN = optimalBlock1D(N);
	int bM = optimalBlock1D(M);
	int bModM = optimalBlock1D(numModels * M);
	int bMM = optimalBlock1D(M * M);
	int bModN = optimalBlock1D(numModels * N);
	int bMod = optimalBlock1D(numModels);
		
	// Declare all lars variables
	int *nVars, *step, *lasso, *done, *cidx, *act, *dropidx;
	int *pivot, *info, *ctrl;
	int *lVars;
	int *hNVars, *hdone, *hact, *hLasso, *hDropidx;
	int *hctrl;
	precision *cmax, *a1, *a2, *gamma;
	precision *y, *mu, *r, *betaOls, *d, *gamma_tilde, *buf;
	precision *beta, *c, *cd;
	precision *alp[numModels], *bet[numModels];
	precision *XA[numModels], *XA1[numModels], *G[numModels], *I[numModels], **dG, **dI;
	precision *ha1, *ha2;

	// Initialize all lars variables
	init_var<int>(nVars, numModels);
	init_var<int>(step, numModels);
	init_var<int>(lasso, numModels);
	init_var<int>(done, numModels);
	init_var<int>(cidx, numModels);
	init_var<int>(act, numModels);
	init_var<int>(dropidx, numModels);
	
	init_var<int>(pivot, numModels * M);
	init_var<int>(info, numModels);
	init_var<int>(ctrl, 1);
	
	init_var<int>(lVars, numModels * M);
	
	hNVars = new int[numModels];
	hdone = new int[numModels];
	hact = new int[numModels];
	hLasso = new int[numModels];
	hDropidx = new int[numModels];
	
	hctrl = new int[1];
	
	init_var<precision>(cmax, numModels);
	init_var<precision>(a1, numModels);
	init_var<precision>(a2, numModels);
	init_var<precision>(gamma, numModels);

	init_var<precision>(y, numModels * M);
	init_var<precision>(mu, numModels * M);
	init_var<precision>(r, numModels * M);
	init_var<precision>(betaOls, numModels * M);
	init_var<precision>(d, numModels * M);
	init_var<precision>(gamma_tilde, numModels * M);
	init_var<precision>(buf, numModels * 64);
		
	init_var<precision>(beta, numModels * N);
	init_var<precision>(c, numModels * N);
	init_var<precision>(cd, numModels * N);

	ha1 = new precision[numModels];
	ha2 = new precision[numModels];

	int maxVariables = min(M, N - 1);
	int maxSteps = 8 * maxVariables;

	int top = numModels;
	double totalFlop = 0;
	double times[25] = {0};

	cublasHandle_t hnd;
	cublasCreate(&hnd);
	cudaStream_t streams[numStreams];
	cublasSetPointerMode(hnd, CUBLAS_POINTER_MODE_DEVICE);

	for (int i = 0; i < numModels; i++) {
		init_var<precision>(alp[i], 1);
		init_var<precision>(bet[i], 1);
		init_var<precision>(XA[i], M * M);
		init_var<precision>(XA1[i], M * M);
		init_var<precision>(G[i], M * M);
		init_var<precision>(I[i], M * M);
	}
		
	cudaMalloc(&dG, numModels * sizeof(precision *));
	cudaMemcpy(dG, G, numModels * sizeof(precision *), cudaMemcpyHostToDevice);
	cudaMalloc(&dI, numModels * sizeof(precision *));
	cudaMemcpy(dI, I, numModels * sizeof(precision *), cudaMemcpyHostToDevice);

	for (int i = 0; i < numStreams; i++) cudaStreamCreate(&streams[i]);

	for (int i = 0; i < numModels; i++) set_model<precision>(Y, y, mu, beta, alp[i], bet[i], nVars, lasso, step, done, act, M, N, i, i, streams[i & (numStreams - 1)], *(new dim3(bN)));
	cudaDeviceSynchronize();

	Debug<precision> debug[totalModels];
	GpuTimer timer;

	int e = 0;
	int completed = 0;
	printf("\rCompleted %d models", completed);
	while (true) {

		timer.start();
		cudaMemset(ctrl, 0, 1 * sizeof(int));
		check(nVars, step, maxVariables, maxSteps, done, ctrl, numModels);
		cudaMemcpy(hctrl, ctrl, 1 * sizeof(int), cudaMemcpyDeviceToHost);
		if (hctrl[0] == 1) {
			cudaMemcpy(ha1, a1, numModels * sizeof(precision), cudaMemcpyDeviceToHost);
			cudaMemcpy(ha2, a2, numModels * sizeof(precision), cudaMemcpyDeviceToHost);
			cudaMemcpy(hdone, done, numModels * sizeof(int), cudaMemcpyDeviceToHost);
			cudaMemcpy(hNVars, nVars, numModels * sizeof(int), cudaMemcpyDeviceToHost);
			cudaMemcpy(hact, act, numModels * sizeof(int), cudaMemcpyDeviceToHost);
			for (int i = 0; i < numModels; i++) {
				if (hdone[i]) {
					if (hdone[i] == 2) completed++;
					if (top < totalModels) {
						set_model<precision>(Y, y, mu, beta, alp[i], bet[i], nVars, lasso, step, done, act, M, N, i, top++, streams[i & (numStreams - 1)], *(new dim3(bN)));
					}
					if (debug[hact[i]].nVars == -1) {
						debug[hact[i]].nVars = hNVars[i];
						debug[hact[i]].a1 = ha1[i];
						debug[hact[i]].a2 = ha2[i];
					}
				}
			}
		}
		printf("\rCompleted %d models", completed);
		if (completed == totalModels) {
			break;
		}
		timer.stop();
		times[0] += timer.elapsed();

		timer.start();
		mat_sub<precision>(y, mu, r, numModels * M, *(new dim3(bModM)));
		cudaDeviceSynchronize();
		timer.stop();
		times[1] += timer.elapsed();
		
		timer.start();
		cublasSetStream(hnd, NULL);
		gemm(hnd, CUBLAS_OP_N, CUBLAS_OP_N, N, numModels, M, alp[0], X, N, r, M, bet[0], c, N);
		cudaDeviceSynchronize();
		timer.stop();
		times[2] += timer.elapsed();
		
		timer.start();
		exclude<precision>(c, lVars, nVars, act, M, N, numModels, 0, *(new dim3(bModM)));
		cudaDeviceSynchronize();
		timer.stop();
		times[3] += timer.elapsed();
		
		timer.start();
		fabsMaxReduce<precision>(c, cmax, buf, numModels, N);
		cudaDeviceSynchronize();
		timer.stop();
		times[4] += timer.elapsed();
		
		timer.start();
		set_cidx<precision>(cmax, cidx, c, N, numModels, *(new dim3(bModN)));
		cudaDeviceSynchronize();
		timer.stop();
		times[5] += timer.elapsed();
		
		timer.start();
		lasso_add(lasso, lVars, nVars, cidx, M, N, numModels, *(new dim3(bMod)));
		cudaDeviceSynchronize();
		timer.stop();
		times[6] += timer.elapsed();

		cudaMemcpy(hNVars, nVars, numModels * sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(hLasso, lasso, numModels * sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(hDropidx, dropidx, numModels * sizeof(int), cudaMemcpyDeviceToHost);

		timer.start();
		for (int i = 0; i < numModels; i++) {
			gather<precision>(XA[i], XA1[i], X, lVars, hNVars[i], hLasso[i], hDropidx[i] - 1, M, N, i, streams[i & (numStreams - 1)]);
		}
		cudaDeviceSynchronize();
		timer.stop();
		times[7] += timer.elapsed();

		timer.start();
		for (int i = 0; i < numModels; i++) {
			cublasSetStream(hnd, streams[i & (numStreams - 1)]);
			gemm(hnd, CUBLAS_OP_T, CUBLAS_OP_N, hNVars[i], hNVars[i], M, alp[i], XA[i], M, XA[i], M, bet[i], G[i], hNVars[i]);
		}
		cudaDeviceSynchronize();
		timer.stop();
		times[8] += timer.elapsed();
		
		timer.start();
		for (int i = 0; i < numModels; i++) {
			cublasSetStream(hnd, streams[i & (numStreams - 1)]);
			gemv(hnd, CUBLAS_OP_T, M, hNVars[i], alp[i], XA[i], M, y + i * M, 1, bet[i], r + i * M, 1);
		}
		cudaDeviceSynchronize();
		timer.stop();
		times[9] += timer.elapsed();
		
		timer.start();
		for (int i = 0; i < numModels; i++) {
			cublasSetStream(hnd, streams[i & (numStreams - 1)]);
			getrfBatched(hnd, hNVars[i], dG + i, hNVars[i], pivot + i * M, info + i, 1);
		}
		cudaDeviceSynchronize();
		timer.stop();
		times[10] += timer.elapsed();
		
		timer.start();
		for (int i = 0; i < numModels; i++) {
			cublasSetStream(hnd, streams[i & (numStreams - 1)]);
			getriBatched(hnd, hNVars[i], dG + i, hNVars[i], pivot + i * M, dI + i, hNVars[i], info + i, 1);
		}
		cudaDeviceSynchronize();
		timer.stop();
		times[11] += timer.elapsed();
		
		timer.start();
		for (int i = 0; i < numModels; i++) {
			cublasSetStream(hnd, streams[i & (numStreams - 1)]);
			gemv(hnd, CUBLAS_OP_N, hNVars[i], hNVars[i], alp[i], I[i], hNVars[i], r + i * M, 1, bet[i], betaOls + i * M, 1);
		}
		cudaDeviceSynchronize();
		timer.stop();
		times[12] += timer.elapsed();
		
		timer.start();
		for (int i = 0; i < numModels; i++) {
			cublasSetStream(hnd, streams[i & (numStreams - 1)]);    
			gemv(hnd, CUBLAS_OP_N, M, hNVars[i], alp[i], XA[i], M, betaOls + i * M, 1, bet[i], d + i * M, 1);
		}
		cudaDeviceSynchronize();
		timer.stop();
		times[13] += timer.elapsed();
		
		timer.start();
		mat_sub<precision>(d, mu, d, numModels * M, *(new dim3(bModM)));
		cudaDeviceSynchronize();
		timer.stop();
		times[14] += timer.elapsed();
		
		timer.start();
		gammat<precision>(gamma_tilde, beta, betaOls, lVars, nVars, lasso, M, N, numModels, *(new dim3(bModM)));
		cudaDeviceSynchronize();
		timer.stop();
		times[15] += timer.elapsed();
		
		timer.start();
		for (int i = 0; i < numModels; i++) {
			cublasSetStream(hnd, streams[i & (numStreams - 1)]);
			int length = (hNVars[i] > 1)? hNVars[i] - 1: 1;
			amin(hnd, length, gamma_tilde + i * M, 1, dropidx + i);
		}
		cudaDeviceSynchronize();
		timer.stop();
		times[16] += timer.elapsed();
		
		timer.start();
		cublasSetStream(hnd, NULL);
		gemm(hnd, CUBLAS_OP_N, CUBLAS_OP_N, N, numModels, M, alp[0], X, N, d, M, bet[0], cd, N);
		cudaDeviceSynchronize();
		timer.stop();
		times[17] += timer.elapsed();
		
		timer.start();
		cdMinReduce<precision>(c, cd, cmax, r, buf, numModels, N);
		cudaDeviceSynchronize();
		timer.stop();
		times[18] += timer.elapsed();
		
		timer.start();
		set_gamma<precision>(gamma, gamma_tilde, r, dropidx, lasso, nVars, maxVariables, M, numModels, *(new dim3(bMod)));
		cudaDeviceSynchronize();
		timer.stop();
		times[19] += timer.elapsed();
		
		timer.start();
		update<precision>(beta, mu, d, betaOls, gamma, lVars, nVars, M, N, numModels, *(new dim3(bModM)));
		cudaDeviceSynchronize();
		timer.stop();
		times[20] += timer.elapsed();

		timer.start();
		drop(lVars, dropidx, nVars, lasso, M, numModels);
		cudaDeviceSynchronize();
		timer.stop();
		times[21] += timer.elapsed();

		timer.start();
		sqrAddReduce<precision>(y, mu, r, buf, numModels, M);
		cudaDeviceSynchronize();
		timer.stop();
		times[22] += timer.elapsed();
		
		timer.start();
		fabsAddReduce<precision>(beta, cmax, buf, numModels, N);
		cudaDeviceSynchronize();
		timer.stop();
		times[23] += timer.elapsed();

		timer.start();
		final<precision>(a1, a2, cmax, r, step, done, numModels, 0.43, *(new dim3(bMod)));
		cudaDeviceSynchronize();
		timer.stop();

		times[24] += timer.elapsed();
		
		totalFlop += flopCounter(M, N, numModels, hNVars);
		e++;
	}
	
	// Statistics
	double transferTime = times[0];
	double execTime = 0;
	for (int i = 1; i < 25; i++) execTime += times[i];
	printf("\n");

	// for (int i = 0; i < numModels; i++) {
	//     printf("Model = %d: a1 = %f: a2 = %f: nVars = %d\n", i, debug[i].a1, debug[i].a2, debug[i].nVars);
	// }
	
	for (int i = 0; i < 25; i++) {
		printf("Kernel %2d time = %10.4f\n", i, times[i]);
	}
	printf("Execution time = %f\n", execTime * 1.0e-3);
	printf("Transfer time = %f\n", transferTime * 1.0e-3);
	printf("Total Gflop count = %f\n", totalFlop * 1.0e-9);
	printf("Execution Gflops = %f\n", (totalFlop * 1.0e-9) / (execTime * 1.0e-3));

	cudaFree(nVars);
	cudaFree(step);
	cudaFree(lasso);
	cudaFree(done);
	cudaFree(cidx);
	cudaFree(act);
	cudaFree(dropidx);
	
	cudaFree(pivot);
	cudaFree(info);
	cudaFree(ctrl);
	
	cudaFree(lVars);
	
	cudaFree(cmax);
	cudaFree(a1);
	cudaFree(a2);
	cudaFree(gamma);

	cudaFree(y);
	cudaFree(mu);
	cudaFree(r);
	cudaFree(betaOls);
	cudaFree(d);
	cudaFree(gamma_tilde);
	cudaFree(buf);
	
	cudaFree(beta);
	cudaFree(c);
	cudaFree(cd);
	
	cublasDestroy(hnd);
	for (int i = 0; i < numModels; i++) {
		cudaFree(alp[i]);
		cudaFree(bet[i]);
		cudaFree(XA[i]);
		cudaFree(XA1[i]);
		cudaFree(G[i]);
		cudaFree(I[i]);
	}

	for (int i = 0; i < numStreams; i++) cudaStreamDestroy(streams[i]);
		
	cudaFree(dG);
	cudaFree(dI);

	return 0;
}
