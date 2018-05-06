#include <bits/stdc++.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "utilities.h"
#include "blas.h"
#include "kernels.h"

typedef float precision;

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
	if (argc < 5) {
		printf("Insufficient parameters, required 4! (flatMriPath, numModels, numStreams, g)\n");
		return 0;
	}

	// Reading flattened mri image
	precision *X, *Y;
	IntegerTuple tuple = read_flat_mri<precision>(argv[1], X, Y);
	int M = tuple.M, N = tuple.N;
	printf("Read FMRI Data X of shape: (%d,%d)\n", M, N);
	printf("Read FMRI Data Y of shape: (%d,%d)\n", M, N);

	// Number of models to solve in ||l
	int numModels = atoi(argv[2]);
	int totalModels = N;
	printf("Total number of models: %d\n", totalModels);
	printf("Number of models in ||l: %d\n", numModels);

	int numStreams = atoi(argv[3]);
	numStreams = pow(2, int(log(numStreams) / log(2)));
	printf("Number of streams: %d\n", numStreams);

	precision g = atof(argv[4]);
	printf("Lambda: %f\n", g);

	// Computimal optimal block sizes
	int bN = optimalBlock1D(N);
	int bM = optimalBlock1D(M);
	int bModM = optimalBlock1D(numModels * M);
	int bMM = optimalBlock1D(M * M);
	int bModN = optimalBlock1D(numModels * N);
	int bMod = optimalBlock1D(numModels);
		
	// Declare all lars variables
	int *nVars, *step, *lasso, *done, *cidx, *act, *dropidx;
	int *pivot, *info, *intBuf;
	int *lVars;
	int *hNVars, *hStep, *hdone, *hact, *hLasso, *hDropidx;
	precision *cmax, *a1, *a2, *gamma;
	precision *y, *mu, *r, *betaOls, *d, *gamma_tilde, *buf;
	precision *beta, *c, *cd;
	precision alp = 1, bet = 0;
	precision *XA[numModels], *XA1[numModels], *G[numModels], *I[numModels], **dXA, **dG, **dI;
	precision *ha1, *ha2;

	// Initialize all lars variables
	init_var<int>(nVars, numModels);
	init_var<int>(step, numModels);
	init_var<int>(lasso, numModels);
	init_var<int>(done, numModels);
	init_var<int>(cidx, numModels);
	init_var<int>(act, numModels);
	init_var<int>(dropidx, numModels);
	
	init_var<int>(pivot, M * numModels * M);
	init_var<int>(info, M * numModels);
	init_var<int>(intBuf, numModels * 128);
	
	init_var<int>(lVars, numModels * M);
	
	hNVars = new int[numModels];
	hStep = new int[numModels];
	hdone = new int[numModels];
	hact = new int[numModels];
	hLasso = new int[numModels];
	hDropidx = new int[numModels];
	
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
	init_var<precision>(buf, numModels * 128);
		
	init_var<precision>(beta, numModels * N);
	init_var<precision>(c, numModels * N);
	init_var<precision>(cd, numModels * N);

	ha1 = new precision[numModels];
	ha2 = new precision[numModels];

	int maxVariables = min(M, N - 1);
	int maxSteps = 8 * maxVariables;

	int top = numModels;
	double totalFlop = 0;
	double times[24] = {0};

	cublasHandle_t hnd;
	cublasCreate(&hnd);
	cudaStream_t streams[numStreams];

	for (int i = 0; i < numModels; i++) {
		init_var<precision>(XA[i], M * M);
		init_var<precision>(XA1[i], M * M);
		init_var<precision>(G[i], M * M);
		init_var<precision>(I[i], M * M);
	}
	
	cudaMalloc(&dXA, numModels * sizeof(precision *));
	cudaMemcpy(dXA, XA, numModels * sizeof(precision *), cudaMemcpyHostToDevice);	
	cudaMalloc(&dG, numModels * sizeof(precision *));
	cudaMemcpy(dG, G, numModels * sizeof(precision *), cudaMemcpyHostToDevice);
	cudaMalloc(&dI, numModels * sizeof(precision *));
	cudaMemcpy(dI, I, numModels * sizeof(precision *), cudaMemcpyHostToDevice);

	for (int i = 0; i < numStreams; i++) cudaStreamCreate(&streams[i]);

	for (int i = 0; i < numModels; i++) set_model<precision>(Y, y, mu, beta, nVars, lasso, step, done, act, M, N, i, i, streams[i & (numStreams - 1)], *(new dim3(bN)));
	cudaDeviceSynchronize();

	GpuTimer timer;
	std::ofstream stepf("step.csv"), nvarsf("nvars.csv"), a1f("a1.csv"), a2f("a2.csv"), betaf("beta.csv");

	precision **batchXA[maxVariables], **batchG[maxVariables], **batchI[maxVariables], **dBatchXA[maxVariables], **dBatchG[maxVariables], **dBatchI[maxVariables];
	for (int i = 0; i < maxVariables; i++) {
		batchXA[i] = new precision *[numModels];
		batchG[i] = new precision *[numModels];
		batchI[i] = new precision *[numModels];
		cudaMalloc(&dBatchXA[i], numModels * sizeof(precision *));
		cudaMalloc(&dBatchG[i], numModels * sizeof(precision *));
		cudaMalloc(&dBatchI[i], numModels * sizeof(precision *));
	}
	int batchLen[maxVariables];

	int e = 0;
	int completed = 0;
	printf("\rCompleted %d models", completed);
	while (true) {

		timer.start();
		check(nVars, step, maxVariables, maxSteps, done, numModels);
		int ctrl = 0;
		cudaMemcpy(hdone, done, numModels * sizeof(int), cudaMemcpyDeviceToHost);
		for (int i = 0; i < numModels; i++) {
			if (hdone[i]) {
				ctrl = 1;
				break;
			}
		}
		if (ctrl) {
			cudaMemcpy(ha1, a1, numModels * sizeof(precision), cudaMemcpyDeviceToHost);
			cudaMemcpy(ha2, a2, numModels * sizeof(precision), cudaMemcpyDeviceToHost);
			cudaMemcpy(hStep, step, numModels * sizeof(int), cudaMemcpyDeviceToHost);
			cudaMemcpy(hNVars, nVars, numModels * sizeof(int), cudaMemcpyDeviceToHost);
			cudaMemcpy(hact, act, numModels * sizeof(int), cudaMemcpyDeviceToHost);

			for (int i = 0, s = 0; i < numModels; i++) {
				if (hdone[i] == 2) {
					compress<precision>(beta, r, lVars, hNVars[i], i, M, N, streams[s & (numStreams - 1)]);
					s++;
				}
			}
			cudaDeviceSynchronize();

			for (int i = 0; i < numModels; i++) {
				if (hdone[i] == 2) {
					completed++;
					stepf << hact[i] << ", " << hStep[i] << "\n";
					nvarsf << hact[i] << ", " << hNVars[i] << "\n";
					a1f << hact[i] << ", " << ha1[i] << "\n";
					a2f << hact[i] << ", " << ha2[i] << "\n";
					int hlVars[hNVars[i]];
					precision hbeta[hNVars[i]];
					cudaMemcpy(hlVars, lVars + i * M, hNVars[i] * sizeof(int), cudaMemcpyDeviceToHost);
					cudaMemcpy(hbeta, r + i * M, hNVars[i] * sizeof(precision), cudaMemcpyDeviceToHost);
					for (int j = 0; j < hNVars[i]; j++) betaf << hact[i] << ", " << hlVars[j] << ", " << hbeta[j] << "\n";
				}
			}

			for (int i = 0, s = 0; i < numModels; i++) {
				if (hdone[i]) {
					if (top < totalModels) {
						set_model<precision>(Y, y, mu, beta, nVars, lasso, step, done, act, M, N, i, top++, streams[s & (numStreams - 1)], *(new dim3(bN)));
						s++;
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
		gemm(hnd, CUBLAS_OP_N, CUBLAS_OP_N, N, numModels, M, &alp, X, N, r, M, &bet, c, N);
		cudaDeviceSynchronize();
		timer.stop();
		times[2] += timer.elapsed();
		
		timer.start();
		exclude<precision>(c, lVars, nVars, act, M, N, numModels, 0, *(new dim3(bModM)));
		cudaDeviceSynchronize();
		timer.stop();
		times[3] += timer.elapsed();
		
		timer.start();
		fabsMaxReduce<precision>(c, cmax, buf, cidx, intBuf, numModels, N);
		cudaDeviceSynchronize();
		timer.stop();
		times[4] += timer.elapsed();

		timer.start();
		lasso_add(lasso, lVars, nVars, cidx, M, N, numModels, *(new dim3(bMod)));
		cudaDeviceSynchronize();
		timer.stop();
		times[5] += timer.elapsed();

		timer.start();
		cudaMemcpy(hNVars, nVars, numModels * sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(hLasso, lasso, numModels * sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(hDropidx, dropidx, numModels * sizeof(int), cudaMemcpyDeviceToHost);
		int maxVar = hNVars[0];
		for (int i = 0; i < maxVariables; i++) batchLen[i] = 0;
		for (int i = 0; i < numModels; i++) {
			if (hNVars[i] > maxVar) maxVar = hNVars[i];
			batchXA[hNVars[i]][batchLen[hNVars[i]]] = XA[i];
			batchG[hNVars[i]][batchLen[hNVars[i]]] = G[i];
			batchI[hNVars[i]][batchLen[hNVars[i]]] = I[i];
			batchLen[hNVars[i]]++;
		}
		for (int i = 0; i < maxVariables; i++) {
			if (batchLen[i] > 0) {
				cudaMemcpy(dBatchXA[i], batchXA[i], batchLen[i] * sizeof(precision *), cudaMemcpyHostToDevice);
				cudaMemcpy(dBatchG[i], batchG[i], batchLen[i] * sizeof(precision *), cudaMemcpyHostToDevice);
				cudaMemcpy(dBatchI[i], batchI[i], batchLen[i] * sizeof(precision *), cudaMemcpyHostToDevice);
			}
		}
		cudaDeviceSynchronize();
		timer.stop();
		times[0] += timer.elapsed();

		timer.start();
		for (int i = 0; i < numModels; i++) {
			gather<precision>(XA[i], XA1[i], X, lVars, hNVars[i], hLasso[i], hDropidx[i] - 1, M, N, i, streams[i & (numStreams - 1)]);
		}
		cudaDeviceSynchronize();
		timer.stop();
		times[6] += timer.elapsed();

		timer.start();
		for (int i = 0, s = 0; i < maxVariables; i++) {
			if (batchLen[i] > 0) {
				cublasSetStream(hnd, streams[s & (numStreams - 1)]);
				gemmBatched(hnd, CUBLAS_OP_T, CUBLAS_OP_N, i, i, M, &alp, dBatchXA[i], M, dBatchXA[i], M, &bet, dBatchG[i], i, batchLen[i]);
				s++;
			}
		}
		cudaDeviceSynchronize();
		timer.stop();
		times[7] += timer.elapsed();
		
		timer.start();
		XAyBatched<precision>(dXA, y, r, nVars, M, numModels);
		cudaDeviceSynchronize();
		timer.stop();
		times[8] += timer.elapsed();
		
		timer.start();
		for (int i = 0, s = 0; i < maxVariables; i++) {
			if (batchLen[i] > 0) {
				cublasSetStream(hnd, streams[s & (numStreams - 1)]);
				getrfBatched(hnd, i, dBatchG[i], i, pivot + i * numModels * M, info + i * numModels, batchLen[i]);
				s++;
			}
		}
		cudaDeviceSynchronize();
		timer.stop();
		times[9] += timer.elapsed();
		
		timer.start();
		for (int i = 0, s = 0; i < maxVariables; i++) {
			if (batchLen[i] > 0) {
				cublasSetStream(hnd, streams[s & (numStreams - 1)]);
				getriBatched(hnd, i, dBatchG[i], i, pivot + i * numModels * M, dBatchI[i], i, info + i * numModels, batchLen[i]);
				s++;
			}
		}
		cudaDeviceSynchronize();
		timer.stop();
		times[10] += timer.elapsed();
		
		timer.start();
		IrBatched<precision>(dI, r, betaOls, nVars, M, numModels, maxVar);
		cudaDeviceSynchronize();
		timer.stop();
		times[11] += timer.elapsed();
		
		timer.start();
		XAbetaOlsBatched<precision>(dXA, betaOls, d, nVars, M, numModels, maxVar);
		cudaDeviceSynchronize();
		timer.stop();
		times[12] += timer.elapsed();
		
		timer.start();
		mat_sub<precision>(d, mu, d, numModels * M, *(new dim3(bModM)));
		cudaDeviceSynchronize();
		timer.stop();
		times[13] += timer.elapsed();
		
		timer.start();
		gammat<precision>(gamma_tilde, beta, betaOls, lVars, nVars, lasso, M, N, numModels, *(new dim3(bModM)));
		cudaDeviceSynchronize();
		timer.stop();
		times[14] += timer.elapsed();
		
		timer.start();
		minGamma<precision>(gamma_tilde, dropidx, nVars, numModels, M, *(new dim3(bMod)));
		cudaDeviceSynchronize();
		timer.stop();
		times[15] += timer.elapsed();
		
		timer.start();
		cublasSetStream(hnd, NULL);
		gemm(hnd, CUBLAS_OP_N, CUBLAS_OP_N, N, numModels, M, &alp, X, N, d, M, &bet, cd, N);
		cudaDeviceSynchronize();
		timer.stop();
		times[16] += timer.elapsed();
		
		timer.start();
		cdMinReduce<precision>(c, cd, cmax, r, buf, numModels, N);
		cudaDeviceSynchronize();
		timer.stop();
		times[17] += timer.elapsed();
		
		timer.start();
		set_gamma<precision>(gamma, gamma_tilde, r, dropidx, lasso, nVars, maxVariables, M, numModels, *(new dim3(bMod)));
		cudaDeviceSynchronize();
		timer.stop();
		times[18] += timer.elapsed();
		
		timer.start();
		update<precision>(beta, mu, d, betaOls, gamma, lVars, nVars, M, N, numModels, *(new dim3(bModM)));
		cudaDeviceSynchronize();
		timer.stop();
		times[19] += timer.elapsed();

		timer.start();
		drop(lVars, dropidx, nVars, lasso, M, numModels);
		cudaDeviceSynchronize();
		timer.stop();
		times[20] += timer.elapsed();

		timer.start();
		sqrAddReduce<precision>(y, mu, r, buf, numModels, M);
		cudaDeviceSynchronize();
		timer.stop();
		times[21] += timer.elapsed();
		
		timer.start();
		fabsAddReduce<precision>(beta, cmax, buf, numModels, N);
		cudaDeviceSynchronize();
		timer.stop();
		times[22] += timer.elapsed();

		timer.start();
		final<precision>(a1, a2, cmax, r, step, done, numModels, g, *(new dim3(bMod)));
		cudaDeviceSynchronize();
		timer.stop();
		times[23] += timer.elapsed();
		
		totalFlop += flopCounter(M, N, numModels, hNVars);
		e++;
	}

	stepf.close();
	nvarsf.close();
	a1f.close();
	a2f.close();
	betaf.close();

	// Statistics
	double transferTime = times[0];
	double execTime = 0;
	for (int i = 1; i < 24; i++) execTime += times[i];
	printf("\n");

	for (int i = 0; i < 24; i++) {
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
	cudaFree(intBuf);
	
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
		cudaFree(XA[i]);
		cudaFree(XA1[i]);
		cudaFree(G[i]);
		cudaFree(I[i]);
	}

	for (int i = 0; i < maxVariables; i++) {
		cudaFree(dBatchXA[i]);
		cudaFree(dBatchG[i]);
		cudaFree(dBatchI[i]);
	}

	for (int i = 0; i < numStreams; i++) cudaStreamDestroy(streams[i]);
		
	cudaFree(dXA);
	cudaFree(dG);
	cudaFree(dI);

	return 0;
}
