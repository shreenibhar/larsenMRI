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
	// abs(c)
	flop += (double) N * (double) numModels;
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
		// norm1
		flop += 2.0 * (double) hNVars[i];
	}
	// cd = X'*d
	flop += 2.0 * (double) M * (double) N * (double) numModels;
	// gamma
	flop += 6.0 * (double) N * (double) numModels;
	// mu update
	flop += 2.0 * (double) M * (double) numModels;
	// norm2
	flop += 3.0 * (double) M * (double) numModels;
	// norm2 sqrt, step, sb, err
	flop += 4.0 * (double) numModels;
	// G
	flop += 3.0 * (double) M * (double) numModels;
	return flop;
}

int main(int argc, char *argv[]) {
	if (argc < 9) {
		printf("Insufficient parameters, required 8! (flatMriPath, numModels, numStreams, max l1, min l2, min g, max vars, max steps)\nInput 0 for a parameter to use it's default value!\n");
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
	numModels = (numModels == 0)? 512: numModels;
	int totalModels = N;
	printf("Total number of models: %d\n", totalModels);
	printf("Number of models in ||l: %d\n", numModels);

	int numStreams = atoi(argv[3]);
	numStreams = (numStreams == 0)? 8: numStreams;
	numStreams = pow(2, int(log(numStreams) / log(2)));
	printf("Number of streams: %d\n", numStreams);

	precision l1 = atof(argv[4]);
	l1 = (l1 == 0)? 1000: l1;
	printf("Max L1: %f\n", l1);

	precision l2 = atof(argv[5]);
	printf("Min L2: %f\n", l2);

	precision g = atof(argv[6]);
	g = (g == 0)? 0.1: g;
	printf("Lambda: %f\n", g);

	int maxVariables = atoi(argv[7]);
	maxVariables = (maxVariables == 0)? N - 1: maxVariables;
	maxVariables = min(min(M, N - 1), maxVariables);
	printf("Max Variables: %d\n", maxVariables);

	int maxSteps = atoi(argv[8]);
	maxSteps = (maxSteps == 0)? 8 * maxVariables: maxSteps;
	maxSteps = min(8 * maxVariables, maxSteps);
	printf("Max Steps: %d\n", maxSteps);

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
	precision *cmax, *a1, *a2, *lambda, *gamma;
	precision *y, *mu, *r, *betaOls, *d, *gamma_tilde, *buf;
	precision *beta, *c, *cd, *beta_prev;
	precision alp = 1, bet = 0;
	precision *XA[numModels], *XA1[numModels], *G[numModels], *I[numModels], **dXA, **dG, **dI;
	precision *ha1, *ha2, *hlambda;
	double corr_alp = 1, corr_bet = 0;
	double *corr_beta, *corr_sb, *corr_y, *corr_tmp, *corr_betaols, *corr_yh, *corr_z;
	double *corr_XA[numModels], *corr_G[numModels], *corr_I[numModels], **corr_dXA, **corr_dG, **corr_dI;

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
	init_var<precision>(lambda, numModels);
	init_var<precision>(gamma, numModels);

	init_var<precision>(y, numModels * M);
	init_var<precision>(mu, numModels * M);
	init_var<precision>(r, numModels * M);
	init_var<precision>(betaOls, numModels * M);
	init_var<precision>(d, numModels * M);
	init_var<precision>(gamma_tilde, numModels);
	init_var<precision>(buf, numModels * 128);
		
	init_var<precision>(beta, numModels * N);
	init_var<precision>(c, numModels * N);
	init_var<precision>(cd, numModels * N);
	init_var<precision>(beta_prev, numModels * N);

	init_var<double>(corr_beta, numModels * M);
	init_var<double>(corr_sb, numModels * M);
	init_var<double>(corr_y, numModels * M);
	init_var<double>(corr_tmp, numModels * M);
	init_var<double>(corr_betaols, numModels * M);
	init_var<double>(corr_yh, numModels * M);
	init_var<double>(corr_z, numModels * M);

	ha1 = new precision[numModels];
	ha2 = new precision[numModels];
	hlambda = new precision[numModels];

	for (int i = 0; i < numModels; i++) {
		init_var<precision>(XA[i], M * M);
		init_var<precision>(XA1[i], M * M);
		init_var<precision>(G[i], M * M);
		init_var<precision>(I[i], M * M);

		init_var<double>(corr_XA[i], M * M);
		init_var<double>(corr_G[i], M * M);
		init_var<double>(corr_I[i], M * M);
	}
	
	cudaMalloc(&dXA, numModels * sizeof(precision *));
	cudaMemcpy(dXA, XA, numModels * sizeof(precision *), cudaMemcpyHostToDevice);	
	cudaMalloc(&dG, numModels * sizeof(precision *));
	cudaMemcpy(dG, G, numModels * sizeof(precision *), cudaMemcpyHostToDevice);
	cudaMalloc(&dI, numModels * sizeof(precision *));
	cudaMemcpy(dI, I, numModels * sizeof(precision *), cudaMemcpyHostToDevice);

	cudaMalloc(&corr_dXA, numModels * sizeof(double *));
	cudaMemcpy(corr_dXA, corr_XA, numModels * sizeof(double *), cudaMemcpyHostToDevice);
	cudaMalloc(&corr_dG, numModels * sizeof(double *));
	cudaMemcpy(corr_dG, corr_G, numModels * sizeof(double *), cudaMemcpyHostToDevice);
	cudaMalloc(&corr_dI, numModels * sizeof(double *));
	cudaMemcpy(corr_dI, corr_I, numModels * sizeof(double *), cudaMemcpyHostToDevice);

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

	cublasHandle_t hnd;
	cublasCreate(&hnd);
	cudaStream_t streams[numStreams];
	for (int i = 0; i < numStreams; i++) cudaStreamCreate(&streams[i]);

	for (int i = 0; i < numModels; i++) set_model<precision>(Y, y, mu, beta, a1, a2, lambda, nVars, lasso, step, done, act, M, N, i, i, streams[i & (numStreams - 1)], *(new dim3(bN)));
	cudaDeviceSynchronize();

	GpuTimer timer;
	std::ofstream stepf("step.csv"), nvarsf("nvars.csv"), a1f("a1.csv"), a2f("a2.csv"), lambdaf("G.csv"), betaf("beta.csv");

	int top = numModels;
	double totalFlop = 0;
	double times[25] = {0};
	int e = 0;
	int completed_count = 0;
	std::map<int, int> completed;
	while (true) {
		int t = 0;

		timer.start();
		check(nVars, step, a1, a2, lambda, maxVariables, maxSteps, l1, l2, g, done, numModels);
		int ctrl = 0;
		cudaMemcpy(hdone, done, numModels * sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(hact, act, numModels * sizeof(int), cudaMemcpyDeviceToHost);
		for (int i = 0; i < numModels; i++) {
			if (hdone[i] && !completed[hact[i]]) {
				ctrl = 1;
				break;
			}
		}

		if (ctrl) {
			cudaMemcpy(hStep, step, numModels * sizeof(int), cudaMemcpyDeviceToHost);
			cudaMemcpy(hNVars, nVars, numModels * sizeof(int), cudaMemcpyDeviceToHost);

			for (int i = 0, s = 0; i < numModels; i++) {
				if (hdone[i] && !completed[hact[i]]) {
					copyUp<precision>(corr_XA[i], XA[i], hNVars[i] * M, streams[s & (numStreams - 1)]);
					s++;
				}
			}
			cudaDeviceSynchronize();

			for (int i = 0, s = 0; i < numModels; i++) {
				if (hdone[i] && !completed[hact[i]]) {
					copyUp<precision>(corr_y + i * M, y + i * M, M, streams[s & (numStreams - 1)]);
					s++;
				}
			}
			cudaDeviceSynchronize();

			for (int i = 0, s = 0; i < numModels; i++) {
				if (hdone[i] && !completed[hact[i]]) {
					computeSign<precision>(corr_sb + i * M, beta + i * N, beta_prev + i * N, lVars + i * M, hNVars[i], streams[s & (numStreams - 1)]);
					s++;
				}
			}
			cudaDeviceSynchronize();

			for (int i = 0, s = 0; i < numModels; i++) {
				if (hdone[i] && !completed[hact[i]]) {
					cublasSetStream(hnd, streams[s & (numStreams - 1)]);
					gemm(hnd, CUBLAS_OP_T, CUBLAS_OP_N, hNVars[i], hNVars[i], M, &corr_alp, corr_XA[i], M, corr_XA[i], M, &corr_bet, corr_G[i], hNVars[i]);
					s++;
				}
			}
			cudaDeviceSynchronize();

			for (int i = 0, s = 0; i < numModels; i++) {
				if (hdone[i] && !completed[hact[i]]) {
					cublasSetStream(hnd, streams[s & (numStreams - 1)]);
					getrfBatched(hnd, hNVars[i], corr_dG + i, hNVars[i], pivot + i * M, info + i, 1);
					s++;
				}
			}
			cudaDeviceSynchronize();

			for (int i = 0, s = 0; i < numModels; i++) {
				if (hdone[i] && !completed[hact[i]]) {
					cublasSetStream(hnd, streams[s & (numStreams - 1)]);
					getriBatched(hnd, hNVars[i], corr_dG + i, hNVars[i], pivot + i * M, corr_dI + i, hNVars[i], info + i, 1);
					s++;
				}
			}
			cudaDeviceSynchronize();

			for (int i = 0, s = 0; i < numModels; i++) {
				if (hdone[i] && !completed[hact[i]]) {
					cublasSetStream(hnd, streams[s & (numStreams - 1)]);
					gemv(hnd, CUBLAS_OP_T, M, hNVars[i], &corr_alp, corr_XA[i], M, corr_y + i * M, 1, &corr_bet, corr_tmp + i * M, 1);
					s++;
				}
			}
			cudaDeviceSynchronize();

			for (int i = 0, s = 0; i < numModels; i++) {
				if (hdone[i] && !completed[hact[i]]) {
					cublasSetStream(hnd, streams[s & (numStreams - 1)]);
					gemv(hnd, CUBLAS_OP_N, hNVars[i], hNVars[i], &corr_alp, corr_I[i], hNVars[i], corr_tmp + i * M, 1, &corr_bet, corr_betaols + i * M, 1);
					s++;
				}
			}
			cudaDeviceSynchronize();

			for (int i = 0, s = 0; i < numModels; i++) {
				if (hdone[i] && !completed[hact[i]]) {
					cublasSetStream(hnd, streams[s & (numStreams - 1)]);
					gemv(hnd, CUBLAS_OP_N, M, hNVars[i], &corr_alp, corr_XA[i], M, corr_betaols + i * M, 1, &corr_bet, corr_yh + i * M, 1);
					s++;
				}
			}
			cudaDeviceSynchronize();

			for (int i = 0, s = 0; i < numModels; i++) {
				if (hdone[i] && !completed[hact[i]]) {
					cublasSetStream(hnd, streams[s & (numStreams - 1)]);
					gemv(hnd, CUBLAS_OP_N, hNVars[i], hNVars[i], &corr_alp, corr_I[i], hNVars[i], corr_sb + i * M, 1, &corr_bet, corr_tmp + i * M, 1);
					s++;
				}
			}
			cudaDeviceSynchronize();

			for (int i = 0, s = 0; i < numModels; i++) {
				if (hdone[i] && !completed[hact[i]]) {
					cublasSetStream(hnd, streams[s & (numStreams - 1)]);
					gemv(hnd, CUBLAS_OP_N, M, hNVars[i], &corr_alp, corr_XA[i], M, corr_tmp + i * M, 1, &corr_bet, corr_z + i * M, 1);
					s++;
				}
			}
			cudaDeviceSynchronize();

			for (int i = 0, s = 0; i < numModels; i++) {
				if (hdone[i] && !completed[hact[i]]) {
					correct<precision>(corr_beta + i * M, corr_betaols + i * M, corr_tmp + i * M, corr_y + i * M, corr_yh + i * M, corr_z + i * M, a1 + i, a2 + i, lambda + i, l2, g, hNVars[i], M, streams[s & (numStreams - 1)]);
					s++;
				}
			}
			cudaDeviceSynchronize();

			cudaMemcpy(ha1, a1, numModels * sizeof(precision), cudaMemcpyDeviceToHost);
			cudaMemcpy(ha2, a2, numModels * sizeof(precision), cudaMemcpyDeviceToHost);
			cudaMemcpy(hlambda, lambda, numModels * sizeof(precision), cudaMemcpyDeviceToHost);

			for (int i = 0; i < numModels; i++) {
				if (hdone[i] && !completed[hact[i]]) {
					completed[hact[i]] = 1;
					completed_count++;
					stepf << hact[i] << ", " << hStep[i] << "\n";
					nvarsf << hact[i] << ", " << hNVars[i] << "\n";
					a1f << hact[i] << ", " << ha1[i] << "\n";
					a2f << hact[i] << ", " << ha2[i] << "\n";
					lambdaf << hact[i] << ", " << hlambda[i] << "\n";
					int hlVars[hNVars[i]];
					double hbeta[hNVars[i]];
					cudaMemcpy(hlVars, lVars + i * M, hNVars[i] * sizeof(int), cudaMemcpyDeviceToHost);
					cudaMemcpy(hbeta, corr_beta + i * M, hNVars[i] * sizeof(double), cudaMemcpyDeviceToHost);
					for (int j = 0; j < hNVars[i]; j++) betaf << hact[i] << ", " << hlVars[j] << ", " << hbeta[j] << "\n";
				}
			}

			for (int i = 0, s = 0; i < numModels && top < totalModels; i++) {
				if (hdone[i] && completed[hact[i]]) {
					set_model<precision>(Y, y, mu, beta, a1, a2, lambda, nVars, lasso, step, done, act, M, N, i, top++, streams[s & (numStreams - 1)], *(new dim3(bN)));
					s++;
					hdone[i] = 0;
				}
			}
			cudaDeviceSynchronize();
		}
		printf("\rCompleted %d models", completed_count);
		if (completed_count == totalModels) {
			break;
		}
		timer.stop();
		times[t++] += timer.elapsed();

		timer.start();
		drop(lVars, dropidx, nVars, lasso, M, numModels);
		cudaDeviceSynchronize();
		timer.stop();
		times[t++] += timer.elapsed();

		timer.start();
		mat_sub<precision>(y, mu, r, numModels * M, *(new dim3(bModM)));
		cudaDeviceSynchronize();
		timer.stop();
		times[t++] += timer.elapsed();
		
		timer.start();
		cublasSetStream(hnd, NULL);
		gemm(hnd, CUBLAS_OP_N, CUBLAS_OP_N, N, numModels, M, &alp, X, N, r, M, &bet, c, N);
		cudaDeviceSynchronize();
		timer.stop();
		times[t++] += timer.elapsed();
		
		timer.start();
		exclude<precision>(c, lVars, nVars, act, M, N, numModels, 0, *(new dim3(bModM)));
		cudaDeviceSynchronize();
		timer.stop();
		times[t++] += timer.elapsed();
		
		timer.start();
		fabsMaxReduce<precision>(c, cmax, buf, cidx, intBuf, numModels, N);
		cudaDeviceSynchronize();
		timer.stop();
		times[t++] += timer.elapsed();

		timer.start();
		lasso_add(lasso, lVars, nVars, cidx, M, N, numModels, *(new dim3(bMod)));
		cudaDeviceSynchronize();
		timer.stop();
		times[t++] += timer.elapsed();

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
		for (int i = 0, s = 0; i < numModels; i++) {
			gather<precision>(XA[i], XA1[i], X, lVars, hNVars[i], hLasso[i], hDropidx[i], M, N, i, streams[s & (numStreams - 1)]);
			s++;
		}
		cudaDeviceSynchronize();
		timer.stop();
		times[t++] += timer.elapsed();

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
		times[t++] += timer.elapsed();
		
		timer.start();
		XAyBatched<precision>(dXA, y, r, nVars, M, numModels);
		cudaDeviceSynchronize();
		timer.stop();
		times[t++] += timer.elapsed();
		
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
		times[t++] += timer.elapsed();
		
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
		times[t++] += timer.elapsed();
		
		timer.start();
		IrBatched<precision>(dI, r, betaOls, nVars, M, numModels, maxVar);
		cudaDeviceSynchronize();
		timer.stop();
		times[t++] += timer.elapsed();
		
		timer.start();
		XAbetaOlsBatched<precision>(dXA, betaOls, d, nVars, M, numModels, maxVar);
		cudaDeviceSynchronize();
		timer.stop();
		times[t++] += timer.elapsed();

		timer.start();
		mat_sub<precision>(d, mu, d, numModels * M, *(new dim3(bModM)));
		cudaDeviceSynchronize();
		timer.stop();
		times[t++] += timer.elapsed();
		
		timer.start();
		gammat<precision>(gamma_tilde, beta, betaOls, dropidx, lVars, nVars, lasso, M, N, numModels, *(new dim3(bModM)));
		cudaDeviceSynchronize();
		timer.stop();
		times[t++] += timer.elapsed();
		
		timer.start();
		cublasSetStream(hnd, NULL);
		gemm(hnd, CUBLAS_OP_N, CUBLAS_OP_N, N, numModels, M, &alp, X, N, d, M, &bet, cd, N);
		cudaDeviceSynchronize();
		timer.stop();
		times[t++] += timer.elapsed();
		
		timer.start();
		cdMinReduce<precision>(c, cd, cmax, r, buf, numModels, N);
		cudaDeviceSynchronize();
		timer.stop();
		times[t++] += timer.elapsed();
		
		timer.start();
		set_gamma<precision>(gamma, gamma_tilde, r, lasso, nVars, maxVariables, M, numModels, *(new dim3(bMod)));
		cudaDeviceSynchronize();
		timer.stop();
		times[t++] += timer.elapsed();
		
		timer.start();
		update<precision>(beta, beta_prev, mu, d, betaOls, gamma, dXA, y, a1, a2, lambda, lVars, nVars, step, M, N, numModels, l1, *(new dim3(bMod)));
		cudaDeviceSynchronize();
		timer.stop();
		times[t++] += timer.elapsed();
		
		totalFlop += flopCounter(M, N, numModels, hNVars);
		e++;
	}

	stepf.close();
	nvarsf.close();
	a1f.close();
	a2f.close();
	lambdaf.close();
	betaf.close();

	// Statistics
	double transferTime = times[0];
	double execTime = 0;
	for (int i = 1; i < 25; i++) execTime += times[i];
	printf("\n");

 	for (int i = 0; i < 25; i++) {
 		printf("Kernel %2d time = %10.4f\n", i, times[i]);
 	}
 	printf("Execution time(s) = %f\n", execTime * 1.0e-3);
 	printf("Transfer time(s) = %f\n", transferTime * 1.0e-3);
	printf("Total Flop count(gflop) = %f\n", totalFlop * 1.0e-9);
	printf("Execution Flops(gflops) = %f\n", (totalFlop * 1.0e-9) / (execTime * 1.0e-3));

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
	cudaFree(lambda);
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
	cudaFree(beta_prev);

	cudaFree(corr_beta);
	cudaFree(corr_sb);
	cudaFree(corr_y);
	cudaFree(corr_tmp);
	cudaFree(corr_betaols);
	cudaFree(corr_yh);
	cudaFree(corr_z);

	for (int i = 0; i < numModels; i++) {
		cudaFree(XA[i]);
		cudaFree(XA1[i]);
		cudaFree(G[i]);
		cudaFree(I[i]);

		cudaFree(corr_XA[i]);
		cudaFree(corr_G[i]);
		cudaFree(corr_I[i]);
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

	cudaFree(corr_dXA);
	cudaFree(corr_dG);
	cudaFree(corr_dI);

	cublasDestroy(hnd);

	return 0;
}
