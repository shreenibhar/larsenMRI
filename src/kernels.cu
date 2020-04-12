#ifndef KERNELS_CU
#define KERNELS_CU

#include "kernels.h"

__global__
void set_model_kernel(precision *Y, precision *y, precision *mu, precision *beta, precision *a1, precision *a2, precision *lambda, precision *randnrm, int *nVars, int *eVars, int *lasso, int *step, int *done, int *act, int M, int N, int hact) {
	int ind = threadIdx.x + blockIdx.x * blockDim.x;
	if (ind < N) {
		beta[ind] = 0;
	}
	if (ind < M) {
		mu[ind] = 0;
		y[ind] = Y[ind * N + hact];
	}
	if (ind == 0) {
		nVars[0] = 0;
		eVars[0] = 0;
		lasso[0] = 0;
		step[0] = 0;
		done[0] = 0;
		act[0] = hact;
		a1[0] = 0;
		a2[0] = 1e6;
		lambda[0] = 1e6;
		randnrm[0] = 1;
	}
}

void set_model(precision *Y, precision *y, precision *mu, precision *beta, precision *a1, precision *a2, precision *lambda, precision *randnrm, int *nVars, int *eVars, int *lasso, int *step, int *done, int *act, int M, int N, int hact, cudaStream_t &stream) {
	dim3 blockDim(1024);
	dim3 gridDim((N + blockDim.x - 1) / blockDim.x);
	set_model_kernel<<<gridDim, blockDim, 0, stream>>>(Y, y, mu, beta, a1, a2, lambda, randnrm, nVars, eVars, lasso, step, done, act, M, N, hact);
}

__global__
void check_kernel(int *nVars, int *step, precision *a1, precision *a2, precision *lambda, int maxVariables, int maxSteps, precision l1, precision l2, precision g, int *done, int numModels) {
	int mod = threadIdx.x + blockIdx.x * blockDim.x;
	if (mod < numModels) {
		if (nVars[mod] >= maxVariables || step[mod] >= maxSteps || a1[mod] >= l1 || a2[mod] <= l2 || lambda[mod] <= g || done[mod]) {
			done[mod] = 1;
		}
	}
}
void check(int *nVars, int *step, precision *a1, precision *a2, precision *lambda, int maxVariables, int maxSteps, precision l1, precision l2, precision g, int *done, int numModels) {
	dim3 blockDim(min(numModels, 1024));
	dim3 gridDim((numModels + blockDim.x - 1) / blockDim.x);
	check_kernel<<<gridDim, blockDim>>>(nVars, step, a1, a2, lambda, maxVariables, maxSteps, l1, l2, g, done, numModels);
}

__global__
void drop_kernel(int *lVars, int *dropidx, int *nVars, int *lasso, int M, int numModels) {
	int mod = threadIdx.x + blockIdx.x * blockDim.x;
	if (mod < numModels && lasso[mod]) {
		int ni = nVars[mod];
		int drop = dropidx[mod];
		for (int i = drop + 1; i < ni; i++) {
			int val = lVars[mod * M + i];
			lVars[mod * M + i - 1] = val;
		}
		nVars[mod] = ni - 1;
	}
}

void drop(int *lVars, int *dropidx, int *nVars, int *lasso, int M, int numModels) {
	dim3 blockDim(min(numModels, 1024));
	dim3 gridDim((numModels + blockDim.x - 1) / blockDim.x);
	drop_kernel<<<gridDim, blockDim>>>(lVars, dropidx, nVars, lasso, M, numModels);
}

__global__
void mat_sub_kernel(precision *a, precision *b, precision *c, int size) {
	int ind = threadIdx.x + blockIdx.x * blockDim.x;
	if (ind < size) {
		c[ind] = a[ind] - b[ind];
	}
}

void mat_sub(precision *a, precision *b, precision *c, int size) {
	dim3 blockDim(min(size, 1024));
	dim3 gridDim((size + blockDim.x - 1) / blockDim.x);
	mat_sub_kernel<<<gridDim, blockDim>>>(a, b, c, size);
}

__global__
void exclude_kernel(precision *absC, int *lVars, int *nVars, int *eVars, int *act, int M, int N, int numModels, precision def) {
	int mod = threadIdx.x + blockIdx.x * blockDim.x;
	if (mod < numModels) {
		int ni = nVars[mod];
		absC[mod * N + act[mod]] = def;
		for (int i = 0; i < ni; i++) {
			int li = lVars[mod * M + i];
			absC[mod * N + li] = def;
		}
		int ei = eVars[mod];
		for (int i = 0; i < ei; i++) {
			int li = lVars[mod * M + M - 1 - i];
			absC[mod * N + li] = def;
		}
	}
}

void exclude(precision *absC, int *lVars, int *nVars, int *eVars, int *act, int M, int N, int numModels, precision def) {
	dim3 blockDim(min(numModels, 1024));
	dim3 gridDim((numModels + blockDim.x - 1) / blockDim.x);
	exclude_kernel<<<gridDim, blockDim>>>(absC, lVars, nVars, eVars, act, M, N, numModels, def);
}

__global__
void lasso_add_kernel(precision *c, int *lasso, int *lVars, int *nVars, int *cidx, int M, int N, int numModels) {
	int mod = threadIdx.x + blockIdx.x * blockDim.x;
	if (mod < numModels) {
		if (!lasso[mod]) {
			int ni = nVars[mod];
			int id = cidx[mod];
			lVars[mod * M + ni] = id;
			nVars[mod] = ni + 1;
			c[mod * N + id] = 0;
		}
	}
}

void lasso_add(precision *c, int *lasso, int *lVars, int *nVars, int *cidx, int M, int N, int numModels) {
	dim3 blockDim(min(numModels, 1024));
	dim3 gridDim((numModels + blockDim.x - 1) / blockDim.x);
	lasso_add_kernel<<<gridDim, blockDim>>>(c, lasso, lVars, nVars, cidx, M, N, numModels);
}

__global__
void gather_add_kernel(precision *XA, precision *XA1, precision *X, int *lVars, int ni, int M, int N, int mod) {
	int ind = threadIdx.x + blockIdx.x * blockDim.x;
	if (ind < M) {
		XA[(ni - 1) * M + ind] = XA1[(ni - 1) * M + ind] = X[ind * N + lVars[mod * M + ni - 1]];
	}
}

__global__
void gather_del_kernel(precision *XA, precision *XA1, precision *X, int ni, int drop, int M, int N, int mod) {
	int mj = threadIdx.x + blockIdx.x * blockDim.x;
	int mi = mj / M;
	mj -= mi * M;
	mi += drop;
	if (mi >= drop && mi < ni && mj < M) {
		XA[mi * M + mj] = XA1[(mi + 1) * M + mj];
	}
}

__global__
void gather_cop_kernel(precision *XA, precision *XA1, precision *X, int ni, int drop, int M, int N, int mod) {
	int mj = threadIdx.x + blockIdx.x * blockDim.x;
	int mi = mj / M;
	mj -= mi * M;
	mi += drop;
	if (mi >= drop && mi < ni && mj < M) {
		XA1[mi * M + mj] = XA[mi * M + mj];
	}
}

void gather(precision *XA, precision *XA1, precision *X, int *lVars, int ni, int lassoCond, int drop, int M, int N, int mod, cudaStream_t &stream) {
	if (!lassoCond) {
		gather_add_kernel<<<1, M, 0, stream>>>(XA, XA1, X, lVars, ni, M, N, mod);
	}
	else {
		if (ni == drop) return;
		dim3 blockDim(min((ni - drop) * M, 1024));
		dim3 gridDim(((ni - drop) * M + blockDim.x - 1) / blockDim.x);
		gather_del_kernel<<<gridDim, blockDim, 0, stream>>>(XA, XA1, X, ni, drop, M, N, mod);
		gather_cop_kernel<<<gridDim, blockDim, 0, stream>>>(XA, XA1, X, ni, drop, M, N, mod);
	}
}

__global__
void checkNan_kernel(int *nVars, int *eVars, int *lVars, int *info, precision *r, precision *d, precision *randnrm, int M, int numModels) {
	int mod = threadIdx.x + blockIdx.x * blockDim.x;
	if (mod < numModels) {
		int ni = nVars[mod];
		precision nrm1 = 0, nrm2 = 0;
		for (int i = 0; i < ni; i++) {
			precision val1 = r[mod * M + i];
			precision val2 = d[mod * M + i];
			nrm1 += val1 * val1;
			nrm2 += val2 * val2;
		}
		nrm1 = sqrt(nrm1);
		nrm2 = sqrt(nrm2);
		precision ratio = (nrm1 + nrm2) / randnrm[mod];
		int infer = info[mod];
		if (ratio > 100 || isnan(ratio) || isinf(ratio) || infer) {
			nVars[mod] -= 1;
			lVars[mod * M + M - 1 - eVars[mod]] = lVars[mod * M + nVars[mod]];
			eVars[mod] += 1;
			infer = 1;
		}
		else randnrm[mod] = nrm1 + nrm2;
		info[mod] = infer;
	}
}

void checkNan(int *nVars, int *eVars, int *lVars, int *info, precision *r, precision *d, precision *randnrm, int M, int numModels) {
	dim3 blockDim(min(numModels, 1024));
	dim3 gridDim((numModels + blockDim.x - 1) / blockDim.x);
	checkNan_kernel<<<gridDim, blockDim>>>(nVars, eVars, lVars, info, r, d, randnrm, M, numModels);
}

__global__
void gammat_kernel(precision *gamma_tilde, precision *beta, precision *betaOls, int *dropidx, int *lVars, int *nVars, int *lasso, int M, int N, int numModels) {
	int mod = threadIdx.x + blockIdx.x * blockDim.x;
	if (mod < numModels) {
		if (lasso[mod]) lasso[mod] = 0;
		int ni = nVars[mod];
		precision miner = inf;
		int id = -1;
		for (int i = 0; i < ni; i++) {
			int si = lVars[mod * M + i];
			precision val = beta[mod * N + si] / (beta[mod * N + si] - betaOls[mod * M + i]);
			val = (val < eps)? inf: val;
			if (val < miner) {
				miner = val;
				id = i;
			}
		}
		gamma_tilde[mod] = miner;
		dropidx[mod] = id;
	}
}

void gammat(precision *gamma_tilde, precision *beta, precision *betaOls, int *dropidx, int *lVars, int *nVars, int *lasso, int M, int N, int numModels) {
	dim3 blockDim(min(numModels, 1024));
	dim3 gridDim((numModels + blockDim.x - 1) / blockDim.x);
	gammat_kernel<<<gridDim, blockDim>>>(gamma_tilde, beta, betaOls, dropidx, lVars, nVars, lasso, M, N, numModels);
}

__global__
void set_gamma_kernel(precision *gamma, precision *gamma_tilde, int *lasso, int *nVars, int maxVariables, int M, int numModels) {
	int mod = threadIdx.x + blockIdx.x * blockDim.x;
	if (mod < numModels) {
		precision gamma_t = gamma_tilde[mod];
		precision gamma_val = gamma[mod];
		if (nVars[mod] == maxVariables) {
			gamma[mod] = 1;
		}
		if (gamma_t < gamma_val) {
			lasso[mod] = 1;
			gamma[mod] = gamma_t;
		}
		else {
			gamma[mod] = gamma_val;
		}
	}
}

void set_gamma(precision *gamma, precision *gamma_tilde, int *lasso, int *nVars, int maxVariables, int M, int numModels) {
	dim3 blockDim(min(numModels, 1024));
	dim3 gridDim((numModels + blockDim.x - 1) / blockDim.x);
	set_gamma_kernel<<<gridDim, blockDim>>>(gamma, gamma_tilde, lasso, nVars, maxVariables, M, numModels);
}

__global__
void update_kernel(precision *beta, precision *beta_prev, precision *mu, precision *d, precision *betaOls, precision *gamma, precision **dXA, precision *y, precision *a1, precision *a2, precision *lambda, int *lVars, int *nVars, int *step, int M, int N, int numModels, precision max_l1) {
	int mod = threadIdx.x + blockIdx.x * blockDim.x;
	if (mod < numModels) {
		int ni = nVars[mod];
		precision gamma_val = gamma[mod];
		precision l1 = 0, l1_prev = 0, delta = 0;
		for (int i = 0; i < ni; i++) {
			int si = lVars[mod * M + i];
			precision beta_val = beta[mod * N + si];
			precision betaOls_val = betaOls[mod * M + i];
			beta_prev[mod * N + si] = beta_val;
			beta[mod * N + si] = beta_val + gamma_val * (betaOls_val - beta_val);
			si = (beta_val > eps) - (beta_val < -eps);
			l1_prev += abs(beta_val);
			delta += si * (betaOls_val - beta_val);
			l1 += abs(beta_val + gamma_val * (betaOls_val - beta_val));
		}
		if (l1 > max_l1) {
			gamma_val = (max_l1 - l1_prev) / delta;
			for (int i = 0; i < ni; i++) {
				int si = lVars[mod * M + i];
				beta[mod * N + si] = beta_prev[mod * N + si] + gamma_val * (betaOls[mod * M + i] - beta_prev[mod * N + si]);
			}
			l1 = max_l1;
		}
		a1[mod] = l1;
		precision l2 = 0, g = 0;
		for (int i = 0; i < M; i++) {
			mu[mod * M + i] += gamma_val * d[mod * M + i];
			precision tmp = y[mod * M + i] - mu[mod * M + i];
			l2 += tmp * tmp;
			g += dXA[mod][i] * tmp;
		}
		l2 = sqrt(l2);
		a2[mod] = l2;
		lambda[mod] = abs(g / l2);
		step[mod] += 1;
	}
}

void update(precision *beta, precision *beta_prev, precision *mu, precision *d, precision *betaOls, precision *gamma, precision **dXA, precision *y, precision *a1, precision *a2, precision *lambda, int *lVars, int *nVars, int *step, int M, int N, int numModels, precision max_l1) {
	dim3 blockDim(min(numModels, 1024));
	dim3 gridDim((numModels + blockDim.x - 1) / blockDim.x);
	update_kernel<<<gridDim, blockDim>>>(beta, beta_prev, mu, d, betaOls, gamma, dXA, y, a1, a2, lambda, lVars, nVars, step, M, N, numModels, max_l1);
}

__global__
void gatherAll_kernel(corr_precision *XA, corr_precision *y, corr_precision *X, int *lVars, int ni, int M, int N, int act) {
	int ind = threadIdx.x + blockIdx.x * blockDim.x;
	int nind = ind / M;
	ind -= nind * M;
	if (nind < ni && ind < M) {
		if (nind == 0) {
			y[ind] = X[ind * N + act];
		}
		XA[nind * M + ind] = X[ind * N + lVars[nind]];
	}
}

void gatherAll(corr_precision *XA, corr_precision *y, corr_precision *X, int *lVars, int ni, int M, int N, int act, cudaStream_t &stream) {
	dim3 blockDim(1024);
	dim3 gridDim((ni * M + blockDim.x - 1) / blockDim.x);
	gatherAll_kernel<<<gridDim, blockDim, 0, stream>>>(XA, y, X, lVars, ni, M, N, act);
}

__global__
void computeSign_kernel(corr_precision *sb, precision *beta, precision *beta_prev, int *lVars, int *dropidx, int *lasso, int *nVars) {
	int ni = nVars[0];
	int lasso_cond = lasso[0], drop_ind = dropidx[0];
	int active_size = 0;
	for (int i = 0; i < ni; i++) {
		int si = lVars[i];
		int sg = (beta[si] > eps) - (beta[si] < -eps);
		if (lasso_cond && i == drop_ind) {
			sg = (beta_prev[si] > eps) - (beta_prev[si] < -eps);
		}
		if (sg != 0) {
			sb[active_size] = (corr_precision) sg;
			lVars[active_size] = si;
			active_size++;
		}
	}
	nVars[0] = active_size;
}

void computeSign(corr_precision *sb, precision *beta, precision *beta_prev, int *lVars, int *dropidx, int *lasso, int *nVars, cudaStream_t &stream) {
	computeSign_kernel<<<1, 1, 0, stream>>>(sb, beta, beta_prev, lVars, dropidx, lasso, nVars);
}

__global__
void correct_kernel(corr_precision *beta, corr_precision *betaols, corr_precision *sb, corr_precision *y, corr_precision *yh, corr_precision *z, precision *a1, precision *a2, precision *lambda, corr_precision min_l2, corr_precision g, int ni, int M) {
	corr_precision zz = 0, yhyh = 0;
	for (int i = 0; i < M; i++) {
		if (i < ni) zz += sb[i] * z[i];
		yhyh += (y[i] - yh[i]) * (y[i] - yh[i]);
	}
	corr_precision err = a2[0], G = lambda[0];
	if (err < min_l2 && min_l2 * min_l2 >= yhyh) {
		err = min_l2;
		G = sqrt((min_l2 * min_l2 - yhyh) / (min_l2 * min_l2 * zz));
	}
	if (G < g && 1 > g * g * zz) {
		G = g;
		err = sqrt(yhyh / (1 - g * g * zz));
	}
	corr_precision l1 = 0;
	for (int i = 0; i < ni; i++) {
		beta[i] = betaols[i] - err * G * z[i];
		l1 += abs(beta[i]);
	}
	a1[0] = l1;
	a2[0] = err;
	lambda[0] = G;
}

void correct(corr_precision *beta, corr_precision *betaols, corr_precision *sb, corr_precision *y, corr_precision *yh, corr_precision *z, precision *a1, precision *a2, precision *lambda, corr_precision min_l2, corr_precision g, int ni, int M, cudaStream_t &stream) {
	correct_kernel<<<1, 1, 0, stream>>>(beta, betaols, sb, y, yh, z, a1, a2, lambda, min_l2, g, ni, M);
}

#endif
