#ifndef KERNELS_CU
#define KERNELS_CU

#include "kernels.h"

template<typename T>
__global__
void set_model_kernel(T *Y, T *y, T *mu, T *beta, T *a1, T *a2, T *lambda, int *nVars, int *lasso, int *step, int *done, int *act, int M, int N, int hact) {
	int ind = threadIdx.x + blockIdx.x * blockDim.x;
	if (ind == 0) {
		nVars[0] = 0;
		lasso[0] = 0;
		step[0] = 0;
		done[0] = 0;
		act[0] = hact;
		a1[0] = 0;
		a2[0] = 1e6;
		lambda[0] = 1e6;
	}
	if (ind < M) {
		mu[ind] = 0;
		y[ind] = Y[ind * N + hact];
	}
	if (ind < N) {
		beta[ind] = 0;
	}
}

template<typename T>
void set_model(T *Y, T *y, T *mu, T *beta, T *a1, T *a2, T *lambda, int *nVars, int *lasso, int *step, int *done, int *act, int M, int N, int hact, cudaStream_t &stream) {
	dim3 blockDim(1024);
	dim3 gridDim((N + blockDim.x - 1) / blockDim.x);
	set_model_kernel<T><<<gridDim, blockDim, 0, stream>>>(Y, y, mu, beta, a1, a2, lambda, nVars, lasso, step, done, act, M, N, hact);
}

template void set_model<float>(float *Y, float *y, float *mu, float *beta, float *a1, float *a2, float *lambda, int *nVars, int *lasso, int *step, int *done, int *act, int M, int N, int hact, cudaStream_t &stream);
template void set_model<double>(double *Y, double *y, double *mu, double *beta, double *a1, double *a2, double *lambda, int *nVars, int *lasso, int *step, int *done, int *act, int M, int N, int hact, cudaStream_t &stream);

template<typename T>
__global__
void check_kernel(int *nVars, int *step, T *a1, T *a2, T * lambda, int maxVariables, int maxSteps, T l1, T l2, T g, int *done, int numModels) {
	int mod = threadIdx.x + blockIdx.x * blockDim.x;
	if (mod < numModels) {
		if (nVars[mod] < maxVariables && step[mod] < maxSteps && a1[mod] < l1 && a2[mod] > l2 && lambda[mod] > g && !done[mod]) {
		}
		else {
			done[mod] = 1;
		}
	}
}
template<typename T>
void check(int *nVars, int *step, T *a1, T *a2, T * lambda, int maxVariables, int maxSteps, T l1, T l2, T g, int *done, int numModels) {
	dim3 blockDim(min(numModels, 1024));
	dim3 gridDim((numModels + blockDim.x - 1) / blockDim.x);
	check_kernel<T><<<gridDim, blockDim>>>(nVars, step, a1, a2, lambda, maxVariables, maxSteps, l1, l2, g, done, numModels);
}

template void check<float>(int *nVars, int *step, float *a1, float *a2, float * lambda, int maxVariables, int maxSteps, float l1, float l2, float g, int *done, int numModels);
template void check<double>(int *nVars, int *step, double *a1, double *a2, double * lambda, int maxVariables, int maxSteps, double l1, double l2, double g, int *done, int numModels);

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

template<typename T>
__global__
void mat_sub_kernel(T *a, T *b, T *c, int size) {
	int ind = threadIdx.x + blockIdx.x * blockDim.x;
	if (ind < size) {
		c[ind] = a[ind] - b[ind];
	}
}

template<typename T>
void mat_sub(T *a, T *b, T *c, int size) {
	dim3 blockDim(min(size, 1024));
	dim3 gridDim((size + blockDim.x - 1) / blockDim.x);
	mat_sub_kernel<T><<<gridDim, blockDim>>>(a, b, c, size);
}

template void mat_sub<float>(float *a, float *b, float *c, int size);
template void mat_sub<double>(double *a, double *b, double *c, int size);

template<typename T>
__global__
void exclude_kernel(T *absC, int *lVars, int *nVars, int *act, int M, int N, int numModels, T def) {
	int mod = threadIdx.x + blockIdx.x * blockDim.x;
	if (mod < numModels) {
		int ni = nVars[mod];
		absC[mod * N + act[mod]] = def;
		for (int i = 0; i < ni; i++) {
			int li = lVars[mod * M + i];
			absC[mod * N + li] = def;
		}
	}
}

template<typename T>
void exclude(T *absC, int *lVars, int *nVars, int *act, int M, int N, int numModels, T def) {
	dim3 blockDim(min(numModels, 1024));
	dim3 gridDim((numModels + blockDim.x - 1) / blockDim.x);
	exclude_kernel<T><<<gridDim, blockDim>>>(absC, lVars, nVars, act, M, N, numModels, def);
}

template void exclude<float>(float *absC, int *lVars, int *nVars, int *act, int M, int N, int numModels, float def);
template void exclude<double>(double *absC, int *lVars, int *nVars, int *act, int M, int N, int numModels, double def);

__global__
void lasso_add_kernel(int *lasso, int *lVars, int *nVars, int *cidx, int M, int N, int numModels) {
	int mod = threadIdx.x + blockIdx.x * blockDim.x;
	if (mod < numModels) {
		if (!lasso[mod]) {
			int ni = nVars[mod];
			int id = cidx[mod];
			lVars[mod * M + ni] = id;
			nVars[mod] = ni + 1;
		}
	}
}

void lasso_add(int *lasso, int *lVars, int *nVars, int *cidx, int M, int N, int numModels) {
	dim3 blockDim(min(numModels, 1024));
	dim3 gridDim((numModels + blockDim.x - 1) / blockDim.x);
	lasso_add_kernel<<<gridDim, blockDim>>>(lasso, lVars, nVars, cidx, M, N, numModels);
}

template<typename T>
__global__
void gather_add_kernel(T *XA, T *XA1, T *X, int *lVars, int ni, int M, int N, int mod) {
	int ind = threadIdx.x + blockIdx.x * blockDim.x;
	if (ind < M) {
		XA[(ni - 1) * M + ind] = XA1[(ni - 1) * M + ind] = X[ind * N + lVars[mod * M + ni - 1]];
	}
}

template<typename T>
__global__
void gather_del_kernel(T *XA, T *XA1, T *X, int ni, int drop, int M, int N, int mod) {
	int mj = threadIdx.x + blockIdx.x * blockDim.x;
	int mi = mj / M;
	mj -= mi * M;
	mi += drop;
	if (mi >= drop && mi < ni && mj < M) {
		XA[mi * M + mj] = XA1[(mi + 1) * M + mj];
	}
}

template<typename T>
__global__
void gather_cop_kernel(T *XA, T *XA1, T *X, int ni, int drop, int M, int N, int mod) {
	int mj = threadIdx.x + blockIdx.x * blockDim.x;
	int mi = mj / M;
	mj -= mi * M;
	mi += drop;
	if (mi >= drop && mi < ni && mj < M) {
		XA1[mi * M + mj] = XA[mi * M + mj];
	}
}

template<typename T>
void gather(T *XA, T *XA1, T *X, int *lVars, int ni, int lassoCond, int drop, int M, int N, int mod, cudaStream_t &stream) {
	if (!lassoCond) {
		gather_add_kernel<T><<<1, M, 0, stream>>>(XA, XA1, X, lVars, ni, M, N, mod);
	}
	else {
		if (ni == drop) return;
		dim3 blockDim(min((ni - drop) * M, 1024));
		dim3 gridDim(((ni - drop) * M + blockDim.x - 1) / blockDim.x);
		gather_del_kernel<T><<<gridDim, blockDim, 0, stream>>>(XA, XA1, X, ni, drop, M, N, mod);
		gather_cop_kernel<T><<<gridDim, blockDim, 0, stream>>>(XA, XA1, X, ni, drop, M, N, mod);
	}
}

template void gather<float>(float *XA, float *XA1, float *X, int *lVars, int ni, int lassoCond, int drop, int M, int N, int mod, cudaStream_t &stream);
template void gather<double>(double *XA, double *XA1, double *X, int *lVars, int ni, int lassoCond, int drop, int M, int N, int mod, cudaStream_t &stream);

template<typename T>
__global__
void gammat_kernel(T *gamma_tilde, T *beta, T *betaOls, int *dropidx, int *lVars, int *nVars, int *lasso, int M, int N, int numModels) {
	int mod = threadIdx.x + blockIdx.x * blockDim.x;
	if (mod < numModels) {
		if (lasso[mod]) lasso[mod] = 0;
		int ni = nVars[mod];
		T miner = inf;
		int id = -1;
		for (int i = 0; i < ni; i++) {
			int si = lVars[mod * M + i];
			T val = beta[mod * N + si] / (beta[mod * N + si] - betaOls[mod * M + i]);
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

template<typename T>
void gammat(T *gamma_tilde, T *beta, T *betaOls, int *dropidx, int *lVars, int *nVars, int *lasso, int M, int N, int numModels) {
	dim3 blockDim(min(numModels, 1024));
	dim3 gridDim((numModels + blockDim.x - 1) / blockDim.x);
	gammat_kernel<T><<<gridDim, blockDim>>>(gamma_tilde, beta, betaOls, dropidx, lVars, nVars, lasso, M, N, numModels);
}

template void gammat<float>(float *gamma_tilde, float *beta, float *betaOls, int *dropidx, int *lVars, int *nVars, int *lasso, int M, int N, int numModels);
template void gammat<double>(double *gamma_tilde, double *beta, double *betaOls, int *dropidx, int *lVars, int *nVars, int *lasso, int M, int N, int numModels);

template<typename T>
__global__
void set_gamma_kernel(T *gamma, T *gamma_tilde, T *r, int *lasso, int *nVars, int maxVariables, int M, int numModels) {
	int mod = threadIdx.x + blockIdx.x * blockDim.x;
	if (mod < numModels) {
		T gamma_t = gamma_tilde[mod];
		T gamma_val = r[mod];
		if (nVars[mod] == maxVariables) {
			gamma[mod] = 1;
		}
		else if (gamma_t < gamma_val) {
			lasso[mod] = 1;
			gamma[mod] = gamma_t;
		}
		else {
			gamma[mod] = gamma_val;
		}
	}
}

template<typename T>
void set_gamma(T *gamma, T *gamma_tilde, T *r, int *lasso, int *nVars, int maxVariables, int M, int numModels) {
	dim3 blockDim(min(numModels, 1024));
	dim3 gridDim((numModels + blockDim.x - 1) / blockDim.x);
	set_gamma_kernel<T><<<gridDim, blockDim>>>(gamma, gamma_tilde, r, lasso, nVars, maxVariables, M, numModels);
}

template void set_gamma<float>(float *gamma, float *gamma_tilde, float *r, int *lasso, int *nVars, int maxVariables, int M, int numModels);
template void set_gamma<double>(double *gamma, double *gamma_tilde, double *r, int *lasso, int *nVars, int maxVariables, int M, int numModels);

template<typename T>
__global__
void update_kernel(T *beta, T *beta_prev, T *mu, T *d, T *betaOls, T *gamma, T **dXA, T *y, T *a1, T *a2, T *lambda, int *lVars, int *nVars, int *step, int M, int N, int numModels, T max_l1) {
	int mod = threadIdx.x + blockIdx.x * blockDim.x;
	if (mod < numModels) {
		int ni = nVars[mod];
		T gamma_val = gamma[mod];
		T l1 = 0, l1one = 0, l1two = 0, one, two, three;
		for (int i = 0; i < ni; i++) {
			int si = lVars[mod * M + i];
			one = beta[mod * N + si];
			two = betaOls[mod * M + i] - one;
			three = one + gamma_val * two;
			beta_prev[mod * N + si] = one;
			beta[mod * N + si] = three;
			si = (one > eps) - (one < -eps);
			l1one += abs(one);
			l1two += si * two;
			l1 += abs(three);
		}
		if (l1 > max_l1) {
			gamma_val = (max_l1 - l1one) / l1two;
			for (int i = 0; i < ni; i++) {
				int si = lVars[mod * M + i];
				one = beta_prev[mod * N + si];
				two = betaOls[mod * M + i] - one;
				three = one + gamma_val * two;
				beta[mod * N + si] = three;
			}
			l1 = max_l1;
		}
		a1[mod] = l1;
		l1two = 0;
		l1one = 0;
		for (int i = 0; i < M; i++) {
			one = mu[mod * M + i];
			two = d[mod * M + i];
			three = one + gamma_val * two;
			mu[mod * M + i] = three;
			one = y[mod * M + i] - three;
			l1two += one * one;
			l1one += dXA[mod][i] * one;
		}
		l1two = sqrt(l1two);
		l1one = abs(l1one) / l1two;
		a2[mod] = l1two;
		lambda[mod] = l1one;
		step[mod] += 1;
	}
}

template<typename T>
void update(T *beta, T *beta_prev, T *mu, T *d, T *betaOls, T *gamma, T **dXA, T *y, T *a1, T *a2, T *lambda, int *lVars, int *nVars, int *step, int M, int N, int numModels, T max_l1) {
	dim3 blockDim(min(numModels, 1024));
	dim3 gridDim((numModels + blockDim.x - 1) / blockDim.x);
	update_kernel<T><<<gridDim, blockDim>>>(beta, beta_prev, mu, d, betaOls, gamma, dXA, y, a1, a2, lambda, lVars, nVars, step, M, N, numModels, max_l1);
}

template void update<float>(float *beta, float *beta_prev, float *mu, float *d, float *betaOls, float *gamma, float **dXA, float *y, float *a1, float *a2, float *lambda, int *lVars, int *nVars, int *step, int M, int N, int numModels, float max_l1);
template void update<double>(double *beta, double *beta_prev, double *mu, double *d, double *betaOls, double *gamma, double **dXA, double *y, double *a1, double *a2, double *lambda, int *lVars, int *nVars, int *step, int M, int N, int numModels, double max_l1);

template<typename T>
__global__
void copyUp_kernel(double *varUp, T *var, int size) {
	int ind = threadIdx.x + blockIdx.x * blockDim.x;
	if (ind < size) {
		varUp[ind] = (double) var[ind];
	}
}

template<typename T>
void copyUp(double *varUp, T *var, int size, cudaStream_t &stream) {
	dim3 blockDim(min(size, 1024));
	dim3 gridDim((size + blockDim.x - 1) / blockDim.x);
	copyUp_kernel<T><<<gridDim, blockDim, 0, stream>>>(varUp, var, size);
}

template void copyUp<float>(double *varUp, float *var, int size, cudaStream_t &stream);
template void copyUp<double>(double *varUp, double *var, int size, cudaStream_t &stream);

template<typename T>
__global__
void computeSign_kernel(double *sb, T *beta, T *beta_prev, int *lVars, int ni) {
	int ind = threadIdx.x + blockIdx.x * blockDim.x;
	if (ind < ni) {
		int si = lVars[ind];
		int sg = (beta[si] > eps) - (beta[si] < -eps);
		if (!sg) sg = (beta_prev[si] > eps) - (beta_prev[si] < -eps);
		sb[ind] = (T) sg;
	}
}

template<typename T>
void computeSign(double *sb, T *beta, T *beta_prev, int *lVars, int ni, cudaStream_t &stream) {
	dim3 blockDim(min(ni, 1024));
	dim3 gridDim((ni + blockDim.x - 1) / blockDim.x);
	computeSign_kernel<T><<<gridDim, blockDim, 0, stream>>>(sb, beta, beta_prev, lVars, ni);
}

template void computeSign<float>(double *sb, float *beta, float *beta_prev, int *lVars, int ni, cudaStream_t &stream);
template void computeSign<double>(double *sb, double *beta, double *beta_prev, int *lVars, int ni, cudaStream_t &stream);

template<typename T>
__global__
void correct_kernel(double *beta, double *betaols, double *sb, double *y, double *yh, double *z, T *a1, T *a2, T *lambda, double min_l2, double g, int ni, int M) {
	double zz = 0, yhyh = 0;
	for (int i = 0; i < M; i++) {
		if (i < ni) zz += sb[i] * z[i];
		yhyh += (y[i] - yh[i]) * (y[i] - yh[i]);
	}
	double err = a2[0], G = lambda[0];
	if (err < min_l2) {
		err = min_l2;
		G = sqrt((min_l2 * min_l2 - yhyh) / (min_l2 * min_l2 * zz));
	}
	if (G < g) {
		G = g;
		err = sqrt(yhyh / (1 - g * g * zz));
	}
	double l1 = 0;
	for (int i = 0; i < ni; i++) {
		beta[i] = betaols[i] - err * G * z[i];
		l1 += abs(beta[i]);
	}
	a1[0] = l1;
	a2[0] = err;
	lambda[0] = G;
}

template<typename T>
void correct(double *beta, double *betaols, double *sb, double *y, double *yh, double *z, T *a1, T *a2, T *lambda, double min_l2, double g, int ni, int M, cudaStream_t &stream) {
	correct_kernel<T><<<1, 1, 0, stream>>>(beta, betaols, sb, y, yh, z, a1, a2, lambda, min_l2, g, ni, M);
}

template void correct<float>(double *beta, double *betaols, double *sb, double *y, double *yh, double *z, float *a1, float *a2, float *lambda, double min_l2, double g, int ni, int M, cudaStream_t &stream);
template void correct<double>(double *beta, double *betaols, double *sb, double *y, double *yh, double *z, double *a1, double *a2, double *lambda, double min_l2, double g, int ni, int M, cudaStream_t &stream);

#endif
