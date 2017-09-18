#ifndef KERNELS_CU
#define KERNELS_CU

#include "kernels.h"

template<typename T>
__global__
void set_model_kernel(T *Y, T *y, T *mu, T *beta, T *alp, T *bet, int *nVars, int *lasso, int *step, int *done, int *act, int M, int N, int mod, int hact) {
	int ind = threadIdx.x + blockIdx.x * blockDim.x;
	if (ind == 0) {
		nVars[mod] = 0;
		lasso[mod] = 0;
		step[mod] = 0;
		done[mod] = 0;
		act[mod] = hact;
		alp[0] = 1;
		bet[0] = 0;
	}
	if (ind < M) {
		mu[mod * M + ind] = 0;
		y[mod * M + ind] = Y[ind * N + hact];
	}
	if (ind < N) {
		beta[mod * N + ind] = 0;
	}
}

template<typename T>
void set_model(T *Y, T *y, T *mu, T *beta, T *alp, T *bet, int *nVars, int *lasso, int *step, int *done, int *act, int M, int N, int mod, int hact, cudaStream_t stream, dim3 blockDim) {
	dim3 gridDim((N + blockDim.x - 1) / blockDim.x);
	set_model_kernel<T><<<gridDim, blockDim, 0, stream>>>(Y, y, mu, beta, alp, bet, nVars, lasso, step, done, act, M, N, mod, hact);
}

template void set_model<float>(float *Y, float *y, float *mu, float *beta, float *alp, float *bet, int *nVars, int *lasso, int *step, int *done, int *act, int M, int N, int mod, int hact, cudaStream_t stream, dim3 blockDim);
template void set_model<double>(double *Y, double *y, double *mu, double *beta, double *alp, double *bet, int *nVars, int *lasso, int *step, int *done, int *act, int M, int N, int mod, int hact, cudaStream_t stream, dim3 blockDim);

__global__
void check_kernel(int *nVars, int *step, int maxVariables, int maxSteps, int *done, int *ctrl, int numModels) {
	int mod = threadIdx.x + blockIdx.x * blockDim.x;
	if (mod < numModels) {
		if (nVars[mod] < maxVariables && step[mod] < maxSteps && !done[mod]) {
		}
		else {
			if (done[mod] == 0 || done[mod] == 1) {
				done[mod] = 2;
				ctrl[0] = 1;
			}
			else if (done[mod] == 2) {
				done[mod] = 3;
			}
		}
	}
}

void check(int *nVars, int *step, int maxVariables, int maxSteps, int *done, int *ctrl, int numModels) {
	int block = (numModels < 1024)? numModels: 1024;
	dim3 blockDim(block);
	dim3 gridDim((numModels + blockDim.x - 1) / blockDim.x);
	check_kernel<<<gridDim, blockDim>>>(nVars, step, maxVariables, maxSteps, done, ctrl, numModels);
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
void mat_sub(T *a, T *b, T *c, int size, dim3 blockDim) {
	dim3 gridDim((size + blockDim.x - 1) / blockDim.x);
	mat_sub_kernel<T><<<gridDim, blockDim>>>(a, b, c, size);
}

template void mat_sub<float>(float *a, float *b, float *c, int size, dim3 blockDim);
template void mat_sub<double>(double *a, double *b, double *c, int size, dim3 blockDim);

template<typename T>
__global__
void exclude_kernel(T *absC, int *lVars, int *nVars, int *act, int M, int N, int numModels, T def) {
	int ind = threadIdx.x + blockIdx.x * blockDim.x;
	int mod = ind / M;
	ind -= mod * M;
	if (mod < numModels) {
		int ni = nVars[mod];
		if (ind == M - 1) {
			absC[mod * N + act[mod]] = def;
		}
		if (ind < ni) {
			int li = lVars[mod * M + ind];
			absC[mod * N + li] = def;
		}
	}
}

template<typename T>
void exclude(T *absC, int *lVars, int *nVars, int *act, int M, int N, int numModels, T def, dim3 blockDim) {
	dim3 gridDim((numModels * M + blockDim.x - 1) / blockDim.x);
	exclude_kernel<T><<<gridDim, blockDim>>>(absC, lVars, nVars, act, M, N, numModels, def);
}

template void exclude<float>(float *absC, int *lVars, int *nVars, int *act, int M, int N, int numModels, float def, dim3 blockDim);
template void exclude<double>(double *absC, int *lVars, int *nVars, int *act, int M, int N, int numModels, double def, dim3 blockDim);

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

void lasso_add(int *lasso, int *lVars, int *nVars, int *cidx, int M, int N, int numModels, dim3 blockDim) {
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
void gather(T *XA, T *XA1, T *X, int *lVars, int ni, int lassoCond, int drop, int M, int N, int mod, cudaStream_t stream) {
	if (!lassoCond) {
		gather_add_kernel<T><<<1, M, 0, stream>>>(XA, XA1, X, lVars, ni, M, N, mod);
	}
	else {
		dim3 blockDim(1024);
		dim3 gridDim(((ni - drop) * M + blockDim.x - 1) / blockDim.x);
		gather_del_kernel<T><<<gridDim, blockDim, 0, stream>>>(XA, XA1, X, ni, drop, M, N, mod);
		gather_cop_kernel<T><<<gridDim, blockDim, 0, stream>>>(XA, XA1, X, ni, drop, M, N, mod);
	}
}

template void gather<float>(float *XA, float *XA1, float *X, int *lVars, int ni, int lassoCond, int drop, int M, int N, int mod, cudaStream_t stream);
template void gather<double>(double *XA, double *XA1, double *X, int *lVars, int ni, int lassoCond, int drop, int M, int N, int mod, cudaStream_t stream);

template<typename T>
__global__
void gammat_kernel(T *gamma_tilde, T *beta, T *betaOls, int *lVars, int *nVars, int *lasso, int M, int N, int numModels) {
	int ind = threadIdx.x + blockIdx.x * blockDim.x;
	int mod = ind / M;
	ind -= mod * M;
	if (ind < M && mod < numModels) {
		int ni = nVars[mod];
		if (ind < ni - 1) {
			int si = lVars[mod * M + ind];
			T val = beta[mod * N + si] / (beta[mod * N + si] - betaOls[mod * M + ind]);
			val = (val <= 0)? inf: val;
			gamma_tilde[mod * M + ind] = val;
		}
		else if (ind == 0 && ni - 1 <= 0) {
			gamma_tilde[mod * M + ind] = inf;
		}
		if (ind == 0 && lasso[mod]) {
			lasso[mod] = 0;
		}
	}
}

template<typename T>
void gammat(T *gamma_tilde, T *beta, T *betaOls, int *lVars, int *nVars, int *lasso, int M, int N, int numModels, dim3 blockDim) {
	dim3 gridDim((numModels * M + blockDim.x - 1) / blockDim.x);
	gammat_kernel<T><<<gridDim, blockDim>>>(gamma_tilde, beta, betaOls, lVars, nVars, lasso, M, N, numModels);
}

template void gammat<float>(float *gamma_tilde, float *beta, float *betaOls, int *lVars, int *nVars, int *lasso, int M, int N, int numModels, dim3 blockDim);
template void gammat<double>(double *gamma_tilde, double *beta, double *betaOls, int *lVars, int *nVars, int *lasso, int M, int N, int numModels, dim3 blockDim);

template<typename T>
__global__
void set_gamma_kernel(T *gamma, T *gamma_tilde, T *r, int *dropidx, int *lasso, int *nVars, int maxVariables, int M, int numModels) {
	int mod = threadIdx.x + blockIdx.x * blockDim.x;
	if (mod < numModels) {
		T gamma_t = gamma_tilde[mod * M + dropidx[mod] - 1];
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
void set_gamma(T *gamma, T *gamma_tilde, T *r, int *dropidx, int *lasso, int *nVars, int maxVariables, int M, int numModels, dim3 blockDim) {
	dim3 gridDim((numModels + blockDim.x - 1) / blockDim.x);
	set_gamma_kernel<T><<<gridDim, blockDim>>>(gamma, gamma_tilde, r, dropidx, lasso, nVars, maxVariables, M, numModels);
}

template void set_gamma<float>(float *gamma, float *gamma_tilde, float *r, int *dropidx, int *lasso, int *nVars, int maxVariables, int M, int numModels, dim3 blockDim);
template void set_gamma<double>(double *gamma, double *gamma_tilde, double *r, int *dropidx, int *lasso, int *nVars, int maxVariables, int M, int numModels, dim3 blockDim);

template<typename T>
__global__
void update_kernel(T *beta, T *mu, T *d, T *betaOls, T *gamma, int *lVars, int *nVars, int M, int N, int numModels) {
	int ind = threadIdx.x + blockIdx.x * blockDim.x;
	int mod = ind / M;
	ind -= mod * M;
	if (mod < numModels) {
		int ni = nVars[mod];
		T gamma_val = gamma[mod];
		if (ind < M) {
			mu[mod * M + ind] += gamma_val * d[mod * M + ind];
			if (ind < ni) {
				int si = lVars[mod * M + ind];
				beta[mod * N + si] += gamma_val * (betaOls[mod * M + ind] - beta[mod * N + si]);
			}
		}
	}
}

template<typename T>
void update(T *beta, T *mu, T *d, T *betaOls, T *gamma, int *lVars, int *nVars, int M, int N, int numModels, dim3 blockDim) {
	dim3 gridDim((numModels * M + blockDim.x - 1) / blockDim.x);
	update_kernel<T><<<gridDim, blockDim>>>(beta, mu, d, betaOls, gamma, lVars, nVars, M, N, numModels);
}

template void update<float>(float *beta, float *mu, float *d, float *betaOls, float *gamma, int *lVars, int *nVars, int M, int N, int numModels, dim3 blockDim);
template void update<double>(double *beta, double *mu, double *d, double *betaOls, double *gamma, int *lVars, int *nVars, int M, int N, int numModels, dim3 blockDim);

__global__
void drop_kernel(int *lVars, int *dropidx, int *nVars, int *lasso, int M, int numModels) {
	int mod = blockIdx.x;
	int ind = threadIdx.x;
	if (mod < numModels && lasso[mod]) {
		int ni = nVars[mod];
		int drop = dropidx[mod] - 1;
		if (ind < ni && ind > drop) {
			int val = lVars[mod * M + ind];
			__syncthreads();
			lVars[mod * M + ind - 1] = val;
		}
		else return;
		if (ind == ni - 1) nVars[mod] = ni - 1;
	}
	else return;
}

void drop(int *lVars, int *dropidx, int *nVars, int *lasso, int M, int numModels) {
		drop_kernel<<<numModels, M>>>(lVars, dropidx, nVars, lasso, M, numModels);
}

template<typename T>
__global__
void final_kernel(T *a1, T *a2, T *cmax, T *r, int *step, int *done, int numModels, T g) {
	int mod = threadIdx.x + blockIdx.x * blockDim.x;
	if (mod < numModels) {
		step[mod] += 1;
		T a1_val = cmax[mod], a2_val = sqrt(r[mod]);
		if (step[mod] > 1) {
			T G = -(a2_val - a2[mod]) / (a1_val - a1[mod]);
			if (G < g) {
				if (!done[mod]) done[mod] = 1;
				return;
			}
		}
		a1[mod] = a1_val;
		a2[mod] = a2_val;
	}
}

template<typename T>
void final(T *a1, T *a2, T *cmax, T *r, int *step, int *done, int numModels, T g, dim3 blockDim) {
		dim3 gridDim((numModels + blockDim.x - 1) / blockDim.x);
		final_kernel<<<gridDim, blockDim>>>(a1, a2, cmax, r, step, done, numModels, g);
}

template void final<float>(float *a1, float *a2, float *cmax, float *r, int *step, int *done, int numModels, float g, dim3 blockDim);
template void final<double>(double *a1, double *a2, double *cmax, double *r, int *step, int *done, int numModels, double g, dim3 blockDim);

#endif
