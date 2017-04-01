#ifndef LARS_PROC_KERNEL
#define LARS_PROC_KERNEL

#include <bits/stdc++.h>
#include <cuda_runtime.h>

#define INF 50000

using namespace std;

template<class T>
__global__ void d_init_with(T *d_mat, T val, int m, int n) {
	int mi = threadIdx.x + blockIdx.x * blockDim.x;
	int ni = threadIdx.y + blockIdx.y * blockDim.y;
	if (mi < m && ni < n)
		d_mat[mi * n + ni] = val;
}

template<class T>
void h_init_with(T *d_mat, T val, int m, int n) {
	dim3 block_dim(32, 32);
	dim3 grid_dim((m + block_dim.x - 1) / block_dim.x,
		(n + block_dim.y - 1) / block_dim.y);
	d_init_with<T><<<grid_dim, block_dim>>>(d_mat, val, m, n);
}

__global__ void d_check(int *d_ctrl, int *d_step, int *d_nVars,  int *d_done, int M, int Z, int num_models) {
	int mod = threadIdx.x + blockIdx.x * blockDim.x;
	if (mod < num_models && !d_done[mod]) {
		int	max_vars = min(M - 1, Z - 1);
		if (d_nVars[mod] < max_vars && d_step[mod] < 8 * max_vars)
			d_ctrl[0] = 1;
		else {
			d_done[mod] = 1;
			d_ctrl[1] = 1;
		}
	}
}

// d_ctrl is the variable which is set to 0 if all the models are completed hence stop and 1 if not.
// Loop checker of larsen.
void h_check(int *d_ctrl, int *d_step, int *d_nVars,  int *d_done, int M, int Z, int num_models) {
	dim3 block_dim(1024);
	dim3 grid_dim((num_models + block_dim.x - 1) / block_dim.x);
	d_check<<<grid_dim, block_dim>>>(d_ctrl, d_step, d_nVars, d_done, M, Z, num_models);
}

template<class T>
__global__ void d_corr(T *d_X, T *d_Y, T *d_mu, T *d_c, T *d_, int *d_act, int *d_done, int M, int Z, int num_models)
{
	int	ind = threadIdx.x + blockIdx.x * blockDim.x;
	int	mod = threadIdx.y + blockIdx.y * blockDim.y;
	if (mod < num_models && !d_done[mod]) {
		if (ind < Z - 1) {
			int	i = ind;
			int act = d_act[mod];
			if (i >= act)
				i++;
			T tot = 0;
			for (int j = 0; j < M - 1; j++)
				tot += d_X[j * Z + i] * (d_Y[j * Z + act] - d_mu[mod * (M - 1) + j]);
			d_c[mod * (Z - 1) + ind] = d_[mod * (Z - 1) + ind] = tot;
		}
	}
}

// c = _ = X' * (y - mu).
template<class T>
void h_corr(T *d_X, T *d_Y, T *d_mu, T *d_c, T *d_, int *d_act, int *d_done, int M, int Z, int num_models) {
	dim3 block_dim(32, 32);
	dim3 grid_dim((Z - 1 + block_dim.x - 1) / block_dim.x,
		(num_models + block_dim.y - 1) / block_dim.y);
	d_corr<T><<<grid_dim, block_dim>>>(d_X, d_Y, d_mu, d_c, d_, d_act, d_done, M, Z, num_models);
}

template<class T>
__global__ void d_exc_corr(T *d_, int *d_lVars, int *d_nVars, int *d_done, int M, int Z, int num_models) {
	int	ind = threadIdx.x + blockIdx.x * blockDim.x;
	int	mod = threadIdx.y + blockIdx.y * blockDim.y;
	if (mod < num_models && !d_done[mod]) {
		int	max_vars = min(M - 1, Z - 1);
		if (ind < d_nVars[mod]) {
			int	i = d_lVars[mod * max_vars + ind];
			d_[mod * (Z - 1) + i] = 0;
		}
	}
}

// Exclude Active variables in _.
template<class T>
void h_exc_corr(T *d_, int *d_lVars, int *d_nVars, int *d_done, int M, int Z, int num_models) {
	int max_vars = min(M - 1, Z - 1);
	dim3 block_dim(32, 32);
	dim3 grid_dim((max_vars + block_dim.x - 1) / block_dim.x,
		(num_models + block_dim.y - 1) / block_dim.y);
	d_exc_corr<T><<<grid_dim, block_dim>>>(d_, d_lVars, d_nVars, d_done, M, Z, num_models);
}

template<class T>
__global__ void d_max_corr(T *d_, T *d_cmax, int *d_ind, int *d_done, int Z, int num_models) {
	int	mod = threadIdx.x + blockIdx.x * blockDim.x;
	if (mod < num_models && !d_done[mod]) {
		int maxi = -1;
		T max = -INF;
		for (int j = 0; j < Z - 1; j++) {
			T tot = fabs(d_[mod * (Z - 1) + j]);
			if(tot > max) {
				max = tot;
				maxi = j;
			}
		}
		d_cmax[mod] = max;
		d_ind[mod] = maxi;
	}
}

// Take max(abs(_(I))).
template<class T>
void h_max_corr(T *d_, T *d_cmax, int *d_ind, int *d_done, int Z, int num_models) {
	dim3 block_dim(1024);
	dim3 grid_dim((num_models + block_dim.x - 1) / block_dim.x);
	d_max_corr<T><<<grid_dim, block_dim>>>(d_, d_cmax, d_ind, d_done, Z, num_models);
}

__global__ void d_lasso_add(int *d_ind, int *d_lVars, int *d_nVars, int *d_lasso, int *d_done,
	int M, int Z, int num_models)
{
	int mod = threadIdx.x + blockIdx.x * blockDim.x;
	if (mod < num_models && !d_done[mod]) {
		int	max_vars = min(M - 1, Z - 1);
		if (!d_lasso[mod]) {
			d_lVars[mod * max_vars + d_nVars[mod]] = d_ind[mod];
			d_nVars[mod] += 1;
		}
		else
			d_lasso[mod] = 0;
	}
}

// Add cmax to active set.
void h_lasso_add(int *d_ind, int *d_lVars, int *d_nVars, int *d_lasso, int *d_done,
	int M, int Z, int num_models)
{
	dim3 block_dim(1024);
	dim3 grid_dim((num_models + block_dim.x - 1) / block_dim.x);
	d_lasso_add<<<grid_dim, block_dim>>>(d_ind, d_lVars, d_nVars, d_lasso, d_done, M, Z, num_models);
}

template<class T>
__global__ void d_xincty(T *d_X, T *d_Y, T *d_, int *d_lVars, int *d_nVars, int *d_act, int *d_done,
	int M, int Z, int num_models)
{
	int	ind = threadIdx.x + blockIdx.x * blockDim.x;
	int	mod = threadIdx.y + blockIdx.y * blockDim.y;
	if (mod < num_models && !d_done[mod]) {
		int	max_vars = min(M - 1, Z - 1);
		if (ind < d_nVars[mod]) {
			int i = d_lVars[mod * max_vars + ind];
			int act = d_act[mod];
			if (i >= act)
				i++;
			T tot = 0;
			for (int j = 0; j < M - 1; j++)
				tot += d_X[j * Z + i] * d_Y[j * Z + act];
			d_[mod * max_vars + ind] = tot;
		}
	}
}

// Compute _ = X(:, A)' * y.
template<class T>
void h_xincty(T *d_X, T *d_Y, T *d_, int *d_lVars, int *d_nVars, int *d_act, int *d_done,
	int M, int Z, int num_models)
{
	int max_vars = min(M - 1, Z - 1);
	dim3 block_dim(32, 32);
	dim3 grid_dim((max_vars + block_dim.x - 1) / block_dim.x,
		(num_models + block_dim.y - 1) / block_dim.y);
	d_xincty<T><<<grid_dim, block_dim>>>(d_X, d_Y, d_, d_lVars, d_nVars, d_act, d_done, M, Z, num_models);
}

template<class T>
__global__ void d_set_gram(T *d_X, T *d_G, T *d_I, int *d_lVars, int *d_act, int ni,
	int M, int Z, int mod, int num_models)
{
	int	indx = threadIdx.x;
	int	indy = blockIdx.x;
	if (indx < ni && indy < ni && indx <= indy) {
		int max_vars = min(M - 1, Z - 1);
		int act = d_act[mod];
		int i = d_lVars[mod * max_vars + indx];
		int j = d_lVars[mod * max_vars + indy];
		if (i >= act)
			i++;
		if (j >= act)
			j++;
		T tot = 0;
		for (int k = 0; k < M - 1; k++)
			tot += d_X[k * Z + i] * d_X[k * Z + j];
		if (indx == indy) {
			d_G[indx * ni + indy] = tot;
			d_I[indx * ni + indy] = 1;
		}
		else {
			d_G[indx * ni + indy] = d_G[indy * ni + indx] = tot;
			d_I[indx * ni + indy] = d_I[indy * ni + indx] = 0;
		}
	}
}

// Compute G = X(:, A)' * X(:, A).
template<class T>
void h_set_gram(T *d_X, T *d_G, T *d_I, int *d_lVars, int *d_act, int ni,
	int M, int Z, int mod, int num_models)
{
	d_set_gram<T><<<ni, ni>>>(d_X, d_G, d_I, d_lVars, d_act, ni, M, Z, mod, num_models);
}

template<class T>
__global__ void d_betaols(T *d_I, T *d_, T *d_betaOLS,
	int ni, int mod, int M, int Z, int num_models)
{
	int	ind = threadIdx.x;
	if (ind < ni) {
		int max_vars = min(M - 1, Z - 1);
		T tot = 0;
		for (int j = 0; j < ni; j++)
			tot += d_I[ind * ni + j] * d_[mod * max_vars + j];
		d_betaOLS[mod * max_vars + ind] = tot;
	}
}

// Compute betaOLS = I * _.
template<class T>
void h_betaols(T *d_I, T *d_, T *d_betaOLS,
	int ni, int mod, int M, int Z, int num_models)
{
	d_betaols<T><<<1, ni>>>(d_I, d_, d_betaOLS, ni, mod, M, Z, num_models);
}

template<class T>
__global__ void d_dgamma(T *d_X, T *d_betaOLS, T *d_mu, T *d_d, T *d_beta, T *d_gamma,
	int *d_lVars, int *d_nVars, int *d_act, int *d_done,
	int M, int Z, int num_models)
{
	int	ind = threadIdx.x + blockIdx.x * blockDim.x;
	int mod = threadIdx.y + blockIdx.y * blockDim.y;
	if (mod < num_models && !d_done[mod]) {
		if (ind < M - 1) {
			int max_vars = min(M - 1, Z - 1);
			int ni = d_nVars[mod];
			int act = d_act[mod];
			T tot = 0;
			for (int j = 0; j < ni; j++) {
				int i = d_lVars[mod * max_vars + j];
				if (i >= act)
					i++;
				tot += d_X[ind * Z + i] * d_betaOLS[mod * max_vars + j];
			}
			d_d[mod * (M - 1) + ind] = tot - d_mu[mod * (M - 1) + ind];
			if (ind < ni - 1) {
				int i = d_lVars[mod * max_vars + ind];
				tot = d_beta[mod * (Z - 1) + i] / (d_beta[mod * (Z - 1) + i] - d_betaOLS[mod * max_vars + ind]);
             	if (tot <= 0)
             		tot = INF;
                d_gamma[mod * max_vars + ind] = tot;
			}
		}
	}
}

// Computing d = X(:, A) * betaOLS - mu and gamma list.
template<class T>
void h_dgamma(T *d_X, T *d_betaOLS, T *d_mu, T *d_d, T *d_beta, T *d_gamma,
	int *d_lVars, int *d_nVars, int *d_act, int *d_done,
	int M, int Z, int num_models)
{
	dim3 block_dim(32, 32);
	dim3 grid_dim((M - 1 + block_dim.x - 1) / block_dim.x,
		(num_models + block_dim.y - 1) / block_dim.y);
	d_dgamma<T><<<grid_dim, block_dim>>>(d_X, d_betaOLS, d_mu, d_d, d_beta, d_gamma, d_lVars, d_nVars, d_act, d_done, M, Z, num_models);
}

template<class T>
__global__ void d_min_gamma(T *d_gamma, int *d_ind, int *d_nVars, int *d_done,
	int M, int Z, int num_models)
{
	int mod = threadIdx.x + blockIdx.x * blockDim.x;
	if (mod < num_models && !d_done[mod]) {
		int	ni, mini, max_vars = min(M - 1, Z - 1);
		T min;
		ni = d_nVars[mod];
		min = INF;
		mini = -1;
		for (int j = 0; j < ni - 1; j++) {
			T tot = d_gamma[mod * max_vars + j];
			if (tot < min) {
				min = tot;
				mini = j;
			}
		}
		d_gamma[mod] = min;
		d_ind[mod] = mini;
	}
}

// Computing min(gamma(gamma > 0)).
template<class T>
void h_min_gamma(T *d_gamma, int *d_ind, int *d_nVars, int *d_done,
	int M, int Z, int num_models)
{
	dim3 block_dim(1024);
	dim3 grid_dim((num_models + block_dim.x - 1) / block_dim.x);
	d_min_gamma<T><<<grid_dim, block_dim>>>(d_gamma, d_ind, d_nVars, d_done, M, Z, num_models);
}

template<class T>
__global__ void d_xtd(T *d_X, T *d_d, T *d_, T *d_c, T *d_cmax, int *d_act, int *d_done,
	int M, int Z, int num_models)
{
	int ind = threadIdx.x + blockIdx.x * blockDim.x;
	int mod = threadIdx.y + blockIdx.y * blockDim.y;
	if (mod < num_models && !d_done[mod]) {
		if (ind < Z - 1) {
			T cmax = d_cmax[mod];
			int act = d_act[mod];
			int i = ind;
			if (i >= act)
				i++;
			T tot = 0;
			for (int j = 0; j < M - 1; j++)
				tot += d_X[j * Z + i] * d_d[mod * (M - 1) + j];
			T a = (d_c[mod * (Z - 1) + ind] + cmax) / (tot + cmax);
			T b = (d_c[mod * (Z - 1) + ind] - cmax) / (tot - cmax);
			if (a <= 0)
				a = INF;
			if (b <= 0)
				b = INF;
			tot = min(a, b);
			d_[mod * (Z - 1) + ind] = tot;
		}
	}
}

// Computing _ = X' * d.
template<class T>
void h_xtd(T *d_X, T *d_d, T *d_, T *d_c, T *d_cmax, int *d_act, int *d_done,
	int M, int Z, int num_models)
{
	dim3 block_dim(32, 32);
	dim3 grid_dim((Z - 1 + block_dim.x - 1) / block_dim.x,
		(num_models + block_dim.y - 1) / block_dim.y);
	d_xtd<T><<<grid_dim, block_dim>>>(d_X, d_d, d_, d_c, d_cmax, d_act, d_done, M, Z, num_models);
}

template<class T>
__global__ void d_exc_tmp(T *d_, int *d_lVars, int *d_nVars, int *d_done,
	int M, int Z, int num_models)
{
	int ind = threadIdx.x + blockIdx.x * blockDim.x;
	int mod = threadIdx.y + blockIdx.y * blockDim.y;
    if (mod < num_models && !d_done[mod]) {
		int max_vars = min(M - 1, Z - 1);
		int ni = d_nVars[mod];
       	if (ind < ni) {
			int i = d_lVars[mod * max_vars + ind];
			d_[mod * (Z - 1) + i] = INF;
		}
	}
}

// Excluding active variables from gamma_tilde.
template<class T>
void h_exc_tmp(T *d_, int *d_lVars, int *d_nVars, int *d_done,
	int M, int Z, int num_models)
{
	int max_vars = min(M - 1, Z - 1);
	dim3 block_dim(32, 32);
	dim3 grid_dim((max_vars + block_dim.x - 1) / block_dim.x,
		(num_models + block_dim.y - 1) / block_dim.y);
	d_exc_tmp<T><<<grid_dim, block_dim>>>(d_, d_lVars, d_nVars, d_done, M, Z, num_models);
}

template<class T>
__global__ void d_min_tmp(T *d_, int *d_done, int Z, int num_models) {
	int mod = threadIdx.x + blockIdx.x * blockDim.x;
   	if (mod < num_models && !d_done[mod]) {
		T min = INF;
        for (int j = 0; j < Z - 1; j++) {
            T tot = d_[mod * (Z - 1) + j];
            if (tot < min)
              	min = tot;
        }
        d_[mod] = min;
    }
}

// Finding gamma = min(gamma_tilde(gamma_tilde > 0)).
template<class T>
void h_min_tmp(T *d_, int *d_done, int Z, int num_models) {
	dim3 block_dim(1024);
	dim3 grid_dim((num_models + block_dim.x - 1) / block_dim.x);
	d_min_tmp<T><<<grid_dim, block_dim>>>(d_, d_done, Z, num_models);
}

template<class T>
__global__ void d_lasso_dev(T *d_, T *d_gamma, int *d_nVars,
	int *d_lasso, int *d_done, int M, int Z, int num_models)
{
	int mod = threadIdx.x + blockIdx.x * blockDim.x;
	int max_vars = min(M - 1, Z - 1);
    if (mod < num_models && !d_done[mod]) {
		if (d_nVars[mod] == max_vars) {
			if (d_gamma[mod] < 1)
				d_lasso[mod] = 1;
			else
				d_gamma[mod] = 1;
		}
		else {
			if (d_gamma[mod] < d_[mod])
				d_lasso[mod] = 1;
			else
				d_gamma[mod] = d_[mod];
		}
	}
}

// Lasso deviation condition.
template<class T>
void h_lasso_dev(T *d_, T *d_gamma, int *d_nVars,
	int *d_lasso, int *d_done, int M, int Z, int num_models)
{
	dim3 block_dim(1024);
	dim3 grid_dim((num_models + block_dim.x - 1) / block_dim.x);
	d_lasso_dev<T><<<grid_dim, block_dim>>>(d_, d_gamma, d_nVars, d_lasso, d_done, M, Z, num_models);
}

template<class T>
__global__ void d_update(T *d_gamma, T *d_mu, T *d_beta, T *d_betaOLS, T *d_d,
	int *d_lVars, int *d_nVars, int *d_done, int M, int Z, int num_models)
{
	int	mi = threadIdx.x + blockIdx.x * blockDim.x;
	int mod = threadIdx.y + blockIdx.y * blockDim.y;
	if (mod < num_models && !d_done[mod]) {
		if (mi < M - 1) {
			int	max_vars = min(M - 1, Z - 1);
			T gamma = d_gamma[mod];
			d_mu[mod * (M - 1) + mi] += gamma * d_d[mod * (M - 1) + mi];
			if (mi < d_nVars[mod]) {
				int i = d_lVars[mod * max_vars + mi];
				d_beta[mod * (Z - 1) + i] += gamma * (d_betaOLS[mod * max_vars + mi] - d_beta[mod * (Z - 1) + i]);
			}
		}
	}
}

// Updates mu and beta.
template<class T>
void h_update(T *d_gamma, T *d_mu, T *d_beta, T *d_betaOLS, T *d_d,
	int *d_lVars, int *d_nVars, int *d_done, int M, int Z, int num_models)
{
	dim3 block_dim(32, 32);
	dim3 grid_dim((M - 1 + block_dim.x - 1) / block_dim.x,
		(num_models + block_dim.y - 1) / block_dim.y);
	d_update<T><<<grid_dim, block_dim>>>(d_gamma, d_mu, d_beta, d_betaOLS, d_d,
		d_lVars, d_nVars, d_done, M, Z, num_models);
}

__global__ void d_lasso_drop(int *d_ind, int *d_lVars, int *d_nVars,
	int *d_lasso, int *d_done, int M, int Z, int num_models)
{
	int ind = threadIdx.x;
	int mod = blockIdx.x;
	if (mod < num_models && !d_done[mod]) {
		if (d_lasso[mod]) {
			int	st, max_vars = min(M - 1, Z - 1);
			st = d_ind[mod];
			if (ind < d_nVars[mod] - 1 && ind >= st) {
				int tmp = d_lVars[mod * max_vars + ind + 1];
				__syncthreads();
				d_lVars[mod * max_vars + ind] = tmp;
			}
			if (ind == 0)
				d_nVars[mod] -= 1;
		}
	}
}

// Drops deviated lasso variable.
void h_lasso_drop(int *d_ind, int *d_lVars, int *d_nVars,
	int *d_lasso, int *d_done, int M, int Z, int num_models)
{
	int max_vars = min(M - 1, Z - 1);
	d_lasso_drop<<<num_models, max_vars>>>(d_ind, d_lVars, d_nVars, d_lasso, d_done, M, Z, num_models);
}

template<class T>
__global__ void d_ress(T *d_Y, T *d_mu, T *d_,
	int *d_act, int *d_done, int M, int Z, int num_models)
{
	int	mi = threadIdx.x + blockIdx.x * blockDim.x;
	int	mod = threadIdx.y + blockIdx.y * blockDim.y;
	if (mod < num_models && !d_done[mod]) {
		if (mi < M - 1) {
			int	act = d_act[mod];
			d_[mod * (M - 1) + mi] = d_mu[mod * (M - 1) + mi] - d_Y[mi * Z + act];
		}
	}
}

// Computing _ = mu - y.
template<class T>
void h_ress(T *d_Y, T *d_mu, T *d_,
	int *d_act, int *d_done, int M, int Z, int num_models)
{
	dim3 block_dim(32, 32);
	dim3 grid_dim((M - 1 + block_dim.x - 1) / block_dim.x,
		(num_models + block_dim.y - 1) / block_dim.y);
	d_ress<T><<<grid_dim, block_dim>>>(d_Y, d_mu, d_, d_act, d_done, M, Z, num_models);
}

template<class T>
__global__ void d_final(T *d_, T *d_beta, T *d_upper1,
	T *d_normb, int *d_nVars, int *d_step, int *d_done, int *d_ctrl,
	int M, int Z, int num_models, T g)
{
	int mod = threadIdx.x + blockIdx.x * blockDim.x;
	if (mod < num_models && !d_done[mod]) {
		T upper1 = 0, normb = 0;
		for (int i = 0; i < Z - 1; i++)
			normb += fabs(d_beta[mod * (Z - 1) + i]);
		for (int i = 0; i < M - 1; i++) {
			T val = d_[mod * (M - 1) + i];
			upper1 += val * val;
		}
		upper1 = sqrt(upper1);
		if (d_step[mod] > 1) {
			T G = -(d_upper1[mod] - upper1) / (d_normb[mod] - normb);
			if (G < g) {
				d_done[mod] = 1;
				d_ctrl[1] = 1;
			}
		}
		d_upper1[mod] = upper1;
		d_normb[mod] = normb;
		d_step[mod] += 1;
	}
}

// Computes G and breaking condition.
template<class T>
void h_final(T *d_, T *d_beta, T *d_upper1,
	T *d_normb, int *d_nVars, int *d_step, int *d_done, int *d_ctrl,
	int M, int Z, int num_models, T g)
{
	dim3 block_dim(1024);
	dim3 grid_dim((num_models + block_dim.x - 1) / block_dim.x);
	d_final<T><<<grid_dim, block_dim>>>(d_, d_beta, d_upper1, d_normb, d_nVars, d_step,
		d_done, d_ctrl, M, Z, num_models, g);
}

template<class T>
__global__ void d_clear(T *d_beta, T *d_mu, int *d_done, int M, int Z, int num_models) {
	int zi = threadIdx.x + blockIdx.x * blockDim.x;
	int mod = threadIdx.y + blockIdx.y * blockDim.y;
	if (mod < num_models && d_done[mod]) {
		if (zi < Z - 1)
			d_beta[mod * (Z - 1) + zi] = 0;
		if (zi < M - 1)
			d_mu[mod * (M - 1) + zi] = 0;
	}
}

template<class T>
void h_clear(T *d_beta, T *d_mu, int *d_done, int M, int Z, int num_models) {
	dim3 block_dim(32, 32);
	dim3 grid_dim((Z - 1 + block_dim.x - 1) / block_dim.x,
		(num_models + block_dim.y - 1) / block_dim.y);
	d_clear<T><<<grid_dim, block_dim>>>(d_beta, d_mu, d_done, M, Z, num_models);
}

#endif