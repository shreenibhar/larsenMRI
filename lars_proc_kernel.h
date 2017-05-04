#ifndef LARS_PROC_KERNEL
#define LARS_PROC_KERNEL

#include <bits/stdc++.h>
#include <cuda_runtime.h>

#include "utilities.h"

#define INF 50000

using namespace std;

template<class T>
__global__ void d_init_full(dmatrix<T> mat, T val) {
	int col = threadIdx.x + blockIdx.x * blockDim.x;
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	if (row < mat.M && col < mat.N)
		mat.set(row, col, val);
}

template<class T>
void h_init_full(dmatrix<T> mat, T val) {
	dim3 blockDim(32, 32);
	dim3 gridDim((mat.N + blockDim.x - 1) / blockDim.x,
		(mat.M + blockDim.y - 1) / blockDim.y);
	d_init_full<T><<<gridDim, blockDim>>>(mat, val);
}

template<class T>
__global__ void d_init_y(dmatrix<T> y, dmatrix<T> Yt, int mod, int act) {
	int ind = threadIdx.x + blockIdx.x * blockDim.x;
	if (ind < y.N)
		y.set(mod, ind, Yt.get(act, ind));
}

template<class T>
void h_init_y(dmatrix<T> y, dmatrix<T> Yt, int mod, int act) {
	int occup = (y.N < 1024)? y.N: 1024;
	dim3 blockDim(occup);
	dim3 gridDim((y.N + blockDim.x - 1) / blockDim.x);
	d_init_y<T><<<gridDim, blockDim>>>(y, Yt, mod, act);
}

template<class T>
__global__ void d_reset(dmatrix<T> y, dmatrix<T> Yt, dmatrix<T> mu, dmatrix<T> beta, dmatrix<bool> maskVars,
	dmatrix<int> act, dmatrix<int> nVars, dmatrix<int> lasso, dmatrix<int> step, dmatrix<int> done)
{
	int ind = threadIdx.x + blockIdx.x * blockDim.x;
	int mod = threadIdx.y + blockIdx.y * blockDim.y;
	if (mod < mu.M && done.get(0, mod) == 2) {
		int hact = act.get(0, mod);
		if (ind == 0) {
			nVars.set(0, mod, 0);
			lasso.set(0, mod, 0);
			step.set(0, mod, 0);
			done.set(0, mod, 0);
		}
		if (ind < maskVars.N)
			maskVars.set(mod, ind, false);
		if (ind < mu.N)
			mu.set(mod, ind, 0);
		if (ind < beta.N)
			beta.set(mod, ind, 0);
		if (ind < y.N)
			y.set(mod, ind, Yt.get(hact, ind));
	}
}

template<class T>
void h_reset(dmatrix<T> y, dmatrix<T> Yt, dmatrix<T> mu, dmatrix<T> beta, dmatrix<bool> maskVars,
	dmatrix<int> act, dmatrix<int> nVars, dmatrix<int> lasso, dmatrix<int> step, dmatrix<int> done)
{
	dim3 blockDim(32, 32);
	dim3 gridDim((beta.N + blockDim.x - 1) / blockDim.x,
		(beta.M + blockDim.y - 1) / blockDim.y);
	d_reset<T><<<gridDim, blockDim>>>(y, Yt, mu, beta, maskVars, act, nVars, lasso, step, done);
}

template<class T>
__global__ void d_transpose(dmatrix<T> odata, dmatrix<T> idata) {
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	if (row < idata.M && col < idata.N)
		odata.set(col, row, idata.get(row, col));
}

template<class T>
void h_transpose(dmatrix<T> odata, dmatrix<T> idata) {
	dim3 blockDim(32, 32);
	dim3 gridDim((idata.N + blockDim.x - 1) / blockDim.x,
		(idata.M + blockDim.y - 1) / blockDim.y);
	d_transpose<T><<<gridDim, blockDim>>>(odata, idata);
}

template<class T>
__global__ void d_copy(dmatrix<T> to, dmatrix<T> from) {
	int col = threadIdx.x + blockIdx.x * blockDim.x;
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	if (row < from.M && col < from.N)
		to.set(row, col, from.get(row, col));
}

template<class T>
void h_copy(dmatrix<T> to, dmatrix<T> from) {
	if (to.M != from.M || to.N != from.N)
		throw "Dimension mismatch\n";
	dim3 blockDim(32, 32);
	dim3 gridDim((from.N + blockDim.x - 1) / blockDim.x,
		(from.M + blockDim.y - 1) / blockDim.y);
	d_copy<T><<<gridDim, blockDim>>>(to, from);
}

__global__ void d_check(dmatrix<int> step, dmatrix<int> nVars, dmatrix<int> done, int max_vars)
{
	int mod = threadIdx.x + blockIdx.x * blockDim.x;
	if (mod < step.M && !done.get(0, mod)) {
		if (nVars.get(0, mod) < max_vars && step.get(0, mod) < 8 * max_vars) {
			// NOP
		}
		else {
			done.set(0, mod, 1);
		}
	}
}

// Loop checker of larsen.
void h_check(dmatrix<int> step, dmatrix<int> nVars, dmatrix<int> done, int max_vars)
{
	int occup = (step.M < 1024)? step.M: 1024;
	dim3 blockDim(occup);
	dim3 gridDim((step.M + blockDim.x - 1) / blockDim.x);
	d_check<<<gridDim, blockDim>>>(step, nVars, done, max_vars);
}

template<class T>
__global__ void d_corr(dmatrix<T> Xt, dmatrix<T> y, dmatrix<T> mu, dmatrix<T> c,
	dmatrix<int> act, dmatrix<int> done, dmatrix<T> flop)
{
	const int TILE_DIM = 16;
	int col = threadIdx.x + blockIdx.x * TILE_DIM;
	int row = threadIdx.y + blockIdx.y * TILE_DIM;
	int mod = blockIdx.z;

	__shared__ T As[TILE_DIM][TILE_DIM];
	__shared__ T Bs[TILE_DIM];

	T CValue = 0;

	int ARows = c.N;
	int ACols = Xt.N;
	int BRows = y.N;
	int BCols = 1;

	if (mod < c.M && !done.get(0, mod)) {
		if (row == 0 && col == 0) {
			flop.set(mod, 0, Xt.M * Xt.N * 2 + flop.get(mod, 0));
		}

		int hact = act.get(0, mod);

		for (int k = 0; k < (TILE_DIM + ACols - 1) / TILE_DIM; k++) {
			if (k * TILE_DIM + threadIdx.x < ACols && row < ARows)
				As[threadIdx.y][threadIdx.x] = Xt.getmodt(row, k * TILE_DIM + threadIdx.x, hact);
			else
				As[threadIdx.y][threadIdx.x] = 0;
			if (k * TILE_DIM + threadIdx.y < BRows && col < BCols)
				Bs[threadIdx.y] = y.get(mod, k * TILE_DIM + threadIdx.y) - mu.get(mod, k * TILE_DIM + threadIdx.y);
			else
				Bs[threadIdx.y] = 0;
			__syncthreads();
			
			for (int n = 0; n < TILE_DIM; n++)
				CValue += As[threadIdx.y][n] * Bs[n];
			__syncthreads();
		}

		if (row < ARows && col < BCols) {
			c.set(mod, row, CValue);
		}
	}
}

// c = X' * __.
template<class T>
void h_corr(dmatrix<T> Xt, dmatrix<T> y, dmatrix<T> mu, dmatrix<T> c,
	dmatrix<int> act, dmatrix<int> done, dmatrix<T> flop)
{
	int TILE_DIM = 16;
	dim3 blockDim(TILE_DIM, TILE_DIM, 1);
	dim3 gridDim(1, (c.N + blockDim.y - 1) / blockDim.y, c.M);
	d_corr<T><<<gridDim, blockDim>>>(Xt, y, mu, c, act, done, flop);
}

template<class T>
__global__ void d_max_corr(dmatrix<T> c, dmatrix<T> cmax,
	dmatrix<int> ind, dmatrix<bool> maskVars, dmatrix<int> done)
{
	int mod = threadIdx.x + blockIdx.x * blockDim.x;
	if (mod < c.M && !done.get(0, mod)) {
		int maxi = -1;
		T max = -INF;
		for (int j = 0; j < c.N; j++) {
			if (maskVars.get(mod, j))
				continue;
			T tot = fabs(c.get(mod, j));
			if(tot > max) {
				max = tot;
				maxi = j;
			}
		}
		cmax.set(0, mod, max);
		ind.set(0, mod, maxi);
	}
}

// Take max(abs(c(I))).
template<class T>
void h_max_corr(dmatrix<T> c, dmatrix<T> cmax,
	dmatrix<int> ind, dmatrix<bool> maskVars, dmatrix<int> done)
{
	int occup = (c.M < 1024)? c.M: 1024;
	dim3 blockDim(occup);
	dim3 gridDim((c.M + blockDim.x - 1) / blockDim.x);
	d_max_corr<T><<<gridDim, blockDim>>>(c, cmax, ind, maskVars, done);
}

template<class T>
__global__ void d_lasso_add(dmatrix<int> ind, dmatrix<int> lVars, dmatrix<int> nVars, dmatrix<bool> maskVars, dmatrix<int> lasso, dmatrix<int> done, int max_vars, dmatrix<T> flop)
{
	int mod = threadIdx.x + blockIdx.x * blockDim.x;
	if (mod < maskVars.M && !done.get(0, mod)) {
		if (!lasso.get(0, mod)) {
			int hind = ind.get(0, mod);
			int ni = nVars.get(0, mod);

			flop.set(mod, 2, 3 * (ni + 1) * (ni + 1) + flop.get(mod, 2));

			lVars.set(mod, ni, hind);
			maskVars.set(mod, hind, true);
			nVars.set(0, mod, ni + 1);
		}
		else
			lasso.set(0, mod, 0);
	}
}

// Add cmax to active set.
template<class T>
void h_lasso_add(dmatrix<int> ind, dmatrix<int> lVars, dmatrix<int> nVars, dmatrix<bool> maskVars, dmatrix<int> lasso, dmatrix<int> done, int max_vars, dmatrix<T> flop)
{
	int occup = (maskVars.M < 1024)? maskVars.M: 1024;
	dim3 blockDim(occup);
	dim3 gridDim((maskVars.M + blockDim.x - 1) / blockDim.x);
	d_lasso_add<<<gridDim, blockDim>>>(ind, lVars, nVars, maskVars, lasso, done, max_vars, flop);
}

template<class T>
__global__ void d_xincty(dmatrix<T> Xt, dmatrix<T> y, dmatrix<T> __,
	dmatrix<int> lVars, dmatrix<int> nVars, dmatrix<int> act, dmatrix<int> done, dmatrix<T> flop)
{
	const int TILE_DIM = 16;
	int col = threadIdx.x + blockIdx.x * TILE_DIM;
	int row = threadIdx.y + blockIdx.y * TILE_DIM;
	int mod = blockIdx.z;

	__shared__ T As[TILE_DIM][TILE_DIM];
	__shared__ T Bs[TILE_DIM];

	T CValue = 0;

	if (mod < y.M && !done.get(0, mod)) {
		int ni = nVars.get(0, mod);
		int ARows = ni;
		int ACols = Xt.N;
		int BRows = y.N;
		int BCols = 1;
		int hact = act.get(0, mod);

		if (row == 0 && col == 0) {
			flop.set(mod, 1, ni * Xt.N * 2 + 2 * ni * ni + flop.get(mod, 1));
		}

		for (int k = 0; k < (TILE_DIM + ACols - 1) / TILE_DIM; k++) {
			if (k * TILE_DIM + threadIdx.x < ACols && row < ARows)
				As[threadIdx.y][threadIdx.x] = Xt.getmodt(lVars.get(mod, row), k * TILE_DIM + threadIdx.x, hact);
			else
				As[threadIdx.y][threadIdx.x] = 0;
			if (k * TILE_DIM + threadIdx.y < BRows && col < BCols)
				Bs[threadIdx.y] = y.get(mod, k * TILE_DIM + threadIdx.y);
			else
				Bs[threadIdx.y] = 0;
			__syncthreads();
			
			for (int n = 0; n < TILE_DIM; n++)
				CValue += As[threadIdx.y][n] * Bs[n];
			__syncthreads();
		}

		if (row < ARows && col < BCols) {
			__.set(mod, row, CValue);
		}
	}
}

// Compute __ = X(:, A)' * y.
template<class T>
void h_xincty(dmatrix<T> Xt, dmatrix<T> y, dmatrix<T> __,
	dmatrix<int> lVars, dmatrix<int> nVars, dmatrix<int> act, dmatrix<int> done, dmatrix<T> flop)
{
	int max_vars = y.N;
	int TILE_DIM = 16;
	dim3 blockDim(TILE_DIM, TILE_DIM, 1);
	dim3 gridDim(1, (max_vars + blockDim.y - 1) / blockDim.y, y.M);
	d_xincty<T><<<gridDim, blockDim>>>(Xt, y, __, lVars, nVars, act, done, flop); 
}

template<class T>
__global__ void d_set_gram(dmatrix<T> Xt, dmatrix<T> G,
	dmatrix<int> lVars, dmatrix<int> act, int ni, int mod, dmatrix<T> flop)
{
	const int TILE_DIM = 16;
	int col = threadIdx.x + blockIdx.x * TILE_DIM;
	int row = threadIdx.y + blockIdx.y * TILE_DIM;

	__shared__ T As[TILE_DIM][TILE_DIM];
	__shared__ T Bs[TILE_DIM][TILE_DIM];

	T CValue = 0;

	int ARows = ni;
	int ACols = Xt.N;
	int BRows = Xt.N;
	int BCols = ni;
	int hact = act.get(0, mod);

	if (row == 0 && col == 0) {
		flop.set(mod, 4, ni * ni * ni * 3 + flop.get(mod, 4));
	}

	for (int k = 0; k < (TILE_DIM + ACols - 1) / TILE_DIM; k++) {
		if (k * TILE_DIM + threadIdx.x < ACols && row < ARows)
			As[threadIdx.y][threadIdx.x] = Xt.getmodt(lVars.get(mod, row), k * TILE_DIM + threadIdx.x, hact);
		else
			As[threadIdx.y][threadIdx.x] = 0;
		if (k * TILE_DIM + threadIdx.y < BRows && col < BCols)
			Bs[threadIdx.y][threadIdx.x] = Xt.getmodt(lVars.get(mod, col), k * TILE_DIM + threadIdx.y, hact);
		else
			Bs[threadIdx.y][threadIdx.x] = 0;
		__syncthreads();
			
		for (int n = 0; n < TILE_DIM; n++)
			CValue += As[threadIdx.y][n] * Bs[n][threadIdx.x];
		__syncthreads();
	}

	if (row < ARows && col < BCols)
		G.d_mat[row * ni + col] = CValue;
}

// Compute G = X(:, A)' * X(:, A).
template<class T>
void h_set_gram(dmatrix<T> Xt, dmatrix<T> G,
	dmatrix<int> lVars, dmatrix<int> act, int ni, int mod, dmatrix<T> flop)
{
	int TILE_DIM = 16;
	int max_vars = Xt.N;
	dim3 blockDim(TILE_DIM, TILE_DIM);
	dim3 gridDim((max_vars + blockDim.x - 1) / blockDim.x,
		(max_vars + blockDim.y - 1) / blockDim.y);
	d_set_gram<T><<<gridDim, blockDim>>>(Xt, G, lVars, act, ni, mod, flop);
}

template<class T>
__global__ void d_betaols(dmatrix<T> I, dmatrix<T> __, dmatrix<T> betaOls,
	int ni, int mod)
{
	const int TILE_DIM = 16;
	int col = threadIdx.x + blockIdx.x * TILE_DIM;
	int row = threadIdx.y + blockIdx.y * TILE_DIM;

	__shared__ T As[TILE_DIM][TILE_DIM];
	__shared__ T Bs[TILE_DIM];

	T CValue = 0;

	int ARows = ni;
	int ACols = ni;
	int BRows = ni;
	int BCols = 1;

	for (int k = 0; k < (TILE_DIM + ACols - 1) / TILE_DIM; k++) {
		if (k * TILE_DIM + threadIdx.x < ACols && row < ARows)
			As[threadIdx.y][threadIdx.x] = I.d_mat[row * ni + k * TILE_DIM + threadIdx.x];
		else
			As[threadIdx.y][threadIdx.x] = 0;
		if (k * TILE_DIM + threadIdx.y < BRows && col < BCols)
			Bs[threadIdx.y] = __.get(mod, k * TILE_DIM + threadIdx.y);
		else
			Bs[threadIdx.y] = 0;
		__syncthreads();
		
		for (int n = 0; n < TILE_DIM; n++)
			CValue += As[threadIdx.y][n] * Bs[n];
		__syncthreads();
	}

	if (row < ARows && col < BCols) {
		betaOls.set(mod, row, CValue);
	}
}

// Compute betaOLS = I * __.
template<class T>
void h_betaols(dmatrix<T> I, dmatrix<T> __, dmatrix<T> betaOls,
	int ni, int mod)
{
	int TILE_DIM = 16;
	dim3 blockDim(TILE_DIM, TILE_DIM);
	dim3 gridDim(1, (ni + blockDim.y - 1) / blockDim.y);
	d_betaols<T><<<gridDim, blockDim>>>(I, __, betaOls, ni, mod);
}

template<class T>
__global__ void d_d(dmatrix<T> X, dmatrix<T> betaOls, dmatrix<T> mu, dmatrix<T> d,
	dmatrix<int> lVars, dmatrix<int> nVars, dmatrix<int> act, dmatrix<int> done, dmatrix<T> flop)
{
	const int TILE_DIM = 16;
	int col = threadIdx.x + blockIdx.x * TILE_DIM;
	int row = threadIdx.y + blockIdx.y * TILE_DIM;
	int mod = blockIdx.z;

	__shared__ T As[TILE_DIM][TILE_DIM];
	__shared__ T Bs[TILE_DIM];

	T CValue = 0;

	if (mod < mu.M && !done.get(0, mod)) {
		int ni = nVars.get(0, mod);
		int ARows = X.M;
		int ACols = ni;
		int BRows = ni;
		int BCols = 1;
		int hact = act.get(0, mod);

		if (row == 0 && col == 0) {
			flop.set(mod, 5, X.M * ni * 2 + X.M + flop.get(mod, 5));
		}

		for (int k = 0; k < (TILE_DIM + ACols - 1) / TILE_DIM; k++) {
			if (k * TILE_DIM + threadIdx.x < ACols && row < ARows)
				As[threadIdx.y][threadIdx.x] = X.getmod(row, lVars.get(mod, k * TILE_DIM + threadIdx.x), hact);
			else
				As[threadIdx.y][threadIdx.x] = 0;
			if (k * TILE_DIM + threadIdx.y < BRows && col < BCols)
				Bs[threadIdx.y] = betaOls.get(mod, k * TILE_DIM + threadIdx.y);
			else
				Bs[threadIdx.y] = 0;
			__syncthreads();
			
			for (int n = 0; n < TILE_DIM; n++)
				CValue += As[threadIdx.y][n] * Bs[n];
			__syncthreads();
		}

		if (row < ARows && col < BCols) {
			d.set(mod, row, CValue - mu.get(mod, row));
		}
	}
}

// Computing d = X(:, A) * betaOLS - mu and gamma list.
template<class T>
void h_d(dmatrix<T> X, dmatrix<T> betaOls, dmatrix<T> mu, dmatrix<T> d,
	dmatrix<int> lVars, dmatrix<int> nVars, dmatrix<int> act, dmatrix<int> done, dmatrix<T> flop)
{
	int TILE_DIM = 16;
	dim3 blockDim(TILE_DIM, TILE_DIM, 1);
	dim3 gridDim(1, (d.N + blockDim.y - 1) / blockDim.y, d.M);
	
	d_d<T><<<gridDim, blockDim>>>(X, betaOls, mu, d, lVars, nVars, act, done, flop);
}

template<class T>
__global__ void d_gammat(dmatrix<T> __, dmatrix<T> beta, dmatrix<T> betaOls,
	dmatrix<int> lVars, dmatrix<int> nVars, dmatrix<int> done, dmatrix<T> flop)
{
	int ind = threadIdx.x + blockIdx.x * blockDim.x;
	int mod = threadIdx.y + blockIdx.y * blockDim.y;
	if (mod < beta.M && !done.get(0, mod)) {
		int ni = nVars.get(0, mod);

		if (ind == 0) {
			flop.set(mod, 6, ni * 2 + flop.get(mod, 6));
		}

		if (ind < ni - 1) {
			int i = lVars.get(mod, ind);
			T tot = beta.get(mod, i) / (beta.get(mod, i) - betaOls.get(mod, ind));
			if (tot <= 0)
				tot = INF;
			__.set(mod, ind, tot);
		}
	}
}

template<class T>
void h_gammat(dmatrix<T> __, dmatrix<T> beta, dmatrix<T> betaOls,
	dmatrix<int> lVars, dmatrix<int> nVars, dmatrix<int> done, dmatrix<T> flop)
{
	int max_vars = lVars.N;
	dim3 blockDim(32, 32);
	dim3 gridDim((max_vars + blockDim.x - 1) / blockDim.x,
		(lVars.M + blockDim.y - 1) / blockDim.y);
	d_gammat<T><<<gridDim, blockDim>>>(__, beta, betaOls, lVars, nVars, done, flop);
}

template<class T>
__global__ void d_min_gammat(dmatrix<T> gamma, dmatrix<T> __,
	dmatrix<int> ind, dmatrix<int> nVars, dmatrix<int> done)
{
	int mod = threadIdx.x + blockIdx.x * blockDim.x;
	if (mod < gamma.M && !done.get(0, mod)) {
		int ni = nVars.get(0, mod);
		T min = INF;
		int mini = -1;
		for (int j = 0; j < ni - 1; j++) {
			T tot = __.get(mod, j);
			if (tot < min) {
				min = tot;
				mini = j;
			}
		}
		gamma.set(0, mod, min);
		ind.set(0, mod, mini);
	}
}

// Computing min(gamma(gamma > 0)).
template<class T>
void h_min_gammat(dmatrix<T> gamma, dmatrix<T> __,
	dmatrix<int> ind, dmatrix<int> nVars, dmatrix<int> done)
{
	int occup = (gamma.M < 1024)? gamma.M: 1024;
	dim3 blockDim(occup);
	dim3 gridDim((gamma.M + blockDim.x - 1) / blockDim.x);
	d_min_gammat<T><<<gridDim, blockDim>>>(gamma, __, ind, nVars, done);
}

template<class T>
__global__ void d_xtd(dmatrix<T> Xt, dmatrix<T> d, dmatrix<T> _, dmatrix<T> c, dmatrix<T> cmax,
	dmatrix<int> act, dmatrix<int> done, dmatrix<T> flop)
{
	const int TILE_DIM = 16;
	int col = threadIdx.x + blockIdx.x * TILE_DIM;
	int row = threadIdx.y + blockIdx.y * TILE_DIM;
	int mod = blockIdx.z;

	__shared__ T As[TILE_DIM][TILE_DIM];
	__shared__ T Bs[TILE_DIM];

	T CValue = 0;

	int ARows = _.N;
	int ACols = Xt.N;
	int BRows = d.N;
	int BCols = 1;

	if (mod < d.M && !done.get(0, mod)) {
		int hact = act.get(0, mod);

		if (row == 0 && col == 0) {
			flop.set(mod, 7, Xt.M * Xt.N * 2 + flop.get(mod, 7));
			flop.set(mod, 8, Xt.M * 6 + flop.get(mod, 8));
		}

		for (int k = 0; k < (TILE_DIM + ACols - 1) / TILE_DIM; k++) {
			if (k * TILE_DIM + threadIdx.x < ACols && row < ARows)
				As[threadIdx.y][threadIdx.x] = Xt.getmodt(row, k * TILE_DIM + threadIdx.x, hact);
			else
				As[threadIdx.y][threadIdx.x] = 0;
			if (k * TILE_DIM + threadIdx.y < BRows && col < BCols)
				Bs[threadIdx.y] = d.get(mod, k * TILE_DIM + threadIdx.y);
			else
				Bs[threadIdx.y] = 0;
			__syncthreads();
			
			for (int n = 0; n < TILE_DIM; n++)
				CValue += As[threadIdx.y][n] * Bs[n];
			__syncthreads();
		}

		if (row < ARows && col < BCols) {
			T cm = cmax.get(0, mod);
			T a = (c.get(mod, row) + cm) / (CValue + cm);
			T b = (c.get(mod, row) - cm) / (CValue - cm);
			if (a <= 0)
				a = INF;
			if (b <= 0)
				b = INF;
			CValue = min(a, b);
			_.set(mod, row, CValue);
		}
	}
}

// Computing _ = X' * d.
template<class T>
void h_xtd(dmatrix<T> Xt, dmatrix<T> d, dmatrix<T> _, dmatrix<T> c, dmatrix<T> cmax,
	dmatrix<int> act, dmatrix<int> done, dmatrix<T> flop)
{
	int TILE_DIM = 16;
	dim3 blockDim(TILE_DIM, TILE_DIM, 1);
	dim3 gridDim(1, (_.N + blockDim.x - 1) / blockDim.x, d.M);
	d_xtd<T><<<gridDim, blockDim>>>(Xt, d, _, c, cmax, act, done, flop);
}

template<class T>
__global__ void d_min_tmp(dmatrix<T> _, dmatrix<T> __,
	dmatrix<bool> maskVars, dmatrix<int> done)
{
	int mod = threadIdx.x + blockIdx.x * blockDim.x;
	if (mod < _.M && !done.get(0, mod)) {
		T min = INF;
		for (int j = 0; j < _.N; j++) {
			if (maskVars.get(mod, j))
				continue;
			T tot = _.get(mod, j);
			if (tot < min)
				min = tot;
		}
		__.set(0, mod, min);
	}
}

// Finding gamma = min(gamma_tilde(gamma_tilde > 0)).
template<class T>
void h_min_tmp(dmatrix<T> _, dmatrix<T> __,
	dmatrix<bool> maskVars, dmatrix<int> done)
{
	int occup = (_.M < 1024)? _.M: 1024;
	dim3 blockDim(occup);
	dim3 gridDim((_.M + blockDim.x - 1) / blockDim.x);
	d_min_tmp<T><<<gridDim, blockDim>>>(_, __, maskVars, done);
}

template<class T>
__global__ void d_lasso_dev(dmatrix<T> __, dmatrix<T> gamma,
	dmatrix<int> nVars, dmatrix<int> lasso, dmatrix<int> done)
{
	int mod = threadIdx.x + blockIdx.x * blockDim.x;
	int max_vars = __.N;
	if (mod < __.M && !done.get(0, mod)) {
		if (nVars.get(0, mod) == max_vars) {
			if (gamma.get(0, mod) < 1)
				lasso.set(0, mod, 1);
			else
				gamma.set(0, mod, 1);
		}
		else {
			if (gamma.get(0, mod) < __.get(0, mod))
				lasso.set(0, mod, 1);
			else
				gamma.set(0, mod, __.get(0, mod));
		}
	}
}

// Lasso deviation condition.
template<class T>
void h_lasso_dev(dmatrix<T> __, dmatrix<T> gamma,
	dmatrix<int> nVars, dmatrix<int> lasso, dmatrix<int> done)
{
	int occup = (__.M < 1024)? __.M: 1024;
	dim3 blockDim(occup);
	dim3 gridDim((__.M + blockDim.x - 1) / blockDim.x);
	d_lasso_dev<T><<<gridDim, blockDim>>>(__, gamma, nVars, lasso, done);
}

template<class T>
__global__ void d_update(dmatrix<T> gamma, dmatrix<T> mu, dmatrix<T> beta, dmatrix<T> betaOls, dmatrix<T> d,
	dmatrix<int> lVars, dmatrix<int> nVars, dmatrix<int> done, dmatrix<T> flop)
{
	int mi = threadIdx.x + blockIdx.x * blockDim.x;
	int mod = threadIdx.y + blockIdx.y * blockDim.y;
	if (mod < gamma.M && !done.get(0, mod)) {
		int ni = nVars.get(0, mod);

		if (mi == 0) {
			flop.set(mod, 9, 2 * mu.N + 3 * ni + flop.get(mod, 9));
		}

		if (mi < gamma.N) {
			T gma = gamma.get(0, mod);
			T val = gma * d.get(mod, mi) + mu.get(mod, mi);
			mu.set(mod, mi, val);
			if (mi < ni) {
				int i = lVars.get(mod, mi);
				val = gma * (betaOls.get(mod, mi) - beta.get(mod, i)) + beta.get(mod, i);
				beta.set(mod, i, val);
			}
		}
	}
}

// Updates mu and beta.
template<class T>
void h_update(dmatrix<T> gamma, dmatrix<T> mu, dmatrix<T> beta, dmatrix<T> betaOls, dmatrix<T> d,
	dmatrix<int> lVars, dmatrix<int> nVars, dmatrix<int> done, dmatrix<T> flop)
{
	dim3 blockDim(32, 32);
	dim3 gridDim((gamma.N + blockDim.x - 1) / blockDim.x,
		(gamma.M + blockDim.y - 1) / blockDim.y);
	d_update<T><<<gridDim, blockDim>>>(gamma, mu, beta, betaOls, d, lVars, nVars, done, flop);
}

template<class T>
__global__ void d_lasso_drop(dmatrix<int> ind, dmatrix<int> lVars, dmatrix<int> nVars, dmatrix<bool> maskVars,
	dmatrix<int> lasso, dmatrix<int> done, dmatrix<T> flop)
{
	int i = threadIdx.x;
	int mod = blockIdx.x;
	if (mod < lVars.M && !done.get(0, mod)) {
		if (lasso.get(0, mod)) {
			int st, ni = nVars.get(0, mod);
			st = ind.get(0, mod);
			if (i < ni - 1 && i >= st) {
				int tmp = lVars.get(mod, i + 1);
				__syncthreads();
				lVars.set(mod, i, tmp);
			}
			if (i == 0) {
				nVars.set(0, mod, ni - 1);
				maskVars.set(mod, st, false);

				flop.set(mod, 3, 4 * (ni - st) * (ni - st) + flop.get(mod, 3));
			}
		}
	}
}

// Drops deviated lasso variable.
template<class T>
void h_lasso_drop(dmatrix<int> ind, dmatrix<int> lVars, dmatrix<int> nVars, dmatrix<bool> maskVars,
	dmatrix<int> lasso, dmatrix<int> done, dmatrix<T> flop)
{
	d_lasso_drop<<<lVars.M, lVars.N>>>(ind, lVars, nVars, maskVars, lasso, done, flop);
}

template<class T>
__global__ void d_final(dmatrix<T> mu, dmatrix<T> y, dmatrix<T> beta, dmatrix<T> upper1, dmatrix<T> normb,
	dmatrix<int> nVars, dmatrix<int> step, dmatrix<int> act, dmatrix<int> done, T g, dmatrix<T> flop)
{
	int mod = threadIdx.x + blockIdx.x * blockDim.x;
	if (mod < mu.M && !done.get(0, mod)) {
		flop.set(mod, 10, beta.N * 3 + mu.N * 3 + flop.get(mod, 10));

		int hact = act.get(0, mod);
		step.set(0, mod, step.get(0, mod) + 1);
		T hupper1 = 0, hnormb = 0;
		for (int i = 0; i < beta.N; i++)
			hnormb += fabs(beta.get(mod, i));
		for (int i = 0; i < mu.N; i++) {
			T val = mu.get(mod, i) - y.get(mod, i);
			hupper1 += val * val;
		}
		hupper1 = sqrt(hupper1);
		if (step.get(0, mod) > 1) {
			T G = -(upper1.get(0, mod) - hupper1) / (normb.get(0, mod) - hnormb);
			if (G < g) {
				done.set(0, mod, 1);
				return;
			}
		}
		upper1.set(0, mod, hupper1);
		normb.set(0, mod, hnormb);
	}
}

// Computes G and breaking condition.
template<class T>
void h_final(dmatrix<T> mu, dmatrix<T> y, dmatrix<T> beta, dmatrix<T> upper1, dmatrix<T> normb,
	dmatrix<int> nVars, dmatrix<int> step, dmatrix<int> act, dmatrix<int> done, T g, dmatrix<T> flop)
{
	int occup = (mu.M < 1024)? mu.M: 1024;
	dim3 blockDim(occup);
	dim3 gridDim((mu.M + blockDim.x - 1) / blockDim.x);
	d_final<T><<<gridDim, blockDim>>>(mu, y, beta, upper1, normb, nVars, step, act, done, g, flop);
}

#endif