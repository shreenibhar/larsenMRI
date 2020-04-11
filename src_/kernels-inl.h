#ifndef KERNELS_INL_H
#define KERNELS_INL_H

#include "headers.h"


// Initializes a model's initial values.
template<typename ProcPrec>
__global__ void set_model_kernel(
  ProcPrec *Y, ProcPrec *y, ProcPrec *mu,
  ProcPrec *a1, ProcPrec *a2, ProcPrec *lambda, ProcPrec *randnrm,
  int *nVars, int *eVars, int *lasso, int *step, int *done, int *act,
  int M, int N, int bufModel, int actModel
)
{
  int ind = threadIdx.x + blockIdx.x * blockDim.x;
  mu[bufModel * M + ind] = 0;
  y[bufModel * M + ind] = Y[ind * N + actModel];
  if (ind == 0) {
    nVars[bufModel] = 0;
    eVars[bufModel] = 0;
    lasso[bufModel] = 0;
    step[bufModel] = 0;
    done[bufModel] = 0;
    act[bufModel] = actModel;
    a1[bufModel] = 0;
    a2[bufModel] = 1e6;
    lambda[bufModel] = 1e6;
    randnrm[bufModel] = 1;
  }
}


// Checks if the larsen loop has ended.
template<typename ProcPrec>
struct checkOp {
    int maxVariables, maxSteps;
    ProcPrec l1, l2, g;

    checkOp(int _maxVariables, int _maxSteps, ProcPrec _l1, ProcPrec _l2, ProcPrec _g) : maxVariables(_maxVariables), maxSteps(_maxSteps), l1(_l1), l2(_l2), g(_g) {}

    __host__ __device__
    int operator()(thrust::tuple<int, int, ProcPrec, ProcPrec, ProcPrec, int> input) {
        if (thrust::get<0>(input) < maxVariables && thrust::get<1>(input) < maxSteps && thrust::get<2>(input) < l1 && thrust::get<3>(input) > l2 && thrust::get<4>(input) > g && !thrust::get<5>(input)) return 0;
        else return 1;
    }
};


// Dropping variables if condition is met.
__global__ void drop_kernel(int *lVars, int *dropidx, int *nVars, int *lasso, int M, int numModels, int *done) {
  int mod = threadIdx.x + blockIdx.x * blockDim.x;
  if (mod < numModels && lasso[mod] && !done[mod]) {
    int ni = nVars[mod];
    int drop = dropidx[mod];
    for (int i = drop + 1; i < ni; i++) {
      int val = lVars[mod * M + i];
      lVars[mod * M + i - 1] = val;
    }
    nVars[mod] = ni - 1;
  }
}


// Compute matrix subtraction c = a - b.
template<typename ProcPrec>
__global__ void mat_sub_kernel(ProcPrec *a, ProcPrec *b, ProcPrec *c, int size) {
	int ind = threadIdx.x + blockIdx.x * blockDim.x;
	if (ind < size) {
		c[ind] = a[ind] - b[ind];
	}
}


// Wrapper for mat_sub_kernel.
template<typename ProcPrec>
void mat_sub(ProcPrec *a, ProcPrec *b, ProcPrec *c, int size) {
	dim3 blockDim(1024);
	dim3 gridDim((size + blockDim.x - 1) / blockDim.x);
	mat_sub_kernel<ProcPrec><<<gridDim, blockDim>>>(a, b, c, size);
}


// Excluding variales specified by a list.
template<typename ProcPrec>
__global__ void exclude_kernel(ProcPrec *c, int *lVars, int *nVars, int *eVars, int *act, int M, int N, int numModels, ProcPrec def, int *done) {
  int mod = threadIdx.x + blockIdx.x * blockDim.x;
  if (mod < numModels && !done[mod]) {
    int ni = nVars[mod];
    c[mod * N + act[mod]] = def; // Trick to convolve gemv into gemm (c = X' * r).
    for (int i = 0; i < ni; i++) {
      c[mod * N + lVars[mod * M + i]] = def;
    }
    // To exclude variables from eVars which is stored from the end in lVars.
    int ei = eVars[mod];
    for (int i = 0; i < ei; i++) {
      c[mod * N + lVars[mod * M + M - 1 - i]] = def;
    }
  }
}


template<typename ProcPrec>
struct absoluteOp {
  __host__ __device__
  ProcPrec operator()(ProcPrec x) {
    return abs(x);
  }
};


// Adding variables to active set.
template<typename ProcPrec>
__global__ void lasso_add_kernel(ProcPrec *c, int *lasso, int *lVars, int *nVars, int *cidx, int M, int N, int numModels, int *done) {
  int mod = threadIdx.x + blockIdx.x * blockDim.x;
  if (mod < numModels && !lasso[mod] && !done[mod]) {
    int ni = nVars[mod];
    int id = cidx[mod];
    lVars[mod * M + ni] = id;
    nVars[mod] = ni + 1;
    c[mod * N + id] = 0; // Excluding the added variable from c for future steps.
  }
}


// Used to add a new row corresponding to the new variable in X(:, A) and X1(:, A).
template<typename ProcPrec>
__global__ void gather_add_kernel(ProcPrec *XA, ProcPrec *XA1, ProcPrec *X, int *lVars, int ni, int M, int N, int mod) {
  int ind = threadIdx.x + blockIdx.x * blockDim.x;
  XA[(ni - 1) * M + ind] = XA1[(ni - 1) * M + ind] = X[ind * N + lVars[mod * M + ni - 1]];
}


// Used to delete a row from X(:, A) using the backup X1(:, A).
template<typename ProcPrec>
__global__ void gather_del_kernel(ProcPrec *XA, ProcPrec *XA1, ProcPrec *X, int ni, int drop, int M, int N, int mod) {
  int mj = threadIdx.x + blockIdx.x * blockDim.x;
  int mi = mj / M;
  mj -= mi * M;
  mi += drop;
  if (mi >= drop && mi < ni && mj < M) { // mi < ni is because ni is the nVars after deletion from active set.
    XA[mi * M + mj] = XA1[(mi + 1) * M + mj];
  }
}


// Copy to XA1 to have a backup to perform efficient deletes later.
template<typename ProcPrec>
__global__ void gather_cop_kernel(ProcPrec *XA, ProcPrec *XA1, ProcPrec *X, int ni, int drop, int M, int N, int mod) {
  int mj = threadIdx.x + blockIdx.x * blockDim.x;
  int mi = mj / M;
  mj -= mi * M;
  mi += drop;
  if (mi >= drop && mi < ni && mj < M) {
    XA1[mi * M + mj] = XA[mi * M + mj];
  }
}


// The logic related to finding X(:, A) efficiently.
template<typename ProcPrec>
void gather(ProcPrec *XA, ProcPrec *XA1, ProcPrec *X, int *lVars, int ni, int lassoCond, int drop, int M, int N, int mod, cudaStream_t &stream) {
  if (!lassoCond) {
    gather_add_kernel<ProcPrec><<<1, M, 0, stream>>>(XA, XA1, X, lVars, ni, M, N, mod);
  }
  else {
    if (ni == drop) return; // No need to do anything if the last row is dropped.
    // Shift all rows by one from the dropped row onwards.
    dim3 blockDim(min((ni - drop) * M, 1024));
    dim3 gridDim(((ni - drop) * M + blockDim.x - 1) / blockDim.x);
    gather_del_kernel<ProcPrec><<<gridDim, blockDim, 0, stream>>>(XA, XA1, X, ni, drop, M, N, mod);
    gather_cop_kernel<ProcPrec><<<gridDim, blockDim, 0, stream>>>(XA, XA1, X, ni, drop, M, N, mod);
  }
}


// Computing XA' * y.
template<typename ProcPrec>
__global__ void XAt_yBatched_kernel(ProcPrec **XA, ProcPrec *y, ProcPrec *r, int *nVars, int M, int numModels) {
  int mod = threadIdx.x + blockIdx.x * blockDim.x;
  int ind = threadIdx.y + blockIdx.y * blockDim.y;
  if (mod < numModels) {
    int ni = nVars[mod];
    extern __shared__ char buf[];
    ProcPrec *smem = reinterpret_cast<ProcPrec *>(buf);
    smem[ind] = (ind < M)? y[mod * M + ind]: 0;
    __syncthreads();
    if (ind < ni) {
      ProcPrec val = 0;
      for (int i = 0; i < M; i++) {
        val += XA[mod][ind * M + i] * smem[i];
      }
      r[mod * M + ind] = val;
    }
  }
}


// Wrapper for XAt_yBatched_kernel.
template<typename ProcPrec>
void XAt_yBatched(ProcPrec **XA, ProcPrec *y, ProcPrec *r, int *nVars, int M, int numModels) {
  dim3 blockDim(1, M);
  dim3 gridDim(numModels, 1);
  XAt_yBatched_kernel<ProcPrec><<<gridDim, blockDim, M * sizeof(ProcPrec)>>>(XA, y, r, nVars, M, numModels);
}


// Computing I * r(XA' * y).
template<typename ProcPrec>
__global__ void IrBatched_kernel(ProcPrec **I, ProcPrec *r, ProcPrec *betaOls, int *nVars, int M, int numModels) {
  int mod = threadIdx.x + blockIdx.x * blockDim.x;
  int ind = threadIdx.y + blockIdx.y * blockDim.y;
  if (mod < numModels) {
    int ni = nVars[mod];
    extern __shared__ char buf[];
    ProcPrec *smem = reinterpret_cast<ProcPrec *>(buf);
    smem[ind] = (ind < ni)? r[mod * M + ind]: 0;
    __syncthreads();
    if (ind < ni) {
      ProcPrec val = 0;
      for (int i = 0; i < ni; i++) {
        val += I[mod][ind * ni + i] * smem[i];
      }
      betaOls[mod * M + ind] = val;
    }
  }
}


// Wrapper for IrBatched_kernel.
template<typename ProcPrec>
void IrBatched(ProcPrec **I, ProcPrec *r, ProcPrec *betaOls, int *nVars, int M, int numModels, int maxVar) {
  dim3 blockDim(1, maxVar);
  dim3 gridDim(numModels, 1);
  IrBatched_kernel<ProcPrec><<<gridDim, blockDim, maxVar * sizeof(ProcPrec)>>>(I, r, betaOls, nVars, M, numModels);
}


// Computing XA * betaOls.
template<typename ProcPrec>
__global__ void XAbetaOlsBatched_kernel(ProcPrec **XA, ProcPrec *betaOls, ProcPrec *d, int *nVars, int M, int numModels) {
  int mod = threadIdx.x + blockIdx.x * blockDim.x;
  int ind = threadIdx.y + blockIdx.y * blockDim.y;
  if (mod < numModels) {
    int ni = nVars[mod];
    extern __shared__ char buf[];
    ProcPrec *smem = reinterpret_cast<ProcPrec *>(buf);
    if (ind < ni) smem[ind] = betaOls[mod * M + ind];
    __syncthreads();
    if (ind < M) {
      ProcPrec val = 0;
      for (int i = 0; i < ni; i++) {
        val += XA[mod][i * M + ind] * smem[i];
      }
      d[mod * M + ind] = val;
    }
  }
}


// Wrapper for XAbetaOlsBatched_kernel.
template<typename ProcPrec>
void XAbetaOlsBatched(ProcPrec **XA, ProcPrec *betaOls, ProcPrec *d, int *nVars, int M, int numModels, int maxVar) {
  dim3 blockDim(1, M);
  dim3 gridDim(numModels, 1);
  XAbetaOlsBatched_kernel<ProcPrec><<<gridDim, blockDim, maxVar * sizeof(ProcPrec)>>>(XA, betaOls, d, nVars, M, numModels);
}


// Check if I is well conditioned.
template<typename ProcPrec>
__global__ void checkNan_kernel(int *nVars, int *eVars, int *lVars, int *info, ProcPrec *r, ProcPrec *d, ProcPrec *randnrm, int M, int numModels, int *done) {
  int mod = threadIdx.x + blockIdx.x * blockDim.x;
  if (mod < numModels && !done[mod]) {
    int ni = nVars[mod];
    ProcPrec nrm1 = 0, nrm2 = 0;
    for (int i = 0; i < ni; i++) {
      ProcPrec val1 = r[mod * M + i];
      ProcPrec val2 = d[mod * M + i];
      nrm1 += val1 * val1;
      nrm2 += val2 * val2;
    }
    nrm1 = sqrt(nrm1);
    nrm2 = sqrt(nrm2);
    ProcPrec ratio = (nrm1 + nrm2) / randnrm[mod];
    int infer = info[mod];
    if (infer || ratio > 100 || isnan(ratio) || isinf(ratio)) {
      int ei = eVars[mod];
      lVars[mod * M + M - 1 - ei] = lVars[mod * M + ni - 1];
      nVars[mod] = ni - 1;
      eVars[mod] = ei + 1;
      infer = 1;
    }
    else randnrm[mod] = nrm1 + nrm2;
    info[mod] = infer;
  }
}


// Computing gamma tilde and dropidx.
template<typename ProcPrec>
__global__ void gammat_kernel(ProcPrec *gamma_tilde, ProcPrec *beta, ProcPrec *betaOls, int *dropidx, int *lVars, int *nVars, int *lasso, int M, int N, int numModels, int *done) {
  int mod = threadIdx.x + blockIdx.x * blockDim.x;
  if (mod < numModels && !done[mod]) {
    if (lasso[mod]) lasso[mod] = 0;
    int ni = nVars[mod];
    ProcPrec miner = inf;
    int id = -1;
    for (int i = 0; i < ni; i++) {
      int si = lVars[mod * M + i];
      ProcPrec val = beta[mod * N + si] / (beta[mod * N + si] - betaOls[mod * M + i]);
      val = (val <= 0)? inf: val;
      if (val < miner) {
        miner = val;
        id = i;
      }
    }
    gamma_tilde[mod] = miner;
    dropidx[mod] = id;
  }
}


// cd Op to find gamma.
template<typename ProcPrec>
struct cdTransform {
  __host__ __device__
  ProcPrec operator()(thrust::tuple<ProcPrec, ProcPrec, ProcPrec> x) {
    ProcPrec c_val = thrust::get<0>(x);
    ProcPrec cd_val = thrust::get<1>(x);
    ProcPrec cmax_val = thrust::get<2>(x);
    if (c_val == 0) return inf;
    ProcPrec val1 = (c_val - cmax_val) / (cd_val - cmax_val);
    ProcPrec val2 = (c_val + cmax_val) / (cd_val + cmax_val);
    val1 = (val1 <= 0)? inf: val1;
    val2 = (val2 <= 0)? inf: val2;
    return min(val1, val2);
  }
};


// Set the final value of gamma.
template<typename ProcPrec>
__global__ void set_gamma_kernel(ProcPrec *gamma, ProcPrec *gamma_tilde, int *lasso, int *nVars, int maxVariables, int M, int numModels, int *done) {
  int mod = threadIdx.x + blockIdx.x * blockDim.x;
  if (mod < numModels && !done[mod]) {
    ProcPrec gamma_t = gamma_tilde[mod];
    ProcPrec gamma_val = gamma[mod];
    if (nVars[mod] == maxVariables) {
      gamma_val = 1;
    }
    if (gamma_t < gamma_val) {
      lasso[mod] = 1;
      gamma_val = gamma_t;
    }
    gamma[mod] = gamma_val;
  }
}


// Performs final updates and L1 correction.
template<typename ProcPrec>
__global__ void update_kernel(ProcPrec *beta, ProcPrec *beta_prev, ProcPrec *mu, ProcPrec *d, ProcPrec *betaOls, ProcPrec *gamma, ProcPrec **dXA, ProcPrec *y, ProcPrec *a1, ProcPrec *a2, ProcPrec *lambda, int *lVars, int *nVars, int *step, int M, int N, int numModels, ProcPrec max_l1, int *done) {
  int mod = threadIdx.x + blockIdx.x * blockDim.x;
  if (mod < numModels && !done[mod]) {
    int ni = nVars[mod];
    ProcPrec gamma_val = gamma[mod];
    ProcPrec l1 = 0, beta_prev_l1 = 0, delta = 0;
    for (int i = 0; i < ni; i++) {
      int si = lVars[mod * M + i];
      ProcPrec beta_val = beta[mod * N + si];
      ProcPrec betaOls_val = betaOls[mod * M + i];
      beta_prev[mod * N + si] = beta_val;
      beta[mod * N + si] = beta_val + gamma_val * (betaOls_val - beta_val);
      si = (beta_val > 0) - (beta_val < 0);
      beta_prev_l1 += abs(beta_val);
      delta += si * (betaOls_val - beta_val);
      l1 += abs(beta[mod * N + si]);
    }
    if (l1 > max_l1) {
      gamma_val = (max_l1 - beta_prev_l1) / delta;
      for (int i = 0; i < ni; i++) {
        int si = lVars[mod * M + i];
        beta[mod * N + si] = beta_prev[mod * N + si] + gamma_val * (betaOls[mod * M + i] - beta_prev[mod * N + si]);
      }
      l1 = max_l1;
    }
    a1[mod] = l1;
    ProcPrec l2 = 0, g = 0;
    for (int i = 0; i < M; i++) {
      mu[mod * M + i] += gamma_val * d[mod * M + i];
      ProcPrec r = y[mod * M + i] - mu[mod * M + i];
      l2 += r * r;
      g += dXA[mod][i] * r;
    }
    a2[mod] = sqrt(l2);
    lambda[mod] = abs(g / sqrt(l2));
    step[mod] += 1;
  }
}

// __global__
// void gatherAll_kernel(corr_precision *XA, corr_precision *y, corr_precision *X, int *lVars, int ni, int M, int N, int act) {
// 	int ind = threadIdx.x + blockIdx.x * blockDim.x;
// 	int nind = ind / M;
// 	ind -= nind * M;
// 	if (nind < ni && ind < M) {
// 		if (nind == 0) {
// 			y[ind] = X[ind * N + act];
// 		}
// 		XA[nind * M + ind] = X[ind * N + lVars[nind]];
// 	}
// }

// void gatherAll(corr_precision *XA, corr_precision *y, corr_precision *X, int *lVars, int ni, int M, int N, int act, cudaStream_t &stream) {
// 	dim3 blockDim(1024);
// 	dim3 gridDim((ni * M + blockDim.x - 1) / blockDim.x);
// 	gatherAll_kernel<<<gridDim, blockDim, 0, stream>>>(XA, y, X, lVars, ni, M, N, act);
// }

// __global__
// void computeSign_kernel(corr_precision *sb, precision *beta, precision *beta_prev, int *lVars, int *dropidx, int *lasso, int ni) {
// 	int ind = threadIdx.x + blockIdx.x * blockDim.x;
// 	if (ind < ni) {
// 		int si = lVars[ind];
// 		int sg;
// 		if (lasso[0] && dropidx[0] == ind) {
// 			sg = (beta_prev[si] > eps) - (beta_prev[si] < -eps);
// 		}
// 		else {
// 			sg = (beta[si] > eps) - (beta[si] < -eps);
// 			if (sg == 0) {
// 				sg = (beta_prev[si] > eps) - (beta_prev[si] < -eps);
// 			}
// 		}
// 		sb[ind] = (corr_precision) sg;
// 	}
// }

// void computeSign(corr_precision *sb, precision *beta, precision *beta_prev, int *lVars, int *dropidx, int *lasso, int ni, cudaStream_t &stream) {
// 	dim3 blockDim(min(ni, 1024));
// 	dim3 gridDim((ni + blockDim.x - 1) / blockDim.x);
// 	computeSign_kernel<<<gridDim, blockDim, 0, stream>>>(sb, beta, beta_prev, lVars, dropidx, lasso, ni);
// }

// __global__
// void correct_kernel(corr_precision *beta, corr_precision *betaols, corr_precision *sb, corr_precision *y, corr_precision *yh, corr_precision *z, precision *a1, precision *a2, precision *lambda, corr_precision min_l2, corr_precision g, int ni, int M) {
// 	corr_precision zz = 0, yhyh = 0;
// 	for (int i = 0; i < M; i++) {
// 		if (i < ni) zz += sb[i] * z[i];
// 		yhyh += (y[i] - yh[i]) * (y[i] - yh[i]);
// 	}
// 	corr_precision err = a2[0], G = lambda[0];
// 	if (err < min_l2 && min_l2 * min_l2 >= yhyh) {
// 		err = min_l2;
// 		G = sqrt((min_l2 * min_l2 - yhyh) / (min_l2 * min_l2 * zz));
// 	}
// 	if (G < g && 1 > g * g * zz) {
// 		G = g;
// 		err = sqrt(yhyh / (1 - g * g * zz));
// 	}
// 	corr_precision l1 = 0;
// 	for (int i = 0; i < ni; i++) {
// 		beta[i] = betaols[i] - err * G * z[i];
// 		l1 += abs(beta[i]);
// 	}
// 	a1[0] = l1;
// 	a2[0] = err;
// 	lambda[0] = G;
// }

// void correct(corr_precision *beta, corr_precision *betaols, corr_precision *sb, corr_precision *y, corr_precision *yh, corr_precision *z, precision *a1, precision *a2, precision *lambda, corr_precision min_l2, corr_precision g, int ni, int M, cudaStream_t &stream) {
// 	correct_kernel<<<1, 1, 0, stream>>>(beta, betaols, sb, y, yh, z, a1, a2, lambda, min_l2, g, ni, M);
// }


#endif
