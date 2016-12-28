#include <iostream>
#include <fstream>
#include <cstring>
#include <cmath>
#define INF 50000

using namespace std;
//Inititalise params.
__global__
void dInit(int *d_nVars, int *d_lasso, int *d_done, int *d_step, int l){
int    	ind = threadIdx.x + blockIdx.x * blockDim.x;
	if(ind < l){
		d_nVars[ind] = 0;
		d_lasso[ind] = 0;
		d_done[ind] = 0;
		d_step[ind] = 1;
	}
}
//d_ctrl is the variable which is set to 0 if all the models are completed hence stop and 1 if not.
//Loop checker of larsen.
__global__
void dCheck(int *d_ctrl, int *d_step, int *d_nVars, int M, int Z, int l, int *d_done, int *d_towrite){
int    	ind = threadIdx.x + blockIdx.x * blockDim.x;
	if(ind < l && !d_done[ind]){
	int	n = min(M - 1, Z - 1);
		if(d_nVars[ind] < n && d_step[ind] < 8 * n){
			*d_ctrl = 1;
		}
		else{
			d_done[ind] = 1;
			*d_towrite = 1;
		}
	}
}
//c = _ = X' * (y - mu);
__global__
void dCorr(double *d_X, double *d_Y, double *d_mu, double *d_c, double *d_, int M, int Z, int *d_act, int l, int *d_done, double *d_corr_flop){
int	ind = threadIdx.x + blockIdx.x * blockDim.x;
int	mod = threadIdx.y + blockIdx.y * blockDim.y;
	if(mod < l && !d_done[mod]){	
		if(ind == 0) d_corr_flop[mod] += M * Z * 2.0;
		if(ind < Z - 1){
		int	i, act, j;
		double	tot;
			i = ind;
			act = d_act[mod];
			if(i >= act)i++;
			tot = 0;
			for(j = 0;j < M - 1;j++){
				tot += d_X[j * Z + i] * (d_Y[j * Z + act] - d_mu[mod * (M - 1) + j]);
			}
			d_c[mod * (Z - 1) + ind] = d_[mod * (Z - 1) + ind] = tot;
		}
	}
}
//Exclude Active variables in _.
__global__
void dExcCorr(double *d_, int *d_lVars, int *d_nVars, int M, int Z, int l, int *d_done){
int	ind = threadIdx.x + blockIdx.x * blockDim.x;
int	mod = threadIdx.y + blockIdx.y * blockDim.y;
	if(mod < l && !d_done[mod]){
	int	n = min(M - 1, Z - 1);
		if(ind < d_nVars[mod]){
		int	i = d_lVars[mod * n + ind];
			d_[mod * (Z - 1) + i] = 0;
		}
	}
}
//Take max(abs(_(I)))
__global__
void dMaxcorr(double *d_, double *d_cmax, int *d_ind, int Z, int l, int *d_done){
int	ind = threadIdx.x + blockIdx.x * blockDim.x;
	if(ind < l && !d_done[ind]){
	int	j, maxi;
	double	max, tot;
		maxi = -1;
		max = -INF;
		for(j = 0;j < Z - 1;j++){
			tot = fabs(d_[ind * (Z - 1) + j]);
			if(tot > max){
				max = tot;
				maxi = j;	
			}
		}
		d_cmax[ind] = max;
		d_ind[ind] = maxi;
	}
}
//Add cmax to active set.
__global__
void dLassoAdd(int *d_ind, int *d_lVars, int *d_nVars, int *d_lasso, int M, int Z, int l, int *d_done, double *d_add_flop){
int    	ind = threadIdx.x + blockIdx.x * blockDim.x;
	if(ind < l && !d_done[ind]){
	int	n = min(M - 1, Z - 1);
		if(!d_lasso[ind]){
			d_lVars[ind * n + d_nVars[ind]] = d_ind[ind];
			d_nVars[ind] += 1;
		}
		else{
			d_lasso[ind] = 0;
		}
		d_add_flop[ind] += 3 * pow(double(d_nVars[ind]), 2.0);
	}
}
//Compute _ = X(:, A)' * y.
__global__
void dXincTY(double *d_X, double *d_Y, double *d_, int *d_lVars, int *d_nVars, int M, int Z, int *d_act, int l, int *d_done, double *d_other_flop){
int	ind = threadIdx.x + blockIdx.x * blockDim.x;
int	mod = threadIdx.y + blockIdx.y * blockDim.y;
	if(mod < l && !d_done[mod]){
	int	n = min(M - 1, Z - 1);
		if(ind < d_nVars[mod]){
			if(ind == 0)d_other_flop[mod] += d_nVars[mod] * M * 2;
		int	i, j, act;
		double	tot;
			i = d_lVars[mod * n + ind];
			act = d_act[mod];
			if(i >= act)i++;
			tot = 0;
			for(j = 0;j < M - 1;j++)tot += d_X[j * Z + i] * d_Y[j * Z + act];
			d_[mod * n + ind] = tot;
		}
	}
}
//Compute G = X(:, A)' * X(:, A).
__global__
void dSetGram(double *d_X, double *d_G, double *d_I, int *d_lVars, int n, int M, int Z, int mod, int *d_act, int l, double *d_other_flop){
int	indx = threadIdx.x;
int	indy = blockIdx.x;
	if(indx < n && indy < n && indx <= indy){
		if(indx == 0 && indy == 0)d_other_flop[0] += n * n * M * 2;
	int	i, j, k, act;
	double 	tot;
		act = d_act[mod];
		i = d_lVars[mod * min(M-1,Z-1) + indx];
		j = d_lVars[mod * min(M-1,Z-1) + indy];
		if(i >= act)i++;
		if(j >= act)j++;
		tot = 0;
		for(k = 0;k < M - 1;k++){
			tot += d_X[k * Z + i] * d_X[k * Z + j];
		}
		if(indx == indy){
			d_G[indx * n + indy] = tot;
			d_I[indx * n + indy] = 1;
		}
		else{
			d_G[indx * n + indy] = d_G[indy * n + indx] = tot;
			d_I[indx * n + indy] = d_I[indy * n + indx] = 0;
		}
	}
}
//Guass jordan functions normalize non-diagnal.
__global__ 
void nodiag_normalize(double *A, double *I, int n, int i, double *d_other_flop){
int	y = threadIdx.x;
	if (y < n){
		if(y == 0) d_other_flop[0] += n * 2;
		if (y != i){
			I[i * n + y] /= A[i * n + i];
			A[i * n + y] /= A[i * n + i];
		}
	}
}
//Guass jordan function normalize diagnal.
__global__ 
void diag_normalize(double *A, double *I, int n, int i){
	I[i * n + i] /= A[i * n + i];
	A[i * n + i] = 1;
}
//Guass jordan function row transforms.
__global__ 
void gaussjordan(double *A, double *I, int n, int i, double *d_other_flop){
int	x = threadIdx.x;
int	y = blockIdx.x;
	if (x < n && y < n){
		if(x == 0 && y == 0)d_other_flop[0] += n * n * 4;
		if (x != i){
			I[x * n + y] -= I[i * n + y] * A[x * n + i];
			if(y != i){
				A[x * n + y] -= A[i * n + y] * A[x * n + i];
			}
		}
	}
}
//Set zero.
__global__
void set_zero(double *A, double *I, int n, int i){
int 	x = threadIdx.x;
	if (x < n){
		if (x != i){
			A[x * n + i] = 0;
		}
	}
}
//Compute betaOLS = I * _.
__global__
void dBetaols(double *d_I, double *d_, double *d_betaOLS, int n, int mod, int M, int Z, int l, double *d_ols_flop){
int	ind = threadIdx.x;
	if(ind < n){
		if(ind == 0) d_ols_flop[mod] += M * 2 * n;
	int	j;
	double	tot;
		tot = 0;
		for(j = 0;j < n;j++){
			tot += d_I[ind * n + j] * d_[mod * min(M-1,Z-1) + j];
		}
		d_betaOLS[mod * min(M-1,Z-1) + ind] = tot;
	}
}
//Computing d = X(:, A) * betaOLS and gamma list.
__global__
void ddgamma(double *d_X, double *d_mu, double *d_beta, double *d_betaOLS, double *d_gamma, double *d_d, int *d_lVars, int *d_nVars, int M, int Z, int *d_act, int l, int *d_done, double *d_other_flop){
int	ind = threadIdx.x + blockIdx.x * blockDim.x;
int    	mod = threadIdx.y + blockIdx.y * blockDim.y;
	if(mod < l && !d_done[mod]){
		if(ind < M - 1){
		int	i, j, n, act;
		double	tot;
			n = d_nVars[mod];
			if (ind == 0)d_other_flop[mod] += M * n * 2 + n * 2;
			act = d_act[mod];
			tot = 0;
			for(j = 0;j < n;j++){
				i = d_lVars[mod * min(M-1,Z-1) + j];
				if(i >= act)i++;
				tot += d_X[ind * Z + i] * d_betaOLS[mod * min(M-1,Z-1) + j];
			}
			d_d[mod * (M - 1) + ind] = tot - d_mu[mod * (M - 1) + ind];
			if(ind < n - 1){
				i = d_lVars[mod * min(M-1,Z-1) + ind];
           	     		tot = d_beta[mod * (Z - 1) + i] / (d_beta[mod * (Z - 1) + i] - d_betaOLS[mod * min(M-1,Z-1) + ind]);
             			if(tot <= 0)tot = INF;
                		d_gamma[mod * min(M-1,Z-1) + ind] = tot;
			}
		}
	}
}
//Computing min(gamma(gamma > 0))
__global__
void dGammamin(double *d_gamma, int *d_ind, int *d_nVars, int M, int Z, int l, int *d_done){
int    	ind = threadIdx.x + blockIdx.x * blockDim.x;
	if(ind < l && !d_done[ind]){
	int	j, n, mini, nt = min(M-1,Z-1);
	double	min, tot;
		n = d_nVars[ind];
		min = INF;
		mini = -1;
		tot = 0;
		for(j = 0;j < n - 1;j++){
			tot = d_gamma[ind * nt + j];
			if(tot < min){
				min = tot;
				mini = j;
			}
		}
		d_gamma[ind] = min;
		d_ind[ind] = mini;
	}
}
//Computing _ = X' * d and gamma_tilde.
__global__
void dXTd(double *d_X, double *d_c, double *d_, double *d_d, double *d_cmax, int M, int Z, int *d_act, int l, int *d_done, double *d_other_flop){
int    	ind = threadIdx.x + blockIdx.x * blockDim.x;
int    	mod = threadIdx.y + blockIdx.y * blockDim.y;
	if(mod < l && !d_done[mod]){
		if(ind < Z - 1){
			if (ind == 0) d_other_flop[mod] += Z * M * 2 + 2 * Z;
		int	i, act, j;
		double	tot, cmax, a, b;
			cmax = d_cmax[mod];
			act = d_act[mod];
			i = ind;
			if(i >= act)i++;
			tot = 0;
			for(j = 0;j < M - 1;j++){
				tot += d_X[j * Z + i] * d_d[mod * (M - 1) + j];
			}
			a = (d_c[mod * (Z - 1) + ind] + cmax) / (tot + cmax);
			b = (d_c[mod * (Z - 1) + ind] - cmax) / (tot - cmax);
			if(a <= 0)a = INF;
			if(b <= 0)b = INF;
			tot = min(a, b);
			d_[mod * (Z - 1) + ind] = tot;
		}
	}
}
//Excluding active variables from gamma_tilde.
__global__
void dExctmp(double *d_, int *d_lVars, int *d_nVars, int M, int Z, int l, int *d_done){
int    	ind = threadIdx.x + blockIdx.x * blockDim.x;
int    	mod = threadIdx.y + blockIdx.y * blockDim.y;
        if(mod < l && !d_done[mod]){
	int	i, n;
		n = d_nVars[mod];
                if(ind < n){
			i = d_lVars[mod * min(M-1,Z-1) + ind];
			d_[mod * (Z - 1) + i] = INF;
		}
        }
}
//Finding gamma = min(gamma_tilde(gamma_tilde > 0)).
__global__
void dTmpmin(double *d_, int Z, int l, int *d_done){
int     ind = threadIdx.x + blockIdx.x * blockDim.x;
        if(ind < l && !d_done[ind]){
        int     j;
        double  min, tot;
		min = INF;
                for(j = 0;j < Z - 1;j++){
                        tot = d_[ind * (Z - 1) + j];
                        if(tot < min){
                                min = tot;
                        }
                }
                d_[ind] = min;
        }
}
//Lasso deviation condition.
__global__
void dLassodev(double *d_, double *d_gamma, int *d_nVars, int *d_lasso, int n, int l, int *d_done){
int     ind = threadIdx.x + blockIdx.x * blockDim.x;
        if(ind < l && !d_done[ind]){
		if(d_nVars[ind] == n){
			if(d_gamma[ind] < 1){
				d_lasso[ind] = 1;
			}
			else{
				d_gamma[ind] = 1;
			}
		}
		else{
			if(d_gamma[ind] < d_[ind]){
				d_lasso[ind] = 1;
			}
			else{
				d_gamma[ind] = d_[ind];
			}
		}
	}
}
//Updates mu and beta.
__global__
void dUpdate(double *d_gamma, double *d_mu, double *d_beta, double *d_betaOLS, double *d_d, int *d_lVars, int *d_nVars, int M, int Z, int l, int *d_done, double *d_other_flop){
int	ind = threadIdx.x + blockIdx.x * blockDim.x;
int    	mod = threadIdx.y + blockIdx.y * blockDim.y;
	if(mod < l && !d_done[mod]){
		if(ind < M - 1){
			if(ind == 0)d_other_flop[mod] += M * 2 + d_nVars[mod] * 3;
		int	i, n = min(M - 1, Z - 1);
		double	gamma = d_gamma[mod];
			d_mu[mod * (M - 1) + ind] += gamma * d_d[mod * (M - 1) + ind];
			if(ind < d_nVars[mod]){
				i = d_lVars[mod * n + ind];
				d_beta[mod * (Z - 1) + i] += gamma * (d_betaOLS[mod * n + ind] - d_beta[mod * (Z - 1) + i]);
			}
		}
	}
}
//Drops deviated lasso variable.
__global__
void dLassodrop(int *d_ind, int *d_lVars, int *d_nVars, int *d_lasso, int M, int Z, int l, int *d_done, double *d_drop_flop){
int     ind = threadIdx.x;
int     mod = blockIdx.x;
	if(mod < l && !d_done[mod]){
		if(d_lasso[mod]){
		int	st, tmp, n = min(M - 1, Z - 1);
			st = d_ind[mod];
			if(ind == 0) d_drop_flop[mod] += 4 * pow(double(d_nVars[mod] - st), 2.0);
			if(ind < d_nVars[mod] - 1 && ind >= st){
				tmp = d_lVars[mod * n + ind + 1];
				__syncthreads();
				d_lVars[mod * n + ind] = tmp;
			}
			if(ind == 0){
				d_nVars[mod] -= 1;
			}
		}
	}
}
//Computing _ = mu - y.
__global__
void dRess(double *d_Y, double *d_mu, double *d_, int M, int Z, int *d_act, int l, int *d_done, double *d_other_flop){
int	ind = threadIdx.x + blockIdx.x * blockDim.x;
int	mod = threadIdx.y + blockIdx.y * blockDim.y;
	if(mod < l && !d_done[mod]){
		if(ind < M - 1){
			if(ind == 0)d_other_flop[mod] += M;
		int	act;
			act = d_act[mod];
			d_[mod * (M - 1) + ind] = d_mu[mod * (M - 1) + ind] - d_Y[ind * Z + act];
		}
	}
}
//Computes G and breaking condition.
__global__
void dFinal(double *d_, double *d_beta, double *d_upper1, double *d_normb, int *d_nVars, int *d_step, double g, int M, int Z, int l, int *d_done, double *d_other_flop, int *d_towrite){
int    	ind = threadIdx.x + blockIdx.x * blockDim.x;
	if(ind < l && !d_done[ind]){
		if(ind == 0)d_other_flop[ind] += M * 3 + Z + 5;
	int	i;
	double	upper1 = 0, normb = 0;
		for(i = 0;i < Z - 1;i++)normb += fabs(d_beta[ind * (Z - 1) + i]);
		for(i = 0;i < M - 1;i++)upper1 += pow(d_[ind * (M - 1) + i], 2);
		upper1 = sqrt(upper1);
		if(d_step[ind] > 1){
		double	G = -(d_upper1[ind] - upper1) / (d_normb[ind] - normb);
			if(G < g){
				d_done[ind] = 1;
				*d_towrite = 1;
			}
		}
		d_upper1[ind] = upper1;
		d_normb[ind] = normb;
		d_step[ind] += 1;
	}
}
