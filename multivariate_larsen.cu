#include <bits/stdc++.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define INF 50000

typedef float precision;

using namespace std;

template<class T>
struct dmatrix {
    T *d_mat;
    int M, N;
};

template<class T>
struct debug {
    T upper1;
    T normb;
    int step;
    int nVars;
};

int str_to_int(char *argv) {
    int i = 0, num_models = 0;
    while(argv[i] != '\0') {
        num_models = num_models * 10 + argv[i] - '0';
        i++;
    }
    return num_models;
}

int idx(int i, int j, int lda) {
    return j * lda + i;
}

__device__ int IDX(int i, int j, int lda) {
    return j * lda + i;
}

__device__ cublasHandle_t handle;

__global__ void CUBLAS_START() {
    cublasCreate(&handle);
}

__global__ void CUBLAS_STOP() {
    cublasDestroy(handle);
}

class gpuTimer {
    cudaEvent_t start_time, stop_time;
public:
    gpuTimer() {
        cudaEventCreate(&start_time);
        cudaEventCreate(&stop_time);
    }
    ~gpuTimer() {
        cudaEventDestroy(start_time);
        cudaEventDestroy(stop_time);
    }
    void start() {
        cudaEventRecord(start_time);
    }
    void stop() {
        cudaEventRecord(stop_time);
    }
    float elapsed() {
        cudaEventSynchronize(start_time);
        cudaEventSynchronize(stop_time);
        float mill = 0;
        cudaEventElapsedTime(&mill, start_time, stop_time);
        return mill;
    }
};

template<class T>
void read_flat_mri(char *argv, dmatrix<T> &X, dmatrix<T> &Y) {
    ifstream fp(argv);
    int M, Z;
    fp >> M >> Z;
    
    T *h_number = new T[M * Z];
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < Z; j++) {
            fp >> h_number[idx(i, j, M)];
        }
    }
    fp.close();

    X.M = M - 1;
    X.N = Z;
    cudaMallocManaged(&X.d_mat, X.M * X.N * sizeof(T));
    
    for (int j = 0; j < X.N; j++) {
        T mean = 0;
        for (int i = 0; i < M - 1; i++) {
            X.d_mat[idx(i, j, X.M)] = h_number[idx(i, j, M)];
            mean += X.d_mat[idx(i, j, X.M)];
        }
        mean /= X.M;
        T std = 0;
        for (int i = 0; i < X.M; i++) {
            X.d_mat[idx(i, j, X.M)] = X.d_mat[idx(i, j, X.M)] - mean;
            std += X.d_mat[idx(i, j, X.M)] * X.d_mat[idx(i, j, X.M)];
        }
        std /= X.M - 1;
        std = sqrt(std);
        T norm = 0;
        for (int i = 0; i < X.M; i++) {
            X.d_mat[idx(i, j, X.M)] = X.d_mat[idx(i, j, X.M)] / std;
            norm += X.d_mat[idx(i, j, X.M)] * X.d_mat[idx(i, j, X.M)];
        }
        norm = sqrt(norm);
        for (int i = 0; i < X.M; i++) {
            X.d_mat[idx(i, j, X.M)] = X.d_mat[idx(i, j, X.M)] / norm;
        }
    }

    Y.M = M - 1;
    Y.N = Z;
    cudaMallocManaged(&Y.d_mat, Y.M * Y.N * sizeof(T));
    
    for (int j = 0; j < Y.N; j++) {
        T mean = 0;
        for (int i = 1; i < M; i++) {
            Y.d_mat[idx(i - 1, j, Y.M)] = h_number[idx(i, j, M)];
            mean += Y.d_mat[idx(i - 1, j, Y.M)];
        }
        mean /= Y.M;
        T std = 0;
        for (int i = 0; i < Y.M; i++) {
            Y.d_mat[idx(i, j, Y.M)] = Y.d_mat[idx(i, j, Y.M)] - mean;
            std += Y.d_mat[idx(i, j, Y.M)] * Y.d_mat[idx(i, j, Y.M)];
        }
        std /= Y.M - 1;
        std = sqrt(std);
        T norm = 0;
        for (int i = 0; i < Y.M; i++) {
            Y.d_mat[idx(i, j, Y.M)] = Y.d_mat[idx(i, j, Y.M)] / std;
            norm += Y.d_mat[idx(i, j, Y.M)] * Y.d_mat[idx(i, j, Y.M)];
        }
        norm = sqrt(norm);
        for (int i = 0; i < Y.M; i++) {
            Y.d_mat[idx(i, j, Y.M)] = Y.d_mat[idx(i, j, Y.M)] / norm;
        }
    }
}

template<class T>
__global__ void SET_MODEL(dmatrix<T> X, dmatrix<T> Y, T **x, T **y, T **mu, T **beta,
    int **act, int **nVars, int **lasso, int **step, int **done,
    int M, int N, int mod, int num_models)
{
    int ni = threadIdx.x + blockIdx.x * blockDim.x;
    int mi = threadIdx.y + blockIdx.y * blockDim.y;
    if (mi < M && ni < N) {
        int hact = *act[mod];
        int xcol = (ni >= hact)? ni + 1: ni;
        x[mod][IDX(mi, ni, M)] = X.d_mat[IDX(mi, xcol, X.M)];
        if (mi == 0 && ni == 0) {
            *nVars[mod] = 0;
            *lasso[mod] = 0;
            *step[mod] = 0;
            *done[mod] = 0; 
        }
        if (mi == 0) {
            beta[mod][ni] = 0;
        }
        if (ni == 0) {
            mu[mod][mi] = 0;
            y[mod][mi] = Y.d_mat[IDX(mi, hact, Y.M)];
        }
    }
}

template<class T>
void set_model(dmatrix<T> X, dmatrix<T> Y, T **x, T **y, T **mu, T **beta,
    int **act, int **nVars, int **lasso, int **step, int **done,
    int M, int N, int mod, int num_models)
{
    dim3 blockDim(32, 32);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x,
        (M + blockDim.y - 1) / blockDim.y);
    SET_MODEL<T><<<gridDim, blockDim>>>(X, Y, x, y, mu, beta,
        act, nVars, lasso, step, done, M, N, mod, num_models);
}

template<class T>
__global__ void VEC_SUB(T *a, T *b, T *c, int size) {
    int ind = threadIdx.x + blockIdx.x * blockDim.x;
    if (ind < size) {
        c[ind] = a[ind] - b[ind];
    }
}

template<class T>
__global__ void VEC_ABS(T *a, T *c, int size) {
    int ind = threadIdx.x + blockIdx.x * blockDim.x;
    if (ind < size) {
        c[ind] = fabs(a[ind]);
    }
}

template<class T>
__global__ void VEC_EXC(T *a, int *exc, int ni, T def) {
    int ind = threadIdx.x + blockIdx.x * blockDim.x;
    if (ind < ni) {
        int si = exc[ind];
        a[si] = def;
    }
}

template<class T>
__global__ void GATHER(T *gx, T *x, int *lVars, int ni, int M) {
    int ind = threadIdx.x + blockIdx.x * blockDim.x;
    int mi = threadIdx.y + blockIdx.y * blockDim.y;
    if (ind < M && mi < M) {
        if (ind < ni) {
            int si = lVars[ind];
            gx[IDX(mi, ind, M)] = x[IDX(mi, si, M)];
        }
        else {
            gx[IDX(mi, ind, M)] = 0;
        }
    }
}

template<class T>
__global__ void IDENTITYEXT(T *G, int ni, int M) {
    int ind = threadIdx.x + blockIdx.x * blockDim.x;
    if (ind < M && ind >= ni) {
        G[IDX(ind, ind, M)] = 1;
    }
}

template<class T>
__global__ void GAMMAT(T *gammat, T *beta, T *betaOls, int *lVars, int ni, int M, T def) {
    int ind = threadIdx.x + blockIdx.x * blockDim.x;
    if (ind < M) {
        if (ind < ni - 1) {
            int si = lVars[ind];
            T val = beta[si] / (beta[si] - betaOls[ind]);
            if (val <= 0)
                val = INF;
            gammat[ind] = val;
        }
        else {
            gammat[ind] = def;
        }
    }
}

template<class T>
__global__ void TRANS(T *c, T *cd, T cmax, int N) {
    int ind = threadIdx.x + blockIdx.x * blockDim.x;
    if (ind < N) {
        T a = (c[ind] - cmax) / (cd[ind] - cmax);
        T b = (c[ind] + cmax) / (cd[ind] + cmax);
        if (a <= 0)
            a = INF;
        if (b <= 0)
            b = INF;
        cd[ind] = min(a, b);
    }
}

template<class T>
__global__ void UPDATE(T *beta, T *mu, T *d, T *betaOls, int *lVars, int ni, int M, T gamma) {
    int ind = threadIdx.x + blockIdx.x * blockDim.x;
    if (ind < M) {
        mu[ind] += gamma * d[ind];
        if (ind < ni) {
            int si = lVars[ind];
            beta[si] += gamma * (betaOls[ind] - beta[si]);
        }
    }
}

__global__ void DROP(int *lVars, int dropIdx, int ni) {
    int ind = threadIdx.x;
    if (ind < ni && ind > dropIdx) {
        int val = lVars[ind];
        __syncthreads();
        lVars[ind - 1] = val;
    }
}

template<class T>
__global__ void RESIDUE(T *beta, T *y, T *mu, T *cd, T *r, int M, int N) {
    int ind = threadIdx.x + blockIdx.x * blockDim.x;
    if (ind < M) {
        r[ind] = y[ind] - mu[ind];
    }
    if (ind < N) {
        cd[ind] = fabs(beta[ind]);
    }
}

template<class T>
__global__ void LARS_ITER(T **x, T **y, T **mu, T **beta, T **r, T **c, T **absC, T **cmax, T **gx, T **G,
    int **lVars, int **nVars, int **step, int **lasso, int **done, int M, int N, T *flop)
{
    int mod = threadIdx.x;
    if (*done[mod])
        return;
    int Mdim = (M < 1024)? M: 1024;
    int Ndim = (N < 1024)? N: 1024;

    dim3 bM(Mdim);
    dim3 gM((M + bM.x - 1) / bM.x);
    dim3 bN(Ndim);
    dim3 gN((N + bN.x - 1) / bN.x);
    dim3 bMM(32, 32);
    dim3 gMM((M + bMM.x - 1) / bMM.x, (M + bMM.y - 1) / bMM.y);

    int ni = *nVars[mod];

    if (ni < M && *step[mod] < 8 * M) {}
    else {
        *done[mod] = 1;
        return;
    }

    VEC_SUB<T><<<gM, bM>>>(y[mod], mu[mod], r[mod], M);flop[mod] += M;

    T alp = 1;
    T bet = 0;
    cublasSgemv(handle, CUBLAS_OP_T,
        M, N,
        &alp,
        (const T *)x[mod], M,
        (const T *)r[mod], 1,
        &bet,
        c[mod], 1);flop[mod] += 2 * N * M;

    VEC_ABS<T><<<gN, bN>>>(c[mod], absC[mod], N);

    VEC_EXC<T><<<1, ni>>>(absC[mod], lVars[mod], ni, 0);

    int cidx;
    cublasIsamax(handle,
        N, absC[mod],
        1, &cidx);
    cidx--;
    *cmax[mod] = absC[mod][cidx];

    if (*lasso[mod] == 0) {
        lVars[mod][ni] = cidx;
        *nVars[mod] += 1;
        ni += 1;
    }
    else
        *lasso[mod] = 0;

    GATHER<T><<<gMM, bMM>>>(gx[mod], x[mod], lVars[mod], ni, M);

    cublasSgemv(handle, CUBLAS_OP_T,
        M, M,
        &alp,
        (const T *)gx[mod], M,
        (const T *)y[mod], 1,
        &bet,
        r[mod], 1);flop[mod] += 2 * M * M;

    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
        M, M, M,
        &alp,
        (const T *)gx[mod], M,
        (const T *)gx[mod], M,
        &bet,
        G[mod], M);flop[mod] += 3 * M * M * M;

    IDENTITYEXT<T><<<gM, bM>>>(G[mod], ni, M);
}

template<class T>
__global__ void LARS_ITER_1(T **I, T **r, T **betaOls, T **gx, T **mu, T **d, T **beta, T **gammat, T **x, T **cd, T **c, T **cmax, T **y, T **upper1, T **normb,
    int **lVars, int **nVars, int **lasso, int **step, int **done, int M, int N, T g, T *flop)
{
    int mod = threadIdx.x;
    if (*done[mod])
        return;
    int Mdim = (M < 1024)? M: 1024;
    int Ndim = (N < 1024)? N: 1024;
    dim3 bM(Mdim);
    dim3 gM((M + bM.x - 1) / bM.x);
    dim3 bN(Ndim);
    dim3 gN((N + bN.x - 1) / bN.x);
    int ni = *nVars[mod];

    T alp = 1;
    T bet = 0;
    cublasSgemv(handle, CUBLAS_OP_N,
        M, M,
        &alp,
        (const T *)I[mod], M,
        (const T *)r[mod], 1,
        &bet,
        betaOls[mod], 1);flop[mod] += 2 * M * M;

    cublasSgemv(handle, CUBLAS_OP_N,
        M, M,
        &alp,
        (const T *)gx[mod], M,
        (const T *)betaOls[mod], 1,
        &bet,
        d[mod], 1);flop[mod] += 2 * M * M;

    VEC_SUB<T><<<gM, bM>>>(d[mod], mu[mod], d[mod], M);

    GAMMAT<T><<<gM, bM>>>(gammat[mod], beta[mod], betaOls[mod], lVars[mod], ni, M, INF);flop[mod] += 2 * ni;

    int dropIdx;
    cublasIsamin(handle,
        M, gammat[mod],
        1, &dropIdx);
    dropIdx--;
    T gamma_tilde = gammat[mod][dropIdx];

    cublasSgemv(handle, CUBLAS_OP_T,
        M, N,
        &alp,
        (const T *)x[mod], M,
        (const T *)d[mod], 1,
        &bet,
        cd[mod], 1);

    TRANS<T><<<gN, bN>>>(c[mod], cd[mod], *cmax[mod], N);
    VEC_EXC<T><<<1, ni>>>(cd[mod], lVars[mod], ni, INF);

    int tmpIdx;
    cublasIsamin(handle,
        N, cd[mod],
        1, &tmpIdx);
    tmpIdx--;
    T gamma = cd[mod][tmpIdx];

    if (ni == M)
        gamma = 1;
    if (gamma_tilde < gamma) {
        *lasso[mod] = 1;
        gamma = gamma_tilde;
    }
    UPDATE<T><<<gM, bM>>>(beta[mod], mu[mod], d[mod], betaOls[mod], lVars[mod], ni, M, gamma);flop[mod] += 6 * N;

    int stp = *step[mod] + 1;
    *step[mod] = stp;

    if (*lasso[mod]) {
        DROP<<<1, ni>>>(lVars[mod], dropIdx, ni);
        ni--;
        *nVars[mod] = ni;
    }
    T a1, a2;
    RESIDUE<T><<<gN, bN>>>(beta[mod], y[mod], mu[mod], cd[mod], r[mod], M, N);flop[mod] += M;
    cublasSnrm2(handle, M, r[mod], 1, &a2);flop[mod] += 2 * M;
    cublasSasum(handle, N, cd[mod], 1, &a1);flop[mod] += N;

    if (stp > 1) {
        T G = -(*upper1[mod] - a2) / (*normb[mod] - a1);
        if (G < g) {
            *done[mod] = 1;
            return;
        }
    }
    *upper1[mod] = a2;
    *normb[mod] = a1;
}

int main(int argc, char *argv[]) {
    gpuTimer timer;
    precision g = 0.43;

    dmatrix<precision> X, Y;

    // Reading flattened MRI image.
    read_flat_mri<precision>(argv[1], X, Y);
    printf("Read FMRI Data X of shape:(%d,%d)\n", X.M, X.N);
    printf("Read FMRI Data Y of shape:(%d,%d)\n", Y.M, Y.N);

    int M = X.M, N = X.N - 1, total_models = X.N;

    debug<precision> deb[total_models];

    // Number of models to solve in parallel.
    int num_models = str_to_int(argv[2]);
    printf("Number of models in ||l:%d\n", num_models);

    // Declare all Lars variables.
    int **nVars;
    cudaMallocManaged(&nVars, num_models * sizeof(int *));
    for (int i = 0; i < num_models; i++)
        cudaMallocManaged(&nVars[i], sizeof(int));
    
    int **step;
    cudaMallocManaged(&step, num_models * sizeof(int *));
    for (int i = 0; i < num_models; i++)
        cudaMallocManaged(&step[i], sizeof(int));

    int **lasso;
    cudaMallocManaged(&lasso, num_models * sizeof(int *));
    for (int i = 0; i < num_models; i++)
        cudaMallocManaged(&lasso[i], sizeof(int));

    int **act;
    cudaMallocManaged(&act, num_models * sizeof(int *));
    for (int i = 0; i < num_models; i++)
        cudaMallocManaged(&act[i], sizeof(int));

    int **done;
    cudaMallocManaged(&done, num_models * sizeof(int *));
    for (int i = 0; i < num_models; i++)
        cudaMallocManaged(&done[i], sizeof(int));

    precision **x;
    cudaMallocManaged(&x, num_models * sizeof(precision *));
    for (int i = 0; i < num_models; i++)
        cudaMallocManaged(&x[i], M * N * sizeof(precision));

    precision **y;
    cudaMallocManaged(&y, num_models * sizeof(precision *));
    for (int i = 0; i < num_models; i++)
        cudaMallocManaged(&y[i], M * sizeof(precision));

    precision **mu;
    cudaMallocManaged(&mu, num_models * sizeof(precision *));
    for (int i = 0; i < num_models; i++)
        cudaMallocManaged(&mu[i], M * sizeof(precision));

    precision **beta;
    cudaMallocManaged(&beta, num_models * sizeof(precision *));
    for (int i = 0; i < num_models; i++)
        cudaMallocManaged(&beta[i], N * sizeof(precision));

    precision **r;
    cudaMallocManaged(&r, num_models * sizeof(precision *));
    for (int i = 0; i < num_models; i++)
        cudaMallocManaged(&r[i], M * sizeof(precision));

    precision **c;
    cudaMallocManaged(&c, num_models * sizeof(precision *));
    for (int i = 0; i < num_models; i++)
        cudaMallocManaged(&c[i], N * sizeof(precision));

    precision **absC;
    cudaMallocManaged(&absC, num_models * sizeof(precision *));
    for (int i = 0; i < num_models; i++)
        cudaMallocManaged(&absC[i], N * sizeof(precision));

    precision **cmax;
    cudaMallocManaged(&cmax, num_models * sizeof(precision *));
    for (int i = 0; i < num_models; i++)
        cudaMallocManaged(&cmax[i], sizeof(precision));

    precision **gx;
    cudaMallocManaged(&gx, num_models * sizeof(precision *));
    for (int i = 0; i < num_models; i++)
        cudaMallocManaged(&gx[i], M * M * sizeof(precision));

    precision **G;
    cudaMallocManaged(&G, num_models * sizeof(precision *));
    for (int i = 0; i < num_models; i++)
        cudaMallocManaged(&G[i], M * M * sizeof(precision));

    precision **I;
    cudaMallocManaged(&I, num_models * sizeof(precision *));
    for (int i = 0; i < num_models; i++)
        cudaMallocManaged(&I[i], M * M * sizeof(precision));

    precision **betaOls;
    cudaMallocManaged(&betaOls, num_models * sizeof(precision *));
    for (int i = 0; i < num_models; i++)
        cudaMallocManaged(&betaOls[i], M * sizeof(precision));

    precision **d;
    cudaMallocManaged(&d, num_models * sizeof(precision *));
    for (int i = 0; i < num_models; i++)
        cudaMallocManaged(&d[i], M * sizeof(precision));

    precision **gammat;
    cudaMallocManaged(&gammat, num_models * sizeof(precision *));
    for (int i = 0; i < num_models; i++)
        cudaMallocManaged(&gammat[i], M * sizeof(precision));

    precision **cd;
    cudaMallocManaged(&cd, num_models * sizeof(precision *));
    for (int i = 0; i < num_models; i++)
        cudaMallocManaged(&cd[i], N * sizeof(precision));

    precision **upper1;
    cudaMallocManaged(&upper1, num_models * sizeof(precision *));
    for (int i = 0; i < num_models; i++)
        cudaMallocManaged(&upper1[i], sizeof(precision));

    precision **normb;
    cudaMallocManaged(&normb, num_models * sizeof(precision *));
    for (int i = 0; i < num_models; i++)
        cudaMallocManaged(&normb[i], sizeof(precision));

    int **lVars;
    cudaMallocManaged(&lVars, num_models * sizeof(int *));
    for (int i = 0; i < num_models; i++)
        cudaMallocManaged(&lVars[i], M * sizeof(int));

    for (int i = 0; i < num_models; i++) {
        *act[i] = i;
    }

    for (int i = 0; i < num_models; i++) {
        set_model<precision>(X, Y, x, y, mu, beta,
            act, nVars, lasso, step, done,
            M, N, i, num_models);
    }

    cudaDeviceSynchronize();
    cublasHandle_t hnd;
    cublasCreate(&hnd);
    int *info, *pivot;
    cudaMallocManaged(&info, num_models * sizeof(int));
    cudaMallocManaged(&pivot, num_models * M * sizeof(int));
    precision *flop;
    cudaMallocManaged(&flop, num_models * sizeof(precision));
    cudaMemset(flop, 0, num_models * sizeof(precision));
    CUBLAS_START<<<1, 1>>>();

    int top = num_models;
    float totime = 0;
    while (true) {
        timer.start();
    
        LARS_ITER<precision><<<1, num_models>>>(x, y, mu, beta, r, c, absC, cmax, gx, G,
            lVars, nVars, step, lasso, done, M, N, flop);

        cublasSgetrfBatched(hnd, M, G, M, pivot, info, num_models);
        cublasSgetriBatched(hnd, M, (const precision **)G, M, pivot, I, M, info, num_models);

        LARS_ITER_1<precision><<<1, num_models>>>(I, r, betaOls, gx, mu, d, beta, gammat, x, cd, c, cmax, y, upper1, normb,
            lVars, nVars, lasso, step, done, M, N, g, flop);

        timer.stop();
        totime += timer.elapsed();

        bool toContinue = false;
        for (int i = 0; i < num_models; i++) {
            cudaDeviceSynchronize();
            if (*done[i]) {
                deb[*act[i]].upper1 = *upper1[i];
                deb[*act[i]].normb = *normb[i];
                deb[*act[i]].nVars = *nVars[i];
                deb[*act[i]].step = *step[i];
                if (top < total_models) {
                    *act[i] = top++;
                    printf("\rSolving till model:%d ", top);
                    set_model<precision>(X, Y, x, y, mu, beta,
                        act, nVars, lasso, step, done,
                        M, N, i, num_models);
                    toContinue = true;
                }
            }
            else
                toContinue = true;
        }
        if (!toContinue)
            break;
    }

    printf("\n");
    cudaDeviceSynchronize();

    ofstream f;
    f.open("res.txt");
    for (int i = 0; i < total_models; i++) {
        f << deb[i].upper1 << ":" << deb[i].normb << ":" << deb[i].step << ":" << deb[i].nVars << "\n";
    }
    f.close();

    precision total_flops = 0;
    for (int i = 0; i < num_models; i++)
        total_flops += flop[i];

    printf("Performance in Gflops:%f\n", (total_flops * 1e-9) / (totime * 1e-3));

    CUBLAS_STOP<<<1, 1>>>();
    cublasDestroy(hnd);

    return 0;
}