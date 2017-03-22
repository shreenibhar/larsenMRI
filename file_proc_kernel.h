#include <iostream>
#include <fstream>
#include <cstring>
#include <cuda.h>
#include <cmath>

using namespace std;

// Remove last row for new1 and first row for new from d_number.
__global__
void d_trim(double *d_number, double *d_new1, double *d_new,
        int M, int Z)
{
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        if (y < Z) {
                if (x < M - 1)
                        d_new1[x * Z + y] = d_number[x * Z + y];
                if(x > 0 && x < M)
                        d_new[(x - 1) * Z + y] = d_number[x * Z + y];
        }
}

// Normalize data.
__global__
void d_proc(double *d_new1, double *d_new, int M, int Z) {
        int ind = threadIdx.x + blockIdx.x * blockDim.x;
        int i;
        double tot1, sum1;
        double tot, sum;
	M -= 1;
        if (ind < Z) {
                tot1 = tot = 0;
                for (i = 0; i < M; i++) {
                        tot1 += d_new1[i * Z + ind];
                        tot += d_new[i * Z + ind];
                }
                tot1 /= M;
                tot /= M;
                sum1 = sum = 0;
                for (i = 0; i < M; i++) {
                        d_new1[i * Z + ind] -= tot1;
                        d_new[i * Z + ind] -= tot;
                        sum1 += pow(d_new1[i * Z + ind], 2);
                        sum += pow(d_new[i * Z + ind], 2);
                }
                sum1 = sqrt(sum1 / (M - 1));
                sum = sqrt(sum / (M - 1));
                tot1 = tot = 0;
                for (i = 0; i < M; i++) {
                        d_new1[i * Z + ind] /= sum1;
                        d_new[i * Z + ind] /= sum;
                        tot1 += pow(d_new1[i * Z + ind], 2);
                        tot += pow(d_new[i * Z + ind], 2);
                }
                tot1 = sqrt(tot1);
                tot = sqrt(tot);
                for (i = 0; i < M; i++) {
                        d_new1[i * Z + ind] /= tot1;
                        d_new[i * Z + ind] /= tot;
                }
        }
}
