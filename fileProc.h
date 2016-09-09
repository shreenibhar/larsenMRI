#include <iostream>
#include <fstream>
#include <cmath>
#include <cstring>
#include "fileProcKernel.h"

using namespace std;

struct dMatrix{
double *dmatrix;
int    M, N;
};
/*
Handles universal device pointers.
Fuction to write u_beta as csv file, u_upper1 and u_step as csv file.
Writes a single beta model in sparse csv format with step and upper1 residual.
*/
int writeModel(double *&u_beta, int *&u_step, double *&u_upper1, int M, int Z, int actingModel, int bufferModel){

int     j;
double  eps = pow(10, -6);

ofstream f, o, r;
        f.open("QBeta.csv", ios::out | ios::app);
        o.open("QStep.txt", ios::out | ios::app);
        r.open("QRess.txt", ios::out | ios::app);

        o << actingModel << ',' << u_step[bufferModel]-1 << '\n';
        r << actingModel << ',' << 1-u_upper1[bufferModel] << '\n';
        for(j = 0;j < Z-1;j++){
        double  val = u_beta[bufferModel*(Z-1)+j];
                if(fabs(val) > eps){
                        if(j < actingModel){
                                f << actingModel << ',' << j << ',' << val << '\n';
                        }
                        else{
                                f << actingModel << ',' << j+1 << ',' << val << '\n';
                        }
                }
        }
        f.close();
        o.close();
        r.close();
        return 0;
}
/*
Returns device pointer.
Function to read flattened 2d FMRI image.
*/
dMatrix readFlatMRI(char *argv){
fstream fp(argv);
int     M, Z, i, j;
string  str;
        getline(fp, str);
        getline(fp, str);
        getline(fp, str);
        fp >> str >> str >> M >> str >> str >> Z;
double  *number = new double[M*Z], *d_number;
        for (i = 0;i < M;i++){
                for (j = 0;j < Z;j++)
                        fp >> number[i*Z+j];
        }
        fp.close();
        cudaMalloc((void **)&d_number, M*Z*sizeof(double));
        cudaMemcpy(d_number, number, M*Z*sizeof(double), cudaMemcpyHostToDevice);
        delete[] number;
dMatrix mat;
        mat.dmatrix = d_number;
        mat.M = M;
        mat.N = Z;
        return mat;
}
//Remove first and last row for new1 and new respectively and normalize.
dMatrix * procFlatMRI(dMatrix d_mat){
double	*d_new1, *d_new, *d_number = d_mat.dmatrix;
int	M = d_mat.M, Z = d_mat.N;
	cudaMalloc((void **)&d_new1, (M - 1) * Z * sizeof(double));
        cudaMalloc((void **)&d_new, (M - 1) * Z * sizeof(double));
dim3    bz(1000);
dim3    gz((Z + bz.x - 1) / bz.x);
dim3    bmz(31, 31);
dim3    gmz((M + bmz.x - 1) / bmz.x, (Z + bmz.y - 1) / bmz.y);
        dTrim<<<gmz, bmz>>>(d_number, d_new1, d_new, M, Z);
        cudaFree(d_number);
        dProc<<<gz, bz>>>(d_new1, d_new, M, Z);
dMatrix *list = new dMatrix[2];
	list[0].dmatrix = d_new1;
	list[1].dmatrix = d_new;
	list[0].M = list[1].M = M - 1;
	list[1].N = list[1].N = Z;
	return list;
}
