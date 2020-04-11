#ifndef UTILITIES_CPP
#define UTILITIES_CPP

#include "utilities.h"

template<typename T>
void printDeviceVar(T *var, int size, int *ind, int numInd, int breaker) {
	T *hVar = new T[size];
	cudaMemcpy(hVar, var, size * sizeof(T), cudaMemcpyDeviceToHost);
	for (int i = 0; i < numInd; i++) {
		if (i % breaker == 0) printf("\n");
		if (typeid(T).name()[0] == 'i') printf("%6d ", hVar[ind[i]]);
		else printf("%6.3f ", hVar[ind[i]]);
	}
	printf("\n");
}

template void printDeviceVar<int>(int *var, int size, int *ind, int numInd, int breaker);
template void printDeviceVar<float>(float *var, int size, int *ind, int numInd, int breaker);
template void printDeviceVar<double>(double *var, int size, int *ind, int numInd, int breaker);

void range(int *&ind, int st, int ed) {
	ind = new int[ed - st];
	for (int i = st; i < ed; i++) ind[i - st] = i;
}

GpuTimer::GpuTimer() {
	cudaEventCreate(&startTime);
	cudaEventCreate(&stopTime);
}
	
GpuTimer::~GpuTimer() {
	cudaEventDestroy(startTime);
	cudaEventDestroy(stopTime);
}

void GpuTimer::start() {
	cudaEventRecord(startTime);
}
void GpuTimer::stop() {
	cudaEventRecord(stopTime);
}
	
float GpuTimer::elapsed() {
	cudaEventSynchronize(stopTime);
	float mill = 0;
	cudaEventElapsedTime(&mill, startTime, stopTime);
	return mill;
}

template<typename T>
IntegerTuple read_flat_mri(std::string path, T *&X, T *&Y) {
	std::ifstream fp(path.c_str());
	char hash;
	int M, N;
	fp >> hash >> M >> N;
	
	T *h_number = new T[M * N];
	for (int i = 0; i < M; i++) {
		for (int j = 0; j < N; j++) {
			fp >> h_number[i * N + j];
		}
	}
	fp.close();

	cudaMalloc(&X, (M - 1) * N * sizeof(T));
	T *hX = new T[(M - 1) * N];

	for (int j = 0; j < N; j++) {
		T mean = 0;
		for (int i = 0; i < M - 1; i++) {
			hX[i * N + j] = h_number[i * N + j];
			mean += hX[i * N + j];
		}
		mean /= M - 1;
		T nrm = 0;
		for (int i = 0; i < M - 1; i++) {
			hX[i * N + j] -= mean;
			nrm += hX[i * N + j] * hX[i * N + j];
		}
		nrm = sqrt(nrm);
		for (int i = 0; i < M - 1; i++) {
			hX[i * N + j] /= nrm;
		}
	}

	cudaMemcpy(X, hX, (M - 1) * N * sizeof(T), cudaMemcpyHostToDevice);

	cudaMalloc(&Y, (M - 1) * N * sizeof(T));
	T *hY = new T[(M - 1) * N];
	
	for (int j = 0; j < N; j++) {
		T mean = 0;
		for (int i = 1; i < M; i++) {
			hY[(i - 1) * N + j] = h_number[i * N + j];
			mean += hY[(i - 1) * N + j];
		}
		mean /= M - 1;
		T nrm = 0;
		for (int i = 0; i < M - 1; i++) {
			hY[i * N + j] -= mean;
			nrm += hY[i * N + j] * hY[i * N + j];
		}
		nrm = sqrt(nrm);
		for (int i = 0; i < M - 1; i++) {
			hY[i * N + j] /= nrm;
		}
	}

	cudaMemcpy(Y, hY, (M - 1) * N * sizeof(T), cudaMemcpyHostToDevice);

	IntegerTuple tuple;
	tuple.M = M - 1;
	tuple.N = N;
	return tuple;
}

template IntegerTuple read_flat_mri<float>(std::string path, float *&X, float *&Y);
template IntegerTuple read_flat_mri<double>(std::string path, double *&X, double *&Y);

template<typename T>
void init_var(T *&var, int size) {
	cudaMalloc(&var, size * sizeof(T));
}

template void init_var<int>(int *&var, int size);
template void init_var<float>(float *&var, int size);
template void init_var<double>(double *&var, int size);

void cudaErrorFlush() {
	cudaDeviceSynchronize();
	cudaError_t error = cudaGetLastError();
	printf("Cuda Error (%s): %s\n", cudaGetErrorName(error), cudaGetErrorString(error));
}

#endif
