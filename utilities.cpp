#ifndef UTILITIES_CPP
#define UTILITIES_CPP

#include "utilities.h"

int str_to_int(std::string str) {
	int i = 0, integer = 0;
	while(str[i] != '\0') {
		integer = integer * 10 + str[i] - '0';
		i++;
	}
	return integer;
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
	int M, N;
	fp >> M >> N;
	
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
		T std = 0;
		for (int i = 0; i < M - 1; i++) {
			hX[i * N + j] -= mean;
			std += hX[i * N + j] * hX[i * N + j];
		}
		std /= M - 2;
		std = sqrt(std);
		T norm = 0;
		for (int i = 0; i < M - 1; i++) {
			hX[i * N + j] /= std;
			norm += hX[i * N + j] * hX[i * N + j];
		}
		norm = sqrt(norm);
		for (int i = 0; i < M - 1; i++) {
			hX[i * N + j] /= norm;
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
		T std = 0;
		for (int i = 0; i < M - 1; i++) {
			hY[i * N + j] -= mean;
			std += hY[i * N + j] * hY[i * N + j];
		}
		std /= M - 2;
		std = sqrt(std);
		T norm = 0;
		for (int i = 0; i < M - 1; i++) {
			hY[i * N + j] /= std;
			norm += hY[i * N + j] * hY[i * N + j];
		}
		norm = sqrt(norm);
		for (int i = 0; i < M - 1; i++) {
			hY[i * N + j] /= norm;
		}
	}

	cudaMemcpy(Y, hY, (M - 1) * N * sizeof(T), cudaMemcpyHostToDevice);

	IntegerTuple tuple;
	tuple.M = M - 1;
	tuple.N = N;
	return tuple;
}

template IntegerTuple read_flat_mri<float>(std::string, float *&X, float *&Y);
template IntegerTuple read_flat_mri<double>(std::string, double *&X, double *&Y);

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
