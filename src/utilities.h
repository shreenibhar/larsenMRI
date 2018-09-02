#ifndef UTILITIES_H
#define UTILITIES_H

#include <bits/stdc++.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

typedef float precision;
typedef double corr_precision;

#define inf 50000
#define eps 1e-6

class IntegerTuple {
public:
	int M, N;
};

class GpuTimer {
private:
	cudaEvent_t startTime, stopTime;
public:
	GpuTimer();

	~GpuTimer();

	void start();

	void stop();

	float elapsed();       
};

IntegerTuple read_flat_mri(std::string path, precision *&X, precision *&Y);

template<typename T>
void printDeviceVar(T *var, int size, int *ind, int numInd, int breaker = 1);

void range(int *&ind, int st, int ed);

template<typename T>
void init_var(T *&var, int size);

void cudaErrorFlush();

#endif
