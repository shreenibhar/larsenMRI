#ifndef UTILITIES_H
#define UTILITIES_H

#include <bits/stdc++.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand.h>

#include <thrust/device_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>

typedef float precision;
typedef float corr_precision;

#define inf 50000
#define eps 1e-9

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

template<typename T>
IntegerTuple read_flat_mri(std::string path, T *&X, T *&Y);

template<typename T>
void printDeviceVar(T *var, int size, int *ind, int numInd, int breaker = 1);

void range(int *&ind, int st, int ed);

template<typename T>
void init_var(T *&var, int size);

void cudaErrorFlush();

#endif
