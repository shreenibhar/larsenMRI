#ifndef UTILITIES_H
#define UTILITIES_H

#include <bits/stdc++.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define inf 50000

class IntegerTuple {
public:
	int M, N;
};

template<typename T>
class Debug {
public:
	T a1;
	T a2;
	int nVars;
	Debug() {a1 = -1; a2 = -1; nVars = -1;};
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

int str_to_int(std::string str);

int optimalBlock1D(int problemSize);

template<typename T>
void printDeviceVar(T *var, int size, int *ind, int numInd, int breaker = 1);

void range(int *&ind, int st, int ed);

template<typename T>
void init_var(T *&var, int size);

void cudaErrorFlush();

#endif
