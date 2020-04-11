#ifndef UTILITIES_INL_H
#define UTILITIES_INL_H

#include "headers.h"


// Larsen variables.
template<typename ProcPrec, typename CorrPrec>
class Variable {
public:
  // Facility to allocate using constructor.
  Variable (int N=0) {
    if (N) {
      AllocProc(N);
    }
  }

  // Allocating memory.
  void AllocProc(int N) {
    hVar.resize(N);
    dVar.resize(N);
  }

  void AllocCorr() {
    hVar_.resize(hVar.size());
    dVar_.resize(dVar.size());
  }

  // Sync functions which copy over host to device or device to host.
  void SyncHost() {
    thrust::copy(dVar.begin(), dVar.end(), hVar.begin());
  }

  void SyncHost_() {
    thrust::copy(dVar_.begin(), dVar_.end(), hVar_.begin());
  }

  void SyncDev() {
    thrust::copy(hVar.begin(), hVar.end(), dVar.begin());
  }

  void SyncDev_() {
    thrust::copy(hVar_.begin(), hVar_.end(), dVar_.begin());
  }

  // Converting to raw pointers.
  ProcPrec * GetRawDevPtr() {
    return thrust::raw_pointer_cast(dVar.data());
  }

  CorrPrec * GetRawDevPtr_() {
    return thrust::raw_pointer_cast(dVar_.data());
  }

  // Variables used for Processing.
  thrust::host_vector<ProcPrec> hVar;
  thrust::device_vector<ProcPrec> dVar;

  // Variables used for Correction.
  thrust::host_vector<CorrPrec> hVar_;
  thrust::device_vector<CorrPrec> dVar_;
};


// Timer class.
class GpuTimer {
private:
  cudaEvent_t startTime, stopTime;
public:
  GpuTimer() {
    cudaEventCreate(&startTime);
    cudaEventCreate(&stopTime);
  }

  ~GpuTimer() {
    cudaEventDestroy(startTime);
    cudaEventDestroy(stopTime);
  }

  void start() {
    cudaEventRecord(startTime);
  }

  void stop() {
    cudaEventRecord(stopTime);
  }

  float elapsed() {
    cudaEventSynchronize(stopTime);
    float mill = 0;
    cudaEventElapsedTime(&mill, startTime, stopTime);
    return mill;
  }
};


// Reads the flattened fMRI image and generates X, Y matrices (mean, norm2 normalization).
template<typename ProcPrec, typename CorrPrec>
std::tuple<int, int> ReadFlatMri(std::string path, Variable<ProcPrec, CorrPrec> &X, Variable<ProcPrec, CorrPrec> &Y) {
  std::ifstream fp(path.c_str());
  char ch;
  int M, N;
  fp >> ch >> M >> N;

  CorrPrec *h_number = new CorrPrec[M * N];
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      fp >> h_number[i * N + j];
    }
  }
  fp.close();

  X.AllocProc((M - 1) * N);
  X.AllocCorr();

  for (int j = 0; j < N; j++) {
    CorrPrec mean = 0;
    for (int i = 0; i < M - 1; i++) {
      X.hVar_[i * N + j] = h_number[i * N + j];
      mean += h_number[i * N + j];
    }
    mean /= M - 1;
    CorrPrec nrm = 0;
    for (int i = 0; i < M - 1; i++) {
      X.hVar_[i * N + j] -= mean;
      nrm += X.hVar_[i * N + j] * X.hVar_[i * N + j];
    }
    nrm = sqrt(nrm);
    for (int i = 0; i < M - 1; i++) {
      X.hVar_[i * N + j] /= nrm;
      X.hVar[i * N + j] = X.hVar_[i * N + j];
    }
  }

  Y.AllocProc((M - 1) * N);
  Y.AllocCorr();

  for (int j = 0; j < N; j++) {
    CorrPrec mean = 0;
    for (int i = 1; i < M; i++) {
      Y.hVar_[(i - 1) * N + j] = h_number[i * N + j];
      mean += h_number[i * N + j];
    }
    mean /= M - 1;
    CorrPrec nrm = 0;
    for (int i = 0; i < M - 1; i++) {
      Y.hVar_[i * N + j] -= mean;
      nrm += Y.hVar_[i * N + j] * Y.hVar_[i * N + j];
    }
    nrm = sqrt(nrm);
    for (int i = 0; i < M - 1; i++) {
      Y.hVar_[i * N + j] /= nrm;
      Y.hVar[i * N + j] = Y.hVar_[i * N + j];
    }
  }

  X.SyncDev();
  X.SyncDev_();

  Y.SyncDev();
  Y.SyncDev_();

  return std::make_tuple(M - 1, N);
}


// Cuda error flushing with display.
void cudaErrorFlush() {
  cudaDeviceSynchronize();
  cudaError_t error = cudaGetLastError();
  printf("Cuda Error (%s): %s\n", cudaGetErrorName(error), cudaGetErrorString(error));
}


#endif
