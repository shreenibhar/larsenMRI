#include <iostream> 

using namespace std;

int main() {
  int nDevices;
  cudaGetDeviceCount(&nDevices);
  for (int i = 0; i < nDevices; i++) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);
    cout << "Device Number: " << i << endl;
    cout << "  Device Name: " << prop.name << endl;
    cout << "  Device Clock Rate: " << prop.clockRate << endl;
    cout << "  Device Shared Memory per Block: " << prop.sharedMemPerBlock << endl;
    cout << "  Device Regs per Block: " << prop.regsPerBlock << endl;
    cout << "  Device Warp Size: " << prop.warpSize << endl;
    cout << "  Device Mem Pitch: " << prop.memPitch << endl;
    cout << "  Device Max Threads per Block: " << prop.maxThreadsPerBlock << endl;
    cout << "  Device Max Threads Dim: " << prop.maxThreadsDim[0] << ' ' << prop.maxThreadsDim[1] << ' ' << prop.maxThreadsDim[2] << endl;
    cout << "  Device Max Grid Size: " << prop.maxGridSize[0] << ' ' << prop.maxGridSize[1] << ' ' << prop.maxGridSize[2] << endl;
    cout << "  Device Total Const Memory: " << prop.totalConstMem << endl;
    cout << "  Device Major: " << prop.major << endl;
    cout << "  Device Minor: " << prop.minor << endl;
    cout << "  Device Clock Rate: " << prop.clockRate << endl;
    cout << "  Device Texture Allingment: " << prop.textureAllignment << endl;
    cout << "  Device Device Overlap: " << prop.deviceOverlap << endl;

    printf("  Memory Bus Width (bits): %d\n",
           prop.memoryBusWidth);
    printf("  Peak Memory Bandwidth (GB/s): %f\n\n",
           2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
  }
  return 0;
}
