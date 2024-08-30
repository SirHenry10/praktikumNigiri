#pragma once
#include <cstdio>

#include <iostream>
#include <cuda_runtime.h>


extern "C" {
#define cucheck_dev(call)                                    \
{                                                          \
cudaError_t cucheck_err = (call);                        \
if (cucheck_err != cudaSuccess) {                        \
const char* err_str = cudaGetErrorString(cucheck_err); \
printf("%s (%d): %s\n", __FILE__, __LINE__, err_str);  \
}                                                        \
}
#define cuda_check() \
{ cucheck_dev(cudaGetLastError()); }

inline void cuda_sync_stream(cudaStream_t const& stream) {
  cudaEvent_t event{};
  std::cerr << "Test gpu_raptor::launch_kernel() cuda_sync_stream" << std::endl;
  cudaEventCreateWithFlags(&event,
                           cudaEventBlockingSync | cudaEventDisableTiming);
  cudaEventRecord(event, stream);
  std::cerr << "Test gpu_raptor::launch_kernel() cuda_sync_stream1" << std::endl;
  cudaEventSynchronize(event);
  std::cerr << "Test gpu_raptor::launch_kernel() cuda_sync_stream2" << std::endl;
  cudaEventDestroy(event);
    std::cerr << "Test gpu_raptor::launch_kernel() cuda_sync_stream3" << std::endl;
  cuda_check();
}

}  // extern "C"
