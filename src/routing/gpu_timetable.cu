
#include "nigiri/routing/gpu_timetable.h"
// todo: types.h include nicht möglich erzeugt hashing error
#include <cstdio>

extern "C" {

#define XSTR(s) STR(s)
#define STR(s) #s

#define CUDA_CALL(call)                                   \
    if ((code = call) != cudaSuccess) {                     \
      printf("CUDA error: %s at " STR(call) " %s:%d\n",     \
             cudaGetErrorString(code), __FILE__, __LINE__); \
      goto fail;                                            \
    }

#define CUDA_COPY_TO_DEVICE(type, target, source, size)                        \
    CUDA_CALL(cudaMalloc(&target, size * sizeof(type)))                          \
    CUDA_CALL(                                                                   \
        cudaMemcpy(target, source, size * sizeof(type), cudaMemcpyHostToDevice)) \
    device_bytes += size * sizeof(type);

// TODO: const machen alles?
struct gpu_timetable* create_gpu_timetable(gpu_delta const* route_stop_times,
                                           std::uint32_t  n_route_stop_times,
                                           gpu_vecvec<gpu_route_idx_t,gpu_value_type> const* route_location_seq,
                                           gpu_vecvec<gpu_location_idx_t , gpu_route_idx_t> const* location_routes,
                                           std::uint32_t const* n_locations,
                                           std::uint32_t const* n_routes,
                                           gpu_vector_map<gpu_route_idx_t,gpu_interval<std::uint32_t>> const* route_stop_time_ranges,
                                           gpu_vector_map<gpu_route_idx_t,gpu_interval<gpu_transport_idx_t >> const* route_transport_ranges,
                                           gpu_vector_map<gpu_bitfield_idx_t, gpu_bitfield> const* bitfields,
                                           gpu_vector_map<gpu_transport_idx_t,gpu_bitfield_idx_t> const* transport_traffic_days,
                                           gpu_interval<date::sys_days> const* date_range,
                                           gpu_locations const* locations,
                                           gpu_vector_map<gpu_route_idx_t, gpu_clasz> const* route_clasz) {
  size_t device_bytes = 0U;

  cudaError_t code;
  gpu_timetable* gtt =
      static_cast<gpu_timetable*>(malloc(sizeof(gpu_timetable)));
  if (gtt == nullptr) {
    printf("nigiri gpu raptor: malloc for gpu_timetable failed\n");
    return nullptr;
  }
  // route_stop_times_
  gtt->route_stop_times_ = nullptr;
  CUDA_COPY_TO_DEVICE(gpu_delta, gtt->route_stop_times_, route_stop_times,
                      n_route_stop_times);
  //route_location_seq
  gtt->route_location_seq_ = nullptr;
  using gpu_vecvec_route_value = gpu_vecvec<gpu_route_idx_t,gpu_value_type>;
  CUDA_COPY_TO_DEVICE(gpu_vecvec_route_value , gtt->route_location_seq_,
                      route_location_seq, 1);
  //location_routes_
  gtt->location_routes_ = nullptr;
  using gpu_vecvec_location_route = gpu_vecvec<gpu_location_idx_t , gpu_route_idx_t>;
  CUDA_COPY_TO_DEVICE(gpu_vecvec_location_route, gtt->location_routes_, location_routes,1);
  //n_locations_
  gtt->n_locations_ = nullptr;
  CUDA_COPY_TO_DEVICE(uint32_t , gtt->n_locations_, n_locations,1);
  //n_routes_
  gtt->n_routes_ = nullptr;
  CUDA_COPY_TO_DEVICE(uint32_t , gtt->n_routes_, n_routes,1);
  //route_stop_time_ranges_
  gtt->route_stop_time_ranges_ = nullptr;
  using gpu_vecmap_stop_time_ranges = gpu_vector_map<gpu_route_idx_t,gpu_interval<std::uint32_t>>;
  CUDA_COPY_TO_DEVICE(gpu_vecmap_stop_time_ranges , gtt->route_stop_time_ranges_, route_stop_time_ranges,1);
  //route_transport_ranges_
  gtt->route_transport_ranges_ = nullptr;
  using gpu_vecmap_route_transport_ranges = gpu_vector_map<gpu_route_idx_t,gpu_interval<gpu_transport_idx_t >>;
  CUDA_COPY_TO_DEVICE(gpu_vecmap_route_transport_ranges , gtt->route_transport_ranges_, route_transport_ranges,1);
  //bitfields_
  gtt->bitfields_ = nullptr;
  using gpu_vecmap_bitfields = gpu_vector_map<gpu_bitfield_idx_t, gpu_bitfield>;
  CUDA_COPY_TO_DEVICE(gpu_vecmap_bitfields, gtt->bitfields_, bitfields,1);
  //transport_traffic_days_
  gtt->transport_traffic_days_ = nullptr;
  using gpu_vecmap_transport_traffic_days = gpu_vector_map<gpu_transport_idx_t,gpu_bitfield_idx_t>;
  CUDA_COPY_TO_DEVICE(gpu_vecmap_transport_traffic_days, gtt->transport_traffic_days_, transport_traffic_days,1);
  //date_range_
  gtt->date_range_ = nullptr;
  using gpu_date_range = gpu_interval<date::sys_days>;
  CUDA_COPY_TO_DEVICE(gpu_date_range , gtt->date_range_, date_range,1);
  //locations_
  gtt->locations_ = nullptr;
  CUDA_COPY_TO_DEVICE(gpu_locations , gtt->locations_, locations,1);
  //route_clasz_
  gtt->route_clasz_ = nullptr;
  using gpu_vector_map_clasz = gpu_vector_map<gpu_route_idx_t, gpu_clasz>;
  CUDA_COPY_TO_DEVICE(gpu_vector_map_clasz, gtt->route_clasz_, route_clasz,1);
  return gtt;


fail:
  cudaFree(gtt->route_stop_times_);
  cudaFree(gtt->route_location_seq_);
  cudaFree(gtt->location_routes_);
  cudaFree(gtt->n_locations_);
  cudaFree(gtt->n_routes_);
  cudaFree(gtt->route_stop_time_ranges_);
  cudaFree(gtt->route_transport_ranges_);
  cudaFree(gtt->bitfields_);
  cudaFree(gtt->transport_traffic_days_);
  cudaFree(gtt->date_range_);
  cudaFree(gtt->locations_);
  cudaFree(gtt->route_clasz_);
  free(gtt);
  return nullptr;
}
void destroy_gpu_timetable(gpu_timetable*& gtt) {
  cudaFree(gtt->route_stop_times_);
  cudaFree(gtt->route_location_seq_);
  cudaFree(gtt->location_routes_);
  cudaFree(gtt->n_locations_);
  cudaFree(gtt->n_routes_);
  cudaFree(gtt->route_stop_time_ranges_);
  cudaFree(gtt->route_transport_ranges_);
  cudaFree(gtt->bitfields_);
  cudaFree(gtt->transport_traffic_days_);
  cudaFree(gtt->date_range_);
  cudaFree(gtt->locations_);
  cudaFree(gtt->route_clasz_);
  free(gtt);
  gtt = nullptr;
  cudaDeviceSynchronize();
  auto const last_error = cudaGetLastError();
  if (last_error != cudaSuccess) {
    printf("CUDA error: %s at " STR(last_error) " %s:%d\n",
           cudaGetErrorString(last_error), __FILE__, __LINE__);
  }
}
}  // extern "C"