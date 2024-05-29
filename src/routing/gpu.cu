#include "nigiri/routing/gpu.h"

#include <cstdio>

#include "../../deps/doctest/doctest/doctest.h"

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

  struct gpu_timetable {
    gpu_delta* route_stop_times_{nullptr};
    nigiri::route_idx_t* route_stop_time_ranges_keys {nullptr};
    nigiri::interval<std::uint32_t>* route_stop_time_ranges_values {nullptr};
  };

  struct gpu_timetable* create_gpu_timetable(gpu_delta const* route_stop_times,
                                             std::uint32_t n_route_stop_times,
                                             nigiri::route_idx_t* route_stop_time_ranges_keys_keys,
                                             nigiri::interval<std::uint32_t>* route_stop_time_ranges_values,
                                             std::uint32_t n_route_stop_time_ranges_) {
    size_t device_bytes = 0U;

    cudaError_t code;
    gpu_timetable* gtt =
        static_cast<gpu_timetable*>(malloc(sizeof(gpu_timetable)));
    if (gtt == nullptr) {
      printf("nigiri gpu raptor: malloc for gpu_timetable failed\n");
      return nullptr;
    }
    //route_stop_times_
    gtt->route_stop_times_ = nullptr;

    CUDA_COPY_TO_DEVICE(gpu_delta, gtt->route_stop_times_, route_stop_times,
                        n_route_stop_times);
    //route_stop_time_ranges
    gtt->route_stop_time_ranges_keys = nullptr;
    gtt->route_stop_time_ranges_values = nullptr;
    CUDA_COPY_TO_DEVICE(nigiri::route_idx_t, gtt->route_stop_time_ranges_keys, route_stop_time_ranges_keys_keys,
                        n_route_stop_time_ranges_);
    CUDA_COPY_TO_DEVICE(nigiri::interval<std::uint32_t>, gtt->route_stop_time_ranges_values, route_stop_time_ranges_values,
                        n_route_stop_time_ranges_);
    //location_routes_

    //route_clasz

    //location_transfer_time_count

    //locations_footpaths_out

    //route_location_seq

    //route_transport_ranges

    //bitfields_

    //transport_traffic_days_

    //date_range_

    //trip_display_names_

    //merged_trips_

    //transport_to_trip_section_

    //transport_route_

    return gtt;

  fail:
    //route_stop_times
    cudaFree(gtt->route_stop_times_);
    //route_stop_time_ranges
    cudaFree(gtt->route_stop_time_ranges_keys);
    cudaFree(gtt->route_stop_time_ranges_values);
    //location_routes_

    //route_clasz

    //location_transfer_time_count

    //locations_footpaths_out

    //route_location_seq

    //route_transport_ranges

    //bitfields_

    //transport_traffic_days_

    //date_range_

    //trip_display_names_

    //merged_trips_

    //transport_to_trip_section_

    //transport_route_
    free(gtt);
    return nullptr;
  }
  void destroy_gpu_timetable(gpu_timetable* &gtt) {
      //route_stop_times_
      cudaFree(gtt->route_stop_times_);
      //route_stop_time_ranges
      cudaFree(gtt->route_stop_time_ranges_keys);
      cudaFree(gtt->route_stop_time_ranges_values);
      //location_routes_

      //route_clasz

      //location_transfer_time_count

      //locations_footpaths_out

      //route_location_seq

      //route_transport_ranges

      //bitfields_

      //transport_traffic_days_

      //date_range_

      //trip_display_names_

      //merged_trips_

      //transport_to_trip_section_

      //transport_route_
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