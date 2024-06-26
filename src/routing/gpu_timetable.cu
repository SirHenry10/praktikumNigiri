#include "nigiri/routing/gpu_timetable.h"

#include "../../deps/doctest/doctest/doctest.h"

namespace nigiri{
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


//TODO: const machen alles?
  struct gpu_timetable* create_gpu_timetable(gpu_delta const* route_stop_times,
                                             std::uint32_t n_route_stop_times,
                                             route_idx_t* location_routes,
                                             std::uint32_t n_locations,
                                             route_idx_t* route_clasz_keys,
                                             clasz* route_clasz_values,
                                             std::uint32_t n_route_clasz,
                                             location_idx_t* transfer_time_keys,
                                             u8_minutes* transfer_time_values,
                                             std::uint32_t n_transfer_time,
                                             footpath* footpaths_out,
                                             std::uint32_t n_footpaths_out,
                                             stop::value_type* route_location_seq,
                                             std::uint32_t n_routes, //= n_route_location_seq
                                             route_idx_t* route_transport_ranges_keys,
                                             interval<transport_idx_t>* route_transport_ranges_values,
                                             std::uint32_t n_route_transport_ranges,
                                             bitfield_idx_t* bitfields_keys,
                                             bitfield* bitfields_values,
                                             std::uint32_t n_bitfields,
                                             transport_idx_t* transport_traffic_days_keys,
                                             bitfield_idx_t* transport_traffic_days_values,
                                             std::uint32_t n_transport_traffic_days,
                                             interval<date::sys_days> date_range,
                                             char* trip_display_names,
                                             std::uint32_t n_trip_display_names,
                                             trip_idx_t* merged_trips,
                                             std::uint32_t n_merged_trips,
                                             merged_trips_idx_t* transport_to_trip_section,
                                             std::uint32_t n_transport_to_trip_section,
                                             transport_idx_t* transport_route_keys,
                                             route_idx_t* transport_route_values,
                                             std::uint32_t n_transport_routes,
                                             route_idx_t* route_stop_time_ranges_keys_keys,
                                             interval<std::uint32_t>* route_stop_time_ranges_values,
                                             std::uint32_t n_route_stop_time_ranges) {
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
    CUDA_COPY_TO_DEVICE(route_idx_t, gtt->route_stop_time_ranges_keys, route_stop_time_ranges_keys_keys,
                        n_route_stop_time_ranges);
    CUDA_COPY_TO_DEVICE(interval<std::uint32_t>, gtt->route_stop_time_ranges_values, route_stop_time_ranges_values,
                        n_route_stop_time_ranges);
    //location_routes_
    gtt->location_routes_ = nullptr;
    CUDA_COPY_TO_DEVICE(route_idx_t, gtt->location_routes_, location_routes, n_locations);
    //route_clasz
    gtt->route_clasz_keys_ = nullptr;
    gtt->route_clasz_values_= nullptr;
    CUDA_COPY_TO_DEVICE(route_idx_t, gtt->route_clasz_keys_, route_clasz_keys, n_route_clasz);
    CUDA_COPY_TO_DEVICE(clasz, gtt->route_clasz_values_, route_clasz_values, n_route_clasz);
    //transfer_time_
    gtt->transfer_time_keys_ = nullptr;
    gtt->transfer_time_values_ = nullptr;
    CUDA_COPY_TO_DEVICE(location_idx_t, gtt->transfer_time_keys_, transfer_time_keys, n_transfer_time);
    CUDA_COPY_TO_DEVICE(u8_minutes, gtt->transfer_time_values_, transfer_time_values, n_transfer_time);
    //locations_footpaths_out
    gtt->footpaths_out_ = nullptr;
    CUDA_COPY_TO_DEVICE(footpath, gtt->footpaths_out_, footpaths_out, n_footpaths_out);
    //route_location_seq
    gtt->route_location_seq_ = nullptr;
    CUDA_COPY_TO_DEVICE(stop::value_type, gtt->route_location_seq_, route_location_seq, n_routes);
    //route_transport_ranges
    gtt->route_transport_ranges_keys_ = nullptr;
    gtt->route_transport_ranges_values_ = nullptr;
    CUDA_COPY_TO_DEVICE(route_idx_t, gtt->route_transport_ranges_keys_, route_transport_ranges_keys, n_route_transport_ranges);
    CUDA_COPY_TO_DEVICE(interval<transport_idx_t>, gtt->route_transport_ranges_values_, route_transport_ranges_values, n_route_transport_ranges);
    //bitfields_
    gtt->bitfields_keys_ = nullptr;
    gtt->bitfields_values_ = nullptr;
    CUDA_COPY_TO_DEVICE(bitfield_idx_t, gtt->bitfields_keys_, bitfields_keys, n_bitfields);
    CUDA_COPY_TO_DEVICE(bitfield, gtt->bitfields_values_, bitfields_values, n_bitfields);
    //transport_traffic_days_
    gtt->transport_traffic_days_keys_ = nullptr;
    gtt->transport_traffic_days_values_ = nullptr;
    CUDA_COPY_TO_DEVICE(transport_idx_t, gtt->transport_traffic_days_keys_, transport_traffic_days_keys, n_transport_traffic_days);
    CUDA_COPY_TO_DEVICE(bitfield_idx_t, gtt->transport_traffic_days_values_, transport_traffic_days_values, n_transport_traffic_days);
    //date_range_

    //trip_display_names_
    gtt->trip_display_names_ = nullptr;
    CUDA_COPY_TO_DEVICE(char, gtt->trip_display_names_, trip_display_names, n_trip_display_names);
    //merged_trips_
    gtt->merged_trips_ = nullptr;
    CUDA_COPY_TO_DEVICE(trip_idx_t, gtt->merged_trips_, merged_trips, n_merged_trips);
    //transport_to_trip_section_
    gtt->transport_to_trip_section_ = nullptr;
    CUDA_COPY_TO_DEVICE(merged_trips_idx_t, gtt->transport_to_trip_section_, transport_to_trip_section, n_transport_to_trip_section);
    //transport_route_
    gtt->transport_route_keys_ = nullptr;
    gtt->transport_route_values_ = nullptr;
    CUDA_COPY_TO_DEVICE(transport_idx_t, gtt->transport_route_keys_, transport_route_keys, n_transport_routes);
    CUDA_COPY_TO_DEVICE(route_idx_t, gtt->transport_route_values_, transport_route_values, n_transport_routes);
    gtt->n_route_stop_times_ = n_route_stop_times;
    gtt->n_locations_ = n_locations;
    gtt->n_route_clasz_ = n_route_clasz;
    gtt->n_transfer_time_ = n_transfer_time;
    gtt->n_footpaths_out_ = n_footpaths_out;
    gtt->n_routes_ = n_routes;
    gtt->n_route_transport_ranges_ = n_route_transport_ranges;
    gtt->n_bitfields_ = n_bitfields;
    gtt->n_transport_traffic_days_ = n_transport_traffic_days;
    gtt->n_trip_display_names_ = n_trip_display_names;
    gtt->n_merged_trips_ = n_merged_trips;
    gtt->n_transport_to_trip_section_ = n_transport_to_trip_section;
    gtt->n_route_transport_ranges_ = n_transport_routes;
    gtt->n_route_stop_time_ranges_ = n_route_stop_time_ranges;
    gtt->date_range_ = date_range;
    return gtt;

  fail:
    //route_stop_times
    cudaFree(gtt->route_stop_times_);
    //route_stop_time_ranges
    cudaFree(gtt->route_stop_time_ranges_keys);
    cudaFree(gtt->route_stop_time_ranges_values);
    //location_routes_
    cudaFree(gtt->location_routes_);
    //route_clasz
    cudaFree(gtt->route_clasz_keys_);
    cudaFree(gtt->route_clasz_values_);
    //location_transfer_time_count
    cudaFree(gtt->transfer_time_keys_);
    cudaFree(gtt->transfer_time_values_);
    //locations_footpaths_out
    cudaFree(gtt->footpaths_out_);
    //route_location_seq
    cudaFree(gtt->route_location_seq_);
    //route_transport_ranges
    cudaFree(gtt->route_transport_ranges_keys_);
    cudaFree(gtt->route_transport_ranges_values_);
    //bitfields_
    cudaFree(gtt->bitfields_keys_);
    cudaFree(gtt->bitfields_values_);
    //transport_traffic_days_
    cudaFree(gtt->transport_traffic_days_keys_);
    cudaFree(gtt->transport_traffic_days_values_);
    //date_range_

    //trip_display_names_
    cudaFree(gtt->trip_display_names_);
    //merged_trips_
    cudaFree(gtt->merged_trips_);
    //transport_to_trip_section_
    cudaFree(gtt->transport_to_trip_section_);
    //transport_route_
    cudaFree(gtt->transport_route_keys_);
    cudaFree(gtt->transport_route_values_);

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
      cudaFree(gtt->location_routes_);
      //route_clasz
      cudaFree(gtt->route_clasz_keys_);
      cudaFree(gtt->route_clasz_values_);
      //location_transfer_time_count
      cudaFree(gtt->transfer_time_keys_);
      cudaFree(gtt->transfer_time_values_);
      //locations_footpaths_out
      cudaFree(gtt->footpaths_out_);
      //route_location_seq
      cudaFree(gtt->route_location_seq_);
      //route_transport_ranges
      cudaFree(gtt->route_transport_ranges_keys_);
      cudaFree(gtt->route_transport_ranges_values_);
      //bitfields_
      cudaFree(gtt->bitfields_keys_);
      cudaFree(gtt->bitfields_values_);
      //transport_traffic_days_
      cudaFree(gtt->transport_traffic_days_keys_);
      cudaFree(gtt->transport_traffic_days_values_);
      //date_range_

      //trip_display_names_
      cudaFree(gtt->trip_display_names_);
      //merged_trips_
      cudaFree(gtt->merged_trips_);
      //transport_to_trip_section_
      cudaFree(gtt->transport_to_trip_section_);
      //transport_route_
      cudaFree(gtt->transport_route_keys_);
      cudaFree(gtt->transport_route_values_);

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

//TODO: help methode to convert tt in gtt??
} //namespace nigiri