#pragma once
#include <cinttypes>

#include "cista/allocator.h"
#include "cista/containers/vecvec.h"
#include "date/date.h"
#include "cista/strong.h"
#include "gpu_types.h"
template <typename K, typename V, typename SizeType = cista::base_t<K>>
using gpu_vecvec = cista::raw::gpu_vecvec<K, V, SizeType>;
extern "C" {
  struct gpu_timetable {
    gpu_delta* route_stop_times_{nullptr};
    gpu_vecvec<gpu_route_idx_t,gpu_value_type,unsigned int>* route_location_seq_ {nullptr};
    gpu_vecvec<gpu_location_idx_t,gpu_route_idx_t,unsigned int>* location_routes_ {nullptr};
    std::uint32_t* n_locations_{nullptr};
    std::uint32_t* n_routes_{nullptr};
    /*
    route_idx_t* route_stop_time_ranges_keys {nullptr};
    interval<std::uint32_t>* route_stop_time_ranges_values {nullptr};
    route_idx_t* location_routes_{nullptr};
    route_idx_t* route_clasz_keys_{nullptr};
    clasz* route_clasz_values_{nullptr};
    location_idx_t* transfer_time_keys_{nullptr};
    u8_minutes* transfer_time_values_{nullptr};
    footpath* footpaths_out_{nullptr};
    stop::value_type* route_location_seq_{nullptr};
    route_idx_t* route_transport_ranges_keys_{nullptr};
    interval<transport_idx_t>* route_transport_ranges_values_{nullptr};
    bitfield_idx_t* bitfields_keys_{nullptr};
    bitfield* bitfields_values_{nullptr};
    transport_idx_t* transport_traffic_days_keys_{nullptr};
    bitfield_idx_t* transport_traffic_days_values_{nullptr};
    char* trip_display_names_{nullptr};
    trip_idx_t* merged_trips_{nullptr};
    merged_trips_idx_t* transport_to_trip_section_{nullptr};
    transport_idx_t* transport_route_keys_{nullptr};
    route_idx_t* transport_route_values_{nullptr};
    route_idx_t* route_stop_time_ranges_keys_keys_{nullptr};
    interval<std::uint32_t>* route_stop_time_ranges_values_{nullptr};
    std::uint32_t n_route_stop_times_{}, n_locations_{},
        n_route_clasz_{}, n_transfer_time_{},
        n_footpaths_out_{}, n_routes_{},
        n_route_transport_ranges_{},n_bitfields_{},
        n_transport_traffic_days_{}, n_trip_display_names_{},
        n_merged_trips_{},n_transport_to_trip_section_{},
        n_transport_routes_{}, n_route_stop_time_ranges_{};
    interval<date::sys_days> internal_interval_days() const {
      return {date_range_.from_ - kTimetableOffset,
              date_range_.to_ + date::days{1}};
    }
    // Schedule range.
    interval<date::sys_days> date_range_{};
     */
#ifdef NIGIRI_CUDA
    __host__ __device__ std::span<gpu_delta const> event_times_at_stop(route_idx_t const r,
                                               stop_idx_t const stop_idx,
                                               event_type const ev_type) const {
      auto const n_transports =
          static_cast<unsigned>(route_transport_ranges_[r].size());
      auto const idx = static_cast<unsigned>(
          route_stop_time_ranges_[r].from_ +
          n_transports * (stop_idx * 2 - (ev_type == event_type::kArr ? 1 : 0)));
      return std::span<gpu_delta const>{&route_stop_times_[idx], n_transports};
    }
#endif
  };

  struct gpu_timetable* create_gpu_timetable(gpu_delta const* route_stop_times,
                                             std::uint32_t n_route_stop_times,
                                             gpu_vecvec<gpu_route_idx_t,gpu_value_type> const* route_location_seq, // Route -> list_of_stops
                                             gpu_vecvec<gpu_location_idx_t , gpu_route_idx_t> const* location_routes, // location -> Route
                                             std::uint32_t const* n_locations
                                             /*,
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
                                             route_idx_t* route_transport_ranges_keys,
                                             interval<transport_idx_t>* route_transport_ranges_values,
                                             std::uint32_t n_route_transport_ranges,
                                             bitfield_idx_t* bitfields_keys,
                                             bitfield* bitfields_values,
                                             std::uint32_t n_bitfields,
                                             transport_idx_t* transport_traffic_days_keys,
                                             bitfield_idx_t* transport_traffic_days_values,
                                             std::uint32_t n_transport_traffic_days,
                                             interval<date::sys_days> date_range, //ich glaube durch dieses intervall müssen wir nicht iterieren bzw. dazu sind intervalle gar nicht gedacht
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
                                             std::uint32_t n_route_stop_time_ranges*/);
  void destroy_gpu_timetable(gpu_timetable* &gtt);
}  // extern "C"