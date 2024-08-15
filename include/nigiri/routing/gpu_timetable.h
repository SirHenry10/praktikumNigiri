#pragma once
#include <cinttypes>

#include "cista/allocator.h"
#include "cista/containers/vecvec.h"
#include "date/date.h"
#include "cista/strong.h"
#include "gpu_types.h"
#include <span>
template <typename T>
using gpu_interval = nigiri::gpu_interval<T>;
extern "C" {

  struct gpu_timetable {
    gpu_delta* route_stop_times_{nullptr};
    gpu_vecvec<gpu_route_idx_t,gpu_value_type,unsigned int>* route_location_seq_ {nullptr};
    gpu_vecvec<gpu_location_idx_t,gpu_route_idx_t,unsigned int>* location_routes_ {nullptr};
    std::uint32_t* n_locations_{nullptr};
    std::uint32_t* n_routes_{nullptr};
    gpu_vector_map<gpu_route_idx_t,gpu_interval<std::uint32_t>>* route_stop_time_ranges_{nullptr};
    gpu_vector_map<gpu_route_idx_t,gpu_interval<gpu_transport_idx_t >>* route_transport_ranges_{nullptr};
    gpu_vector_map<gpu_bitfield_idx_t, gpu_bitfield>* bitfields_{nullptr};
    gpu_vector_map<gpu_bitfield_idx_t,std::uint64_t*>* bitfields_data_{nullptr};
    gpu_vector_map<gpu_transport_idx_t,gpu_bitfield_idx_t>* transport_traffic_days_{nullptr};
    gpu_interval<gpu_sys_days>* date_range_{nullptr};
    gpu_locations* locations_{nullptr};
    gpu_vector_map<gpu_route_idx_t, gpu_clasz>* route_clasz_{nullptr};
#ifdef NIGIRI_CUDA
    __host__ __device__ std::span<gpu_delta const> event_times_at_stop(gpu_route_idx_t const r,
                                               gpu_stop_idx_t const stop_idx,
                                               gpu_event_type const ev_type) const {
      auto rtr = *route_transport_ranges_;
      auto const n_transports =
          static_cast<unsigned>(rtr[r].size());
      auto const idx = static_cast<unsigned>(
          rtr[r].from_ +
          n_transports * (stop_idx * 2 - (ev_type == gpu_event_type::kArr ? 1 : 0)));
      return std::span<gpu_delta const>{&route_stop_times_[idx], n_transports};
    }
    __host__ __device__ gpu_interval<gpu_sys_days> gpu_internal_interval_days() const {
      auto date_range = *date_range_;
      return {date_range.from_ - (gpu_days{1} + gpu_days{4}),
              date_range.to_ + gpu_days{1}};
    }
#endif

  };

  struct gpu_timetable* create_gpu_timetable(gpu_delta const* route_stop_times,
                                             std::uint32_t n_route_stop_times,
                                             gpu_vecvec<gpu_route_idx_t,gpu_value_type> const* route_location_seq, // Route -> list_of_stops
                                             gpu_vecvec<gpu_location_idx_t , gpu_route_idx_t> const* location_routes, // location -> Route
                                             std::uint32_t const* n_locations,
                                             std::uint32_t const* n_routes,
                                             gpu_vector_map<gpu_route_idx_t,gpu_interval<std::uint32_t>> const* route_stop_time_ranges,
                                             gpu_vector_map<gpu_route_idx_t,gpu_interval<gpu_transport_idx_t >> const* route_transport_ranges,
                                             gpu_vector_map<gpu_bitfield_idx_t, gpu_bitfield> const* bitfields,
                                             gpu_vector_map<gpu_bitfield_idx_t,std::uint64_t*> const* bitfields_data_, //TODO: müssen dann jedes gpu_bitfield.data nehmen und in eine vector_map ändern
                                             gpu_vector_map<gpu_transport_idx_t,gpu_bitfield_idx_t> const* transport_traffic_days,
                                             gpu_interval<gpu_sys_days> const* date_range,
                                             gpu_locations const* locations,
                                             gpu_vector_map<gpu_route_idx_t, gpu_clasz> const* route_clasz);
  void destroy_gpu_timetable(gpu_timetable* &gtt);
}  // extern "C"