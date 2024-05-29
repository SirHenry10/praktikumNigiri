#pragma once

#include <cinttypes>

#include "cista/containers/vecvec.h"

#include <nigiri/types.h>

extern "C" {

  struct gpu_timetable;

  struct gpu_delta {
    std::uint16_t days_ : 5;
    std::uint16_t mam_ : 11;
  };

  struct gpu_timetable* create_gpu_timetable(gpu_delta const* route_stop_times,
                                           std::uint32_t n_route_stop_times,
                                           nigiri::route_idx_t* gpu_route_stop_time_ranges_keys_keys,
                                           nigiri::interval<std::uint32_t>* gpu_route_stop_time_ranges_values,
                                           std::uint32_t n_gpu_route_stop_time_ranges_);
  void destroy_gpu_timetable(gpu_timetable* &gtt);

}  // extern "C"