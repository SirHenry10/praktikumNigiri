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
                                             nigiri::route_idx_t* location_routes,
                                             std::uint32_t n_locations,
                                             nigiri::route_idx_t* route_clasz_keys,
                                             nigiri::clasz* route_clasz_values,
                                             std::uint32_t n_route_clasz,
                                             nigiri::location_idx_t* transfer_time_keys,
                                             nigiri::u8_minutes* transfer_time_values,
                                             std::uint32_t n_transfer_time,
                                             nigiri::footpath* footpaths_out,
                                             std::uint32_t n_footpaths_out,
                                             stop::value_type* route_location_seq,
                                             std::uint32_t n_route_location_seq,
                                             nigiri::route_idx_t* route_transport_ranges_keys,
                                             nigiri::interval<transport_idx_t>* route_transport_ranges_values,
                                             std::uint32_t n_route_transport_ranges,
                                             nigiri::bitfield_idx_t* bitfields_keys,
                                             nigiri::bitfield* bitfields_values,
                                             std::uint32_t n_bitfields,
                                             nigiri::transport_idx_t* transport_traffic_days_keys,
                                             nigiri::bitfield_idx_t* transport_traffic_days_values,
                                             std::uint32_t n_transport_traffic_days,
                                             interval<date::sys_days> date_range, //ich glaube durch dieses intervall müssen wir nicht iterieren bzw. dazu sind intervalle gar nicht gedacht
                                             char* trip_display_names,
                                             std::uint32_t n_trip_display_names,
                                             nigiri::trip_idx_t* merged_trips,
                                             std::uint32_t n_merged_trips,
                                             nigiri::merged_trips_idx_t* transport_to_trip_section,
                                             std::uint32_t n_transport_to_trip_section,
                                             nigiri::transport_idx_t* transport_route_keys,
                                             nigiri::route_idx_t* transport_route_values,
                                             std::uint32_t n_transport_routes,
                                             nigiri::route_idx_t* route_stop_time_ranges_keys_keys,
                                             nigiri::interval<std::uint32_t>* route_stop_time_ranges_values,
                                             std::uint32_t n_route_stop_time_ranges_);
  void destroy_gpu_timetable(gpu_timetable* &gtt);

}  // extern "C"