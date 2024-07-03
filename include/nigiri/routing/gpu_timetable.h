#pragma once

#include <cinttypes>

#include "cista/containers/vecvec.h"


#include <compare>
#include <filesystem>
#include <span>
#include <type_traits>

#include "cista/memory_holder.h"
#include "cista/reflection/printable.h"

#include "utl/verify.h"
#include "utl/zip.h"

#include "geo/latlng.h"

#include "nigiri/common/interval.h"
#include "nigiri/footpath.h"
#include "nigiri/location.h"
#include "nigiri/logging.h"
#include "nigiri/stop.h"
#include "nigiri/types.h"

namespace nigiri{
extern "C" {

  struct gpu_delta_t {
    std::uint16_t days_ : 5;
    std::uint16_t mam_ : 11;
  };
  struct gpu_timetable {
    gpu_delta_t* route_stop_times_{nullptr};
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
  };

  struct gpu_timetable* create_gpu_timetable(gpu_delta_t const* route_stop_times,
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
                                             std::uint32_t n_routes,
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
                                             std::uint32_t n_route_stop_time_ranges);
  void destroy_gpu_timetable(gpu_timetable* &gtt);

  inline unixtime_t gpu_delta_to_unix(gpu_delta_t d) {
    return d.days_ +
           d.mam_ * unixtime_t::duration{1};
  }

  bool operator== (const gpu_delta_t& a, const gpu_delta_t& b){
    return (a.days_==b.days_ && a.mam_==b.mam_);
  }
}  // extern "C"
}  //namespace nigiri