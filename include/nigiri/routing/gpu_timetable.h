#pragma once
#include <cinttypes>

#include "cista/containers/vecvec.h"
#include "date/date.h"

struct gpu_delta{
  std::uint16_t days_ : 5;
  std::uint16_t mam_ : 11;
  bool operator== (gpu_delta const& a) const{
    return (a.days_== this->days_ && a.mam_==this->mam_);
  }
  bool operator!= (gpu_delta const& a) const{
    return !(operator==(a));
  }
};
//TODO: später raus kicken was nicht brauchen

using vecvec<> = cista::raw::vecvec<>;
using gpu_location_idx_t = cista::strong<std::uint32_t, struct _location_idx>;
using gpu_value_type = gpu_location_idx_t::value_t;
using gpu_bitfield_idx_t = cista::strong<std::uint32_t, struct _bitfield_idx>;
using gpu_location_idx_t = cista::strong<std::uint32_t, struct _location_idx>;
using gpu_route_idx_t = cista::strong<std::uint32_t, struct _route_idx>;
using gpu_section_idx_t = cista::strong<std::uint32_t, struct _section_idx>;
using gpu_section_db_idx_t = cista::strong<std::uint32_t, struct _section_db_idx>;
using gpu_trip_idx_t = cista::strong<std::uint32_t, struct _trip_idx>;
using gpu_trip_id_idx_t = cista::strong<std::uint32_t, struct _trip_id_str_idx>;
using gpu_transport_idx_t = cista::strong<std::uint32_t, struct _transport_idx>;
using gpu_source_idx_t = cista::strong<std::uint16_t, struct _source_idx>;
using gpu_day_idx_t = cista::strong<std::uint16_t, struct _day_idx>;
using gpu_timezone_idx_t = cista::strong<std::uint16_t, struct _timezone_idx>;

enum class gpu_direction { kForward, kBackward };
using gpu_i32_minutes = std::chrono::duration<int32_t, std::ratio<60>>;
using gpu_unixtime_t = std::chrono::sys_time<gpu_i32_minutes>;

template <gpu_direction SearchDir>
inline constexpr auto const kInvalidGpuDelta =
    SearchDir == gpu_direction::kForward ? gpu_delta{31, 2047}
                                             : gpu_delta{0, 0};

gpu_unixtime_t gpu_delta_to_unixtime(date::sys_days const base,
                               gpu_delta const gd) {
  return std::chrono::time_point_cast<gpu_unixtime_t::duration>(base) +
         (gd.days_ * 1440 + gd.mam_) * gpu_unixtime_t::duration{1};
}
gpu_delta unix_to_gpu_delta(date::sys_days const base,
                              gpu_unixtime_t const t) {
  auto mam =
      (t - std::chrono::time_point_cast<gpu_unixtime_t::duration>(base)).count();
  gpu_delta gd;
  assert(x != std::numeric_limits<mam>::min());
  assert(x != std::numeric_limits<mam>::max());
  if (mam < 0) {
    auto const d = -mam / 1440 + 1;
    auto const min = mam + (d * 1440);
    gd.days_ = d;
    gd.mam_ = min;
    return gd;
  } else {
    gd.days_ = mam / 1440;
    gd.mam_ = mam % 1440;
    return gd;
  }
}

extern "C" {
  struct gpu_timetable {
    gpu_delta* route_stop_times_{nullptr};
    cista::raw::vecvec<gpu_route_idx_t,gpu_value_type>* route_location_seq_ {};
    cista::raw::vecvec<gpu_location_idx_t , gpu_route_idx_t>* location_routes_ {};
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
  };

  struct gpu_timetable* create_gpu_timetable(gpu_delta const* route_stop_times,
                                             std::uint32_t n_route_stop_times,
                                             cista::raw::vecvec<gpu_route_idx_t,gpu_value_type> const* route_location_seq_, // Route -> list_of_stops
                                             cista::raw::vecvec<gpu_location_idx_t , gpu_route_idx_t> const* location_routes_ // location -> Route
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