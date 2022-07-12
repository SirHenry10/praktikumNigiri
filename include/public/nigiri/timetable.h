#pragma once

#include "nigiri/types.h"

namespace nigiri {

struct timetable {
  struct expanded_trip_section {
    trip_idx_t trip_idx_{};
    section_idx_t from_section_idx_{}, to_section_idx_{};
  };

  struct external_trip_section {
    external_trip_idx_t trip_idx_{};
    section_idx_t section_idx_{};
  };

  template <typename T>
  struct index_range {
    T from_{}, to_{};
  };

  struct stop {
    location_idx_t location_idx() const { return location_idx_t{location_}; }
    bool in_allowed() const { return in_allowed_; }
    bool out_allowed() const { return out_allowed_; }

  private:
    location_idx_t::value_t location_ : 30;
    bool in_allowed_ : 1;
    bool out_allowed_ : 1;
  };
  static_assert(sizeof(stop) == sizeof(location_idx_t));

  enum class location_type : std::uint8_t {
    track,
    platform,
    station,
    meta_station
  };

  struct footpath {
    location_idx_t target_;
    duration_t duration_;
  };

  struct locations {
    // Station access: external station id -> internal station idx
    hash_map<location_id_t, location_idx_t> location_id_to_idx_;

    // Location hierarchy
    vector_map<location_idx_t, location_type> types_;
    vector_map<location_idx_t, osm_node_id_t> osm_ids_;
    fws_multimap<location_idx_t, location_idx_t> children_;
    fws_multimap<location_idx_t, location_idx_t> parents_;

    // Footpaths
    fws_multimap<location_idx_t, footpath> footpaths_out_;
    fws_multimap<location_idx_t, footpath> footpaths_in_;
  } locations_;

  // Trip access: external trip id -> internal trip index
  hash_map<external_trip_id_t, external_trip_idx_t> trip_id_to_idx_;

  // External trip index -> list of external trip ids (HRD + RI Basis)
  mutable_fws_multimap<external_trip_idx_t, external_trip_id_t> trip_ids_;

  // Route -> From (inclusive) and to index (exclusive) of expanded trips
  fws_multimap<route_idx_t, index_range<trip_idx_t>> route_trip_ranges_;

  // Route -> list of stops
  fws_multimap<route_idx_t, stop> route_location_seq_;

  // Location -> list of routes
  mutable_fws_multimap<location_idx_t, route_idx_t> location_routes_;

  // Trip index -> sequence of stop times
  fws_multimap<trip_idx_t, minutes_after_midnight_t> trip_stop_times_;

  // Trip index -> traffic day bitfield
  vector_map<trip_idx_t, bitfield_idx_t> expanded_trip_traffic_days_;

  // Trip index -> sequence of stop times
  fws_multimap<rt_trip_idx_t, unixtime_t> rt_trip_stop_times_;

  // Unique bitfields
  vector_map<bitfield_idx_t, bitfield> bitfields_;

  // For each trip the corresponding route
  vector_map<trip_idx_t, route_idx_t> trip_route_;

  // Trip index -> trip section meta data db index
  fws_multimap<trip_idx_t, section_db_idx_t> trip_section_meta_data_;

  // Trip index -> merged trips
  fws_multimap<trip_idx_t, merged_trips_idx_t> trip_to_external_trip_section_;

  // Merged trips info
  fws_multimap<merged_trips_idx_t, external_trip_idx_t> merged_trips_;

  // External trip index -> list of section ranges where this trip was expanded
  mutable_fws_multimap<external_trip_idx_t, expanded_trip_section>
      external_trip_idx_to_expanded_trip_idx_;

  // External trip -> debug info
  vector_map<trip_idx_t, string> trip_debug_;
};

}  // namespace nigiri
