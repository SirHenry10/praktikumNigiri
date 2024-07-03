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
    bool operator== (gpu_delta_t a){
      return (a.days_== this->days_ && a.mam_==this->mam_);
    }
    bool operator!= (gpu_delta_t a){
      return !(operator==(a));
    }
  };
  struct gpu_timetable {
    struct locations {
      timezone_idx_t register_timezone(timezone tz) {
        auto const idx = timezone_idx_t{
            static_cast<timezone_idx_t::value_t>(timezones_.size())};
        timezones_.emplace_back(std::move(tz));
        return idx;
      }

      location_idx_t register_location(location const& l) {
        auto const next_idx = static_cast<location_idx_t::value_t>(names_.size());
        auto const l_idx = location_idx_t{next_idx};
        auto const [it, is_new] = location_id_to_idx_.emplace(
            location_id{.id_ = l.id_, .src_ = l.src_}, l_idx);

        if (is_new) {
          names_.emplace_back(l.name_);
          coordinates_.emplace_back(l.pos_);
          ids_.emplace_back(l.id_);
          src_.emplace_back(l.src_);
          types_.emplace_back(l.type_);
          location_timezones_.emplace_back(l.timezone_idx_);
          equivalences_.emplace_back();
          children_.emplace_back();
          preprocessing_footpaths_out_.emplace_back();
          preprocessing_footpaths_in_.emplace_back();
          transfer_time_.emplace_back(l.transfer_time_);
          parents_.emplace_back(l.parent_);
        } else {
          log(log_lvl::error, "timetable.register_location",
              "duplicate station {}", l.id_);
        }

        assert(names_.size() == next_idx + 1);
        assert(coordinates_.size() == next_idx + 1);
        assert(ids_.size() == next_idx + 1);
        assert(src_.size() == next_idx + 1);
        assert(types_.size() == next_idx + 1);
        assert(location_timezones_.size() == next_idx + 1);
        assert(equivalences_.size() == next_idx + 1);
        assert(children_.size() == next_idx + 1);
        assert(preprocessing_footpaths_out_.size() == next_idx + 1);
        assert(preprocessing_footpaths_in_.size() == next_idx + 1);
        assert(transfer_time_.size() == next_idx + 1);
        assert(parents_.size() == next_idx + 1);

        return it->second;
      }

      location get(location_idx_t const idx) const {
        auto l = location{ids_[idx].view(),
                          names_[idx].view(),
                          coordinates_[idx],
                          src_[idx],
                          types_[idx],
                          parents_[idx],
                          location_timezones_[idx],
                          transfer_time_[idx],
                          it_range{equivalences_[idx]}};
        l.l_ = idx;
        return l;
      }

      location get(location_id const& id) const {
        return get(location_id_to_idx_.at(id));
      }

      void resolve_timezones();

      // Station access: external station id -> internal station idx
      hash_map<location_id, location_idx_t> location_id_to_idx_;
      vecvec<location_idx_t, char> names_;
      vecvec<location_idx_t, char> ids_;
      vector_map<location_idx_t, geo::latlng> coordinates_;
      vector_map<location_idx_t, source_idx_t> src_;
      vector_map<location_idx_t, u8_minutes> transfer_time_;
      vector_map<location_idx_t, location_type> types_;
      vector_map<location_idx_t, location_idx_t> parents_;
      vector_map<location_idx_t, timezone_idx_t> location_timezones_;
      mutable_fws_multimap<location_idx_t, location_idx_t> equivalences_;
      mutable_fws_multimap<location_idx_t, location_idx_t> children_;
      mutable_fws_multimap<location_idx_t, footpath> preprocessing_footpaths_out_;
      mutable_fws_multimap<location_idx_t, footpath> preprocessing_footpaths_in_;
      array<vecvec<location_idx_t, footpath>, kMaxProfiles> footpaths_out_;
      array<vecvec<location_idx_t, footpath>, kMaxProfiles> footpaths_in_;
      vector_map<timezone_idx_t, timezone> timezones_;
    } locations_;
    unixtime_t to_unixtime(std::uint16_t const d,
                           std::uint16_t const mam) const {
      return internal_interval_days().from_ + d * 1_days +
             static_cast<std::chrono::duration<int16_t, std::ratio<60>>>(mam);
      //kucken ob cast stimmt
    }

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

}  // extern "C"
}  //namespace nigiri