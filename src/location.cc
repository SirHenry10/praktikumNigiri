#include "nigiri/location.h"

#include "nigiri/timetable.h"

namespace nigiri {

std::ostream& operator<<(std::ostream& out, location const& l) {
  return out << '(' << l.name_ << ", " << l.id_ << ')';
}

location::location(timetable const& tt, location_idx_t idx)
    : l_{idx},
      id_{tt.locations_.ids_[idx].view()},
      name_{tt.locations_.names_[idx].view()},
      pos_{tt.locations_.coordinates_[idx]},
      src_{tt.locations_.src_[idx]},
      type_{tt.locations_.types_[idx]},
      parent_{tt.locations_.parents_[idx]},
      timezone_idx_{tt.locations_.location_timezones_[idx]},
      transfer_time_{tt.locations_.transfer_time_[idx]},
      equivalences_{tt.locations_.equivalences_[idx]} {}

auto init_equivalences(gpu_timetable const& gtt, location_idx_t l) {
  auto n_gpu_equivalences = *reinterpret_cast<mutable_fws_multimap<location_idx_t, location_idx_t>*>(&gtt.locations_->equivalences_);
  return n_gpu_equivalences[l];
}
location::location(gpu_timetable const& gtt, gpu_location_idx_t idx)
    : l_{*reinterpret_cast<location_idx_t*>(&idx)},equivalences_{init_equivalences(gtt,l_)}{
  auto n_gpu_ids = *reinterpret_cast<vecvec<location_idx_t, char>*>(&gtt.locations_->ids_);
  id_ = n_gpu_ids[l_].view();
  auto n_gpu_names = *reinterpret_cast<vecvec<location_idx_t, char>*>(&gtt.locations_->names_);
  name_ = n_gpu_names[l_].view();
  auto n_gpu_coordinates = *reinterpret_cast<vector_map<location_idx_t, geo::latlng>*>(&gtt.locations_->coordinates_);
  pos_ = n_gpu_coordinates[l_];
  auto n_gpu_src = *reinterpret_cast<vector_map<location_idx_t, source_idx_t>*>(&gtt.locations_->src_);
  src_ = n_gpu_src[l_];
  auto n_gpu_types = *reinterpret_cast<vector_map<location_idx_t, location_type>*>(&gtt.locations_->types_);
  type_ = n_gpu_types[l_];
  auto n_gpu_parent = *reinterpret_cast<vector_map<location_idx_t, location_idx_t>*>(&gtt.locations_->parents_);
  parent_ = n_gpu_parent[l_];
  auto n_gpu_timezones = *reinterpret_cast<vector_map<location_idx_t, timezone_idx_t>*>(&gtt.locations_->location_timezones_);
  timezone_idx_ = n_gpu_timezones[l_];
  auto n_gpu_transfer_time = *reinterpret_cast<vector_map<location_idx_t, u8_minutes>*>(&gtt.locations_->transfer_time_);
  transfer_time_ = n_gpu_transfer_time[l_];
}


location::location(
    std::string_view id,
    std::string_view name,
    geo::latlng pos,
    source_idx_t src,
    location_type type,
    location_idx_t parent,
    timezone_idx_t timezone,
    duration_t transfer_time,
    it_range<vector<location_idx_t>::const_iterator> equivalences)
    : l_{location_idx_t::invalid()},
      id_{id},
      name_{name},
      pos_{pos},
      src_{src},
      type_{type},
      parent_{parent},
      timezone_idx_{timezone},
      transfer_time_{transfer_time},
      equivalences_{equivalences} {}

}  // namespace nigiri
