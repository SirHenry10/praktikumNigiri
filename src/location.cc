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


location::location(gpu_timetable const& gtt, gpu_location_idx_t idx)
    : l_{*reinterpret_cast<location_idx_t*>(&idx)},
      id_{gtt.locations_host_.ids_[l_].view()},
      name_{gtt.locations_host_.names_[l_].view()},
      pos_{gtt.locations_host_.coordinates_[l_]},
      src_{gtt.locations_host_.src_[l_]},
      type_{gtt.locations_host_.types_[l_]},
      parent_{gtt.locations_host_.parents_[l_]},
      timezone_idx_{gtt.locations_host_.location_timezones_[l_]},
      transfer_time_{gtt.locations_host_.transfer_time_[l_]},
      equivalences_{gtt.locations_host_.equivalences_[l_]}{
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
