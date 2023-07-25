#pragma once

#include "nigiri/timetable.h"

namespace nigiri::loader {

void build_footpaths(timetable& tt,
                     uint16_t const& no_profiles,
                     bool adjust_footpaths,
                     bool merge_duplicates);

}  // namespace nigiri::loader
