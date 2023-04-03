#pragma once

#include <string_view>

#include "nigiri/types.h"

namespace nigiri {
struct timetable;
}

namespace nigiri::loader::gtfs {

using agency_map_t = hash_map<std::string, provider_idx_t>;

agency_map_t parse_agencies(timetable&, std::string_view file_content);

}  // namespace nigiri::loader::gtfs
