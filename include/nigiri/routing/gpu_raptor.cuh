#pragma once

#include "nigiri/common/delta_t.h"
#include "nigiri/common/linear_lower_bound.h"
#include "nigiri/routing/journey.h"
#include "nigiri/routing/limits.h"
#include "nigiri/routing/pareto_set.h"
#include "nigiri/routing/raptor/debug.h"
#include "nigiri/routing/raptor/raptor_state.h"
#include "nigiri/routing/raptor/reconstruct.h"
#include "nigiri/rt/rt_timetable.h"
#include "nigiri/special_stations.h"
#include "nigiri/timetable.h"
#include "nigiri/types.h"

namespace nigiri::routing {

struct raptor_stats {
  std::uint64_t n_routing_time_{0ULL};
  std::uint64_t n_footpaths_visited_{0ULL};
  std::uint64_t n_routes_visited_{0ULL};
  std::uint64_t n_earliest_trip_calls_{0ULL};
  std::uint64_t n_earliest_arrival_updated_by_route_{0ULL};
  std::uint64_t n_earliest_arrival_updated_by_footpath_{0ULL};
  std::uint64_t fp_update_prevented_by_lower_bound_{0ULL};
  std::uint64_t route_update_prevented_by_lower_bound_{0ULL};
};

template <direction SearchDir, bool Rt>
struct gpu_raptor {
  using algo_state_t = raptor_state;
  using algo_stats_t = raptor_stats;

  static constexpr bool kUseLowerBounds = true;
  static constexpr auto const kFwd = (SearchDir == direction::kForward);
  static constexpr auto const kBwd = (SearchDir == direction::kBackward);
  static constexpr auto const kInvalid = kInvalidDelta<SearchDir>;
  static constexpr auto const kUnreachable =
      std::numeric_limits<std::uint16_t>::max();
  static constexpr auto const kIntermodalTarget =
      to_idx(get_special_station(special_station::kEnd));

  static bool is_better(auto a, auto b) { return kFwd ? a < b : a > b; }
  static bool is_better_or_eq(auto a, auto b) { return kFwd ? a <= b : a >= b; }
  static auto get_best(auto a, auto b) { return is_better(a, b) ? a : b; }
  static auto get_best(auto x, auto... y) {
    ((x = get_best(x, y)), ...);
    return x;
  }
  static auto dir(auto a) { return (kFwd ? 1 : -1) * a; }


};

}  // namespace nigiri::routing
