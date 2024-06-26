#pragma once

#include "nigiri/common/delta_t.h"
#include "nigiri/common/linear_lower_bound.h"
#include "nigiri/routing/journey.h"
#include "nigiri/routing/limits.h"
#include "nigiri/routing/pareto_set.h"
#include "nigiri/routing/raptor/debug.h"
#include "nigiri/routing/raptor/gpu_raptor_state.h"
#include "nigiri/routing/raptor/raptor_state.h" //maybe weg
#include "nigiri/routing/raptor/reconstruct.h"
#include "nigiri/rt/rt_timetable.h"
#include "nigiri/special_stations.h"
#include "nigiri/routing/gpu_timetable.h"
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

  gpu_raptor(gpu_timetable const* gtt,
         //rt_timetable const* rtt,
         gpu_raptor_state& state,
         std::vector<bool>& is_dest,
         std::vector<std::uint16_t>& dist_to_dest,
         std::vector<std::uint16_t>& lb,
         day_idx_t const base,
         clasz_mask_t const allowed_claszes)
      : gtt_{gtt},
        //rtt_{rtt},
        state_{state},
        is_dest_{is_dest},
        dist_to_end_{dist_to_dest},
        lb_{lb},
        base_{base},
        n_days_{gtt_.internal_interval_days().size().count()},
        n_locations_{gtt_.n_locations_},
        n_routes_{gtt_.n_routes_},
        //n_rt_transports_{Rt ? rtt->n_rt_transports() : 0U},
        allowed_claszes_{allowed_claszes} {
    state_.resize(n_locations_, n_routes_, n_rt_transports_);
    utl::fill(time_at_dest_, kInvalid);
    state_.round_times_.reset(kInvalid);
  }

  algo_stats_t get_stats() const {
    return stats_;
  }

  void reset_arrivals() {
    // utl::fill(time_at_dest_, kInvalid);
    // state_.round_times_.reset(kInvalid);
  }

  void next_start_time() {
    /*utl::fill(state_.best_, kInvalid);
    utl::fill(state_.tmp_, kInvalid);
    utl::fill(state_.prev_station_mark_, false);
    utl::fill(state_.station_mark_, false);
    utl::fill(state_.route_mark_, false);
    if constexpr (Rt) {
      utl::fill(state_.rt_transport_mark_, false);
    }*/
  }

  void add_start(location_idx_t const l, unixtime_t const t) {
    /*state_.best_[to_idx(l)] = unix_to_delta(base(), t);
    state_.round_times_[0U][to_idx(l)] = unix_to_delta(base(), t);
    state_.station_mark_[to_idx(l)] = true;*/
  }

  // hier wird Kernel aufgerufen
void execute(unixtime_t const start_time,
             std::uint8_t const max_transfers,
             unixtime_t const worst_time_at_dest,
             profile_idx_t const prf_idx,
             pareto_set<journey>& results){
  void* kernel_arg[] = {(void*)&start_time, (void*)&max_transfers, (void*)&worst_time_at_dest, (void*)&prf_idx, (void*)&results, (void*)&this};
  launchKernel(gpu_raptor_kernel, kernel_args, this.mem_->, this.mem_->);
}


  void reconstruct(query const& q, journey& j) {
    // reconstruct_journey<SearchDir>(tt_, rtt_, q, state_, j, base(), base_);
  }

  gpu_timetable const* gtt_{nullptr};
  //rt_timetable const* rtt_{nullptr};
  // all diese müssen mit malloc (evtl. in anderer Datei)
  gpu_raptor_state& state_;
  std::vector<bool>& is_dest_;
  std::vector<std::uint16_t>& dist_to_end_;
  std::vector<std::uint16_t>& lb_;
  std::array<delta_t, kMaxTransfers + 1> time_at_dest_;
  day_idx_t base_;
  int n_days_;
  raptor_stats stats_;
  std::uint32_t n_locations_, n_routes_, n_rt_transports_;
  clasz_mask_t allowed_claszes_;
};

__global__ void gpu_raptor_kernel(gpu_timetable const tt){

}

}  // namespace nigiri::routing
