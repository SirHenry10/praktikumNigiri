#pragma once

#include <cinttypes>
//#include "nigiri/common/linear_lower_bound.h"
#include "nigiri/common/delta_t.h"
#include "nigiri/routing/gpu_raptor_state.h"
#include "nigiri/routing/gpu_timetable.h"
#include "nigiri/routing/journey.h"
#include "nigiri/routing/pareto_set.h"
#include "nigiri/routing/raptor/raptor.h"
#include "nigiri/routing/raptor/raptor_state.h"
#include "nigiri/timetable.h"
#include "gpu_types.h"

template <gpu_direction SearchDir, bool Rt>
struct gpu_raptor;

using namespace date;
using namespace nigiri;
using namespace nigiri::routing;


template <direction SearchDir, bool Rt>
struct gpu_raptor_translator {
  static constexpr auto const kInvalid = kInvalidDelta<SearchDir>;
  using algo_stats_t = gpu_raptor_stats;
  static direction const cpu_direction_ = SearchDir;
  gpu_direction const gpu_direction_ =
      *reinterpret_cast<enum gpu_direction const*>(&cpu_direction_);
  std::variant<
      std::unique_ptr<gpu_raptor<gpu_direction::kForward, true>>,
      std::unique_ptr<gpu_raptor<gpu_direction::kForward, false>>,
      std::unique_ptr<gpu_raptor<gpu_direction::kBackward, true>>,
      std::unique_ptr<gpu_raptor<gpu_direction::kBackward, false>>
      > gpu_r_;

  gpu_raptor_translator(timetable const& tt,
                        rt_timetable const* rtt,
                        raptor_state& state,
                        std::vector<bool>& is_dest,
                        std::vector<std::uint16_t>& dist_to_dest,
                        std::vector<std::uint16_t>& lb,
                        day_idx_t const base,
                        clasz_mask_t const allowed_claszes);
  algo_stats_t get_stats();

  void reset_arrivals();

  void next_start_time();

  void add_start(location_idx_t const l, unixtime_t const t);

  // hier wird Kernel aufgerufen
  void execute(unixtime_t const start_time,
               uint8_t const max_transfers,
               unixtime_t const worst_time_at_dest,
               profile_idx_t const prf_idx,
               nigiri::pareto_set<nigiri::routing::journey>& results);

  void reconstruct(nigiri::routing::query const& q,
                   nigiri::routing::journey& j){
    reconstruct_journey<SearchDir>(tt_, rtt_, q, state_, j, base(), base_);
        };

  timetable const& tt_;
  rt_timetable const* rtt_{nullptr};
  raptor_state& state_;
  std::vector<bool>& is_dest_;
  std::vector<std::uint16_t>& dist_to_end_;
  std::vector<std::uint16_t>& lb_;
  std::array<delta_t, kMaxTransfers + 1> time_at_dest_;
  day_idx_t base_;
  int n_days_;
  std::uint32_t n_locations_, n_routes_, n_rt_transports_;
  clasz_mask_t allowed_claszes_;
  static bool test(bool hi);
private:
  date::sys_days base() const{
    return tt_.internal_interval_days().from_ +raptor<SearchDir, Rt>::as_int(base_) * date::days{1};
  };
  gpu_timetable* translate_tt_in_gtt(timetable tt);
};