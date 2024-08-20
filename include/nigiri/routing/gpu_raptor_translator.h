#pragma once

#include <cinttypes>
#include "nigiri/routing/gpu_raptor_state.h"
#include "nigiri/routing/gpu_timetable.h"
#include "nigiri/routing/pareto_set.h"
#include "nigiri/routing/journey.h"
#include "nigiri/routing/query.h"
#include "nigiri/routing/raptor/raptor_state.h"
#include "nigiri/rt/rt_timetable.h"
#include "clasz_mask.h"
#include "nigiri/timetable.h"
#include "gpu_types.h"

template <gpu_direction SearchDir, bool Rt>
struct gpu_raptor;


template <nigiri::direction SearchDir, bool Rt>
struct gpu_raptor_translator {
  static constexpr auto const kInvalid = nigiri::kInvalidDelta<SearchDir>;
  static constexpr bool kUseLowerBounds = true;
  using algo_state_t = nigiri::routing::raptor_state; //TODO: maybe zu gpu_raptor_sate ändern und dann auch im Konstruktor
  using algo_stats_t = gpu_raptor_stats;
  static nigiri::direction const cpu_direction_ = SearchDir;
  static gpu_direction const gpu_direction_ =
      static_cast<enum gpu_direction const>(cpu_direction_);
  std::variant<
      std::unique_ptr<gpu_raptor<gpu_direction::kForward, true>>,
      std::unique_ptr<gpu_raptor<gpu_direction::kForward, false>>,
      std::unique_ptr<gpu_raptor<gpu_direction::kBackward, true>>,
      std::unique_ptr<gpu_raptor<gpu_direction::kBackward, false>>
      > gpu_r_;

  gpu_raptor_translator(nigiri::timetable const& tt,
                        nigiri::rt_timetable const* rtt,
                        algo_state_t & state,
                        std::vector<bool>& is_dest,
                        std::vector<std::uint16_t>& dist_to_dest,
                        std::vector<std::uint16_t>& lb,
                        nigiri::day_idx_t const base,
                        nigiri::routing::clasz_mask_t const allowed_claszes);
  algo_stats_t get_stats();

  void reset_arrivals();

  void next_start_time();

  void add_start(nigiri::location_idx_t const l, nigiri::unixtime_t const t);

  // hier wird Kernel aufgerufen
  void execute(nigiri::unixtime_t const start_time,
               uint8_t const max_transfers,
               nigiri::unixtime_t const worst_time_at_dest,
               nigiri::profile_idx_t const prf_idx,
               nigiri::pareto_set<nigiri::routing::journey>& results);

  void reconstruct(nigiri::routing::query const& q,
                   nigiri::routing::journey& j);

  nigiri::timetable const& tt_;
  nigiri::rt_timetable const* rtt_{nullptr};
  algo_state_t & state_;
  std::vector<bool>& is_dest_;
  std::vector<std::uint16_t>& dist_to_end_;
  std::vector<std::uint16_t>& lb_;
  std::array<nigiri::delta_t, nigiri::routing::kMaxTransfers + 1> time_at_dest_;
  nigiri::day_idx_t base_;
  int n_days_;
  std::uint32_t n_locations_, n_routes_, n_rt_transports_;
  nigiri::routing::clasz_mask_t allowed_claszes_;
  static bool test(bool hi);
private:
  date::sys_days base() const;
  gpu_timetable* translate_tt_in_gtt(nigiri::timetable tt);
  gpu_delta_t* get_gpu_roundtimes(nigiri::unixtime_t const start_time,
                                  uint8_t const max_transfers,
                                  nigiri::unixtime_t const worst_time_at_dest,
                                  nigiri::profile_idx_t const prf_idx);
};
#pragma once

#include <cinttypes>
#include "nigiri/common/delta_t.h"
#include "nigiri/routing/gpu_raptor.h"
#include "nigiri/routing/raptor/raptor.h"
#include "nigiri/rt/rt_timetable.h"
#include "nigiri/timetable.h"
#include "nigiri/routing/gpu_types.h"

using namespace date;
using namespace nigiri;
using namespace nigiri::routing;
template <direction SearchDir, bool Rt>
void gpu_raptor_translator<SearchDir, Rt>::execute(
    unixtime_t const start_time,
    uint8_t const max_transfers,
    unixtime_t const worst_time_at_dest,
    profile_idx_t const prf_idx,
    nigiri::pareto_set<journey>& results) {
  gpu_delta_t* gpu_round_times = get_gpu_roundtimes(start_time,max_transfers,worst_time_at_dest,prf_idx);
  // Konstruktion der Ergebnis-Journey
  auto const end_k = std::min(max_transfers, kMaxTransfers) + 1U;
  for (auto i = 0U; i != n_locations_; ++i) {
    auto const is_dest = is_dest_[i];
    if (!is_dest) {
      continue;
    }

    for (auto k = 1U; k != end_k; ++k) {
      auto const dest_time = *reinterpret_cast<delta_t*>(
          &gpu_round_times[k * (gpu_kMaxTransfers + 1U) + i]);
      if (dest_time != kInvalid) {
        trace("ADDING JOURNEY: start={}, dest={} @ {}, transfers={}\n",
              start_time,
              delta_to_unix(
                  base(),
                  (*reinterpret_cast<delta_t*>(
                      &gpu_round_times[k * (gpu_kMaxTransfers + 1U) + i]))),
              location{tt_, location_idx_t{i}}, k - 1);
        auto const [optimal, it, dominated_by] = results.add(
            journey{.legs_ = {},
                    .start_time_ = start_time,
                    .dest_time_ = delta_to_unix(base(), dest_time),
                    .dest_ = location_idx_t{i},
                    .transfers_ = static_cast<std::uint8_t>(k - 1)});
        if (!optimal) {
          trace("  DOMINATED BY: start={}, dest={} @ {}, transfers={}\n",
                dominated_by->start_time_, dominated_by->dest_time_,
                location{tt_, dominated_by->dest_}, dominated_by->transfers_);
        }
      }
    }
  }
}

template <direction SearchDir, bool Rt>
void gpu_raptor_translator<SearchDir, Rt>::reconstruct(const query& q,
                                                       journey& j){
  reconstruct_journey<SearchDir>(tt_, rtt_, q, state_, j, base(), base_);
}
inline int translator_as_int(day_idx_t const d)  { return static_cast<int>(d.v_); } //as_int von raptor rüber kopiert da nicht static dort
template <nigiri::direction SearchDir, bool Rt>
date::sys_days gpu_raptor_translator<SearchDir, Rt>::base() const{
  return tt_.internal_interval_days().from_ +translator_as_int(base_) * date::days{1};
};

template <nigiri::direction SearchDir, bool Rt>
gpu_raptor_translator<SearchDir, Rt>::gpu_raptor_translator(
    nigiri::timetable const& tt,
    nigiri::rt_timetable const* rtt,
    algo_state_t & state,
    std::vector<bool>& is_dest,
    std::vector<std::uint16_t>& dist_to_dest,
    std::vector<std::uint16_t>& lb,
    nigiri::day_idx_t const base,
    nigiri::routing::clasz_mask_t const allowed_claszes)
    : tt_{tt},
      rtt_{rtt},
      state_{state},
      is_dest_{is_dest},
      dist_to_end_{dist_to_dest},
      lb_{lb},
      base_{base},
      n_days_{tt_.internal_interval_days().size().count()},
      n_locations_{tt_.n_locations()},
      n_routes_{tt.n_routes()},
      n_rt_transports_{Rt ? rtt->n_rt_transports() : 0U},
      allowed_claszes_{allowed_claszes}{
  auto gpu_base = *reinterpret_cast<gpu_day_idx_t*>(&base_);
  auto gpu_allowed_claszes = *reinterpret_cast<gpu_clasz_mask_t*>(&allowed_claszes_);
  auto gpu_state_ =  gpu_raptor_state{};
  gpu_r_ = std::make_unique<gpu_raptor<gpu_direction_,Rt>>(translate_tt_in_gtt(tt_),gpu_state_, is_dest_,dist_to_end_, lb_, gpu_base, gpu_allowed_claszes); //TODO maybe: transalte raptor_state into gpu_raptor_state
}
using algo_stats_t = gpu_raptor_stats;
template <nigiri::direction SearchDir, bool Rt>
algo_stats_t gpu_raptor_translator<SearchDir, Rt>::get_stats() {
  if (gpu_direction_ == gpu_direction::kForward && Rt == true) {
    return get<std::unique_ptr<gpu_raptor<gpu_direction::kForward,true>>>(gpu_r_)->get_stats();
  } else if (gpu_direction_ == gpu_direction::kForward && Rt == false) {
    return get<std::unique_ptr<gpu_raptor<gpu_direction::kForward,false>>>(gpu_r_)->get_stats();
  } else if (gpu_direction_ == gpu_direction::kBackward && Rt == true) {
    return get<std::unique_ptr<gpu_raptor<gpu_direction::kBackward,true>>>(gpu_r_)->get_stats();
  } else if (gpu_direction_ == gpu_direction::kBackward && Rt == false) {
    return get<std::unique_ptr<gpu_raptor<gpu_direction::kBackward,false>>>(gpu_r_)->get_stats();
  }
}

template <nigiri::direction SearchDir, bool Rt>
void gpu_raptor_translator<SearchDir, Rt>::reset_arrivals() {
  if (gpu_direction_ == gpu_direction::kForward && Rt == true) {
    get<std::unique_ptr<gpu_raptor<gpu_direction::kForward,true>>>(gpu_r_)->reset_arrivals();
  } else if (gpu_direction_ == gpu_direction::kForward && Rt == false) {
    get<std::unique_ptr<gpu_raptor<gpu_direction::kForward,false>>>(gpu_r_)->reset_arrivals();
  } else if (gpu_direction_ == gpu_direction::kBackward && Rt == true) {
    get<std::unique_ptr<gpu_raptor<gpu_direction::kBackward,true>>>(gpu_r_)->reset_arrivals();
  } else if (gpu_direction_ == gpu_direction::kBackward && Rt == false) {
    get<std::unique_ptr<gpu_raptor<gpu_direction::kBackward,false>>>(gpu_r_)->reset_arrivals();
  }
}

template <nigiri::direction SearchDir, bool Rt>
void gpu_raptor_translator<SearchDir, Rt>::next_start_time() {
  if (gpu_direction_ == gpu_direction::kForward && Rt == true) {
    get<std::unique_ptr<gpu_raptor<gpu_direction::kForward,true>>>(gpu_r_)->next_start_time();
  } else if (gpu_direction_ == gpu_direction::kForward && Rt == false) {
    get<std::unique_ptr<gpu_raptor<gpu_direction::kForward,false>>>(gpu_r_)->next_start_time();
  } else if (gpu_direction_ == gpu_direction::kBackward && Rt == true) {
    get<std::unique_ptr<gpu_raptor<gpu_direction::kBackward,true>>>(gpu_r_)->next_start_time();
  } else if (gpu_direction_ == gpu_direction::kBackward && Rt == false) {
    get<std::unique_ptr<gpu_raptor<gpu_direction::kBackward,false>>>(gpu_r_)->next_start_time();
  }
}

template <nigiri::direction SearchDir, bool Rt>
void gpu_raptor_translator<SearchDir, Rt>::add_start(nigiri::location_idx_t const l,
                                                     nigiri::unixtime_t const t) {
  auto gpu_l = *reinterpret_cast<const gpu_location_idx_t*>(&l);
  auto gpu_t = *reinterpret_cast<const gpu_unixtime_t*>(&t);
  if (gpu_direction_ == gpu_direction::kForward && Rt == true) {
    get<std::unique_ptr<gpu_raptor<gpu_direction::kForward,true>>>(gpu_r_)->add_start(gpu_l,gpu_t);
  } else if (gpu_direction_ == gpu_direction::kForward && Rt == false) {
    get<std::unique_ptr<gpu_raptor<gpu_direction::kForward,false>>>(gpu_r_)->add_start(gpu_l,gpu_t);
  } else if (gpu_direction_ == gpu_direction::kBackward && Rt == true) {
    get<std::unique_ptr<gpu_raptor<gpu_direction::kBackward,true>>>(gpu_r_)->add_start(gpu_l,gpu_t);
  } else if (gpu_direction_ == gpu_direction::kBackward && Rt == false) {
    get<std::unique_ptr<gpu_raptor<gpu_direction::kBackward,false>>>(gpu_r_)->add_start(gpu_l,gpu_t);
  }
}

// hier wird Kernel aufgerufen
template <nigiri::direction SearchDir, bool Rt>
gpu_delta_t* gpu_raptor_translator<SearchDir, Rt>::get_gpu_roundtimes(
    nigiri::unixtime_t const start_time,
    uint8_t const max_transfers,
    nigiri::unixtime_t const worst_time_at_dest,
    nigiri::profile_idx_t const prf_idx) {
  auto gpu_start_time = *reinterpret_cast<gpu_unixtime_t const*>(&start_time);
  auto gpu_worst_time_at_dest =
      *reinterpret_cast<gpu_unixtime_t const*>(&max_transfers);
  auto gpu_prf_idx = *reinterpret_cast<gpu_profile_idx_t const*>(&prf_idx);
  gpu_delta_t* gpu_round_times;
  if (auto gpu_r = get_if<std::unique_ptr<gpu_raptor<gpu_direction::kForward,true>>>(&gpu_r_)->get()) {
    gpu_round_times = gpu_r->execute(start_time,max_transfers,worst_time_at_dest,prf_idx);
  } else if (auto gpu_r = get_if<std::unique_ptr<gpu_raptor<gpu_direction::kForward,false>>>(&gpu_r_)->get()) {
    gpu_round_times = gpu_r->execute(start_time,max_transfers,worst_time_at_dest,prf_idx);
  } else if (auto gpu_r = get_if<std::unique_ptr<gpu_raptor<gpu_direction::kBackward, true>>>(&gpu_r_)->get()) {
    gpu_round_times = gpu_r->execute(start_time,max_transfers,worst_time_at_dest,prf_idx);
  } else if (auto gpu_r = get_if<std::unique_ptr<gpu_raptor<gpu_direction::kBackward,false>>>(&gpu_r_)->get()) {
    gpu_round_times = gpu_r->execute(start_time,max_transfers,worst_time_at_dest,prf_idx);
  }
  return gpu_round_times;
}

template <nigiri::direction SearchDir, bool Rt>
gpu_timetable* gpu_raptor_translator<SearchDir, Rt>::translate_tt_in_gtt(nigiri::timetable tt) {
  vector_map<nigiri::bitfield_idx_t, std::uint64_t*> bitfields_data_ =
      vector_map<nigiri::bitfield_idx_t, std::uint64_t*>();
  for (nigiri::bitfield_idx_t i = nigiri::bitfield_idx_t{0}; i < tt.bitfields_.size(); ++i) {
    auto t = tt.bitfields_.at(i);
    bitfields_data_.emplace_back(t.blocks_.data());     //TODO: überprüfen ob es so funktioniert
  }
  auto gpu_bitfields_data_ =
      reinterpret_cast<gpu_vector_map<gpu_bitfield_idx_t, std::uint64_t*>*>(
          &bitfields_data_);
  gpu_locations locations_ = gpu_locations(
      *reinterpret_cast<gpu_vector_map<gpu_location_idx_t, gpu_u8_minutes>*>(
          &tt.locations_.transfer_time_),
      reinterpret_cast<gpu_vecvec<gpu_location_idx_t, nigiri::gpu_footpath>*>(
          tt.locations_.footpaths_out_.data()),
      reinterpret_cast<gpu_vecvec<gpu_location_idx_t, nigiri::gpu_footpath>*>(
          tt.locations_.footpaths_in_.data()));
  auto n_locations = tt.n_locations();
  auto n_routes = tt.n_routes();
  auto gtt = create_gpu_timetable(
      reinterpret_cast<gpu_delta*>(tt.route_stop_times_.data()),
      tt.route_stop_times_.size(),
      reinterpret_cast<gpu_vecvec<gpu_route_idx_t, gpu_value_type>*>(
          &tt.route_location_seq_),
      reinterpret_cast<gpu_vecvec<gpu_location_idx_t, gpu_route_idx_t>*>(
          &tt.location_routes_),
      &n_locations, &n_routes,
      reinterpret_cast<gpu_vector_map<gpu_route_idx_t,
                                      nigiri::gpu_interval<std::uint32_t>>*>(
          &tt.route_stop_time_ranges_),
      reinterpret_cast<gpu_vector_map<
          gpu_route_idx_t, nigiri::gpu_interval<gpu_transport_idx_t>>*>(
          &tt.route_transport_ranges_),
      reinterpret_cast<gpu_vector_map<gpu_bitfield_idx_t, gpu_bitfield>*>(
          &tt.bitfields_),
      gpu_bitfields_data_,
      reinterpret_cast<
          gpu_vector_map<gpu_transport_idx_t, gpu_bitfield_idx_t>*>(
          &tt.transport_traffic_days_),
      reinterpret_cast<nigiri::gpu_interval<gpu_sys_days>*>(&tt.date_range_),
      &locations_,
      reinterpret_cast<gpu_vector_map<gpu_route_idx_t, gpu_clasz>*>(
          &tt.route_clasz_));
  return gtt;
}
template <nigiri::direction SearchDir, bool Rt>
bool gpu_raptor_translator<SearchDir, Rt>::test(bool hi) {
  return hi;
}
