#pragma once

#include <cinttypes>
#include "nigiri/routing/gpu_raptor.cuh"
#include "nigiri/routing/gpu_raptor_translator.h"
#include "nigiri/rt/rt_timetable.h"
#include "nigiri/timetable.h"

namespace nigiri::routing {

using algo_stats_t = gpu_raptor_state;

template <direction SearchDir, bool Rt>
gpu_raptor_translator<SearchDir, Rt>::gpu_raptor_translator(
    timetable const& tt,
    rt_timetable const* rtt,
    raptor_state& state,
    std::vector<bool>& is_dest,
    std::vector<std::uint16_t>& dist_to_dest,
    std::vector<std::uint16_t>& lb,
    day_idx_t const base,
    clasz_mask_t const allowed_claszes)
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
      allowed_claszes_{allowed_claszes}  {
  auto gpu_base = *reinterpret_cast<gpu_day_idx_t*>(&base_);
  auto gpu_allowed_claszes = *reinterpret_cast<gpu_clasz_mask_t*>(&allowed_claszes_);
  gpu_r_ = gpu_raptor<gpu_direction_, Rt>(translate_tt_in_gtt(tt_), state_, is_dest_, dist_to_dest, lb_, gpu_base, gpu_allowed_claszes);
}

template <direction SearchDir, bool Rt>
algo_stats_t gpu_raptor_translator<SearchDir, Rt>::get_stats() const {
  return gpu_r_.get_stats();
}

template <direction SearchDir, bool Rt>
void gpu_raptor_translator<SearchDir, Rt>::reset_arrivals() {
  gpu_r_.reset_arrivals();
}

template <direction SearchDir, bool Rt>
void gpu_raptor_translator<SearchDir, Rt>::next_start_time() {
  gpu_r_.next_start_time();
}

template <direction SearchDir, bool Rt>
void gpu_raptor_translator<SearchDir, Rt>::add_start(location_idx_t const l,
                                                     unixtime_t const t) {
  auto gpu_l = *reinterpret_cast<const gpu_location_idx_t*>(&l);
  auto gpu_t = *reinterpret_cast<const gpu_unixtime_t*>(&t);
  gpu_r_.add_start(gpu_l, gpu_t);
}

// hier wird Kernel aufgerufen
template <direction SearchDir, bool Rt>
void gpu_raptor_translator<SearchDir, Rt>::execute(
    unixtime_t const start_time,
    uint8_t const max_transfers,
    unixtime_t const worst_time_at_dest,
    profile_idx_t const prf_idx,
    nigiri::pareto_set<nigiri::routing::journey>& results) {
  auto gpu_start_time = *reinterpret_cast<gpu_unixtime_t const*>(&start_time);
  auto gpu_worst_time_at_dest =
      *reinterpret_cast<gpu_unixtime_t const*>(&max_transfers);
  auto gpu_prf_idx = *reinterpret_cast<gpu_profile_idx_t const*>(&prf_idx);
  auto gpu_round_times = gpu_r_.execute(gpu_start_time, max_transfers,
                                        gpu_worst_time_at_dest, gpu_prf_idx);
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
gpu_timetable* gpu_raptor_translator<SearchDir, Rt>::translate_tt_in_gtt(timetable tt) {
  vector_map<bitfield_idx_t, std::uint64_t*> bitfields_data_ =
      vector_map<bitfield_idx_t, std::uint64_t*>();
  for (bitfield_idx_t i = bitfield_idx_t{0}; i < tt.bitfields_.size(); ++i) {
    auto t = tt.bitfields_.at(i);
    bitfields_data_.emplace_back(std::make_pair(i, t.blocks_.data()));
  }
  auto gpu_bitfields_data_ =
      reinterpret_cast<gpu_vector_map<gpu_bitfield_idx_t, std::uint64_t*>*>(
          &bitfields_data_);
  gpu_locations locations_ = gpu_locations(
      *reinterpret_cast<gpu_vector_map<gpu_location_idx_t, gpu_u8_minutes>*>(
          &tt.locations_.transfer_time_),
      reinterpret_cast<gpu_vecvec<gpu_location_idx_t, gpu_footpath>*>(
          tt.locations_.footpaths_out_.data()),
      reinterpret_cast<gpu_vecvec<gpu_location_idx_t, gpu_footpath>*>(
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
}//namespace nigiri::routing;