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
  using algo_state_t = storage_raptor_state;
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
  std::unique_ptr<mem> mem_;
  gpu_timetable const* gtt_;

  gpu_raptor_translator(nigiri::timetable const& tt,
                        nigiri::rt_timetable const* rtt,
                        gpu_timetable const* gtt,
                        algo_state_t & state,
                        std::vector<uint8_t>& is_dest,
                        std::vector<std::uint16_t>& dist_to_dest,
                        std::vector<std::uint16_t>& lb,
                        nigiri::day_idx_t const base,
                        nigiri::routing::clasz_mask_t const allowed_claszes);
  ~gpu_raptor_translator();
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
  algo_state_t& state_;
  nigiri::routing::raptor_state cpu_state_;
  std::vector<uint8_t>& is_dest_;
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
  void get_gpu_roundtimes(nigiri::unixtime_t const start_time,
                                  uint8_t const max_transfers,
                                  nigiri::unixtime_t const worst_time_at_dest,
                                  nigiri::profile_idx_t const prf_idx);
  std::unique_ptr<mem> get_gpu_mem(gpu_timetable const* gtt);
  void gpu_covert_to_r_state();
};
#pragma once

#include <cinttypes>
#include "nigiri/common/delta_t.h"
#include "nigiri/routing/gpu_raptor.h"
#include "nigiri/routing/gpu_types.h"
#include "nigiri/routing/raptor/raptor.h"
#include "nigiri/rt/frun.h"
#include "nigiri/rt/rt_timetable.h"
#include "nigiri/timetable.h"

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
  auto start_execute = std::chrono::high_resolution_clock::now();
  get_gpu_roundtimes(start_time,max_transfers,worst_time_at_dest,prf_idx);
  // Konstruktion der Ergebnis-Journey
  //TODO: kann wieder die normale round_times aus dem raptor_state verwenden da ich ja zurück übersetzen
  auto const end_k = std::min(max_transfers, kMaxTransfers) + 1U;
  auto start_journey = std::chrono::high_resolution_clock::now();
  for (auto i = 0U; i != n_locations_; ++i) {
    auto const is_dest = is_dest_[i];
    if (!is_dest) {
      continue;
    }
    for (auto k = 1U; k != end_k; ++k) {
      auto const dest_time = cpu_state_.round_times_[k][i];
      if (dest_time != kInvalid) {
        trace("ADDING JOURNEY: start={}, dest={} @ {}, transfers={}\n",
              start_time, delta_to_unix(base(), state_.round_times_[k][i]),
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
  }auto end_journey = std::chrono::high_resolution_clock::now();
  auto journey_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_journey - start_journey).count();
  std::cout << "journey Time: " << journey_duration << " microseconds\n";
  auto end_execute = std::chrono::high_resolution_clock::now();
  auto execute_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_execute - start_execute).count();
  std::cout << "execute Time: " << execute_duration << " microseconds\n";
}

template <direction SearchDir, bool Rt>
void gpu_raptor_translator<SearchDir, Rt>::reconstruct(const query& q,
                                                       journey& j){

  auto start_reconstruct = std::chrono::high_resolution_clock::now();
  reconstruct_journey<SearchDir>(tt_, rtt_, q,cpu_state_, j, base(), base_);
  auto end_reconstruct = std::chrono::high_resolution_clock::now();
  auto reconstruct_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_reconstruct - start_reconstruct).count();
  std::cout << "reconstruct Time: " << reconstruct_duration << " microseconds\n";
}
inline int translator_as_int(day_idx_t const d)  { return static_cast<int>(d.v_); } //as_int von raptor rüber kopiert da nicht static dort
template <nigiri::direction SearchDir, bool Rt>
date::sys_days gpu_raptor_translator<SearchDir, Rt>::base() const{
  return tt_.internal_interval_days().from_ +translator_as_int(base_) * date::days{1};
};

#include "gtest/gtest.h"
static gpu_timetable* translate_tt_in_gtt(nigiri::timetable tt) {

  gpu_locations locations_ = gpu_locations(
      reinterpret_cast<gpu_vector_map<gpu_location_idx_t, gpu_u8_minutes>*>(
          &tt.locations_.transfer_time_),
      reinterpret_cast<gpu_vecvec<gpu_location_idx_t, nigiri::gpu_footpath>*>(
          &tt.locations_.footpaths_out_),
      reinterpret_cast<gpu_vecvec<gpu_location_idx_t, nigiri::gpu_footpath>*>(
          &tt.locations_.footpaths_in_));

  uint32_t n_locations = tt.n_locations();;
  uint32_t n_routes = tt.n_routes();
  auto gtt_stop_time_ranges =
      reinterpret_cast<gpu_vector_map<gpu_route_idx_t,
                                      nigiri::gpu_interval<std::uint32_t>>*>(
          &tt.route_stop_time_ranges_);
  for (int i = 0; i < (*gtt_stop_time_ranges).size(); ++i) {
    for (int j = 0; j < (*gtt_stop_time_ranges)[gpu_route_idx_t{i}].size(); ++j) {
      EXPECT_EQ((*gtt_stop_time_ranges)[gpu_route_idx_t{i}].from_,tt.route_stop_time_ranges_[route_idx_t{i}].from_);
    }
  }

  auto gtt = create_gpu_timetable(
      reinterpret_cast<gpu_delta*>(tt.route_stop_times_.data()),
      tt.route_stop_times_.size(),
      reinterpret_cast<gpu_vecvec<gpu_route_idx_t, gpu_value_type>*>(
          &tt.route_location_seq_),
      reinterpret_cast<gpu_vecvec<gpu_location_idx_t, gpu_route_idx_t>*>(
          &tt.location_routes_),
      n_locations,
      n_routes,
      gtt_stop_time_ranges,
      reinterpret_cast<gpu_vector_map<
          gpu_route_idx_t, nigiri::gpu_interval<gpu_transport_idx_t>>*>(
          &tt.route_transport_ranges_),
      reinterpret_cast<gpu_vector_map<gpu_bitfield_idx_t,gpu_bitfield>*>(
          &tt.bitfields_),
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
gpu_raptor_translator<SearchDir, Rt>::gpu_raptor_translator(
    nigiri::timetable const& tt,
    nigiri::rt_timetable const* rtt,
    gpu_timetable const* gtt,
    algo_state_t & state,
    std::vector<uint8_t>& is_dest,
    std::vector<std::uint16_t>& dist_to_dest,
    std::vector<std::uint16_t>& lb,
    nigiri::day_idx_t const base,
    nigiri::routing::clasz_mask_t const allowed_claszes)
    : tt_{tt},
      rtt_{rtt},
      gtt_{gtt},
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
  cpu_state_ = routing::raptor_state{};
  auto start_constuct = std::chrono::high_resolution_clock::now();
  auto& gpu_base = *reinterpret_cast<gpu_day_idx_t*>(&base_);
  auto& gpu_allowed_claszes = *reinterpret_cast<gpu_clasz_mask_t*>(&allowed_claszes_);
  auto start_mem = std::chrono::high_resolution_clock::now();
  mem_ = std::move(get_gpu_mem(gtt));
  auto end_mem = std::chrono::high_resolution_clock::now();
  auto mem_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_mem - start_mem).count();
  std::cout << "mem Time: " << mem_duration << " microseconds\n";
  gpu_r_ = std::make_unique<gpu_raptor<gpu_direction_,Rt>>(gtt_,mem_.get(), is_dest_,dist_to_end_, lb_, gpu_base, gpu_allowed_claszes,tt_.internal_interval_days().size().count()); //TODO: next SEH error also falscher pointer oder so...
  auto end_constuct = std::chrono::high_resolution_clock::now();
  auto constuct_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_constuct - start_constuct).count();
  std::cout << "constuct Time: " << constuct_duration << " microseconds\n";
}
template <nigiri::direction SearchDir, bool Rt>
gpu_raptor_translator<SearchDir, Rt>::~gpu_raptor_translator(){
  if (mem_ != nullptr) {
    mem_.reset();
  }
};
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

  auto start_reset = std::chrono::high_resolution_clock::now();
  if (gpu_direction_ == gpu_direction::kForward && Rt == true) {
    get<std::unique_ptr<gpu_raptor<gpu_direction::kForward,true>>>(gpu_r_)->reset_arrivals();
  } else if (gpu_direction_ == gpu_direction::kForward && Rt == false) {
    get<std::unique_ptr<gpu_raptor<gpu_direction::kForward,false>>>(gpu_r_)->reset_arrivals();
  } else if (gpu_direction_ == gpu_direction::kBackward && Rt == true) {
    get<std::unique_ptr<gpu_raptor<gpu_direction::kBackward,true>>>(gpu_r_)->reset_arrivals();
  } else if (gpu_direction_ == gpu_direction::kBackward && Rt == false) {
    get<std::unique_ptr<gpu_raptor<gpu_direction::kBackward,false>>>(gpu_r_)->reset_arrivals();
  }
  auto end_reset = std::chrono::high_resolution_clock::now();
  auto reset_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_reset - start_reset).count();
  std::cout << "reset Time: " << reset_duration << " microseconds\n";
}

template <nigiri::direction SearchDir, bool Rt>
void gpu_raptor_translator<SearchDir, Rt>::next_start_time() {
  auto start_next = std::chrono::high_resolution_clock::now();
  if (gpu_direction_ == gpu_direction::kForward && Rt == true) {
    get<std::unique_ptr<gpu_raptor<gpu_direction::kForward,true>>>(gpu_r_)->next_start_time();
  } else if (gpu_direction_ == gpu_direction::kForward && Rt == false) {
    get<std::unique_ptr<gpu_raptor<gpu_direction::kForward,false>>>(gpu_r_)->next_start_time();
  } else if (gpu_direction_ == gpu_direction::kBackward && Rt == true) {
    get<std::unique_ptr<gpu_raptor<gpu_direction::kBackward,true>>>(gpu_r_)->next_start_time();
  } else if (gpu_direction_ == gpu_direction::kBackward && Rt == false) {
    get<std::unique_ptr<gpu_raptor<gpu_direction::kBackward,false>>>(gpu_r_)->next_start_time();
  }
  auto end_next = std::chrono::high_resolution_clock::now();
  auto next_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_next - start_next).count();
  std::cout << "next Time: " << next_duration << " microseconds\n";
}

template <nigiri::direction SearchDir, bool Rt>
void gpu_raptor_translator<SearchDir, Rt>::add_start(nigiri::location_idx_t const l,
                                                     nigiri::unixtime_t const t) {

    auto start_add = std::chrono::high_resolution_clock::now();

  auto gpu_l = *reinterpret_cast<const gpu_location_idx_t*>(&l);
  gpu_unixtime_t gpu_t = *reinterpret_cast<gpu_unixtime_t const*>(&t);
  if (gpu_direction_ == gpu_direction::kForward && Rt == true) {
    get<std::unique_ptr<gpu_raptor<gpu_direction::kForward,true>>>(gpu_r_)->add_start(gpu_l,gpu_t);
  } else if (gpu_direction_ == gpu_direction::kForward && Rt == false) {
    get<std::unique_ptr<gpu_raptor<gpu_direction::kForward,false>>>(gpu_r_)->add_start(gpu_l,gpu_t);
  } else if (gpu_direction_ == gpu_direction::kBackward && Rt == true) {
    get<std::unique_ptr<gpu_raptor<gpu_direction::kBackward,true>>>(gpu_r_)->add_start(gpu_l,gpu_t);
  } else if (gpu_direction_ == gpu_direction::kBackward && Rt == false) {
    get<std::unique_ptr<gpu_raptor<gpu_direction::kBackward,false>>>(gpu_r_)->add_start(gpu_l,gpu_t);
  }
  auto end_add = std::chrono::high_resolution_clock::now();
  auto add_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_add - start_add).count();
  std::cout << "add Time: " << add_duration << " microseconds\n";
}

// hier wird Kernel aufgerufen
template <nigiri::direction SearchDir, bool Rt>
void gpu_raptor_translator<SearchDir, Rt>::get_gpu_roundtimes(
    nigiri::unixtime_t const start_time,
    uint8_t const max_transfers,
    nigiri::unixtime_t const worst_time_at_dest,
    nigiri::profile_idx_t const prf_idx) {
  auto& gpu_start_time = *reinterpret_cast<gpu_unixtime_t const*>(&start_time);

  auto& gpu_worst_time_at_dest = *reinterpret_cast<gpu_unixtime_t const*>(&worst_time_at_dest);;

  auto& gpu_prf_idx = *reinterpret_cast<gpu_profile_idx_t const*>(&prf_idx);

  if (gpu_direction_ == gpu_direction::kForward && Rt == true) {
    get<std::unique_ptr<gpu_raptor<gpu_direction::kForward,true>>>(gpu_r_)->execute(gpu_start_time,max_transfers,gpu_worst_time_at_dest,prf_idx);
  } else if (gpu_direction_ == gpu_direction::kForward && Rt == false) {
    get<std::unique_ptr<gpu_raptor<gpu_direction::kForward,false>>>(gpu_r_)->execute(gpu_start_time,max_transfers,gpu_worst_time_at_dest,prf_idx);
  } else if (gpu_direction_ == gpu_direction::kBackward && Rt == true) {
    get<std::unique_ptr<gpu_raptor<gpu_direction::kBackward,true>>>(gpu_r_)->execute(gpu_start_time,max_transfers,gpu_worst_time_at_dest,prf_idx);
  } else if (gpu_direction_ == gpu_direction::kBackward && Rt == false) {
    get<std::unique_ptr<gpu_raptor<gpu_direction::kBackward,false>>>(gpu_r_)->execute(gpu_start_time,max_transfers,gpu_worst_time_at_dest,prf_idx);
  }
  gpu_covert_to_r_state();
}

template <nigiri::direction SearchDir, bool Rt>
std::unique_ptr<mem> gpu_raptor_translator<SearchDir, Rt>::get_gpu_mem(gpu_timetable const* gtt) {
  return gpu_mem(state_,gpu_direction_,gtt);
}

template <nigiri::direction SearchDir, bool Rt>
void gpu_raptor_translator<SearchDir, Rt>::gpu_covert_to_r_state() {
  auto gpu_tmp = mem_->host_.tmp_;
  auto gpu_best = mem_->host_.best_;
  auto gpu_round_times = mem_->host_.round_times_;
  auto gpu_columns = mem_->device_.column_count_round_times_;
  auto gpu_rows = mem_->device_.row_count_round_times_;
  vector<gpu_delta_t> vec(gpu_round_times.data(), gpu_round_times.data() + gpu_rows * gpu_columns);
  cista::raw::flat_matrix<delta_t> matrix;
  matrix.resize(gpu_rows,gpu_columns);
  matrix.entries_ = std::move(vec);

  cpu_state_.tmp_ = gpu_tmp;
  cpu_state_.best_ = gpu_best;
  cpu_state_.round_times_ = matrix;
  state_.tmp_ = mem_->host_.tmp_;
  state_.best_ = mem_->host_.best_;
  state_.station_mark_ = mem_->host_.station_mark_;
  state_.prev_station_mark_ = mem_->host_.prev_station_mark_;
  state_.route_mark_ = mem_->host_.route_mark_;
}
