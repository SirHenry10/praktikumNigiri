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
  using algo_state_t = nigiri::routing::raptor_state;
  using algo_stats_t = gpu_raptor_stats;
  static nigiri::direction const cpu_direction_ = SearchDir;
  static gpu_direction const gpu_direction_ =
      static_cast<enum gpu_direction const>(cpu_direction_); //TODO: reinterprate cast maybe weiß nicht ob static geht
  std::variant<
      std::unique_ptr<gpu_raptor<gpu_direction::kForward, true>>,
      std::unique_ptr<gpu_raptor<gpu_direction::kForward, false>>,
      std::unique_ptr<gpu_raptor<gpu_direction::kBackward, true>>,
      std::unique_ptr<gpu_raptor<gpu_direction::kBackward, false>>
      > gpu_r_;
  std::unique_ptr<mem> mem_;
  std::unique_ptr<loaned_mem> loaned_mem_;

  gpu_raptor_translator(nigiri::timetable const& tt,
                        nigiri::rt_timetable const* rtt,
                        algo_state_t & state,
                        std::vector<bool>& is_dest,
                        std::vector<std::uint16_t>& dist_to_dest,
                        std::vector<std::uint16_t>& lb,
                        nigiri::day_idx_t const base,
                        nigiri::routing::clasz_mask_t const allowed_claszes);
  algo_stats_t get_stats();
  //TODO: destructor bauen mit: if (mem_ != nullptr) {
  //         delete mem_;
  //         mem_ = nullptr;
  //       }
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
  gpu_delta_t* get_gpu_roundtimes(nigiri::unixtime_t const start_time,
                                  uint8_t const max_transfers,
                                  nigiri::unixtime_t const worst_time_at_dest,
                                  nigiri::profile_idx_t const prf_idx);
  std::unique_ptr<mem> get_gpu_mem(gpu_timetable* gtt);
  void gpu_covert_to_r_state();
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
  std::cerr << "Test gpu_raptor_translator::execute()" << std::endl;
  gpu_delta_t* gpu_round_times = get_gpu_roundtimes(start_time,max_transfers,worst_time_at_dest,prf_idx);
  // Konstruktion der Ergebnis-Journey
  //TODO: kann wieder die normale round_times aus dem raptor_state verwenden da ich ja zurück übersetzen
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
  std::cerr << "Test gpu_raptor_translator::execute() ende"<< std::endl;
}

template <direction SearchDir, bool Rt>
void gpu_raptor_translator<SearchDir, Rt>::reconstruct(const query& q,
                                                       journey& j){
  std::cerr << "Test gpu_raptor_translator::reconstruct()" << std::endl;
  reconstruct_journey<SearchDir>(tt_, rtt_, q, state_, j, base(), base_);
  std::cerr << "Test gpu_raptor_translator::reconstruct() ende" << std::endl;
}
inline int translator_as_int(day_idx_t const d)  { return static_cast<int>(d.v_); } //as_int von raptor rüber kopiert da nicht static dort
template <nigiri::direction SearchDir, bool Rt>
date::sys_days gpu_raptor_translator<SearchDir, Rt>::base() const{
  return tt_.internal_interval_days().from_ +translator_as_int(base_) * date::days{1};
};

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
  auto gtt = create_gpu_timetable(
      reinterpret_cast<gpu_delta*>(tt.route_stop_times_.data()),
      tt.route_stop_times_.size(),
      reinterpret_cast<gpu_vecvec<gpu_route_idx_t, gpu_value_type>*>(
          &tt.route_location_seq_),
      reinterpret_cast<gpu_vecvec<gpu_location_idx_t, gpu_route_idx_t>*>(
          &tt.location_routes_),
      n_locations,
      n_routes,
      reinterpret_cast<gpu_vector_map<gpu_route_idx_t,
                                      nigiri::gpu_interval<std::uint32_t>>*>(
          &tt.route_stop_time_ranges_),
      reinterpret_cast<gpu_vector_map<
          gpu_route_idx_t, nigiri::gpu_interval<gpu_transport_idx_t>>*>(
          &tt.route_transport_ranges_),
      reinterpret_cast<gpu_vector_map<gpu_bitfield_idx_t, gpu_bitfield>*>(
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
  std::cerr << "Test gpu_raptor_translator()" << std::endl;
  auto& gpu_base = *reinterpret_cast<gpu_day_idx_t*>(&base_);
  auto& gpu_allowed_claszes = *reinterpret_cast<gpu_clasz_mask_t*>(&allowed_claszes_);
  auto gtt = translate_tt_in_gtt(tt_);
  mem_ = std::move(get_gpu_mem(gtt));
  std::cerr << "Test gpu_raptor_translator() std::make_unique" << std::endl;
  gpu_r_ = std::make_unique<gpu_raptor<gpu_direction_,Rt>>(gtt,mem_.get(), is_dest_,dist_to_end_, lb_, gpu_base, gpu_allowed_claszes,tt_.internal_interval_days().size().count()); //TODO: next SEH error also falscher pointer oder so...
  std::cerr << "Test gpu_raptor_translator() ende" << std::endl;
}
using algo_stats_t = gpu_raptor_stats;
template <nigiri::direction SearchDir, bool Rt>
algo_stats_t gpu_raptor_translator<SearchDir, Rt>::get_stats() {
  std::cerr << "Test gpu_raptor_translator::get_stats()" << std::endl;
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
  std::cerr << "Test gpu_raptor_translator::reset_arrivals()" << std::endl;
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
  std::cerr << "Test gpu_raptor_translator::next_start_time" << std::endl;
  if (gpu_direction_ == gpu_direction::kForward && Rt == true) {
    get<std::unique_ptr<gpu_raptor<gpu_direction::kForward,true>>>(gpu_r_)->next_start_time();
  } else if (gpu_direction_ == gpu_direction::kForward && Rt == false) {
    get<std::unique_ptr<gpu_raptor<gpu_direction::kForward,false>>>(gpu_r_)->next_start_time();
  } else if (gpu_direction_ == gpu_direction::kBackward && Rt == true) {
    get<std::unique_ptr<gpu_raptor<gpu_direction::kBackward,true>>>(gpu_r_)->next_start_time();
  } else if (gpu_direction_ == gpu_direction::kBackward && Rt == false) {
    get<std::unique_ptr<gpu_raptor<gpu_direction::kBackward,false>>>(gpu_r_)->next_start_time();
  }
  std::cerr << "Test gpu_raptor_translator::next_start_time() ende" << std::endl;
}

template <nigiri::direction SearchDir, bool Rt>
void gpu_raptor_translator<SearchDir, Rt>::add_start(nigiri::location_idx_t const l,
                                                     nigiri::unixtime_t const t) {
  std::cerr << "Test gpu_raptor_translator::add_start()" << std::endl;
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
  std::cerr << "Test gpu_raptor_translator::add_start() ende" << std::endl;
}

// hier wird Kernel aufgerufen
template <nigiri::direction SearchDir, bool Rt>
gpu_delta_t* gpu_raptor_translator<SearchDir, Rt>::get_gpu_roundtimes(
    nigiri::unixtime_t const start_time,
    uint8_t const max_transfers,
    nigiri::unixtime_t const worst_time_at_dest,
    nigiri::profile_idx_t const prf_idx) {
  std::cerr << "Test gpu_raptor_translator::get_gpu_roundtimes()" << std::endl;
  gpu_unixtime_t gpu_start_time = *reinterpret_cast<gpu_unixtime_t const*>(&start_time);

  gpu_unixtime_t gpu_worst_time_at_dest = *reinterpret_cast<gpu_unixtime_t const*>(&worst_time_at_dest);;

  auto gpu_prf_idx = *reinterpret_cast<gpu_profile_idx_t const*>(&prf_idx);
  gpu_delta_t* gpu_round_times;
  if (auto gpu_r = get_if<std::unique_ptr<gpu_raptor<gpu_direction::kForward,true>>>(&gpu_r_)->get()) {
    gpu_round_times = gpu_r->execute(gpu_start_time,max_transfers,gpu_worst_time_at_dest,prf_idx);
  } else if (auto gpu_r = get_if<std::unique_ptr<gpu_raptor<gpu_direction::kForward,false>>>(&gpu_r_)->get()) {
    gpu_round_times = gpu_r->execute(gpu_start_time,max_transfers,gpu_worst_time_at_dest,prf_idx);
  } else if (auto gpu_r = get_if<std::unique_ptr<gpu_raptor<gpu_direction::kBackward, true>>>(&gpu_r_)->get()) {
    gpu_round_times = gpu_r->execute(gpu_start_time,max_transfers,gpu_worst_time_at_dest,prf_idx);
  } else if (auto gpu_r = get_if<std::unique_ptr<gpu_raptor<gpu_direction::kBackward,false>>>(&gpu_r_)->get()) {
    gpu_round_times = gpu_r->execute(gpu_start_time,max_transfers,gpu_worst_time_at_dest,prf_idx);
  }
  gpu_covert_to_r_state();
  std::cerr << "Test gpu_raptor_translator::get_gpu_roundtimes() ende" << std::endl;
  return gpu_round_times;
}

template <nigiri::direction SearchDir, bool Rt>
std::unique_ptr<mem> gpu_raptor_translator<SearchDir, Rt>::get_gpu_mem(gpu_timetable* gtt) {
  std::cerr << "Test gpu_raptor_translator::get_gpu_mem()" << std::endl;
  state_.resize(n_locations_,n_routes_,n_rt_transports_);
  auto& tmp = *reinterpret_cast<std::vector<gpu_delta_t>*>(&state_.tmp_);
  auto& best = *reinterpret_cast<std::vector<gpu_delta_t>*>(&state_.best_);
  return gpu_mem(tmp,best,state_.station_mark_,state_.prev_station_mark_,state_.route_mark_,gpu_direction_,gtt);
}

template <nigiri::direction SearchDir, bool Rt>
void gpu_raptor_translator<SearchDir, Rt>::gpu_covert_to_r_state() {
  std::cerr << "Test gpu_raptor_translator::gpu_covert_to_r_state()" << std::endl;
  auto gpu_tmp = mem_->host_.tmp_;
  auto gpu_best = mem_->host_.best_;
  auto gpu_round_times = mem_->host_.round_times_;
  auto gpu_columns = mem_->device_.column_count_round_times_;
  auto gpu_rows = mem_->device_.row_count_round_times_;
  auto gpu_station_mark = mem_->host_.station_mark_;
  auto gpu_prev_station_mark = mem_->host_.prev_station_mark_;
  auto gpu_route_mark = mem_->host_.route_mark_;
  vector<gpu_delta_t> vec(gpu_round_times.data(), gpu_round_times.data() + gpu_rows * gpu_columns);
  cista::raw::flat_matrix<delta_t> matrix;
  matrix.resize(gpu_rows,gpu_columns);
  matrix.entries_ = std::move(vec);
  std::vector<bool> station_mark(mem_->device_.n_locations_);
  for (int i = 0; i < station_mark.size(); ++i) {
    station_mark[i] = gpu_station_mark[i] == 0 ? false : true;
  }
  std::vector<bool> prev_station_mark(mem_->device_.n_locations_);
  for (int i = 0; i < prev_station_mark.size(); ++i) {
    prev_station_mark[i] = gpu_prev_station_mark[i] == 0 ? false : true;
  }
  std::vector<bool> route_mark(mem_->device_.n_routes_);
  for (int i = 0; i < route_mark.size(); ++i) {
    route_mark[i] = gpu_route_mark[i] == 0 ? false : true;
  }

  state_.tmp_ = gpu_tmp;
  state_.best_ = gpu_best;
  state_.round_times_ = matrix;
  state_.station_mark_ = station_mark;
  state_.prev_station_mark_ = prev_station_mark;
  state_.route_mark_ = route_mark;

  std::cerr << "Test gpu_raptor_translator::gpu_covert_to_r_state() ende" << std::endl;
}