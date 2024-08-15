#pragma once

#include <cinttypes>
#include "nigiri/loader/hrd/load_timetable.h"
#include "nigiri/common/linear_lower_bound.h"
#include "nigiri/routing/gpu_raptor_state.cuh"
#include "nigiri/routing/gpu_timetable.h"
#include "nigiri/routing/journey.h"
#include "nigiri/routing/limits.h"
#include "nigiri/routing/pareto_set.h"
#include "nigiri/routing/query.h"
#include "nigiri/routing/raptor/debug.h"
#include "nigiri/routing/raptor/raptor.h"
#include "nigiri/routing/raptor/raptor_state.h"
#include "nigiri/routing/raptor/raptor_state.h"  //maybe weg
#include "nigiri/routing/raptor/reconstruct.h"
#include "nigiri/rt/rt_timetable.h"
#include "nigiri/special_stations.h"
#include "gpu_raptor.cuh"
#include "gpu_types.h"

using namespace date;
using namespace nigiri;
using namespace nigiri::loader;
using namespace nigiri::routing;


template <direction SearchDir, bool Rt>
struct gpu_raptor_translator {
  using algo_stats_t = raptor_stats;
  static direction const cpu_direction_ = SearchDir;
  static gpu_direction const gpu_direction_ = *reinterpret_cast<enum gpu_direction const*>(&cpu_direction_);
  gpu_raptor<gpu_direction_,Rt> gpu_r_;

  gpu_raptor_translator(timetable const* tt,
             rt_timetable const* rtt,
                        raptor_state& state,
             std::vector<bool>& is_dest,
             std::vector<std::uint16_t>& dist_to_dest,
             std::vector<std::uint16_t>& lb,
             day_idx_t const base,
             clasz_mask_t const allowed_claszes)
      :rtt_{rtt},
        state_{state},
        is_dest_{is_dest},
        dist_to_end_{dist_to_end_},
        lb_{lb},
        base_{base},
        allowed_claszes_{allowed_claszes}
  {
    auto gpu_base = *reinterpret_cast<gpu_day_idx_t*>(&base_);
    auto gpu_allowed_claszes = *reinterpret_cast<gpu_clasz_mask_t *>(&allowed_claszes_);
    gpu_r_ = gpu_raptor<gpu_direction_,Rt>(translate_tt_in_gtt(tt_),rtt_,state_,is_dest_,dist_to_dest,lb_,gpu_base,gpu_allowed_claszes);
  }
  algo_stats_t get_stats() const {
    return gpu_r_.get_stats();
  }

  void reset_arrivals() {
    gpu_r_.reset_arrivals();
  }

  void next_start_time() {
    gpu_r_.next_start_time();
  }

  void add_start(location_idx_t const l, unixtime_t const t) {
    auto gpu_l = *reinterpret_cast<const gpu_location_idx_t*>(&l);
    auto gpu_t = *reinterpret_cast<const gpu_unixtime_t*>(&t);
    gpu_r_.add_start(gpu_l,gpu_t);
  }


  // hier wird Kernel aufgerufen
  void execute(unixtime_t const start_time,
               uint8_t const max_transfers,
               unixtime_t const worst_time_at_dest,
               profile_idx_t const prf_idx,
               nigiri::pareto_set<nigiri::routing::journey>& results){

  }


  void reconstruct(nigiri::routing::query const& q, nigiri::routing::journey& j) {
    reconstruct_journey<SearchDir>(tt_, rtt_, q, state_, j, base(), base_);
  }
  timetable const& tt_;
  rt_timetable const* rtt_{nullptr};
  raptor_state& state_;
  std::vector<bool>& is_dest_;
  std::vector<std::uint16_t>& dist_to_end_;
  std::vector<std::uint16_t>& lb_;
  std::array<delta_t, kMaxTransfers + 1> time_at_dest_;
  day_idx_t base_;
  int n_days_;
  raptor_stats stats_;
  std::uint32_t n_locations_, n_routes_, n_rt_transports_;
  clasz_mask_t allowed_claszes_;
private:
  date::sys_days base() const {
    return tt_.internal_interval_days().from_ + raptor<SearchDir, Rt>::as_int(base_) * date::days{1};
  }
  gpu_timetable& translate_tt_in_gtt(timetable& tt){
    vector_map<bitfield_idx_t,std::uint64_t*> bitfields_data_ = vector_map<bitfield_idx_t,std::uint64_t*>();
    for (bitfield_idx_t i = bitfield_idx_t{0}; i< tt.bitfields_.size(); ++i) {
    auto t = tt.bitfields_.at(i);
    bitfields_data_.emplace_back(std::make_pair(i,t.blocks_.data()));
    }
    auto gpu_bitfields_data_ = reinterpret_cast<gpu_vector_map<gpu_bitfield_idx_t,std::uint64_t*>*>(&bitfields_data_);
    gpu_locations locations_ = gpu_locations(*reinterpret_cast<gpu_vector_map<gpu_location_idx_t, gpu_u8_minutes>*>(&tt.locations_.transfer_time_),
                                             *reinterpret_cast<gpu_vecvec<gpu_location_idx_t, gpu_footpath>*>(&tt.locations_.footpaths_out_),
                                             *reinterpret_cast<gpu_vecvec<gpu_location_idx_t, gpu_footpath>*>(&tt.locations_.footpaths_in_));
    auto n_locations = tt.n_locations();
    auto n_routes = tt.n_routes();
    auto gtt = create_gpu_timetable(
        reinterpret_cast<gpu_delta*>(tt.route_stop_times_.data()),
        tt.route_stop_times_.size(),
        reinterpret_cast<gpu_vecvec<gpu_route_idx_t,gpu_value_type>*>(&tt.route_location_seq_),
        reinterpret_cast<gpu_vecvec<gpu_location_idx_t , gpu_route_idx_t>*>(&tt.location_routes_),
        &n_locations,
        &n_routes,
        reinterpret_cast<gpu_vector_map<gpu_route_idx_t,nigiri::gpu_interval<std::uint32_t>>*>(&tt.route_stop_time_ranges_),
        reinterpret_cast<gpu_vector_map<gpu_route_idx_t,nigiri::gpu_interval<gpu_transport_idx_t >>*>(&tt.route_transport_ranges_),
        reinterpret_cast<gpu_vector_map<gpu_bitfield_idx_t, gpu_bitfield>*>(&tt.bitfields_),
        gpu_bitfields_data_,
        reinterpret_cast<gpu_vector_map<gpu_transport_idx_t,gpu_bitfield_idx_t>*>(&tt.transport_traffic_days_),
        reinterpret_cast<nigiri::gpu_interval<gpu_sys_days>*>(&tt.date_range_),
        &locations_,
        reinterpret_cast<gpu_vector_map<gpu_route_idx_t, gpu_clasz>* >(&tt.route_clasz_)
    );
    return gtt;
  }
};