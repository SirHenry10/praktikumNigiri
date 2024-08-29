#pragma once
#include "nigiri/routing/gpu_raptor.h"
#include <iostream>

#include <cooperative_groups.h>
using namespace cooperative_groups;
// leader type must be unsigned 32bit
// no leader is a zero ballot vote (all 0) minus 1 => with underflow all 1's
constexpr unsigned int FULL_MASK = 0xFFFFffff;
constexpr unsigned int NO_LEADER = FULL_MASK;

__device__ __forceinline__ unsigned int get_block_thread_id() {
  return threadIdx.x + (blockDim.x * threadIdx.y);
}

__device__ __forceinline__ unsigned int get_global_thread_id() {
  return get_block_thread_id() + (blockDim.x * blockDim.y * blockIdx.x);
}

__device__ __forceinline__ unsigned int get_block_stride() {
  return blockDim.x * blockDim.y;
}

__device__ __forceinline__ unsigned int get_global_stride() {
  return get_block_stride() * gridDim.x * gridDim.y;
}

// für die uint32_t station/route_marks
__device__ void mark(unsigned int* store, unsigned int const idx) {
  unsigned int const store_idx = (idx >> 5);  // divide by 32
  unsigned int const mask = 1 << (idx % 32);
  atomicOr(&store[store_idx], mask);
}

__device__ bool marked(unsigned int const* const store, unsigned int idx) {
  unsigned int const store_idx = (idx >> 5);  // divide by 32
  unsigned int const val = store[store_idx];
  unsigned int const mask = 1 << (idx % 32);
  return (bool)(val & mask);
}

__device__ void reset_store(unsigned int* store, int const store_size) {
  auto const t_id = get_global_thread_id();
  auto const stride = get_global_stride();

  for (auto idx = t_id; idx < store_size; idx += stride) {
    store[idx] = 0;
  }
}

__device__ bool update_arrival(gpu_delta_t* base_,
                               const unsigned int l_idx, gpu_delta_t const val){
#if __CUDA_ARCH__ >= 700

  auto old_value = base_[l_idx];
  gpu_delta_t assumed;

  do {
    if (old_value <= val) {
      return false;
    }

    assumed = old_value;

    old_value = atomicCAS(reinterpret_cast<int*>(&base_[l_idx]), assumed, val);
  } while (assumed != old_value);

  return true;
#else
  gpu_delta_t* const arr_address = &base_[l_idx];
  auto* base_address = (unsigned int*)((size_t)arr_address & ~2);
  unsigned int old_value, assumed, new_value, compare_val;

  old_value = *base_address;

  do {
    assumed = old_value;

    if ((size_t)arr_address & 2) {
      compare_val = (0x0000FFFF & assumed) ^ (((unsigned int)val) << 16);
    } else {
      compare_val = (0xFFFF0000 & assumed) ^ (unsigned int)val;
    }

    new_value = __vminu2(old_value, compare_val);

    if (new_value == old_value) {
      return false;
    }

    old_value = atomicCAS(base_address, assumed, new_value);
  } while (assumed != old_value);

  return true;
#endif
}

template <gpu_direction SearchDir, bool Rt>
__device__ void update_time_at_dest(unsigned const k, gpu_delta_t const t, gpu_delta_t* time_at_dest_){
  for (auto i = k; i < gpu_kMaxTransfers+1; ++i) {
    time_at_dest_[i] = get_best<SearchDir>(time_at_dest_[i], t);
  }
}

template <gpu_direction SearchDir, bool Rt>
__device__ void convert_station_to_route_marks(unsigned int* station_marks, unsigned int* route_marks,
                                               bool* any_station_marked, gpu_timetable* gtt_) {
  auto const global_t_id = get_global_thread_id();
  auto const global_stride = get_global_stride();
  // anstatt stop_count_ brauchen wir location_routes ?location_idx_{gtt.n_locations}?
  for (uint32_t idx = global_t_id;
       idx < gtt_->n_locations_; idx += global_stride) {
    if (marked(station_marks, idx)) {
      if (!*any_station_marked) {
        *any_station_marked = true;
      }
      auto const& location_routes = (*gtt_->location_routes_)[gpu_location_idx_t{idx}];
      for (auto r : location_routes) {
        mark(route_marks, gpu_to_idx(r));
      }
    }
  }
}

template <gpu_direction SearchDir, bool Rt>
__device__ gpu_delta_t time_at_stop(gpu_route_idx_t const r, gpu_transport const t,
                                    gpu_stop_idx_t const stop_idx,
                                    gpu_event_type const ev_type,
                                    gpu_timetable* gtt_,
                                    gpu_day_idx_t base_){
  auto const range = *gtt_->route_transport_ranges_;
  auto const n_transports = static_cast<unsigned>(range.size());
  auto const route_stop_begin = static_cast<unsigned>(range[r].from_.v_ + n_transports *
                                                                              (stop_idx * 2 - (ev_type==gpu_event_type::kArr ? 1 : 0)));
  return gpu_clamp((as_int(t.day_) - as_int(base_)) * 1440
                   + gtt_->route_stop_times_[route_stop_begin +
                                             (gpu_to_idx(t.day_) - gpu_to_idx(range[r].from_))].count());
}

template <typename It, typename End, typename Key, typename Cmp>
__device__ It linear_lb(It from, End to, Key&& key, Cmp&& cmp) {
  for (auto it = from; it != to; ++it) {
    if (!cmp(*it, key)) {
      return it;
    }
  }
  return to;
}

template <gpu_direction SearchDir, bool Rt>
__device__ bool is_transport_active(gpu_transport_idx_t const t,
                                    std::size_t const day , gpu_timetable* gtt_)  {
  return (*gtt_->bitfields_)[(*gtt_->transport_traffic_days_)[t]].test(day);
}

template <gpu_direction SearchDir>
__device__ bool is_valid(gpu_delta_t t) {
  // Use if constexpr to ensure compile-time evaluation
  if constexpr (SearchDir == gpu_direction::kForward) {
    return t != cuda::std::numeric_limits<gpu_delta_t>::max();
  } else {
    return t != cuda::std::numeric_limits<gpu_delta_t>::min();
  }
}



template <gpu_direction SearchDir, bool Rt>
__device__ bool update_route_smaller32(unsigned const k, gpu_route_idx_t r,
                                       gpu_raptor_stats* stats_,
                                       uint32_t* prev_station_mark_, gpu_delta_t* best_,
                                       gpu_delta_t* round_times_, uint32_t row_count_round_times_,
                                       gpu_delta_t* tmp_,short const* kMaxTravelTimeTicks_,
                                       uint16_t* lb_, int n_days_,
                                       gpu_delta_t* time_at_dest_,
                                       uint32_t* station_mark_, gpu_day_idx_t* base_,
                                       unsigned short kUnreachable, bool any_station_marked_,
                                       gpu_delta* route_stop_times,
                                       gpu_vecvec<gpu_route_idx_t,gpu_value_type>* route_location_seq,
                                       gpu_vecvec<gpu_location_idx_t , gpu_route_idx_t>* location_routes,
                                       std::uint32_t const n_locations,
                                       std::uint32_t const n_routes,
                                       gpu_vector_map<gpu_route_idx_t,gpu_interval<std::uint32_t>>* route_stop_time_ranges,
                                       gpu_vector_map<gpu_route_idx_t,gpu_interval<gpu_transport_idx_t >>* route_transport_ranges,
                                       gpu_vector_map<gpu_bitfield_idx_t, gpu_bitfield>* bitfields,
                                       gpu_vector_map<gpu_transport_idx_t,gpu_bitfield_idx_t>* transport_traffic_days,
                                       gpu_interval<gpu_sys_days>* date_range,
                                       gpu_locations* locations,
                                       gpu_vector_map<gpu_route_idx_t, gpu_clasz>* route_clasz){
  auto const t_id = threadIdx.x;
  auto const stop_seq = (*gtt_->route_location_seq_)[r];
  auto stop_idx = -1;
  gpu_stop stp{};
  unsigned int l_idx;
  //bool is_last;
  gpu_delta_t prev_round_time = ((SearchDir == gpu_direction::kForward) ? cuda::std::numeric_limits<gpu_delta_t>::max()
                                                                 : cuda::std::numeric_limits<gpu_delta_t>::min());

  unsigned leader = stop_seq.size();
  unsigned int active_stop_count = stop_seq.size();

  if(t_id == 0){
    any_station_marked_= false;
  }
  if(t_id < active_stop_count) {
    stop_idx = static_cast<gpu_stop_idx_t>(
        (SearchDir == gpu_direction::kForward) ? t_id : stop_seq.size() - t_id - 1U);
    stp = gpu_stop{stop_seq[stop_idx]};
    l_idx = gpu_to_idx(stp.gpu_location_idx());
    //is_last = t_id == stop_seq.size() - 1U;
    // ist äquivalent zu prev_arrival
    prev_round_time = round_times_[(k - 1) * row_count_round_times_ + l_idx];
  }

  if (!__any_sync(FULL_MASK, prev_round_time!=cuda::std::numeric_limits<gpu_delta_t>::max())) {
    return any_station_marked_;
  }

  // berechnen von allen möglichen trips(Abfahrt-/Ankunftszeiten) von dieser station
  auto const splitter = gpu_split_day_mam(*base_, prev_round_time);
  auto const day_at_stop = splitter.first;
  auto const mam = splitter.second;
  auto const n_days_to_iterate = get_smaller(
      gpu_kMaxTravelTime.count() / 1440 + 1,
      (SearchDir == gpu_direction::kForward) ? n_days_ - as_int(day_at_stop) : as_int(day_at_stop) + 1);
  auto const arrival_times = gtt_->gpu_event_times_at_stop(r, stop_idx, gpu_event_type::kArr);
  auto const departure_times = gtt_->gpu_event_times_at_stop(r, stop_idx, gpu_event_type::kDep);
  auto const seek_first_day = [&]() {
    return linear_lb(gpu_get_begin_it<SearchDir>(departure_times),
                     gpu_get_end_it<SearchDir>(departure_times), mam,
                     [&](gpu_delta const a, gpu_minutes_after_midnight_t const b) {
                       return is_better<SearchDir>(a.mam_, b.count());
                     });
  };

  for (auto i = gpu_day_idx_t::value_t{0U}; i != n_days_to_iterate; ++i){ // die Schleife geht durch alle Tage
    auto const ev_time_range =
        gpu_it_range{i == 0U ? seek_first_day() : gpu_get_begin_it<SearchDir>(departure_times),
                 gpu_get_end_it<SearchDir>(departure_times)};
    if (ev_time_range.empty()) {
      return any_station_marked_;
    }
    auto day = (SearchDir == gpu_direction::kForward) ? day_at_stop + i : day_at_stop - i;
    for (auto it = begin(ev_time_range); it != end(ev_time_range); ++it){ // die Schleife geht durch alle Zeiten
      if(t_id < active_stop_count){
        ++stats_[l_idx>>5].n_earliest_trip_calls_;
        auto const t_offset = static_cast<cuda::std::size_t>(&*it - departure_times.data());
        auto const dep = *it;
        auto const dep_mam = dep.mam_;
        auto const dep_t = to_gpu_delta(day, dep_mam, base_);
        // election of leader
        unsigned ballot = __ballot_sync(
            FULL_MASK, (t_id < active_stop_count) && is_valid<SearchDir>(prev_round_time)  &&
                           is_valid<SearchDir>(dep_t) &&
                           (prev_round_time <= dep_t));
        leader = __ffs(ballot) - 1;
      }

      if(t_id > leader && t_id < active_stop_count){
        auto const t_offset = static_cast<cuda::std::size_t>(&*it - arrival_times.data());
        auto const arr = *it;
        auto const arr_mam = arr.mam_;
        auto const arr_t = to_gpu_delta(day, arr_mam, base_);

        bool updated = update_arrival(tmp_, stop_idx, arr_t);
        if(updated){
          mark(station_mark_, stop_idx);
          any_station_marked_ = true;
        }
      }
      if (leader != NO_LEADER) {
        active_stop_count = leader;
      }
      leader = NO_LEADER;
    }
  }
  return any_station_marked_;
}

template <gpu_direction SearchDir, bool Rt>
__device__ bool update_route_bigger32(unsigned const k, gpu_route_idx_t r,
                                      gpu_raptor_stats* stats_,
                                      uint32_t* prev_station_mark_, gpu_delta_t* best_,
                                      gpu_delta_t* round_times_, uint32_t row_count_round_times_,
                                      gpu_delta_t* tmp_, short const* kMaxTravelTimeTicks_,
                                      uint16_t* lb_, int n_days_,
                                      gpu_delta_t* time_at_dest_,
                                      uint32_t* station_mark_, gpu_day_idx_t* base_,
                                      unsigned short kUnreachable, bool any_station_marked_,
                                      gpu_delta* route_stop_times,
                                      gpu_vecvec<gpu_route_idx_t,gpu_value_type>* route_location_seq,
                                      gpu_vecvec<gpu_location_idx_t , gpu_route_idx_t>* location_routes,
                                      std::uint32_t const n_locations,
                                      std::uint32_t const n_routes,
                                      gpu_vector_map<gpu_route_idx_t,gpu_interval<std::uint32_t>>* route_stop_time_ranges,
                                      gpu_vector_map<gpu_route_idx_t,gpu_interval<gpu_transport_idx_t >>* route_transport_ranges,
                                      gpu_vector_map<gpu_bitfield_idx_t, gpu_bitfield>* bitfields,
                                      gpu_vector_map<gpu_transport_idx_t,gpu_bitfield_idx_t>* transport_traffic_days,
                                      gpu_interval<gpu_sys_days>* date_range,
                                      gpu_locations* locations,
                                      gpu_vector_map<gpu_route_idx_t, gpu_clasz>* route_clasz){
  auto const t_id = threadIdx.x;

  unsigned int leader = NO_LEADER;
  unsigned int any_arrival = 0;

  auto const stop_seq = (*gtt_->route_location_seq_)[r];
  auto stop_idx = -1;
  gpu_stop stp{};
  unsigned int l_idx;
  //bool is_last;
  gpu_delta_t prev_round_time = ((SearchDir == gpu_direction::kForward) ? cuda::std::numeric_limits<gpu_delta_t>::max()
                                                                        : cuda::std::numeric_limits<gpu_delta_t>::min());
  unsigned int active_stop_count = stop_seq.size();
  int const stage_count = (stop_seq.size() + (32 - 1)) >> 5;
  int active_stage_count = stage_count;

  // berechnen von allen möglichen trips(Abfahrt-/Ankunftszeiten) von dieser station
  auto const splitter = gpu_split_day_mam(*base_, prev_round_time);
  auto const day_at_stop = splitter.first;
  auto const mam = splitter.second;
  auto const n_days_to_iterate = get_smaller(
      gpu_kMaxTravelTime.count() / 1440 + 1,
      (SearchDir == gpu_direction::kForward) ? n_days_ - as_int(day_at_stop) : as_int(day_at_stop) + 1);
  auto const arrival_times = gtt_->gpu_event_times_at_stop(r, stop_idx, gpu_event_type::kArr);
  auto const departure_times = gtt_->gpu_event_times_at_stop(r, stop_idx, gpu_event_type::kDep);
  auto const seek_first_day = [&]() {
    return linear_lb(gpu_get_begin_it<SearchDir>(departure_times),
                     gpu_get_end_it<SearchDir>(departure_times), mam,
                     [&](gpu_delta const a, gpu_minutes_after_midnight_t const b) {
                       return is_better<SearchDir>(a.mam_, b.count());
                     });
  };

  for (auto i = gpu_day_idx_t::value_t{0U}; i != n_days_to_iterate; ++i){
    auto const ev_time_range =
        gpu_it_range{i == 0U ? seek_first_day() : gpu_get_begin_it<SearchDir>(departure_times),
                     gpu_get_end_it<SearchDir>(departure_times)};
    if (ev_time_range.empty()) {
      return any_station_marked_;
    }
    auto day = (SearchDir == gpu_direction::kForward) ? day_at_stop + i : day_at_stop - i;
    for (auto it = begin(ev_time_range); it != end(ev_time_range); ++it){
      for(int current_stage=0; current_stage<active_stage_count; ++current_stage){
        int stage_id = (current_stage << 5) + t_id;
        if(stage_id < active_stop_count){
          stop_idx = static_cast<gpu_stop_idx_t>(
              (SearchDir == gpu_direction::kForward) ? t_id : stop_seq.size() - t_id - 1U);
          stp = gpu_stop{stop_seq[stop_idx]};
          l_idx = gpu_to_idx(stp.gpu_location_idx());
          //is_last = t_id == stop_seq.size() - 1U;
          // ist äquivalent zu prev_arrival
          prev_round_time = round_times_[(k - 1) * row_count_round_times_ + l_idx];
        }
        any_arrival |= __any_sync(FULL_MASK, is_valid<SearchDir>(prev_round_time));
        if (current_stage == active_stage_count - 1 && !any_arrival) {
          return any_station_marked_;
        }
        if(!any_arrival){
          continue;
        }
        if(stage_id < active_stop_count){
          ++stats_[l_idx>>5].n_earliest_trip_calls_;
          auto const t_offset = static_cast<cuda::std::size_t>(&*it - departure_times.data());
          auto const dep = *it;
          auto const dep_mam = dep.mam_;
          auto const dep_t = to_gpu_delta(day, dep_mam, base_);
          // election of leader
          unsigned ballot = __ballot_sync(
              FULL_MASK, (t_id < active_stop_count) && is_valid<SearchDir>(prev_round_time)  &&
                             is_valid<SearchDir>(dep_t) &&
                             (prev_round_time <= dep_t));
          leader = __ffs(ballot) - 1;
        }
        if(leader != NO_LEADER){
          leader += current_stage << 5;
        }
        // zuerst current stage updaten
        if(leader != NO_LEADER && stage_id < active_stop_count){
          if(stage_id > leader){
            auto const t_offset = static_cast<cuda::std::size_t>(&*it - arrival_times.data());
            auto const arr = *it;
            auto const arr_mam = arr.mam_;
            auto const arr_t = to_gpu_delta(day, arr_mam, base_);

            bool updated = update_arrival(tmp_, stop_idx, arr_t);
            if(updated){
              mark(station_mark_, stop_idx);
              any_station_marked_ = true;
            }
          }
        }
        //TODO stop_idx stimmt nicht -> sollte t_id/stage_id/upward_id enthalten?
        // dann alle späteren stages updaten
        if(leader != NO_LEADER){
          for(int upward_stage = current_stage+1; upward_stage < active_stage_count; ++upward_stage){
            int upwards_id = (upward_stage<<5) + t_id;
            if(upwards_id < active_stop_count){
              auto const t_offset = static_cast<cuda::std::size_t>(&*it - arrival_times.data());
              auto const arr = *it;
              auto const arr_mam = arr.mam_;
              auto const arr_t = to_gpu_delta(day, arr_mam, base_);

              bool updated = update_arrival(tmp_, stop_idx, arr_t);
              if(updated){
                mark(station_mark_, stop_idx);
                any_station_marked_ = true;
              }
            }
          }
          active_stop_count = leader;
          active_stage_count = (active_stop_count + (32 - 1)) >> 5;
          leader = NO_LEADER;
        }
      }

    }
  }
  return any_station_marked_; //TODO sollten wir any_station_marked_ lieber über parameter zurückgeben lassen?
}

template <gpu_direction SearchDir, bool Rt, bool WithClaszFilter>
__device__ bool loop_routes(unsigned const k, bool any_station_marked_, uint32_t* route_mark_,
                            gpu_clasz_mask_t const* allowed_claszes_,
                            gpu_raptor_stats* stats_,
                            short const* kMaxTravelTimeTicks_, uint32_t* prev_station_mark_,
                            gpu_delta_t* best_,
                            gpu_delta_t* round_times_, uint32_t row_count_round_times_,
                            gpu_delta_t* tmp_,
                            uint16_t* lb_, int n_days_,
                            gpu_delta_t* time_at_dest_,
                            uint32_t* station_mark_, gpu_day_idx_t* base_,
                            unsigned short kUnreachable,
                            gpu_delta* route_stop_times,
                            gpu_vecvec<gpu_route_idx_t,gpu_value_type>* route_location_seq,
                            gpu_vecvec<gpu_location_idx_t , gpu_route_idx_t>* location_routes,
                            std::uint32_t const n_locations,
                            std::uint32_t const n_routes,
                            gpu_vector_map<gpu_route_idx_t,gpu_interval<std::uint32_t>>* route_stop_time_ranges,
                            gpu_vector_map<gpu_route_idx_t,gpu_interval<gpu_transport_idx_t >>* route_transport_ranges,
                            gpu_vector_map<gpu_bitfield_idx_t, gpu_bitfield>* bitfields,
                            gpu_vector_map<gpu_transport_idx_t,gpu_bitfield_idx_t>* transport_traffic_days,
                            gpu_interval<gpu_sys_days>* date_range,
                            gpu_locations* locations,
                            gpu_vector_map<gpu_route_idx_t, gpu_clasz>* route_clasz){
  auto const global_t_id = get_global_thread_id();
  auto const global_stride = get_global_stride();

  if(get_global_thread_id()==0){
    any_station_marked_ = false;
  }
  //Hier gehen wir durch alle Routen wie in update_routes_dev von Julian
  for(auto r_idx = global_t_id;
       r_idx <= gtt_->n_routes_; r_idx += global_stride){
    auto const r = gpu_route_idx_t{r_idx};
    if(marked(route_mark_, r_idx)){
      if constexpr (WithClaszFilter){
        auto const as_mask = static_cast<gpu_clasz_mask_t>(1U << static_cast<std::underlying_type_t<gpu_clasz>>(static_cast<gpu_clasz>((*gtt_->route_clasz_)[r])));
        if(!((*allowed_claszes_ & as_mask)==as_mask)){
          continue;
        }
      }
      ++stats_[global_t_id>>5].n_routes_visited_; // wir haben 32 Stellen und es schreiben nur 32 threads gleichzeitig
      // hier in smaller32 und bigger32 aufteilen?
      // TODO hier in smaller32 und bigger32 aufteilen? → aber hier geht nur ein thread rein...
      // also sollte vielleicht diese Schleife mit allen auf einmal durchgangen werden???
      // parameter stimmen noch nicht
      printf("gpu_raptor_kernel: bevor update_route");
      if((*gtt_).route_location_seq_[r_idx].size() <= 32){ // die Route hat <= 32 stops
        any_station_marked_ = update_route_smaller32<SearchDir, Rt>(k, r, stats_, prev_station_mark_, best_,
                                                      round_times_, row_count_round_times_, tmp_, kMaxTravelTimeTicks_,
                                                      lb_, n_days_, time_at_dest_, station_mark_, base_, kUnreachable, any_station_marked_,
                                                                    route_stop_times,
                                                                    route_location_seq,
                                                                    location_routes,
                                                                    n_locations,
                                                                    n_routes,
                                                                    route_stop_time_ranges,
                                                                    route_transport_ranges,
                                                                    bitfields,
                                                                    transport_traffic_days,
                                                                    date_range,
                                                                    locations,
                                                                    route_clasz);
      }
      else{ // diese Route hat > 32 Stops
        any_station_marked_ = update_route_bigger32<SearchDir, Rt>(k, r, stats_, prev_station_mark_, best_,
        /*any_station_marked_ = update_route_bigger32<SearchDir, Rt>(k, r, gtt_, stats_, prev_station_mark_, best_,
        */                                             round_times_, row_count_round_times_, tmp_, kMaxTravelTimeTicks_,
                                                     lb_, n_days_, time_at_dest_, station_mark_, base_, kUnreachable, any_station_marked_,
                                                                   route_stop_times,
                                                                   route_location_seq,
                                                                   location_routes,
                                                                   n_locations,
                                                                   n_routes,
                                                                   route_stop_time_ranges,
                                                                   route_transport_ranges,
                                                                   bitfields,
                                                                   transport_traffic_days,
                                                                   date_range,
                                                                   locations,
                                                                   route_clasz);
      }

    }
  }
  return any_station_marked_;
}

template <gpu_direction SearchDir, bool Rt>
__device__ void update_transfers(unsigned const k, gpu_timetable* gtt_,
                                 bool const * is_dest_, uint16_t* dist_to_end_, uint32_t dist_to_end_size_, gpu_delta_t* tmp_,
                                 gpu_delta_t* best_, gpu_delta_t* time_at_dest_, unsigned short kUnreachable,
                                 uint16_t* lb_, gpu_delta_t* round_times_, uint32_t row_count_round_times_,
                                 uint32_t* station_mark_, uint32_t* prev_station_mark_,
                                 gpu_raptor_stats* stats_){
  auto const global_t_id = get_global_thread_id();
  auto const global_stride = get_global_stride();
  for(auto l_idx = global_t_id;
       l_idx <= gtt_->n_locations_; l_idx += global_stride){
    if(!marked(prev_station_mark_, l_idx)){
      continue;
    }
    auto const is_dest = is_dest_[l_idx];
    auto const transfer_time = (dist_to_end_size_==0 && is_dest)
        ? 0 : dir<SearchDir>((*gtt_->locations_->transfer_time_)[gpu_location_idx_t{l_idx}]).count();
    const auto fp_target_time =
        static_cast<gpu_delta_t>(tmp_[l_idx] + transfer_time);
    if(is_better<SearchDir>(fp_target_time, best_[l_idx])
        && is_better<SearchDir>(fp_target_time, time_at_dest_[k])){
      if(lb_[l_idx] == kUnreachable
          || !is_better<SearchDir>(fp_target_time + dir<SearchDir>(lb_[l_idx]), time_at_dest_[k])){
        ++stats_[l_idx>>5].fp_update_prevented_by_lower_bound_;
        continue;
      }
      ++stats_[l_idx>>5].n_earliest_arrival_updated_by_footpath_;
      bool updated = update_arrival(round_times_, l_idx, fp_target_time);
      best_[l_idx] = fp_target_time;
      if(updated){
        mark(station_mark_, l_idx);
      }
      if(is_dest){
        update_time_at_dest<SearchDir, Rt>(k, fp_target_time, time_at_dest_);
      }
    }
  }
}

template <gpu_direction SearchDir, bool Rt>
__device__ void update_footpaths(unsigned const k, gpu_profile_idx_t const prf_idx, unsigned short kUnreachable,
                                 uint32_t* prev_station_mark_,
                                 gpu_timetable* gtt_, gpu_delta_t* tmp_, gpu_delta_t* best_,
                                 uint16_t const* lb_, uint32_t* station_mark_, gpu_delta_t* time_at_dest_,
                                 bool const* is_dest_, gpu_delta_t* round_times_,
                                 uint32_t row_count_round_times_,
                                 gpu_raptor_stats* stats_){
  auto const global_t_id = get_global_thread_id();
  auto const global_stride = get_global_stride();
  for(auto idx = global_t_id;
       idx <= gtt_->n_locations_; idx += global_stride){
    if(!marked(prev_station_mark_, idx)){
      continue;
    }
    auto const l_idx = gpu_location_idx_t{idx};
    auto const& fps = (SearchDir == gpu_direction::kForward)
         ? gtt_->locations_->gpu_footpaths_out_[prf_idx][l_idx]
           : gtt_->locations_->gpu_footpaths_in_[prf_idx][l_idx];
    for(auto const& fp: fps){
      ++stats_[idx>>5].n_footpaths_visited_;
      auto const target = gpu_to_idx(gpu_location_idx_t{fp.target_});
      auto const fp_target_time =
          gpu_clamp(tmp_[idx] + dir<SearchDir>(fp.duration()).count());

      if(is_better<SearchDir>(fp_target_time, best_[target])
          && is_better<SearchDir>(fp_target_time, time_at_dest_[k])){
        auto const lower_bound = lb_[gpu_to_idx(gpu_location_idx_t{fp.target_})];
        if(lower_bound == kUnreachable
            || !is_better<SearchDir>(fp_target_time + dir<SearchDir>(lower_bound), time_at_dest_[k])){
          ++stats_[idx>>5].fp_update_prevented_by_lower_bound_;
          continue;
        }
      }
      ++stats_[idx>>5].n_earliest_arrival_updated_by_footpath_;
      bool updated = update_arrival(round_times_, gpu_to_idx(gpu_location_idx_t{fp.target_}), fp_target_time);
      best_[gpu_to_idx(gpu_location_idx_t{fp.target_})] = fp_target_time;
      if(updated){
        mark(station_mark_, gpu_to_idx(gpu_location_idx_t{fp.target_}));
      }
      if(is_dest_[gpu_to_idx(gpu_location_idx_t{fp.target_})]){
        update_time_at_dest<SearchDir, Rt>(k, fp_target_time, time_at_dest_);
      }
    }
  }

}

template <gpu_direction SearchDir, bool Rt>
__device__ void update_intermodal_footpaths(unsigned const k, gpu_timetable* gtt_,
                                            uint16_t* dist_to_end_, uint32_t dist_to_end_size_, uint32_t* station_mark_,
                                            uint32_t* prev_station_mark_, gpu_delta_t* time_at_dest_,
                                            unsigned short kUnreachable, gpu_location_idx_t* gpu_kIntermodalTarget,
                                            gpu_delta_t* best_, gpu_delta_t* tmp_,
                                            gpu_delta_t* round_times_, uint32_t row_count_round_times_){
  if(get_global_thread_id()==0 && dist_to_end_size_==0){
    return;
  }
  auto const global_t_id = get_global_thread_id();
  auto const global_stride = get_global_stride();
  for(auto idx = global_t_id;
       idx <= gtt_->n_locations_; idx += global_stride){
    if((marked(prev_station_mark_, idx) || marked(station_mark_, idx)) && dist_to_end_[idx] != kUnreachable){
      auto const end_time = gpu_clamp(get_best<SearchDir>(best_[idx], tmp_[idx]) + dir<SearchDir>(dist_to_end_[idx]));
      if(is_better<SearchDir>(end_time, gpu_kIntermodalTarget[(*best_)].v_)){
        bool updated = update_arrival(round_times_, gpu_kIntermodalTarget->v_, end_time);
        gpu_kIntermodalTarget[(*best_)].v_ = end_time;
        update_time_at_dest<SearchDir, Rt>(k, end_time, time_at_dest_);
      }
    }
  }
}


template <gpu_direction SearchDir, bool Rt>
__device__ void raptor_round(unsigned const k, gpu_profile_idx_t const prf_idx,
                             gpu_day_idx_t* base_,
                             gpu_clasz_mask_t allowed_claszes_, uint16_t* dist_to_end_,
                             uint32_t dist_to_end_size_,
                             bool* is_dest_, uint16_t* lb_, int n_days_,
                             gpu_delta_t* time_at_dest_,
                             bool* any_station_marked_, uint32_t* route_mark_,
                             uint32_t* station_mark_, gpu_delta_t* best_,
                             unsigned short kUnreachable, uint32_t* prev_station_mark_,
                             gpu_delta_t* round_times_, gpu_delta_t* tmp_,
                             uint32_t row_count_round_times_,
                             uint32_t column_count_round_times_,
                             gpu_location_idx_t* gpu_kIntermodalTarget,
                             gpu_raptor_stats* stats_, short* kMaxTravelTimeTicks_,
                             gpu_delta* route_stop_times,
                             gpu_vecvec<gpu_route_idx_t,gpu_value_type>* route_location_seq,
                             gpu_vecvec<gpu_location_idx_t , gpu_route_idx_t>* location_routes,
                             std::uint32_t const n_locations,
                             std::uint32_t const n_routes,
                             gpu_vector_map<gpu_route_idx_t,gpu_interval<std::uint32_t>>* route_stop_time_ranges,
                             gpu_vector_map<gpu_route_idx_t,gpu_interval<gpu_transport_idx_t >>* route_transport_ranges,
                             gpu_vector_map<gpu_bitfield_idx_t, gpu_bitfield>* bitfields,
                             gpu_vector_map<gpu_transport_idx_t,gpu_bitfield_idx_t>* transport_traffic_days,
                             gpu_interval<gpu_sys_days>* date_range,
                             gpu_locations* locations,
                             gpu_vector_map<gpu_route_idx_t, gpu_clasz>* route_clasz){

  // update_time_at_dest für alle locations
  printf("raptor_round");
  auto const global_t_id = get_global_thread_id();
  auto const global_stride = get_global_stride();
  //TODO sicher, dass man über n_locations iterieren muss? -> aufpassen, dass round_times nicht out of range zugegriffen wird
  for(auto idx = global_t_id; idx < gtt_->n_locations_; idx += global_stride){
    best_[global_t_id] = get_best<SearchDir>
        (round_times_[k*row_count_round_times_+idx], best_[idx]);
    if(is_dest_[idx]){
      update_time_at_dest<SearchDir, Rt>(k, best_[global_t_id], time_at_dest_);
    }
  }
  this_grid().sync();

  // für jede location & für jede location_route state_.route_mark_
  if(get_global_thread_id()==0){
    *any_station_marked_ = false;
  }
  convert_station_to_route_marks<SearchDir, Rt>(station_mark_, route_mark_,
                                                  any_station_marked_, gtt_);
  this_grid().sync();

  if(get_global_thread_id()==0){
    if(!*any_station_marked_){
      return;
    }
    // swap
    cuda::std::swap(prev_station_mark_,station_mark_);
    // fill
    for(int j = 0; j < gtt_->n_locations_; j++){
      station_mark_[j] = 0xFFFF;
    }
  }
  this_grid().sync();
  // loop_routes mit true oder false
  // any_station_marked soll nur einmal gesetzt werden, aber loop_routes soll mit allen threads durchlaufen werden?

  printf("gpu_raptor_kernel loop_routes");
  *any_station_marked_ = (allowed_claszes_ == 0xffff)
                         ? loop_routes<SearchDir, Rt, false>(k, any_station_marked_, route_mark_, &allowed_claszes_,
                                                             stats_, kMaxTravelTimeTicks_, prev_station_mark_, best_,
                                                             round_times_, row_count_round_times_, tmp_, lb_, n_days_,
                                                             time_at_dest_, station_mark_, base_, kUnreachable,
                                                                 route_stop_times,
                                                                 route_location_seq,
                                                                 location_routes,
                                                                 n_locations,
                                                                 n_routes,
                                                                 route_stop_time_ranges,
                                                                 route_transport_ranges,
                                                                 bitfields,
                                                                 transport_traffic_days,
                                                                 date_range,
                                                                 locations,
                                                                 route_clasz)
                           : loop_routes<SearchDir, Rt, true>(k, any_station_marked_, route_mark_, &allowed_claszes_,
                                                             stats_, kMaxTravelTimeTicks_, prev_station_mark_, best_,
                                                             round_times_, row_count_round_times_, tmp_, lb_, n_days_,
                                                             time_at_dest_, station_mark_, base_, kUnreachable,
                                                                route_stop_times,
                                                                route_location_seq,
                                                                location_routes,
                                                                n_locations,
                                                                n_routes,
                                                                route_stop_time_ranges,
                                                                route_transport_ranges,
                                                                bitfields,
                                                                transport_traffic_days,
                                                                date_range,
                                                                locations,
                                                                route_clasz);

  this_grid().sync();
  if(get_global_thread_id()==0){
    if(!*any_station_marked_){
      return;
    }
    // fill
    for(int i = 0; i < gtt_->n_routes_; i++){
      route_mark_[i] = 0xFFFF;
    }
    // swap
    cuda::std::swap(prev_station_mark_,station_mark_);
    // fill
    for(int j = 0; j < gtt_->n_locations_; j++){
      station_mark_[j] = 0xFFFF; // soll es auf false setzen
    }
  }
  this_grid().sync();
  // update_transfers
  update_transfers<SearchDir, Rt>(k, gtt_, is_dest_, dist_to_end_, dist_to_end_size_,
                   tmp_, best_, time_at_dest_, kUnreachable, lb_, round_times_,
                   row_count_round_times_, station_mark_, prev_station_mark_, stats_);
  this_grid().sync();

  // update_footpaths
  update_footpaths<SearchDir, Rt>(k, prf_idx, kUnreachable, prev_station_mark_,
                   gtt_, tmp_, best_, lb_, station_mark_, time_at_dest_,
                   is_dest_, round_times_, row_count_round_times_, stats_);
  this_grid().sync();

  // update_intermodal_footpaths
  update_intermodal_footpaths<SearchDir, Rt>(k, gtt_, dist_to_end_, dist_to_end_size_, station_mark_,
                             prev_station_mark_, time_at_dest_, kUnreachable,
                             gpu_kIntermodalTarget, best_, tmp_, round_times_, row_count_round_times_);

}

template <gpu_direction SearchDir, bool Rt>
__device__ void init_arrivals(gpu_unixtime_t const worst_time_at_dest,
                              gpu_day_idx_t* base_, gpu_delta_t* time_at_dest,
                              gpu_delta* route_stop_times,
                              gpu_vector_map<gpu_route_idx_t,gpu_interval<gpu_transport_idx_t >>* route_transport_ranges,
                              gpu_interval<gpu_sys_days>* date_range){
  auto const t_id = get_global_thread_id();
  if(t_id < gpu_kMaxTransfers+1){
    printf("test haengen start -1");
    auto test0 = base(base_,date_range);
    printf("test haengen start");
    auto test1 = unix_to_gpu_delta(test0, worst_time_at_dest);
    printf("test haengen mid");
    auto test2 = time_at_dest[t_id];
    printf("test haengen mid2");
    time_at_dest[t_id] = get_best<SearchDir>(test1, test2);
    printf("test haengen end");
  }

}

// größten Teil von raptor.execute() wird hierdrin ausgeführt
// kernel muss sich außerhalb der gpu_raptor Klasse befinden
template <gpu_direction SearchDir, bool Rt>
__global__ void gpu_raptor_kernel(gpu_unixtime_t* start_time,
                                  uint8_t max_transfers,
                                  gpu_unixtime_t* worst_time_at_dest,
                                  gpu_profile_idx_t* prf_idx,
                                  gpu_raptor<SearchDir,Rt> gr,
                                  gpu_delta_t* tmp,
                                  gpu_delta_t* best,
                                  gpu_delta_t* round_times,
                                  gpu_delta_t* time_at_dest,
                                  uint32_t* station_mark,
                                  uint32_t* prev_station_mark,
                                  uint32_t* route_mark,
                                  bool* any_station_marked,
                                  uint32_t row_count_round_times,
                                  uint32_t column_count_round_times,
                                  gpu_raptor_stats* stats,
                                  gpu_delta* route_stop_times,
                                  gpu_vecvec<gpu_route_idx_t,gpu_value_type>* route_location_seq,
                                  gpu_vecvec<gpu_location_idx_t , gpu_route_idx_t>* location_routes,
                                  std::uint32_t const n_locations,
                                  std::uint32_t const n_routes,
                                  gpu_vector_map<gpu_route_idx_t,gpu_interval<std::uint32_t>>* route_stop_time_ranges,
                                  gpu_vector_map<gpu_route_idx_t,gpu_interval<gpu_transport_idx_t >>* route_transport_ranges,
                                  gpu_vector_map<gpu_bitfield_idx_t, gpu_bitfield>* bitfields,
                                  gpu_vector_map<gpu_transport_idx_t,gpu_bitfield_idx_t>* transport_traffic_days,
                                  gpu_interval<gpu_sys_days>* date_range,
                                  gpu_locations* locations,
                                  gpu_vector_map<gpu_route_idx_t, gpu_clasz>* route_clasz){
  auto const end_k =
      get_smaller(max_transfers, gpu_kMaxTransfers) + 1U;
  // 1. Initialisierung

  init_arrivals<SearchDir, Rt>(*worst_time_at_dest, gr.base_,
                time_at_dest, route_stop_times,route_transport_ranges,date_range);

  this_grid().sync();
  printf("gpu_raptor_kernel3");
  // 2. Update Routes

  for (auto k = 1U; k != end_k; ++k) { // diese Schleife bleibt, da alle Threads in jede Runde gehen

    // Resultate aus lezter Runde von device in variable speichern?  //TODO: typen von kIntermodalTarget und dist_to_end_size falsch???
    raptor_round<SearchDir, Rt>(k, *prf_idx, gr.base_, *gr.allowed_claszes_,
                 gr.dist_to_end_, *gr.dist_to_end_size_, gr.is_dest_, gr.lb_, *gr.n_days_,
                                time_at_dest,
                                any_station_marked, route_mark,
                                station_mark, best,
                 gr.kUnreachable, prev_station_mark,
                                round_times, tmp,
                                row_count_round_times,
                                column_count_round_times,
                 gr.kIntermodalTarget_, stats, gr.kMaxTravelTimeTicks_,route_stop_times,
                                route_location_seq,
                                location_routes,
                                n_locations,
                                n_routes,
                                route_stop_time_ranges,
                                route_transport_ranges,
                                bitfields,
                                transport_traffic_days,
                                date_range,
                                locations,
                                route_clasz);
    this_grid().sync();
  }
  this_grid().sync();

  //construct journey

  this_grid().sync();

}

#define XSTR(s) STR(s)
#define STR(s) #s

#define CUDA_CALL(call) \
    if ((code = (call)) != cudaSuccess) {                     \
      printf("CUDA error: %s at " XSTR(call) " %s:%d\n",     \
             cudaGetErrorString(code), __FILE__, __LINE__); \
      goto fail;                                            \
    }

#define CUDA_COPY_TO_DEVICE(type, target, source, size)                        \
    CUDA_CALL(cudaMalloc(&(target), (size) * sizeof(type)))                          \
    CUDA_CALL(                                                                   \
        cudaMemcpy(target, source, (size) * sizeof(type), cudaMemcpyHostToDevice))

void copy_to_devices(gpu_clasz_mask_t const& allowed_claszes,
                     std::vector<std::uint16_t> const& dist_to_dest,
                     gpu_day_idx_t const& base,
                     std::unique_ptr<bool[]> const& is_dest,
                     std::size_t is_dest_size,
                     std::vector<std::uint16_t> const& lb,
                     int const& n_days,
                     std::uint16_t const& kUnreachable,
                     short const& kMaxTravelTimeTicks,
                     unsigned int const& kIntermodalTarget,
                     gpu_clasz_mask_t*& allowed_claszes_,
                     std::uint16_t* & dist_to_end_,
                     std::uint32_t* & dist_to_end_size_,
                     gpu_day_idx_t* & base_,
                     bool* & is_dest_,
                     std::uint16_t* & lb_,
                     int* & n_days_,
                     std::uint16_t* & kUnreachable_,
                     gpu_location_idx_t* & kIntermodalTarget_,
                     short* & kMaxTravelTimeTicks_){
  cudaError_t code;
  std::cerr << "copy_to_device start" << std::endl;
  auto dist_to_end_size = dist_to_dest.size();
  allowed_claszes_ = nullptr;
  CUDA_COPY_TO_DEVICE(gpu_clasz_mask_t, allowed_claszes_, &allowed_claszes, 1);
  dist_to_end_ = nullptr;
  CUDA_COPY_TO_DEVICE(std::uint16_t, dist_to_end_, dist_to_dest.data(),
                      dist_to_dest.size());
  dist_to_end_size_ = nullptr;
  CUDA_COPY_TO_DEVICE(std::uint32_t, dist_to_end_size_, &dist_to_end_size_, 1);
  base_ = nullptr;
  CUDA_COPY_TO_DEVICE(gpu_day_idx_t, base_, &base, 1);
  is_dest_ = nullptr;
  CUDA_COPY_TO_DEVICE(bool, is_dest_, is_dest.get(), is_dest_size);
  lb_ = nullptr;
  CUDA_COPY_TO_DEVICE(std::uint16_t, lb_, lb.data(), lb.size());
  n_days_ = nullptr;
  CUDA_COPY_TO_DEVICE(int, n_days_, &n_days, 1);
  kUnreachable_ = nullptr;
  CUDA_COPY_TO_DEVICE(std::uint16_t, kUnreachable_, &kUnreachable, 1);
  kIntermodalTarget_ = nullptr;
  CUDA_COPY_TO_DEVICE(gpu_location_idx_t, kIntermodalTarget_,
                      &kIntermodalTarget, 1);
  kMaxTravelTimeTicks_ = nullptr;
  CUDA_COPY_TO_DEVICE(short, kMaxTravelTimeTicks_, &kMaxTravelTimeTicks, 1);
  std::cerr << "copy_to_device end" << std::endl;
  return;
fail:
  cudaFree(allowed_claszes_);
  cudaFree(dist_to_end_);
  cudaFree(dist_to_end_size_);
  cudaFree(base_);
  cudaFree(is_dest_);
  cudaFree(lb_);
  cudaFree(n_days_);
  cudaFree(kUnreachable_);
  cudaFree(kIntermodalTarget_);
  cudaFree(kMaxTravelTimeTicks_);
  return;
};
void copy_to_device_destroy(
    gpu_clasz_mask_t*& allowed_claszes_,
    std::uint16_t* & dist_to_end_,
    std::uint32_t* & dist_to_end_size_,
    gpu_day_idx_t* & base_,
    bool* & is_dest_,
    std::uint16_t* & lb_,
    int* & n_days_,
    std::uint16_t* & kUnreachable_,
    gpu_location_idx_t* & kIntermodalTarget_,
    short* & kMaxTravelTimeTicks_){
  cudaFree(allowed_claszes_);
  cudaFree(dist_to_end_);
  cudaFree(dist_to_end_size_);
  cudaFree(base_);
  cudaFree(is_dest_);
  cudaFree(lb_);
  cudaFree(n_days_);
  cudaFree(kUnreachable_);
  cudaFree(kIntermodalTarget_);
  cudaFree(kMaxTravelTimeTicks_);
  cudaDeviceSynchronize();
  auto const last_error = cudaGetLastError();
  if (last_error != cudaSuccess) {
    printf("CUDA error: %s at " STR(last_error) " %s:%d\n",
           cudaGetErrorString(last_error), __FILE__, __LINE__);
  }
};

void launch_kernel(void** args,
                          device_context const& device,
                          cudaStream_t s,
                          gpu_direction search_dir,
                          bool rt) {
  std::cerr << "Test gpu_raptor::launch_kernel() start" << std::endl;
  cudaSetDevice(device.id_);
  // Kernel-Auswahl basierend auf Parametern
  void* kernel_func = nullptr;
  if (search_dir == gpu_direction::kForward && rt == true) {
    kernel_func = (void*)gpu_raptor_kernel<gpu_direction::kForward, true>;
  } else if (search_dir == gpu_direction::kForward && rt == false) {
    kernel_func = (void*)gpu_raptor_kernel<gpu_direction::kForward, false>;
  } else if (search_dir == gpu_direction::kBackward && rt == true) {
    kernel_func = (void*)gpu_raptor_kernel<gpu_direction::kBackward, true>;
  } else if (search_dir == gpu_direction::kBackward && rt == false) {
    kernel_func = (void*)gpu_raptor_kernel<gpu_direction::kBackward, false>;
  }

  std::cerr << "Test gpu_raptor::launch_kernel() kernel_start" << std::endl;
  cudaLaunchCooperativeKernel(kernel_func, device.grid_, device.threads_per_block_, args, 0, s);
  cudaDeviceSynchronize();
  cuda_check();
  std::cerr << "Test gpu_raptor::launch_kernel() ende" << std::endl;

}

inline void fetch_arrivals_async(mem*& mem, cudaStream_t s) {
  cudaMemcpyAsync(
      mem->host_.round_times_.data(), mem->device_.round_times_,
      sizeof(gpu_delta_t)*mem->host_.row_count_round_times_*mem->host_.column_count_round_times_, cudaMemcpyDeviceToHost, s);
  cuda_check();
  cudaMemcpyAsync(
      mem->host_.stats_.data(), mem->device_.stats_,
      sizeof(gpu_raptor_stats)*32, cudaMemcpyDeviceToHost, s);
  cuda_check();
  cudaMemcpyAsync(
      mem->host_.tmp_.data(), mem->device_.tmp_,
      sizeof(gpu_delta_t)*mem->device_.n_locations_, cudaMemcpyDeviceToHost, s);
  cuda_check();
  cudaMemcpyAsync(
      mem->host_.best_.data(), mem->device_.best_,
      sizeof(gpu_delta_t)*mem->device_.n_locations_, cudaMemcpyDeviceToHost, s);
  cuda_check();
  cudaMemcpyAsync(
      mem->host_.station_mark_.data(), mem->device_.station_mark_,
      sizeof(uint32_t)*mem->device_.n_locations_, cudaMemcpyDeviceToHost, s);
  cuda_check();
  cudaMemcpyAsync(
      mem->host_.prev_station_mark_.data(), mem->device_.prev_station_mark_,
      sizeof(uint32_t)*mem->device_.n_locations_, cudaMemcpyDeviceToHost, s);
  cuda_check();
  cudaMemcpyAsync(
      mem->host_.route_mark_.data(), mem->device_.route_mark_,
      sizeof(uint32_t)*mem->device_.n_routes_, cudaMemcpyDeviceToHost, s);
  cuda_check();
}
void copy_back(mem*& mem){
  cuda_check();
  cuda_sync_stream(mem->context_.proc_stream_);
  cuda_check();
  fetch_arrivals_async(mem,mem->context_.transfer_stream_);
  cuda_check();
  cuda_sync_stream(mem->context_.transfer_stream_);
  cuda_check();
}

void add_start_gpu(gpu_location_idx_t const l, gpu_unixtime_t const t,mem* mem_,gpu_timetable* gtt_,gpu_day_idx_t base_,short const kInvalid){
  std::cerr << "Test gpu_raptor::add_start_gpu() start" << std::endl;
  trace_upd("adding start {}: {}\n", location{gtt_, l}, t);
  std::vector<gpu_delta_t> best_new(mem_->device_.n_locations_,kInvalid);
  std::cerr << "Test gpu_raptor::add_start_gpu() 1" << std::endl;
  std::vector<gpu_delta_t> round_times_new((mem_->device_.column_count_round_times_*mem_->device_.row_count_round_times_),kInvalid);
  std::cerr << "Test gpu_raptor::add_start_gpu() 1.5" << std::endl;
  best_new[gpu_to_idx(l)] = unix_to_gpu_delta(cpu_base(gtt_,base_), t);
  std::cerr << "Test gpu_raptor::add_start_gpu() 2" << std::endl;
  //TODO: hier fehler da base nur auf device funktioniert!
  round_times_new[0U*mem_->device_.row_count_round_times_+ gpu_to_idx(l)] = unix_to_gpu_delta(cpu_base(gtt_,base_), t);
  //TODO: fix station_mark ist kein bool!
  std::vector<uint32_t> gpu_station_mark(mem_->device_.n_locations_,0);
  gpu_station_mark[gpu_to_idx(l)] = 1;
  std::cerr << "Test gpu_raptor::add_start_gpu() 3" << std::endl;
  cudaMemcpy(mem_->device_.best_, best_new.data(), mem_->device_.n_locations_*sizeof(gpu_delta_t), cudaMemcpyHostToDevice);
  cudaMemcpy(mem_->device_.round_times_, round_times_new.data(), round_times_new.size()*sizeof(gpu_delta_t), cudaMemcpyHostToDevice);
  cudaMemcpy(mem_->device_.station_mark_, gpu_station_mark.data(), mem_->device_.n_locations_*sizeof(uint32_t), cudaMemcpyHostToDevice);
  std::cerr << "Test gpu_raptor::add_start_gpu() end" << std::endl;
}
std::unique_ptr<mem> gpu_mem(
    std::vector<gpu_delta_t>& tmp,
    std::vector<gpu_delta_t>& best,
    std::vector<bool>& station_mark,
    std::vector<bool>& prev_station_mark,
    std::vector<bool>& route_mark,
    gpu_direction search_dir,
    gpu_timetable* gtt){
  std::cerr << "Test gpu_raptor::gpu_mem()" << std::endl;
  short kInvalid = 0;
  if(search_dir == gpu_direction::kForward){
    kInvalid = kInvalidGpuDelta<gpu_direction::kForward>;
  } else{
    kInvalid = kInvalidGpuDelta<gpu_direction::kBackward>;
  }
  std::vector<uint32_t> gpu_station_mark(gtt->n_locations_);
  for (size_t i = 0; i < station_mark.size(); ++i) {
    gpu_station_mark[i] = station_mark[i];
  }
  std::vector<uint32_t> gpu_prev_station_mark(gtt->n_locations_);
  for (size_t i = 0; i < prev_station_mark.size(); ++i) {
    gpu_prev_station_mark[i] = prev_station_mark[i];
  }
  std::vector<uint32_t> gpu_route_mark(gtt->n_locations_);
  for (size_t i = 0; i < route_mark.size(); ++i) {
    gpu_route_mark[i] = route_mark[i];
  }
  gpu_raptor_state state;
  std::cerr << "Test gpu_raptor::gpu_mem() 1" << std::endl;
  state.init(*gtt, kInvalid);
  loaned_mem loan(state,kInvalid);
  std::unique_ptr<mem> mem = std::move(loan.mem_);

  cudaMemcpy(mem.get()->device_.tmp_, tmp.data(), (gtt->n_locations_) * sizeof(gpu_delta_t), cudaMemcpyHostToDevice);
  cuda_check();
  cudaMemcpy(mem.get()->device_.best_, best.data(), (gtt->n_locations_) * sizeof(gpu_delta_t), cudaMemcpyHostToDevice);
  cuda_check();
  cudaMemcpy(mem.get()->device_.station_mark_, gpu_station_mark.data(), (gtt->n_locations_) * sizeof(uint32_t), cudaMemcpyHostToDevice);
  cuda_check();
  cudaMemcpy(mem.get()->device_.prev_station_mark_, gpu_prev_station_mark.data(), (gtt->n_locations_) * sizeof(uint32_t), cudaMemcpyHostToDevice);
  cuda_check();
  cudaMemcpy(mem.get()->device_.route_mark_, gpu_route_mark.data(), (gtt->n_routes_) * sizeof(uint32_t), cudaMemcpyHostToDevice);
  cuda_check();
  cudaDeviceSynchronize();
  std::cerr << "Test gpu_raptor::gpu_mem() ende" << std::endl;
  return mem;
}

void copy_to_gpu_args(gpu_unixtime_t const* start_time,
                      gpu_unixtime_t const* worst_time_at_dest,
                      gpu_profile_idx_t const* prf_idx,
                      gpu_unixtime_t*& start_time_ptr,
                      gpu_unixtime_t*& worst_time_at_dest_ptr,
                      gpu_profile_idx_t*& prf_idx_ptr){
  cudaError_t code;
  CUDA_COPY_TO_DEVICE(gpu_unixtime_t,start_time_ptr,start_time,1);
  CUDA_COPY_TO_DEVICE(gpu_unixtime_t,worst_time_at_dest_ptr,worst_time_at_dest,1);
  CUDA_COPY_TO_DEVICE(gpu_profile_idx_t ,prf_idx_ptr,prf_idx,1);
  return;
  fail:
    cudaFree(start_time_ptr);
    cudaFree(worst_time_at_dest_ptr);
    cudaFree(prf_idx_ptr);
    return;
}
void destroy_copy_to_gpu_args(gpu_unixtime_t*& start_time_ptr,
                              gpu_unixtime_t*& worst_time_at_dest_ptr,
                              gpu_profile_idx_t*& prf_idx_ptr){
  cudaFree(start_time_ptr);
  cudaFree(worst_time_at_dest_ptr);
  cudaFree(prf_idx_ptr);
  cuda_check();
}