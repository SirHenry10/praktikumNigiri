
#include "nigiri/routing/gpu_raptor.h"

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
       idx < *gtt_->n_locations_; idx += global_stride) {
    if (marked(station_marks, idx)) {
      if (!*any_station_marked) {
        *any_station_marked = true;
      }
      auto const location_routes = gtt_->location_routes_;
      for (auto const& r :  (*location_routes)[gpu_location_idx_t{idx}]) {
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
                                    gpu_strong<uint16_t, _day_idx> base_){
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
  const auto traffic_day = (*gtt_->bitfields_)[(*gtt_->transport_traffic_days_)[t]];
  // test(i=day) methode
  if (day >= traffic_day.size()) {
    return false;
  }
  const auto bitfields_data = (*gtt_->bitfields_data_)[(*gtt_->transport_traffic_days_)[t]];
  auto const block = bitfields_data[day / traffic_day.bits_per_block];
  auto const bit = (day % traffic_day.bits_per_block);
  return (block & (std::uint64_t{1U} << bit)) != 0U;
}


template <gpu_direction SearchDir, bool Rt>
__device__ bool update_route_smaller32(unsigned const k, gpu_route_idx_t r,
                                       gpu_timetable* gtt_,
                                       gpu_raptor_stats* stats_,
                                       uint32_t* prev_station_mark_, gpu_delta_t* best_,
                                       gpu_delta_t* round_times_, uint32_t row_count_round_times_,
                                       gpu_delta_t* tmp_, short const* kMaxTravelTimeTicks_,
                                       uint16_t* lb_, int n_days_,
                                       gpu_delta_t* time_at_dest_,
                                       uint32_t* station_mark_, gpu_strong<uint16_t, _day_idx> base_,
                                       unsigned short kUnreachable, bool any_station_marked_){
  auto const t_id = threadIdx.x;
  auto const stop_seq = gtt_->route_location_seq_->operator[](r);
  auto stop_idx = -1;
  gpu_stop stp{};
  unsigned int l_idx;
  bool is_last;
  gpu_delta_t prev_round_time;
  unsigned leader = stop_seq.size();
  unsigned int active_stop_count = stop_seq.size();
  if(t_id == 0){
    any_station_marked_= false;
  }

  if(t_id < active_stop_count) {  // wir gehen durch alle Stops der Route r
    stop_idx = static_cast<gpu_stop_idx_t>(
        (SearchDir == gpu_direction::kForward) ? t_id : stop_seq.size() - t_id - 1U);
    stp = gpu_stop{stop_seq[stop_idx]};
    l_idx = gpu_to_idx(gpu_location_idx_t{stp.location_});
    is_last = t_id == stop_seq.size() - 1U;
    // ist äquivalent zu prev_arrival
    prev_round_time = round_times_[(k - 1) * row_count_round_times_ + l_idx];
  }
  if (!__any_sync(FULL_MASK, prev_round_time!=std::numeric_limits<gpu_delta_t>::max())) {
    return any_station_marked_;
  }

  if(t_id < active_stop_count) {
    //zuerst müssen wir earliest transport für jede station/thread berechnen
    auto const splitter = gpu_split_day_mam(base_, prev_round_time);
    auto const day_at_stop = splitter.first;
    auto const mam = splitter.second;
    auto et = gpu_transport{};

    // Anfang get_earliest_transport Methode
    ++stats_[l_idx>>5].n_earliest_trip_calls_;
    auto const n_days_to_iterate = get_smaller(*kMaxTravelTimeTicks_/1440 +1,
                                               (SearchDir == gpu_direction::kForward) ?
                                                                                      n_days_ - as_int(day_at_stop) : as_int(day_at_stop)+1);
    auto const event_times =
        gtt_->event_times_at_stop(r, stop_idx, (SearchDir == gpu_direction::kForward) ?
                                                                                      gpu_event_type::kDep : gpu_event_type::kArr);
    auto const seek_first_day = [&]() {
      return linear_lb(get_begin_it<decltype(event_times), SearchDir>(event_times),
                       get_end_it<std::span<gpu_delta>, SearchDir>(event_times), mam,
                       [&](gpu_delta const a, gpu_minutes_after_midnight_t const b) {
                         return is_better<SearchDir, Rt>(a.mam_, b.count());
                       });
    };

    for(auto i = gpu_day_idx_t::value_t{0U}; i != n_days_to_iterate; ++i){
      auto const ev_time_range = gpu_it_range{i==0U ? seek_first_day() : get_begin_it<SearchDir>(event_times), get_end_it<SearchDir>(event_times)};
      if(ev_time_range.emmpty()){
        continue;
      }
      auto const day = (SearchDir == gpu_direction::kForward) ? day_at_stop + i : day_at_stop - i;
      for(auto it = begin(ev_time_range); it != end(ev_time_range); ++i){ // hier werden alle trips/transports durchgangen
        auto const t_offset = static_cast<std::size_t>(&*it - event_times.data());
        auto const ev = *it;
        auto const ev_mam = ev.mam();
        auto const t = (*gtt_->route_transport_ranges_)[r][t_offset];
        auto const ev_day_offset = ev.days();
        auto const start_day = static_cast<std::size_t>(as_int(day) - ev_day_offset);
        if(is_better_or_equal<SearchDir>(time_at_dest_[k], unix_to_gpu_delta(day, ev_mam)+dir<SearchDir>(lb_[l_idx]))){
          et = {gpu_transport_idx_t::invalid(), gpu_day_idx_t::invalid()};
        }
        else if(i == 0U && !is_better_or_eq(mam.count(), ev_mam)){
          et = gpu_transport{};
        }
        else if(!is_transport_active<SearchDir, Rt>(t, start_day, gtt_)){
          et= gpu_transport{};
        }
        else {
          et = {t, static_cast<gpu_day_idx_t>(as_int(day) - ev_day_offset)};
        }
        auto const et_time_at_stop = et.is_valid()
                   ? time_at_stop<SearchDir, Rt>(r, et, stop_idx, (SearchDir == gpu_direction::kForward) ?
                     gpu_event_type::kDep : gpu_event_type::kArr) : kInvalidGpuDelta<SearchDir>;
        // hier leader election? -> jeder thread hat zu diesem Punkt sein et von diesem trip/transport bekommen
        // elect leader via two CUDA functions
        // every participating thread (1.parameter) evaluates predicate (2.parameter)
        // & also works as a barrier, d.h. we have to wait for all threads to cast their vote
        // predicate is true if it is possible to enter current trip at station
        unsigned ballot = __ballot_sync(
            FULL_MASK, (t_id < active_stop_count) && prev_round_time!=std::numeric_limits<gpu_delta_t>::max() &&
                        et_time_at_stop!=std::numeric_limits<gpu_delta_t>::max() &&
                        (prev_round_time <= et_time_at_stop));
        leader = __ffs(ballot) - 1; // returns smallest thread, for which predicate is true
        // jeder thread, dessen station nach der von leader liegt,
        // kann jetzt seine jeweilige trip arrival time versuchen zu aktualisieren
        if (t_id > leader && t_id < active_stop_count){
          if (et.is_valid() || marked(prev_station_mark_, l_idx)) {
            auto current_best = kInvalidGpuDelta<SearchDir>;
            if(et.is_valid() && ((SearchDir == gpu_direction::kForward) ? stp.out_allowed_ : stp.in_allowed_)){
              auto const by_transport = time_at_stop<SearchDir, Rt>(
                  r, et, stop_idx, (SearchDir == gpu_direction::kForward) ? gpu_event_type::kArr : gpu_event_type::kDep, gtt_, base_);
              current_best = get_best<SearchDir>(round_times_[(k - 1)*row_count_round_times_ + l_idx],
                                                 tmp_[l_idx], best_[l_idx]);

              if (is_better<SearchDir>(by_transport, current_best) &&
                  is_better<SearchDir>(by_transport, time_at_dest_[k]) &&
                  lb_[l_idx] != kUnreachable &&
                  is_better<SearchDir>(by_transport + dir<SearchDir>(lb_[l_idx]), time_at_dest_[k])){
                ++stats_[get_global_thread_id()>>5].n_earliest_arrival_updated_by_route_;
                tmp_[l_idx] = get_best<SearchDir>(by_transport, tmp_[l_idx]);
                mark(station_mark_, l_idx);
                current_best = by_transport;
                atomicOr(reinterpret_cast<int*>(any_station_marked_), true); // keine Ahnung, ob das hier mit dem cast korrekt funktioniert
              }
            }
            if(!is_last && ((SearchDir == gpu_direction::kForward) ? (stp.in_allowed_!=0U) : (stp.out_allowed_!=0U)) && marked(prev_station_mark_, l_idx)){ //wieder umgekehrte Bedingung
              if(lb_[l_idx] == kUnreachable) {
                return any_station_marked_;
              }
              if(is_better_or_eq<SearchDir, Rt>(prev_round_time, et_time_at_stop)){

              }
            }
          }
        }


        if (leader != NO_LEADER) {
          active_stop_count = leader;
        }
        leader = NO_LEADER;
      }
    }
  }

  return any_station_marked_;
}

template <gpu_direction SearchDir, bool Rt>
__device__ bool update_route_bigger32(unsigned const k, gpu_route_idx_t r,
                                      gpu_timetable* gtt_,
                                      gpu_raptor_stats* stats_,
                                      uint32_t* prev_station_mark_, gpu_delta_t* best_,
                                      gpu_delta_t* round_times_, uint32_t row_count_round_times_,
                                      gpu_delta_t* tmp_, short* kMaxTravelTimeTicks_,
                                      uint16_t* lb_, int n_days_,
                                      gpu_delta_t* time_at_dest_,
                                      uint32_t* station_mark_, gpu_strong<uint16_t, _day_idx> base_,
                                      unsigned short kUnreachable, bool any_station_marked_){
  return false;
}

template <gpu_direction SearchDir, bool Rt, bool WithClaszFilter>
__device__ bool loop_routes(unsigned const k, bool any_station_marked_,
                            gpu_timetable* gtt_, uint32_t* route_mark_,
                            gpu_clasz_mask_t* allowed_claszes_,
                            gpu_raptor_stats* stats_,
                            short* kMaxTravelTimeTicks_, uint32_t* prev_station_mark_,
                            gpu_delta_t* best_,
                            gpu_delta_t* round_times_, uint32_t row_count_round_times_,
                            gpu_delta_t* tmp_,
                            uint16_t* lb_, int n_days_,
                            gpu_delta_t* time_at_dest_,
                            uint32_t* station_mark_, gpu_strong<uint16_t, _day_idx> base_,
                            unsigned short kUnreachable){
  auto const global_t_id = get_global_thread_id();
  auto const global_stride = get_global_stride();

  if(get_global_thread_id()==0){
    any_station_marked_ = false;
  }
  //Hier gehen wir durch alle Routen wie in update_routes_dev von Julian
  for(auto r_idx = global_t_id;
       r_idx <= *gtt_->n_routes_; r_idx += global_stride){
    auto const r = gpu_route_idx_t{r_idx};
    if(marked(route_mark_, r_idx)){
      if constexpr (WithClaszFilter){
        auto const as_mask = static_cast<gpu_clasz_mask_t>(1U << static_cast<std::underlying_type_t<gpu_clasz>>(static_cast<gpu_clasz>((*gtt_->route_clasz_)[r])));
        if(!((*allowed_claszes_ & as_mask)==as_mask)){
          continue;
        }
      }
      ++stats_[global_t_id>>5].n_routes_visited_; // wir haben 32 Stellen und es schreiben nur 32 threads gleichzeitig
      // TODO hier in smaller32 und bigger32 aufteilen? → aber hier geht nur ein thread rein...
      // also sollte vielleicht diese Schleife mit allen auf einmal durchgangen werden???
      // parameter stimmen noch nicht
      if((*gtt_).route_location_seq_[r_idx].size() <= 32){ // die Route hat <= 32 stops
        any_station_marked_ |= update_route_smaller32<SearchDir, Rt>(k, r, gtt_, stats_, prev_station_mark_, best_,
                                                      round_times_, row_count_round_times_, tmp_, kMaxTravelTimeTicks_, lb_, n_days_, time_at_dest_, station_mark_, base_, kUnreachable, any_station_marked_);
      }
      else{ // diese Route hat > 32 Stops
        any_station_marked_ |= update_route_bigger32<SearchDir, Rt>(k, r, gtt_, stats_, prev_station_mark_, best_,
                                                     round_times_, row_count_round_times_, tmp_, kMaxTravelTimeTicks_, lb_, n_days_, time_at_dest_, station_mark_, base_, kUnreachable, any_station_marked_);
      }

    }
  }
  return any_station_marked_;
}

template <gpu_direction SearchDir, bool Rt>
__device__ void update_transfers(unsigned const k, gpu_timetable* gtt_,
                                 bool* is_dest_, uint16_t* dist_to_end_, uint32_t dist_to_end_size_, gpu_delta_t* tmp_,
                                 gpu_delta_t* best_, gpu_delta_t* time_at_dest_, unsigned short kUnreachable,
                                 uint16_t* lb_, gpu_delta_t* round_times_, uint32_t row_count_round_times_,
                                 uint32_t* station_mark_, uint32_t* prev_station_mark_,
                                 gpu_raptor_stats* stats_){
  auto const global_t_id = get_global_thread_id();
  auto const global_stride = get_global_stride();
  for(auto l_idx = global_t_id;
       l_idx <= *gtt_->n_locations_; l_idx += global_stride){
    if(!marked(prev_station_mark_, l_idx)){
      continue;
    }
    auto const is_dest = is_dest_[l_idx];
    auto const transfer_time = (dist_to_end_size_==0 && is_dest)
        ? 0 : dir<SearchDir, Rt>(gtt_->locations_->transfer_time_[gpu_location_idx_t{l_idx}]).count();
    const short fp_target_time =
        static_cast<gpu_delta_t>(tmp_[l_idx] + transfer_time);
    if(is_better<SearchDir, Rt>(fp_target_time, best_[l_idx])
        && is_better<SearchDir, Rt>(fp_target_time, time_at_dest_[k])){
      if(lb_[l_idx] == kUnreachable
          || !is_better(fp_target_time + dir<SearchDir, Rt>(lb_[l_idx]), time_at_dest_[k])){
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
                                 uint16_t* lb_, uint32_t* station_mark_, gpu_delta_t* time_at_dest_,
                                 bool* is_dest_, gpu_delta_t* round_times_,
                                 uint32_t row_count_round_times_,
                                 gpu_raptor_stats* stats_){
  auto const global_t_id = get_global_thread_id();
  auto const global_stride = get_global_stride();
  for(auto idx = global_t_id;
       idx <= *gtt_->n_locations_; idx += global_stride){
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
          gpu_clamp(tmp_[idx] + dir<SearchDir, Rt>(fp.duration()).count());

      if(is_better(fp_target_time, best_[target])
          && is_better(fp_target_time, time_at_dest_[k])){
        auto const lower_bound = lb_[gpu_to_idx(gpu_location_idx_t{fp.target_})];
        if(lower_bound == kUnreachable
            || !is_better(fp_target_time + dir<SearchDir, Rt>(lower_bound), time_at_dest_[k])){
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
        update_time_at_dest(k, fp_target_time, time_at_dest_);
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
       idx <= *gtt_->n_locations_; idx += global_stride){
    if((marked(prev_station_mark_, idx) || marked(station_mark_, idx)) && dist_to_end_[idx] != kUnreachable){
      auto const end_time = clamp(get_best<SearchDir, Rt>(best_[idx], tmp_[idx]) + dir<SearchDir, Rt>(dist_to_end_[idx]));
      if(is_better(end_time, (*best_)[gpu_kIntermodalTarget])){
        bool updated = update_arrival(round_times_, gpu_kIntermodalTarget->v_, end_time);
        (*best_)[gpu_kIntermodalTarget] = end_time;
        update_time_at_dest(k, end_time, time_at_dest_);
      }
    }
  }
}


template <gpu_direction SearchDir, bool Rt>
__device__ void raptor_round(unsigned const k, gpu_profile_idx_t const prf_idx,
                             gpu_timetable* gtt_, gpu_strong<uint16_t, _day_idx> base_,
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
                             gpu_raptor_stats stats_, short* kMaxTravelTimeTicks_){

  // update_time_at_dest für alle locations
  auto const global_t_id = get_global_thread_id();
  auto const global_stride = get_global_stride();
  //TODO sicher, dass man über n_locations iterieren muss? -> aufpassen, dass round_times nicht out of range zugegriffen wird
  for(auto idx = global_t_id; idx < *gtt_->n_locations_; idx += global_stride){
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
    uint32_t const size = *gtt_->n_locations_;
    uint32_t dummy_marks[size];
    for(int i=0; i < *gtt_->n_locations_; i++){
      dummy_marks[i] = station_mark_[i];
      station_mark_[i] = prev_station_mark_[i];
      prev_station_mark_[i] = station_mark_[i];
    }
    // fill
    for(int j = 0; j < *gtt_->n_locations_; j++){
      station_mark_[j] = 0xFFFF;
    }
  }
  this_grid().sync();
  // loop_routes mit true oder false
  // any_station_marked soll nur einmal gesetzt werden, aber loop_routes soll mit allen threads durchlaufen werden?
  *any_station_marked_ = (allowed_claszes_ == 0xffff)
                         ? loop_routes<SearchDir, Rt, false>(k, any_station_marked_, gtt_, route_mark_, &allowed_claszes_,
                                                             &stats_, kMaxTravelTimeTicks_, prev_station_mark_, best_,
                                                             round_times_, row_count_round_times_, tmp_, lb_, n_days_,
                                                             time_at_dest_, station_mark_, base_, kUnreachable)
                           : loop_routes<SearchDir, Rt,true>(k, any_station_marked_, gtt_, route_mark_, allowed_claszes_,
                                                             stats_, kMaxTravelTimeTicks_, prev_station_mark_, best_,
                                                             round_times_, row_count_round_times_, tmp_, lb_, n_days_,
                                                             time_at_dest_, station_mark_, base_, kUnreachable);

  this_grid().sync();
  if(get_global_thread_id()==0){
    if(!*any_station_marked_){
      return;
    }
    // fill
    for(int i = 0; reinterpret_cast<uint32_t*>(i) < gtt_->n_routes_; i++){
      route_mark_[i] = 0xFFFF;
    }
    // swap
    uint32_t const size = *gtt_->n_locations_;
    uint32_t dummy_marks[size];
    for(int i=0; i < *gtt_->n_locations_; i++){
      dummy_marks[i] = station_mark_[i];
      station_mark_[i] = prev_station_mark_[i];
      prev_station_mark_[i] = station_mark_[i];
    }
    // fill
    for(int j = 0; j < *gtt_->n_locations_; j++){
      station_mark_[j] = 0xFFFF; // soll es auf false setzen
    }
  }
  this_grid().sync();
  // update_transfers
  update_transfers<SearchDir, Rt>(k, gtt_, is_dest_, dist_to_end_, dist_to_end_size_,
                   tmp_, best_, time_at_dest_, kUnreachable, lb_, round_times_,
                   row_count_round_times_, station_mark_, prev_station_mark_);
  this_grid().sync();

  // update_footpaths
  update_footpaths<SearchDir, Rt>(k, prf_idx, kUnreachable, prev_station_mark_,
                   gtt_, tmp_, best_, lb_, station_mark_, time_at_dest_,
                   is_dest_, round_times_, row_count_round_times_);
  this_grid().sync();

  // update_intermodal_footpaths
  update_intermodal_footpaths<SearchDir, Rt>(k, gtt_, dist_to_end_, dist_to_end_size_, station_mark_,
                             prev_station_mark_, time_at_dest_, kUnreachable,
                             gpu_kIntermodalTarget, best_, tmp_, round_times_);

}

template <gpu_direction SearchDir, bool Rt>
__device__ void init_arrivals(gpu_delta_t d_worst_at_dest,
                              gpu_unixtime_t const worst_time_at_dest,
                              gpu_day_idx_t* base_, gpu_delta_t* time_at_dest,
                              gpu_timetable* gtt_){
  auto const t_id = get_global_thread_id();

  if(t_id==0){
    d_worst_at_dest = unix_to_gpu_delta(base(gtt_, base_), worst_time_at_dest);
  }

  if(t_id < gpu_kMaxTransfers+1){
    time_at_dest[t_id] = get_best<SearchDir>(d_worst_at_dest, time_at_dest[t_id]);
  }

}

// größten Teil von raptor.execute() wird hierdrin ausgeführt
// kernel muss sich außerhalb der gpu_raptor Klasse befinden
template <gpu_direction SearchDir, bool Rt>
__global__ void gpu_raptor_kernel(gpu_unixtime_t const start_time,
                                  uint8_t const max_transfers,
                                  gpu_unixtime_t const worst_time_at_dest,
                                  gpu_profile_idx_t const prf_idx,
                                  gpu_raptor<SearchDir,Rt>& gr){
  auto const end_k =
      get_smaller(max_transfers, gpu_kMaxTransfers) + 1U;
  // 1. Initialisierung
  gpu_delta_t d_worst_at_dest{};
  init_arrivals<SearchDir, Rt>(d_worst_at_dest, worst_time_at_dest, gr.base_,
                gr.mem_->device_.time_at_dest_, gr.gtt_);
  this_grid().sync();

  // 2. Update Routes
  for (auto k = 1U; k != end_k; ++k) { // diese Schleife bleibt, da alle Threads in jede Runde gehen

    // Resultate aus lezter Runde von device in variable speichern?  //TODO: typen von kIntermodalTarget und dist_to_end_size falsch???
    raptor_round<SearchDir, Rt>(k, prf_idx, gr.gtt_, *gr.base_, *gr.allowed_claszes_,
                 gr.dist_to_end_, *gr.dist_to_end_size_, gr.is_dest_, gr.lb_, *gr.n_days_,
                 gr.mem_->device_.time_at_dest_,
                 gr.mem_->device_.any_station_marked_, gr.mem_->device_.route_mark_,
                 gr.mem_->device_.station_mark_, gr.mem_->device_.best_,
                 gr.kUnreachable, gr.mem_->device_.prev_station_mark_,
                 gr.mem_->device_.round_times_, gr.mem_->device_.tmp_,
                 gr.mem_->device_.row_count_round_times_,
                 gr.mem_->device_.column_count_round_times_,
                 gr.kIntermodalTarget_, gr.stats_, gr.kMaxTravelTimeTicks_);
    this_grid().sync();
  }
  this_grid().sync();

  //construct journey

  this_grid().sync();

}

#define XSTR(s) STR(s)
#define STR(s) #s

#define CUDA_CALL(call) \
    if ((code = call) != cudaSuccess) {                     \
      printf("CUDA error: %s at " STR(call) " %s:%d\n",     \
             cudaGetErrorString(code), __FILE__, __LINE__); \
      goto fail;                                            \
    }

#define CUDA_COPY_TO_DEVICE(type, target, source, size)                        \
    CUDA_CALL(cudaMalloc(&target, size * sizeof(type)))                          \
    CUDA_CALL(                                                                   \
        cudaMemcpy(target, source, size * sizeof(type), cudaMemcpyHostToDevice))

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
                     unsigned int* & kIntermodalTarget_,
                     short* & kMaxTravelTimeTicks_){
  cudaError_t code;
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
    unsigned int* & kIntermodalTarget_,
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

void inline launch_kernel(void** args,
                          device_context const& device,
                          cudaStream_t s,
                          gpu_direction search_dir,
                          bool rt) {
  cudaSetDevice(device.id_);
  if(search_dir == gpu_direction::kForward && rt == true){
  cudaLaunchCooperativeKernel(gpu_raptor_kernel<gpu_direction::kForward,true>, device.grid_,  //  NOLINT
                              device.threads_per_block_, args, 0, s);
  } else if(search_dir == gpu_direction::kForward && rt == false){
    cudaLaunchCooperativeKernel(gpu_raptor_kernel<gpu_direction::kForward,false>, device.grid_,  //  NOLINT
                                device.threads_per_block_, args, 0, s);
  } else if(search_dir == gpu_direction::kBackward && rt == true){
    cudaLaunchCooperativeKernel(gpu_raptor_kernel<gpu_direction::kBackward,true>, device.grid_,  //  NOLINT
                                device.threads_per_block_, args, 0, s);
  }else if(search_dir == gpu_direction::kBackward && rt == false){
    cudaLaunchCooperativeKernel(gpu_raptor_kernel<gpu_direction::kBackward,false>, device.grid_,  //  NOLINT
                                device.threads_per_block_, args, 0, s);
  }
  cuda_check();
}

inline void fetch_arrivals_async(mem* mem, cudaStream_t s) {
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
void copy_back(mem* mem){
  cuda_check();
  cuda_sync_stream(mem->context_.proc_stream_);
  cuda_check();
  fetch_arrivals_async(mem,mem->context_.transfer_stream_);
  cuda_check();
  cuda_sync_stream(mem->context_.transfer_stream_);
  cuda_check();
}

void add_start_gpu(gpu_location_idx_t const l, gpu_unixtime_t const t,mem* mem_,gpu_timetable* gtt_,gpu_day_idx_t* base_,short const kInvalid){
  trace_upd("adding start {}: {}\n", location{gtt_, l}, t);
  std::vector<gpu_delta_t> best_new(mem_->device_.n_locations_,kInvalid);
  std::vector<gpu_delta_t> round_times_new((mem_->device_.column_count_round_times_*mem_->device_.row_count_round_times_),kInvalid);
  best_new[gpu_to_idx(l)] = unix_to_gpu_delta(base(gtt_,base_), t);
  round_times_new[0U*mem_->device_.row_count_round_times_+ gpu_to_idx(l)] = unix_to_gpu_delta(base(gtt_,base_), t);
  //TODO: fix station_mark ist kein bool!
  std::vector<uint32_t> gpu_station_mark(mem_->device_.n_locations_,0);
  gpu_station_mark[gpu_to_idx(l)] = 1;
  cudaMemcpy(mem_->device_.best_, best_new.data(), mem_->device_.n_locations_*sizeof(gpu_delta_t), cudaMemcpyHostToDevice);
  cudaMemcpy(mem_->device_.round_times_, round_times_new.data(), round_times_new.size()*sizeof(gpu_delta_t), cudaMemcpyHostToDevice);
  cudaMemcpy(mem_->device_.station_mark_, gpu_station_mark.data(), mem_->device_.n_locations_*sizeof(uint32_t), cudaMemcpyHostToDevice);
}

mem* gpu_mem(
    std::vector<gpu_delta_t> tmp,
    std::vector<gpu_delta_t> best,
    std::vector<bool> station_mark,
    std::vector<bool> prev_station_mark,
    std::vector<bool> route_mark,
    gpu_direction search_dir,
    gpu_timetable* gtt){
  short kInvalid = 0;
  if(search_dir == gpu_direction::kForward){
    kInvalid = kInvalidGpuDelta<gpu_direction::kForward>;
  } else{
    kInvalid = kInvalidGpuDelta<gpu_direction::kBackward>;
  }
  auto state = gpu_raptor_state{};
  state.init(*gtt, kInvalid);
  loaned_mem loan(state,kInvalid);
  mem* mem = loan.mem_;
  std::vector<uint32_t> gpu_station_mark(*gtt->n_locations_);
  for (size_t i = 0; i < station_mark.size(); ++i) {
    gpu_station_mark[i] = station_mark[i];
  }
  std::vector<uint32_t> gpu_prev_station_mark(*gtt->n_locations_);
  for (size_t i = 0; i < prev_station_mark.size(); ++i) {
    gpu_prev_station_mark[i] = prev_station_mark[i];
  }
  std::vector<uint32_t> gpu_route_mark(*gtt->n_locations_);
  for (size_t i = 0; i < route_mark.size(); ++i) {
    gpu_route_mark[i] = route_mark[i];
  }

  //TODO: Maybe tmp und best entfernen da eh überschieben???
  cudaMemcpy(mem->device_.tmp_, tmp.data(), (*gtt->n_locations_) * sizeof(gpu_delta_t), cudaMemcpyHostToDevice);
  cuda_check();
  cudaMemcpy(mem->device_.best_, best.data(), (*gtt->n_locations_) * sizeof(gpu_delta_t), cudaMemcpyHostToDevice);
  cuda_check();
  cudaMemcpy(mem->device_.station_mark_, gpu_station_mark.data(), (*gtt->n_locations_) * sizeof(uint32_t), cudaMemcpyHostToDevice);
  cuda_check();
  cudaMemcpy(mem->device_.prev_station_mark_, gpu_prev_station_mark.data(), (*gtt->n_locations_) * sizeof(uint32_t), cudaMemcpyHostToDevice);
  cuda_check();
  cudaMemcpy(mem->device_.route_mark_, gpu_route_mark.data(), (*gtt->n_routes_) * sizeof(uint32_t), cudaMemcpyHostToDevice);
  cuda_check();
  return mem;
}