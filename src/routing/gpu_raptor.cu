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

    printf("update arrival von wert %d", val);
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

    printf("update arrival von wert %d", new_value);
    old_value = atomicCAS(base_address, assumed, new_value);
  } while (assumed != old_value);

  return true;
#endif
}

template <gpu_direction SearchDir, bool Rt>
__device__ void update_time_at_dest(unsigned const k, gpu_delta_t const t, gpu_delta_t * time_at_dest_){
  for (auto i = k; i < gpu_kMaxTransfers+1; ++i) {
    time_at_dest_[i] = get_best<SearchDir>(time_at_dest_[i], t);
  }
}

template <gpu_direction SearchDir, bool Rt>
__device__ void convert_station_to_route_marks(unsigned int* station_marks, unsigned int* route_marks,
                                               bool* any_station_marked,
                                               gpu_vecvec<gpu_location_idx_t , gpu_route_idx_t> const* location_routes_,
                                               std::uint32_t const n_locations) {
  auto const global_t_id = get_global_thread_id();
  auto const global_stride = get_global_stride();
  if(global_t_id == 0)printf("test_convert\n"); //ToDo: nur test delete later
  // anstatt stop_count_ brauchen wir location_routes ?location_idx_{n_locations}?
  for (uint32_t idx = global_t_id; idx < n_locations; idx += global_stride) {
    if (marked(station_marks, idx)) {
      printf("marked!"); //ToDo: nur test delete later
      if (!*any_station_marked) {
        atomicOr(reinterpret_cast<int*>(any_station_marked),1); //TODO atomicor?
      }
      for (auto r : (*location_routes_)[gpu_location_idx_t{idx}]) {
        printf("marked! TEST %d round: %d",r);
        mark(route_marks, gpu_to_idx(r));
      }
    }
  }
}

template <gpu_direction SearchDir, bool Rt>
__device__ gpu_delta_t time_at_stop(gpu_route_idx_t const r, gpu_transport const t,
                                    gpu_stop_idx_t const stop_idx,
                                    gpu_event_type const ev_type,
                                    gpu_day_idx_t base_,
                                    gpu_vector_map<gpu_route_idx_t,gpu_interval<std::uint32_t>> const* route_stop_time_ranges,
                                    gpu_vector_map<gpu_route_idx_t,gpu_interval<gpu_transport_idx_t >> const* route_transport_ranges,
                                    gpu_delta const* route_stop_times){
  auto const n_transports = static_cast<unsigned>((*route_transport_ranges)[r].size());
  auto const route_stop_begin = static_cast<unsigned>( (*route_stop_time_ranges)[r].from_ + n_transports *
                                                                              (stop_idx * 2 - (ev_type==gpu_event_type::kArr ? 1 : 0)));
  printf("GPU n_transports: %d,route stop begin %d",n_transports,route_stop_begin);
  printf("GPU clamp %d",gpu_clamp((as_int(t.day_) - as_int(base_)) * 1440
                                  + route_stop_times[route_stop_begin +
                                                     (gpu_to_idx(t.t_idx_) - gpu_to_idx((*route_transport_ranges)[r].from_))].count()));
  printf("event_mam gpu: %d",route_stop_times[route_stop_begin +
                                               (gpu_to_idx(t.t_idx_) - gpu_to_idx((*route_transport_ranges)[r].from_))].count());
  return gpu_clamp((as_int(t.day_) - as_int(base_)) * 1440
                   + route_stop_times[route_stop_begin +
                                             (gpu_to_idx(t.t_idx_) - gpu_to_idx((*route_transport_ranges)[r].from_))].count());
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
                                    std::size_t const day,
                                    gpu_vector_map<gpu_transport_idx_t,gpu_bitfield_idx_t> const* transport_traffic_days,
                                    gpu_vector_map<gpu_bitfield_idx_t, gpu_bitfield> const* bitfields)  {
  assert((*transport_traffic_days).el_ != nullptr);
  assert((*bitfields).el_ !=  nullptr);
  if (day >= (*bitfields)[(*transport_traffic_days)[t]].size()) {
    return false;
  }
  auto const block = (*bitfields)[(*transport_traffic_days)[t]].blocks_[day / (*bitfields)[(*transport_traffic_days)[t]].bits_per_block];
  auto const bit = (day % (*bitfields)[(*transport_traffic_days)[t]].bits_per_block);
  printf("gpu test bit: %d",bit);
  printf("gpu test block: %d",block);
  printf("gpu test day: %d",day);
  return (block & (std::uint64_t{1U} << bit)) != 0U;
}

template <gpu_direction SearchDir>
__device__ bool valid(gpu_delta_t t) {
  // Use if constexpr to ensure compile-time evaluation
  if constexpr (SearchDir == gpu_direction::kForward) {
    return t != cuda::std::numeric_limits<gpu_delta_t>::max();
  } else {
    return t != cuda::std::numeric_limits<gpu_delta_t>::min();
  }
}



template <gpu_direction SearchDir, bool Rt>
__device__ void update_route_smaller32(const unsigned k,const gpu_route_idx_t r,
                                       gpu_raptor_stats* stats_,
                                       uint32_t* prev_station_mark_, gpu_delta_t* best_,
                                       gpu_delta_t* round_times_, uint32_t column_count_round_times_,
                                       gpu_delta_t* tmp_,short const* kMaxTravelTimeTicks_,
                                       uint16_t* lb_, int n_days_,
                                       gpu_delta_t* time_at_dest_,
                                       uint32_t* station_mark_, gpu_day_idx_t* base_,
                                       unsigned short kUnreachable, bool* any_station_marked_,
                                       gpu_delta const* route_stop_times,
                                       gpu_vecvec<gpu_route_idx_t,gpu_value_type> const* route_location_seq,
                                       gpu_vecvec<gpu_location_idx_t , gpu_route_idx_t> const* location_routes,
                                       std::uint32_t const n_locations,
                                       std::uint32_t const n_routes,
                                       gpu_vector_map<gpu_route_idx_t,gpu_interval<std::uint32_t>> const* route_stop_time_ranges,
                                       gpu_vector_map<gpu_route_idx_t,gpu_interval<gpu_transport_idx_t >> const* route_transport_ranges,
                                       gpu_vector_map<gpu_bitfield_idx_t, gpu_bitfield> const* bitfields,
                                       gpu_vector_map<gpu_transport_idx_t,gpu_bitfield_idx_t> const* transport_traffic_days,
                                       gpu_interval<gpu_sys_days> const* date_range,
                                       gpu_vector_map<gpu_location_idx_t, gpu_u8_minutes> const* transfer_time,
                                       gpu_vecvec<gpu_location_idx_t, nigiri::gpu_footpath> const* gpu_footpaths_out,
                                       gpu_vecvec<gpu_location_idx_t, nigiri::gpu_footpath> const* gpu_footpaths_in,
                                       gpu_vector_map<gpu_route_idx_t, gpu_clasz> const* route_clasz){

  auto local_any_station = false;
  printf("smaller");
  auto const t_id = threadIdx.x;
  auto const stop_seq = (*route_location_seq)[r];
  printf("smaller after stop seq");
  auto stop_idx = -1;
  gpu_stop stp{};
  unsigned int l_idx;
  //bool is_last;
  gpu_delta_t prev_round_time = ((SearchDir == gpu_direction::kForward) ? cuda::std::numeric_limits<gpu_delta_t>::max()
                                                                 : cuda::std::numeric_limits<gpu_delta_t>::min());
  printf("ind test: %d", stop_seq.size());
  unsigned leader = stop_seq.size();
  unsigned int active_stop_count = stop_seq.size();


  //TODO: muss eigentlich hier weil prev_round_time falsch wird
  if(t_id >= active_stop_count) { //solte doch eigentlich egal sein da wir smaller sind oder?
    return;
  }
  printf("smaller after any_station_marked");
    stop_idx = static_cast<gpu_stop_idx_t>(
        (SearchDir == gpu_direction::kForward) ? t_id : stop_seq.size() - t_id - 1U);

    printf("smaller after any_station_marked1");
    stp = gpu_stop{stop_seq[stop_idx]};

    printf("smaller after any_station_marked2");
    l_idx = gpu_to_idx(stp.gpu_location_idx());

    printf("smaller after any_station_marked3");
    auto is_last = t_id == stop_seq.size() - 1U;
    // ist äquivalent zu prev_arrival
    if (is_last || !(SearchDir == gpu_direction::kForward ? stp.in_allowed() : stp.out_allowed()) ||
        !marked(prev_station_mark_,l_idx)) {
      //dann wird diese übersprungen
      return; //übersprungen ist gleich dieser thread returned
    }
    if (lb_[l_idx] == kUnreachable) { //TODO: maybe kucken ob richtig? bzw. ob weg
      // dann wird Durchgehen dieser Route abgebrochen
      return; //break ist gleich alle thread returnen
    }
    prev_round_time = round_times_[(k-1) * column_count_round_times_ + l_idx];
    auto rt = round_times_[(k-1) * column_count_round_times_ + l_idx];
    printf("rt value: %d",rt);
    if(!__any_sync(FULL_MASK, rt!=cuda::std::numeric_limits<gpu_delta_t>::max())||
        !__any_sync(FULL_MASK, rt!=cuda::std::numeric_limits<gpu_delta_t>::min())){
      printf("round_times marked??? %d, L_idx: %d",marked(prev_station_mark_,l_idx),l_idx);
      printf("round_times value: %d , k = %d",round_times_[0],k);
      printf("round_times_max/min %d",(k-1) * column_count_round_times_ + l_idx);
    }
    printf("smaller after any_station_marked4");

    assert(__any_sync(FULL_MASK, prev_round_time!=cuda::std::numeric_limits<gpu_delta_t>::max())&&
             __any_sync(FULL_MASK, prev_round_time!=cuda::std::numeric_limits<gpu_delta_t>::min()));

  printf("smaller 2");
  // berechnen von allen möglichen trips(Abfahrt-/Ankunftszeiten) von dieser station
  auto const splitter = gpu_split_day_mam(*base_, prev_round_time);
  printf("smaller 3");
  //TODO: hier ist prev_round_time numeric_limit need fix
  auto const day_at_stop = splitter.first;
  auto const mam = splitter.second;
  printf("smaller 3.6");
  auto const n_days_to_iterate = get_smaller(
      gpu_kMaxTravelTime.count() / 1440 + 1,
      (SearchDir == gpu_direction::kForward) ? n_days_ - as_int(day_at_stop) : as_int(day_at_stop) + 1);
  printf("smaller 3.7");
  assert(route_stop_time_ranges->el_ != nullptr);
  auto const arrival_times = gpu_event_times_at_stop(r, stop_idx, gpu_event_type::kArr, route_stop_time_ranges,route_transport_ranges, route_stop_times);
  printf("smaller 3.8");
  auto const departure_times = gpu_event_times_at_stop(r, stop_idx, gpu_event_type::kDep, route_stop_time_ranges,route_transport_ranges, route_stop_times);
  printf("smaller 3.9");
  auto const seek_first_day = [&]() {
    return linear_lb(gpu_get_begin_it<SearchDir>(departure_times),
                     gpu_get_end_it<SearchDir>(departure_times), mam,
                     [&](gpu_delta const a, gpu_minutes_after_midnight_t const b) {
                       return is_better<SearchDir>(a.mam_, b.count());
                     });
  };
  printf("smaller 4");
  gpu_transport_idx_t t = gpu_transport_idx_t::invalid();
  gpu_delta_t dep_t = (SearchDir == gpu_direction::kForward) ? dep_t = cuda::std::numeric_limits<gpu_delta_t>::max() : cuda::std::numeric_limits<gpu_delta_t>::min();
  int i_f = 0;
  for (auto i = 0U; i < n_days_to_iterate; ++i){ // die Schleife geht durch alle Tage
    auto const ev_time_range =
        gpu_it_range{gpu_day_idx_t{i} == 0U ? seek_first_day() : gpu_get_begin_it<SearchDir>(departure_times),
                 gpu_get_end_it<SearchDir>(departure_times)};
    printf("UPDATE!!!!!!!1");
    if (ev_time_range.empty()) {
      return;
    }
    printf(" 2 ");
    auto day = (SearchDir == gpu_direction::kForward) ? day_at_stop + gpu_day_idx_t{i} : day_at_stop - gpu_day_idx_t{i};
    for (auto it = begin(ev_time_range); it < end(ev_time_range); ++it){ // die Schleife geht durch alle Zeiten
      printf(" 3 ");
      if(t_id < active_stop_count){
        ++stats_[t_id%32].n_earliest_trip_calls_;
        auto const t_offset = (SearchDir == gpu_direction::kForward) ? static_cast<cuda::std::size_t>(&*it - departure_times.data()) : static_cast<cuda::std::size_t>(&*it - arrival_times.data());
        t = (*route_transport_ranges)[r][t_offset];
        auto const dep = *it;
        auto const dep_mam = dep.mam_;
        dep_t = to_gpu_delta(day, dep_mam, base_);

        printf("test 4 ");

      }
      // election of leader
      unsigned ballot = __ballot_sync(
          FULL_MASK, (t_id < active_stop_count) && valid<SearchDir>(prev_round_time)  &&
                         valid<SearchDir>(dep_t) &&
                         (is_better_or_eq<SearchDir>(prev_round_time,dep_t)) &&
                         is_transport_active<SearchDir, Rt>(t, as_int(day), transport_traffic_days, bitfields));
      //ist immer true
      leader = __ffs(ballot) - 1;
      printf(" 6 ");
      // alle station nach leader können jetzt updaten, wenn der transport an dem Tag fährt
      auto const t_offset = (SearchDir == gpu_direction::kForward) ?  static_cast<cuda::std::size_t>(&*it - departure_times.data()) : static_cast<cuda::std::size_t>(&*it - arrival_times.data());

      assert((*route_transport_ranges)[r].size() > t_offset);
      auto const t2 = (*route_transport_ranges)[r][t_offset];
      if(t_id > leader ) {
        printf("if0");
        i_f++;
        if (t_id < active_stop_count){
          printf("if1");
          i_f++;
          if (is_transport_active<SearchDir, Rt>(
                  t2, as_int(day), transport_traffic_days, bitfields)) {
            i_f++;
            printf("UPDATE!!!!!!!3");
            auto const arr = *it;
            auto const arr_mam = arr.mam_;
            auto const arr_t = to_gpu_delta(day, arr_mam, base_);

            auto updated = atomicExch(&tmp_[stop_idx], arr_t);
            if (updated != tmp_[stop_idx]) {
              printf("UPDATE!!!!!!!4");
              mark(station_mark_, stop_idx);
              local_any_station = true;
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
  if(get_global_thread_id() == 0){
    for (int i = 0; i < n_locations; i++) {
      printf("n_stations update: %d", station_mark_[i]);
    }

  }
  printf("any_station update: %d", any_station_marked_);
  if(i_f>0)printf("test if: %d",i_f);
  printf("smaller3");
  if (local_any_station){
    atomicOr(reinterpret_cast<int*>(any_station_marked_),1);
  }
}

template <gpu_direction SearchDir, bool Rt>
__device__ void update_route_bigger32(unsigned const k, gpu_route_idx_t r,
                                      gpu_raptor_stats* stats_,
                                      uint32_t* prev_station_mark_, gpu_delta_t* best_,
                                      gpu_delta_t* round_times_, uint32_t column_count_round_times_,
                                      gpu_delta_t* tmp_, short const* kMaxTravelTimeTicks_,
                                      uint16_t* lb_, int n_days_,
                                      gpu_delta_t* time_at_dest_,
                                      uint32_t* station_mark_, gpu_day_idx_t* base_,
                                      unsigned short kUnreachable, bool* any_station_marked_,
                                      gpu_delta const* route_stop_times,
                                      gpu_vecvec<gpu_route_idx_t,gpu_value_type> const* route_location_seq,
                                      gpu_vecvec<gpu_location_idx_t , gpu_route_idx_t> const* location_routes,
                                      std::uint32_t const n_locations,
                                      std::uint32_t const n_routes,
                                      gpu_vector_map<gpu_route_idx_t,gpu_interval<std::uint32_t>> const* route_stop_time_ranges,
                                      gpu_vector_map<gpu_route_idx_t,gpu_interval<gpu_transport_idx_t >> const* route_transport_ranges,
                                      gpu_vector_map<gpu_bitfield_idx_t, gpu_bitfield> const* bitfields,
                                      gpu_vector_map<gpu_transport_idx_t,gpu_bitfield_idx_t> const* transport_traffic_days,
                                      gpu_interval<gpu_sys_days> const* date_range,
                                      gpu_vector_map<gpu_location_idx_t, gpu_u8_minutes> const* transfer_time,
                                      gpu_vecvec<gpu_location_idx_t, nigiri::gpu_footpath> const* gpu_footpaths_out,
                                      gpu_vecvec<gpu_location_idx_t, nigiri::gpu_footpath> const* gpu_footpaths_in,
                                      gpu_vector_map<gpu_route_idx_t, gpu_clasz> const* route_clasz){
  printf("UPDATE BIGGER");
  auto local_any_station = false;
  auto const t_id = threadIdx.x;

  unsigned int leader = NO_LEADER;
  unsigned int any_arrival = 0;

  auto const stop_seq = (*route_location_seq)[r];
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
  auto const arrival_times = gpu_event_times_at_stop(r, stop_idx, gpu_event_type::kArr, route_stop_time_ranges,route_transport_ranges, route_stop_times);
  auto const departure_times = gpu_event_times_at_stop(r, stop_idx, gpu_event_type::kDep, route_stop_time_ranges,route_transport_ranges, route_stop_times);
  auto const seek_first_day = [&]() {
    return linear_lb(gpu_get_begin_it<SearchDir>(departure_times),
                     gpu_get_end_it<SearchDir>(departure_times), mam,
                     [&](gpu_delta const a, gpu_minutes_after_midnight_t const b) {
                       return is_better<SearchDir>(a.mam_, b.count());
                     });
  };

  for (auto i = 0U; i < n_days_to_iterate; ++i){
    auto const ev_time_range =
        gpu_it_range{gpu_day_idx_t{i} == 0U ? seek_first_day() : gpu_get_begin_it<SearchDir>(departure_times),
                     gpu_get_end_it<SearchDir>(departure_times)};
    if (ev_time_range.empty()) {
      return;
    }
    auto day = (SearchDir == gpu_direction::kForward) ? day_at_stop + gpu_day_idx_t{i} : day_at_stop - gpu_day_idx_t{i};
    for (auto it = begin(ev_time_range); it < end(ev_time_range); ++it){
      for(int current_stage=0; current_stage<active_stage_count; ++current_stage){
        int stage_id = (current_stage << 5) + t_id;
        if(stage_id < active_stop_count){
          stop_idx = static_cast<gpu_stop_idx_t>(
              (SearchDir == gpu_direction::kForward) ? t_id : stop_seq.size() - t_id - 1U);
          stp = gpu_stop{stop_seq[stop_idx]};
          l_idx = gpu_to_idx(stp.gpu_location_idx());
          //is_last = t_id == stop_seq.size() - 1U;
          // ist äquivalent zu prev_arrival
          prev_round_time = round_times_[(k - 1) * column_count_round_times_ + l_idx];
        }
        any_arrival |= __any_sync(FULL_MASK, valid<SearchDir>(prev_round_time));
        if (current_stage == active_stage_count - 1 && !any_arrival) {
          return;
        }
        if(!any_arrival){
          continue;
        }
        if(stage_id < active_stop_count){
          ++stats_[t_id%32].n_earliest_trip_calls_;
          auto const t_offset = static_cast<cuda::std::size_t>(&*it - departure_times.data());
          auto const t = (*route_transport_ranges)[r][t_offset];
          auto const dep = *it;
          auto const dep_mam = dep.mam_;
          auto const dep_t = to_gpu_delta(day, dep_mam, base_);
          // election of leader
          unsigned ballot = __ballot_sync(
              FULL_MASK, (t_id < active_stop_count) && valid<SearchDir>(prev_round_time)  &&
                             valid<SearchDir>(dep_t) &&
                             (prev_round_time <= dep_t) &&
                             is_transport_active<SearchDir, Rt>(t, as_int(day), transport_traffic_days, bitfields));
          leader = __ffs(ballot) - 1;
        }
        if(leader != NO_LEADER){
          leader += current_stage << 5;
        }
        // zuerst current stage updaten

        if(leader != NO_LEADER && stage_id < active_stop_count){
          auto const t_offset = static_cast<cuda::std::size_t>(&*it - arrival_times.data());
          auto const t = (*route_transport_ranges)[r][t_offset];
          if(stage_id > leader && is_transport_active<SearchDir, Rt>(t, as_int(day), transport_traffic_days, bitfields)){
            auto const arr = *it;
            auto const arr_mam = arr.mam_;
            auto const arr_t = to_gpu_delta(day, arr_mam, base_);

            auto updated = atomicExch(&tmp_[stop_idx], arr_t);
            if(updated != tmp_[stop_idx]){
              mark(station_mark_, stop_idx);
              local_any_station = true;
            }
          }
        }
        //TODO stop_idx stimmt nicht -> sollte t_id/stage_id/upward_id enthalten?
        // dann alle späteren stages updaten
        if(leader != NO_LEADER){
          for(int upward_stage = current_stage+1; upward_stage < active_stage_count; ++upward_stage){
            int upwards_id = (upward_stage<<5) + t_id;
            auto const t_offset = static_cast<cuda::std::size_t>(&*it - arrival_times.data());
            auto const t = (*route_transport_ranges)[r][t_offset];
            if(upwards_id < active_stop_count && is_transport_active<SearchDir, Rt>(t,as_int(day), transport_traffic_days, bitfields)){
              auto const arr = *it;
              auto const arr_mam = arr.mam_;
              auto const arr_t = to_gpu_delta(day, arr_mam, base_);

              auto updated = atomicExch(&tmp_[stop_idx], arr_t);
              if(updated != tmp_[stop_idx]){
                mark(station_mark_, stop_idx);
                local_any_station = true;
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
  if(local_any_station){
    atomicOr(reinterpret_cast<int*>(any_station_marked_),1);
  }
}


//hilfsmethode für update_route
template <gpu_direction SearchDir, bool Rt>
__device__ gpu_transport get_earliest_transport(unsigned const k,
                                                gpu_route_idx_t const r,
                                                gpu_stop_idx_t const stop_idx,
                                                gpu_day_idx_t const day_at_stop,
                                                gpu_minutes_after_midnight_t const mam_at_stop,
                                                gpu_location_idx_t const l,
                                                gpu_raptor_stats* stats_,
                                                uint16_t* lb_,
                                                gpu_delta_t* time_at_dest_,
                                                gpu_vector_map<gpu_route_idx_t,gpu_interval<gpu_transport_idx_t >> const* route_transport_ranges,
                                                gpu_vector_map<gpu_route_idx_t,gpu_interval<uint32_t >> const* route_stop_time_ranges,
                                                int n_days_, gpu_day_idx_t* base_,
                                                gpu_delta const* route_stop_times,
                                                gpu_vector_map<gpu_bitfield_idx_t, gpu_bitfield> const* bitfields,
                                                gpu_vector_map<gpu_transport_idx_t,gpu_bitfield_idx_t> const* transport_traffic_days) {
  ++stats_[get_global_thread_id()%32].n_earliest_trip_calls_;

  auto const n_days_to_iterate = get_smaller(
      gpu_kMaxTravelTime.count() / 1440 + 1,
      (SearchDir == gpu_direction::kForward) ? n_days_ - as_int(day_at_stop) : as_int(day_at_stop) + 1);

  auto const event_times = gpu_event_times_at_stop(
      r, stop_idx, (SearchDir == gpu_direction::kForward) ? gpu_event_type::kDep : gpu_event_type::kArr, route_stop_time_ranges,route_transport_ranges, route_stop_times);
  printf("gpu r %d, stop_idx %d",r,stop_idx);

  auto const seek_first_day = [&]() {
    return linear_lb(gpu_get_begin_it<SearchDir>(event_times), gpu_get_end_it<SearchDir>(event_times),
                     mam_at_stop,
                     [&](gpu_delta const a, gpu_minutes_after_midnight_t const b) {
                       return is_better<SearchDir>(a.mam_, b.count());
                     });
  };
  printf("GPU mem: %d",mam_at_stop);

#if defined(NIGIRI_TRACING)
  auto const l_idx =
      stop{tt_.route_location_seq_[r][stop_idx]}.location_idx();

  trace(
      "┊ │k={}    et: current_best_at_stop={}, stop_idx={}, location={}, "
      "n_days_to_iterate={}\n",
      k, tt_.to_unixtime(day_at_stop, mam_at_stop), stop_idx,
      location{tt_, l_idx}, n_days_to_iterate);
#endif

  for (auto i = gpu_day_idx_t::value_t{0U}; i != n_days_to_iterate; ++i) {
    auto const ev_time_range =
        gpu_it_range{i == 0U ? seek_first_day() : gpu_get_begin_it<SearchDir>(event_times),
                 gpu_get_end_it<SearchDir>(event_times)};

    printf("GPU i: %d,seek_first_day: %d",i,seek_first_day); //TODO: seek_first_day() ist 0
    for (auto r :event_times) {
      printf("gpu event_time: %d",r);
    }
    printf("gpu ev_time_range %d",ev_time_range.begin());
    printf("gpu ev_time_range %d",ev_time_range.end());
    if (ev_time_range.empty()) {
      continue;
    }

    auto const day = (SearchDir == gpu_direction::kForward) ? day_at_stop + i : day_at_stop - i;
    for (auto it = begin(ev_time_range); it != end(ev_time_range); ++it) {
      auto const t_offset =
          static_cast<std::size_t>(&*it - event_times.data());
      auto const ev = *it;
      auto const ev_mam = ev.mam_;

      if (is_better_or_eq<SearchDir>(time_at_dest_[k],
                          to_gpu_delta(day, ev_mam, base_) + dir<SearchDir>(lb_[gpu_to_idx(l)]))) {
        //hier geht cpu rein
        return {gpu_transport_idx_t::invalid(), gpu_day_idx_t::invalid()};
      }
      printf("GPU Test1: rtr , t_offset: %d, k: %d , it %d!",t_offset,k,it);
      auto const t = (*route_transport_ranges)[r][t_offset];
      printf("Test2: rtr");
      if (i == 0U && !is_better_or_eq<SearchDir>(mam_at_stop.count(), ev_mam)) {
        continue;
      }

      auto const ev_day_offset = ev.days_;
      auto const start_day =
          static_cast<std::size_t>(as_int(day) - ev_day_offset);

      if (k==1 && i==0)
        printf("return gpu");
      if(!is_transport_active<SearchDir, Rt>(t, start_day, transport_traffic_days, bitfields)) {
        continue;
      }
      return {t, static_cast<gpu_day_idx_t>(as_int(day) - ev_day_offset)};
    }
  }
  return {};
}

//nicht parallele update_route
template <gpu_direction SearchDir, bool Rt>
__device__ void update_route(unsigned const k, gpu_route_idx_t const r,
                  gpu_vecvec<gpu_route_idx_t,gpu_value_type> const* route_location_seq,
                  gpu_raptor_stats* stats_,
                  uint32_t* prev_station_mark_, gpu_delta_t* best_,
                  gpu_delta_t* round_times_, uint32_t column_count_round_times_,
                  gpu_delta_t* tmp_,
                  uint16_t* lb_,
                  gpu_delta_t* time_at_dest_,
                  uint32_t* station_mark_,
                  unsigned short kUnreachable,
                  bool* any_station_marked_,
                  gpu_day_idx_t* base_,
                             gpu_vector_map<gpu_route_idx_t,gpu_interval<gpu_transport_idx_t >> const* route_transport_ranges,
                  gpu_vector_map<gpu_route_idx_t,gpu_interval<uint32_t>> const* route_stop_time_ranges,
                  int n_days_, gpu_vector_map<gpu_bitfield_idx_t, gpu_bitfield> const* bitfields,
                  gpu_delta const* route_stop_times,
                  gpu_vector_map<gpu_transport_idx_t,gpu_bitfield_idx_t> const* transport_traffic_days) {
  auto const stop_seq = (*route_location_seq)[r];
  printf("Test k: %d",k);
  // diese Variable ist das Problem beim Parallelisieren
  auto et = gpu_transport{};
  // hier gehen wir durch alle Stops der Route r → das wollen wir in update_smaller/bigger machen
  for (auto i = 0U; i != stop_seq.size(); ++i) {
    auto const stop_idx =
        static_cast<gpu_stop_idx_t>((SearchDir == gpu_direction::kForward) ? i : stop_seq.size() - i - 1U);
    auto const stp = gpu_stop{stop_seq[stop_idx]};
    auto const l_idx = gpu_to_idx(stp.gpu_location_idx());
    auto const is_last = i == stop_seq.size() - 1U;

    // wenn transportmittel an dem Tag nicht fährt &
    // wenn station nicht markiert ist, wird diese übersprungen → springt zur nächsten station
    if (!et.is_valid() && !marked(prev_station_mark_, l_idx)) {
      continue;
    }
    auto current_best = kInvalidGpuDelta<SearchDir>;
    //wenn station ausgehende/eingehende Transportmittel hat & transportmittel an dem Tag fährt

    if (et.is_valid() && ((SearchDir == gpu_direction::kForward) ? stp.out_allowed() : stp.in_allowed())) {
      // wann transportmittel an dieser station ankommt
      auto const by_transport = time_at_stop<SearchDir, Rt>(
          r, et, stop_idx, (SearchDir == gpu_direction::kForward) ? gpu_event_type::kArr : gpu_event_type::kDep, *base_, route_stop_time_ranges, route_transport_ranges, route_stop_times);
      // beste Zeit für diese station bekommen
      current_best = get_best<SearchDir>(round_times_[(k - 1)*column_count_round_times_ + l_idx],
                              tmp_[l_idx], best_[l_idx]);
      assert(by_transport != cuda::std::numeric_limits<gpu_delta_t>::min() &&
             by_transport != cuda::std::numeric_limits<gpu_delta_t>::max());
      // wenn Ankunftszeit dieses Transportmittels besser ist als beste Ankunftszeit für station
      // & vor frühster Ankunftszeit am Ziel liegt

      if (is_better<SearchDir>(by_transport, current_best) &&
          is_better<SearchDir>(by_transport, time_at_dest_[k]) &&
          lb_[l_idx] != kUnreachable &&
          is_better<SearchDir>(by_transport + dir<SearchDir>(lb_[l_idx]), time_at_dest_[k])) {
        // dann wird frühste Ankunftszeit an dieser Station aktualisiert
        // hier einziger Punkt, wo gemeinsame Variablen verändert werden → ATOMIC
        ++stats_[get_global_thread_id()%32].n_earliest_arrival_updated_by_route_;
        tmp_[l_idx] = get_best<SearchDir>(by_transport, tmp_[l_idx]);
        mark(station_mark_, l_idx);
        current_best = by_transport;
        atomicOr(reinterpret_cast<int*>(any_station_marked_),1);
        printf("any_station_marked must be true! GPU %d , k: %d",*any_station_marked_,k);
      }
    }

    // wenn es die letzte Station in der Route ist
    // oder es keine ausgehenden/eingehenden transportmittel gibt
    // oder die Station nicht markiert war
    if(k==2)
      printf("counter k==2 GPU" );
    if (is_last || !((SearchDir == gpu_direction::kForward) ? stp.in_allowed() : stp.out_allowed()) ||
        !marked(prev_station_mark_, l_idx)) {
      //dann wird diese übersprungen
      continue;
    }

    // wenn der lowerBound von der Station nicht erreichbar ist,
    // werden die darauffolgenden Stationen auch nicht erreichbar sein

    if (lb_[l_idx] == kUnreachable) {
      // dann wird Durchgehen dieser Route abgebrochen
      break;
    }

    // wenn Transportmittel an dem Tag fährt, dann ist das hier Ankunftszeit am Stop
    auto const et_time_at_stop =
        et.is_valid()
            ? time_at_stop<SearchDir, Rt>(r, et, stop_idx,
                           (SearchDir == gpu_direction::kForward) ? gpu_event_type::kDep : gpu_event_type::kArr,
                           *base_, route_stop_time_ranges, route_transport_ranges, route_stop_times)
            : kInvalidGpuDelta<SearchDir>;
    // vorherige Ankunftszeit an der Station
    printf("GPU K %d", k);
    auto const prev_round_time = round_times_[(k-1) * column_count_round_times_ + l_idx];
    if(k==2) printf("GPU l_idx: %d",l_idx);
    if(k==3)printf("GPU round_times k=1 %d, idx: %d", prev_round_time, (k-1) * column_count_round_times_ + l_idx);
    assert(prev_round_time != kInvalidGpuDelta<SearchDir>);
    // wenn vorherige Ankunftszeit besser ist → dann sucht man weiter nach besserem Umstieg in ein Transportmittel
    //printf("GPU prev_round_time %d, et_time_at_stop %d", prev_round_time, et_time_at_stop);
    printf("GPU prev_round_time: %d et_time_at_stop: %d",prev_round_time,et_time_at_stop);
    printf("is better or eq: %d",is_better_or_eq<SearchDir>(prev_round_time, et_time_at_stop));

    if (is_better_or_eq<SearchDir>(prev_round_time, et_time_at_stop)) {

      auto const [day, mam] = gpu_split_day_mam(*base_, prev_round_time);
      // Hier muss leader election stattfinden
      // dann wird neues Transportmittel, das am frühsten von station abfährt

      auto const new_et = get_earliest_transport<SearchDir, Rt>(k, r, stop_idx, day, mam,
                                                 stp.gpu_location_idx(), stats_, lb_, time_at_dest_,route_transport_ranges,
                                                                route_stop_time_ranges, n_days_, base_, route_stop_times, bitfields, transport_traffic_days);
      printf("GPU new_et valid %d , K: %d", new_et.is_valid(), k);
      current_best =
          get_best<SearchDir>(current_best, best_[l_idx], tmp_[l_idx]);
      // wenn neues Transportmittel an diesem Tag fährt und
      // bisherige beste Ankunftszeit Invalid ist ODER Ankunftszeit an Station besser als Ankunftszeit von neuem Transportmittel

      if (new_et.is_valid() &&
          (current_best == kInvalidGpuDelta<SearchDir> ||
           is_better_or_eq<SearchDir>(
               time_at_stop<SearchDir, Rt>(r, new_et, stop_idx,
                            (SearchDir == gpu_direction::kForward) ? gpu_event_type::kDep : gpu_event_type::kArr, *base_, route_stop_time_ranges, route_transport_ranges, route_stop_times),
               et_time_at_stop))) {

        // dann wird neues Transportmittel genommen
        et = new_et;
      }
    }
  }
}

template <gpu_direction SearchDir, bool Rt, bool WithClaszFilter>
__device__ void loop_routes(unsigned const k, bool* any_station_marked_, uint32_t* route_mark_,
                            gpu_clasz_mask_t const* allowed_claszes_,
                            gpu_raptor_stats* stats_,
                            short const* kMaxTravelTimeTicks_, uint32_t* prev_station_mark_,
                            gpu_delta_t* best_,
                            gpu_delta_t* round_times_, uint32_t column_count_round_times_,
                            gpu_delta_t* tmp_,
                            uint16_t* lb_, int n_days_,
                            gpu_delta_t* time_at_dest_,
                            uint32_t* station_mark_, gpu_day_idx_t* base_,
                            unsigned short kUnreachable,
                            gpu_delta const* route_stop_times,
                            gpu_vecvec<gpu_route_idx_t,gpu_value_type> const* route_location_seq,
                            gpu_vecvec<gpu_location_idx_t , gpu_route_idx_t> const* location_routes,
                            std::uint32_t const n_locations,
                            std::uint32_t const n_routes,
                            gpu_vector_map<gpu_route_idx_t,gpu_interval<std::uint32_t>> const* route_stop_time_ranges,
                            gpu_vector_map<gpu_route_idx_t,gpu_interval<gpu_transport_idx_t >> const* route_transport_ranges,
                            gpu_vector_map<gpu_bitfield_idx_t, gpu_bitfield> const* bitfields,
                            gpu_vector_map<gpu_transport_idx_t,gpu_bitfield_idx_t> const* transport_traffic_days,
                            gpu_interval<gpu_sys_days> const* date_range,
                            gpu_vector_map<gpu_location_idx_t, gpu_u8_minutes> const* transfer_time,
                            gpu_vecvec<gpu_location_idx_t, nigiri::gpu_footpath> const* gpu_footpaths_out,
                            gpu_vecvec<gpu_location_idx_t, nigiri::gpu_footpath> const* gpu_footpaths_in,
                            gpu_vector_map<gpu_route_idx_t, gpu_clasz> const* route_clasz){
  if(get_global_thread_id() == 0){
    atomicAnd(reinterpret_cast<int*>(any_station_marked_),0);
  }
  this_grid().sync();
  //printf("loop routs intern");
  auto const global_t_id = get_global_thread_id();
  if(get_global_thread_id() ==0){
    printf("loop_routes: begin2");
  }
  //Hier gehen wir durch alle Routen wie in update_routes_dev von Julian
  auto const stride = blockDim.y * gridDim.x;
  auto const start_r_id = threadIdx.y + (blockDim.y * blockIdx.x);
  /*
  for(auto r_idx = start_r_id;
       r_idx < n_routes; r_idx += stride){
    */
  if(get_global_thread_id() == 0)
  for (auto r_idx = 0U; r_idx != n_routes; ++r_idx) {

    auto const r = gpu_route_idx_t{r_idx};
    if(!marked(route_mark_, r_idx)) {
      continue;
    }
      printf("gpu_raptor_kernel: after marked");
      if constexpr (WithClaszFilter){
        auto const as_mask = static_cast<gpu_clasz_mask_t>(1U << static_cast<std::underlying_type_t<gpu_clasz>>((*route_clasz)[r]));
        printf("gpu_raptor_kernel: continue");
        if(!((*allowed_claszes_ & as_mask)==as_mask)){
          continue;
        }
      }
      printf("gpu_raptor_kernel: bevor update_route %d", global_t_id);
      ++stats_[global_t_id%32].n_routes_visited_; // wir haben 32 Stellen und es schreiben nur 32 threads gleichzeitig
      // hier in smaller32 und bigger32 aufteilen?
      // TODO hier in smaller32 und bigger32 aufteilen? → aber hier geht nur ein thread rein...
      // also sollte vielleicht diese Schleife mit allen auf einmal durchgangen werden???
      // parameter stimmen noch nicht
      // parallele update_route
      /*
      if((*route_location_seq)[gpu_route_idx_t{r_idx}].size() <= 32){ // die Route hat <= 32 stops
        update_route_smaller32<SearchDir, Rt>(k, r, stats_, prev_station_mark_, best_,
                                                      round_times_, column_count_round_times_, tmp_, kMaxTravelTimeTicks_,
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
                                                                    transfer_time,
                                                                    gpu_footpaths_in,
                                                                    gpu_footpaths_out,
                                                                    route_clasz);
      }
      else{ // diese Route hat > 32 Stops
         update_route_bigger32<SearchDir, Rt>(k, r, stats_, prev_station_mark_, best_,
                                                                   round_times_, column_count_round_times_, tmp_, kMaxTravelTimeTicks_,
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
                                                                   transfer_time,
                                                                    gpu_footpaths_in,
                                                                   gpu_footpaths_out,
                                                                   route_clasz);

      }
        */
        update_route<SearchDir, Rt>(k, r, route_location_seq, stats_, prev_station_mark_, best_, round_times_,
                                    column_count_round_times_, tmp_, lb_, time_at_dest_, station_mark_, kUnreachable, any_station_marked_,
                                    base_,route_transport_ranges,route_stop_time_ranges, n_days_, bitfields, route_stop_times, transport_traffic_days);

       }

  this_grid().sync();
  if(get_global_thread_id() == 0)printf("loop routes marked end: %d, round %d", *any_station_marked_,k); //  immer 0
}

template <gpu_direction SearchDir, bool Rt>
__device__ void update_transfers(unsigned const k, bool const * is_dest_, uint16_t* dist_to_end_,
                                 uint32_t dist_to_end_size_, gpu_delta_t* tmp_,
                                 gpu_delta_t* best_, gpu_delta_t* time_at_dest_, unsigned short kUnreachable,
                                 uint16_t* lb_, gpu_delta_t* round_times_, uint32_t column_count_round_times_,
                                 uint32_t* station_mark_, uint32_t* prev_station_mark_,
                                 std::uint32_t const n_locations,
                                 gpu_vector_map<gpu_location_idx_t, gpu_u8_minutes> const* transfer_time,
                                 gpu_vecvec<gpu_location_idx_t, nigiri::gpu_footpath> const* gpu_footpaths_out,
                                 gpu_vecvec<gpu_location_idx_t, nigiri::gpu_footpath> const* gpu_footpaths_in_,
                                 gpu_raptor_stats* stats_){
  if(k==2){
    printf("GPU error!!!");
  }
  auto const global_t_id = get_global_thread_id();
  auto const global_stride = get_global_stride();
  for(auto l_idx = global_t_id;
       l_idx < n_locations; l_idx += global_stride){

    printf("TRANSFERS2");
    if(!marked(prev_station_mark_, l_idx)){
      printf("TRANSFERS3");
      continue;
    }
    printf("TRANSFERS4");
    auto const is_dest = is_dest_[l_idx];
    auto const tt = (dist_to_end_size_==0 && is_dest)
        ? 0 : dir<SearchDir>((*transfer_time)[gpu_location_idx_t{l_idx}]).count();
    const auto fp_target_time =
        static_cast<gpu_delta_t>(tmp_[l_idx] + tt);
    if(is_better<SearchDir>(fp_target_time, best_[l_idx])
        && is_better<SearchDir>(fp_target_time, time_at_dest_[k])){
      if(lb_[l_idx] == kUnreachable
          || !is_better<SearchDir>(fp_target_time + dir<SearchDir>(lb_[l_idx]), time_at_dest_[k])){
        ++stats_[l_idx%32].fp_update_prevented_by_lower_bound_;
        continue;
      }
      ++stats_[l_idx%32].n_earliest_arrival_updated_by_footpath_;
      printf("column_count_round_times_: %d",column_count_round_times_);
      if(fp_target_time==435 || fp_target_time==495 || fp_target_time==465){
        printf("GPU Fehler transfers k=%d, l_idx=%d fptt=%d at %d", k, l_idx, fp_target_time, k * column_count_round_times_ + l_idx);
      }
      auto updated = atomicExch(reinterpret_cast<int*>(
                         &round_times_[k * column_count_round_times_ + l_idx]), fp_target_time);
      best_[l_idx] = fp_target_time;
      if(updated != round_times_[k * column_count_round_times_ + l_idx]){
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
                                 gpu_delta_t* tmp_, gpu_delta_t* best_,
                                 uint16_t const* lb_, uint32_t* station_mark_, gpu_delta_t* time_at_dest_,
                                 bool const* is_dest_, gpu_delta_t* round_times_,
                                 uint32_t column_count_round_times_,
                                 std::uint32_t const n_locations,
                                 gpu_vector_map<gpu_location_idx_t, gpu_u8_minutes> const* transfer_time,
                                gpu_vecvec<gpu_location_idx_t, nigiri::gpu_footpath> const* gpu_footpaths_in,
                                 gpu_vecvec<gpu_location_idx_t, nigiri::gpu_footpath> const* gpu_footpaths_out,
                                 gpu_raptor_stats* stats_){
  if(get_global_thread_id()== 0)printf("update_footpaths: k:%d",k);
  auto const global_t_id = get_global_thread_id();
  auto const global_stride = get_global_stride();
  for(auto idx = global_t_id;
       idx < n_locations; idx += global_stride){
    if(!marked(prev_station_mark_, idx)){
      continue;
    }
    auto const l_idx = gpu_location_idx_t{idx};
    auto const& fps = (SearchDir == gpu_direction::kForward)
         ? gpu_footpaths_out[prf_idx][l_idx]
           : gpu_footpaths_in[prf_idx][l_idx];
    for(auto const& fp: fps){
      ++stats_[idx%32].n_footpaths_visited_;
      auto const target = gpu_to_idx(gpu_location_idx_t{fp.target_});
      auto const fp_target_time =
          gpu_clamp(tmp_[idx] + dir<SearchDir>(fp.duration()).count());

      if(is_better<SearchDir>(fp_target_time, best_[target])
          && is_better<SearchDir>(fp_target_time, time_at_dest_[k])){
        auto const lower_bound = lb_[gpu_to_idx(gpu_location_idx_t{fp.target_})];
        if(lower_bound == kUnreachable
            || !is_better<SearchDir>(fp_target_time + dir<SearchDir>(lower_bound), time_at_dest_[k])){
          ++stats_[idx%32].fp_update_prevented_by_lower_bound_;
          continue;
        }
      }
      ++stats_[idx%32].n_earliest_arrival_updated_by_footpath_;
      printf("GPU footpaths update_arrivals %d ,value: %d", (k) * column_count_round_times_ + gpu_to_idx(gpu_location_idx_t{fp.target_}),fp_target_time);
      if(fp_target_time==435 || fp_target_time==495 || fp_target_time==465){
        printf("GPU Fehler footpaths k=%d, l_idx=%d fptt=%d", k, idx, fp_target_time);
      }
      auto updated = atomicExch(
          reinterpret_cast<int*>(
              &round_times_[k * column_count_round_times_ +
                            gpu_to_idx(gpu_location_idx_t{fp.target_})]), fp_target_time);
      best_[gpu_to_idx(gpu_location_idx_t{fp.target_})] = fp_target_time;
      if(updated !=round_times_[k * column_count_round_times_ + gpu_to_idx(gpu_location_idx_t{fp.target_})]){
        mark(station_mark_, gpu_to_idx(gpu_location_idx_t{fp.target_}));
      }
      if(is_dest_[gpu_to_idx(gpu_location_idx_t{fp.target_})]){
        update_time_at_dest<SearchDir, Rt>(k, fp_target_time, time_at_dest_);
      }
    }
  }

}

template <gpu_direction SearchDir, bool Rt>
__device__ void update_intermodal_footpaths(unsigned const k, std::uint32_t const n_locations,
                                            uint16_t* dist_to_end_, uint32_t dist_to_end_size_, uint32_t* station_mark_,
                                            uint32_t* prev_station_mark_, gpu_delta_t* time_at_dest_,
                                            unsigned short kUnreachable, gpu_location_idx_t* gpu_kIntermodalTarget,
                                            gpu_delta_t* best_, gpu_delta_t* tmp_,
                                            gpu_delta_t* round_times_, uint32_t column_count_round_times_){
  if(dist_to_end_size_==0){
    return;
  }
  auto const global_t_id = get_global_thread_id();
  auto const global_stride = get_global_stride();
  if(global_t_id == 0){
    for(auto idx = 0U; idx != n_locations; ++idx){
      if((marked(prev_station_mark_, idx) || marked(station_mark_, idx)) && dist_to_end_[idx] != kUnreachable){
        auto const end_time = gpu_clamp(get_best<SearchDir>(best_[idx], tmp_[idx]) + dir<SearchDir>(dist_to_end_[idx]));
        printf("GPU intermodal_footpaths end_time: %d, gpu_kIntermodalTarget: %d",end_time,best_[gpu_to_idx(*gpu_kIntermodalTarget)]);
        if(is_better<SearchDir>(end_time, best_[gpu_to_idx(*gpu_kIntermodalTarget)])){
          printf("GPU intermodal_footpaths update_arrivals %d ,value: %d", (k) * column_count_round_times_ + gpu_kIntermodalTarget->v_,end_time);
          if(end_time==435 || end_time==495 || end_time==465){
            printf("GPU Fehler intermodalfootpaths k=%d, l_idx=%d et=%d", k, idx, end_time);
          }
          auto updated = atomicExch(reinterpret_cast<int*>(
                             &round_times_[k * column_count_round_times_ +
                                           gpu_kIntermodalTarget->v_]), end_time);
          best_[gpu_to_idx(*gpu_kIntermodalTarget)] = end_time;
          update_time_at_dest<SearchDir, Rt>(k, end_time, time_at_dest_);
        }
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
                             gpu_delta const* route_stop_times,
                             gpu_vecvec<gpu_route_idx_t,gpu_value_type> const* route_location_seq,
                             gpu_vecvec<gpu_location_idx_t , gpu_route_idx_t> const* location_routes,
                             std::uint32_t const n_locations,
                             std::uint32_t const n_routes,
                             gpu_vector_map<gpu_route_idx_t,gpu_interval<std::uint32_t>> const* route_stop_time_ranges,
                             gpu_vector_map<gpu_route_idx_t,gpu_interval<gpu_transport_idx_t >> const* route_transport_ranges,
                             gpu_vector_map<gpu_bitfield_idx_t, gpu_bitfield> const* bitfields,
                             gpu_vector_map<gpu_transport_idx_t,gpu_bitfield_idx_t> const* transport_traffic_days,
                             gpu_interval<gpu_sys_days> const* date_range,
                             gpu_vector_map<gpu_location_idx_t, gpu_u8_minutes> const* transfer_time,
                            gpu_vecvec<gpu_location_idx_t, nigiri::gpu_footpath> const* gpu_footpaths_in,
                             gpu_vecvec<gpu_location_idx_t, nigiri::gpu_footpath> const* gpu_footpaths_out,
                             gpu_vector_map<gpu_route_idx_t, gpu_clasz> const* route_clasz){

  assert((*route_location_seq).data_.el_ != nullptr);
  assert((*route_location_seq).bucket_starts_.el_ != nullptr);
  assert((*location_routes).data_.el_ != nullptr);
  assert((*location_routes).bucket_starts_.el_ != nullptr);
  assert((*route_stop_time_ranges).el_ != nullptr);
  assert((*route_transport_ranges).el_ != nullptr);
  assert((*bitfields).el_ != nullptr);
  assert((*transport_traffic_days).el_ != nullptr);
  assert((*transfer_time).el_ != nullptr);
  assert((*gpu_footpaths_in).data_.el_ != nullptr);
  assert((*gpu_footpaths_in).bucket_starts_.el_ != nullptr);
  assert((*gpu_footpaths_out).data_.el_ != nullptr);
  assert((*gpu_footpaths_out).bucket_starts_.el_ != nullptr);
  assert((*route_clasz).el_ != nullptr);
  // update_time_at_dest für alle locations
  if(get_global_thread_id() ==0){
  printf("raptor_round: %d \n",k);
  }
  if(get_global_thread_id() == 0)printf("station marked? %d \n",marked(station_mark_,0)) ;

  auto const global_t_id = get_global_thread_id();
  auto const global_stride = get_global_stride();
  //TODO sicher, dass man über n_locations iterieren muss? -> aufpassen, dass round_times nicht out of range zugegriffen wird
  for(auto idx = global_t_id; idx < n_locations; idx += global_stride){
    auto test =round_times_[(k) * column_count_round_times_ +idx];
    auto test2 = best_[idx];
    best_[global_t_id] =get_best<SearchDir>(test, test2);
    if(is_dest_[idx]){
      update_time_at_dest<SearchDir, Rt>(k, best_[global_t_id], time_at_dest_);
    }
  }

  this_grid().sync();

  // für jede location & für jede location_route state_.route_mark_
  if(get_global_thread_id()==0){
    atomicAnd(reinterpret_cast<int*>(any_station_marked_),0);
  }
  this_grid().sync();
  if(get_global_thread_id() == 0){
    for (int i = 0; i< n_locations; i++) {
      //printf("nlocation: %d", station_mark_[i]);
    }
  }
  if(get_global_thread_id() == 0 && k == 3){
    for (int i = 0; i< n_locations; ++i){
      printf("GPU stations_marked bevor convert round 2: %d, stelle i: %d", station_mark_[i],i);
    }
  }
  convert_station_to_route_marks<SearchDir, Rt>(station_mark_, route_mark_,
                                 any_station_marked_, location_routes, n_locations);
  if(get_global_thread_id() == 0){
    for (int i = 0; i< n_routes; i++) {
      //printf("nroutes: %d", route_mark_[i]);
    }
  }
  this_grid().sync();

  if(!*any_station_marked_){
    return;
  }

  if(get_global_thread_id()==0){
    // swap
    cuda::std::swap(prev_station_mark_,station_mark_);
    // fill

    for(int j = 0; j < ((n_locations/32)+1); j++){
      station_mark_[j] = 0x0000;
    }
  }

  if(global_t_id == 0)printf("waiting %d\n",k);
  this_grid().sync();

  if(get_global_thread_id() == 0){
    for (int i = 0; i < n_locations; i++) {
      //printf("nprev_stations0: %d", prev_station_mark_[i]);
    }
  }
  //ToDo: ich hab return raus gezogen da sonst endlosschleife...
  if(global_t_id == 0)printf("waiting finished %d\n",k);
  // loop_routes mit true oder false
  // any_station_marked soll nur einmal gesetzt werden, aber loop_routes soll mit allen threads durchlaufen werden?

  if (get_global_thread_id() == 0) {
         printf("Before loop_routes: any_station_marked_: %d\n", *any_station_marked_);
  }

  if(get_global_thread_id() == 0 && k == 2){
    for (int i = 0; i< n_routes; ++i){
      printf("GPU route_marked bevor round 3: %d, stelle i: %d", route_mark_[i],i);
    }
    for(int j = 0; j<row_count_round_times_;++j) {
      for (int i = 0; i < column_count_round_times_; ++i) {
        printf("round_time GPU bevor 2 Runde: %d", round_times_[j*column_count_round_times_+i]);
      }
    }
  }
  if(get_global_thread_id() == 0) printf("rounde k: %d",k);
  (allowed_claszes_ == 0xffff)? loop_routes<SearchDir, Rt, false>(k, any_station_marked_, route_mark_, &allowed_claszes_,
                                                             stats_, kMaxTravelTimeTicks_, prev_station_mark_, best_,
                                                             round_times_, column_count_round_times_, tmp_, lb_, n_days_,
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
                                                                 transfer_time,
                                                                 gpu_footpaths_in,
                                                                 gpu_footpaths_out,
                                                                 route_clasz)
                           : loop_routes<SearchDir, Rt, true>(k, any_station_marked_, route_mark_, &allowed_claszes_,
                                                             stats_, kMaxTravelTimeTicks_, prev_station_mark_, best_,
                                                             round_times_, column_count_round_times_, tmp_, lb_, n_days_,
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
                                                                transfer_time,
                                                                gpu_footpaths_in,
                                                                gpu_footpaths_out,
                                                                route_clasz);
  this_grid().sync();
  if (get_global_thread_id() == 0) {
    printf("raptor_round: any_station_marked_ after loop_routes: %d,k: %d\n",
           *any_station_marked_,k);
  }
  if(!*any_station_marked_){
    printf("Test GPU round 2 break");
    return;
  }
  //ToDo: ICH habe mal das return raus geschoben weil warum sollte nur der 0 thread returnen

  if(get_global_thread_id() == 0){
    for (int i = 0; i < n_locations; i++) {
      printf("n_stations: %d", station_mark_[i]);
    }
  }
  if(get_global_thread_id()==0){

    // fill
    for(int i = 0; i < ((n_routes/32)+1); i++){
      route_mark_[i] = 0x0000;
    }
    // swap
    cuda::std::swap(prev_station_mark_,station_mark_);
    // fill
    for(int j = 0; j < ((n_locations/32)+1); j++){
      station_mark_[j] = 0x0000; // soll es auf false setzen
    }

  }
  if(get_global_thread_id() == 0){
    for (int i = 0; i < n_locations; i++) {
      printf("nprev_stations1: %d", prev_station_mark_[i]);
    }
  }
  this_grid().sync();
  //TODO: warum hier eigentlich noch ein check?


  // update_transfers
  update_transfers<SearchDir, Rt>(k, is_dest_, dist_to_end_, dist_to_end_size_,
                   tmp_, best_, time_at_dest_, kUnreachable, lb_, round_times_,
                                  column_count_round_times_, station_mark_, prev_station_mark_,
                   n_locations,
                                  transfer_time,
                                  gpu_footpaths_in,
                                  gpu_footpaths_out, stats_);
  this_grid().sync();



  // update_footpaths
  update_footpaths<SearchDir, Rt>(k, prf_idx, kUnreachable, prev_station_mark_,
                   tmp_, best_, lb_, station_mark_, time_at_dest_,
                   is_dest_, round_times_, column_count_round_times_,
                   n_locations,
                                  transfer_time,
                                  gpu_footpaths_in,
                                  gpu_footpaths_out, stats_);
  this_grid().sync();
  if(get_global_thread_id() == 0 && k==2)
    for(int j = 0; j<row_count_round_times_;++j) {
      for (int i = 0; i < column_count_round_times_; ++i) {
        printf("round_time GPU after update_footpaths: %d", round_times_[j*row_count_round_times_+i]);
      }
    }
  // update_intermodal_footpaths
  update_intermodal_footpaths<SearchDir, Rt>(k, n_locations, dist_to_end_, dist_to_end_size_, station_mark_,
                             prev_station_mark_, time_at_dest_, kUnreachable,
                             gpu_kIntermodalTarget, best_, tmp_, round_times_, column_count_round_times_);

}

template <gpu_direction SearchDir, bool Rt>
__device__ void init_arrivals(gpu_unixtime_t const worst_time_at_dest,
                              gpu_day_idx_t* base_, gpu_delta_t* time_at_dest,
                              gpu_delta const* route_stop_times,
                              gpu_vector_map<gpu_route_idx_t,gpu_interval<gpu_transport_idx_t >> const* route_transport_ranges,
                              gpu_interval<gpu_sys_days> const* date_range){
  auto const t_id = get_global_thread_id();
  if(t_id < gpu_kMaxTransfers+1){
    time_at_dest[t_id] = get_best<SearchDir>(unix_to_gpu_delta(base(base_,date_range), worst_time_at_dest), time_at_dest[t_id]);
  }

}

// größten Teil von raptor.execute() wird hierdrin ausgeführt
// kernel muss sich außerhalb der gpu_raptor Klasse befinden
template <gpu_direction SearchDir, bool Rt>
__global__ void gpu_raptor_kernel(gpu_unixtime_t* start_time,
                                  uint8_t max_transfers,
                                  gpu_unixtime_t* worst_time_at_dest,
                                  gpu_profile_idx_t* prf_idx,
                                  gpu_clasz_mask_t* allowed_claszes,
                                  std::uint16_t* dist_to_end,
                                  std::uint32_t* dist_to_end_size,
                                  gpu_day_idx_t* base,
                                  bool* is_dest,
                                  std::uint16_t* lb,
                                  int* n_days,
                                  std::uint16_t* kUnreachable,
                                  gpu_location_idx_t* kIntermodalTarget,
                                  short* kMaxTravelTimeTicks,
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
                                  gpu_delta const* route_stop_times,
                                  gpu_vecvec<gpu_route_idx_t,gpu_value_type> const* route_location_seq,
                                  gpu_vecvec<gpu_location_idx_t , gpu_route_idx_t> const* location_routes,
                                  std::uint32_t const n_locations,
                                  std::uint32_t const n_routes,
                                  gpu_vector_map<gpu_route_idx_t,gpu_interval<std::uint32_t>> const* route_stop_time_ranges,
                                  gpu_vector_map<gpu_route_idx_t,gpu_interval<gpu_transport_idx_t >> const* route_transport_ranges,
                                  gpu_vector_map<gpu_bitfield_idx_t, gpu_bitfield> const* bitfields,
                                  gpu_vector_map<gpu_transport_idx_t,gpu_bitfield_idx_t> const* transport_traffic_days,
                                  gpu_interval<gpu_sys_days> const* date_range,
                                  gpu_vector_map<gpu_location_idx_t, gpu_u8_minutes> const* transfer_time,
                                  gpu_vecvec<gpu_location_idx_t, nigiri::gpu_footpath> const* gpu_footpaths_in,
                                  gpu_vecvec<gpu_location_idx_t, nigiri::gpu_footpath> const* gpu_footpaths_out,
                                  gpu_vector_map<gpu_route_idx_t, gpu_clasz> const* route_clasz){
  auto const end_k =
      get_smaller(max_transfers, gpu_kMaxTransfers) + 1U;
  // 1. Initialisierung
  printf("gpu kernel from %d",(*route_stop_time_ranges)[gpu_route_idx_t{1}].from_);
  assert((*route_location_seq).data_.el_ != nullptr);
  assert((*route_location_seq).bucket_starts_.el_ != nullptr);
  assert((*location_routes).data_.el_ != nullptr);
  assert((*location_routes).bucket_starts_.el_ != nullptr);
  assert((*route_stop_time_ranges).el_ != nullptr);
  assert((*route_transport_ranges).el_ != nullptr);
  assert((*bitfields).el_ != nullptr);
  assert((*transport_traffic_days).el_ != nullptr);
  assert((*transfer_time).el_ != nullptr);
  assert((*gpu_footpaths_in).data_.el_ != nullptr);
  assert((*gpu_footpaths_in).bucket_starts_.el_ != nullptr);
  assert((*gpu_footpaths_out).data_.el_ != nullptr);
  assert((*gpu_footpaths_out).bucket_starts_.el_ != nullptr);
  assert((*route_clasz).el_ != nullptr);
  init_arrivals<SearchDir, Rt>(*worst_time_at_dest, base,
                time_at_dest, route_stop_times,route_transport_ranges,date_range);

  this_grid().sync();
  //++stats[get_global_thread_id()>>5].n_routes_visited_; TODO: so ist out of range
  // ausprobieren, ob folgende daten noch weiter entschachtelt werden müssen
  if(get_global_thread_id() == 0) printf("is marked? %d \n", marked(station_mark,0));
  //locations->gpu_footpaths_out_[1][1]; // hiervon sind auch gpu_footpaths_out und transfer_time betroffem
  // 2. Update Routes

  for (auto k = 1U; k != end_k; ++k) { // diese Schleife bleibt, da alle Threads in jede Runde gehen
    // Resultate aus lezter Runde von device in variable speichern?  //TODO: typen von kIntermodalTarget und dist_to_end_size falsch???
    if(k!= 1 && (!(*any_station_marked))){
      break;
    }
    raptor_round<SearchDir, Rt>(k, *prf_idx, base, *allowed_claszes,
                 dist_to_end, *dist_to_end_size, is_dest, lb, *n_days,
                 time_at_dest, any_station_marked, route_mark,
                 station_mark, best,
                 *kUnreachable, prev_station_mark,
                 round_times, tmp,
                                column_count_round_times,
                 column_count_round_times,
                 kIntermodalTarget, stats, kMaxTravelTimeTicks,route_stop_times,
                 route_location_seq,
                                location_routes,
                                n_locations,
                                n_routes,
                                route_stop_time_ranges,
                                route_transport_ranges,
                                bitfields,
                                transport_traffic_days,
                                date_range,
                                transfer_time,
                                gpu_footpaths_in,
                                gpu_footpaths_out,
                                route_clasz);
    this_grid().sync();
  }
  this_grid().sync();
  if (get_global_thread_id() == 0){
    for(int j = 0; j<row_count_round_times;++j) {
      for (int i = 0; i < n_locations; ++i) {
        printf("round_time GPU at end of kernel: %d", round_times[j*row_count_round_times+i]);
      }
    }
  }
  if(get_global_thread_id() == 0)printf("Testerin2");
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
  auto dist_to_end_size = dist_to_dest.size();
  std::cerr << "copy_to_device start" << dist_to_end_size << std::endl;

  allowed_claszes_ = nullptr;
  CUDA_COPY_TO_DEVICE(gpu_clasz_mask_t, allowed_claszes_, &allowed_claszes, 1);
  dist_to_end_ = nullptr;
  CUDA_COPY_TO_DEVICE(std::uint16_t, dist_to_end_, dist_to_dest.data(),
                      dist_to_dest.size());
  dist_to_end_size_ = nullptr;
  CUDA_COPY_TO_DEVICE(std::uint32_t, dist_to_end_size_, &dist_to_end_size, 1);
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
  cudaLaunchCooperativeKernel(kernel_func, device.grid_, device.threads_per_block_, args, 0, s);
  cudaDeviceSynchronize();
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
      sizeof(uint32_t)*((mem->device_.n_locations_/32)+1), cudaMemcpyDeviceToHost, s);
  cuda_check();
  cudaMemcpyAsync(
      mem->host_.prev_station_mark_.data(), mem->device_.prev_station_mark_,
      sizeof(uint32_t)*((mem->device_.n_locations_/32)+1), cudaMemcpyDeviceToHost, s);
  cuda_check();
  cudaMemcpyAsync(
      mem->host_.route_mark_.data(), mem->device_.route_mark_,
      sizeof(uint32_t)*((mem->device_.n_routes_/32)+1), cudaMemcpyDeviceToHost, s);
  cuda_check(); //TODO: keine ahnung warum hier kein cuda check geht
}
void copy_back(mem* mem){
  std::cerr << "Test gpu_raptor::launch_kernel() bevor proc" << std::endl;
  cuda_check();
  //cuda_sync_stream(mem->context_.proc_stream_);
  std::cerr << "Test gpu_raptor::launch_kernel() bevor transfer" << std::endl;
  fetch_arrivals_async(mem,mem->context_.transfer_stream_);
  cuda_check();
  std::cerr << "Test gpu_raptor::launch_kernel() sync_stream" << std::endl;
  //cuda_sync_stream(mem->context_.transfer_stream_);

}

void add_start_gpu(gpu_location_idx_t const l, gpu_unixtime_t const t,mem* mem_,gpu_timetable const* gtt_,gpu_day_idx_t base_,short const kInvalid){
  std::cerr << "Test gpu_raptor::add_start_gpu() start" << std::endl;
  trace_upd("adding start {}: {}\n", location{gtt_, l}, t);
  std::vector<gpu_delta_t> best_new(mem_->device_.n_locations_,kInvalid);
  std::cerr << "Test gpu_raptor::add_start_gpu() 1" << std::endl;
  std::vector<gpu_delta_t> round_times_new((mem_->device_.column_count_round_times_*mem_->device_.row_count_round_times_),kInvalid);
  std::cerr << "Test gpu_raptor::add_start_gpu() 1.5" << std::endl;
  best_new[gpu_to_idx(l)] = unix_to_gpu_delta(cpu_base(gtt_,base_), t);
  std::cerr << "Test gpu_raptor::add_start_gpu() 2" << std::endl;
  //TODO: hier fehler da base nur auf device funktioniert!
  round_times_new[0U*mem_->device_.column_count_round_times_+ gpu_to_idx(l)] = unix_to_gpu_delta(cpu_base(gtt_,base_), t);

  //TODO: fix station_mark ist kein bool!
  std::vector<uint32_t> gpu_station_mark((( mem_->device_.n_locations_/32)+1),0);
  unsigned int const store_idx = (gpu_to_idx(l) >> 5);  // divide by 32
  unsigned int const mask = 1 << (gpu_to_idx(l) % 32);
  gpu_station_mark[store_idx] |= mask; //TODO: das ist bullshit so makiert man keine station also mit nur l
  unsigned int const store_idx2 = (gpu_to_idx(l) >> 5);  // divide by 32
  unsigned int const val = gpu_station_mark[store_idx];
  unsigned int const mask2 = 1 << (gpu_to_idx(l) % 32);
  auto testerina = (bool)(val & mask2);
  std::cerr << "Test if marked: " << testerina << "l: "<<gpu_to_idx(l)<< std::endl;
  std::cerr << "Test gpu_raptor::add_start_gpu() 3" << std::endl;
  cudaMemcpy(mem_->device_.best_, best_new.data(), mem_->device_.n_locations_*sizeof(gpu_delta_t), cudaMemcpyHostToDevice);
  cuda_check();
  cudaMemcpy(mem_->device_.round_times_, round_times_new.data(), round_times_new.size()*sizeof(gpu_delta_t), cudaMemcpyHostToDevice);
  cuda_check();
  cudaMemcpy(mem_->device_.station_mark_, gpu_station_mark.data(), ((mem_->device_.n_locations_/32)+1)*sizeof(uint32_t), cudaMemcpyHostToDevice);
  cuda_check();
  std::cerr << "Test gpu_raptor::add_start_gpu() end" << std::endl;
}
std::unique_ptr<mem> gpu_mem(
    std::vector<gpu_delta_t>& tmp,
    std::vector<gpu_delta_t>& best,
    std::vector<bool>& station_mark,
    std::vector<bool>& prev_station_mark,
    std::vector<bool>& route_mark,
    gpu_direction search_dir,
    gpu_timetable const* gtt){
  std::cerr << "Test gpu_raptor::gpu_mem()" << std::endl;
  short kInvalid = 0;
  if(search_dir == gpu_direction::kForward){
    kInvalid = kInvalidGpuDelta<gpu_direction::kForward>;
  } else{
    kInvalid = kInvalidGpuDelta<gpu_direction::kBackward>;
  }
  std::vector<uint32_t> gpu_station_mark(((gtt->n_locations_/32)+1));
  for (size_t i = 0; i < station_mark.size(); ++i) {
    size_t uint32_i = i / 32;
    size_t bit_i = i % 32;
    if (station_mark[i]) {
      gpu_station_mark[uint32_i] |= (1u << bit_i);
    }
  }
  std::vector<uint32_t> gpu_prev_station_mark(((gtt->n_locations_/32)+1));
  for (size_t i = 0; i < prev_station_mark.size(); ++i) {
    size_t uint32_i = i / 32;
    size_t bit_i = i % 32;
    if (prev_station_mark[i]) {
      gpu_prev_station_mark[uint32_i] |= (1u << bit_i);
    }
  }
  std::vector<uint32_t> gpu_route_mark(((gtt->n_routes_/32)+1));
  for (size_t i = 0; i < route_mark.size(); ++i) {
    size_t uint32_i = i / 32;
    size_t bit_i = i % 32;
    if (route_mark[i]) {
      gpu_route_mark[uint32_i] |= (1u << bit_i);
    }
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
  cudaMemcpy(mem.get()->device_.station_mark_, gpu_station_mark.data(), ((gtt->n_locations_/32)+1) * sizeof(uint32_t), cudaMemcpyHostToDevice);
  cuda_check();
  cudaMemcpy(mem.get()->device_.prev_station_mark_, gpu_prev_station_mark.data(), ((gtt->n_locations_/32)+1) * sizeof(uint32_t), cudaMemcpyHostToDevice);
  cuda_check();
  cudaMemcpy(mem.get()->device_.route_mark_, gpu_route_mark.data(), ((gtt->n_routes_/32)+1) * sizeof(uint32_t), cudaMemcpyHostToDevice);
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
void destroy_copy_to_gpu_args(gpu_unixtime_t* start_time_ptr,
                              gpu_unixtime_t* worst_time_at_dest_ptr,
                              gpu_profile_idx_t* prf_idx_ptr){
  std::cerr << "Test gpu_raptor::launch_kernel() start_time" << std::endl;
  cudaFree(start_time_ptr);
  std::cerr << "Test gpu_raptor::launch_kernel() worst_time" << std::endl;
  cudaFree(worst_time_at_dest_ptr);
  std::cerr << "Test gpu_raptor::launch_kernel() prf" << std::endl;
  cudaFree(prf_idx_ptr);
  //cuda_check();
}