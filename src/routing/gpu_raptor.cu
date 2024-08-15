#include "nigiri/routing/gpu_raptor.cuh"
#include "nigiri/routing/query.h"
#include "nigiri/routing/query.h"
#include "cooperative_groups.h"

using namespace cooperative_groups;

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

template <gpu_direction SearchDir, bool Rt>
__device__ void update_time_at_dest(unsigned const k, gpu_delta_t const t, gpu_delta_t* time_at_dest_){
  for (auto i = k; i < gpu_kMaxTransfers+1; ++i) {
    time_at_dest_[i] = get_best<SearchDir, Rt>(time_at_dest_[i], t);
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
      /*if constexpr (Rt) {
        for (auto const& rt_t :
             rtt_->location_rt_transports_[location_idx_t{i}]) {
          any_marked = true;
          state_.rt_transport_mark_[to_idx(rt_t)] = true;
        }
      }*/
    }
  }
}

template <gpu_direction SearchDir, bool Rt>
void reconstruct(nigiri::routing::query const& q, gpu_journey& j){
  //reconstruct_journey<SearchDir, Rt>(...);
}

template <gpu_direction SearchDir, bool Rt>
__device__ bool update_route_smaller32(unsigned const k, gpu_route_idx_t r,
                                       gpu_timetable* gtt_, raptor_stats* stats_,
                                       uint32_t* prev_station_mark_, gpu_delta_t* best_,
                                       gpu_delta_t* round_times_, gpu_delta_t* tmp_,
                                       uint16_t* lb_, int n_days_,
                                       gpu_delta_t* time_at_dest_,
                                       uint32_t* station_mark_){
  auto const t_id = threadIdx.x;

  gpu_stop_idx_t stop_id = 0xffff;
  uint16_t prev_arrival = 0xffff;
  uint16_t stop_arrival = 0xffff;
  uint16_t stop_departure = 0xffff;

  unsigned leader = (*gtt_).route_location_seq_->size();
  unsigned int active_stop_count = (*gtt_).route_location_seq_->size();

  if(t_id < active_stop_count){

  }

  return false;
}

template <gpu_direction SearchDir, bool Rt>
__device__ bool update_route_bigger32(unsigned const k, gpu_route_idx_t r,
                                      gpu_timetable* gtt_, raptor_stats* stats_,
                                      uint32_t* prev_station_mark_, gpu_delta_t* best_,
                                      gpu_delta_t* round_times_, gpu_delta_t* tmp_,
                                      uint16_t* lb_, int n_days_,
                                      gpu_delta_t* time_at_dest_,
                                      uint32_t* station_mark_){
  return false;
}

template <gpu_direction SearchDir, bool Rt, bool WithClaszFilter>
__device__ bool loop_routes(unsigned const k, bool any_station_marked_,
                            gpu_timetable* gtt_, uint32_t* route_mark_,
                            gpu_clasz_mask_t* allowed_claszes_, raptor_stats* stats_,
                            short* kMaxTravelTimeTicks_, uint32_t* prev_station_mark_,
                            gpu_delta_t* best_,
                            gpu_delta_t* round_times_, gpu_delta_t* tmp_,
                            uint16_t* lb_, int n_days_,
                            gpu_delta_t* time_at_dest_,
                            uint32_t* station_mark_){
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
        auto const as_mask = static_cast<gpu_clasz_mask_t>(1U << static_cast<std::underlying_type_t<gpu_clasz>>(static_cast<nigiri::clasz>((*gtt_->route_clasz_)[r])));
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
                                                      round_times_, tmp_, lb_, n_days_, time_at_dest_, station_mark_);
      }
      else{ // diese Route hat > 32 Stops
        any_station_marked_ |= update_route_bigger32<SearchDir, Rt>(k, r, gtt_, stats_, prev_station_mark_, best_,
                                                     round_times_, tmp_, lb_, n_days_, time_at_dest_, station_mark_);
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
                                 raptor_stats* stats_){
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
      round_times_[k * row_count_round_times_ + l_idx] = fp_target_time;
      best_[l_idx] = fp_target_time;
      mark(station_mark_, l_idx);
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
                                 uint32_t row_count_round_times_, raptor_stats* stats_){
  auto const global_t_id = get_global_thread_id();
  auto const global_stride = get_global_stride();
  for(auto idx = global_t_id;
       idx <= *gtt_->n_locations_; idx += global_stride){
    if(!marked(prev_station_mark_, idx)){
      continue;
    }
    auto const l_idx = gpu_location_idx_t{idx};
    auto const& fps = (SearchDir == gpu_direction::kForward)
         ? gtt_->locations_->gpu_footpaths_out_.data_[prf_idx][l_idx.v_]
           : gtt_->locations_->gpu_footpaths_in_.data_[prf_idx][l_idx.v_];
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
      ++stats_[idx>>5].n_earliest_arrival_updated_by_footpath_; //
      round_times_[k * row_count_round_times_ + gpu_to_idx(gpu_location_idx_t{fp.target_})] = fp_target_time;
      best_[gpu_to_idx(gpu_location_idx_t{fp.target_})] = fp_target_time;
      mark(station_mark_, gpu_to_idx(gpu_location_idx_t{fp.target_}));
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
        round_times_[k*row_count_round_times_ + gpu_kIntermodalTarget->v_] = end_time;
        (*best_)[gpu_kIntermodalTarget] = end_time;
        update_time_at_dest(k, end_time, time_at_dest_);
      }
    }
  }
}

/*
template <gpu_direction SearchDir, bool Rt>
__device__ gpu_transport get_earliest_transport(unsigned const k, gpu_route_idx_t const r,
                                                gpu_stop_idx_t const stop_idx,
                                                gpu_day_idx_t const day_at_stop,
                                                gpu_minutes_after_midnight_t const mam_at_stop,
                                                gpu_location_idx_t const l,
                                                int n_days_, gpu_timetable* gtt_,
                                                gpu_direction search_dir_,
                                                raptor_stats* stats_, short* kMaxTravelTimeTicks_){
  ++stats_[l.v_>>5].n_earliest_trip_calls_;
  auto const n_days_to_iterate = get_smaller(*kMaxTravelTimeTicks_/1440 +1,
                                          (SearchDir == gpu_direction::kForward) ?
                                           n_days_ - as_int(day_at_stop) : as_int(day_at_stop)+1);
  auto const event_times =
      gtt_->event_times_at_stop(r, stop_idx, (SearchDir == gpu_direction::kForward) ?
                                              gpu_event_type::kDep : gpu_event_type::kArr);
  auto const seek_first_day = [&]() {
    return linear_lb(get_begin_it<decltype(event_times), SearchDir>(event_times),
                     get_end_it<std::span<gpu_delta>, SearchDir>(event_times), mam_at_stop,
                     [&](gpu_delta const a, gpu_minutes_after_midnight_t const b) {
                       return is_better<SearchDir, Rt>(a.mam_, b.count()); // anders mit gpu_delta umgehen
                     });
  };

  // for Schleife über n_days_to_iterate
}
 */

template <gpu_direction SearchDir, bool Rt, std::size_t Size>
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

template <gpu_direction SearchDir, bool Rt>
__device__ void raptor_round(unsigned const k, gpu_profile_idx_t const prf_idx,
                             gpu_timetable* gtt_, gpu_strong<uint16_t, _day_idx> base_,
                             gpu_clasz_mask_t allowed_claszes_, uint16_t* dist_to_end_,
                             uint32_t dist_to_end_size_,
                             bool* is_dest_, uint16_t* lb_, int n_days_,
                             gpu_delta_t* time_at_dest_,
                             bool any_station_marked_, uint32_t* route_mark_,
                             uint32_t* station_mark_, gpu_delta_t* best_,
                             unsigned short kUnreachable, uint32_t* prev_station_mark_,
                             gpu_delta_t* round_times_, gpu_delta_t* tmp_,
                             uint32_t size_best_, uint32_t size_tmp_,
                             uint32_t row_count_round_times_,
                             uint32_t column_count_round_times_,
                             uint32_t size_route_mark_, uint32_t size_station_mark_,
                             gpu_location_idx_t* gpu_kIntermodalTarget,
                             raptor_stats stats_, short* kMaxTravelTimeTicks_){

  // update_time_at_dest für alle locations
  auto const global_t_id = get_global_thread_id();
  auto const global_stride = get_global_stride();
  //TODO sicher, dass man über n_locations iterieren muss? -> aufpassen, dass round_times nicht out of range zugegriffen wird
  for(auto idx = global_t_id; idx < *gtt_->n_locations_; idx += global_stride){
    best_[global_t_id] = get_best<SearchDir, Rt>
        (round_times_[k*row_count_round_times_+idx], best_[idx]);
    if(is_dest_[idx]){
      update_time_at_dest<SearchDir, Rt>(k, best_[global_t_id], time_at_dest_);
    }
  }
  this_grid().sync();

  // für jede location & für jede location_route state_.route_mark_
  if(get_global_thread_id()==0){
    any_station_marked_ = false;
  }
  convert_station_to_route_marks<SearchDir, Rt>(station_mark_, route_mark_,
                                                  any_station_marked_, gtt_);
  this_grid().sync();

  if(get_global_thread_id()==0){
    if(!any_station_marked_){
      return;
    }
    // swap
    uint32_t const size = size_station_mark_;
    uint32_t dummy_marks[size];
    for(int i=0; i<size_station_mark_; i++){
      dummy_marks[i] = station_mark_[i];
      station_mark_[i] = prev_station_mark_[i];
      prev_station_mark_[i] = station_mark_[i];
    }
    // fill
    for(int j = 0; j < size_station_mark_; j++){
      station_mark_[j] = 0xFFFF;
    }
  }
  this_grid().sync();
  // loop_routes mit true oder false
  // any_station_marked soll nur einmal gesetzt werden, aber loop_routes soll mit allen threads durchlaufen werden?
  any_station_marked_ = (allowed_claszes_ = 0xffff)
                         ? loop_routes<SearchDir, Rt, false>(k, any_station_marked_, gtt_, route_mark_, allowed_claszes_,
                                                             stats_, kMaxTravelTimeTicks_, prev_station_mark_, best_,
                                                             round_times_, tmp_, lb_, n_days_, time_at_dest_, station_mark_)
                           : loop_routes<SearchDir, Rt,true>(k, any_station_marked_, gtt_, route_mark_, allowed_claszes_,
                                                             stats_, kMaxTravelTimeTicks_, prev_station_mark_, best_,
                                                             round_times_, tmp_, lb_, n_days_, time_at_dest_, station_mark_);

  this_grid().sync();
  if(get_global_thread_id()==0){
    if(!any_station_marked_){
      return;
    }
    // fill
    for(int i = 0; i<size_route_mark_; i++){
      route_mark_[i] = 0xFFFF;
    }
    // swap
    uint32_t const size = size_station_mark_;
    uint32_t dummy_marks[size];
    for(int i=0; i<size_station_mark_; i++){
      dummy_marks[i] = station_mark_[i];
      station_mark_[i] = prev_station_mark_[i];
      prev_station_mark_[i] = station_mark_[i];
    }
    // fill
    for(int j = 0; j < size_station_mark_; j++){
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
                                  nigiri::pareto_set<gpu_journey>& results,
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

    // Resultate aus lezter Runde von device in variable speichern?
    raptor_round<SearchDir, Rt>(k, prf_idx, gr.gtt_, *gr.base_, *gr.allowed_claszes_,
                 gr.dist_to_end_, gr.dist_to_end_size_, gr.is_dest_, gr.lb_, *gr.n_days_,
                 gr.mem_->device_.time_at_dest_,
                 *gr.mem_->device_.any_station_marked_, gr.mem_->device_.route_mark_,
                 gr.mem_->device_.station_mark_, gr.mem_->device_.best_,
                 gr.kUnreachable, gr.mem_->device_.prev_station_mark_,
                 gr.mem_->device_.round_times_, gr.mem_->device_.tmp_,
                 gr.mem_->device_.size_best_, gr.mem_->device_.size_tmp_,
                 gr.mem_->device_.row_count_round_times_,
                 gr.mem_->device_.column_count_round_times_,
                 gr.mem_->device_.size_route_mark_,
                 gr.mem_->device_.size_station_mark_,
                 gr.kIntermodalTarget, gr.stats_, gr.kMaxTravelTimeTicks_);
    this_grid().sync();
  }
  this_grid().sync();

  //construct journey

  this_grid().sync();

}


