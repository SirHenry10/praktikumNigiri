#include "nigiri/routing/gpu_raptor.cuh"
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
  for (auto i = k; i < nigiri::routing::kMaxTransfers+1; ++i) {
    time_at_dest_[i] = get_best<SearchDir, Rt>(time_at_dest_[i], t);
  }
}

template <gpu_direction SearchDir, bool Rt>
__device__ void convert_station_to_route_marks(unsigned int* station_marks,
                                               unsigned int* route_marks,
                                               bool* any_station_marked,
                                               gpu_raptor<SearchDir, Rt>& gr) {
  auto const global_t_id = get_global_thread_id();
  auto const global_stride = get_global_stride();
  // anstatt stop_count_ brauchen wir location_routes ?location_idx_{gtt.n_locations}?
  for (uint32_t idx = global_t_id; idx < *gr.gtt_.n_locations_; idx += global_stride) {
    if (marked(station_marks, idx)) {
      if (!*any_station_marked) {
        *any_station_marked = true;
      }
      auto const location_routes = gr.gtt_.location_routes_;
      for (auto const& r :  location_routes[gpu_location_idx_t{idx}]) {
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
void reconstruct(nigiri::routing::query const& q, nigiri::routing::journey& j){
  //reconstruct_journey<SearchDir, Rt>(...);
}

template <gpu_direction SearchDir, bool Rt, bool WithClaszFilter>
__device__ bool loop_routes(unsigned const k, gpu_raptor<SearchDir,Rt>& gr){
  auto const global_t_id = get_global_thread_id();
  auto const global_stride = get_global_stride();
  if(get_global_thread_id()==0){
    gr.mem_->device_.any_station_marked_ = false;
  }
  for(auto r_idx = global_t_id; r_idx <= gr.gtt_->n_routes_; r_idx += global_stride){
    auto const r = gpu_route_idx_t{r_idx};
    if(gr.mem_->device_.route_mark_[r_idx]){
      if constexpr (WithClaszFilter){
        if(!is_allowed(gr.allowed_claszes_, gr.gtt_.route_clasz_[r])){
          continue;
        }
      }
      ++gr.stats_.n_routes_visited_;
      // TODO hier in smaller32 und bigger32 aufteilen? → aber hier geht nur ein thread rein...
      // also sollte vielleicht diese Schleife mit allen auf einmal durchgangen werden???
      gr.mem_->device_.any_station_marked_ |= update_route(k, r, gr);
    }
  }
}

template <gpu_direction SearchDir, bool Rt>
__device__ bool update_route(unsigned const k, gpu_route_idx_t const r, gpu_raptor<SearchDir,Rt>& gr){
  return false;
}

template <gpu_direction SearchDir, bool Rt>
__device__ bool update_route_smaller32(unsigned const k, gpu_raptor<SearchDir,Rt>& gr){
  return false;
}

template <gpu_direction SearchDir, bool Rt>
__device__ bool update_route_bigger32(unsigned const k, gpu_raptor<SearchDir,Rt>& gr){
  return false;
}

template <gpu_direction SearchDir, bool Rt>
__device__ void update_transfers(unsigned const k, gpu_timetable* gtt_, gpu_raptor<SearchDir, Rt>& gr){
  auto const global_t_id = get_global_thread_id();
  auto const global_stride = get_global_stride();
  for(auto l_idx = global_t_id; l_idx <= gr.gtt_.n_locations_; l_idx += global_stride){
    if(!gr.mem_->device_.prev_station_mark){ //PREV_Stations????
      continue;
    }
    auto const is_dest = gr.is_dest_[l_idx];
    // wo sind unsere locations bzw.wie heißen sie?
    auto const transfer_time = (gr.dist_to_end_.empty() && is_dest)
        ? 0 : gr.dir(gr.gtt_.locations_.transfer_time_[gpu_location_idx_t{l_idx}]).count();
    auto const fp_target_time =
        static_cast<gpu_delta>(gr.mem_->device_.tmp_[l_idx] + transfer_time);
    if(gr.is_better(fp_target_time, gr.mem_->device_.best_[l_idx])
        && gr.is_better(fp_target_time, gr.time_at_dest_[k])){
      if(gr.lb_[l_idx] == gr.kUnreachable
          || !gr.is_better(fp_target_time + gr.dir(gr.lb_[l_idx]), gr.time_at_dest_[k])){
        ++gr.stats_.fp_update_prevented_by_lower_bound_;
        continue;
      }

      ++gr.stats_.n_earliest_arrival_updated_by_footpath_;
      gr.mem_->device_.round_times_[k][l_idx] = fp_target_time;
      gr.mem_->device_.best_[l_idx] = fp_target_time;
      gr.mem_->device_.station_mark_[l_idx] = true;
      if(is_dest){
        update_time_at_dest(k, fp_target_time, gr);
      }
    }
  }
}

template <gpu_direction SearchDir, bool Rt>
__device__ void update_footpaths(unsigned const k, gpu_profile_idx_t const prf_idx, gpu_raptor<SearchDir,Rt>& gr){
  auto const global_t_id = get_global_thread_id();
  auto const global_stride = get_global_stride();
  for(auto idx = global_t_id; idx <= gr.gtt_.n_locations_; idx += global_stride){
    if(!gr.mem_->device_.prev_station_mark){//PREV_Stations??
      continue;
    }
    auto const l_idx = gpu_location_idx_t{idx};
    auto const& fps = gr.kFwd
         ? gr.gtt_.locations_.footpaths_out_[prf_idx][l_idx] : gr.gtt_.locations_.footpaths_in_[prf_idx][l_idx];
    for(auto const& fp: fps){
      ++gr.stats_.n_footpaths_visited_;
      auto const target = to_idx(fp.target());
      auto const fp_target_time =
          clamp(gr.mem_->device_.tmp_[idx] + gr.dir(fp.duration()).count());

      if(gr.is_better(fp_target_time, gr.mem_->device_.best_[target])
          && gr.is_better(fp_target_time, gr.time_at_dest_[k])){
        auto const lower_bound = gr.lb_[to_idx(fp.target())];
        if(lower_bound == gr.kUnreachable
            || !gr.is_better(fp_target_time + gr.dir(lower_bound), gr.time_at_dest_[k])){
          ++gr.stats_.fp_update_prevented_by_lower_bound_;
          continue;
        }
      }
      ++gr.stats_.n_earliest_arrival_updated_by_footpath_;
      gr.mem_->device_.round_times_[k][to_idx(fp.target())] = fp_target_time;
      gr.mem_->device_.best_[to_idx(fp.target())] = fp_target_time;
      gr.mem_->device_.station_mark_[to_idx(fp.target())] = true;
      if(gr.is_dest_[to_idx(fp.target())]){
        update_time_at_dest(k, fp_target_time, gr);
      }
    }
  }

}

template <gpu_direction SearchDir, bool Rt>
__device__ void update_intermodal_footpaths(unsigned const k, gpu_raptor<SearchDir,Rt>& gr){
  if(get_global_thread_id()==0 && gr.dist_to_end_.empty()){
    return;
  }
  auto const global_t_id = get_global_thread_id();
  auto const global_stride = get_global_stride();
  for(auto idx = global_t_id; idx <= gr.gtt_.n_locations_; idx += global_stride){
    if((gr.mem_->device_ || gr.mem_->device_.station_mark_[idx]) && gr.dist_to_end_[idx] != gr.kUnreachable){
      auto const end_time = clamp(gr.get_best(gr.mem_->device_.best_[idx], gr.mem_->device_.tmp_[idx]) + gr.dir(gr.dist_to_end_[idx]));
      if(gr.is_better(end_time, gr.mem_->device_.best_[gr.kIntermodalTarget])){
        gr.mem_->device_.round_times_[k][gr.kIntermodalTarget] = end_time;
        gr.mem_->device_.best_[gr.IntermodalTarget] = end_time;
        update_time_at_dest(k, end_time, gr);
      }
    }
  }
}

template <gpu_direction SearchDir, bool Rt>
__device__ gpu_transport get_earliest_transport(unsigned const k,
                                       gpu_route_idx_t const r,
                                       gpu_stop_idx_t const stop_idx,
                                       gpu_day_idx_t const day_at_stop,
                                       gpu_minutes_after_midnight_t const mam_at_stop,
                                       gpu_location_idx_t const l,
                                       gpu_raptor<SearchDir,Rt>& gr){
  ++gr.stats_.n_earliest_trip_calls_;
  auto const n_days_to_iterate = std::min(gr.gtt_.kMaxTravelTime.count()/1440 +1,
                                          gr.kFwd ? gr.n_days_ - gr.as_int(day_at_stop) : gr.as_int(day_at_stop)+1);
  auto const event_times =
      gr.gtt_.event_times_at_stop(r, stop_idx, gr.kFwd ? gpu_event_type::kDep : gpu_event_type::kArr);
  auto const seek_first_day = [&]() {
    return linear_lb(get_begin_it(event_times), get_end_it(event_times), mam_at_stop,
                     [&](gpu_delta const a, gpu_minutes_after_midnight_t const b) {
                       return gr.is_better(a.mam_, b.count()); // anders mit gpu_delta umgehen
                     });
  };

  // for Schleife über n_days_to_iterate
}

template <gpu_direction SearchDir, bool Rt>
__device__ bool is_transport_active(gpu_transport_idx_t const t, std::size_t const day , gpu_timetable* gtt_)  {
  return gtt_->bitfields_[gtt_->transport_traffic_days_[t]].test(day);
}

template <gpu_direction SearchDir, bool Rt>
__device__ gpu_delta_t time_at_stop(gpu_route_idx_t const r, gpu_transport const t, gpu_stop_idx_t const stop_idx, gpu_event_type const ev_type, gpu_timetable* gtt_, gpu_strong<uint16_t, _day_idx> base_){
  auto const range = gtt_->route_transport_ranges_;
  auto const range2 = *range;
  auto const n_transports = static_cast<unsigned>(range->used_size_);
  auto const route_stop_begin = static_cast<unsigned>(range[r].from_ + n_transports *
                                (stop_idx * 2 - (ev_type==gpu_event_type::kArr ? 1 : 0)));
  return gpu_clamp((as_int(t.day_) - as_int(base_)) * 1440
                   + gtt_->route_stop_times_[route_stop_begin + (gpu_to_idx(t.day_) - gpu_to_idx(range[r].from_))]);
}

template <gpu_direction SearchDir, bool Rt>
__device__ void raptor_round(unsigned const k, gpu_profile_idx_t const prf_idx, gpu_timetable* gtt_,
                             gpu_strong<uint16_t, _day_idx> base_, gpu_clasz_mask_t allowed_claszes_, uint16_t* dist_to_end_,
                             bool* is_dest_, uint16_t* lb_, int n_days_, gpu_direction search_dir_, gpu_delta_t* time_at_dest_,
                             bool any_station_marked_, uint32_t* route_mark_, uint32_t* station_mark_, gpu_delta_t* best_,
                             gpu_delta_t* round_times_, gpu_delta_t* tmp, uint32_t size_best_, uint32_t size_tmp_,
                             uint32_t row_count_round_times_, uint32_t column_count_round_times_,
                             uint32_t size_route_mark_, uint32_t size_station_mark_){

  // update_time_at_dest für alle locations
  auto const global_t_id = get_global_thread_id();
  auto const global_stride = get_global_stride();
  //TODO sicher, dass man über n_locations iterieren muss? -> aufpassen, dass round_times nicht out of range zugegriffen wird
  for(auto idx = global_t_id; idx < *gtt_->n_locations_; idx += global_stride){
    best_[global_t_id] = get_best<search_dir_, Rt>(round_times_[k*row_count_round_times_+idx], best_[idx]);
    if(is_dest_[idx]){
      update_time_at_dest<search_dir_, Rt>(k, best_[global_t_id], time_at_dest_);
    }
  }
  this_grid().sync();

  // für jede location & für jede location_route state_.route_mark_
  if(get_global_thread_id()==0){
    any_station_marked_ = false;
  }
  convert_station_to_route_marks<search_dir_, Rt>(station_mark_, route_mark_, any_station_marked_, gtt_);
  this_grid().sync();

  if(get_global_thread_id()==0){
    if(!any_station_marked_){
      return;
    }
    // SWAP ???
    utl::fill(station_mark_, false); //TODO fill() auf dem device nutzbar machen
  }
  this_grid().sync();
  // loop_routes mit true oder false
  // any_station_marked soll nur einmal gesetzt werden, aber loop_routes soll mit allen threads durchlaufen werden?
  any_station_marked_ = (allowed_claszes_ = nigiri::routing::all_clasz_allowed())
                                              ? loop_routes<false>(k, gr) : loop_routes<false>(k, gr);
  this_grid().sync();
  if(get_global_thread_id()==0){
    if(!any_station_marked_){
      return;
    }
    utl::fill(route_mark_, false);
    // SWAP ???
    utl::fill(station_mark_, false);
  }
  this_grid().sync();

  // update_transfers
  update_transfers(k, gr);
  this_grid().sync();
  // update_footpaths
  update_footpaths(k, prf_idx, gr); //müssen wir gtt hier mit einsetzen??
  this_grid().sync();
  // update_intermodal_footpaths
  update_intermodal_footpaths(k, gr);

}

template <gpu_direction SearchDir, bool Rt>
__device__ void init_arrivals(gpu_delta_t d_worst_at_dest, gpu_unixtime_t const worst_time_at_dest, gpu_day_idx_t* base_, gpu_delta_t* time_at_dest, gpu_timetable* gtt_){
  auto const t_id = get_global_thread_id();

  if(t_id==0){
    d_worst_at_dest = unix_to_gpu_delta(base(gtt_, base_), worst_time_at_dest);
  }

  if(t_id < nigiri::routing::kMaxTransfers+1){
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
                                  nigiri::pareto_set<nigiri::routing::journey>& results,
                                  gpu_raptor<SearchDir,Rt>& gr){
  auto const end_k = std::min(max_transfers, nigiri::routing::kMaxTransfers) + 1U;
  // 1. Initialisierung
  gpu_delta_t d_worst_at_dest{};
  init_arrivals(d_worst_at_dest, worst_time_at_dest, gr.base_, gr.mem_->device_.time_at_dest_, gr.gtt_);
  this_grid().sync();

  // 2. Update Routes
  for (auto k = 1U; k != end_k; ++k) { // diese Schleife bleibt, da alle Threads in jede Runde gehen

    // Resultate aus lezter Runde von device in variable speichern?
    raptor_round(k, prf_idx, gr.gtt_, *gr.base_, *gr.allowed_claszes_,
                 gr.dist_to_end_, gr.is_dest_, gr.lb_, *gr.n_days_, gr.mem_->device_.search_dir_,
                 gr.mem_->device_.time_at_dest_, *gr.mem_->device_.any_station_marked_,
                 gr.mem_->device_.route_mark_, gr.mem_->device_.station_mark_,
                 gr.mem_->device_.best_, gr.mem_->device_.round_times_, gr.mem_->device_.tmp_,
                 gr.mem_->device_.size_best_, gr.mem_->device_.size_tmp_,
                 gr.mem_->device_.row_count_round_times_, gr.mem_->device_.column_count_round_times_,
                 gr.mem_->device_.size_route_mark_, gr.mem_->device_.size_station_mark_);
    this_grid().sync();
  }
  this_grid().sync();

  //construct journey

  this_grid().sync();

}


