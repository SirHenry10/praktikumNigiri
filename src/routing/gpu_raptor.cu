#include "nigiri/routing/gpu_raptor.cuh"
#include "cooperative_groups.h"

namespace nigiri::routing{
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

void reset_arrivals(){

}

// algo_stats_t get_stats() const { return stats_; }

void next_start_time(){

}

// hier wird Startpunkt hinzugefügt?
void add_start(location_idx_t const l, unixtime_t const t){

}

template <direction SearchDir, bool Rt>
void reconstruct(query const& q, journey& j){
  //reconstruct_journey<SearchDir, Rt>(...);
}

__device__ bool loop_routes(unsigned const k){
  return false;
}

__device__ void update_transfers(unsigned const k){

}

__device__ void update_footpaths(unsigned const k, gpu_timetable const tt){

}

__device__ bool update_intermodal_footpaths(unsigned const k){
  return false;
}

__device__ bool update_route(unsigned const k, gpu_timetable const tt){
  return false;
}

__device__ transport get_earliest_transport(unsigned const k, gpu_timetable const tt){

}

// nur für trace
__device__ bool is_transport_active(transport_idx_t const t, std::size_t const day)  {
  return false;
}

// nur für trace
__device__ delta_t time_at_stop(route_idx_t const r, transport const t, stop_idx_t const stop_idx, event_type const ev_type){

}

__device__ void update_time_at_dest(unsigned const k, delta_t const t){

}

template <direction SearchDir, bool Rt>
__device__ void raptor_round(profile_idx_t const prf_idx, gpu_raptor<SearchDir,Rt>& gr){

  // update_time_at_dest für alle locations

  // für jede location & für jede location_route state_.route_mark_

  // loop_routes mit true oder false

  this_grid().sync();
  // update_transfers

  this_grid().sync();
  // update_footpaths

  this_grid().sync();
  // update_intermodal_footpaths

}

template <direction SearchDir, bool Rt>
__device__ void init_arrivals(gpu_delta_t const d_worst_at_dest, unixtime_t const worst_time_at_dest, gpu_raptor<SearchDir, Rt>& gr){
  auto const t_id = get_global_thread_id();

  if(t_id==0){
    d_worst_at_dest = unix_to_gpu_delta(gr.base, worst_time_at_dest);
  }

  if(t_id < gr.time_at_dest_.size){
    gr.time_at_dest[t_id] = get_best(d_worst_at_dest, gr.time_at_dest[t_id]);
  }

}

// größten Teil von raptor.execute() wird hierdrin ausgeführt
// kernel muss sich außerhalb der gpu_raptor Klasse befinden
template <direction SearchDir, bool Rt>
__global__ void gpu_raptor_kernel(unixtime_t const start_time,
                                  std::uint8_t const max_transfers,
                                  unixtime_t const worst_time_at_dest,
                                  profile_idx_t const prf_idx,
                                  pareto_set<journey>& results,
                                  gpu_raptor<SearchDir,Rt>& gr){
  auto const end_k = std::min(max_transfers, kMaxTransfers) + 1U;
  // 1. Initialisierung
  gpu_delta_t const d_worst_at_dest{};
  init_arrivals(d_worst_at_dest, worst_time_at_dest, gr);
  this_grid().sync();

  // 2. Update Routes
  for (auto k = 1U; k != end_k; ++k) { // diese Schleife bleibt, da alle Threads in jede Runde gehen
    // Resultate aus lezter Runde von device in variable speichern?
    raptor_round(prf_idx, gr);
  }
  this_grid().sync();

  //construct journey

  this_grid().sync();

}


}  // extern "C"
