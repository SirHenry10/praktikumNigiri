#include "nigiri/routing/gpu_raptor.cuh"

namespace nigiri::routing{


void reset_arrivals(){

}
algo_stats_t get_stats() const { return stats_; }

void next_start_time(){

}

// hier wird Startpunkt hinzugefügt?
void add_start(location_idx_t const l, unixtime_t const t){

}

void reconstruct(query const& q, journey& j){
  reconstruct_journey<SearchDir>(...);
}

__device__ bool loop_routes(unsigned const k){

}

__device__ void update_transfers(unsigned const k){

}

__device__ void update_footpaths(unsigned const k, gpu_timetable const tt){

}

__device__ update_intermodal_footpaths(unsigned const k){

}

__device__ update_route(unsigned const k, gpu_timetable const tt){

}

__device__ transport get_earliest_transport(unsigned const k, gpu_timetable const tt){

}

__device__ bool is_transport_active(transport_idx_t const t, std::size_t const day) const {

}

__device__ delta_t time_at_stop(route_idx_t const r, transport const t, stop_idx_t const stop_idx, event_type const ev_type){

}

__device__ void update_time_at_dest(unsigned const k, delta_t const t){

}

__device__ void init_arrivals(gpu_delta_t const d_worst_at_dest, gpu_raptor& gr){
  auto const t_id = get_global_thread_id();

  if(t_id < gr.time_at_dest_.size){
    gr.time_at_dest[t_id] = get_best(d_worst_at_dest, gr.time_at_dest[t_id]);
  }

}

// größten Teil von raptor.execute() wird hierdrin ausgeführt
// kernel muss sich außerhalb der gpu_raptor Klasse befinden
__global__ void gpu_raptor_kernel(unixtime_t const start_time,
                                  std::uint8_t const max_transfers,
                                  unixtime_t const worst_time_at_dest,
                                  profile_idx_t const prf_idx,
                                  pareto_set<journey>& results,
                                  gpu_raptor& gr){
  // 1. Initialisierung
  gpu_delta_t const d_worst_at_dest = to_gpu_delta(gr.base, worst_time_at_dest);
  init_arrivals(d_worst_at_dest, gr);
  this_grid().sync();

  // 2. Update Routes
  for (auto k = 1U; k != end_k; ++k) { // diese Schleife bleibt, da alle Threads in jede Runde gehen

  }
  this_grid().sync();


}


}  // extern "C"
