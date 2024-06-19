#include "nigiri/routing/gpu_raptor.cuh"

namespace nigiri::routing{


void reset_arrivals(){

}
algo_stats_t get_stats() const { return stats_; }

void next_start_time(){

}

// hier wird Startpunkt hinzugefügt
void add_start(location, unixtime_t const t){

}

__device__ void reconstruct(query const& q, journey& j){
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

bool is_transport_active(transport_idx_t const t, std::size_t const day) const {

}

delta_t time_at_stop(route_idx_t const r, transport const t, stop_idx_t const stop_idx, event_type const ev_type){

}

__device__ void update_time_at_dest(unsigned const k, delta_t const t){

}


// größten Teil von raptor.execute() wird hierdrin ausgeführt
__global__ void gpu_raptor_kernel(gpu_timetable const tt){

}

void execute(gpu_timetable const tt){

}

}  // extern "C"
