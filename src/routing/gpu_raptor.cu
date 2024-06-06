#include "nigiri/routing/gpu_raptor.cuh"

extern "C" {


static bool is_better(auto a, auto b) { return kFwd ? a < b : a > b; }
static bool is_better_or_eq(auto a, auto b) { return kFwd ? a <= b : a >= b; }
static auto get_best(auto a, auto b) { return is_better(a, b) ? a : b; }
static auto get_best(auto x, auto... y) {
  ((x = get_best(x, y)), ...);
  return x;
}
static auto dir(auto a) { return (kFwd ? 1 : -1) * a; }

void reset_arrivals(){

}

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
