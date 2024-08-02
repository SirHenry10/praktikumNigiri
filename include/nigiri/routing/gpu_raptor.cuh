#pragma once

#include <cinttypes>
#include "nigiri/common/linear_lower_bound.h"
#include "nigiri/routing/journey.h"
#include "nigiri/routing/query.h"
#include "nigiri/routing/limits.h"
#include "nigiri/routing/pareto_set.h"
#include "nigiri/routing/raptor/debug.h"
#include "nigiri/routing/gpu_raptor_state.cuh"
#include "nigiri/routing/raptor/raptor_state.h" //maybe weg
#include "nigiri/routing/raptor/reconstruct.h"
#include "nigiri/rt/rt_timetable.h"
#include "nigiri/special_stations.h"
#include "nigiri/routing/gpu_timetable.h"
#include "nigiri/special_stations.h"
extern "C" {
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
}//extern "C"

extern "C" {
void copy_to_devices(gpu_clasz_mask_t const& allowed_claszes,
                     std::vector<std::uint16_t> const& dist_to_dest,
                     gpu_day_idx_t const& base,
                     bool* const& is_dest,
                     std::size_t is_dest_size,
                     std::vector<std::uint16_t> const& lb,
                     gpu_direction const& search_dir,
                     int const& n_days,
                     gpu_clasz_mask_t*& allowed_claszes_,
                     std::uint16_t* & dist_to_end_,
                     gpu_day_idx_t* & base_,
                     bool* & is_dest_,
                     std::uint16_t* & lb_,
                     gpu_direction* & search_dir_,
                     int* & n_days_){
  cudaError_t code;
  allowed_claszes_ = nullptr;
  CUDA_COPY_TO_DEVICE(gpu_clasz_mask_t,allowed_claszes_,&allowed_claszes,1);
  dist_to_end_ = nullptr;
  CUDA_COPY_TO_DEVICE(std::uint16_t,dist_to_end_,dist_to_dest.data(),dist_to_dest.size());
  base_ = nullptr;
  CUDA_COPY_TO_DEVICE(gpu_day_idx_t,base_,&base,1);
  is_dest_ = nullptr;
  CUDA_COPY_TO_DEVICE(bool,is_dest_,is_dest,is_dest_size);
  lb_ = nullptr;
  CUDA_COPY_TO_DEVICE(std::uint16_t ,lb_,lb.data(),lb.size());
  search_dir_ = nullptr;
  CUDA_COPY_TO_DEVICE(gpu_direction,search_dir_,&search_dir,1);
  n_days_ = nullptr;
  CUDA_COPY_TO_DEVICE(int,n_days_,&n_days,1);
fail:
  cudaFree(allowed_claszes_);
  cudaFree(dist_to_end_);
  cudaFree(base_);
  cudaFree(is_dest_);
  cudaFree(lb_);
  cudaFree(search_dir_);
};
}//extern "C"
struct raptor_stats {
  std::uint64_t n_routing_time_{0ULL};
  std::uint64_t n_footpaths_visited_{0ULL};
  std::uint64_t n_routes_visited_{0ULL};
  std::uint64_t n_earliest_trip_calls_{0ULL};
  std::uint64_t n_earliest_arrival_updated_by_route_{0ULL};
  std::uint64_t n_earliest_arrival_updated_by_footpath_{0ULL};
  std::uint64_t fp_update_prevented_by_lower_bound_{0ULL};
  std::uint64_t route_update_prevented_by_lower_bound_{0ULL};
};

template <typename Kernel>
void inline launch_kernel(Kernel kernel, void** args,
                          device_context const& device, cudaStream_t s) {
  cudaSetDevice(device.id_);

  cudaLaunchCooperativeKernel((void*)kernel, device.grid_,  //  NOLINT
                              device.threads_per_block_, args, 0, s);
  cuda_check();
}

inline void fetch_arrivals_async(mem* const& mem, cudaStream_t s) {
  cudaMemcpyAsync(
      mem->host_.round_times_, mem->device_.round_times_,
      mem->host_.row_count_round_times_ * mem->host_.column_count_round_times_ * sizeof(gpu_delta), cudaMemcpyDeviceToHost, s);
  cuda_check();
}
/*
inline void fetch_arrivals_async(mem* const& mem, raptor_round const round_k,
                                 cudaStream_t s) {
  cudaMemcpyAsync((*dq.mem_->host_.result_)[round_k],
                  dq.mem_->device_.result_[round_k],
                  dq.mem_->host_.result_->stop_count_ * sizeof(time),
                  cudaMemcpyDeviceToHost, s);
  cuda_check();
}
 */
template <gpu_direction SearchDir, bool Rt>
struct gpu_raptor;

template <gpu_direction SearchDir, bool Rt>
__global__ void gpu_raptor_kernel(gpu_unixtime_t const start_time,
                                  uint8_t const max_transfers,
                                  gpu_unixtime_t const worst_time_at_dest,
                                  gpu_profile_idx_t const prf_idx,
                                  nigiri::pareto_set<nigiri::routing::journey>& results,
                                  gpu_raptor<SearchDir,Rt>& gr);

template<gpu_direction SearchDir>
__host__ __device__ static bool is_better(auto a, auto b) { return SearchDir==gpu_direction::kForward ? a < b : a > b; }
template<gpu_direction SearchDir>
__host__ __device__ static bool is_better_or_eq(auto a, auto b) { return SearchDir==gpu_direction::kForward ? a <= b : a >= b; }
__host__ __device__ static auto get_best(auto a, auto b) { return is_better(a, b) ? a : b; }
__host__ __device__ static auto get_best(auto x, auto... y) {
  ((x = get_best(x, y)), ...);
  return x;
}
__host__ __device__ int as_int(gpu_day_idx_t d)  { return static_cast<int>(d.v_); }
__host__ __device__ date::sys_days base(gpu_timetable* const gtt, gpu_day_idx_t* base) {
  return gtt->gpu_internal_interval_days().from_ + as_int(*base) * date::days{1};
}

template <gpu_direction SearchDir, bool Rt>
struct gpu_raptor {
  using algo_stats_t = raptor_stats;

  static constexpr bool kUseLowerBounds = true;
  static constexpr auto const kFwd = (SearchDir == gpu_direction::kForward);
  static constexpr auto const kBwd = (SearchDir == gpu_direction::kBackward);
  static constexpr auto const kInvalid = kInvalidGpuDelta<SearchDir>;
  static constexpr auto const kUnreachable =
      std::numeric_limits<std::uint16_t>::max();
  static constexpr auto const kIntermodalTarget =
      to_idx(get_special_station(nigiri::special_station::kEnd));


  __host__ __device__ static auto dir(auto a) { return (kFwd ? 1 : -1) * a; }
  gpu_raptor(gpu_timetable const* gtt,
         nigiri::rt_timetable const* rtt,
         gpu_raptor_state& state,
         std::vector<bool>& is_dest,
         std::vector<std::uint16_t>& dist_to_dest,
         std::vector<std::uint16_t>& lb,
         gpu_day_idx_t const base,
         gpu_clasz_mask_t const allowed_claszes)
      : gtt_{gtt},
        rtt_{rtt},
        state_{state}
        {
    state_.init(*gtt_,kInvalid);
    loaned_mem loan(state_,kInvalid);
    mem_ = loan.mem_;
    bool copy_array[is_dest.size()];
    for (int i = 0; i<is_dest.size();i++){
      copy_array[i] = is_dest[i];
    }
    copy_to_devices(allowed_claszes,
                    dist_to_dest,
                    base,
                    copy_array,
                    is_dest.size(),
                    lb,
                    SearchDir,
                    gtt_->gpu_internal_interval_days().size().count(),
                    allowed_claszes_,
                    dist_to_end_,
                    base_,
                    is_dest_,
                    lb_,
                    search_dir_,
                    n_days_);
  }
  //TODO: BUILD DESTRUKTOR TO DESTORY mallocs
  algo_stats_t get_stats() const {
    //TODO:
    return stats_;
  }

  void reset_arrivals() {
    utl::fill(mem_->device_.time_at_dest_, kInvalid);
    mem_->host_.reset(kInvalid);
  }

  void next_start_time() {
    std::vector<gpu_delta_t> best_new(mem_->device_.size_best_,kInvalid);
    std::vector<gpu_delta_t> tmp_new(mem_->device_.size_tmp_,kInvalid);
    bool copy_array_station[mem_->device_.size_station_mark_];
    std::fill(std::begin(copy_array_station), std::end(copy_array_station), false);
    bool copy_array_route[mem_->device_.size_route_mark_];
    std::fill(std::begin(copy_array_route), std::end(copy_array_route), false);
    cudaMemcpy(mem_->device_.best_, best_new.data(), mem_->device_.size_best_*sizeof(gpu_delta_t), cudaMemcpyHostToDevice);
    cudaMemcpy(mem_->device_.tmp_, tmp_new.data(), mem_->device_.size_tmp_*sizeof(gpu_delta_t), cudaMemcpyHostToDevice);
    cudaMemcpy(mem_->device_.prev_station_mark_, copy_array_station, mem_->device_.size_station_mark_*sizeof(bool), cudaMemcpyHostToDevice);
    cudaMemcpy(mem_->device_.station_mark_, copy_array_station, mem_->device_.size_station_mark_*sizeof(bool), cudaMemcpyHostToDevice);
    cudaMemcpy(mem_->device_.route_mark_, copy_array_route, mem_->device_.size_route_mark_*sizeof(bool), cudaMemcpyHostToDevice);
  }

  void add_start(gpu_location_idx_t const l, gpu_unixtime_t const t) {
    trace_upd("adding start {}: {}\n", location{tt_, l}, t);
    std::vector<gpu_delta_t> best_new(mem_->device_.size_best_,kInvalid);
    std::vector<gpu_delta_t> round_times_new((mem_->device_.column_count_round_times_*mem_->device_.row_count_round_times_),kInvalid);
    best_new[to_idx(l).v_] = unix_to_gpu_delta(base(), t);
    round_times_new[0U*mem_->device_.row_count_round_times_+ as_int(to_idx(l))] = unix_to_gpu_delta(base(), t);
    bool copy_array[mem_->device_.size_station_mark_];
    std::fill(std::begin(copy_array), std::end(copy_array), false);
    copy_array[to_idx(l).v_] = true;
    cudaMemcpy(mem_->device_.best_, best_new.data(), mem_->device_.size_best_*sizeof(gpu_delta_t), cudaMemcpyHostToDevice);
    cudaMemcpy(mem_->host_.round_times_, round_times_new.data(), round_times_new.size()*sizeof(gpu_delta_t), cudaMemcpyHostToDevice);
    cudaMemcpy(mem_->device_.round_times_, round_times_new.data(), round_times_new.size()*sizeof(gpu_delta_t), cudaMemcpyHostToDevice);
    cudaMemcpy(mem_->device_.station_mark_, copy_array, mem_->device_.size_station_mark_*sizeof(bool), cudaMemcpyHostToDevice);
  }


  // hier wird Kernel aufgerufen
void execute(gpu_unixtime_t const start_time,
             uint8_t const max_transfers,
             gpu_unixtime_t const worst_time_at_dest,
             gpu_profile_idx_t const prf_idx,
             nigiri::pareto_set<nigiri::routing::journey>& results){

  void* kernel_args[] = {(void*)&start_time, (void*)&max_transfers, (void*)&worst_time_at_dest, (void*)&prf_idx, (void*)&results, (void*)&this};
  launch_kernel(gpu_raptor_kernel<SearchDir,Rt>, kernel_args,mem_->context_,mem_->context_.proc_stream_);
  cuda_check();
  //TODO: copy result back
  //TODO: ALLES LÖSCHEN!!!!!!!!!!!!!
}


  void reconstruct(nigiri::routing::query const& q, nigiri::routing::journey& j) {
    // reconstruct_journey<SearchDir>(tt_, rtt_, q, state_, j, base(), base_);
  }

  template <typename T>
  auto get_begin_it(T const& t) {
    if constexpr (kFwd) {
      return t.begin();
    } else {
      return t.rbegin();
    }
  }

  template <typename T>
  auto get_end_it(T const& t) {
    if constexpr (kFwd) {
      return t.end();
    } else {
      return t.rend();
    }
  }

  gpu_timetable const* gtt_{nullptr};
  nigiri::rt_timetable const* rtt_{nullptr};
  // all diese müssen mit malloc (evtl. in anderer Datei)
  gpu_raptor_state& state_;
  mem* mem_;
  bool* is_dest_;
  uint16_t* dist_to_end_;
  uint16_t* lb_;
  gpu_direction* search_dir_;
  gpu_day_idx_t* base_;
  int* n_days_;
  raptor_stats stats_;
  uint32_t n_rt_transports_;
  gpu_clasz_mask_t* allowed_claszes_;
};