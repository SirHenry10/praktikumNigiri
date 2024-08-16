#pragma once

#include <cinttypes>
#include "nigiri/common/linear_lower_bound.h"
#include "nigiri/routing/gpu_raptor_state.cuh"
#include "nigiri/routing/gpu_timetable.h"
#include "nigiri/routing/pareto_set.h"
#include "nigiri/routing/raptor/debug.h"
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
  CUDA_COPY_TO_DEVICE(gpu_clasz_mask_t,allowed_claszes_,&allowed_claszes,1);
  dist_to_end_ = nullptr;
  CUDA_COPY_TO_DEVICE(std::uint16_t,dist_to_end_,dist_to_dest.data(),dist_to_dest.size());
  dist_to_end_size_ = nullptr;
  CUDA_COPY_TO_DEVICE(std::uint32_t,dist_to_end_size_,&dist_to_end_size_,1);
  base_ = nullptr;
  CUDA_COPY_TO_DEVICE(gpu_day_idx_t,base_,&base,1);
  is_dest_ = nullptr;
  CUDA_COPY_TO_DEVICE(bool,is_dest_,is_dest,is_dest_size);
  lb_ = nullptr;
  CUDA_COPY_TO_DEVICE(std::uint16_t ,lb_,lb.data(),lb.size());
  n_days_ = nullptr;
  CUDA_COPY_TO_DEVICE(int,n_days_,&n_days,1);
  kUnreachable_ = nullptr;
  CUDA_COPY_TO_DEVICE(std::uint16_t,kUnreachable_,&kUnreachable,1);
  kIntermodalTarget_ = nullptr;
  CUDA_COPY_TO_DEVICE(gpu_location_idx_t,kIntermodalTarget_,&kIntermodalTarget,1);
  kMaxTravelTimeTicks_ = nullptr;
  CUDA_COPY_TO_DEVICE(short,kMaxTravelTimeTicks_,&kMaxTravelTimeTicks,1);
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
}//extern "C"


template <typename Kernel>
void inline launch_kernel(Kernel kernel, void** args,
                          device_context const& device, cudaStream_t s) {
  cudaSetDevice(device.id_);

  cudaLaunchCooperativeKernel((void*)kernel, device.grid_,  //  NOLINT
                              device.threads_per_block_, args, 0, s);
  cuda_check();
}

inline void fetch_arrivals_async(mem* const& mem, cudaStream_t s) {
  //TODO:
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
                                  gpu_raptor<SearchDir,Rt>& gr);

template<gpu_direction SearchDir>
__host__ __device__ static bool is_better(auto a, auto b) { return SearchDir==gpu_direction::kForward ? a < b : a > b; }
__host__ __device__ static auto get_smaller(auto a, auto b) { return a < b ? a : b ;}
template<gpu_direction SearchDir>
__host__ __device__ static bool is_better_or_eq(auto a, auto b) { return SearchDir==gpu_direction::kForward ? a <= b : a >= b; }
__host__ __device__ static auto get_best(auto a, auto b) { return is_better(a, b) ? a : b; }
__host__ __device__ static auto get_best(auto x, auto... y) {
  ((x = get_best(x, y)), ...);
  return x;
}

__host__ __device__ int as_int(gpu_location_idx_t d)  { return static_cast<int>(d.v_); }
__host__ __device__ int as_int(gpu_day_idx_t d)  { return static_cast<int>(d.v_); }
__host__ __device__ gpu_sys_days base(gpu_timetable const* gtt, gpu_day_idx_t* base) {
  return gtt->gpu_internal_interval_days().from_ + as_int(*base) * gpu_days{1};
}
template<gpu_direction SearchDir>
__host__ __device__ static auto dir(auto a) { return (SearchDir==gpu_direction::kForward ? 1 : -1) * a; }

template <typename T, gpu_direction SearchDir>
__host__ __device__ auto get_begin_it(T const& t) {
  if constexpr (SearchDir == gpu_direction::kForward) {
    return t.begin();
  } else {
    return t.rbegin();
  }
}

template <typename T, gpu_direction SearchDir>
__host__ __device__ auto get_end_it(T const& t) {
  if constexpr ((SearchDir == gpu_direction::kForward)) {
    return t.end();
  } else {
    return t.rend();
  }
}

template <gpu_direction SearchDir, bool Rt>
struct gpu_raptor {
  using algo_stats_t = raptor_stats;

  static constexpr auto const kMaxTravelTimeTicks = gpu_kMaxTravelTime.count();
  static constexpr bool kUseLowerBounds = true;
  static constexpr auto const kFwd = (SearchDir == gpu_direction::kForward);
  static constexpr auto const kBwd = (SearchDir == gpu_direction::kBackward);
  static constexpr auto const kInvalid = kInvalidGpuDelta<SearchDir>;
  static constexpr auto const kUnreachable =
      std::numeric_limits<std::uint16_t>::max();
  static auto const kIntermodalTarget =
      gpu_to_idx(get_gpu_special_station(gpu_special_station::kEnd));


  gpu_raptor(gpu_timetable const* gtt,
         gpu_raptor_state& state,
         std::vector<bool>& is_dest,
         std::vector<std::uint16_t>& dist_to_dest,
         std::vector<std::uint16_t>& lb,
         gpu_day_idx_t const base,
         gpu_clasz_mask_t const allowed_claszes)
      : gtt_{gtt},
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
                    gtt_->gpu_internal_interval_days().size().count(),
                    kUnreachable,
                    kMaxTravelTimeTicks,
                    kIntermodalTarget,
                    allowed_claszes_,
                    dist_to_end_,
                    dist_to_end_size_,
                    base_,
                    is_dest_,
                    lb_,
                    n_days_,
                    kUnreachable_,
                    kIntermodalTarget_,
                    kMaxTravelTimeTicks_);
  }
  //TODO: BUILD DESTRUKTOR TO DESTORY mallocs
  algo_stats_t get_stats() const {
    return stats_;
  }

  void reset_arrivals() {
    //TODO: alles reset arrivals
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
    trace_upd("adding start {}: {}\n", location{gtt_, l}, t);
    std::vector<gpu_delta_t> best_new(mem_->device_.size_best_,kInvalid);
    std::vector<gpu_delta_t> round_times_new((mem_->device_.column_count_round_times_*mem_->device_.row_count_round_times_),kInvalid);
    best_new[gpu_to_idx(l)] = unix_to_gpu_delta(base(gtt_,base_), t);
    round_times_new[0U*mem_->device_.row_count_round_times_+ gpu_to_idx(l)] = unix_to_gpu_delta(base(gtt_,base_), t);
    bool copy_array[mem_->device_.size_station_mark_];
    std::fill(std::begin(copy_array), std::end(copy_array), false);
    copy_array[gpu_to_idx(l)] = true;
    cudaMemcpy(mem_->device_.best_, best_new.data(), mem_->device_.size_best_*sizeof(gpu_delta_t), cudaMemcpyHostToDevice);
    //TODO: MAYBE noch auf host kopieren weis aber nicht ob notwendig
    cudaMemcpy(mem_->device_.round_times_, round_times_new.data(), round_times_new.size()*sizeof(gpu_delta_t), cudaMemcpyHostToDevice);
    cudaMemcpy(mem_->device_.station_mark_, copy_array, mem_->device_.size_station_mark_*sizeof(bool), cudaMemcpyHostToDevice);
  }


  // hier wird Kernel aufgerufen
  gpu_delta_t* execute(gpu_unixtime_t const start_time,
             uint8_t const max_transfers,
             gpu_unixtime_t const worst_time_at_dest,
             gpu_profile_idx_t const prf_idx){

  void* kernel_args[] = {(void*)&start_time, (void*)&max_transfers, (void*)&worst_time_at_dest, (void*)&prf_idx, (void*)&this};
  launch_kernel(gpu_raptor_kernel<SearchDir,Rt>, kernel_args,mem_->context_,mem_->context_.proc_stream_);
  cuda_check();
  //TODO: copy result back
  //TODO: ALLES LÖSCHEN!!!!!!!!!!!
  //copy stats from host to raptor attribute
  raptor_stats tmp{};
  for (int i = 0; i<32; ++i) {
    tmp.n_routing_time_ += mem_->host_.stats_.get()[i].n_routing_time_;
    tmp.n_footpaths_visited_ += mem_->host_.stats_.get()[i].n_footpaths_visited_;
    tmp.n_routes_visited_ += mem_->host_.stats_.get()[i].n_routes_visited_;
    tmp.n_earliest_trip_calls_ += mem_->host_.stats_.get()[i].n_earliest_trip_calls_;
    tmp.n_earliest_arrival_updated_by_route_ += mem_->host_.stats_.get()[i].n_earliest_arrival_updated_by_route_;
    tmp.n_earliest_arrival_updated_by_footpath_ += mem_->host_.stats_.get()[i].n_earliest_arrival_updated_by_footpath_;
    tmp.fp_update_prevented_by_lower_bound_ += mem_->host_.stats_.get()[i].fp_update_prevented_by_lower_bound_;
    tmp.route_update_prevented_by_lower_bound_ += mem_->host_.stats_.get()[i].route_update_prevented_by_lower_bound_;
  }
  stats_ = tmp;
  return mem_->host_.round_times_.get();
}
  gpu_timetable const* gtt_{nullptr};
  gpu_raptor_state& state_;
  mem* mem_;
  bool* is_dest_;
  uint16_t* dist_to_end_;
  uint32_t* dist_to_end_size_;
  uint16_t* lb_;
  gpu_day_idx_t* base_;
  int* n_days_;
  raptor_stats stats_;
  uint32_t n_rt_transports_;
  gpu_clasz_mask_t* allowed_claszes_;
  std::uint16_t* kUnreachable_;
  unsigned int* kIntermodalTarget_;
  short* kMaxTravelTimeTicks_;
};