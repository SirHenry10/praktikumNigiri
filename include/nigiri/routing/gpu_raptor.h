#pragma once

#include <cinttypes>
#include "nigiri/common/linear_lower_bound.h"
#include "nigiri/routing/gpu_raptor_state.h"
#include "nigiri/routing/gpu_timetable.h"
#include "nigiri/routing/pareto_set.h"
#include "nigiri/routing/raptor/debug.h"
//selbst sabotiert wegen import
#include <variant>
extern "C"{
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
                     short* & kMaxTravelTimeTicks_);

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
    short* & kMaxTravelTimeTicks_);
}
/*
inline void fetch_arrivals_async(mem* const& mem, cudaStream_t s) {
  //TODO:
  cuda_check();
}
 */
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
void execute_gpu(void** args,
                 device_context const& device,
                 cudaStream_t s,
                 gpu_direction const search_dir,
                 bool const rt);
void add_start_gpu(gpu_location_idx_t const l, gpu_unixtime_t const t,mem* mem_,gpu_timetable* gtt_,gpu_day_idx_t* base_,short const kInvalid);

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

__host__ __device__ inline int as_int(gpu_location_idx_t d) { return static_cast<int>(d.v_); }
__host__ __device__ inline int as_int(gpu_day_idx_t d)  { return static_cast<int>(d.v_); }
__host__ __device__ inline gpu_sys_days base(gpu_timetable const* gtt, gpu_day_idx_t* base) {
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
  using algo_stats_t = gpu_raptor_stats;
  static constexpr auto const kMaxTravelTimeTicks = gpu_kMaxTravelTime.count();
  static constexpr bool kUseLowerBounds = true;
  static constexpr auto const kFwd = (SearchDir == gpu_direction::kForward);
  static constexpr auto const kBwd = (SearchDir == gpu_direction::kBackward);
  static constexpr auto const kInvalid = kInvalidGpuDelta<SearchDir>;
  static constexpr auto const kUnreachable =
      std::numeric_limits<std::uint16_t>::max();
  static auto const kIntermodalTarget =
      gpu_to_idx(get_gpu_special_station(gpu_special_station::kEnd));


  gpu_raptor(gpu_timetable* gtt,
         gpu_raptor_state& state,
         std::vector<bool>& is_dest,
         std::vector<std::uint16_t>& dist_to_dest,
         std::vector<std::uint16_t>& lb,
         gpu_day_idx_t const base,
         gpu_clasz_mask_t const allowed_claszes)
      : gtt_{gtt},
        state_{state}
        {
    state_.init(*gtt_,kInvalid); //TODO: das im gpu_raptor_translator machen und dann das von raptor_state rüber kopieren und dann mit mem_-> reset_arrival löschen!?
    loaned_mem loan(state_,kInvalid);
    mem_ = loan.mem_;
    //das darüber sollte in gpu_raptor_translator und von raptor_state die daten rüber kopiert werden... dass raptor_state = gpu_raptor_state ist bei der initialisierung müssen wir danach den
    //TODO: überlegen: raptor_state mit den werten setzten die gpu_raptor_state hatte...???? also muss nach abschluss wieder raptor_state = gpu_raptor_state sein???
    mem_->reset_arrivals_async();
    std::unique_ptr<bool[]> copy_array(new bool[is_dest.size()]);
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
  ~gpu_raptor(){
      copy_to_device_destroy(allowed_claszes_,
                           dist_to_end_,
                           dist_to_end_size_,
                           base_,
                           is_dest_,
                           lb_,
                           n_days_,
                           kUnreachable_,
                           kIntermodalTarget_,
                           kMaxTravelTimeTicks_);
      destroy_gpu_timetable(gtt_);
      if (mem_ != nullptr) {
        delete mem_;
        mem_ = nullptr;
      }

  }
  algo_stats_t get_stats() const {
    return stats_;
  }

  void reset_arrivals() {
    mem_->reset_arrivals_async(); //hier nur reset von time_at_dest und round_times
  }

  void next_start_time() {
    //hier reset von tmp_,best_,prev_station_mark_,station_mark_,route_mark_ //TODO: why not any_station_marked?
    mem_->next_start_time_async();
  }

  void add_start(gpu_location_idx_t const l, gpu_unixtime_t const t) {
      add_start_gpu(l,t,mem_,gtt_,base_,kInvalid);
  }


  // hier wird Kernel aufgerufen
  gpu_delta_t* execute(gpu_unixtime_t const start_time,
             uint8_t const max_transfers,
             gpu_unixtime_t const worst_time_at_dest,
             gpu_profile_idx_t const prf_idx){
    //TODO: alles bis cuda_check in .cu datei schieben also hilfsmethode schreiben
    //TODO: wie benutzten wir start_time bzw... muss das noch rüber kopiert werden???? nike fragen...
  void* kernel_args[] = {(void*)&start_time, (void*)&max_transfers, (void*)&worst_time_at_dest, (void*)&prf_idx, (void*)this};
  execute_gpu(kernel_args, mem_->context_, mem_->context_.proc_stream_,SearchDir,Rt);

  //copy stats from host to raptor attribute
  gpu_raptor_stats tmp{};
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
  gpu_timetable* gtt_{nullptr};
  gpu_raptor_state& state_;
  mem* mem_;
  bool* is_dest_;
  uint16_t* dist_to_end_;
  uint32_t* dist_to_end_size_;
  uint16_t* lb_;
  gpu_day_idx_t* base_;
  int* n_days_;
  gpu_raptor_stats stats_;
  gpu_clasz_mask_t* allowed_claszes_;
  std::uint16_t* kUnreachable_;
  unsigned int* kIntermodalTarget_;
  short* kMaxTravelTimeTicks_;
};
