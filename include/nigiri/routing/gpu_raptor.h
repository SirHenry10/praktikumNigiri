#pragma once

#include <iostream>
#include <cinttypes>
#include "nigiri/common/linear_lower_bound.h"
#include "nigiri/routing/gpu_raptor_state.h"
#include "nigiri/routing/gpu_timetable.h"
#include "nigiri/routing/pareto_set.h"
#include "nigiri/routing/raptor/debug.h"
#include "utl/helpers/algorithm.h"
#include <variant>
extern "C"{
void copy_to_devices(gpu_clasz_mask_t const& allowed_claszes,
                     std::vector<std::uint16_t> const& dist_to_dest,
                     gpu_day_idx_t const& base,
                     std::vector<uint8_t> const& is_dest,
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
    gpu_location_idx_t* & kIntermodalTarget_,
    short* & kMaxTravelTimeTicks_);
}
void launch_kernel(void** args,
                          device_context const& device,
                          cudaStream_t s,
                          gpu_direction search_dir,
                          bool rt);
void copy_back(mem* mem);

std::unique_ptr<mem> gpu_mem(
    std::vector<gpu_delta_t>& tmp,
    std::vector<gpu_delta_t>& best,
    std::vector<bool>& station_mark,
    std::vector<bool>& prev_station_mark,
    std::vector<bool>& route_mark,
    gpu_direction search_dir,
    gpu_timetable const* gtt);
void add_start_gpu(std::vector<gpu_delta_t>& best, std::vector<gpu_delta_t>& round_times,std::vector<uint32_t>& station_mark,mem* mem);

void copy_to_gpu_args(gpu_unixtime_t const* start_time,
                      gpu_unixtime_t const* worst_time_at_dest,
                      gpu_profile_idx_t const* prf_idx,
                      gpu_unixtime_t*& start_time_ptr,
                      gpu_unixtime_t*& worst_time_at_dest_ptr,
                      gpu_profile_idx_t*& prf_idx_ptr);
void destroy_copy_to_gpu_args(gpu_unixtime_t* start_time_ptr,
                              gpu_unixtime_t* worst_time_at_dest_ptr,
                              gpu_profile_idx_t* prf_idx_ptr);
template<gpu_direction SearchDir>
__host__ __device__ static bool is_better(auto a, auto b) { return SearchDir==gpu_direction::kForward ? a < b : a > b; }
__host__ __device__ static auto get_smaller(auto a, auto b) { return a < b ? a : b ;}
template<gpu_direction SearchDir>
__host__ __device__ static bool is_better_or_eq(auto a, auto b) { return SearchDir==gpu_direction::kForward ? a <= b : a >= b; }

//TODO: frage nochmal wann wird oben und wann unten genutz 2. hab gefixt das diese kein template haben
//TODO: nochmal überlegen ob get_best und etc.. immer nach SearchDir kucken sollte oder nur bei gpu_delta_t???
template<gpu_direction SearchDir>
__host__ __device__ static auto get_best(auto a, auto b) { return is_better<SearchDir>(a, b) ? a : b; }

template<gpu_direction SearchDir>
__host__ __device__ static auto get_best(auto x, auto... y) {
  ((x = get_best<SearchDir>(x, y)), ...);
  return x;
}

__host__ __device__ inline int as_int(gpu_location_idx_t d) { return static_cast<int>(d.v_); }
__host__ __device__ inline int as_int(gpu_day_idx_t d)  { return static_cast<int>(d.v_); }
//TODO: base funktioniert nur auf device!!
__device__ inline gpu_sys_days base(gpu_day_idx_t* base,gpu_interval<gpu_sys_days> const* date_range_ptr) {
  return gpu_internal_interval_days(date_range_ptr).from_ + as_int(*base) * gpu_days{1};
}
__host__ inline gpu_sys_days cpu_base(gpu_timetable const* gtt, gpu_day_idx_t base) {
  return gtt->cpu_internal_interval_days().from_ + as_int(base) * gpu_days{1};
}
template<gpu_direction SearchDir>
__host__ __device__ static auto dir(auto a) { return (SearchDir==gpu_direction::kForward ? 1 : -1) * a; }

__host__ __device__ inline gpu_delta_t to_gpu_delta(gpu_day_idx_t const day, std::int16_t const mam,gpu_day_idx_t* base_) {
  return gpu_clamp((as_int(day) - as_int(*base_)) * 1440 + mam);
}

template <gpu_direction SearchDir, typename T>
__host__ __device__ auto gpu_get_begin_it(T const& t) {
  if constexpr (SearchDir == gpu_direction::kForward) {
    return t.begin();
  } else {
    return t.rbegin();
  }
}

template <gpu_direction SearchDir, typename T>
__host__ __device__ auto gpu_get_end_it(T const& t) {
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


  gpu_raptor(gpu_timetable const* gtt,
             mem* mem,
         std::vector<uint8_t>& is_dest,
         std::vector<std::uint16_t>& dist_to_dest,
         std::vector<std::uint16_t>& lb,
         gpu_day_idx_t const& base,
         gpu_clasz_mask_t const& allowed_claszes,
             int const& n_days)
      : gtt_{gtt},
        mem_{mem},
        best_(mem_->device_.n_locations_, kInvalid),
        round_times_(mem_->device_.column_count_round_times_ * mem_->device_.row_count_round_times_, kInvalid),
        station_mark_((mem_->device_.n_locations_ / 32) + 1, 0),
        added_start_(false)
        {
    auto start_bevor_copy = std::chrono::high_resolution_clock::now();


    auto start_reset_a = std::chrono::high_resolution_clock::now();
    mem_->reset_arrivals_async();
    auto end_reset_a = std::chrono::high_resolution_clock::now();
    auto reset_a_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_reset_a - start_reset_a).count();
    std::cout << "reset_a Time: " << reset_a_duration << " microseconds\n";

    auto const kIntermodalTarget  =
        gpu_to_idx(get_gpu_special_station(gpu_special_station::kEnd));
    cpu_base_ = base;

    auto end_bevor_copy = std::chrono::high_resolution_clock::now();
    auto end_bevor_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_bevor_copy - start_bevor_copy).count();
    std::cout << "end_bevor Time: " << end_bevor_duration << " microseconds\n";
    auto start_copy = std::chrono::high_resolution_clock::now();
    copy_to_devices(allowed_claszes,
                    dist_to_dest,
                    base,
                    is_dest,
                    lb,
                    n_days,//error,
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
    auto end_copy = std::chrono::high_resolution_clock::now();
    auto copy_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_copy - start_copy).count();
    std::cout << "copy Time: " << copy_duration << " microseconds\n";
  }
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
  }
  algo_stats_t get_stats() const {
    return stats_;
  }

  void reset_arrivals() {
    utl::fill(round_times_,kInvalid);
    added_start_ = true;
    mem_->reset_arrivals_async();
  }

  void next_start_time() {
    utl::fill(best_, kInvalid);
    utl::fill(station_mark_, 0);
    added_start_ = true;
    mem_->next_start_time_async();
  }

  void add_start(gpu_location_idx_t const l, gpu_unixtime_t const t) {
    trace_upd("adding start {}: {}\n", location{gtt_, l}, t);
    best_[gpu_to_idx(l)] = unix_to_gpu_delta(cpu_base(gtt_, cpu_base_), t);
    round_times_[0U * mem_->device_.column_count_round_times_ + gpu_to_idx(l)] = unix_to_gpu_delta(cpu_base(gtt_, cpu_base_), t);
    unsigned int const store_idx = (gpu_to_idx(l) >> 5);  // divide by 32
    unsigned int const mask = 1 << (gpu_to_idx(l) % 32);
    station_mark_[store_idx] |= mask;
    added_start_ = true;
  }


  // hier wird Kernel aufgerufen
  void execute(gpu_unixtime_t const& start_time,
             uint8_t const& max_transfers,
             gpu_unixtime_t const& worst_time_at_dest,
             gpu_profile_idx_t const& prf_idx){
    //start_time muss rüber das bei trace,max_transfers muss nicht malloced werden, worst_time_at_Dest muss rüber kopiert werden, prf_idx muss kopiert werden

    auto start_add_new = std::chrono::high_resolution_clock::now();
    if (added_start_){
      add_start_gpu(best_,round_times_,station_mark_,mem_);
      added_start_ = false;
    }
    auto end_add_new = std::chrono::high_resolution_clock::now();
    auto add_new_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_add_new - start_add_new).count();
    std::cout << "add_new Time: " << add_new_duration << " microseconds\n";
    gpu_unixtime_t* start_time_ptr = nullptr;
    gpu_unixtime_t* worst_time_at_dest_ptr = nullptr;
    gpu_profile_idx_t* prf_idx_ptr = nullptr;
    copy_to_gpu_args(&start_time,
                     &worst_time_at_dest,
                     &prf_idx,
                     start_time_ptr,
                     worst_time_at_dest_ptr,
                     prf_idx_ptr);
    void* kernel_args[] = {(void*)&start_time_ptr,
                           (void*)&max_transfers,
                           (void*)&worst_time_at_dest_ptr,
                           (void*)&prf_idx_ptr,
                           (void*)&allowed_claszes_,
                           (void*)&dist_to_end_,
                           (void*)&dist_to_end_size_,
                           (void*)&base_,
                           (void*)&is_dest_,
                           (void*)&lb_,
                           (void*)&n_days_,
                           (void*)&kUnreachable_,
                           (void*)&kIntermodalTarget_,
                           (void*)&kMaxTravelTimeTicks_,
                           (void*)&mem_->device_.tmp_,
                           (void*)&mem_->device_.best_,
                           (void*)&mem_->device_.round_times_,
                           (void*)&mem_->device_.time_at_dest_,
                           (void*)&mem_->device_.station_mark_,
                           (void*)&mem_->device_.prev_station_mark_,
                           (void*)&mem_->device_.route_mark_,
                           (void*)&mem_->device_.any_station_marked_,
                           (void*)&mem_->device_.row_count_round_times_,
                           (void*)&mem_->device_.column_count_round_times_,
                           (void*)&mem_->device_.stats_,
                           (void*)&gtt_->route_stop_times_,
                           (void*)&gtt_->route_location_seq_,
                           (void*)&gtt_->location_routes_,
                           (void*)&gtt_->n_locations_,
                           (void*)&gtt_->n_routes_,
                           (void*)&gtt_->route_stop_time_ranges_,
                           (void*)&gtt_->route_transport_ranges_,
                           (void*)&gtt_->bitfields_,
                           (void*)&gtt_->transport_traffic_days_,
                           (void*)&gtt_->date_range_,
                           (void*)&gtt_->locations_.transfer_time_,
                           (void*)&gtt_->locations_.gpu_footpaths_in_,
                           (void*)&gtt_->locations_.gpu_footpaths_out_,
                           (void*)&gtt_->route_clasz_};
    launch_kernel(kernel_args, mem_->context_, mem_->context_.proc_stream_,SearchDir,Rt);
    copy_back(mem_);
    //copy stats from host to raptor attribute
    gpu_raptor_stats tmp{};
    for (int i = 0; i<32; ++i) {
      tmp.n_routing_time_ += mem_->host_.stats_[i].n_routing_time_;
      tmp.n_footpaths_visited_ += mem_->host_.stats_[i].n_footpaths_visited_;
      tmp.n_routes_visited_ += mem_->host_.stats_[i].n_routes_visited_;
      tmp.n_earliest_trip_calls_ += mem_->host_.stats_[i].n_earliest_trip_calls_;
      tmp.n_earliest_arrival_updated_by_route_ += mem_->host_.stats_[i].n_earliest_arrival_updated_by_route_;
      tmp.n_earliest_arrival_updated_by_footpath_ += mem_->host_.stats_[i].n_earliest_arrival_updated_by_footpath_;
      tmp.fp_update_prevented_by_lower_bound_ += mem_->host_.stats_[i].fp_update_prevented_by_lower_bound_;
      tmp.route_update_prevented_by_lower_bound_ += mem_->host_.stats_[i].route_update_prevented_by_lower_bound_;
    }

    stats_ = tmp;
    destroy_copy_to_gpu_args(start_time_ptr,worst_time_at_dest_ptr,prf_idx_ptr);
  }
  gpu_timetable const* gtt_{nullptr};
  mem* mem_{nullptr};
  bool* is_dest_{nullptr};
  uint16_t* dist_to_end_{nullptr};
  uint32_t* dist_to_end_size_{nullptr};
  uint16_t* lb_{nullptr};
  gpu_day_idx_t* base_{nullptr};
  gpu_day_idx_t cpu_base_;
  int* n_days_{nullptr};
  gpu_raptor_stats stats_;
  gpu_clasz_mask_t* allowed_claszes_{nullptr};
  std::uint16_t* kUnreachable_{nullptr};
  gpu_location_idx_t* kIntermodalTarget_{nullptr};
  short* kMaxTravelTimeTicks_{nullptr};
  std::vector<gpu_delta_t> best_;
  std::vector<gpu_delta_t> round_times_;
  std::vector<uint32_t> station_mark_;
  bool added_start_;
};
