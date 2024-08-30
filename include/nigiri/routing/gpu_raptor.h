#pragma once

#include <iostream>
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
void copy_back(mem*& mem);

std::unique_ptr<mem> gpu_mem(
    std::vector<gpu_delta_t>& tmp,
    std::vector<gpu_delta_t>& best,
    std::vector<bool>& station_mark,
    std::vector<bool>& prev_station_mark,
    std::vector<bool>& route_mark,
    gpu_direction search_dir,
    gpu_timetable* gtt);
void add_start_gpu(gpu_location_idx_t const l, gpu_unixtime_t const t,mem* mem_,gpu_timetable* gtt_,gpu_day_idx_t base_,short const kInvalid);

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
__device__ inline gpu_sys_days base(gpu_day_idx_t* base,gpu_interval<gpu_sys_days>* date_range_ptr) {
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
    return t.data();
  } else {
    return t.data() + t.size() - 1;
  }
}

template <gpu_direction SearchDir, typename T>
__host__ __device__ auto gpu_get_end_it(T const& t) {
  if constexpr ((SearchDir == gpu_direction::kForward)) {
    return t.data() + t.size();;
  } else {
    return t.data() - 1;
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


  gpu_raptor(gpu_timetable* gtt,
             mem* mem,
         std::vector<bool>& is_dest,
         std::vector<std::uint16_t>& dist_to_dest,
         std::vector<std::uint16_t>& lb,
         gpu_day_idx_t const& base,
         gpu_clasz_mask_t const& allowed_claszes,
             int const& n_days)
      : gtt_{gtt},
        mem_{mem}//ToDO: verwaltung von mem nicht übergebn drausen lassen...
        {
    std::cerr << "gpu_raptor()" << std::endl;
    mem_->reset_arrivals_async();
    std::cerr << "gpu_raptor() before copy_array" << std::endl;
    std::unique_ptr<bool[]> copy_array(new bool[is_dest.size()]);
    for (int i = 0; i<is_dest.size();i++){
      copy_array[i] = is_dest[i];
    }
    std::cerr << "gpu_raptor() 1 " << dist_to_dest.size() << std::endl;
    auto const kIntermodalTarget  =
        gpu_to_idx(get_gpu_special_station(gpu_special_station::kEnd));
    std::cerr << "gpu_raptor() before copy_to_Devices" << std::endl;
    cpu_base_ = base;
    copy_to_devices(allowed_claszes,
                    dist_to_dest,
                    base,
                    copy_array,
                    is_dest.size(),
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
  }
  ~gpu_raptor(){

    std::cerr << "gpu_raptor() destroy" << std::endl;
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
    std::cerr << "Test gpu_raptor::add_start() start" << std::endl;
      add_start_gpu(l,t,mem_,gtt_,cpu_base_,kInvalid);
      std::cerr << "Test gpu_raptor::add_start() end" << std::endl;
  }


  // hier wird Kernel aufgerufen
  void execute(gpu_unixtime_t const& start_time,
             uint8_t const& max_transfers,
             gpu_unixtime_t const& worst_time_at_dest,
             gpu_profile_idx_t const& prf_idx){
    std::cerr << "Test gpu_raptor::execute() start" << std::endl;
    //start_time muss rüber das bei trace,max_transfers muss nicht malloced werden, worst_time_at_Dest muss rüber kopiert werden, prf_idx muss kopiert werden
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
    std::cerr << "Test gpu_raptor::launch_kernel() bevor mem" << std::endl;
    copy_back(mem_);
    std::cerr << "Test gpu_raptor::launch_kernel() bevor mem2" << std::endl;
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
    std::cerr << "n_routing_time"<<tmp.n_routes_visited_ << std::endl;
    stats_ = tmp;
    std::cerr << "Test gpu_raptor::execute() bevor destroy" << std::endl;
    destroy_copy_to_gpu_args(start_time_ptr,worst_time_at_dest_ptr,prf_idx_ptr);
    std::cerr << "Test gpu_raptor::execute() ende" << std::endl;
  }
  gpu_timetable* gtt_{nullptr};
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
};
