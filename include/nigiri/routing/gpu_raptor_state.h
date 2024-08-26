#pragma once
#include "nigiri/routing/cuda_util.h"
#include "nigiri/routing/gpu_timetable.h"
#include <atomic>
#include <boost/url/grammar/error.hpp>
#include <memory>
#include <mutex>
#include <type_traits>
#include <vector>

struct cudaDeviceProp;
namespace std {
class mutex;
}

struct gpu_raptor_stats {
  __host__ __device__ gpu_raptor_stats(std::uint64_t const& n_routing_time = 0ULL,
               std::uint64_t const& n_footpaths_visited = 0ULL,
               std::uint64_t const& n_routes_visited = 0ULL,
               std::uint64_t const &n_earliest_trip_calls = 0ULL,
               std::uint64_t const& n_earliest_arrival_updated_by_route = 0ULL,
               std::uint64_t const& n_earliest_arrival_updated_by_footpath = 0ULL,
               std::uint64_t const& fp_update_prevented_by_lower_bound = 0ULL,
               std::uint64_t const& route_update_prevented_by_lower_bound = 0ULL)
      : n_routing_time_(n_routing_time),
        n_footpaths_visited_(n_footpaths_visited),
        n_routes_visited_(n_routes_visited),
        n_earliest_trip_calls_(n_earliest_trip_calls),
        n_earliest_arrival_updated_by_route_(n_earliest_arrival_updated_by_route),
        n_earliest_arrival_updated_by_footpath_(n_earliest_arrival_updated_by_footpath),
        fp_update_prevented_by_lower_bound_(fp_update_prevented_by_lower_bound),
        route_update_prevented_by_lower_bound_(route_update_prevented_by_lower_bound) {}

  std::uint64_t n_routing_time_{0ULL};
  std::uint64_t n_footpaths_visited_{0ULL};
  std::uint64_t n_routes_visited_{0ULL};
  std::uint64_t n_earliest_trip_calls_{0ULL};
  std::uint64_t n_earliest_arrival_updated_by_route_{0ULL};
  std::uint64_t n_earliest_arrival_updated_by_footpath_{0ULL};
  std::uint64_t fp_update_prevented_by_lower_bound_{0ULL};
  std::uint64_t route_update_prevented_by_lower_bound_{0ULL};
};
using device_id = int32_t;

std::pair<dim3, dim3> get_launch_parameters(cudaDeviceProp const& prop,
                                           int32_t concurrency_per_device);

struct device_context {
  device_context() = delete;
  device_context(device_context const&) = delete;
  device_context(device_context const&&) = delete;
  device_context operator=(device_context const&) = delete;
  device_context operator=(device_context const&&) = delete;
  device_context(device_id device_id);

  ~device_context() = default;

  void destroy();

  device_id id_{};
  cudaDeviceProp props_{};

  dim3 threads_per_block_;
  dim3 grid_;

  cudaStream_t proc_stream_{};
  cudaStream_t transfer_stream_{};
};

struct host_memory {
  host_memory() = delete;
  host_memory(host_memory const&) = delete;
  host_memory(host_memory const&&) = delete;
  host_memory operator=(host_memory const&) = delete;
  host_memory operator=(host_memory const&&) = delete;
  explicit host_memory(uint32_t row_count_round_times, uint32_t column_count_round_times,uint32_t n_locations,uint32_t n_routes);

  ~host_memory() = default;


  std::vector<gpu_delta_t> round_times_; // round_times ist flat_matrix -> mit entries_ auf alle Elemente zugreifen
  std::vector<gpu_raptor_stats> stats_;
  std::vector<gpu_delta_t> tmp_;
  std::vector<gpu_delta_t> best_;
  std::vector<uint32_t> station_mark_;
  std::vector<uint32_t> prev_station_mark_;
  std::vector<uint32_t> route_mark_;
  uint32_t row_count_round_times_;
  uint32_t column_count_round_times_;
};



struct device_memory {
  device_memory() = delete;
  device_memory(device_memory const&) = delete;
  device_memory(device_memory const&&) = delete;
  device_memory operator=(device_memory const&) = delete;
  device_memory operator=(device_memory const&&) = delete;
  device_memory(uint32_t n_locations, uint32_t n_routes, uint32_t row_count_round_times_, uint32_t column_count_round_times_,gpu_delta_t invalid);

  ~device_memory() = default;

  void next_start_time_async(cudaStream_t s);
  void reset_arrivals_async(cudaStream_t s);
  void destroy();

  // vielleicht getter Methoden
  /*
  void resize(unsigned n_locations,
              unsigned n_routes);
  */

  void reset_async(cudaStream_t s);

  gpu_delta_t* tmp_{};
  gpu_delta_t* best_{};
  gpu_delta_t* round_times_{}; // round_times ist flat_matrix -> mit entries_ auf alle Elemente zugreifen
  gpu_delta_t* time_at_dest_{};
  gpu_delta_t invalid_{};
  //uint32_t da wir 32 Threads haben die jeweils ihre route die marks setzen
  uint32_t* station_mark_{};
  uint32_t* prev_station_mark_{};
  uint32_t* route_mark_{};
  bool* any_station_marked_{};
  uint32_t n_locations_{};
  uint32_t n_routes_{};
  uint32_t row_count_round_times_{}
  ;uint32_t column_count_round_times_{};
  gpu_raptor_stats* stats_{};
};

struct mem {
  mem() = delete;
  mem(mem const&) = delete;
  mem(mem const&&) = delete;
  mem operator=(mem const&) = delete;
  mem operator=(mem const&&) = delete;

  mem(uint32_t n_locations, uint32_t n_routes, uint32_t row_count_round_times_, uint32_t column_count_round_times_,gpu_delta_t invalid,
      device_id device_id);

  void reset_arrivals_async();
  void next_start_time_async();
  ~mem();

  host_memory host_;
  device_memory device_;
  device_context context_;
};


struct gpu_raptor_state {
  using mem_idx = uint32_t;
  static_assert(std::is_unsigned_v<mem_idx>);

  gpu_raptor_state() = default;
  void init(gpu_timetable const& gtt,gpu_delta_t invalid);

  mem_idx get_mem_idx();

  std::atomic<mem_idx> current_idx_{0};
  static_assert(std::is_unsigned_v<decltype(current_idx_)::value_type>);

  std::vector<std::unique_ptr<mem>> memory_;
  std::vector<std::mutex> memory_mutexes_;
};

static_assert(
    std::is_unsigned_v<decltype(std::declval<gpu_raptor_state>().get_mem_idx())>);

struct loaned_mem {
  loaned_mem() = delete;
  loaned_mem(loaned_mem const&) = delete;
  loaned_mem(loaned_mem const&&) = delete;
  loaned_mem operator=(loaned_mem const&) = delete;
  loaned_mem operator=(loaned_mem const&&) = delete;

  explicit loaned_mem(gpu_raptor_state& store,gpu_delta_t invalid);

  ~loaned_mem();

  mem* mem_{nullptr};
  std::unique_lock<std::mutex> lock_{};
};


