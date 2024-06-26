#pragma once
#include <atomic>
#include <boost/url/grammar/error.hpp>
#include <memory>
#include <mutex>
#include <type_traits>
#include <vector>
#include "cuda_util.h"

struct cudaDeviceProp;
namespace std {
class mutex;
}
namespace nigiri::routing {

using device_id = int32_t;

std::pair<dim3, dim3> get_launch_parameters(cudaDeviceProp const& prop,
                                           int32_t concurrency_per_device);

struct device_context {
  device_context() = delete;
  device_context(device_context const&) = delete;
  device_context(device_context const&&) = delete;
  device_context operator=(device_context const&) = delete;
  device_context operator=(device_context const&&) = delete;
  device_context(device_id device_id, int32_t concurrency_per_device);

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
  explicit host_memory(stop_id stop_count);

  ~host_memory() = default;

  void destroy();
  void reset() const;

  std::unique_ptr<raptor_result_pinned> result_{nullptr};
  bool* any_station_marked_{nullptr};
};

struct device_memory {
  device_memory() = delete;
  device_memory(device_memory const&) = delete;
  device_memory(device_memory const&&) = delete;
  device_memory operator=(device_memory const&) = delete;
  device_memory operator=(device_memory const&&) = delete;
  device_memory(uint32_t size_tmp_, uint32_t size_best_, uint32_t row_count_round_times_, uint32_t column_count_round_times_, uint32_t size_station_mark_, uint32_t size_prev_station_mark, uint32_t size_route_mark);

  ~device_memory() = default;

  void destroy();

  // vielleicht getter Methoden
  
  void resize(unsigned n_locations,
              unsigned n_routes);
  void print(timetable const& tt, date::sys_days, delta_t invalid);

  void reset_async(cudaStream_t s);

  // TODO(julian) move from uint32_t to char or something
  delta_t* tmp_, best_, round_times_; // round_times ist flat_matrix -> mit entries_ auf alle Elemente zugreifen
  bool* station_mark_, prev_station_mark_, route_mark_;
  uint32_t size_tmp_, size_best_, row_count_round_times_, column_count_round_times_, size_station_mark_, size_prev_station_mark, size_route_mark;

};

struct mem {
  mem() = delete;
  mem(mem const&) = delete;
  mem(mem const&&) = delete;
  mem operator=(mem const&) = delete;
  mem operator=(mem const&&) = delete;

  mem(uint32_t size_tmp_, uint32_t size_best_, uint32_t row_count_round_times_, uint32_t column_count_round_times_, uint32_t size_station_mark_, uint32_t size_prev_station_mark, uint32_t size_route_mark,
      device_id device_id, int32_t concurrency_per_device);

  ~mem();

  host_memory host_;
  device_memory device_;
  device_context context_;
};

struct memory_store {
  using mem_idx = uint32_t;
  static_assert(std::is_unsigned_v<mem_idx>);

  void init(raptor_meta_info const& meta_info, gpu_timetable const& gtt,
            int32_t concurrency_per_device);

  mem_idx get_mem_idx();

  std::atomic<mem_idx> current_idx_{0};
  static_assert(std::is_unsigned_v<decltype(current_idx_)::value_type>);

  std::vector<std::unique_ptr<mem>> memory_;
  std::vector<std::mutex> memory_mutexes_;
};

static_assert(
    std::is_unsigned_v<decltype(std::declval<memory_store>().get_mem_idx())>);

struct loaned_mem {
  loaned_mem() = delete;
  loaned_mem(loaned_mem const&) = delete;
  loaned_mem(loaned_mem const&&) = delete;
  loaned_mem operator=(loaned_mem const&) = delete;
  loaned_mem operator=(loaned_mem const&&) = delete;
  explicit loaned_mem(memory_store& store);

  ~loaned_mem();

  mem* mem_{nullptr};
  std::unique_lock<std::mutex> lock_{};
};


}  // namespace nigiri::routing
