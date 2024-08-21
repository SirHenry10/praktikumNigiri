#pragma once

#include "nigiri/routing/gpu_raptor_state.h"

#include <cuda_runtime.h>

std::pair<dim3, dim3> get_launch_paramters(
    cudaDeviceProp const& prop, int32_t const concurrency_per_device) {
  int32_t block_dim_x = 32;  // must always be 32!
  int32_t block_dim_y = 32;  // range [1, ..., 32]
  int32_t block_size = block_dim_x * block_dim_y;
  int32_t max_blocks_per_sm = prop.maxThreadsPerMultiProcessor / block_size;

  auto const mp_count = prop.multiProcessorCount / concurrency_per_device;

  int32_t num_blocks = mp_count * max_blocks_per_sm;

  dim3 threads_per_block(block_dim_x, block_dim_y, 1);
  dim3 grid(num_blocks, 1, 1);

  return {threads_per_block, grid};
}

device_context::device_context(device_id const device_id)
    : id_(device_id) {
  cudaSetDevice(id_);
  cuda_check();

  cudaGetDeviceProperties(&props_, device_id);
  cuda_check();

  std::tie(threads_per_block_, grid_) =
      get_launch_paramters(props_, 1);

  cudaStreamCreate(&proc_stream_);
  cuda_check();
  cudaStreamCreate(&transfer_stream_);
  cuda_check();
}

void device_context::destroy() {
  cudaSetDevice(id_);
  cudaStreamDestroy(proc_stream_);
  proc_stream_ = cudaStream_t{};
  cudaStreamDestroy(transfer_stream_);
  transfer_stream_ = cudaStream_t{};
  cuda_check();
}

// Attribute, die von Host benötigt werden
host_memory::host_memory(uint32_t row_count_round_times,
                         uint32_t column_count_round_times
                         ):row_count_round_times_{row_count_round_times},column_count_round_times_{column_count_round_times},
round_times_{std::make_unique<gpu_delta_t>(row_count_round_times*column_count_round_times)},
stats_{std::make_unique<gpu_raptor_stats>(32)}{}

void host_memory::destroy() {
  round_times_ = nullptr;
  stats_ = nullptr;
}
void host_memory::reset(gpu_delta_t invalid) const {
  auto rt = round_times_.get();
  for (auto k = 0U; k != row_count_round_times_; ++k) {
    for (auto l = 0U; l != column_count_round_times_; ++l) {
      rt[k * row_count_round_times_ + l] = invalid;
    }
  }
  auto s = stats_.get();
  gpu_raptor_stats init = {};
  for (auto i = 0U; i < 32; ++i) {
    s[i] = init;
  }
}
// Zuweisung von Speicherplatz an Attribute, die in devices verwendet werden
device_memory::device_memory(uint32_t size_tmp,
                             uint32_t size_best,
                             uint32_t row_count_round_times,
                             uint32_t column_count_round_times,
                             uint32_t size_station_mark,
                             //uint32_t size_prev_station_mark,
                             uint32_t size_route_mark,
                             gpu_delta_t invalid)
    : size_tmp_{size_tmp},
      size_best_{size_best},
      row_count_round_times_{row_count_round_times},
      column_count_round_times_{column_count_round_times},
      size_station_mark_{size_station_mark},
      //size_prev_station_mark_{size_prev_station_mark},
      size_route_mark_{size_route_mark}{


  cudaMalloc(&tmp_, size_tmp_ * sizeof(gpu_delta_t));
  time_at_dest_ = nullptr;
  cudaMalloc(&time_at_dest_, (gpu_kMaxTransfers+1) *sizeof(gpu_delta_t));
  cudaMalloc(&best_, size_best_ * sizeof(gpu_delta_t));
  cudaMalloc(&round_times_, row_count_round_times_ * column_count_round_times_ *
                                sizeof(gpu_delta_t));
  cudaMalloc(&station_mark_, size_station_mark_ * sizeof(uint32_t));
  cudaMalloc(&prev_station_mark_, size_station_mark_ * sizeof(uint32_t));
  cudaMalloc(&route_mark_, size_route_mark_ * sizeof(uint32_t));
  cudaMalloc(&any_station_marked_, sizeof(bool));
  cudaMalloc(&stats_,32*sizeof(gpu_raptor_stats));
  invalid_ = invalid;
  cuda_check();
  this->reset_async(nullptr);
}

void device_memory::destroy() {
  cudaFree(time_at_dest_);
  cudaFree(tmp_);
  cudaFree(best_);
  cudaFree(round_times_);
  cudaFree(station_mark_);
  cudaFree(prev_station_mark_);
  cudaFree(route_mark_);
  cudaFree(stats_);
}

void device_memory::reset_async(cudaStream_t s) {
  cudaMemsetAsync(time_at_dest_,invalid_, (gpu_kMaxTransfers+1)*sizeof(gpu_delta_t), s);
  cudaMemsetAsync(tmp_,invalid_, size_tmp_*sizeof(gpu_delta_t), s);
  cudaMemsetAsync(best_, invalid_, size_best_*sizeof(gpu_delta_t), s);
  cudaMemsetAsync(round_times_, invalid_, column_count_round_times_*row_count_round_times_*sizeof(gpu_delta_t), s);
  cudaMemsetAsync(station_mark_, 0, size_station_mark_*sizeof(uint32_t), s);
  cudaMemsetAsync(prev_station_mark_, 0, size_station_mark_*sizeof(uint32_t), s);
  cudaMemsetAsync(route_mark_, 0, size_route_mark_*sizeof(uint32_t), s);
  cudaMemsetAsync(any_station_marked_, 0, sizeof(bool), s);
  gpu_raptor_stats init_value = {};
  for (int i = 0; i < 32; ++i) {
    cudaMemcpyAsync(&stats_[i], &init_value, sizeof(gpu_raptor_stats), cudaMemcpyHostToDevice, s);
  }
  //additional_start_count_ = invalid<decltype(additional_start_count_)>;
}
void device_memory::next_start_time_async(cudaStream_t s) {
  cudaMemsetAsync(tmp_,invalid_, size_tmp_*sizeof(gpu_delta_t), s);
  cudaMemsetAsync(best_, invalid_, size_best_*sizeof(gpu_delta_t), s);
  cudaMemsetAsync(station_mark_, 0, size_station_mark_*sizeof(uint32_t), s);
  cudaMemsetAsync(prev_station_mark_, 0, size_station_mark_*sizeof(uint32_t), s);
  cudaMemsetAsync(route_mark_, 0, size_route_mark_*sizeof(uint32_t), s);
}
void device_memory::reset_arrivals_async(cudaStream_t s) {
  cudaMemsetAsync(time_at_dest_,invalid_, (gpu_kMaxTransfers+1)*sizeof(gpu_delta_t), s);
  cudaMemsetAsync(round_times_, invalid_, column_count_round_times_*row_count_round_times_*sizeof(gpu_delta_t), s);
}
mem::mem(uint32_t size_tmp_, uint32_t size_best_,
         uint32_t row_count_round_times_, uint32_t column_count_round_times_,
         uint32_t size_station_mark_, uint32_t size_route_mark_,gpu_delta_t invalid,
         device_id const device_id)
    : host_{row_count_round_times_, column_count_round_times_},
      device_{size_tmp_, size_best_, row_count_round_times_, column_count_round_times_, size_station_mark_,  size_route_mark_, invalid},
      context_{device_id} {}

mem::~mem() {
  host_.destroy();
  device_.destroy();
  context_.destroy();
}

void gpu_raptor_state::init(gpu_timetable const& gtt,gpu_delta_t invalid) {
  int32_t device_count = 0;
  cudaGetDeviceCount(&device_count);


  for (auto device_id = 0; device_id < device_count; ++device_id) {
      memory_.emplace_back(std::make_unique<struct mem>(
        *gtt.n_locations_,*gtt.n_locations_,gpu_kMaxTransfers + 1U,*gtt.n_locations_,*gtt.n_locations_,*gtt.n_routes_,invalid, device_id));
  }
  memory_mutexes_ = std::vector<std::mutex>(memory_.size());
}

gpu_raptor_state::mem_idx gpu_raptor_state::get_mem_idx() {
  return current_idx_.fetch_add(1) % memory_.size();
}


loaned_mem::loaned_mem(gpu_raptor_state& store,gpu_delta_t invalid) {
  auto const idx = store.get_mem_idx();
  lock_ = std::unique_lock(store.memory_mutexes_[idx]);
  mem_ = store.memory_[idx].get();
  mem_->device_.invalid_ = invalid;
}

loaned_mem::~loaned_mem() {
  mem_->device_.reset_async(mem_->context_.proc_stream_);
  mem_->host_.reset(mem_->device_.invalid_);
  cuda_sync_stream(mem_->context_.proc_stream_);
}
void mem::reset_arrivals_async(){
  device_.reset_arrivals_async(context_.proc_stream_);
  cuda_sync_stream(context_.proc_stream_);
}
void mem::next_start_time_async(){
  device_.next_start_time_async(context_.proc_stream_);
  cuda_sync_stream(context_.proc_stream_);
}