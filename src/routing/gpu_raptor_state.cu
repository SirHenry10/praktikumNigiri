#pragma once

#include "nigiri/routing/gpu_raptor_state.cuh"

namespace nigiri::routing {

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
host_memory::host_memory(uint32_t row_count_round_times_,
                         uint32_t column_count_round_times_) {
  cudaMallocHost(
      &round_times_,
      row_count_round_times_ * column_count_round_times_ * sizeof(gpu_delta_t));
}

void host_memory::destroy() {
  cudaFreeHost(round_times_);
  round_times_ = nullptr;
}
/* Brauchen wir glaube nicht
void host_memory::reset() const {
  *any_station_marked_ = false;
  result_->reset();
}
 */

// Zuweisung von Speicherplatz an Attribute, die in devices verwendet werden
device_memory::device_memory(uint32_t size_tmp,
                             uint32_t size_best,
                             uint32_t row_count_round_times,
                             uint32_t column_count_round_times,
                             uint32_t size_station_mark,
                             uint32_t size_prev_station_mark,
                             uint32_t size_route_mark)
    : size_tmp_{size_tmp},
      size_best_{size_best},
      row_count_round_times_{row_count_round_times},
      column_count_round_times_{column_count_round_times},
      size_station_mark_{size_station_mark},
      size_prev_station_mark_{size_prev_station_mark},
      size_route_mark_{size_route_mark} {

  cudaMalloc(&tmp_, size_tmp_ * sizeof(gpu_delta_t));
  cudaMalloc(&best_, size_best_ * sizeof(gpu_delta_t));
  cudaMalloc(&round_times_, row_count_round_times_ * column_count_round_times_ *
                                sizeof(gpu_delta_t));
  cudaMalloc(&station_mark_, size_station_mark_ * sizeof(bool));
  cudaMalloc(&prev_station_mark_, size_prev_station_mark_ * sizeof(bool));
  cudaMalloc(&route_mark_, size_route_mark_ * sizeof(bool));
  cuda_check();
  // result inizialisieren??
  /*
  for (auto k = 1U; k < result_.size(); ++k) {
    result_[k] = result_[k - 1] + stop_count;
  }
  */
  this->reset_async(nullptr);
}

void device_memory::destroy() {
  cudaFree(tmp_);
  cudaFree(best_);
  cudaFree(round_times_);
  cudaFree(station_mark_);
  cudaFree(prev_station_mark_);
  cudaFree(route_mark_);
}

void device_memory::reset_async(cudaStream_t s) {
  cudaMemsetAsync(tmp_, 0xFF, size_tmp_*sizeof(gpu_delta_t), s);
  cudaMemsetAsync(best_, 0xFF, size_best_*sizeof(gpu_delta_t), s);
  cudaMemsetAsync(round_times_, 0xFF, column_count_round_times_*row_count_round_times_*sizeof(gpu_delta_t), s);
  cudaMemsetAsync(station_mark_, 0, size_station_mark_*sizeof(bool), s);
  cudaMemsetAsync(prev_station_mark_, 0, size_prev_station_mark_*sizeof(bool), s);
  cudaMemsetAsync(route_mark_, 0, size_route_mark_*sizeof(bool), s);
  //additional_start_count_ = invalid<decltype(additional_start_count_)>;
}

mem::mem(uint32_t size_tmp_, uint32_t size_best_,
         uint32_t row_count_round_times_, uint32_t column_count_round_times_,
         uint32_t size_station_mark_, uint32_t size_prev_station_mark_, uint32_t size_route_mark_,
         device_id const device_id)
    : host_{row_count_round_times_, column_count_round_times_},
      device_{size_tmp_, size_best_, row_count_round_times_, column_count_round_times_, size_station_mark_, size_prev_station_mark_, size_route_mark_},
      context_{device_id} {}

mem::~mem() {
  host_.destroy();
  device_.destroy();
  context_.destroy();
}

void memory_store::init(gpu_timetable const& gtt) {
  int32_t device_count = 0;
  cudaGetDeviceCount(&device_count);


  for (auto device_id = 0; device_id < device_count; ++device_id) {
      memory_.emplace_back(std::make_unique<struct mem>(
        gtt.n_locations_,gtt.n_locations_,kMaxTransfers + 1U,gtt.n_locations_,gtt.n_locations_,gtt.n_locations_,gtt.n_routes_, device_id));
  }

  memory_mutexes_ = std::vector<std::mutex>(memory_.size());
}

memory_store::mem_idx memory_store::get_mem_idx() {
  return current_idx_.fetch_add(1) % memory_.size();
}

loaned_mem::loaned_mem(memory_store& store) {
  auto const idx = store.get_mem_idx();
  lock_ = std::unique_lock(store.memory_mutexes_[idx]);
  mem_ = store.memory_[idx].get();
}

loaned_mem::~loaned_mem() {
  mem_->device_.reset_async(mem_->context_.proc_stream_);
  //mem_->host_.reset();
  cuda_sync_stream(mem_->context_.proc_stream_);
}
}  // namespace nigiri::routing
