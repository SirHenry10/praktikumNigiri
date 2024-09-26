#pragma once

#include "nigiri/routing/gpu_raptor_state.h"
#include <iostream>


#include <cuda_runtime.h>

std::pair<dim3, dim3> get_launch_paramters(
    cudaDeviceProp const& prop, int32_t const concurrency_per_device) {
   //TODO: funktioniert nicht wie bei julian
   int32_t block_dim_x = 32;  // must always be 32!
   int32_t block_dim_y = 4;  // range [1, ..., 32]
   int32_t block_size = block_dim_x * block_dim_y;

   auto const mp_count = prop.multiProcessorCount / concurrency_per_device;

     //TODO: Changed for GTX 1080 herausfinden wie allgemein halten
   int32_t max_blocks_per_sm = 2;
   int32_t num_blocks = mp_count * max_blocks_per_sm;

   int32_t num_sms = prop.multiProcessorCount;  // 20 SMs bei GTX 1080
   int32_t total_blocks = num_sms * max_blocks_per_sm;

   dim3 threads_per_block(block_dim_x, block_dim_y, 1);
   dim3 grid(total_blocks, 1, 1);  // Grid auf 40 Blöcke setzen
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
  std::cerr << "device_context ende" << std::endl;
}

void device_context::destroy() {
  std::cerr << "device_context destroy!!!" << std::endl;
  cudaSetDevice(id_);
  cudaStreamDestroy(proc_stream_);
  proc_stream_ = cudaStream_t{};
  cudaStreamDestroy(transfer_stream_);
  transfer_stream_ = cudaStream_t{};
  cuda_check();
}

// Attribute, die von Host benötigt werden
host_memory::host_memory(uint32_t n_locations,
                         uint32_t n_routes,
                         uint32_t row_count_round_times,
                         uint32_t column_count_round_times
                         ):row_count_round_times_{row_count_round_times},
                             column_count_round_times_{column_count_round_times},
                             round_times_(row_count_round_times*column_count_round_times),
                             stats_(32),
                             tmp_(n_locations),
                             best_(n_locations),
                             station_mark_(n_locations),
                             prev_station_mark_(n_locations),
                             route_mark_(n_routes){}

// Zuweisung von Speicherplatz an Attribute, die in devices verwendet werden
device_memory::device_memory(uint32_t n_locations,
                             uint32_t n_routes,
                             uint32_t row_count_round_times,
                             uint32_t column_count_round_times,
                             gpu_delta_t invalid)
    : n_locations_{n_locations},
      n_routes_{n_routes},
      row_count_round_times_{row_count_round_times},
      column_count_round_times_{column_count_round_times}{
  tmp_ = nullptr;
  cudaMalloc(&tmp_, n_locations_ * sizeof(gpu_delta_t));
  cuda_check();
  time_at_dest_ = nullptr;
  cudaMalloc(&time_at_dest_, (gpu_kMaxTransfers+1) *sizeof(gpu_delta_t));
  cuda_check();
  best_ = nullptr;
  cudaMalloc(&best_, n_locations_ * sizeof(gpu_delta_t));
  cuda_check();
  round_times_ = nullptr;
  cudaMalloc(&round_times_, row_count_round_times_ * column_count_round_times_ *
                                sizeof(gpu_delta_t));
  cuda_check();
  station_mark_ = nullptr;
  cudaMalloc(&station_mark_, ((n_locations_/32)+1) * sizeof(uint32_t));
  cuda_check();
  prev_station_mark_ = nullptr;
  cudaMalloc(&prev_station_mark_, ((n_locations_/32)+1) * sizeof(uint32_t));
  cuda_check();
  route_mark_ = nullptr;
  cudaMalloc(&route_mark_, ((n_routes_/32)+1) * sizeof(uint32_t));
  cuda_check();
  any_station_marked_ = nullptr;
  cudaMalloc(&any_station_marked_, sizeof(bool));
  cuda_check();
  stats_ = nullptr;
  cudaMalloc(&stats_,32*sizeof(gpu_raptor_stats));
  cuda_check();
  invalid_ = invalid;
  cudaDeviceSynchronize();
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
  cudaFree(any_station_marked_);
  cudaFree(stats_);
}

void device_memory::reset_async(cudaStream_t s) {
  std::cerr << "reset_all" << std::endl;
  std::vector<gpu_delta_t> invalid_time_at_dest((gpu_kMaxTransfers+1), invalid_);
  cudaMemcpyAsync(time_at_dest_, invalid_time_at_dest.data(), (gpu_kMaxTransfers+1) * sizeof(gpu_delta_t), cudaMemcpyHostToDevice, s);
  std::vector<gpu_delta_t> invalid_n_locations(n_locations_, invalid_);
  cudaMemcpyAsync(tmp_,invalid_n_locations.data(), n_locations_ * sizeof(gpu_delta_t), cudaMemcpyHostToDevice, s);
  cudaMemcpyAsync(best_,invalid_n_locations.data(), n_locations_ * sizeof(gpu_delta_t), cudaMemcpyHostToDevice, s);
  std::vector<gpu_delta_t> invalid_round_times(column_count_round_times_*row_count_round_times_, invalid_);
  cudaMemcpyAsync(round_times_,invalid_round_times.data(),column_count_round_times_*row_count_round_times_ * sizeof(gpu_delta_t), cudaMemcpyHostToDevice, s);
  cudaMemsetAsync(station_mark_, 0000, ((n_locations_/32)+1)*sizeof(uint32_t), s);
  cudaMemsetAsync(prev_station_mark_, 0000, ((n_locations_/32)+1)*sizeof(uint32_t), s);
  cudaMemsetAsync(route_mark_, 0000, ((n_routes_/32)+1)*sizeof(uint32_t), s);
  cudaMemsetAsync(any_station_marked_, 0000, sizeof(bool), s);
  gpu_raptor_stats init_value = {};

  for (int i = 0; i < 32; ++i) {
    cudaMemcpyAsync(&stats_[i], &init_value, sizeof(gpu_raptor_stats), cudaMemcpyHostToDevice, s);
  }
  //additional_start_count_ = invalid<decltype(additional_start_count_)>;
}
void device_memory::next_start_time_async(cudaStream_t s) {
  std::cerr << "reset_start_time" << std::endl;
  std::vector<gpu_delta_t> invalid_n_locations(n_locations_, invalid_);
  cudaMemcpyAsync(tmp_,invalid_n_locations.data(), n_locations_ * sizeof(gpu_delta_t), cudaMemcpyHostToDevice, s);
  cudaMemcpyAsync(best_,invalid_n_locations.data(), n_locations_ * sizeof(gpu_delta_t), cudaMemcpyHostToDevice, s);
  cudaMemsetAsync(station_mark_, 0000, ((n_locations_/32)+1)*sizeof(uint32_t), s);
  cudaMemsetAsync(prev_station_mark_, 0000, ((n_locations_/32)+1)*sizeof(uint32_t), s);
  cudaMemsetAsync(route_mark_, 0000, ((n_routes_/32)+1)*sizeof(uint32_t), s);
}
void device_memory::reset_arrivals_async(cudaStream_t s) {
  std::cerr << "reset_arrivals async start" << std::endl;
  std::vector<gpu_delta_t> invalid_time_at_dest((gpu_kMaxTransfers+1), invalid_);

  size_t size = (gpu_kMaxTransfers + 1) * sizeof(gpu_delta_t);
  std::cerr << "reset_arrivals async test1" << std::endl;
  cudaMemcpyAsync(time_at_dest_, invalid_time_at_dest.data(), (gpu_kMaxTransfers+1) * sizeof(gpu_delta_t), cudaMemcpyHostToDevice, s);

  std::cerr << "reset_arrivals async mid" << std::endl;
  std::vector<gpu_delta_t> invalid_round_times(column_count_round_times_*row_count_round_times_, invalid_);
  cudaMemcpyAsync(round_times_,invalid_round_times.data(),column_count_round_times_*row_count_round_times_ * sizeof(gpu_delta_t), cudaMemcpyHostToDevice, s);
  std::cerr << "reset_arrivals async end" << std::endl;
}
mem::mem(uint32_t n_locations,
         uint32_t n_routes,
         uint32_t row_count_round_times_,
         uint32_t column_count_round_times_,
         gpu_delta_t invalid,
         device_id const device_id)
    : host_{n_locations,n_routes, row_count_round_times_, column_count_round_times_},
      device_{n_locations, n_routes, row_count_round_times_, column_count_round_times_, invalid},
      context_{device_id} {}

mem::~mem() {
  device_.destroy();
  context_.destroy();
}

void gpu_raptor_state::init(gpu_timetable const& gtt,gpu_delta_t invalid) {
  int32_t device_count = 0;
  cudaGetDeviceCount(&device_count);

  for (auto device_id = 0; device_id < device_count; ++device_id) {
      memory_.emplace_back(std::make_unique<struct mem>(
        gtt.n_locations_,gtt.n_routes_,gpu_kMaxTransfers + 1U,gtt.n_locations_,invalid, device_id));
  }
  memory_mutexes_ = std::vector<std::mutex>(memory_.size());
}

gpu_raptor_state::mem_idx gpu_raptor_state::get_mem_idx() {
  return current_idx_.fetch_add(1) % memory_.size();
}


loaned_mem::loaned_mem(gpu_raptor_state& store,gpu_delta_t invalid) {
  auto const idx = store.get_mem_idx();
  lock_ = std::unique_lock(store.memory_mutexes_[idx]);
  mem_ = std::move(store.memory_[idx]);
  mem_.get()->device_.invalid_ = invalid;
}

loaned_mem::~loaned_mem() {
  if (mem_ != nullptr) {
    mem_->device_.reset_async(mem_->context_.proc_stream_);
    cuda_sync_stream(mem_->context_.proc_stream_);
  }
}
void mem::reset_arrivals_async(){
  device_.reset_arrivals_async(context_.proc_stream_);
  cuda_sync_stream(context_.proc_stream_);
  //TODO: weiß nicht ob ucda_sync_stream?
  std::cerr << "reset_arrivals async ende ende" << std::endl;
}
void mem::next_start_time_async(){
  device_.next_start_time_async(context_.proc_stream_);
  cuda_sync_stream(context_.proc_stream_);
}