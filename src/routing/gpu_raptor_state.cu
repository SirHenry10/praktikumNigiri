#pragma once

#include "nigiri/routing/gpu_raptor_state.cuh"


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
void host_memory::reset(gpu_delta_t invalid) const {
  for (auto k = 0U; k != row_count_round_times_; ++k) {
    for (auto l = 0U; l != column_count_round_times_; ++l) {
      round_times_[k*row_count_round_times_+l] = invalid;
    }
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
  cudaMalloc(&time_at_dest_, (nigiri::routing::kMaxTransfers+1) *sizeof(gpu_delta_t));
  cudaMalloc(&best_, size_best_ * sizeof(gpu_delta_t));
  cudaMalloc(&round_times_, row_count_round_times_ * column_count_round_times_ *
                                sizeof(gpu_delta_t));
  cudaMalloc(&station_mark_, size_station_mark_ * sizeof(uint32_t));
  cudaMalloc(&prev_station_mark_, size_station_mark_ * sizeof(uint32_t));
  cudaMalloc(&route_mark_, size_route_mark_ * sizeof(uint32_t));
  cudaMalloc(&any_station_marked_, sizeof(bool));
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
}


void device_memory::reset_async(cudaStream_t s) {
  cudaMemsetAsync(time_at_dest_,invalid_, (nigiri::routing::kMaxTransfers+1)*sizeof(gpu_delta_t), s);
  cudaMemsetAsync(tmp_,invalid_, size_tmp_*sizeof(gpu_delta_t), s);
  cudaMemsetAsync(best_, invalid_, size_best_*sizeof(gpu_delta_t), s);
  cudaMemsetAsync(round_times_, invalid_, column_count_round_times_*row_count_round_times_*sizeof(gpu_delta_t), s);
  cudaMemsetAsync(station_mark_, 0, size_station_mark_*sizeof(uint32_t), s);
  cudaMemsetAsync(prev_station_mark_, 0, size_station_mark_*sizeof(uint32_t), s);
  cudaMemsetAsync(route_mark_, 0, size_route_mark_*sizeof(uint32_t), s);
  cudaMemsetAsync(any_station_marked_, 0, sizeof(bool), s);
  //additional_start_count_ = invalid<decltype(additional_start_count_)>;
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
        *gtt.n_locations_,*gtt.n_locations_,nigiri::routing::kMaxTransfers + 1U,*gtt.n_locations_,*gtt.n_locations_,*gtt.n_routes_,invalid, device_id));
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
void device_memory::print(const gpu_timetable& gtt, date::sys_days sys_days, gpu_delta_t invalid) {
  auto const has_empty_rounds = [&](std::uint32_t const l) {
    for (auto k = 0U; k != row_count_round_times_; ++k) {
      if (round_times_[k*row_count_round_times_+l] != invalid) {
        return false;
      }
    }
    return true;
  };

  auto const print_delta = [&](gpu_delta_t const gd) {
    if (gd == invalid) {
      fmt::print("________________");
    } else {
      fmt::print("{:16}", gpu_delta_to_unix(sys_days,gd));
    }
  };

  for (auto l = 0U; l != *gtt.n_locations_; ++l) {
    if (best_[l] == invalid && has_empty_rounds(l)) {
      continue;
    }

    fmt::print("{:80}  ", location{gtt, location_idx_t{l}});

    auto const b = best_[l];
    fmt::print("best=");
    print_delta(b);
    fmt::print(", round_times: ");
    for (auto i = 0U; i != row_count_round_times_; ++i) {
      auto const t = round_times_[i*row_count_round_times_+l];
      print_delta(t);
      fmt::print(" ");
    }
    fmt::print("\n");
  }
}
