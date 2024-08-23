#pragma once

#include <cuda_runtime.h>
#include "nigiri/routing/gpu_timetable.h"
#include <cstdio>

#define XSTR(s) STR(s)
#define STR(s) #s

#define CUDA_CALL(call)                                   \
    if ((code = call) != cudaSuccess) {                     \
      printf("CUDA error: %s at " STR(call) " %s:%d\n",     \
             cudaGetErrorString(code), __FILE__, __LINE__); \
      goto fail;                                            \
    }

#define CUDA_COPY_TO_DEVICE(type, target, source, size)                        \
    CUDA_CALL(cudaMalloc(&target, size * sizeof(type)))                          \
    CUDA_CALL(                                                                   \
        cudaMemcpy(target, source, size * sizeof(type), cudaMemcpyHostToDevice)) \
    device_bytes += size * sizeof(type);

template <typename KeyType, typename ValueType>
gpu_vecvec<KeyType, ValueType>* copy_gpu_vecvec_to_device(gpu_vecvec<KeyType, ValueType> const* host_vecvec, size_t& device_bytes, cudaError_t& code) {
  gpu_vecvec<KeyType, ValueType>* device_vecvec = nullptr;


  // Datenstruktur für bucket_starts_
  const gpu_vector<gpu_base_t<KeyType>>* host_bucket_starts_vec = &host_vecvec->bucket_starts_;
  gpu_vector<gpu_base_t<KeyType>>* device_bucket_starts_vec = nullptr;

  const gpu_vector<ValueType>* host_data_vec = &host_vecvec->data_;
  gpu_vector<ValueType>* device_data_vec = nullptr;

  ValueType* device_data = nullptr;

  gpu_base_t<KeyType>* device_bucket_starts = nullptr;


  // Allokiere Speicher für die gpu_vecvec Struktur selbst
  CUDA_CALL(cudaMalloc(&device_vecvec, sizeof(gpu_vecvec<KeyType, ValueType>)));
  CUDA_CALL(cudaMemcpy(device_vecvec, host_vecvec, sizeof(gpu_vecvec<KeyType, ValueType>), cudaMemcpyHostToDevice));
  device_bytes += sizeof(gpu_vecvec<KeyType, ValueType>);


  // Allokiere gpu_basic_vector für data_
  CUDA_CALL(cudaMalloc(&device_data_vec, sizeof(gpu_vector<ValueType>)));
  CUDA_CALL(cudaMemcpy(device_data_vec, host_data_vec, sizeof(gpu_vector<ValueType>), cudaMemcpyHostToDevice));
  device_bytes += sizeof(gpu_vector<ValueType>);

  // Allokiere und kopiere Daten in data_ (el_)
  CUDA_CALL(cudaMalloc(&device_data, host_vecvec->size()* sizeof(ValueType)));
  CUDA_CALL(cudaMemcpy(device_data, host_data_vec->el_, host_data_vec->size() * sizeof(ValueType), cudaMemcpyHostToDevice));
  device_bytes += host_vecvec->size() * sizeof(ValueType);

  // Setze den Zeiger auf el_ in device_data_vec
  CUDA_CALL(cudaMemcpy(&(device_data_vec->el_), &device_data, sizeof(ValueType*), cudaMemcpyHostToDevice));


  // Allokiere gpu_basic_vector für bucket_starts_
  CUDA_CALL(cudaMalloc(&device_bucket_starts_vec, sizeof(gpu_vector<KeyType>)));
  CUDA_CALL(cudaMemcpy(device_bucket_starts_vec, host_bucket_starts_vec, sizeof(gpu_vector<KeyType>), cudaMemcpyHostToDevice));
  device_bytes += sizeof(gpu_vector<gpu_base_t<KeyType>>);

  // Allokiere und kopiere Daten in bucket_starts_ (el_)
  CUDA_CALL(cudaMalloc(&device_bucket_starts, host_vecvec->size() * sizeof(gpu_base_t<KeyType>)));
  CUDA_CALL(cudaMemcpy(device_bucket_starts, host_bucket_starts_vec->el_, host_vecvec->size() * sizeof(gpu_base_t<KeyType>), cudaMemcpyHostToDevice));
  device_bytes += host_vecvec->size() * sizeof(gpu_base_t<KeyType>);

  // Setze den Zeiger auf el_ in device_bucket_starts_vec
  CUDA_CALL(cudaMemcpy(&(device_bucket_starts_vec->el_), &device_bucket_starts, sizeof(gpu_base_t<KeyType>*), cudaMemcpyHostToDevice));

  // Setze die Zeiger in der gpu_vecvec-Struktur auf die allokierten gpu_basic_vector-Strukturen
  CUDA_CALL(cudaMemcpy(&(device_vecvec->data_), &device_data_vec, sizeof(gpu_vector<ValueType>*), cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(&(device_vecvec->bucket_starts_), &device_bucket_starts_vec, sizeof(gpu_vector<gpu_base_t<KeyType>>*), cudaMemcpyHostToDevice));

  return device_vecvec;

fail:
  if (device_data) cudaFree(device_data);
  if (device_bucket_starts) cudaFree(device_bucket_starts);
  if (device_data_vec) cudaFree(device_data_vec);
  if (device_bucket_starts_vec) cudaFree(device_bucket_starts_vec);
  if (device_vecvec) cudaFree(device_vecvec);
  return nullptr;
}

template <typename KeyType, typename ValueType>
void copy_gpu_vector_map_to_device(gpu_vector_map<KeyType, ValueType> const* host_map,
                               gpu_vector_map<KeyType, ValueType>** device_map_ptr,size_t& device_bytes, cudaError_t& code) {
  //TODO:muss Keys auch noch rüber? aber wo sind die??
  auto host_elements = host_map->el_;
  auto num_elements = host_map->size();

  ValueType* device_elements = nullptr;

  gpu_vector_map<KeyType, ValueType> tmp_map;

  CUDA_CALL(cudaMalloc(&device_elements, num_elements * sizeof(ValueType)));
  CUDA_CALL(cudaMemcpy(device_elements, host_elements, num_elements * sizeof(ValueType), cudaMemcpyHostToDevice));
  device_bytes +=num_elements * sizeof(ValueType);

  tmp_map.el_ = device_elements;
  tmp_map.used_size_ = host_map->used_size_;
  tmp_map.allocated_size_ = host_map->allocated_size_;
  tmp_map.self_allocated_ = true;

  CUDA_CALL(cudaMalloc(device_map_ptr, sizeof(gpu_vector_map<KeyType, ValueType>)));
  CUDA_CALL(cudaMemcpy(*device_map_ptr, &tmp_map, sizeof(gpu_vector_map<KeyType, ValueType>), cudaMemcpyHostToDevice));
  device_bytes += sizeof(gpu_vector_map<KeyType, ValueType>);

  return;

  fail:
    if (device_elements) cudaFree(device_elements);
    if (*device_map_ptr) cudaFree(*device_map_ptr);
}


struct gpu_timetable* create_gpu_timetable(gpu_delta const* route_stop_times,
                                           std::uint32_t  n_route_stop_times,
                                           gpu_vecvec<gpu_route_idx_t,gpu_value_type> const* route_location_seq,
                                           gpu_vecvec<gpu_location_idx_t , gpu_route_idx_t> const* location_routes,
                                           std::uint32_t const* n_locations,
                                           std::uint32_t const* n_routes,
                                           gpu_vector_map<gpu_route_idx_t,gpu_interval<std::uint32_t>> const* route_stop_time_ranges,
                                           gpu_vector_map<gpu_route_idx_t,gpu_interval<gpu_transport_idx_t >> const* route_transport_ranges,
                                           gpu_vector_map<gpu_bitfield_idx_t, gpu_bitfield> const* bitfields,
                                           gpu_vector_map<gpu_bitfield_idx_t,std::uint64_t*> const* bitfields_data,
                                           gpu_vector_map<gpu_transport_idx_t,gpu_bitfield_idx_t> const* transport_traffic_days,
                                           gpu_interval<gpu_sys_days> const* date_range,
                                           gpu_locations const* locations,
                                           gpu_vector_map<gpu_route_idx_t, gpu_clasz> const* route_clasz) {
  size_t device_bytes = 0U;

  cudaError_t code;
  gpu_timetable* gtt =
      static_cast<gpu_timetable*>(malloc(sizeof(gpu_timetable)));
  if (gtt == nullptr) {
    printf("nigiri gpu raptor: malloc for gpu_timetable failed\n");
    return nullptr;
  }
  // route_stop_times_
  gtt->route_stop_times_ = nullptr;
  CUDA_COPY_TO_DEVICE(gpu_delta, gtt->route_stop_times_, route_stop_times,
                      n_route_stop_times);
  //route_location_seq
  gtt->route_location_seq_ = copy_gpu_vecvec_to_device(route_location_seq,device_bytes,code);
  //location_routes_
  gtt->location_routes_ = copy_gpu_vecvec_to_device(location_routes, device_bytes, code);
  //n_locations_
  gtt->n_locations_ = nullptr;
  CUDA_COPY_TO_DEVICE(uint32_t , gtt->n_locations_, n_locations,1);
  //n_routes_
  gtt->n_routes_ = nullptr;
  CUDA_COPY_TO_DEVICE(uint32_t , gtt->n_routes_, n_routes,1);
  //route_stop_time_ranges_
  gtt->route_stop_time_ranges_ = nullptr;
  copy_gpu_vector_map_to_device(route_stop_time_ranges,&gtt->route_stop_time_ranges_,device_bytes,code);
  //route_transport_ranges_
  gtt->route_transport_ranges_ = nullptr;
  copy_gpu_vector_map_to_device(route_transport_ranges,&gtt->route_transport_ranges_,device_bytes,code);
  //bitfields_
  gtt->bitfields_ = nullptr;
  copy_gpu_vector_map_to_device(bitfields,&gtt->bitfields_,device_bytes,code);
  //bitfields_data_
  gtt->bitfields_data_ = nullptr;
  copy_gpu_vector_map_to_device(bitfields_data,&gtt->bitfields_data_,device_bytes,code);
  //transport_traffic_days_
  gtt->transport_traffic_days_ = nullptr;
  copy_gpu_vector_map_to_device(transport_traffic_days,&gtt->transport_traffic_days_,device_bytes,code);
  //date_range_
  gtt->date_range_ = nullptr;
  using gpu_date_range = gpu_interval<gpu_sys_days>;
  CUDA_COPY_TO_DEVICE(gpu_date_range , gtt->date_range_, date_range,1);
  //locations_
  gtt->locations_ = nullptr;
  CUDA_COPY_TO_DEVICE(gpu_locations , gtt->locations_, locations,1);
  copy_gpu_vector_map_to_device(locations->transfer_time_,&gtt->locations_->transfer_time_,device_bytes,code);
  gtt->locations_->gpu_footpaths_in_ = copy_gpu_vecvec_to_device(locations->gpu_footpaths_in_,device_bytes,code);
  gtt->locations_->gpu_footpaths_out_ = copy_gpu_vecvec_to_device(locations->gpu_footpaths_out_,device_bytes,code);

  //route_clasz_
  gtt->route_clasz_ = nullptr;
  copy_gpu_vector_map_to_device(route_clasz,&gtt->route_clasz_,device_bytes,code);

  cudaDeviceSynchronize();
  return gtt;


fail:
  destroy_gpu_timetable(gtt);
  return gtt;
}
void destroy_gpu_timetable(gpu_timetable* gtt) {
  if (!gtt) return;

  if (gtt->route_stop_times_) cudaFree(gtt->route_stop_times_);

  if (gtt->route_location_seq_) {
    if (gtt->route_location_seq_->data_.el_) cudaFree(gtt->route_location_seq_->data_.el_);
    if (gtt->route_location_seq_->bucket_starts_.el_) cudaFree(gtt->route_location_seq_->bucket_starts_.el_);
    if (gtt->route_location_seq_->data_.data()) cudaFree(gtt->route_location_seq_->data_.data());
    if (gtt->route_location_seq_->bucket_starts_.data()) cudaFree(gtt->route_location_seq_->bucket_starts_.data());
    cudaFree(gtt->route_location_seq_);
  }

  if (gtt->location_routes_) {
    if (gtt->location_routes_->data_.el_) cudaFree(gtt->location_routes_->data_.el_);
    if (gtt->location_routes_->bucket_starts_.el_) cudaFree(gtt->location_routes_->bucket_starts_.el_);
    if (gtt->location_routes_->data_.data()) cudaFree(gtt->location_routes_->data_.data());
    if (gtt->location_routes_->bucket_starts_.data()) cudaFree(gtt->location_routes_->bucket_starts_.data());
    cudaFree(gtt->location_routes_);
  }

  if (gtt->n_locations_) cudaFree(gtt->n_locations_);
  if (gtt->n_routes_) cudaFree(gtt->n_routes_);

  if (gtt->route_stop_time_ranges_) {
    if (gtt->route_stop_time_ranges_->data()) cudaFree(gtt->route_stop_time_ranges_->data());
    cudaFree(gtt->route_stop_time_ranges_);
  }

  if (gtt->route_transport_ranges_) {
    if (gtt->route_transport_ranges_->data()) cudaFree(gtt->route_transport_ranges_->data());
    cudaFree(gtt->route_transport_ranges_);
  }

  if (gtt->bitfields_) {
    if (gtt->bitfields_->data()) cudaFree(gtt->bitfields_->data());
    cudaFree(gtt->bitfields_);
  }

  if (gtt->bitfields_data_) {
    if (gtt->bitfields_data_->data()) cudaFree(gtt->bitfields_data_->data());
    cudaFree(gtt->bitfields_data_);
  }

  if (gtt->transport_traffic_days_) {
    if (gtt->transport_traffic_days_->data()) cudaFree(gtt->transport_traffic_days_->data());
    cudaFree(gtt->transport_traffic_days_);
  }

  if (gtt->date_range_) cudaFree(gtt->date_range_);

  if (gtt->locations_) {
    if (gtt->locations_->transfer_time_) {
      if (gtt->locations_->transfer_time_->data()) cudaFree(gtt->locations_->transfer_time_->data());
      cudaFree(gtt->locations_->transfer_time_);
    }
    if (gtt->locations_->gpu_footpaths_in_) {
      if (gtt->locations_->gpu_footpaths_in_->data_.el_) cudaFree(gtt->locations_->gpu_footpaths_in_->data_.el_);
      if (gtt->locations_->gpu_footpaths_in_->bucket_starts_.el_) cudaFree(gtt->locations_->gpu_footpaths_in_->bucket_starts_.el_);
      if (gtt->locations_->gpu_footpaths_in_->data_.data()) cudaFree(gtt->locations_->gpu_footpaths_in_->data_.data());
      if (gtt->locations_->gpu_footpaths_in_->bucket_starts_.data()) cudaFree(gtt->locations_->gpu_footpaths_in_->bucket_starts_.data());
      cudaFree(gtt->locations_->gpu_footpaths_in_);
    }
    if (gtt->locations_->gpu_footpaths_out_) {
      if (gtt->locations_->gpu_footpaths_out_->data_.el_) cudaFree(gtt->locations_->gpu_footpaths_out_->data_.el_);
      if (gtt->locations_->gpu_footpaths_out_->bucket_starts_.el_) cudaFree(gtt->locations_->gpu_footpaths_out_->bucket_starts_.el_);
      if (gtt->locations_->gpu_footpaths_out_->data_.data()) cudaFree(gtt->locations_->gpu_footpaths_out_->data_.data());
      if (gtt->locations_->gpu_footpaths_out_->bucket_starts_.data()) cudaFree(gtt->locations_->gpu_footpaths_out_->bucket_starts_.data());
      cudaFree(gtt->locations_->gpu_footpaths_out_);
    }
    cudaFree(gtt->locations_);
  }

  if (gtt->route_clasz_) {
    if (gtt->route_clasz_->data()) cudaFree(gtt->route_clasz_->data());
    cudaFree(gtt->route_clasz_);
  }
  free(gtt);
  gtt = nullptr;
  cudaDeviceSynchronize();
  auto const last_error = cudaGetLastError();
  if (last_error != cudaSuccess) {
    printf("CUDA error: %s at " STR(last_error) " %s:%d\n",
           cudaGetErrorString(last_error), __FILE__, __LINE__);
  }
}