#pragma once

#include <cuda_runtime.h>
#include "nigiri/routing/gpu_timetable.h"
#include <iostream>
#include <cstdio>

#define XSTR(s) STR(s)
#define STR(s) #s

#define CUDA_CALL(call)                                   \
    if ((code = call) != cudaSuccess) {                     \
      printf("CUDA error: %s at " XSTR(call) " %s:%d\n",     \
             cudaGetErrorString(code), __FILE__, __LINE__); \
      goto fail;                                            \
    }

#define CUDA_COPY_TO_DEVICE(type, target, source, size)                        \
    CUDA_CALL(cudaMalloc(&target, size * sizeof(type)))                          \
    CUDA_CALL(                                                                   \
        cudaMemcpy(target, source, size * sizeof(type), cudaMemcpyHostToDevice)) \
    device_bytes += size * sizeof(type);

template <typename KeyType, typename ValueType>
void copy_gpu_vecvec_to_device(gpu_vecvec<KeyType, ValueType> const* h_vecvec,gpu_vecvec<KeyType, ValueType>*& d_vecvec, size_t& device_bytes, cudaError_t& code) {
  // Device-Speicher für `gpu_vecvec` zuweisen
  d_vecvec = nullptr;
  gpu_vector<ValueType>* d_data = nullptr;
  gpu_vector<gpu_base_t<KeyType>>* d_bucket_starts = nullptr;
  CUDA_CALL(cudaMalloc(&d_vecvec, sizeof(gpu_vecvec<KeyType, ValueType>)));

  // Device-Speicher für `bucket_starts_` und `data_` zuweisen
  CUDA_CALL(cudaMalloc(&d_bucket_starts, h_vecvec->bucket_starts_.size() * sizeof(gpu_base_t<KeyType>)));
  CUDA_CALL(cudaMemcpy(d_bucket_starts, h_vecvec->bucket_starts_.data(),
                        h_vecvec->bucket_starts_.size() * sizeof(gpu_base_t<KeyType>), cudaMemcpyHostToDevice));

  CUDA_CALL(cudaMalloc(&d_data, h_vecvec->data_.size() * sizeof(ValueType)));
  CUDA_CALL(cudaMemcpy(d_data, h_vecvec->data_.data(),
                        h_vecvec->data_.size() * sizeof(ValueType), cudaMemcpyHostToDevice));

  // Die Kopie der Vektorzeiger auf die GPU-Struktur setzen
  CUDA_CALL(cudaMemcpy(&(d_vecvec->bucket_starts_), &d_bucket_starts, sizeof(gpu_vector<gpu_base_t<KeyType>>), cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(&(d_vecvec->data_), &d_data, sizeof(gpu_vector<ValueType>), cudaMemcpyHostToDevice));

  CUDA_CALL(cudaMemcpy(&(d_vecvec->bucket_starts_.used_size_), &h_vecvec->bucket_starts_.used_size_, sizeof(h_vecvec->bucket_starts_.used_size_), cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(&(d_vecvec->bucket_starts_.allocated_size_), &h_vecvec->bucket_starts_.allocated_size_, sizeof(h_vecvec->bucket_starts_.allocated_size_), cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(&(d_vecvec->bucket_starts_.self_allocated_), &h_vecvec->bucket_starts_.self_allocated_, sizeof(h_vecvec->bucket_starts_.self_allocated_), cudaMemcpyHostToDevice));

  CUDA_CALL(cudaMemcpy(&(d_vecvec->data_.used_size_), &h_vecvec->data_.used_size_, sizeof(h_vecvec->data_.used_size_), cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(&(d_vecvec->data_.allocated_size_), &h_vecvec->data_.allocated_size_, sizeof(h_vecvec->data_.allocated_size_), cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(&(d_vecvec->data_.self_allocated_), &h_vecvec->data_.self_allocated_, sizeof(h_vecvec->data_.self_allocated_), cudaMemcpyHostToDevice));

  device_bytes += sizeof(gpu_vecvec<KeyType, ValueType>);
  return;
fail:
  std::cerr << "ERROR VECVEC" << std::endl;
  if (d_bucket_starts) cudaFree(d_bucket_starts);
  if (d_data) cudaFree(d_data);
  if (d_vecvec) cudaFree(d_vecvec);
  d_vecvec = nullptr;
  return;
}

template <typename KeyType, typename ValueType>
void copy_gpu_vector_map_to_device(
    const gpu_vector_map<KeyType, ValueType>* h_map,gpu_vector_map<KeyType, ValueType>*& d_map,
    size_t& device_bytes, cudaError_t& code) {

  // Device-Speicher für gpu_vector_map zuweisen
  d_map = nullptr;
  gpu_vector<ValueType>* d_data = nullptr;
  CUDA_CALL(cudaMalloc(&d_map, sizeof(gpu_vector_map<KeyType, ValueType>)));
  // Device-Speicher für die Daten innerhalb von gpu_vector_map zuweisen
  CUDA_CALL(cudaMalloc(&d_data, h_map->size() * sizeof(ValueType)));
  CUDA_CALL(cudaMemcpy(d_data, h_map->data(), h_map->size() * sizeof(ValueType), cudaMemcpyHostToDevice));
  // Die Kopie des Datenzeigers auf die GPU-Struktur setzen
  CUDA_CALL(cudaMemcpy(&(d_map->el_), &d_data, sizeof(gpu_vector<ValueType>), cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(&(d_map->used_size_), &h_map->used_size_, sizeof(h_map->used_size_), cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(&(d_map->allocated_size_), &h_map->allocated_size_, sizeof(h_map->allocated_size_), cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(&(d_map->self_allocated_), &h_map->self_allocated_, sizeof(h_map->self_allocated_), cudaMemcpyHostToDevice))

  device_bytes += sizeof(gpu_vector_map<KeyType, ValueType>);
  return;
fail:
  std::cerr << "ERROR VECMAP" << std::endl;
    if (d_data) cudaFree(d_data);
    if (d_map) cudaFree(d_map);
    d_map = nullptr;
    return;
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
  auto locations_copy = *locations;
  // route_stop_times_
  gtt->route_stop_times_ = nullptr;
  CUDA_COPY_TO_DEVICE(gpu_delta, gtt->route_stop_times_, route_stop_times,
                      n_route_stop_times);
  //route_location_seq
   copy_gpu_vecvec_to_device(route_location_seq,gtt->route_location_seq_,device_bytes,code);


  //location_routes_
  copy_gpu_vecvec_to_device(location_routes,gtt->location_routes_, device_bytes, code);
  //n_locations_
  gtt->n_locations_ = nullptr;
  CUDA_COPY_TO_DEVICE(uint32_t , gtt->n_locations_, n_locations,1);
  //n_routes_
  gtt->n_routes_ = nullptr;
  CUDA_COPY_TO_DEVICE(uint32_t , gtt->n_routes_, n_routes,1);
  //route_stop_time_ranges_
  copy_gpu_vector_map_to_device(route_stop_time_ranges,gtt->route_stop_time_ranges_,device_bytes,code);
  //route_transport_ranges_
  copy_gpu_vector_map_to_device(route_transport_ranges,gtt->route_transport_ranges_,device_bytes,code);
  //bitfields_
  copy_gpu_vector_map_to_device(bitfields,gtt->bitfields_,device_bytes,code);
  //transport_traffic_days_
  copy_gpu_vector_map_to_device(transport_traffic_days,gtt->transport_traffic_days_,device_bytes,code);
  //date_range_
  gtt->date_range_ = nullptr;
  using gpu_date_range = gpu_interval<gpu_sys_days>;
  CUDA_COPY_TO_DEVICE(gpu_date_range , gtt->date_range_, date_range,1);
  //locations_
  gtt->locations_ = nullptr;
  //TODO:maybe fix locations?
  std::cerr << "Locations" << std::endl;
  copy_gpu_vector_map_to_device(locations->transfer_time_,locations_copy.transfer_time_,device_bytes,code);
  copy_gpu_vecvec_to_device(locations->gpu_footpaths_in_,locations_copy.gpu_footpaths_in_,device_bytes,code);
  copy_gpu_vecvec_to_device(locations->gpu_footpaths_out_,locations_copy.gpu_footpaths_out_,device_bytes,code);
  CUDA_COPY_TO_DEVICE(gpu_locations, gtt->locations_, &locations_copy,1);
  //route_clasz_
  copy_gpu_vector_map_to_device(route_clasz,gtt->route_clasz_,device_bytes,code);

  cudaDeviceSynchronize();
  if(!gtt->route_stop_times_||!gtt->route_location_seq_ || !gtt->location_routes_||!gtt->n_locations_||!gtt->n_routes_||!gtt->route_stop_time_ranges_||!gtt->route_transport_ranges_||!gtt->bitfields_||
      !gtt->transport_traffic_days_||! gtt->date_range_||!gtt->locations_||!gtt->route_clasz_){
    std::cerr << "something went wrong, one attribute ist nullptr" << std::endl;
    goto fail;
  }
  return gtt;


fail:
  destroy_gpu_timetable(gtt);
  return nullptr;
}
void destroy_gpu_timetable(gpu_timetable* gtt) {
  if (!gtt) return;

  if (gtt->route_stop_times_) cudaFree(gtt->route_stop_times_);

  if (gtt->route_location_seq_) {
    if (&(gtt->route_location_seq_->data_.el_)) cudaFree(&(gtt->route_location_seq_->data_.el_));
    if (&(gtt->route_location_seq_->bucket_starts_.el_)) cudaFree(&(gtt->route_location_seq_->bucket_starts_.el_));
    cudaFree(gtt->route_location_seq_);
  }

  if (gtt->location_routes_) {
    if (&(gtt->location_routes_->data_.el_)) cudaFree(&(gtt->location_routes_->data_.el_));
    if (&(gtt->location_routes_->bucket_starts_.el_)) cudaFree(&(gtt->location_routes_->bucket_starts_.el_));
    cudaFree(gtt->location_routes_);
  }

  if (gtt->n_locations_) cudaFree(gtt->n_locations_);
  if (gtt->n_routes_) cudaFree(gtt->n_routes_);

  if (gtt->route_stop_time_ranges_) {
    if (&(gtt->route_stop_time_ranges_->el_)) cudaFree(&(gtt->route_stop_time_ranges_->el_));
    cudaFree(gtt->route_stop_time_ranges_);
  }

  if (gtt->route_transport_ranges_) {
    if (&(gtt->route_transport_ranges_->el_)) cudaFree(&(gtt->route_transport_ranges_->el_));
    cudaFree(gtt->route_transport_ranges_);
  }

  if (gtt->bitfields_) {
    if (&(gtt->bitfields_->el_)) cudaFree(&(gtt->bitfields_->el_));
    cudaFree(gtt->bitfields_);
  }

  if (gtt->transport_traffic_days_) {
    if (&(gtt->transport_traffic_days_->el_)) cudaFree(&(gtt->transport_traffic_days_->el_));
    cudaFree(gtt->transport_traffic_days_);
  }

  if (gtt->date_range_) cudaFree(gtt->date_range_);

  std::cerr << "Locations del" << std::endl;
  if (gtt->locations_) {
    if (&(gtt->locations_->transfer_time_)) {
      if (&(gtt->locations_->transfer_time_->el_)) cudaFree(&(gtt->locations_->transfer_time_->el_));
      cudaFree(gtt->locations_->transfer_time_);
    }
    if (&(gtt->locations_->gpu_footpaths_in_)) {
      if (&(gtt->locations_->gpu_footpaths_in_->data_.el_)) cudaFree(&(gtt->locations_->gpu_footpaths_in_->data_.el_));
      if (&(gtt->locations_->gpu_footpaths_in_->bucket_starts_.el_)) cudaFree(&(gtt->locations_->gpu_footpaths_in_->bucket_starts_.el_));
      cudaFree(gtt->locations_->gpu_footpaths_in_);
    }
    if (&(gtt->locations_->gpu_footpaths_out_)) {
      if (&(gtt->locations_->gpu_footpaths_out_->data_.el_)) cudaFree(&(gtt->locations_->gpu_footpaths_out_->data_.el_));
      if (&(gtt->locations_->gpu_footpaths_out_->bucket_starts_.el_)) cudaFree(&(gtt->locations_->gpu_footpaths_out_->bucket_starts_.el_));
      cudaFree(gtt->locations_->gpu_footpaths_out_);
    }
    cudaFree(gtt->locations_);
  }

  std::cerr << "Locations del end" << std::endl;
  if (gtt->route_clasz_) {
    if (&(gtt->route_clasz_->el_)) cudaFree(&(gtt->route_clasz_->el_));
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