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

  using VecVec = gpu_vecvec<KeyType, ValueType>;

  // Step 1: Allocate memory on the device for the `gpu_vecvec` structure
  VecVec* device_vecvec = nullptr;
  gpu_base_t<KeyType>* device_bucket_starts = nullptr;
  ValueType* device_data = nullptr;
  VecVec host_vecvec_copy = *host_vecvec; // Create a host copy to modify the pointers
  CUDA_CALL(cudaMalloc(&device_vecvec, sizeof(VecVec)));

  // Step 2: Allocate memory on the device for `bucket_starts`
  CUDA_CALL(cudaMalloc(&device_bucket_starts, host_vecvec->bucket_starts_.size() * sizeof(gpu_base_t<KeyType>)));

  // Copy `bucket_starts` from host to device
  CUDA_CALL(cudaMemcpy(device_bucket_starts, host_vecvec->bucket_starts_.data(),
                    host_vecvec->bucket_starts_.size() * sizeof(gpu_base_t<KeyType>), cudaMemcpyHostToDevice));

  // Step 3: Allocate memory on the device for `data`
  CUDA_CALL(cudaMalloc(&device_data, host_vecvec->data_.size() * sizeof(ValueType)));

  // Copy `data` from host to device
  CUDA_CALL(cudaMemcpy(device_data, host_vecvec->data_.data(),
                    host_vecvec->data_.size() * sizeof(ValueType), cudaMemcpyHostToDevice));

  // Step 4: Update the `gpu_vecvec` on the device to point to device memory
  host_vecvec_copy.bucket_starts_.el_ = device_bucket_starts;
  host_vecvec_copy.data_.el_ = device_data;

  // Copy the modified `gpu_vecvec` from host to device
  CUDA_CALL(cudaMemcpy(device_vecvec, &host_vecvec_copy, sizeof(VecVec), cudaMemcpyHostToDevice));

  device_bytes += sizeof(VecVec);
  device_bytes += host_vecvec->bucket_starts_.size() * sizeof(gpu_base_t<KeyType>);
  device_bytes += host_vecvec->data_.size() * sizeof(ValueType);

  return device_vecvec;
fail:
  if (device_bucket_starts) cudaFree(device_bucket_starts);
  if (device_data) cudaFree(device_data);
  if (device_vecvec) cudaFree(device_vecvec);
  return nullptr;
}

template <typename KeyType, typename ValueType>
gpu_vector_map<KeyType, ValueType>* copy_gpu_vector_map_to_device(
    const gpu_vector_map<KeyType, ValueType>* host_vector_map,
    size_t& device_bytes, cudaError_t& code) {

  using MapType = gpu_vector_map<KeyType, ValueType>;

  // Schritt 1: Allokieren des Gerätespeichers für die `gpu_vector_map`-Struktur
  MapType* device_vector_map = nullptr;
  ValueType* device_data = nullptr;
  MapType host_map_copy = *host_vector_map;
  CUDA_CALL(cudaMalloc(&device_vector_map, sizeof(MapType)));

  // Schritt 2: Allokieren des Gerätespeichers für `el_` (die Daten)
  CUDA_CALL(cudaMalloc(&device_data, host_vector_map->size() * sizeof(ValueType)));

  // Schritt 3: Kopieren der Daten vom Host zum Gerät
  CUDA_CALL(cudaMemcpy(device_data, host_vector_map->data(),
                    host_vector_map->size() * sizeof(ValueType),
                    cudaMemcpyHostToDevice));

  // Schritt 4: Aktualisieren der `el_`-Zeiger im `gpu_vector_map` auf dem Gerät
  host_map_copy.el_ = device_data;

  // Kopieren der aktualisierten `gpu_vector_map`-Struktur zurück auf das Gerät
  CUDA_CALL(cudaMemcpy(device_vector_map, &host_map_copy, sizeof(MapType),
                    cudaMemcpyHostToDevice));

  device_bytes += sizeof(MapType);
  device_bytes += host_vector_map->size() * sizeof(ValueType);

  return device_vector_map;
fail:
    if (device_data) cudaFree(device_data);
    if (device_vector_map) cudaFree(device_vector_map);
    return nullptr;
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
  gtt->route_stop_time_ranges_ = copy_gpu_vector_map_to_device(route_stop_time_ranges,device_bytes,code);
  //route_transport_ranges_
  gtt->route_transport_ranges_ = copy_gpu_vector_map_to_device(route_transport_ranges,device_bytes,code);
  //bitfields_
  gtt->bitfields_ = copy_gpu_vector_map_to_device(bitfields,device_bytes,code);
  //transport_traffic_days_
  gtt->transport_traffic_days_ = copy_gpu_vector_map_to_device(transport_traffic_days,device_bytes,code);
  //date_range_
  gtt->date_range_ = nullptr;
  using gpu_date_range = gpu_interval<gpu_sys_days>;
  CUDA_COPY_TO_DEVICE(gpu_date_range , gtt->date_range_, date_range,1);
  //locations_
  gtt->locations_ = nullptr;
  locations_copy.transfer_time_ = copy_gpu_vector_map_to_device(locations->transfer_time_,device_bytes,code);
  locations_copy.gpu_footpaths_in_ = copy_gpu_vecvec_to_device(locations->gpu_footpaths_in_,device_bytes,code);
  locations_copy.gpu_footpaths_out_ = copy_gpu_vecvec_to_device(locations->gpu_footpaths_out_,device_bytes,code);
  CUDA_COPY_TO_DEVICE(gpu_locations, gtt->locations_, &locations_copy,1);

  //route_clasz_
  gtt->route_clasz_ = copy_gpu_vector_map_to_device(route_clasz,device_bytes,code);

  cudaDeviceSynchronize();
  return gtt;


fail:
  destroy_gpu_timetable(gtt);
  return nullptr;
}
void destroy_gpu_timetable(gpu_timetable* gtt) {
  if (!gtt) return;

  if (gtt->route_stop_times_) cudaFree(gtt->route_stop_times_);

  if (gtt->route_location_seq_) {
    if (gtt->route_location_seq_->data_.el_) cudaFree(gtt->route_location_seq_->data_.el_);
    if (gtt->route_location_seq_->bucket_starts_.el_) cudaFree(gtt->route_location_seq_->bucket_starts_.el_);
    cudaFree(gtt->route_location_seq_);
  }

  if (gtt->location_routes_) {
    if (gtt->location_routes_->data_.el_) cudaFree(gtt->location_routes_->data_.el_);
    if (gtt->location_routes_->bucket_starts_.el_) cudaFree(gtt->location_routes_->bucket_starts_.el_);
    cudaFree(gtt->location_routes_);
  }

  if (gtt->n_locations_) cudaFree(gtt->n_locations_);
  if (gtt->n_routes_) cudaFree(gtt->n_routes_);

  if (gtt->route_stop_time_ranges_) {
    if (gtt->route_stop_time_ranges_->el_) cudaFree(gtt->route_stop_time_ranges_->el_);
    cudaFree(gtt->route_stop_time_ranges_);
  }

  if (gtt->route_transport_ranges_) {
    if (gtt->route_transport_ranges_->el_) cudaFree(gtt->route_transport_ranges_->el_);
    cudaFree(gtt->route_transport_ranges_);
  }

  if (gtt->bitfields_) {
    if (gtt->bitfields_->el_) cudaFree(gtt->bitfields_->el_);
    cudaFree(gtt->bitfields_);
  }

  if (gtt->transport_traffic_days_) {
    if (gtt->transport_traffic_days_->el_) cudaFree(gtt->transport_traffic_days_->el_);
    cudaFree(gtt->transport_traffic_days_);
  }

  if (gtt->date_range_) cudaFree(gtt->date_range_);

  if (gtt->locations_) {
    if (gtt->locations_->transfer_time_) {
      if (gtt->locations_->transfer_time_->el_) cudaFree(gtt->locations_->transfer_time_->el_);
      cudaFree(gtt->locations_->transfer_time_);
    }
    if (gtt->locations_->gpu_footpaths_in_) {
      if (gtt->locations_->gpu_footpaths_in_->data_.el_) cudaFree(gtt->locations_->gpu_footpaths_in_->data_.el_);
      if (gtt->locations_->gpu_footpaths_in_->bucket_starts_.el_) cudaFree(gtt->locations_->gpu_footpaths_in_->bucket_starts_.el_);
      cudaFree(gtt->locations_->gpu_footpaths_in_);
    }
    if (gtt->locations_->gpu_footpaths_out_) {
      if (gtt->locations_->gpu_footpaths_out_->data_.el_) cudaFree(gtt->locations_->gpu_footpaths_out_->data_.el_);
      if (gtt->locations_->gpu_footpaths_out_->bucket_starts_.el_) cudaFree(gtt->locations_->gpu_footpaths_out_->bucket_starts_.el_);
      cudaFree(gtt->locations_->gpu_footpaths_out_);
    }
    cudaFree(gtt->locations_);
  }

  if (gtt->route_clasz_) {
    if (gtt->route_clasz_->el_) cudaFree(gtt->route_clasz_->el_);
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