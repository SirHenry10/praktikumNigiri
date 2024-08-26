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
void copy_gpu_vecvec_to_device(const gpu_vecvec<KeyType, ValueType>* h_vecvec,gpu_vecvec<KeyType, ValueType>*& d_vecvec, size_t& device_bytes, cudaError_t& code) {
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
void copy_gpu_vector_map_to_device(const gpu_vector_map<KeyType, ValueType>* h_map,
                                   gpu_vector_map<KeyType, ValueType>*& d_map,
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
void copy_gpu_locations_to_device(const gpu_locations* h_locations, gpu_locations*& d_locations, size_t& device_bytes, cudaError_t& code) {
  // Allocate memory for the `gpu_locations` structure on the GPU
  //TODO: unsicher ob die device_bytes stimmen
  d_locations = nullptr;
  gpu_vector_map<gpu_location_idx_t, gpu_u8_minutes>* d_transfer_time = nullptr;
  gpu_vecvec<gpu_location_idx_t, nigiri::gpu_footpath>* d_gpu_footpaths_in = nullptr;
  gpu_vecvec<gpu_location_idx_t, nigiri::gpu_footpath>* d_gpu_footpaths_out = nullptr;

  CUDA_CALL(cudaMalloc(&d_locations, sizeof(gpu_locations)));

  // Copy each nested structure individually
  // 1. Copy `transfer_time_`
  copy_gpu_vector_map_to_device(h_locations->transfer_time_, d_transfer_time, device_bytes, code);
  CUDA_CALL(cudaMemcpy(&(d_locations->transfer_time_), &d_transfer_time, sizeof(gpu_vector_map<gpu_location_idx_t, gpu_u8_minutes>*), cudaMemcpyHostToDevice));

  // 2. Copy `gpu_footpaths_in_`
  copy_gpu_vecvec_to_device(h_locations->gpu_footpaths_in_, d_gpu_footpaths_in, device_bytes, code);
  CUDA_CALL(cudaMemcpy(&(d_locations->gpu_footpaths_in_), &d_gpu_footpaths_in, sizeof(gpu_vecvec<gpu_location_idx_t, nigiri::gpu_footpath>*), cudaMemcpyHostToDevice));

  // 3. Copy `gpu_footpaths_out_`
  copy_gpu_vecvec_to_device(h_locations->gpu_footpaths_out_, d_gpu_footpaths_out, device_bytes, code);
  CUDA_CALL(cudaMemcpy(&(d_locations->gpu_footpaths_out_), &d_gpu_footpaths_out, sizeof(gpu_vecvec<gpu_location_idx_t, nigiri::gpu_footpath>*), cudaMemcpyHostToDevice));

  device_bytes += sizeof(gpu_locations);

  return;

fail:
  std::cerr << "ERROR LOCATIONS" << std::endl;
  // Free allocated GPU memory if something goes wrong
  if (d_transfer_time) cudaFree(d_transfer_time);
  if (d_gpu_footpaths_in) cudaFree(d_gpu_footpaths_in);
  if (d_gpu_footpaths_out) cudaFree(d_gpu_footpaths_out);
  if (d_locations) cudaFree(d_locations);
  d_locations = nullptr;
  return;
}
template <typename KeyType, typename ValueType>
void free_gpu_vecvec(gpu_vecvec<KeyType, ValueType>* d_vecvec) {
  if (!d_vecvec) return;

  cudaError_t code;
  // Free each nested data structure
  gpu_base_t<KeyType>* d_bucket_starts = nullptr;
  ValueType* d_data = nullptr;

  cudaMemcpy(&d_bucket_starts, &(d_vecvec->bucket_starts_), sizeof(gpu_base_t<KeyType>*), cudaMemcpyDeviceToHost);
  cudaMemcpy(&d_data, &(d_vecvec->data_), sizeof(ValueType*), cudaMemcpyDeviceToHost);

  if (d_bucket_starts) {
    code = cudaFree(d_bucket_starts);
    if (code != cudaSuccess) {
      std::cerr << "Error freeing d_bucket_starts: " << cudaGetErrorString(code) << std::endl;
    }
  }
  if (d_data) {
    code = cudaFree(d_data);
    if (code != cudaSuccess) {
      std::cerr << "Error freeing d_data: " << cudaGetErrorString(code) << std::endl;
    }
  }

  code = cudaFree(d_vecvec);
  if (code != cudaSuccess) {
    std::cerr << "Error freeing gpu_vecvec: " << cudaGetErrorString(code) << std::endl;
  }
}

template <typename KeyType, typename ValueType>
void free_gpu_vector_map(gpu_vector_map<KeyType, ValueType>* d_map) {
  if (!d_map) return;

  cudaError_t code;
  ValueType* d_data = nullptr;

  cudaMemcpy(&d_data, &(d_map->el_), sizeof(ValueType*), cudaMemcpyDeviceToHost);

  if (d_data) {
    code = cudaFree(d_data);
    if (code != cudaSuccess) {
      std::cerr << "Error freeing d_data: " << cudaGetErrorString(code) << std::endl;
    }
  }

  code = cudaFree(d_map);
  if (code != cudaSuccess) {
    std::cerr << "Error freeing gpu_vector_map: " << cudaGetErrorString(code) << std::endl;
  }
}

void free_gpu_locations(gpu_locations* d_locations) {
  if (!d_locations) return;

  cudaError_t code;

  // Retrieve pointers to nested structures
  gpu_vector_map<gpu_location_idx_t, gpu_u8_minutes>* d_transfer_time = nullptr;
  gpu_vecvec<gpu_location_idx_t, nigiri::gpu_footpath>* d_gpu_footpaths_in = nullptr;
  gpu_vecvec<gpu_location_idx_t, nigiri::gpu_footpath>* d_gpu_footpaths_out = nullptr;

  cudaMemcpy(&d_transfer_time, &(d_locations->transfer_time_), sizeof(gpu_vector_map<gpu_location_idx_t, gpu_u8_minutes>*), cudaMemcpyDeviceToHost);
  cudaMemcpy(&d_gpu_footpaths_in, &(d_locations->gpu_footpaths_in_), sizeof(gpu_vecvec<gpu_location_idx_t, nigiri::gpu_footpath>*), cudaMemcpyDeviceToHost);
  cudaMemcpy(&d_gpu_footpaths_out, &(d_locations->gpu_footpaths_out_), sizeof(gpu_vecvec<gpu_location_idx_t, nigiri::gpu_footpath>*), cudaMemcpyDeviceToHost);

  // Free each nested structure
  if (d_transfer_time) free_gpu_vector_map(d_transfer_time);
  if (d_gpu_footpaths_in) free_gpu_vecvec(d_gpu_footpaths_in);
  if (d_gpu_footpaths_out) free_gpu_vecvec(d_gpu_footpaths_out);

  // Free the `gpu_locations` structure itself
  code = cudaFree(d_locations);
  if (code != cudaSuccess) {
    std::cerr << "Error freeing gpu_locations: " << cudaGetErrorString(code) << std::endl;
  }
}

struct gpu_timetable* create_gpu_timetable(gpu_delta const* route_stop_times,
                                           std::uint32_t  n_route_stop_times,
                                           gpu_vecvec<gpu_route_idx_t,gpu_value_type> const* route_location_seq,
                                           gpu_vecvec<gpu_location_idx_t , gpu_route_idx_t> const* location_routes,
                                           std::uint32_t const n_locations,
                                           std::uint32_t const n_routes,
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
  gtt->n_locations_ = n_locations;
  //n_routes_
  gtt->n_routes_ = n_routes;
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
  copy_gpu_locations_to_device(locations,gtt->locations_,device_bytes,code);
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
void destroy_gpu_timetable(gpu_timetable*& gtt) {
  if (!gtt) return;  // If the pointer is null, there is nothing to free.

  cudaError_t code;

  // 1. Free each nested structure in `gpu_timetable`

  // Free route_stop_times_ (assuming it's a pointer to GPU memory)
  if (gtt->route_stop_times_) {
    code = cudaFree(gtt->route_stop_times_);
    if (code != cudaSuccess) {
      std::cerr << "Error freeing route_stop_times_: " << cudaGetErrorString(code) << std::endl;
    }
    gtt->route_stop_times_ = nullptr;
  }

  // Free route_location_seq_ (complex nested structure)
  if (gtt->route_location_seq_) {
    free_gpu_vecvec(gtt->route_location_seq_);
    gtt->route_location_seq_ = nullptr;
  }

  // Free location_routes_ (complex nested structure)
  if (gtt->location_routes_) {
    free_gpu_vecvec(gtt->location_routes_);
    gtt->location_routes_ = nullptr;
  }
  // Free route_stop_time_ranges_ (complex nested structure)
  if (gtt->route_stop_time_ranges_) {
    free_gpu_vector_map(gtt->route_stop_time_ranges_);
    gtt->route_stop_time_ranges_ = nullptr;
  }

  // Free route_transport_ranges_ (complex nested structure)
  if (gtt->route_transport_ranges_) {
    free_gpu_vector_map(gtt->route_transport_ranges_);
    gtt->route_transport_ranges_ = nullptr;
  }

  // Free bitfields_ (complex nested structure)
  if (gtt->bitfields_) {
    free_gpu_vector_map(gtt->bitfields_);
    gtt->bitfields_ = nullptr;
  }

  // Free transport_traffic_days_ (complex nested structure)
  if (gtt->transport_traffic_days_) {
    free_gpu_vector_map(gtt->transport_traffic_days_);
    gtt->transport_traffic_days_ = nullptr;
  }

  // Free date_range_ (single pointer)
  if (gtt->date_range_) {
    code = cudaFree(gtt->date_range_);
    if (code != cudaSuccess) {
      std::cerr << "Error freeing date_range_: " << cudaGetErrorString(code) << std::endl;
    }
    gtt->date_range_ = nullptr;
  }

  // Free locations_ (complex nested structure)
  if (gtt->locations_) {
    free_gpu_locations(gtt->locations_);
    gtt->locations_ = nullptr;
  }

  // Free route_clasz_ (complex nested structure)
  if (gtt->route_clasz_) {
    free_gpu_vector_map(gtt->route_clasz_);
    gtt->route_clasz_ = nullptr;
  }

  // 2. Finally, free the `gpu_timetable` structure itself
  free(gtt);
  gtt = nullptr;
  cudaDeviceSynchronize();
  auto const last_error = cudaGetLastError();
  if (last_error != cudaSuccess) {
    printf("CUDA error: %s at " STR(last_error) " %s:%d\n",
           cudaGetErrorString(last_error), __FILE__, __LINE__);
  }
}