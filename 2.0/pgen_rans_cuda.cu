// SPDX-License-Identifier: GPL-3.0-or-later

#include "pgen_rans_cuda.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <limits>
#include <string>
#include <vector>

namespace pgen_rans {
namespace {

constexpr uint32_t kRansLowerBound = 1U << 23;
constexpr uint32_t kWarpSize = 32;
constexpr uint32_t kThreadsPerBlock = 256;

enum DeviceDecodeError : uint32_t {
  kDeviceDecodeSuccess = 0,
  kDeviceDecodeInvalidRecord = 1,
  kDeviceDecodeInvalidModel = 2,
  kDeviceDecodeInvalidSelector = 3,
  kDeviceDecodeInvalidState = 4,
  kDeviceDecodeTruncatedPayload = 5,
  kDeviceDecodeNoncanonicalLane = 6,
};

struct DeviceRecordDescriptor {
  uint32_t record_offset;
  uint32_t record_size;
  uint32_t output_variant;
  uint32_t block_output_variant;
  uint32_t block_variant_ct;
  uint32_t anchor_ct;
};

struct DeviceModelRow {
  uint32_t frequencies[4];
  uint32_t cumulative[4];
  uint8_t active_symbol_ct;
  uint8_t deterministic_symbol;
  uint8_t padding[2];
};

struct DeviceWarpModel {
  DeviceModelRow rows[16];
  uint32_t entropy_offset;
  uint16_t context_mask;
  uint8_t mode;
  uint8_t has_entropy;
  uint8_t has_interleaved_payload;
  uint8_t reference1;
  uint8_t reference2;
  uint8_t padding[1];
  uint32_t error;
};

void SetError(const std::string& message, std::string* error) {
  if (error) {
    *error = message;
  }
}

bool CheckCuda(cudaError_t status, const char* operation,
               std::string* error) {
  if (status == cudaSuccess) {
    return true;
  }
  SetError(std::string(operation) + ": " + cudaGetErrorString(status),
           error);
  return false;
}

const char* DeviceErrorMessage(uint32_t error) {
  switch (error) {
    case kDeviceDecodeInvalidRecord:
      return "CUDA decoder rejected a record prefix.";
    case kDeviceDecodeInvalidModel:
      return "CUDA decoder rejected a probability model.";
    case kDeviceDecodeInvalidSelector:
      return "CUDA decoder rejected an anchor selector.";
    case kDeviceDecodeInvalidState:
      return "CUDA decoder rejected an initial rANS state.";
    case kDeviceDecodeTruncatedPayload:
      return "CUDA decoder encountered a truncated rANS lane.";
    case kDeviceDecodeNoncanonicalLane:
      return "CUDA decoder encountered a noncanonical rANS lane.";
    default:
      return "CUDA decoder returned an unknown error.";
  }
}

template <typename T>
bool EnsureDeviceCapacity(T** pointer, size_t* capacity, size_t requested,
                          std::string* error) {
  if (requested <= *capacity) {
    return true;
  }
  T* replacement = nullptr;
  if (!CheckCuda(cudaMalloc(reinterpret_cast<void**>(&replacement),
                            requested * sizeof(T)),
                 "cudaMalloc", error)) {
    return false;
  }
  if (*pointer) {
    cudaFree(*pointer);
  }
  *pointer = replacement;
  *capacity = requested;
  return true;
}

template <typename T>
bool EnsurePinnedCapacity(T** pointer, size_t* capacity, size_t requested,
                          std::string* error) {
  if (requested <= *capacity) {
    return true;
  }
  T* replacement = nullptr;
  if (!CheckCuda(
          cudaHostAlloc(reinterpret_cast<void**>(&replacement),
                        requested * sizeof(T), cudaHostAllocPortable),
          "cudaHostAlloc", error)) {
    return false;
  }
  if (*pointer) {
    cudaFreeHost(*pointer);
  }
  *pointer = replacement;
  *capacity = requested;
  return true;
}

__device__ uint16_t ReadDeviceU16(const uint8_t* input) {
  return static_cast<uint16_t>(
      static_cast<uint16_t>(input[0]) |
      (static_cast<uint16_t>(input[1]) << 8));
}

__device__ uint32_t ReadDeviceU32(const uint8_t* input) {
  return static_cast<uint32_t>(input[0]) |
         (static_cast<uint32_t>(input[1]) << 8) |
         (static_cast<uint32_t>(input[2]) << 16) |
         (static_cast<uint32_t>(input[3]) << 24);
}

__device__ uint64_t SpreadBits32(uint32_t value) {
  uint64_t result = value;
  result = (result | (result << 16)) & 0x0000ffff0000ffffULL;
  result = (result | (result << 8)) & 0x00ff00ff00ff00ffULL;
  result = (result | (result << 4)) & 0x0f0f0f0f0f0f0f0fULL;
  result = (result | (result << 2)) & 0x3333333333333333ULL;
  result = (result | (result << 1)) & 0x5555555555555555ULL;
  return result;
}

__device__ uint32_t ScheduledAnchorOffset(uint32_t anchor_idx,
                                          uint32_t block_variant_ct,
                                          uint32_t anchor_ct) {
  return static_cast<uint32_t>(
      (static_cast<uint64_t>(2 * anchor_idx + 1) * block_variant_ct) /
      (2 * anchor_ct));
}

__device__ void ParseDeviceModel(const uint8_t* record,
                                 uint32_t record_size,
                                 uint32_t scale_bits,
                                 uint32_t anchor_ct,
                                 DeviceWarpModel* model) {
  model->error = kDeviceDecodeSuccess;
  model->context_mask = 0;
  model->has_entropy = 0;
  model->has_interleaved_payload = 0;
  model->reference1 = 0;
  model->reference2 = 0;
  for (uint32_t context = 0; context != 16; ++context) {
    DeviceModelRow& row = model->rows[context];
    row.active_symbol_ct = 0;
    row.deterministic_symbol = 0;
    for (uint32_t symbol = 0; symbol != 4; ++symbol) {
      row.frequencies[symbol] = 0;
      row.cumulative[symbol] = 0;
    }
  }
  if (!record_size) {
    model->error = kDeviceDecodeInvalidRecord;
    return;
  }
  const uint8_t flags = record[0];
  model->mode = flags & 3U;
  model->has_entropy = (flags >> 2) & 1U;
  model->has_interleaved_payload = (flags >> 3) & 1U;
  if ((flags & 0xf0U) || (model->mode > 2) ||
      (model->has_interleaved_payload && !model->has_entropy)) {
    model->error = kDeviceDecodeInvalidRecord;
    return;
  }
  uint32_t offset = 1;
  if (model->mode != 0) {
    if (offset == record_size) {
      model->error = kDeviceDecodeInvalidRecord;
      return;
    }
    model->reference1 = record[offset++];
    if (model->reference1 >= anchor_ct) {
      model->error = kDeviceDecodeInvalidSelector;
      return;
    }
  }
  if (model->mode == 2) {
    if (offset == record_size) {
      model->error = kDeviceDecodeInvalidRecord;
      return;
    }
    model->reference2 = record[offset++];
    if ((model->reference2 >= anchor_ct) ||
        (model->reference1 == model->reference2)) {
      model->error = kDeviceDecodeInvalidSelector;
      return;
    }
  }
  uint32_t row_ct = 1;
  if (model->mode == 0) {
    model->context_mask = 1;
  } else if (model->mode == 1) {
    row_ct = 4;
    if (offset == record_size) {
      model->error = kDeviceDecodeInvalidModel;
      return;
    }
    model->context_mask = record[offset++];
    if (model->context_mask & 0xfff0U) {
      model->error = kDeviceDecodeInvalidModel;
      return;
    }
  } else {
    row_ct = 16;
    if (offset + 2 > record_size) {
      model->error = kDeviceDecodeInvalidModel;
      return;
    }
    model->context_mask = ReadDeviceU16(record + offset);
    offset += 2;
  }
  if (!model->context_mask) {
    model->error = kDeviceDecodeInvalidModel;
    return;
  }
  const uint32_t total_frequency = 1U << scale_bits;
  uint32_t parsed_has_entropy = 0;
  for (uint32_t context = 0; context != row_ct; ++context) {
    if (!(model->context_mask & (1U << context))) {
      continue;
    }
    if (offset == record_size) {
      model->error = kDeviceDecodeInvalidModel;
      return;
    }
    DeviceModelRow& row = model->rows[context];
    const uint8_t symbol_mask = record[offset++];
    if ((!symbol_mask) || (symbol_mask & 0xf0U)) {
      model->error = kDeviceDecodeInvalidModel;
      return;
    }
    for (uint32_t symbol = 0; symbol != 4; ++symbol) {
      if (symbol_mask & (1U << symbol)) {
        row.deterministic_symbol = static_cast<uint8_t>(symbol);
        ++row.active_symbol_ct;
      }
    }
    uint32_t frequency_sum = 0;
    uint32_t remaining = row.active_symbol_ct;
    for (uint32_t symbol = 0; symbol != 4; ++symbol) {
      row.cumulative[symbol] = frequency_sum;
      if (!(symbol_mask & (1U << symbol))) {
        continue;
      }
      --remaining;
      uint32_t frequency;
      if (remaining) {
        if (offset + 2 > record_size) {
          model->error = kDeviceDecodeInvalidModel;
          return;
        }
        frequency = ReadDeviceU16(record + offset);
        offset += 2;
        if (!frequency) {
          model->error = kDeviceDecodeInvalidModel;
          return;
        }
      } else {
        if (frequency_sum >= total_frequency) {
          model->error = kDeviceDecodeInvalidModel;
          return;
        }
        frequency = total_frequency - frequency_sum;
      }
      row.frequencies[symbol] = frequency;
      frequency_sum += frequency;
    }
    if (frequency_sum != total_frequency) {
      model->error = kDeviceDecodeInvalidModel;
      return;
    }
    parsed_has_entropy |= (row.active_symbol_ct > 1);
  }
  if (parsed_has_entropy != model->has_entropy) {
    model->error = kDeviceDecodeInvalidModel;
    return;
  }
  model->entropy_offset = offset;
  if ((!model->has_entropy) && (offset != record_size)) {
    model->error = kDeviceDecodeInvalidRecord;
  }
}

__global__ void DecodeRecordsKernel(
    const uint8_t* record_bytes,
    const DeviceRecordDescriptor* descriptors, uint32_t descriptor_ct,
    uint32_t sample_ct, uint32_t scale_bits, uint64_t* output,
    uint32_t output_word_stride, uint32_t* global_error) {
  const uint32_t linear_thread_idx =
      blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t warp_idx = linear_thread_idx / kWarpSize;
  const uint32_t lane = threadIdx.x & (kWarpSize - 1);
  if (warp_idx >= descriptor_ct) {
    return;
  }
  constexpr uint32_t kWarpsPerBlock = kThreadsPerBlock / kWarpSize;
  __shared__ DeviceWarpModel warp_models[kWarpsPerBlock];
  DeviceWarpModel* model =
      &(warp_models[threadIdx.x / kWarpSize]);
  const DeviceRecordDescriptor descriptor = descriptors[warp_idx];
  const uint8_t* record = record_bytes + descriptor.record_offset;
  if (!lane) {
    ParseDeviceModel(record, descriptor.record_size, scale_bits,
                     descriptor.anchor_ct, model);
    if (model->error) {
      atomicCAS(global_error, kDeviceDecodeSuccess, model->error);
    }
  }
  __syncwarp();
  if (model->error) {
    return;
  }

  const uint64_t* reference1 = nullptr;
  const uint64_t* reference2 = nullptr;
  if (model->mode != 0) {
    const uint32_t anchor_offset = ScheduledAnchorOffset(
        model->reference1, descriptor.block_variant_ct,
        descriptor.anchor_ct);
    reference1 =
        output +
        static_cast<size_t>(descriptor.block_output_variant +
                            anchor_offset) *
            output_word_stride;
  }
  if (model->mode == 2) {
    const uint32_t anchor_offset = ScheduledAnchorOffset(
        model->reference2, descriptor.block_variant_ct,
        descriptor.anchor_ct);
    reference2 =
        output +
        static_cast<size_t>(descriptor.block_output_variant +
                            anchor_offset) *
            output_word_stride;
  }

  uint32_t state = kRansLowerBound;
  const uint8_t* lane_start = nullptr;
  const uint8_t* lane_iter = nullptr;
  uint32_t interleaved_offset = 0;
  uint32_t interleaved_end_offset = 0;
  bool lane_error = false;
  if (model->has_entropy) {
    const uint32_t state_offset = model->entropy_offset;
    const uint32_t state_end_offset =
        state_offset + kWarpSize * sizeof(uint32_t);
    if (state_end_offset > descriptor.record_size) {
      lane_error = true;
    } else {
      state = ReadDeviceU32(record + state_offset +
                            lane * sizeof(uint32_t));
      if (state < kRansLowerBound) {
        lane_error = true;
      }
      if (model->has_interleaved_payload) {
        constexpr uint32_t kPaddingByteCt = kWarpSize / 2 - 1;
        if (descriptor.record_size <
            state_end_offset + kPaddingByteCt) {
          lane_error = true;
        } else {
          interleaved_offset = state_end_offset;
          interleaved_end_offset =
              descriptor.record_size - kPaddingByteCt;
          if ((lane < kPaddingByteCt) &&
              record[interleaved_end_offset + lane]) {
            lane_error = true;
          }
        }
      } else {
        const uint32_t boundary_offset = state_end_offset;
        const uint32_t payload_offset =
            boundary_offset +
            (kWarpSize - 1) * sizeof(uint32_t);
        if (payload_offset > descriptor.record_size) {
          lane_error = true;
        } else {
          const uint32_t payload_size =
              descriptor.record_size - payload_offset;
          const uint32_t lane_begin =
              lane ? ReadDeviceU32(
                         record + boundary_offset +
                         (lane - 1) * sizeof(uint32_t))
                   : 0;
          const uint32_t lane_end =
              (lane + 1 == kWarpSize)
                  ? payload_size
                  : ReadDeviceU32(
                        record + boundary_offset +
                        lane * sizeof(uint32_t));
          if ((lane_begin > lane_end) ||
              (lane_end > payload_size)) {
            lane_error = true;
          } else {
            lane_start = record + payload_offset + lane_begin;
            lane_iter = record + payload_offset + lane_end;
          }
        }
      }
    }
  }

  const uint32_t word_ct = (sample_ct + 31) / 32;
  uint64_t* target =
      output + static_cast<size_t>(descriptor.output_variant) * word_ct;
  const uint32_t slot_mask = (1U << scale_bits) - 1;
  for (uint32_t word_idx = 0; word_idx != word_ct; ++word_idx) {
    const uint32_t sample_idx = word_idx * kWarpSize + lane;
    uint32_t symbol = 0;
    if ((sample_idx < sample_ct) && (!lane_error)) {
      uint32_t context = 0;
      if (model->mode != 0) {
        context = static_cast<uint32_t>(
            (reference1[word_idx] >> (2 * lane)) & 3U);
      }
      if (model->mode == 2) {
        context =
            4 * context +
            static_cast<uint32_t>(
                (reference2[word_idx] >> (2 * lane)) & 3U);
      }
      const DeviceModelRow& row = model->rows[context];
      if (!row.active_symbol_ct) {
        lane_error = true;
      } else if (row.active_symbol_ct == 1) {
        symbol = row.deterministic_symbol;
      } else {
        const uint32_t slot = state & slot_mask;
        symbol =
            static_cast<uint32_t>(slot >= row.cumulative[1]) +
            static_cast<uint32_t>(slot >= row.cumulative[2]) +
            static_cast<uint32_t>(slot >= row.cumulative[3]);
        const uint32_t frequency = row.frequencies[symbol];
        state = frequency * (state >> scale_bits) + slot -
                row.cumulative[symbol];
        if (!model->has_interleaved_payload) {
          while (state < kRansLowerBound) {
            if (lane_iter == lane_start) {
              lane_error = true;
              break;
            }
            state = (state << 8) | *--lane_iter;
          }
        }
      }
    }
    if (model->has_interleaved_payload) {
      for (uint32_t group = 0; group != 2; ++group) {
        const bool lane_in_group = (lane / 16) == group;
        while (true) {
          const uint32_t refill_mask = __ballot_sync(
              0xffffffffU,
              lane_in_group && (sample_idx < sample_ct) &&
                  (!lane_error) && (state < kRansLowerBound));
          if (!refill_mask) {
            break;
          }
          const uint32_t lower_lane_mask =
              (1U << lane) - 1U;
          const uint32_t refill_rank =
              __popc(refill_mask & lower_lane_mask);
          const uint32_t refill_byte_ct = __popc(refill_mask);
          if (refill_mask & (1U << lane)) {
            if (interleaved_offset + refill_rank >=
                interleaved_end_offset) {
              lane_error = true;
            } else {
              state =
                  (state << 8) |
                  record[interleaved_offset + refill_rank];
            }
          }
          interleaved_offset += refill_byte_ct;
        }
      }
    }
    const uint32_t low_bits =
        __ballot_sync(0xffffffffU,
                      (sample_idx < sample_ct) && (symbol & 1U));
    const uint32_t high_bits =
        __ballot_sync(0xffffffffU,
                      (sample_idx < sample_ct) && (symbol & 2U));
    if (!lane) {
      target[word_idx] =
          SpreadBits32(low_bits) | (SpreadBits32(high_bits) << 1);
    }
  }
  if (model->has_entropy) {
    if (model->has_interleaved_payload) {
      if ((state != kRansLowerBound) ||
          ((!lane) &&
           (interleaved_offset != interleaved_end_offset))) {
        lane_error = true;
      }
    } else if ((lane_iter != lane_start) ||
               (state != kRansLowerBound)) {
      lane_error = true;
    }
  }
  const uint32_t error_mask =
      __ballot_sync(0xffffffffU, lane_error);
  if ((!lane) && error_mask) {
    const uint32_t error =
        (lane_start || lane_iter) ? kDeviceDecodeNoncanonicalLane
                                  : kDeviceDecodeTruncatedPayload;
    atomicCAS(global_error, kDeviceDecodeSuccess, error);
  }
}

}  // namespace

struct CudaBlockDecoder::Impl {
  Impl() {
    int device_ct = 0;
    is_available =
        (cudaGetDeviceCount(&device_ct) == cudaSuccess) && (device_ct > 0);
  }

  ~Impl() {
    cudaFree(device_record_bytes);
    cudaFree(device_anchor_descriptors);
    cudaFree(device_target_descriptors);
    cudaFree(device_error);
    cudaFreeHost(host_record_bytes);
    cudaFreeHost(host_anchor_descriptors);
    cudaFreeHost(host_target_descriptors);
    cudaFreeHost(host_error);
    for (cudaEvent_t event : timing_events) {
      if (event) {
        cudaEventDestroy(event);
      }
    }
  }

  bool EnsureTimingEvents(std::string* error) {
    if (timing_events[0]) {
      return true;
    }
    for (uint32_t event_idx = 0; event_idx != 4; ++event_idx) {
      if (!CheckCuda(cudaEventCreate(&(timing_events[event_idx])),
                     "cudaEventCreate", error)) {
        for (uint32_t cleanup_idx = 0; cleanup_idx != event_idx;
             ++cleanup_idx) {
          cudaEventDestroy(timing_events[cleanup_idx]);
          timing_events[cleanup_idx] = nullptr;
        }
        return false;
      }
    }
    return true;
  }

  bool Decode(const std::vector<const EncodedBlockView*>& blocks,
              uint32_t sample_ct, const CodecParams& params,
              void* device_output_void, size_t output_word_ct,
              void* stream_void, CudaDecodeTimings* timings,
              std::string* error) {
    if (timings) {
      *timings = {};
    }
    if ((!is_available) || blocks.empty() || (sample_ct < kWarpSize) ||
        (params.state_ct != kWarpSize) || (params.scale_bits < 8) ||
        (params.scale_bits > 16) || (!device_output_void)) {
      SetError("Invalid conditional-rANS CUDA decoder arguments.", error);
      return false;
    }
    size_t host_record_byte_ct = 0;
    size_t host_anchor_descriptor_ct = 0;
    size_t host_target_descriptor_ct = 0;
    uint64_t output_variant_ct = 0;
    for (const EncodedBlockView* block : blocks) {
      if ((!block) || (!block->variant_ct()) || (!block->anchor_ct())) {
        SetError("CUDA batch contains an invalid block view.", error);
        return false;
      }
      if (output_variant_ct + block->variant_ct() > UINT32_MAX) {
        SetError("CUDA batch contains too many variants.", error);
        return false;
      }
      for (uint32_t variant_offset = 0;
           variant_offset != block->variant_ct(); ++variant_offset) {
        const ByteSpan record = block->record(variant_offset);
        if ((!record.data) || (!record.size) ||
            (record.size > UINT32_MAX) ||
            (host_record_byte_ct >
             UINT32_MAX - record.size)) {
          SetError("CUDA batch record bytes exceed format limits.",
                   error);
          return false;
        }
        host_record_byte_ct += record.size;
      }
      host_anchor_descriptor_ct += block->anchor_ct();
      host_target_descriptor_ct +=
          block->variant_ct() - block->anchor_ct();
      output_variant_ct += block->variant_ct();
    }
    const uint32_t word_stride = (sample_ct + 31) / 32;
    if ((output_variant_ct >
         std::numeric_limits<size_t>::max() / word_stride) ||
        (output_word_ct <
         static_cast<size_t>(output_variant_ct) * word_stride)) {
      SetError("CUDA decoder output buffer is too small.", error);
      return false;
    }
    if ((timings && (!EnsureTimingEvents(error))) ||
        (!EnsurePinnedCapacity(
             &host_record_bytes, &host_record_byte_capacity,
             host_record_byte_ct, error)) ||
        (!EnsurePinnedCapacity(
             &host_anchor_descriptors, &host_anchor_descriptor_capacity,
             host_anchor_descriptor_ct, error)) ||
        (!EnsurePinnedCapacity(
             &host_target_descriptors, &host_target_descriptor_capacity,
             host_target_descriptor_ct, error)) ||
        (!EnsurePinnedCapacity(
             &host_error, &host_error_capacity, 1, error)) ||
        (!EnsureDeviceCapacity(
             &device_record_bytes, &record_byte_capacity,
             host_record_byte_ct, error)) ||
        (!EnsureDeviceCapacity(
             &device_anchor_descriptors, &anchor_descriptor_capacity,
             host_anchor_descriptor_ct, error)) ||
        (!EnsureDeviceCapacity(
             &device_target_descriptors, &target_descriptor_capacity,
             host_target_descriptor_ct, error)) ||
        (!EnsureDeviceCapacity(&device_error, &error_capacity, 1, error))) {
      return false;
    }

    size_t record_byte_offset = 0;
    size_t anchor_descriptor_idx = 0;
    size_t target_descriptor_idx = 0;
    output_variant_ct = 0;
    for (const EncodedBlockView* block : blocks) {
      const uint32_t block_output_variant =
          static_cast<uint32_t>(output_variant_ct);
      const std::vector<uint32_t> anchor_offsets =
          ScheduledAnchorOffsets(block->variant_ct(), block->anchor_ct());
      std::vector<uint8_t> is_anchor(block->variant_ct(), 0);
      for (const uint32_t offset : anchor_offsets) {
        is_anchor[offset] = 1;
      }
      for (uint32_t variant_offset = 0;
           variant_offset != block->variant_ct(); ++variant_offset) {
        const ByteSpan record = block->record(variant_offset);
        if ((!record.data) || (!record.size) ||
            (record.size > UINT32_MAX) ||
            (record_byte_offset >
             host_record_byte_ct - record.size)) {
          SetError("CUDA batch record bytes exceed format limits.", error);
          return false;
        }
        RecordMetadata metadata;
        std::string metadata_error;
        if (!ParseRecordMetadata(record.data, record.size, &metadata,
                                 &metadata_error)) {
          SetError("CUDA batch metadata: " + metadata_error, error);
          return false;
        }
        if (is_anchor[variant_offset] &&
            (metadata.mode != RecordMode::kMarginal)) {
          SetError("CUDA batch contains a conditional anchor.", error);
          return false;
        }
        const DeviceRecordDescriptor descriptor = {
            static_cast<uint32_t>(record_byte_offset),
            static_cast<uint32_t>(record.size),
            block_output_variant + variant_offset,
            block_output_variant,
            block->variant_ct(),
            block->anchor_ct()};
        memcpy(host_record_bytes + record_byte_offset, record.data,
               record.size);
        record_byte_offset += record.size;
        if (is_anchor[variant_offset]) {
          host_anchor_descriptors[anchor_descriptor_idx++] = descriptor;
        } else {
          host_target_descriptors[target_descriptor_idx++] = descriptor;
        }
      }
      output_variant_ct += block->variant_ct();
    }

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_void);
    const cudaEvent_t start_event = timing_events[0];
    const cudaEvent_t upload_event = timing_events[1];
    const cudaEvent_t anchor_event = timing_events[2];
    const cudaEvent_t target_event = timing_events[3];
    bool success =
        ((!timings) ||
         CheckCuda(cudaEventRecord(start_event, stream),
                   "cudaEventRecord", error)) &&
        CheckCuda(cudaMemcpyAsync(
                      device_record_bytes, host_record_bytes,
                      host_record_byte_ct, cudaMemcpyHostToDevice,
                      stream),
                  "cudaMemcpyAsync(record bytes)", error) &&
        CheckCuda(cudaMemcpyAsync(
                      device_anchor_descriptors,
                      host_anchor_descriptors,
                      host_anchor_descriptor_ct *
                          sizeof(DeviceRecordDescriptor),
                      cudaMemcpyHostToDevice, stream),
                  "cudaMemcpyAsync(anchor descriptors)", error) &&
        CheckCuda(cudaMemsetAsync(device_error, 0, sizeof(uint32_t),
                                  stream),
                  "cudaMemsetAsync", error);
    if (success && host_target_descriptor_ct) {
      success = CheckCuda(
          cudaMemcpyAsync(device_target_descriptors,
                          host_target_descriptors,
                          host_target_descriptor_ct *
                              sizeof(DeviceRecordDescriptor),
                          cudaMemcpyHostToDevice, stream),
          "cudaMemcpyAsync(target descriptors)", error);
    }
    if (success && timings) {
      success = CheckCuda(cudaEventRecord(upload_event, stream),
                          "cudaEventRecord", error);
    }
    uint64_t* device_output =
        reinterpret_cast<uint64_t*>(device_output_void);
    if (success && host_anchor_descriptor_ct) {
      const uint32_t grid_size = static_cast<uint32_t>(
          (host_anchor_descriptor_ct * kWarpSize +
           kThreadsPerBlock - 1) /
          kThreadsPerBlock);
      DecodeRecordsKernel<<<grid_size, kThreadsPerBlock, 0, stream>>>(
          device_record_bytes, device_anchor_descriptors,
          static_cast<uint32_t>(host_anchor_descriptor_ct), sample_ct,
          params.scale_bits, device_output, word_stride, device_error);
      success = CheckCuda(cudaGetLastError(), "anchor kernel launch", error);
    }
    success =
        success &&
        ((!timings) ||
         CheckCuda(cudaEventRecord(anchor_event, stream),
                   "cudaEventRecord", error));
    if (success && host_target_descriptor_ct) {
      const uint32_t grid_size = static_cast<uint32_t>(
          (host_target_descriptor_ct * kWarpSize +
           kThreadsPerBlock - 1) /
          kThreadsPerBlock);
      DecodeRecordsKernel<<<grid_size, kThreadsPerBlock, 0, stream>>>(
          device_record_bytes, device_target_descriptors,
          static_cast<uint32_t>(host_target_descriptor_ct), sample_ct,
          params.scale_bits, device_output, word_stride, device_error);
      success = CheckCuda(cudaGetLastError(), "target kernel launch", error);
    }
    success =
        success &&
        ((!timings) ||
         CheckCuda(cudaEventRecord(target_event, stream),
                   "cudaEventRecord", error));
    success =
        success &&
        CheckCuda(cudaMemcpyAsync(host_error, device_error,
                                  sizeof(*host_error),
                                  cudaMemcpyDeviceToHost, stream),
                  "cudaMemcpyAsync(error)", error) &&
        CheckCuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize",
                                  error);
    if (success && *host_error) {
      SetError(DeviceErrorMessage(*host_error), error);
      success = false;
    }
    if (success && timings) {
      success =
          CheckCuda(cudaEventElapsedTime(
                        &(timings->upload_milliseconds), start_event,
                        upload_event),
                    "cudaEventElapsedTime(upload)", error) &&
          CheckCuda(cudaEventElapsedTime(
                        &(timings->anchor_milliseconds), upload_event,
                        anchor_event),
                    "cudaEventElapsedTime(anchor)", error) &&
          CheckCuda(cudaEventElapsedTime(
                        &(timings->target_milliseconds), anchor_event,
                        target_event),
                    "cudaEventElapsedTime(target)", error);
      if (success) {
        timings->genotype_call_ct = output_variant_ct * sample_ct;
      }
    }
    return success;
  }

  bool is_available = false;
  uint8_t* device_record_bytes = nullptr;
  DeviceRecordDescriptor* device_anchor_descriptors = nullptr;
  DeviceRecordDescriptor* device_target_descriptors = nullptr;
  uint32_t* device_error = nullptr;
  size_t record_byte_capacity = 0;
  size_t anchor_descriptor_capacity = 0;
  size_t target_descriptor_capacity = 0;
  size_t error_capacity = 0;
  uint8_t* host_record_bytes = nullptr;
  DeviceRecordDescriptor* host_anchor_descriptors = nullptr;
  DeviceRecordDescriptor* host_target_descriptors = nullptr;
  uint32_t* host_error = nullptr;
  size_t host_record_byte_capacity = 0;
  size_t host_anchor_descriptor_capacity = 0;
  size_t host_target_descriptor_capacity = 0;
  size_t host_error_capacity = 0;
  cudaEvent_t timing_events[4] = {};
};

CudaBlockDecoder::CudaBlockDecoder() : impl_(new Impl()) {}

CudaBlockDecoder::~CudaBlockDecoder() = default;

bool CudaBlockDecoder::available() const {
  return impl_->is_available;
}

bool CudaBlockDecoder::Decode(
    const std::vector<const EncodedBlockView*>& blocks,
    uint32_t sample_ct, const CodecParams& params, void* device_output,
    size_t output_word_ct, void* cuda_stream, CudaDecodeTimings* timings,
    std::string* error) {
  return impl_->Decode(blocks, sample_ct, params, device_output,
                       output_word_ct, cuda_stream, timings, error);
}

}  // namespace pgen_rans
