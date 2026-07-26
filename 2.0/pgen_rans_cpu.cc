// SPDX-License-Identifier: GPL-3.0-or-later

#include "pgen_rans_cpu.h"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstring>
#include <limits>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

namespace pgen_rans {
namespace {

void SetError(const std::string& message, std::string* error) {
  if (error) {
    *error = message;
  }
}

}  // namespace

struct CpuBlockDecoder::Impl {
  explicit Impl(uint32_t requested_thread_ct)
      : thread_ct(requested_thread_ct
                      ? requested_thread_ct
                      : std::max(1U, std::thread::hardware_concurrency())) {
    workers.reserve(thread_ct);
    try {
      for (uint32_t thread_idx = 0; thread_idx != thread_ct; ++thread_idx) {
        workers.emplace_back([this]() { Worker(); });
      }
    } catch (...) {
      {
        std::lock_guard<std::mutex> lock(work_mutex);
        stopping = true;
      }
      work_ready.notify_all();
      for (std::thread& worker : workers) {
        worker.join();
      }
      workers.clear();
      thread_ct = 0;
    }
  }

  ~Impl() {
    {
      std::lock_guard<std::mutex> lock(work_mutex);
      stopping = true;
    }
    work_ready.notify_all();
    for (std::thread& worker : workers) {
      worker.join();
    }
  }

  void SetFailure(const std::string& message) {
    failed.store(true, std::memory_order_relaxed);
    std::lock_guard<std::mutex> lock(error_mutex);
    if (worker_error.empty()) {
      worker_error = message;
    }
  }

  void RunTasks() {
    while (!failed.load(std::memory_order_relaxed)) {
      const uint32_t task_idx =
          next_task.fetch_add(1, std::memory_order_relaxed);
      if (task_idx >= task_offsets->size()) {
        break;
      }
      const uint32_t variant_offset = (*task_offsets)[task_idx];
      if (project_phase) {
        uint8_t* destination =
            project_output +
            static_cast<size_t>(variant_offset) * project_output_stride;
        memset(destination, 0, project_output_byte_ct);
        const uint64_t* source =
            project_input +
            static_cast<size_t>(variant_offset) * project_input_word_stride;
        for (uint32_t subset_idx = 0;
             subset_idx != project_subset_sample_ct; ++subset_idx) {
          const uint32_t source_idx = project_sample_indices[subset_idx];
          const uint8_t genotype = static_cast<uint8_t>(
              (source[source_idx / 32] >>
               (2 * (source_idx % 32))) &
              3U);
          destination[subset_idx / 4] |=
              static_cast<uint8_t>(
                  genotype << (2 * (subset_idx % 4)));
        }
        continue;
      }
      const ByteSpan record = block->record(variant_offset);
      if ((!record.data) || (!record.size)) {
        SetFailure("Block decoder received an invalid record view.");
        break;
      }
      RecordMetadata metadata;
      std::string local_error;
      const uint64_t* const* task_anchors =
          anchor_phase ? nullptr : anchors->data();
      const uint32_t task_anchor_ct =
          anchor_phase ? 0 : static_cast<uint32_t>(anchors->size());
      if (!DecodeRecordToBufferFromValidatedBlock(
              record.data, record.size, task_anchors, task_anchor_ct,
              sample_ct, params,
              output + static_cast<size_t>(variant_offset) * word_stride,
              word_stride, &metadata, &local_error)) {
        SetFailure("Variant " +
                   std::to_string(block->first_variant() + variant_offset) +
                   ": " + local_error);
        break;
      }
      if (anchor_phase && (metadata.mode != RecordMode::kMarginal)) {
        SetFailure("Scheduled anchor variant " +
                   std::to_string(block->first_variant() + variant_offset) +
                   " is conditionally encoded.");
        break;
      }
    }
  }

  void Worker() {
    uint64_t observed_generation = 0;
    std::unique_lock<std::mutex> lock(work_mutex);
    while (true) {
      work_ready.wait(lock, [&]() {
        return stopping || (generation != observed_generation);
      });
      if (stopping) {
        return;
      }
      observed_generation = generation;
      lock.unlock();
      RunTasks();
      lock.lock();
      if (!--pending_worker_ct) {
        work_done.notify_one();
      }
    }
  }

  bool RunPhase(const std::vector<uint32_t>& offsets,
                bool is_anchor_phase, bool is_project_phase,
                std::string* error) {
    if (offsets.empty()) {
      return true;
    }
    {
      std::lock_guard<std::mutex> lock(work_mutex);
      task_offsets = &offsets;
      anchor_phase = is_anchor_phase;
      project_phase = is_project_phase;
      next_task.store(0, std::memory_order_relaxed);
      failed.store(false, std::memory_order_relaxed);
      {
        std::lock_guard<std::mutex> error_lock(error_mutex);
        worker_error.clear();
      }
      pending_worker_ct = static_cast<uint32_t>(workers.size());
      ++generation;
    }
    work_ready.notify_all();
    {
      std::unique_lock<std::mutex> lock(work_mutex);
      work_done.wait(lock, [&]() { return !pending_worker_ct; });
    }
    if (failed.load(std::memory_order_relaxed)) {
      std::lock_guard<std::mutex> lock(error_mutex);
      SetError(worker_error, error);
      return false;
    }
    return true;
  }

  bool Decode(const EncodedBlockView& block_arg, uint32_t sample_ct_arg,
              const CodecParams& params_arg, uint64_t* output_arg,
              size_t output_word_ct, std::string* error) {
    if (workers.empty()) {
      SetError("Could not start conditional-rANS CPU decoder threads.",
               error);
      return false;
    }
    if ((!block_arg.variant_ct()) || (!sample_ct_arg) || (!output_arg)) {
      SetError("Invalid conditional-rANS block decoder arguments.", error);
      return false;
    }
    const uint32_t stride = PackedWordCt(sample_ct_arg);
    if (block_arg.variant_ct() >
        std::numeric_limits<size_t>::max() / stride) {
      SetError("Decoded block size exceeds platform limits.", error);
      return false;
    }
    const size_t required_word_ct =
        static_cast<size_t>(block_arg.variant_ct()) * stride;
    if (output_word_ct < required_word_ct) {
      SetError("Decoded block output buffer is too small.", error);
      return false;
    }
    std::vector<uint32_t> anchor_offsets =
        ScheduledAnchorOffsets(block_arg.variant_ct(),
                               block_arg.anchor_ct());
    if (anchor_offsets.size() != block_arg.anchor_ct()) {
      SetError("Conditional-rANS block anchor schedule is invalid.", error);
      return false;
    }
    std::vector<uint8_t> is_anchor(block_arg.variant_ct(), 0);
    for (const uint32_t offset : anchor_offsets) {
      is_anchor[offset] = 1;
    }
    std::vector<uint32_t> target_offsets;
    target_offsets.reserve(block_arg.variant_ct() - anchor_offsets.size());
    for (uint32_t offset = 0; offset != block_arg.variant_ct(); ++offset) {
      if (!is_anchor[offset]) {
        target_offsets.push_back(offset);
      }
    }

    block = &block_arg;
    sample_ct = sample_ct_arg;
    params = params_arg;
    output = output_arg;
    word_stride = stride;
    if (!RunPhase(anchor_offsets, true, false, error)) {
      return false;
    }
    std::vector<const uint64_t*> anchor_ptrs(anchor_offsets.size());
    for (uint32_t anchor_idx = 0; anchor_idx != anchor_offsets.size();
         ++anchor_idx) {
      anchor_ptrs[anchor_idx] =
          output + static_cast<size_t>(anchor_offsets[anchor_idx]) * stride;
    }
    anchors = &anchor_ptrs;
    return RunPhase(target_offsets, false, false, error);
  }

  bool ProjectSampleSubset(const uint64_t* input_arg,
                           uint32_t variant_ct,
                           uint32_t raw_sample_ct,
                           const uint32_t* sample_indices,
                           uint32_t subset_sample_ct,
                           uint8_t* output_arg,
                           size_t output_variant_stride,
                           std::string* error) {
    if (workers.empty()) {
      SetError("Could not start conditional-rANS CPU decoder threads.",
               error);
      return false;
    }
    if ((!input_arg) || (!variant_ct) || (!raw_sample_ct) ||
        (!sample_indices) || (!subset_sample_ct) ||
        (subset_sample_ct > raw_sample_ct) || (!output_arg)) {
      SetError("Invalid conditional-rANS sample projection arguments.",
               error);
      return false;
    }
    const size_t output_byte_ct =
        (static_cast<size_t>(subset_sample_ct) + 3) / 4;
    if (output_variant_stride < output_byte_ct) {
      SetError("Conditional-rANS sample projection stride is too small.",
               error);
      return false;
    }
    for (uint32_t subset_idx = 0; subset_idx != subset_sample_ct;
         ++subset_idx) {
      if ((sample_indices[subset_idx] >= raw_sample_ct) ||
          (subset_idx &&
           (sample_indices[subset_idx - 1] >=
            sample_indices[subset_idx]))) {
        SetError(
            "Conditional-rANS sample projection indices must be sorted "
            "and unique.",
            error);
        return false;
      }
    }
    if (variant_ct >
        std::numeric_limits<size_t>::max() / output_variant_stride) {
      SetError("Conditional-rANS projected output exceeds platform limits.",
               error);
      return false;
    }
    std::vector<uint32_t> variant_offsets(variant_ct);
    for (uint32_t variant_idx = 0; variant_idx != variant_ct;
         ++variant_idx) {
      variant_offsets[variant_idx] = variant_idx;
    }
    project_input = input_arg;
    project_input_word_stride = PackedWordCt(raw_sample_ct);
    project_sample_indices = sample_indices;
    project_subset_sample_ct = subset_sample_ct;
    project_output = output_arg;
    project_output_stride = output_variant_stride;
    project_output_byte_ct = output_byte_ct;
    return RunPhase(variant_offsets, false, true, error);
  }

  uint32_t thread_ct = 0;
  std::vector<std::thread> workers;
  std::mutex work_mutex;
  std::condition_variable work_ready;
  std::condition_variable work_done;
  bool stopping = false;
  uint64_t generation = 0;
  uint32_t pending_worker_ct = 0;

  const EncodedBlockView* block = nullptr;
  const std::vector<uint32_t>* task_offsets = nullptr;
  const std::vector<const uint64_t*>* anchors = nullptr;
  uint32_t sample_ct = 0;
  CodecParams params;
  uint64_t* output = nullptr;
  size_t word_stride = 0;
  bool anchor_phase = false;
  bool project_phase = false;
  const uint64_t* project_input = nullptr;
  size_t project_input_word_stride = 0;
  const uint32_t* project_sample_indices = nullptr;
  uint32_t project_subset_sample_ct = 0;
  uint8_t* project_output = nullptr;
  size_t project_output_stride = 0;
  size_t project_output_byte_ct = 0;
  std::atomic<uint32_t> next_task = {0};
  std::atomic<bool> failed = {false};
  std::mutex error_mutex;
  std::string worker_error;
};

CpuBlockDecoder::CpuBlockDecoder(uint32_t thread_ct)
    : impl_(new Impl(thread_ct)) {}

CpuBlockDecoder::~CpuBlockDecoder() = default;

bool CpuBlockDecoder::Decode(const EncodedBlockView& block,
                             uint32_t sample_ct,
                             const CodecParams& params, uint64_t* output,
                             size_t output_word_ct, std::string* error) {
  return impl_->Decode(block, sample_ct, params, output, output_word_ct,
                       error);
}

bool CpuBlockDecoder::ProjectSampleSubset(
    const uint64_t* input, uint32_t variant_ct, uint32_t raw_sample_ct,
    const uint32_t* sample_indices, uint32_t subset_sample_ct,
    uint8_t* output, size_t output_variant_stride, std::string* error) {
  return impl_->ProjectSampleSubset(
      input, variant_ct, raw_sample_ct, sample_indices,
      subset_sample_ct, output, output_variant_stride, error);
}

uint32_t CpuBlockDecoder::thread_ct() const {
  return impl_->thread_ct;
}

}  // namespace pgen_rans
