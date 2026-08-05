// SPDX-License-Identifier: GPL-3.0-or-later

#include "pgen_rans_reader.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <list>
#include <limits>
#include <utility>
#include <vector>

#include "pgen_rans_codec.h"
#include "pgen_rans_container.h"
#include "pgen_rans_cpu.h"
#include "pgen_rans_hybrid.h"

namespace pgen_rans {
namespace {

void SetError(const std::string& message, std::string* error) {
  if (error) {
    *error = message;
  }
}

}  // namespace

struct PackedVariantReader::Impl {
  struct SparseParseCacheEntry {
    bool parsed = false;
    bool is_sparse_predictor = false;
    ParsedSparsePredictor record;

    size_t ByteCt() const {
      return record.exception_sample_ids.size() * sizeof(uint32_t) +
             record.exception_genotypes.size();
    }
  };

  struct CachedBlock {
    size_t ByteCt() const {
      return storage.size() +
             decoded.size() * sizeof(uint64_t) +
             decoded_flags.size() + projected.size() +
             projected_flags.size() +
             anchor_offsets.size() * sizeof(uint32_t) + SparseByteCt();
    }

    size_t SparseByteCt() const {
      size_t result =
          sparse_records.size() * sizeof(SparseParseCacheEntry);
      for (const SparseParseCacheEntry& entry : sparse_records) {
        result += entry.ByteCt();
      }
      return result;
    }

    uint32_t block_idx = UINT32_MAX;
    std::vector<uint8_t> storage;
    EncodedBlockView view;
    std::vector<uint64_t> decoded;
    std::vector<uint8_t> decoded_flags;
    std::vector<uint8_t> projected;
    std::vector<uint8_t> projected_flags;
    bool anchor_offsets_initialized = false;
    std::vector<uint32_t> anchor_offsets;
    std::vector<SparseParseCacheEntry> sparse_records;
  };

  static constexpr size_t kCachedBlockByteLimit =
      64ULL * 1024 * 1024;

  bool Open(const std::string& path, uint32_t thread_ct,
            std::string* error) {
    Close();
    if (!reader.Open(path, error)) {
      return false;
    }
    const uint16_t endian_probe = 1;
    if (!*reinterpret_cast<const uint8_t*>(&endian_probe)) {
      SetError("Conditional-rANS packed reader requires little-endian words.",
               error);
      Close();
      return false;
    }
    decoder.reset(new CpuBlockDecoder(thread_ct));
    if (!decoder->thread_ct()) {
      SetError("Could not start conditional-rANS CPU decoder workers.",
               error);
      Close();
      return false;
    }
    params = {reader.params().state_ct, reader.params().scale_bits};
    word_stride = PackedWordCt(reader.params().sample_ct);
    ClearSampleSubset();
    return true;
  }

  void Close() {
    decoder.reset();
    reader.Close();
    params = {};
    word_stride = 0;
    output_sample_ct = 0;
    packed_byte_ct = 0;
    sample_subset.clear();
    cached_block_idx = UINT32_MAX;
    block_storage.clear();
    decoded_block.clear();
    decoded_flags.clear();
    projected_block.clear();
    projected_flags.clear();
    anchor_offsets_initialized = false;
    anchor_offsets.clear();
    sparse_records.clear();
    sparse_scratch = SparseHardcallResult();
    block_view = {};
    cached_blocks.clear();
    cached_block_byte_ct = 0;
  }

  void ClearSampleSubset() {
    sample_subset.clear();
    output_sample_ct = reader.params().sample_ct;
    packed_byte_ct =
        (static_cast<size_t>(output_sample_ct) + 3) / 4;
  }

  bool SetSampleSubset(const uint32_t* sample_indices,
                       uint32_t subset_sample_ct, std::string* error) {
    if ((!decoder) || (!sample_indices) || (!subset_sample_ct) ||
        (subset_sample_ct > reader.params().sample_ct)) {
      SetError("Invalid conditional-rANS sample subset.", error);
      return false;
    }
    std::vector<uint32_t> candidate(
        sample_indices, sample_indices + subset_sample_ct);
    for (uint32_t subset_idx = 0; subset_idx != subset_sample_ct;
         ++subset_idx) {
      if ((candidate[subset_idx] >= reader.params().sample_ct) ||
          (subset_idx &&
           (candidate[subset_idx - 1] >= candidate[subset_idx]))) {
        SetError(
            "Conditional-rANS sample subset must be sorted and unique.",
            error);
        return false;
      }
    }
    sample_subset.swap(candidate);
    output_sample_ct = subset_sample_ct;
    packed_byte_ct =
        (static_cast<size_t>(output_sample_ct) + 3) / 4;
    if (subset_sample_ct == reader.params().sample_ct) {
      bool identity = true;
      for (uint32_t sample_idx = 0; sample_idx != subset_sample_ct;
           ++sample_idx) {
        if (sample_subset[sample_idx] != sample_idx) {
          identity = false;
          break;
        }
      }
      if (identity) {
        sample_subset.clear();
      }
    }
    projected_block.clear();
    projected_flags.clear();
    cached_block_byte_ct = 0;
    for (CachedBlock& block : cached_blocks) {
      block.projected.clear();
      block.projected_flags.clear();
      cached_block_byte_ct += block.ByteCt();
    }
    TrimBlockCache();
    return true;
  }

  void TrimBlockCache() {
    while ((cached_block_byte_ct > kCachedBlockByteLimit) &&
           (!cached_blocks.empty())) {
      cached_block_byte_ct -= cached_blocks.back().ByteCt();
      cached_blocks.pop_back();
    }
  }

  void StashActiveBlock() {
    if (cached_block_idx == UINT32_MAX) {
      return;
    }
    CachedBlock block;
    block.block_idx = cached_block_idx;
    block.storage = std::move(block_storage);
    block.view = std::move(block_view);
    block.decoded = std::move(decoded_block);
    block.decoded_flags = std::move(decoded_flags);
    block.projected = std::move(projected_block);
    block.projected_flags = std::move(projected_flags);
    block.anchor_offsets_initialized = anchor_offsets_initialized;
    block.anchor_offsets = std::move(anchor_offsets);
    block.sparse_records = std::move(sparse_records);
    cached_block_byte_ct += block.ByteCt();
    cached_blocks.push_front(std::move(block));
    cached_block_idx = UINT32_MAX;
    block_view = {};
    TrimBlockCache();
  }

  bool RestoreCachedBlock(uint32_t block_idx) {
    auto iter = std::find_if(
        cached_blocks.begin(), cached_blocks.end(),
        [block_idx](const CachedBlock& block) {
          return block.block_idx == block_idx;
        });
    if (iter == cached_blocks.end()) {
      return false;
    }
    CachedBlock requested = std::move(*iter);
    cached_block_byte_ct -= requested.ByteCt();
    cached_blocks.erase(iter);
    StashActiveBlock();
    cached_block_idx = requested.block_idx;
    block_storage = std::move(requested.storage);
    block_view = std::move(requested.view);
    decoded_block = std::move(requested.decoded);
    decoded_flags = std::move(requested.decoded_flags);
    projected_block = std::move(requested.projected);
    projected_flags = std::move(requested.projected_flags);
    anchor_offsets_initialized = requested.anchor_offsets_initialized;
    anchor_offsets = std::move(requested.anchor_offsets);
    sparse_records = std::move(requested.sparse_records);
    return true;
  }

  bool LoadBlock(uint32_t block_idx, PackedReadStats* stats,
                 std::string* error) {
    if (block_idx == cached_block_idx) {
      return true;
    }
    if (RestoreCachedBlock(block_idx)) {
      return true;
    }
    StashActiveBlock();
    const auto read_start = std::chrono::steady_clock::now();
    if (!reader.ReadBlockView(block_idx, &block_storage, &block_view,
                              error)) {
      return false;
    }
    const double read_seconds =
        std::chrono::duration<double>(
            std::chrono::steady_clock::now() - read_start)
            .count();
    if (block_view.variant_ct() >
        std::numeric_limits<size_t>::max() / word_stride) {
      SetError("Decoded conditional-rANS block exceeds platform limits.",
               error);
      return false;
    }
    decoded_block.clear();
    decoded_flags.clear();
    projected_block.clear();
    projected_flags.clear();
    anchor_offsets_initialized = false;
    anchor_offsets.clear();
    sparse_records.clear();
    cached_block_idx = block_idx;
    if (stats) {
      ++stats->block_read_ct;
      stats->block_byte_ct += reader.block_index()[block_idx].byte_ct;
      stats->block_read_seconds += read_seconds;
    }
    return true;
  }

  bool GetSparseRecord(uint32_t variant_offset,
                       const ParsedSparsePredictor** parsed,
                       bool* is_sparse_predictor, std::string* error) {
    if ((variant_offset >= block_view.variant_ct()) || (!parsed) ||
        (!is_sparse_predictor)) {
      SetError("Invalid conditional-rANS sparse record request.", error);
      return false;
    }
    if (sparse_records.empty()) {
      sparse_records.resize(block_view.variant_ct());
    }
    SparseParseCacheEntry& entry = sparse_records[variant_offset];
    if (!entry.parsed) {
      const ByteSpan record = block_view.record(variant_offset);
      bool is_sparse = false;
      if ((!record.data) || (!record.size) ||
          (!ParseSparsePredictorRecord(
              record.data, record.size, reader.params().sample_ct,
              &entry.record, &is_sparse, error))) {
        return false;
      }
      entry.is_sparse_predictor = is_sparse;
      entry.parsed = true;
    }
    *parsed = &entry.record;
    *is_sparse_predictor = entry.is_sparse_predictor;
    return true;
  }

  bool TryReadSparse(uint32_t variant, uint32_t max_difflist_len,
                     SparseHardcallResult* sparse, bool* is_sparse,
                     PackedReadStats* stats, std::string* error) {
    *is_sparse = false;
    uint32_t block_idx;
    if (!reader.FindBlock(variant, &block_idx, error) ||
        !LoadBlock(block_idx, stats, error)) {
      return false;
    }
    const uint32_t variant_offset = variant - block_view.first_variant();
    const ParsedSparsePredictor* target;
    bool target_is_sparse;
    if (!GetSparseRecord(
            variant_offset, &target, &target_is_sparse, error)) {
      return false;
    }
    if (!target_is_sparse) {
      return true;
    }

    if (!anchor_offsets_initialized) {
      anchor_offsets = ScheduledAnchorOffsets(
          block_view.variant_ct(), block_view.anchor_ct());
      if (anchor_offsets.size() != block_view.anchor_ct()) {
        SetError("Conditional-rANS block anchor schedule is invalid.", error);
        return false;
      }
      anchor_offsets_initialized = true;
    }

    const uint32_t row_ct =
        (target->mode == RecordMode::kMarginal)
            ? 1
            : ((target->mode == RecordMode::kOneReference) ? 4 : 16);
    bool constant_prediction = true;
    for (uint32_t context = 1; context != row_ct; ++context) {
      if (target->predictions[context] != target->predictions[0]) {
        constant_prediction = false;
        break;
      }
    }

    if ((target->mode != RecordMode::kMarginal) &&
        (target->reference1 >= anchor_offsets.size())) {
      SetError("Sparse record has an invalid first anchor selector.", error);
      return false;
    }
    if ((target->mode == RecordMode::kTwoReference) &&
        (target->reference2 >= anchor_offsets.size())) {
      SetError("Sparse record has an invalid second anchor selector.", error);
      return false;
    }

    const ParsedSparsePredictor* anchors[2] = {nullptr, nullptr};
    if ((target->mode != RecordMode::kMarginal) && !constant_prediction) {
      const uint32_t anchor_idx[2] = {
          target->reference1, target->reference2};
      const uint32_t needed_anchor_ct =
          (target->mode == RecordMode::kTwoReference) ? 2 : 1;
      for (uint32_t idx = 0; idx != needed_anchor_ct; ++idx) {
        bool anchor_is_sparse;
        if (!GetSparseRecord(anchor_offsets[anchor_idx[idx]], &anchors[idx],
                             &anchor_is_sparse, error)) {
          return false;
        }
        if (!anchor_is_sparse) {
          return true;
        }
        if (anchors[idx]->mode != RecordMode::kMarginal) {
          SetError("Scheduled sparse anchor is conditionally encoded.", error);
          return false;
        }
      }
    }

    const uint8_t anchor_common[2] = {
        anchors[0] ? anchors[0]->predictions[0] : static_cast<uint8_t>(0),
        anchors[1] ? anchors[1]->predictions[0] : static_cast<uint8_t>(0)};
    uint32_t common_context = 0;
    if (!constant_prediction &&
        (target->mode != RecordMode::kMarginal)) {
      common_context = anchor_common[0];
      if (target->mode == RecordMode::kTwoReference) {
        common_context = 4 * common_context + anchor_common[1];
      }
    }
    SparseHardcallResult* const raw =
        sample_subset.empty() ? sparse : &sparse_scratch;
    raw->common_genotype = target->predictions[common_context];
    raw->sample_ids.clear();
    raw->genotypes.clear();
    const uint64_t requested_reserve_ct =
        static_cast<uint64_t>(target->exception_sample_ids.size()) +
        (anchors[0] ? anchors[0]->exception_sample_ids.size() : 0) +
        (anchors[1] ? anchors[1]->exception_sample_ids.size() : 0);
    const size_t reserve_ct = static_cast<size_t>(std::min<uint64_t>(
        reader.params().sample_ct, requested_reserve_ct));
    raw->sample_ids.reserve(reserve_ct);
    raw->genotypes.reserve(reserve_ct);

    size_t target_idx = 0;
    size_t anchor_idx[2] = {0, 0};
    while ((target_idx != target->exception_sample_ids.size()) ||
           (anchors[0] &&
            (anchor_idx[0] != anchors[0]->exception_sample_ids.size())) ||
           (anchors[1] &&
            (anchor_idx[1] != anchors[1]->exception_sample_ids.size()))) {
      uint32_t sample_idx = UINT32_MAX;
      if (target_idx != target->exception_sample_ids.size()) {
        sample_idx = target->exception_sample_ids[target_idx];
      }
      for (uint32_t idx = 0; idx != 2; ++idx) {
        if (anchors[idx] &&
            (anchor_idx[idx] !=
             anchors[idx]->exception_sample_ids.size())) {
          sample_idx = std::min(
              sample_idx,
              anchors[idx]->exception_sample_ids[anchor_idx[idx]]);
        }
      }
      uint8_t anchor_genotype[2] = {
          anchor_common[0], anchor_common[1]};
      for (uint32_t idx = 0; idx != 2; ++idx) {
        if (anchors[idx] &&
            (anchor_idx[idx] !=
             anchors[idx]->exception_sample_ids.size()) &&
            (anchors[idx]->exception_sample_ids[anchor_idx[idx]] ==
             sample_idx)) {
          anchor_genotype[idx] =
              anchors[idx]->exception_genotypes[anchor_idx[idx]++];
        }
      }
      uint32_t context = 0;
      if (!constant_prediction &&
          (target->mode != RecordMode::kMarginal)) {
        context = anchor_genotype[0];
        if (target->mode == RecordMode::kTwoReference) {
          context = 4 * context + anchor_genotype[1];
        }
      }
      uint8_t genotype = target->predictions[context];
      if ((target_idx != target->exception_sample_ids.size()) &&
          (target->exception_sample_ids[target_idx] == sample_idx)) {
        const uint8_t exception_genotype =
            target->exception_genotypes[target_idx++];
        if (exception_genotype == genotype) {
          SetError("Sparse exception repeats its predictor.", error);
          return false;
        }
        genotype = exception_genotype;
      }
      if (genotype != raw->common_genotype) {
        raw->sample_ids.push_back(sample_idx);
        raw->genotypes.push_back(genotype);
      }
    }

    if (sample_subset.empty()) {
      if (sparse->sample_ids.size() > max_difflist_len) {
        sparse->common_genotype = UINT32_MAX;
        sparse->sample_ids.clear();
        sparse->genotypes.clear();
        return true;
      }
    } else {
      sparse->common_genotype = raw->common_genotype;
      sparse->sample_ids.clear();
      sparse->genotypes.clear();
      sparse->sample_ids.reserve(
          std::min<size_t>(raw->sample_ids.size(), max_difflist_len));
      sparse->genotypes.reserve(
          std::min<size_t>(raw->genotypes.size(), max_difflist_len));
      // Sparse records should remain O(k log N), even when a tiny difflist is
      // projected onto a very large subset.  Scanning the entire subset would
      // negate the main benefit of this API for 500k-sample cohorts.
      for (size_t raw_idx = 0; raw_idx != raw->sample_ids.size(); ++raw_idx) {
        const std::vector<uint32_t>::const_iterator iter = std::lower_bound(
            sample_subset.begin(), sample_subset.end(),
            raw->sample_ids[raw_idx]);
        if ((iter == sample_subset.end()) ||
            (*iter != raw->sample_ids[raw_idx])) {
          continue;
        }
        if (sparse->sample_ids.size() == max_difflist_len) {
          sparse->common_genotype = UINT32_MAX;
          sparse->sample_ids.clear();
          sparse->genotypes.clear();
          return true;
        }
        sparse->sample_ids.push_back(
            static_cast<uint32_t>(iter - sample_subset.begin()));
        sparse->genotypes.push_back(raw->genotypes[raw_idx]);
      }
    }
    *is_sparse = true;
    return true;
  }

  bool DecodeOffsets(uint32_t block_idx,
                     const std::vector<uint32_t>& variant_offsets,
                     PackedReadStats* stats, std::string* error) {
    if (variant_offsets.empty()) {
      return true;
    }
    if (!LoadBlock(block_idx, stats, error)) {
      return false;
    }
    if (decoded_flags.empty()) {
      decoded_block.resize(
          static_cast<size_t>(block_view.variant_ct()) * word_stride);
      decoded_flags.assign(block_view.variant_ct(), 0);
    }
    std::vector<uint32_t> requested_offsets;
    requested_offsets.reserve(variant_offsets.size());
    std::vector<uint8_t> requested_flags(block_view.variant_ct(), 0);
    uint32_t missing_request_ct = 0;
    for (const uint32_t variant_offset : variant_offsets) {
      if (variant_offset >= block_view.variant_ct()) {
        SetError("Conditional-rANS block request is out of range.", error);
        return false;
      }
      if (!requested_flags[variant_offset]) {
        requested_flags[variant_offset] = 1;
        requested_offsets.push_back(variant_offset);
        missing_request_ct += !decoded_flags[variant_offset];
      }
    }
    if (!missing_request_ct) {
      if (sample_subset.empty()) {
        return true;
      }
    }

    // At near-total density, decoding the whole block is slightly faster
    // than building the sparse dependency closure.  Sparse reads below this
    // measured crossover decode only requested records plus their one or two
    // scheduled marginal anchors.
    const bool decode_dense =
        missing_request_ct &&
        (missing_request_ct == block_view.variant_ct());
    const auto decode_start = std::chrono::steady_clock::now();
    uint32_t decoded_variant_ct = 0;
    if (decode_dense) {
      if (!decoder->Decode(block_view, reader.params().sample_ct, params,
                           decoded_block.data(), decoded_block.size(),
                           error)) {
        return false;
      }
      std::fill(decoded_flags.begin(), decoded_flags.end(), 1);
      decoded_variant_ct = block_view.variant_ct();
    } else if (missing_request_ct) {
      if (!decoder->DecodeSelected(
              block_view, reader.params().sample_ct, params,
              requested_offsets.data(),
              static_cast<uint32_t>(requested_offsets.size()),
              decoded_block.data(), decoded_block.size(),
              decoded_flags.data(), decoded_flags.size(),
              &decoded_variant_ct, error)) {
        return false;
      }
    }
    const double decode_seconds =
        std::chrono::duration<double>(
            std::chrono::steady_clock::now() - decode_start)
            .count();
    double projection_seconds = 0.0;
    if (!sample_subset.empty()) {
      if (projected_block.empty()) {
        projected_block.resize(
            static_cast<size_t>(block_view.variant_ct()) * packed_byte_ct);
        projected_flags.assign(block_view.variant_ct(), 0);
      }
      const auto projection_start =
          std::chrono::steady_clock::now();
      if (decode_dense) {
        if (!decoder->ProjectSampleSubset(
                decoded_block.data(), block_view.variant_ct(),
                reader.params().sample_ct, sample_subset.data(),
                output_sample_ct, projected_block.data(), packed_byte_ct,
                error)) {
          return false;
        }
        std::fill(projected_flags.begin(), projected_flags.end(), 1);
      } else {
        std::vector<uint32_t> projection_offsets;
        projection_offsets.reserve(requested_offsets.size());
        for (const uint32_t variant_offset : requested_offsets) {
          if (!projected_flags[variant_offset]) {
            projection_offsets.push_back(variant_offset);
          }
        }
        if ((!projection_offsets.empty()) &&
            (!decoder->ProjectSampleSubsetSelected(
                decoded_block.data(), block_view.variant_ct(),
                reader.params().sample_ct, projection_offsets.data(),
                static_cast<uint32_t>(projection_offsets.size()),
                sample_subset.data(), output_sample_ct,
                projected_block.data(), packed_byte_ct, error))) {
          return false;
        }
        for (const uint32_t variant_offset : projection_offsets) {
          projected_flags[variant_offset] = 1;
        }
      }
      projection_seconds =
          std::chrono::duration<double>(
              std::chrono::steady_clock::now() - projection_start)
              .count();
    }
    if (stats) {
      stats->decoded_variant_ct += decoded_variant_ct;
      stats->decode_seconds += decode_seconds;
      stats->projection_seconds += projection_seconds;
    }
    return true;
  }

  bool ReadList(const uint32_t* variants, uint32_t variant_ct,
                uint8_t* output, size_t output_variant_stride,
                PackedReadStats* stats, std::string* error) {
    if ((!decoder) || (!variant_ct) || (!variants) || (!output) ||
        (output_variant_stride < packed_byte_ct)) {
      SetError("Invalid conditional-rANS packed read request.", error);
      return false;
    }
    if (variant_ct >
        std::numeric_limits<size_t>::max() /
            output_variant_stride) {
      SetError("Conditional-rANS packed output exceeds platform limits.",
               error);
      return false;
    }
    std::vector<std::pair<uint32_t, uint32_t>> requests;
    requests.reserve(variant_ct);
    for (uint32_t output_idx = 0; output_idx != variant_ct; ++output_idx) {
      if (variants[output_idx] >= reader.params().variant_ct) {
        SetError("Requested conditional-rANS variant is out of range.",
                 error);
        return false;
      }
      requests.emplace_back(variants[output_idx], output_idx);
    }
    std::sort(requests.begin(), requests.end());
    size_t request_begin = 0;
    while (request_begin != requests.size()) {
      uint32_t block_idx;
      if (!reader.FindBlock(requests[request_begin].first, &block_idx,
                            error)) {
        return false;
      }
      const BlockIndexEntry& entry = reader.block_index()[block_idx];
      const uint32_t block_end = entry.first_variant + entry.variant_ct;
      size_t request_end = request_begin;
      std::vector<uint32_t> variant_offsets;
      while ((request_end != requests.size()) &&
             (requests[request_end].first < block_end)) {
        variant_offsets.push_back(
            requests[request_end].first - entry.first_variant);
        ++request_end;
      }
      if (!DecodeOffsets(block_idx, variant_offsets, stats, error)) {
        return false;
      }
      for (size_t request_idx = request_begin;
           request_idx != request_end; ++request_idx) {
        const auto& request = requests[request_idx];
        uint8_t* destination =
            output +
            static_cast<size_t>(request.second) * output_variant_stride;
        CopyDecodedVariant(request.first, destination);
      }
      request_begin = request_end;
    }
    if (stats) {
      stats->returned_variant_ct += variant_ct;
    }
    return true;
  }

  void CopyDecodedVariant(uint32_t variant, uint8_t* output) const {
    const uint32_t variant_offset =
        variant - block_view.first_variant();
    const uint8_t* source = reinterpret_cast<const uint8_t*>(
        decoded_block.data() +
        static_cast<size_t>(variant_offset) * word_stride);
    if (!sample_subset.empty()) {
      source = projected_block.data() +
               static_cast<size_t>(variant_offset) * packed_byte_ct;
    }
    memcpy(output, source, packed_byte_ct);
  }

  ContainerReader reader;
  std::unique_ptr<CpuBlockDecoder> decoder;
  CodecParams params;
  uint32_t word_stride = 0;
  uint32_t output_sample_ct = 0;
  size_t packed_byte_ct = 0;
  std::vector<uint32_t> sample_subset;
  uint32_t cached_block_idx = UINT32_MAX;
  std::vector<uint8_t> block_storage;
  EncodedBlockView block_view;
  std::vector<uint64_t> decoded_block;
  std::vector<uint8_t> decoded_flags;
  std::vector<uint8_t> projected_block;
  std::vector<uint8_t> projected_flags;
  bool anchor_offsets_initialized = false;
  std::vector<uint32_t> anchor_offsets;
  std::vector<SparseParseCacheEntry> sparse_records;
  SparseHardcallResult sparse_scratch;
  std::list<CachedBlock> cached_blocks;
  size_t cached_block_byte_ct = 0;
};

PackedVariantReader::PackedVariantReader() : impl_(new Impl()) {}

PackedVariantReader::~PackedVariantReader() = default;

bool PackedVariantReader::Open(const std::string& path,
                               uint32_t thread_ct,
                               std::string* error) {
  return impl_->Open(path, thread_ct, error);
}

void PackedVariantReader::Close() {
  impl_->Close();
}

uint32_t PackedVariantReader::sample_ct() const {
  return impl_->output_sample_ct;
}

uint32_t PackedVariantReader::raw_sample_ct() const {
  return impl_->reader.params().sample_ct;
}

uint32_t PackedVariantReader::variant_ct() const {
  return impl_->reader.params().variant_ct;
}

uint32_t PackedVariantReader::max_allele_ct() const {
  return impl_->reader.params().max_allele_ct;
}

bool PackedVariantReader::all_nonref() const {
  return impl_->reader.metadata().all_nonref;
}

bool PackedVariantReader::has_mixed_nonref_flags() const {
  return !impl_->reader.metadata().nonref_flags.empty();
}

bool PackedVariantReader::variant_is_nonref(uint32_t variant) const {
  if (variant >= impl_->reader.params().variant_ct) {
    return false;
  }
  if (impl_->reader.metadata().all_nonref) {
    return true;
  }
  const std::vector<uint8_t>& flags =
      impl_->reader.metadata().nonref_flags;
  return !flags.empty() &&
         ((flags[variant / 8] >> (variant % 8)) & 1U);
}

size_t PackedVariantReader::packed_variant_byte_ct() const {
  return impl_->packed_byte_ct;
}

bool PackedVariantReader::SetSampleSubset(
    const uint32_t* sample_indices, uint32_t subset_sample_ct,
    std::string* error) {
  return impl_->SetSampleSubset(sample_indices, subset_sample_ct, error);
}

void PackedVariantReader::ClearSampleSubset() {
  impl_->ClearSampleSubset();
}

bool PackedVariantReader::ReadVariant(
    uint32_t variant, uint8_t* output, size_t output_byte_ct,
    PackedReadStats* stats, std::string* error) {
  if ((!output) || (output_byte_ct < impl_->packed_byte_ct) ||
      (variant >= impl_->reader.params().variant_ct)) {
    SetError("Invalid conditional-rANS packed variant request.", error);
    return false;
  }
  uint32_t block_idx;
  if (!impl_->reader.FindBlock(variant, &block_idx, error)) {
    return false;
  }
  const uint32_t variant_offset =
      variant - impl_->reader.block_index()[block_idx].first_variant;
  const std::vector<uint32_t> variant_offsets = {variant_offset};
  if (!impl_->DecodeOffsets(block_idx, variant_offsets, stats, error)) {
    return false;
  }
  impl_->CopyDecodedVariant(variant, output);
  if (stats) {
    ++stats->returned_variant_ct;
  }
  return true;
}

bool PackedVariantReader::ReadVariantMaybeSparse(
    uint32_t variant, uint32_t max_difflist_len, uint8_t* output,
    size_t output_byte_ct, SparseHardcallResult* sparse,
    PackedReadStats* stats, std::string* error) {
  if ((!output) || (output_byte_ct < impl_->packed_byte_ct) || (!sparse) ||
      (variant >= impl_->reader.params().variant_ct) ||
      (max_difflist_len > impl_->output_sample_ct)) {
    SetError("Invalid conditional-rANS sparse variant request.", error);
    return false;
  }
  sparse->common_genotype = UINT32_MAX;
  sparse->sample_ids.clear();
  sparse->genotypes.clear();
  bool is_sparse = false;
  if (!impl_->TryReadSparse(
          variant, max_difflist_len, sparse, &is_sparse, stats, error)) {
    return false;
  }
  if (!is_sparse) {
    return ReadVariant(
        variant, output, output_byte_ct, stats, error);
  }
  if (stats) {
    ++stats->returned_variant_ct;
    ++stats->returned_sparse_variant_ct;
  }
  return true;
}

bool PackedVariantReader::ReadVariantPatches(
    uint32_t variant, MultiallelicPatches* patches,
    PackedReadStats* stats, std::string* error) {
  if ((!patches) ||
      (variant >= impl_->reader.params().variant_ct)) {
    SetError("Invalid conditional-rANS multiallelic patch request.",
             error);
    return false;
  }
  uint32_t block_idx;
  if (!impl_->reader.FindBlock(variant, &block_idx, error) ||
      !impl_->LoadBlock(block_idx, stats, error)) {
    return false;
  }
  const ByteSpan record =
      impl_->block_view.record(
          variant - impl_->block_view.first_variant());
  return DecodeMultiallelicPatches(
      record.data, record.size, impl_->reader.params().sample_ct,
      patches, error);
}

bool PackedVariantReader::ReadRange(
    uint32_t first_variant, uint32_t variant_ct, uint8_t* output,
    size_t output_variant_stride, PackedReadStats* stats,
    std::string* error) {
  if ((!variant_ct) ||
      (first_variant > impl_->reader.params().variant_ct) ||
      (variant_ct >
       impl_->reader.params().variant_ct - first_variant)) {
    SetError("Conditional-rANS packed range is out of bounds.", error);
    return false;
  }
  if ((!output) || (output_variant_stride < impl_->packed_byte_ct) ||
      (variant_ct >
       std::numeric_limits<size_t>::max() / output_variant_stride)) {
    SetError("Invalid conditional-rANS packed range output.", error);
    return false;
  }
  uint32_t output_offset = 0;
  while (output_offset != variant_ct) {
    const uint32_t variant = first_variant + output_offset;
    uint32_t block_idx;
    if (!impl_->reader.FindBlock(variant, &block_idx, error)) {
      return false;
    }
    const BlockIndexEntry& entry =
        impl_->reader.block_index()[block_idx];
    const uint32_t block_offset = variant - entry.first_variant;
    const uint32_t chunk_variant_ct =
        std::min(variant_ct - output_offset,
                 entry.variant_ct - block_offset);
    std::vector<uint32_t> variant_offsets(chunk_variant_ct);
    for (uint32_t chunk_offset = 0;
         chunk_offset != chunk_variant_ct; ++chunk_offset) {
      variant_offsets[chunk_offset] = block_offset + chunk_offset;
    }
    if (!impl_->DecodeOffsets(block_idx, variant_offsets, stats, error)) {
      return false;
    }
    for (uint32_t chunk_offset = 0;
         chunk_offset != chunk_variant_ct; ++chunk_offset) {
      impl_->CopyDecodedVariant(
          variant + chunk_offset,
          output +
              static_cast<size_t>(output_offset + chunk_offset) *
                  output_variant_stride);
    }
    output_offset += chunk_variant_ct;
  }
  if (stats) {
    stats->returned_variant_ct += variant_ct;
  }
  return true;
}

bool PackedVariantReader::ReadList(
    const uint32_t* variants, uint32_t variant_ct, uint8_t* output,
    size_t output_variant_stride, PackedReadStats* stats,
    std::string* error) {
  return impl_->ReadList(variants, variant_ct, output,
                         output_variant_stride, stats, error);
}

}  // namespace pgen_rans
