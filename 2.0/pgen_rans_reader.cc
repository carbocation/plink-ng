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

namespace pgen_rans {
namespace {

void SetError(const std::string& message, std::string* error) {
  if (error) {
    *error = message;
  }
}

}  // namespace

struct PackedVariantReader::Impl {
  struct CachedBlock {
    size_t ByteCt() const {
      return storage.size() +
             decoded.size() * sizeof(uint64_t) +
             decoded_flags.size() + projected.size() +
             projected_flags.size();
    }

    uint32_t block_idx = UINT32_MAX;
    std::vector<uint8_t> storage;
    EncodedBlockView view;
    std::vector<uint64_t> decoded;
    std::vector<uint8_t> decoded_flags;
    std::vector<uint8_t> projected;
    std::vector<uint8_t> projected_flags;
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
    cached_block_idx = block_idx;
    if (stats) {
      ++stats->block_read_ct;
      stats->block_byte_ct += reader.block_index()[block_idx].byte_ct;
      stats->block_read_seconds += read_seconds;
    }
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
