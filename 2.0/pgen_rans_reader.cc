// SPDX-License-Identifier: GPL-3.0-or-later

#include "pgen_rans_reader.h"

#include <algorithm>
#include <chrono>
#include <cstring>
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
    block_view = {};
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
    cached_block_idx = UINT32_MAX;
    projected_block.clear();
    return true;
  }

  bool DecodeBlock(uint32_t block_idx, PackedReadStats* stats,
                   std::string* error) {
    if (block_idx == cached_block_idx) {
      return true;
    }
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
    decoded_block.resize(
        static_cast<size_t>(block_view.variant_ct()) * word_stride);
    const auto decode_start = std::chrono::steady_clock::now();
    if (!decoder->Decode(block_view, reader.params().sample_ct, params,
                         decoded_block.data(), decoded_block.size(),
                         error)) {
      return false;
    }
    const double decode_seconds =
        std::chrono::duration<double>(
            std::chrono::steady_clock::now() - decode_start)
            .count();
    double projection_seconds = 0.0;
    if (!sample_subset.empty()) {
      projected_block.resize(
          static_cast<size_t>(block_view.variant_ct()) * packed_byte_ct);
      const auto projection_start = std::chrono::steady_clock::now();
      if (!decoder->ProjectSampleSubset(
              decoded_block.data(), block_view.variant_ct(),
              reader.params().sample_ct, sample_subset.data(),
              output_sample_ct, projected_block.data(), packed_byte_ct,
              error)) {
        return false;
      }
      projection_seconds =
          std::chrono::duration<double>(
              std::chrono::steady_clock::now() - projection_start)
              .count();
    } else {
      projected_block.clear();
    }
    cached_block_idx = block_idx;
    if (stats) {
      ++stats->block_read_ct;
      stats->block_byte_ct += reader.block_index()[block_idx].byte_ct;
      stats->decoded_variant_ct += block_view.variant_ct();
      stats->block_read_seconds += read_seconds;
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
    uint32_t active_block_idx = UINT32_MAX;
    for (const auto& request : requests) {
      uint32_t block_idx;
      if (!reader.FindBlock(request.first, &block_idx, error)) {
        return false;
      }
      if (block_idx != active_block_idx) {
        if (!DecodeBlock(block_idx, stats, error)) {
          return false;
        }
        active_block_idx = block_idx;
      }
      uint8_t* destination =
          output +
          static_cast<size_t>(request.second) * output_variant_stride;
      CopyDecodedVariant(request.first, destination);
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
  std::vector<uint8_t> projected_block;
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
  if (!impl_->reader.FindBlock(variant, &block_idx, error) ||
      !impl_->DecodeBlock(block_idx, stats, error)) {
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
      !impl_->DecodeBlock(block_idx, stats, error)) {
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
  for (uint32_t offset = 0; offset != variant_ct; ++offset) {
    const uint32_t variant = first_variant + offset;
    uint32_t block_idx;
    if (!impl_->reader.FindBlock(variant, &block_idx, error) ||
        !impl_->DecodeBlock(block_idx, stats, error)) {
      return false;
    }
    impl_->CopyDecodedVariant(
        variant, output + static_cast<size_t>(offset) *
                              output_variant_stride);
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
