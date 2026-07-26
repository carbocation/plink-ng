// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef PGEN_RANS_CPU_H_
#define PGEN_RANS_CPU_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

#include "pgen_rans_codec.h"
#include "pgen_rans_container.h"

namespace pgen_rans {

class CpuBlockDecoder {
 public:
  explicit CpuBlockDecoder(uint32_t thread_ct);
  ~CpuBlockDecoder();
  CpuBlockDecoder(const CpuBlockDecoder&) = delete;
  CpuBlockDecoder& operator=(const CpuBlockDecoder&) = delete;

  bool Decode(const EncodedBlockView& block, uint32_t sample_ct,
              const CodecParams& params, uint64_t* output,
              size_t output_word_ct, std::string* error);

  // Decodes only the requested variant offsets and the scheduled anchors
  // they reference.  decoded_flags must have one byte per block variant and
  // is preserved across calls so adjacent sparse reads can extend the cache.
  bool DecodeSelected(const EncodedBlockView& block, uint32_t sample_ct,
                      const CodecParams& params,
                      const uint32_t* variant_offsets,
                      uint32_t variant_offset_ct, uint64_t* output,
                      size_t output_word_ct, uint8_t* decoded_flags,
                      size_t decoded_flag_ct, uint32_t* decoded_variant_ct,
                      std::string* error);

  bool ProjectSampleSubset(const uint64_t* input, uint32_t variant_ct,
                           uint32_t raw_sample_ct,
                           const uint32_t* sample_indices,
                           uint32_t subset_sample_ct, uint8_t* output,
                           size_t output_variant_stride,
                           std::string* error);

  bool ProjectSampleSubsetSelected(
      const uint64_t* input, uint32_t variant_ct,
      uint32_t raw_sample_ct, const uint32_t* variant_offsets,
      uint32_t variant_offset_ct, const uint32_t* sample_indices,
      uint32_t subset_sample_ct, uint8_t* output,
      size_t output_variant_stride, std::string* error);

  uint32_t thread_ct() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace pgen_rans

#endif  // PGEN_RANS_CPU_H_
