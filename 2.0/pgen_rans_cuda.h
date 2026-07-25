// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef PGEN_RANS_CUDA_H_
#define PGEN_RANS_CUDA_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "pgen_rans_codec.h"
#include "pgen_rans_container.h"

namespace pgen_rans {

struct CudaDecodeTimings {
  float upload_milliseconds = 0.0F;
  float anchor_milliseconds = 0.0F;
  float target_milliseconds = 0.0F;
  uint64_t genotype_call_ct = 0;
};

class CudaBlockDecoder {
 public:
  CudaBlockDecoder();
  ~CudaBlockDecoder();
  CudaBlockDecoder(const CudaBlockDecoder&) = delete;
  CudaBlockDecoder& operator=(const CudaBlockDecoder&) = delete;

  bool available() const;

  // Blocks are decoded consecutively into device_output in the order given.
  // cuda_stream is a cudaStream_t passed as an opaque pointer; null selects
  // the default stream.
  bool Decode(const std::vector<const EncodedBlockView*>& blocks,
              uint32_t sample_ct, const CodecParams& params,
              void* device_output, size_t output_word_ct,
              void* cuda_stream, CudaDecodeTimings* timings,
              std::string* error);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace pgen_rans

#endif  // PGEN_RANS_CUDA_H_
