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

  uint32_t thread_ct() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace pgen_rans

#endif  // PGEN_RANS_CPU_H_
