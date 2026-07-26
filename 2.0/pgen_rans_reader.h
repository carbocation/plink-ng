// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef PGEN_RANS_READER_H_
#define PGEN_RANS_READER_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

#include "pgen_rans_codec.h"

namespace pgen_rans {

struct PackedReadStats {
  uint64_t block_read_ct = 0;
  uint64_t block_byte_ct = 0;
  uint64_t decoded_variant_ct = 0;
  uint64_t returned_variant_ct = 0;
  double block_read_seconds = 0.0;
  double decode_seconds = 0.0;
  double projection_seconds = 0.0;
};

class PackedVariantReader {
 public:
  PackedVariantReader();
  ~PackedVariantReader();
  PackedVariantReader(const PackedVariantReader&) = delete;
  PackedVariantReader& operator=(const PackedVariantReader&) = delete;

  bool Open(const std::string& path, uint32_t thread_ct,
            std::string* error);
  void Close();

  uint32_t raw_sample_ct() const;
  uint32_t sample_ct() const;
  uint32_t variant_ct() const;
  size_t packed_variant_byte_ct() const;

  // Indices are zero-based positions in the stored sample order and must be
  // strictly increasing. Decoding still visits every stored genotype; this
  // controls the packed projection copied to callers.
  bool SetSampleSubset(const uint32_t* sample_indices,
                       uint32_t subset_sample_ct, std::string* error);
  void ClearSampleSubset();

  bool ReadVariant(uint32_t variant, uint8_t* output,
                   size_t output_byte_ct, PackedReadStats* stats,
                   std::string* error);
  // Patch sample IDs always refer to the stored (raw) sample order, even
  // when SetSampleSubset() is active.
  bool ReadVariantPatches(uint32_t variant,
                          MultiallelicPatches* patches,
                          PackedReadStats* stats, std::string* error);
  bool ReadRange(uint32_t first_variant, uint32_t variant_ct,
                 uint8_t* output, size_t output_variant_stride,
                 PackedReadStats* stats, std::string* error);
  bool ReadList(const uint32_t* variants, uint32_t variant_ct,
                uint8_t* output, size_t output_variant_stride,
                PackedReadStats* stats, std::string* error);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace pgen_rans

#endif  // PGEN_RANS_READER_H_
