// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef PGEN_RANS_READER_H_
#define PGEN_RANS_READER_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "pgen_rans_codec.h"

namespace pgen_rans {

struct PackedReadStats {
  uint64_t block_read_ct = 0;
  uint64_t block_byte_ct = 0;
  uint64_t decoded_variant_ct = 0;
  uint64_t returned_variant_ct = 0;
  uint64_t returned_sparse_variant_ct = 0;
  double block_read_seconds = 0.0;
  double decode_seconds = 0.0;
  double projection_seconds = 0.0;
};

// A PGEN-compatible hardcall difference list.  Sample IDs refer to the
// reader's current output sample order (the stored order unless
// SetSampleSubset() is active), and genotypes use the standard 0/1/2/3 PGEN
// hardcall codes.  Callers may reuse one instance across reads to retain vector
// capacity.
struct SparseHardcallResult {
  uint32_t common_genotype = UINT32_MAX;
  std::vector<uint32_t> sample_ids;
  std::vector<uint8_t> genotypes;
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
  uint32_t max_allele_ct() const;
  bool all_nonref() const;
  bool has_mixed_nonref_flags() const;
  bool variant_is_nonref(uint32_t variant) const;
  size_t packed_variant_byte_ct() const;

  // Indices are zero-based positions in the stored sample order and must be
  // strictly increasing. Packed decoding still visits every stored sample;
  // direct sparse reads instead project only their exception IDs.
  bool SetSampleSubset(const uint32_t* sample_indices,
                       uint32_t subset_sample_ct, std::string* error);
  void ClearSampleSubset();

  bool ReadVariant(uint32_t variant, uint8_t* output,
                   size_t output_byte_ct, PackedReadStats* stats,
                   std::string* error);
  // Returns a difference list when the requested record and any referenced
  // anchors have an exact sparse representation no longer than
  // max_difflist_len.  Otherwise this follows ReadVariant() and writes packed
  // hardcalls to output.  output is required because dense fallback remains a
  // normal successful result; sparse->common_genotype == UINT32_MAX denotes
  // dense fallback.
  bool ReadVariantMaybeSparse(uint32_t variant, uint32_t max_difflist_len,
                              uint8_t* output, size_t output_byte_ct,
                              SparseHardcallResult* sparse,
                              PackedReadStats* stats, std::string* error);
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
