// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef PGEN_RANS_PLINK_H_
#define PGEN_RANS_PLINK_H_

#include <cstdint>
#include <string>

#include "include/pgenlib_read.h"
#include "pgen_rans_reader.h"

namespace pgen_rans {

// Adapts a PGR hardcall stream to the narrow packed-hardcall seam in
// PgenReader.  PLINK's existing sample-subset, allele, score, and export code
// remains responsible for interpretation above this boundary.
class PlinkPgrAdapter {
 public:
  PlinkPgrAdapter();
  ~PlinkPgrAdapter();
  PlinkPgrAdapter(const PlinkPgrAdapter&) = delete;
  PlinkPgrAdapter& operator=(const PlinkPgrAdapter&) = delete;

  bool Open(const std::string& path, uint32_t thread_ct,
            std::string* error);
  void Close();
  void Install(plink2::PgenFileInfo* pgfi, plink2::PgenReader* pgr);

  uint32_t sample_ct() const;
  uint32_t variant_ct() const;
  uint32_t max_allele_ct() const;
  bool all_nonref() const;
  bool has_mixed_nonref_flags() const;
  bool variant_is_nonref(uint32_t variant) const;
  const std::string& last_error() const;

 private:
  static plink2::PglErr Get(
      void* context, const uintptr_t* sample_include,
      const uint32_t* sample_include_cumulative_popcounts,
      uint32_t sample_ct, uint32_t vidx, uintptr_t* genovec);
  static plink2::PglErr GetAllele(
      void* context, const uintptr_t* sample_include,
      const uint32_t* sample_include_cumulative_popcounts,
      uint32_t sample_ct, uint32_t vidx, uint32_t allele_idx,
      uintptr_t* allele_countvec);
  static plink2::PglErr GetCounts(
      void* context, const uintptr_t* sample_include,
      const uint32_t* sample_include_cumulative_popcounts,
      uint32_t sample_ct, uint32_t vidx, uint32_t* genocounts);
  static plink2::PglErr GetPacked(
      void* context, uint32_t vidx, unsigned char* packed_genotypes,
      uint32_t packed_byte_ct);
  static plink2::PglErr GetRaw(
      void* context, uint32_t vidx,
      plink2::PgenGlobalFlags read_gflags,
      uintptr_t** loadbuf_iter_ptr,
      unsigned char* loaded_vrtype_ptr);

  plink2::PglErr ReadBase(
      const uintptr_t* sample_include, uint32_t sample_ct,
      uint32_t vidx, uintptr_t* genovec);
  plink2::PglErr ReadAllele(
      const uintptr_t* sample_include, uint32_t sample_ct,
      uint32_t vidx, uint32_t allele_idx, uintptr_t* allele_countvec);
  bool ReadRaw(uint32_t vidx);
  plink2::PglErr ReadRawRecord(
      uint32_t vidx, plink2::PgenGlobalFlags read_gflags,
      uintptr_t** loadbuf_iter_ptr,
      unsigned char* loaded_vrtype_ptr);
  void CopySubset(const uintptr_t* sample_include, uint32_t sample_ct,
                  uintptr_t* destination) const;

  PackedVariantReader reader_;
  plink2::PgrHardcallBackend backend_;
  uintptr_t* raw_genovec_ = nullptr;
  uintptr_t* count_genovec_ = nullptr;
  uint32_t raw_word_ct_ = 0;
  std::string last_error_;
};

}  // namespace pgen_rans

#endif  // PGEN_RANS_PLINK_H_
