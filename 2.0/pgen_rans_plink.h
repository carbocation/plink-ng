// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef PGEN_RANS_PLINK_H_
#define PGEN_RANS_PLINK_H_

#include <cstdint>
#include <string>

#include "plink2_common.h"
#include "pgen_rans_reader.h"

namespace pgen_rans {

plink2::PglErr PrintPgenInfo(const char* pgenname);

plink2::PglErr MakePgen(
    const uintptr_t* sample_include, const uintptr_t* variant_include,
    const plink2::ChrInfo* cip, const uint32_t* variant_bps,
    const uintptr_t* allele_idx_offsets, uint32_t raw_sample_ct,
    uint32_t sample_ct, uint32_t raw_variant_ct, uint32_t variant_ct,
    uint32_t max_thread_ct, uint32_t discard_phase,
    uint32_t discard_dosage, plink2::PgenFileInfo* pgfip,
    plink2::PgenReader* pgrp, char* outname, char* outname_end);

struct PlinkLoadOptions {
  const char* pgenname = nullptr;
  const char* pvarname = nullptr;
  uintptr_t* allele_idx_offsets = nullptr;
  uint32_t raw_variant_ct = 0;
  uint32_t raw_sample_ct = 0;
  uint32_t max_allele_ct = 2;
  uint32_t thread_ct = 1;
  bool unsupported_command = false;
  bool unsupported_export = false;
  bool phased_ld = false;
  bool unsupported_multiallelic_pca = false;
  bool unsupported_multiallelic_scan = false;
  bool validate = false;
  bool print_info = false;
};

// Adapts a conditional-rANS PGEN hardcall stream to the narrow
// packed-hardcall seam in PgenReader.  PLINK's existing sample-subset, allele,
// score, and export code remains responsible for interpretation above this
// boundary.
class PlinkRansAdapter {
 public:
  PlinkRansAdapter();
  ~PlinkRansAdapter();
  PlinkRansAdapter(const PlinkRansAdapter&) = delete;
  PlinkRansAdapter& operator=(const PlinkRansAdapter&) = delete;

  bool Open(const std::string& path, uint32_t thread_ct,
            std::string* error);
  bool Validate(std::string* error);
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
  static plink2::PglErr GetPackedBatch(
      void* context, const uintptr_t* variant_include,
      uint32_t variant_uidx_start, uint32_t variant_uidx_end,
      uint32_t load_variant_ct, unsigned char* output,
      uint32_t raw_variant_stride);
  static plink2::PglErr GetM(
      void* context, const uintptr_t* sample_include,
      const uint32_t* sample_include_cumulative_popcounts,
      uint32_t sample_ct, uint32_t vidx, plink2::PgenVariant* pgv);
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
  plink2::PglErr ReadMultiallelic(
      const uintptr_t* sample_include,
      const uint32_t* sample_include_cumulative_popcounts,
      uint32_t sample_ct, uint32_t vidx, plink2::PgenVariant* pgv);
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

plink2::PglErr LoadPgen(
    const PlinkLoadOptions& options, uintptr_t** nonref_flags_ptr,
    plink2::PgenFileInfo* pgfi, plink2::PgenReader* pgr,
    uintptr_t* pgr_alloc_cacheline_ct_ptr,
    PlinkRansAdapter** adapter_ptr);

}  // namespace pgen_rans

#endif  // PGEN_RANS_PLINK_H_
