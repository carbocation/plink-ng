// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef PGEN_RANS_PGENLIB_H_
#define PGEN_RANS_PGENLIB_H_

#include <cstdint>
#include <string>
#include <vector>

#include "include/pgenlib_read.h"
#include "pgen_rans_reader.h"

namespace pgen_rans {

// Adapts a conditional-rANS hardcall stream to pgenlib's alternate-backend
// seam.  The backend object owns the callback context and must outlive every
// PgenReader on which Install() has been called.  One backend instance can be
// installed on one reader at a time and must not be called concurrently;
// thread_ct controls parallel work inside a decode, not concurrent access.
class PgenlibRansBackend {
 public:
  PgenlibRansBackend();
  ~PgenlibRansBackend();
  PgenlibRansBackend(const PgenlibRansBackend&) = delete;
  PgenlibRansBackend& operator=(const PgenlibRansBackend&) = delete;

  bool Open(const std::string& path, uint32_t thread_ct,
            std::string* error);
  bool Validate(std::string* error);
  void Close();
  // Requires preinitialized destinations whose sample and variant counts
  // agree with the open container.
  bool Install(plink2::PgenFileInfo* pgfi, plink2::PgenReader* pgr,
               std::string* error);

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
  static plink2::PglErr GetDifflistOrGenovec(
      void* context, const uintptr_t* sample_include,
      const uint32_t* sample_include_cumulative_popcounts,
      uint32_t sample_ct, uint32_t vidx, uint32_t max_difflist_len,
      uintptr_t* genovec, uint32_t* difflist_common_geno_ptr,
      uintptr_t* main_raregeno, uint32_t* difflist_sample_ids,
      uint32_t* difflist_len_ptr);
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
  plink2::PglErr ReadDifflistOrGenovec(
      const uintptr_t* sample_include,
      const uint32_t* sample_include_cumulative_popcounts,
      uint32_t sample_ct, uint32_t vidx, uint32_t max_difflist_len,
      uintptr_t* genovec, uint32_t* difflist_common_geno_ptr,
      uintptr_t* main_raregeno, uint32_t* difflist_sample_ids,
      uint32_t* difflist_len_ptr);
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
  SparseHardcallResult sparse_result_;
  uint32_t raw_word_ct_ = 0;
  plink2::PgenFileInfo* installed_pgfi_ = nullptr;
  plink2::PgenReader* installed_pgr_ = nullptr;
  bool is_open_ = false;
  std::string last_error_;
};

// Compatibility name for the PLINK application layer.  New downstream code
// should use PgenlibRansBackend or UnifiedPgenReader.
using PlinkRansAdapter = PgenlibRansBackend;

enum class PgenStorageMode : uint8_t {
  kStandard,
  kConditionalRans,
};

struct UnifiedPgenOpenOptions {
  uint32_t thread_ct = 1;
  // Required for multiallelic conditional-rANS files.  When supplied, this
  // variant_ct + 1 entry PVAR-derived array is validated and copied.
  const uintptr_t* allele_idx_offsets = nullptr;
};

// Owns a pgenlib reader and all of its initialization state.  Open() selects
// the ordinary pgenlib path or the conditional-rANS backend from the file
// header.  Backend-aware hardcall, allele-count, unphased, and sparse-dosage
// entry points use the returned PgenReader identically in either case.
class UnifiedPgenReader {
 public:
  UnifiedPgenReader();
  ~UnifiedPgenReader();
  UnifiedPgenReader(const UnifiedPgenReader&) = delete;
  UnifiedPgenReader& operator=(const UnifiedPgenReader&) = delete;

  bool Open(const std::string& path,
            const UnifiedPgenOpenOptions& options,
            std::string* error);
  bool Open(const std::string& path, std::string* error) {
    return Open(path, UnifiedPgenOpenOptions(), error);
  }
  void Close();

  bool is_open() const { return is_open_; }
  PgenStorageMode storage_mode() const { return storage_mode_; }
  uint32_t sample_ct() const { return pgfi_.raw_sample_ct; }
  uint32_t variant_ct() const { return pgfi_.raw_variant_ct; }
  uint32_t max_allele_ct() const { return pgfi_.max_allele_ct; }
  plink2::PgenFileInfo* file_info() { return &pgfi_; }
  const plink2::PgenFileInfo* file_info() const { return &pgfi_; }
  plink2::PgenReader* pgen_reader() { return &pgr_; }
  const plink2::PgenReader* pgen_reader() const { return &pgr_; }

 private:
  bool OpenStandard(const std::string& path,
                    plink2::PgenHeaderCtrl header_ctrl,
                    uintptr_t pgfi_alloc_cacheline_ct,
                    std::string* error);
  bool OpenRans(const std::string& path,
                const UnifiedPgenOpenOptions& options,
                std::string* error);

  plink2::PgenFileInfo pgfi_{};
  plink2::PgenReader pgr_{};
  unsigned char* pgfi_alloc_ = nullptr;
  unsigned char* pgr_alloc_ = nullptr;
  std::vector<uintptr_t> allele_idx_offsets_;
  std::vector<uintptr_t> nonref_flags_;
  PgenlibRansBackend rans_backend_;
  PgenStorageMode storage_mode_ = PgenStorageMode::kStandard;
  bool is_open_ = false;
};

}  // namespace pgen_rans

#endif  // PGEN_RANS_PGENLIB_H_
