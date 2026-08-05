// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef PGEN_RANS_PLINK_H_
#define PGEN_RANS_PLINK_H_

#include <cstdint>
#include <string>

#include "plink2_common.h"
#include "pgen_rans_pgenlib.h"

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

plink2::PglErr LoadPgen(
    const PlinkLoadOptions& options, uintptr_t** nonref_flags_ptr,
    plink2::PgenFileInfo* pgfi, plink2::PgenReader* pgr,
    uintptr_t* pgr_alloc_cacheline_ct_ptr,
    PlinkRansAdapter** adapter_ptr);

}  // namespace pgen_rans

#endif  // PGEN_RANS_PLINK_H_
