// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef PGEN_RANS_ENCODE_H_
#define PGEN_RANS_ENCODE_H_

#include <cstdint>
#include <string>

#include "include/pgenlib_read.h"

namespace pgen_rans {

struct VariantMetadata {
  uint32_t chrom_code = 0;
  uint16_t allele_ct = 2;
  uint64_t bp = 0;
};

struct EncodeParams {
  uint32_t block_variant_ct = 128;
  uint32_t anchor_ct = 32;
  uint32_t two_ref_shortlist = 4;
  uint64_t max_anchor_bp = 1000000;
  uint32_t restart_variant_ct = 64;
  uint32_t rans_state_ct = 32;
  uint32_t rans_scale_bits = 12;
  uint32_t thread_ct = 1;
};

struct EncodeInput {
  uint32_t raw_sample_ct = 0;
  uint32_t sample_ct = 0;
  uint32_t raw_variant_ct = 0;
  uint32_t variant_ct = 0;
  const uintptr_t* sample_include = nullptr;
  // Null denotes the first variant_ct records in their original order.
  const uint32_t* variant_uidxs = nullptr;
  const VariantMetadata* variant_metadata = nullptr;
  plink2::PgenFileInfo* pgfi = nullptr;
  plink2::PgenReader* pgen_reader = nullptr;
  bool discard_phase = false;
  bool discard_dosage = false;
};

struct EncodeStats {
  uint32_t variant_ct = 0;
  uint32_t block_ct = 0;
  uint64_t pgen_payload_bytes = 0;
  uint64_t output_bytes = 0;
  uint64_t marginal_ct = 0;
  uint64_t one_reference_ct = 0;
  uint64_t two_reference_ct = 0;
  uint64_t anchor_ct = 0;
  uint64_t multiallelic_ct = 0;
  uint64_t patch_01_ct = 0;
  uint64_t patch_10_ct = 0;
  uint64_t multiallelic_patch_bytes = 0;
  double elapsed_seconds = 0.0;
};

plink2::PglErr EncodePgenRans(const std::string& output_path,
                              const EncodeInput& input,
                              const EncodeParams& params, EncodeStats* stats,
                              std::string* error);

}  // namespace pgen_rans

#endif  // PGEN_RANS_ENCODE_H_
