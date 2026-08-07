// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef PGEN_RANS_CONCAT_H_
#define PGEN_RANS_CONCAT_H_

#include <cstddef>
#include <cstdint>

namespace pgen_rans {

constexpr uint8_t kPgenRansStorageMode = 0x80;

struct ContainerConcatStats {
  uint32_t input_file_ct = 0;
  uint32_t sample_ct = 0;
  uint32_t variant_ct = 0;
  uint32_t block_ct = 0;
  uint64_t copied_block_byte_ct = 0;
  uint64_t output_byte_ct = 0;
};

struct ContainerSummary {
  uint32_t sample_ct = 0;
  uint32_t variant_ct = 0;
  uint32_t max_allele_ct = 0;
  uint32_t all_nonref = 0;
  uint32_t nonref_flag_byte_ct = 0;
};

bool ReadContainerSummaryRaw(const char* input_path,
                             ContainerSummary* summary,
                             unsigned char* nonref_flags,
                             size_t nonref_flags_capacity,
                             char* error_buf,
                             size_t error_buf_size);

// STL-free bridge for PLINK's bigstack-based merge path.
bool ConcatenateContainersRaw(const char* const* input_paths,
                              uint32_t input_path_ct,
                              const char* output_path,
                              ContainerConcatStats* stats,
                              char* error_buf,
                              size_t error_buf_size);

}  // namespace pgen_rans

#endif  // PGEN_RANS_CONCAT_H_
