// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef PGEN_RANS_SIZE_H_
#define PGEN_RANS_SIZE_H_

#include <cstdint>

namespace pgen_rans {

// Conditional-rANS container serialized sizes.  Keep these synchronized
// with SerializeHeader(), SerializeBlockIndex(), and
// ContainerWriter::WriteBlock().
constexpr uint64_t kContainerHeaderByteCt = 96;
constexpr uint64_t kContainerBlockIndexByteCt = 32;
constexpr uint64_t kContainerBlockHeaderByteCt = 16;
constexpr uint64_t kContainerRecordLengthByteCt = 3;
constexpr uint64_t kContainerRestartOffsetByteCt = 4;

inline uint64_t ContainerGlobalOverheadByteCt(
    uint64_t block_ct, uint64_t metadata_byte_ct) {
  return kContainerHeaderByteCt +
         kContainerBlockIndexByteCt * block_ct + metadata_byte_ct;
}

inline uint64_t ContainerBlockOverheadByteCt(
    uint32_t variant_ct, uint32_t restart_variant_ct) {
  if ((!variant_ct) || (!restart_variant_ct)) {
    return 0;
  }
  const uint64_t restart_offset_ct =
      (static_cast<uint64_t>(variant_ct) + restart_variant_ct - 1) /
          restart_variant_ct -
      1;
  return kContainerBlockHeaderByteCt +
         kContainerRecordLengthByteCt * variant_ct +
         kContainerRestartOffsetByteCt * restart_offset_ct;
}

}  // namespace pgen_rans

#endif  // PGEN_RANS_SIZE_H_
