// SPDX-License-Identifier: GPL-3.0-or-later

#include "pgen_rans_container.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstddef>
#include <cstring>
#include <limits>
#include <sys/stat.h>

#ifdef _WIN32
#include <windows.h>
#endif

#if defined(__x86_64__) && \
    (defined(__GNUC__) || defined(__clang__))
#include <nmmintrin.h>
#define PGEN_RANS_X86_CRC32C_DISPATCH 1
#else
#define PGEN_RANS_X86_CRC32C_DISPATCH 0
#endif

#if defined(__aarch64__) && defined(__ARM_FEATURE_CRC32)
#include <arm_acle.h>
#define PGEN_RANS_ARM_CRC32C 1
#else
#define PGEN_RANS_ARM_CRC32C 0
#endif

namespace pgen_rans {
namespace {

constexpr std::array<uint8_t, 3> kFileMagic = {
    0x6c, 0x1b, kPgenRansStorageMode};
constexpr uint32_t kFileHeaderByteCt = 96;
constexpr uint32_t kBlockIndexByteCt = 32;
constexpr uint32_t kBlockHeaderByteCt = 16;
constexpr uint32_t kBlockMagic = 0x314b4c42U;
constexpr uint32_t kMaximumRecordByteCt = 0xffffffU;
constexpr uint32_t kMaximumVariantCt = 0x7ffffffdU;
constexpr uint32_t kContainerFlagNonrefBitmap = 1U << 0;
constexpr uint32_t kContainerFlagAllNonref = 1U << 1;
constexpr uint32_t kContainerKnownFlags =
    kContainerFlagNonrefBitmap | kContainerFlagAllNonref;
#if !PGEN_RANS_ARM_CRC32C
constexpr uint32_t kCrc32cPolynomial = 0x82f63b78U;
#endif

void SetError(const std::string& message, std::string* error) {
  if (error) {
    *error = message;
  }
}

void RemovePartialOutput(const std::string& path, std::string* error) {
  if (std::remove(path.c_str()) && (errno != ENOENT) && error) {
    *error += " Failed to remove partial output " + path + ": " +
              strerror(errno) + ".";
  }
}

bool PathExists(const std::string& path) {
#ifdef _WIN32
  return GetFileAttributesA(path.c_str()) != INVALID_FILE_ATTRIBUTES;
#else
  struct stat path_stat;
  return !lstat(path.c_str(), &path_stat);
#endif
}

bool SameParams(const ContainerParams& lhs, const ContainerParams& rhs) {
  return (lhs.sample_ct == rhs.sample_ct) &&
         (lhs.variant_ct == rhs.variant_ct) &&
         (lhs.block_variant_ct == rhs.block_variant_ct) &&
         (lhs.anchor_ct == rhs.anchor_ct) &&
         (lhs.state_ct == rhs.state_ct) &&
         (lhs.scale_bits == rhs.scale_bits) &&
         (lhs.restart_variant_ct == rhs.restart_variant_ct) &&
         (lhs.block_ct == rhs.block_ct) &&
         (lhs.max_allele_ct == rhs.max_allele_ct);
}

bool SameBlockIndex(const std::vector<BlockIndexEntry>& lhs,
                    const std::vector<BlockIndexEntry>& rhs) {
  if (lhs.size() != rhs.size()) {
    return false;
  }
  for (size_t block_idx = 0; block_idx != lhs.size(); ++block_idx) {
    const BlockIndexEntry& left = lhs[block_idx];
    const BlockIndexEntry& right = rhs[block_idx];
    if ((left.first_variant != right.first_variant) ||
        (left.variant_ct != right.variant_ct) ||
        (left.file_offset != right.file_offset) ||
        (left.byte_ct != right.byte_ct) ||
        (left.checksum != right.checksum)) {
      return false;
    }
  }
  return true;
}

#if !PGEN_RANS_ARM_CRC32C
const std::array<uint32_t, 256>& Crc32cTable() {
  static const std::array<uint32_t, 256> table = []() {
    std::array<uint32_t, 256> result = {};
    for (uint32_t value = 0; value != result.size(); ++value) {
      uint32_t remainder = value;
      for (uint32_t bit = 0; bit != 8; ++bit) {
        remainder = (remainder >> 1) ^
                    ((0U - (remainder & 1U)) & kCrc32cPolynomial);
      }
      result[value] = remainder;
    }
    return result;
  }();
  return table;
}

uint32_t Crc32cPortable(const uint8_t* data, size_t byte_ct) {
  const std::array<uint32_t, 256>& table = Crc32cTable();
  uint32_t checksum = UINT32_MAX;
  for (size_t byte_idx = 0; byte_idx != byte_ct; ++byte_idx) {
    checksum =
        table[(checksum ^ data[byte_idx]) & 0xffU] ^ (checksum >> 8);
  }
  return ~checksum;
}
#endif

#if PGEN_RANS_X86_CRC32C_DISPATCH
__attribute__((target("sse4.2")))
uint32_t Crc32cX86(const uint8_t* data, size_t byte_ct) {
  uint64_t checksum = UINT32_MAX;
  while (byte_ct >= sizeof(uint64_t)) {
    uint64_t word;
    memcpy(&word, data, sizeof(word));
    checksum = _mm_crc32_u64(checksum, word);
    data += sizeof(word);
    byte_ct -= sizeof(word);
  }
  uint32_t tail_checksum = static_cast<uint32_t>(checksum);
  while (byte_ct) {
    tail_checksum = _mm_crc32_u8(tail_checksum, *data++);
    --byte_ct;
  }
  return ~tail_checksum;
}
#endif

#if PGEN_RANS_ARM_CRC32C
uint32_t Crc32cArm(const uint8_t* data, size_t byte_ct) {
  uint32_t checksum = UINT32_MAX;
  while (byte_ct >= sizeof(uint64_t)) {
    uint64_t word;
    memcpy(&word, data, sizeof(word));
    checksum = __crc32cd(checksum, word);
    data += sizeof(word);
    byte_ct -= sizeof(word);
  }
  while (byte_ct) {
    checksum = __crc32cb(checksum, *data++);
    --byte_ct;
  }
  return ~checksum;
}
#endif

uint32_t Crc32c(const uint8_t* data, size_t byte_ct) {
#if PGEN_RANS_X86_CRC32C_DISPATCH
  static const bool has_sse42 = []() {
    __builtin_cpu_init();
    return __builtin_cpu_supports("sse4.2");
  }();
  if (has_sse42) {
    return Crc32cX86(data, byte_ct);
  }
#endif
#if PGEN_RANS_ARM_CRC32C
  return Crc32cArm(data, byte_ct);
#else
  return Crc32cPortable(data, byte_ct);
#endif
}

void AppendU16(uint16_t value, std::vector<uint8_t>* output) {
  output->push_back(static_cast<uint8_t>(value));
  output->push_back(static_cast<uint8_t>(value >> 8));
}

void AppendU24(uint32_t value, std::vector<uint8_t>* output) {
  output->push_back(static_cast<uint8_t>(value));
  output->push_back(static_cast<uint8_t>(value >> 8));
  output->push_back(static_cast<uint8_t>(value >> 16));
}

void AppendU32(uint32_t value, std::vector<uint8_t>* output) {
  output->push_back(static_cast<uint8_t>(value));
  output->push_back(static_cast<uint8_t>(value >> 8));
  output->push_back(static_cast<uint8_t>(value >> 16));
  output->push_back(static_cast<uint8_t>(value >> 24));
}

void AppendU64(uint64_t value, std::vector<uint8_t>* output) {
  for (uint32_t byte_idx = 0; byte_idx != 8; ++byte_idx) {
    output->push_back(static_cast<uint8_t>(value >> (8 * byte_idx)));
  }
}

bool ReadU16(const uint8_t* input, size_t input_size, size_t* offset,
             uint16_t* value) {
  if (*offset + 2 > input_size) {
    return false;
  }
  *value = static_cast<uint16_t>(
      static_cast<uint16_t>(input[*offset]) |
      (static_cast<uint16_t>(input[*offset + 1]) << 8));
  *offset += 2;
  return true;
}

bool ReadU24(const uint8_t* input, size_t input_size, size_t* offset,
             uint32_t* value) {
  if (*offset + 3 > input_size) {
    return false;
  }
  *value = static_cast<uint32_t>(input[*offset]) |
           (static_cast<uint32_t>(input[*offset + 1]) << 8) |
           (static_cast<uint32_t>(input[*offset + 2]) << 16);
  *offset += 3;
  return true;
}

bool ReadU32(const uint8_t* input, size_t input_size, size_t* offset,
             uint32_t* value) {
  if (*offset + 4 > input_size) {
    return false;
  }
  *value = static_cast<uint32_t>(input[*offset]) |
           (static_cast<uint32_t>(input[*offset + 1]) << 8) |
           (static_cast<uint32_t>(input[*offset + 2]) << 16) |
           (static_cast<uint32_t>(input[*offset + 3]) << 24);
  *offset += 4;
  return true;
}

bool ReadU64(const uint8_t* input, size_t input_size, size_t* offset,
             uint64_t* value) {
  if (*offset + 8 > input_size) {
    return false;
  }
  *value = 0;
  for (uint32_t byte_idx = 0; byte_idx != 8; ++byte_idx) {
    *value |= static_cast<uint64_t>(input[*offset + byte_idx])
              << (8 * byte_idx);
  }
  *offset += 8;
  return true;
}

bool Seek(FILE* file, uint64_t offset, std::string* error) {
  if (offset > static_cast<uint64_t>(std::numeric_limits<off_t>::max())) {
    SetError("File offset exceeds platform limits.", error);
    return false;
  }
  if (fseeko(file, static_cast<off_t>(offset), SEEK_SET)) {
    SetError(std::string("File seek failed: ") + strerror(errno), error);
    return false;
  }
  return true;
}

bool Tell(FILE* file, uint64_t* offset, std::string* error) {
  const off_t position = ftello(file);
  if (position < 0) {
    SetError(std::string("File position query failed: ") + strerror(errno),
             error);
    return false;
  }
  *offset = static_cast<uint64_t>(position);
  return true;
}

bool WriteBytes(FILE* file, const uint8_t* data, size_t byte_ct,
                std::string* error) {
  if (byte_ct && (fwrite(data, 1, byte_ct, file) != byte_ct)) {
    SetError(std::string("File write failed: ") + strerror(errno), error);
    return false;
  }
  return true;
}

bool ReadBytes(FILE* file, uint8_t* data, size_t byte_ct,
               std::string* error) {
  if (byte_ct && (fread(data, 1, byte_ct, file) != byte_ct)) {
    SetError(feof(file) ? "Unexpected end of file."
                        : std::string("File read failed: ") + strerror(errno),
             error);
    return false;
  }
  return true;
}

std::vector<uint8_t> SerializeHeader(const ContainerParams& params,
                                     const ContainerMetadata& metadata) {
  std::vector<uint8_t> output;
  uint32_t flags = 0;
  if (!metadata.nonref_flags.empty()) {
    flags |= kContainerFlagNonrefBitmap;
  }
  if (metadata.all_nonref) {
    flags |= kContainerFlagAllNonref;
  }
  const uint64_t block_table_offset = kFileHeaderByteCt;
  const uint64_t metadata_offset =
      block_table_offset +
      static_cast<uint64_t>(params.block_ct) * kBlockIndexByteCt;
  const uint64_t metadata_byte_ct = metadata.nonref_flags.size();
  const uint64_t data_offset = metadata_offset + metadata_byte_ct;
  output.insert(output.end(), kFileMagic.begin(), kFileMagic.end());
  AppendU32(params.variant_ct, &output);
  AppendU32(params.sample_ct, &output);
  uint8_t header_ctrl = 0x40;
  if (metadata.all_nonref) {
    header_ctrl = 0x80;
  } else if (!metadata.nonref_flags.empty()) {
    header_ctrl = 0xc0;
  }
  output.push_back(header_ctrl);
  AppendU32(kPgenRansFormatVersion, &output);
  AppendU32(kFileHeaderByteCt, &output);
  AppendU32(params.block_variant_ct, &output);
  AppendU32(params.anchor_ct, &output);
  AppendU32(params.state_ct, &output);
  AppendU32(params.scale_bits, &output);
  AppendU32(params.restart_variant_ct, &output);
  AppendU32(params.block_ct, &output);
  AppendU32(params.max_allele_ct, &output);
  AppendU32(flags, &output);
  AppendU64(block_table_offset, &output);
  AppendU64(metadata_offset, &output);
  AppendU64(metadata_byte_ct, &output);
  AppendU64(data_offset, &output);
  AppendU64(0, &output);
  AppendU32(0, &output);
  return output;
}

std::vector<uint8_t> SerializeBlockIndex(
    const std::vector<BlockIndexEntry>& entries) {
  std::vector<uint8_t> output;
  output.reserve(entries.size() * kBlockIndexByteCt);
  for (const BlockIndexEntry& entry : entries) {
    AppendU32(entry.first_variant, &output);
    AppendU32(entry.variant_ct, &output);
    AppendU64(entry.file_offset, &output);
    AppendU64(entry.byte_ct, &output);
    AppendU32(entry.checksum, &output);
    AppendU32(0, &output);
  }
  return output;
}

bool ValidateParams(const ContainerParams& params, std::string* error) {
  if ((!params.sample_ct) || (!params.variant_ct) ||
      (params.variant_ct > kMaximumVariantCt) ||
      (!params.block_variant_ct) || (params.block_variant_ct > UINT16_MAX) ||
      (!params.anchor_ct) || (params.anchor_ct > 256) ||
      (params.anchor_ct > params.block_variant_ct) || (!params.state_ct) ||
      (params.state_ct > 256) || (params.scale_bits < 8) ||
      (params.scale_bits > 16) || (!params.restart_variant_ct) ||
      (params.restart_variant_ct > UINT16_MAX) || (!params.block_ct) ||
      (params.block_ct > params.variant_ct) ||
      (params.max_allele_ct < 2) || (params.max_allele_ct > 255)) {
    SetError("Invalid conditional-rANS container parameters.", error);
    return false;
  }
  return true;
}

bool ValidateMetadata(const ContainerParams& params,
                      const ContainerMetadata& metadata,
                      std::string* error) {
  const size_t expected_byte_ct =
      (static_cast<size_t>(params.variant_ct) + 7) / 8;
  if (metadata.all_nonref && !metadata.nonref_flags.empty()) {
    SetError("Conditional-rANS nonreference metadata is contradictory.",
             error);
    return false;
  }
  if (!metadata.nonref_flags.empty()) {
    if (metadata.nonref_flags.size() != expected_byte_ct) {
      SetError("Conditional-rANS nonreference bitmap has the wrong size.",
               error);
      return false;
    }
    const uint32_t trailing_bit_ct = params.variant_ct % 8;
    if (trailing_bit_ct &&
        (metadata.nonref_flags.back() &
         static_cast<uint8_t>(0xffU << trailing_bit_ct))) {
      SetError("Conditional-rANS nonreference bitmap has nonzero padding.",
               error);
      return false;
    }
  }
  return true;
}

}  // namespace

std::vector<uint32_t> ScheduledAnchorOffsets(uint32_t block_variant_ct,
                                             uint32_t requested_anchor_ct) {
  const uint32_t anchor_ct =
      std::min(block_variant_ct, requested_anchor_ct);
  std::vector<uint32_t> result;
  result.reserve(anchor_ct);
  for (uint32_t anchor_idx = 0; anchor_idx != anchor_ct; ++anchor_idx) {
    const uint64_t numerator =
        static_cast<uint64_t>(2 * anchor_idx + 1) * block_variant_ct;
    result.push_back(
        static_cast<uint32_t>(numerator / (2 * anchor_ct)));
  }
  return result;
}

ContainerWriter::~ContainerWriter() {
  if (file_) {
    fclose(file_);
  }
}

bool ContainerWriter::Open(const std::string& path,
                           const ContainerParams& params,
                           const ContainerMetadata& metadata,
                           std::string* error) {
  if (file_) {
    SetError("Container writer is already open.", error);
    return false;
  }
  if (!ValidateParams(params, error) ||
      !ValidateMetadata(params, metadata, error)) {
    return false;
  }
  file_ = fopen(path.c_str(), "w+b");
  if (!file_) {
    SetError(std::string("Could not create container: ") + strerror(errno),
             error);
    return false;
  }
  params_ = params;
  metadata_ = metadata;
  block_index_.reserve(params.block_ct);
  const uint64_t data_offset =
      kFileHeaderByteCt +
      static_cast<uint64_t>(params.block_ct) * kBlockIndexByteCt +
      metadata_.nonref_flags.size();
  if (!Seek(file_, data_offset, error)) {
    failed_ = true;
    fclose(file_);
    file_ = nullptr;
    RemovePartialOutput(path, error);
    return false;
  }
  return true;
}

bool ContainerWriter::WriteBlock(const EncodedBlock& block,
                                 std::string* error) {
  if ((!file_) || failed_) {
    SetError("Container writer is not writable.", error);
    return false;
  }
  if ((block_index_.size() == params_.block_ct) ||
      (block.first_variant != next_variant_) || block.records.empty() ||
      (block.records.size() > params_.block_variant_ct) ||
      (block.records.size() > UINT16_MAX) ||
      (next_variant_ + block.records.size() > params_.variant_ct)) {
    SetError("Invalid or out-of-order container block.", error);
    failed_ = true;
    return false;
  }
  const uint32_t block_variant_ct =
      static_cast<uint32_t>(block.records.size());
  const uint32_t block_anchor_ct =
      std::min(params_.anchor_ct, block_variant_ct);
  const uint32_t restart_group_ct =
      (block_variant_ct + params_.restart_variant_ct - 1) /
      params_.restart_variant_ct;
  const uint32_t restart_offset_ct = restart_group_ct - 1;

  std::vector<uint8_t> serialized;
  uint64_t record_payload_byte_ct = 0;
  for (const std::vector<uint8_t>& record : block.records) {
    if (record.empty() || (record.size() > kMaximumRecordByteCt)) {
      SetError("Record is empty or exceeds the v1 24-bit length limit.",
               error);
      failed_ = true;
      return false;
    }
    record_payload_byte_ct += record.size();
  }
  if (record_payload_byte_ct > UINT32_MAX) {
    SetError("Block record payload exceeds the v1 32-bit offset limit.",
             error);
    failed_ = true;
    return false;
  }
  const uint64_t block_byte_ct =
      kBlockHeaderByteCt + 3LLU * block_variant_ct +
      4LLU * restart_offset_ct + record_payload_byte_ct;
  if (block_byte_ct > std::numeric_limits<size_t>::max()) {
    SetError("Block exceeds platform memory limits.", error);
    failed_ = true;
    return false;
  }
  serialized.reserve(static_cast<size_t>(block_byte_ct));
  AppendU32(kBlockMagic, &serialized);
  AppendU32(block.first_variant, &serialized);
  AppendU16(static_cast<uint16_t>(block_variant_ct), &serialized);
  AppendU16(static_cast<uint16_t>(block_anchor_ct), &serialized);
  AppendU16(static_cast<uint16_t>(params_.restart_variant_ct), &serialized);
  AppendU16(static_cast<uint16_t>(restart_offset_ct), &serialized);
  for (const std::vector<uint8_t>& record : block.records) {
    AppendU24(static_cast<uint32_t>(record.size()), &serialized);
  }
  uint64_t cumulative_record_bytes = 0;
  uint32_t next_restart_variant = params_.restart_variant_ct;
  for (uint32_t variant_offset = 0; variant_offset != block_variant_ct;
       ++variant_offset) {
    if (variant_offset == next_restart_variant) {
      if (cumulative_record_bytes > UINT32_MAX) {
        SetError("Block restart offset exceeds the v1 32-bit limit.", error);
        failed_ = true;
        return false;
      }
      AppendU32(static_cast<uint32_t>(cumulative_record_bytes), &serialized);
      next_restart_variant += params_.restart_variant_ct;
    }
    cumulative_record_bytes += block.records[variant_offset].size();
  }
  for (const std::vector<uint8_t>& record : block.records) {
    serialized.insert(serialized.end(), record.begin(), record.end());
  }
  if (serialized.size() != block_byte_ct) {
    SetError("Internal block-size mismatch.", error);
    failed_ = true;
    return false;
  }
  uint64_t file_offset;
  if ((!Tell(file_, &file_offset, error)) ||
      (!WriteBytes(file_, serialized.data(), serialized.size(), error))) {
    failed_ = true;
    return false;
  }
  block_index_.push_back(
      {block.first_variant, block_variant_ct, file_offset, block_byte_ct,
       Crc32c(serialized.data(), serialized.size())});
  next_variant_ += block_variant_ct;
  return true;
}

bool ContainerWriter::WriteSerializedBlock(
    uint32_t first_variant, uint32_t variant_ct,
    std::vector<uint8_t>* serialized, std::string* error) {
  if ((!file_) || failed_ || (!serialized)) {
    SetError("Container writer is not writable.", error);
    return false;
  }
  if ((block_index_.size() == params_.block_ct) ||
      (first_variant != next_variant_) || (!variant_ct) ||
      (variant_ct > params_.block_variant_ct) ||
      (variant_ct > UINT16_MAX) ||
      (variant_ct > params_.variant_ct) ||
      (next_variant_ > params_.variant_ct - variant_ct) ||
      (serialized->size() < kBlockHeaderByteCt)) {
    SetError("Invalid or out-of-order serialized container block.", error);
    failed_ = true;
    return false;
  }

  size_t offset = 0;
  uint32_t magic;
  uint32_t old_first_variant;
  uint16_t embedded_variant_ct;
  uint16_t anchor_ct;
  uint16_t restart_variant_ct;
  uint16_t restart_offset_ct;
  if ((!ReadU32(serialized->data(), serialized->size(), &offset, &magic)) ||
      (!ReadU32(serialized->data(), serialized->size(), &offset,
                &old_first_variant)) ||
      (!ReadU16(serialized->data(), serialized->size(), &offset,
                &embedded_variant_ct)) ||
      (!ReadU16(serialized->data(), serialized->size(), &offset,
                &anchor_ct)) ||
      (!ReadU16(serialized->data(), serialized->size(), &offset,
                &restart_variant_ct)) ||
      (!ReadU16(serialized->data(), serialized->size(), &offset,
                &restart_offset_ct)) ||
      (magic != kBlockMagic) || (embedded_variant_ct != variant_ct) ||
      (anchor_ct != std::min(params_.anchor_ct, variant_ct)) ||
      (restart_variant_ct != params_.restart_variant_ct) ||
      (restart_offset_ct !=
       (variant_ct + restart_variant_ct - 1) / restart_variant_ct - 1)) {
    SetError("Invalid serialized conditional-rANS block header.", error);
    failed_ = true;
    return false;
  }
  // Silence compilers which do not recognize that parsing this field is
  // itself part of validating the fixed header.
  (void)old_first_variant;

  uint64_t record_payload_byte_ct = 0;
  std::vector<uint32_t> expected_restart_offsets;
  expected_restart_offsets.reserve(restart_offset_ct);
  for (uint32_t variant_idx = 0; variant_idx != variant_ct; ++variant_idx) {
    uint32_t record_byte_ct;
    if ((!ReadU24(serialized->data(), serialized->size(), &offset,
                  &record_byte_ct)) ||
        (!record_byte_ct)) {
      SetError("Invalid serialized block record-length table.", error);
      failed_ = true;
      return false;
    }
    record_payload_byte_ct += record_byte_ct;
    const uint32_t completed_variant_ct = variant_idx + 1;
    if ((completed_variant_ct < variant_ct) &&
        (!(completed_variant_ct % restart_variant_ct))) {
      if (record_payload_byte_ct > UINT32_MAX) {
        SetError("Serialized block restart offset exceeds format limits.",
                 error);
        failed_ = true;
        return false;
      }
      expected_restart_offsets.push_back(
          static_cast<uint32_t>(record_payload_byte_ct));
    }
  }
  for (uint32_t restart_idx = 0; restart_idx != restart_offset_ct;
       ++restart_idx) {
    uint32_t restart_offset;
    if ((!ReadU32(serialized->data(), serialized->size(), &offset,
                  &restart_offset)) ||
        (restart_offset != expected_restart_offsets[restart_idx])) {
      SetError("Invalid serialized block restart-offset table.", error);
      failed_ = true;
      return false;
    }
  }
  if ((record_payload_byte_ct > UINT32_MAX) ||
      (record_payload_byte_ct != serialized->size() - offset)) {
    SetError("Serialized block record lengths do not span its payload.",
             error);
    failed_ = true;
    return false;
  }

  (*serialized)[4] = static_cast<uint8_t>(first_variant);
  (*serialized)[5] = static_cast<uint8_t>(first_variant >> 8);
  (*serialized)[6] = static_cast<uint8_t>(first_variant >> 16);
  (*serialized)[7] = static_cast<uint8_t>(first_variant >> 24);
  uint64_t file_offset;
  if ((!Tell(file_, &file_offset, error)) ||
      (!WriteBytes(file_, serialized->data(), serialized->size(), error))) {
    failed_ = true;
    return false;
  }
  block_index_.push_back(
      {first_variant, variant_ct, file_offset, serialized->size(),
       Crc32c(serialized->data(), serialized->size())});
  next_variant_ += variant_ct;
  return true;
}

bool ContainerWriter::Close(std::string* error) {
  if (!file_) {
    SetError("Container writer is not open.", error);
    return false;
  }
  if (failed_ || (block_index_.size() != params_.block_ct) ||
      (next_variant_ != params_.variant_ct)) {
    SetError("Container is incomplete and cannot be finalized.", error);
    fclose(file_);
    file_ = nullptr;
    return false;
  }
  const std::vector<uint8_t> header =
      SerializeHeader(params_, metadata_);
  const std::vector<uint8_t> block_index =
      SerializeBlockIndex(block_index_);
  bool success = Seek(file_, 0, error) &&
                 WriteBytes(file_, header.data(), header.size(), error) &&
                 WriteBytes(file_, block_index.data(), block_index.size(),
                            error) &&
                 WriteBytes(file_, metadata_.nonref_flags.data(),
                            metadata_.nonref_flags.size(), error);
  if (success && fflush(file_)) {
    SetError(std::string("File flush failed: ") + strerror(errno), error);
    success = false;
  }
  if (fclose(file_)) {
    if (success) {
      SetError(std::string("File close failed: ") + strerror(errno), error);
    }
    success = false;
  }
  file_ = nullptr;
  return success;
}

ContainerReader::~ContainerReader() {
  Close();
}

bool ContainerReader::Open(const std::string& path, std::string* error) {
  Close();
  file_ = fopen(path.c_str(), "rb");
  if (!file_) {
    SetError(std::string("Could not open container: ") + strerror(errno),
             error);
    return false;
  }
  if (fseeko(file_, 0, SEEK_END)) {
    SetError(std::string("File size query failed: ") + strerror(errno),
             error);
    Close();
    return false;
  }
  const off_t file_size = ftello(file_);
  if (file_size < static_cast<off_t>(kFileHeaderByteCt)) {
    SetError("Container is shorter than its fixed header.", error);
    Close();
    return false;
  }
  file_size_ = static_cast<uint64_t>(file_size);
  return OpenIndex(error);
}

bool ContainerReader::OpenReadAt(uint64_t file_size,
                                 ReadAtCallback read_at,
                                 void* read_at_context,
                                 std::string* error) {
  Close();
  if ((!read_at) || (file_size < kFileHeaderByteCt)) {
    SetError("Invalid conditional-rANS ranged reader.", error);
    return false;
  }
  read_at_ = read_at;
  read_at_context_ = read_at_context;
  file_size_ = file_size;
  return OpenIndex(error);
}

bool ContainerReader::OpenIndex(std::string* error) {
  if ((!file_) && (!read_at_)) {
    SetError("Container reader has no byte source.", error);
    return false;
  }
  if (file_size_ < kFileHeaderByteCt) {
    SetError("Container is shorter than its fixed header.", error);
    Close();
    return false;
  }
  std::array<uint8_t, kFileHeaderByteCt> header;
  if (!ReadAt(0, header.data(), header.size(), error)) {
    Close();
    return false;
  }
  if (!std::equal(kFileMagic.begin(), kFileMagic.end(), header.begin())) {
    SetError("Invalid conditional-rANS PGEN storage mode.", error);
    Close();
    return false;
  }
  size_t offset = kFileMagic.size();
  uint32_t version;
  uint32_t header_byte_ct;
  uint32_t header_ctrl;
  uint32_t flags;
  uint64_t block_table_offset;
  uint64_t metadata_offset;
  uint64_t metadata_byte_ct;
  uint64_t data_offset;
  uint64_t reserved64;
  uint32_t reserved32;
  if ((!ReadU32(header.data(), header.size(), &offset,
                &params_.variant_ct)) ||
      (!ReadU32(header.data(), header.size(), &offset, &params_.sample_ct)) ||
      (offset >= header.size())) {
    SetError("Truncated conditional-rANS PGEN header.", error);
    Close();
    return false;
  }
  header_ctrl = header[offset++];
  if ((!ReadU32(header.data(), header.size(), &offset, &version)) ||
      (!ReadU32(header.data(), header.size(), &offset, &header_byte_ct)) ||
      (!ReadU32(header.data(), header.size(), &offset,
                &params_.block_variant_ct)) ||
      (!ReadU32(header.data(), header.size(), &offset, &params_.anchor_ct)) ||
      (!ReadU32(header.data(), header.size(), &offset, &params_.state_ct)) ||
      (!ReadU32(header.data(), header.size(), &offset, &params_.scale_bits)) ||
      (!ReadU32(header.data(), header.size(), &offset,
                &params_.restart_variant_ct)) ||
      (!ReadU32(header.data(), header.size(), &offset, &params_.block_ct)) ||
      (!ReadU32(header.data(), header.size(), &offset,
                &params_.max_allele_ct)) ||
      (!ReadU32(header.data(), header.size(), &offset, &flags)) ||
      (!ReadU64(header.data(), header.size(), &offset,
                &block_table_offset)) ||
      (!ReadU64(header.data(), header.size(), &offset,
                &metadata_offset)) ||
      (!ReadU64(header.data(), header.size(), &offset,
                &metadata_byte_ct)) ||
      (!ReadU64(header.data(), header.size(), &offset, &data_offset)) ||
      (!ReadU64(header.data(), header.size(), &offset, &reserved64)) ||
      (!ReadU32(header.data(), header.size(), &offset, &reserved32))) {
    SetError("Truncated conditional-rANS PGEN header.", error);
    Close();
    return false;
  }
  const uint64_t expected_data_offset =
      kFileHeaderByteCt +
      static_cast<uint64_t>(params_.block_ct) * kBlockIndexByteCt +
      metadata_byte_ct;
  const uint64_t expected_metadata_offset =
      kFileHeaderByteCt +
      static_cast<uint64_t>(params_.block_ct) * kBlockIndexByteCt;
  const uint64_t expected_nonref_byte_ct =
      (static_cast<uint64_t>(params_.variant_ct) + 7) / 8;
  const uint32_t expected_nonref_storage =
      (flags & kContainerFlagNonrefBitmap)
          ? 3
          : ((flags & kContainerFlagAllNonref) ? 2 : 1);
  if ((version < kPgenRansMinimumFormatVersion) ||
      (version > kPgenRansFormatVersion) ||
      (header_byte_ct != kFileHeaderByteCt) ||
      (block_table_offset != kFileHeaderByteCt) ||
      (header_ctrl & 0x3f) ||
      ((header_ctrl >> 6) != expected_nonref_storage) ||
      (flags & ~kContainerKnownFlags) ||
      ((flags & kContainerFlagNonrefBitmap) &&
       (flags & kContainerFlagAllNonref)) ||
      (metadata_offset != expected_metadata_offset) ||
      (metadata_byte_ct !=
       ((flags & kContainerFlagNonrefBitmap)
            ? expected_nonref_byte_ct
            : 0)) ||
      (data_offset != expected_data_offset) || (data_offset > file_size_) ||
      reserved64 || reserved32 || (offset != header.size()) ||
      (!ValidateParams(params_, error))) {
    if (error && error->empty()) {
      *error = "Invalid conditional-rANS PGEN header.";
    }
    Close();
    return false;
  }
  metadata_.all_nonref = (flags & kContainerFlagAllNonref) != 0;
  const size_t serialized_index_byte_ct =
      static_cast<size_t>(params_.block_ct) * kBlockIndexByteCt;
  std::vector<uint8_t> serialized_index_and_metadata(
      serialized_index_byte_ct + static_cast<size_t>(metadata_byte_ct));
  if (!ReadAt(kFileHeaderByteCt, serialized_index_and_metadata.data(),
              serialized_index_and_metadata.size(), error)) {
    Close();
    return false;
  }
  metadata_.nonref_flags.assign(
      serialized_index_and_metadata.begin() +
          static_cast<std::ptrdiff_t>(serialized_index_byte_ct),
      serialized_index_and_metadata.end());
  if (!ValidateMetadata(params_, metadata_, error)) {
    Close();
    return false;
  }
  const uint8_t* serialized_index =
      serialized_index_and_metadata.data();
  block_index_.resize(params_.block_ct);
  offset = 0;
  uint32_t next_variant = 0;
  uint64_t next_file_offset = data_offset;
  for (BlockIndexEntry& entry : block_index_) {
    uint32_t checksum;
    uint32_t entry_reserved;
    if ((!ReadU32(serialized_index, serialized_index_byte_ct, &offset,
                  &entry.first_variant)) ||
        (!ReadU32(serialized_index, serialized_index_byte_ct, &offset,
                  &entry.variant_ct)) ||
        (!ReadU64(serialized_index, serialized_index_byte_ct, &offset,
                  &entry.file_offset)) ||
        (!ReadU64(serialized_index, serialized_index_byte_ct, &offset,
                  &entry.byte_ct)) ||
        (!ReadU32(serialized_index, serialized_index_byte_ct, &offset,
                  &checksum)) ||
        (!ReadU32(serialized_index, serialized_index_byte_ct, &offset,
                  &entry_reserved)) ||
        entry_reserved || (entry.first_variant != next_variant) ||
        (!entry.variant_ct) ||
        (entry.variant_ct > params_.block_variant_ct) ||
        (entry.file_offset != next_file_offset) ||
        (entry.byte_ct < kBlockHeaderByteCt) ||
        (entry.file_offset > file_size_) ||
        (entry.byte_ct > file_size_ - entry.file_offset)) {
      SetError("Invalid conditional-rANS block index.", error);
      Close();
      return false;
    }
    entry.checksum = checksum;
    next_variant += entry.variant_ct;
    next_file_offset += entry.byte_ct;
  }
  if ((next_variant != params_.variant_ct) ||
      (next_file_offset != file_size_)) {
    SetError("Container index does not span the file.", error);
    Close();
    return false;
  }
  return true;
}

void ContainerReader::Close() {
  if (file_) {
    fclose(file_);
  }
  file_ = nullptr;
  read_at_ = nullptr;
  read_at_context_ = nullptr;
  params_ = {};
  metadata_ = {};
  block_index_.clear();
  file_size_ = 0;
}

bool ContainerReader::ReadAt(uint64_t offset, uint8_t* destination,
                             size_t byte_ct, std::string* error) {
  if ((offset > file_size_) || (byte_ct > file_size_ - offset)) {
    SetError("Conditional-rANS ranged read is outside the file.", error);
    return false;
  }
  if (read_at_) {
    return read_at_(read_at_context_, offset, destination, byte_ct, error);
  }
  return Seek(file_, offset, error) &&
         ReadBytes(file_, destination, byte_ct, error);
}

ByteSpan EncodedBlockView::record(uint32_t variant_offset) const {
  if (variant_offset >= variant_ct()) {
    return {};
  }
  const uint32_t begin = record_offsets_[variant_offset];
  const uint32_t end = record_offsets_[variant_offset + 1];
  return {record_data_ + begin, static_cast<size_t>(end - begin)};
}

bool ContainerReader::ReadBlockView(uint32_t block_idx,
                                    std::vector<uint8_t>* storage,
                                    EncodedBlockView* block,
                                    std::string* error) {
  if (((!file_) && (!read_at_)) || (block_idx >= block_index_.size()) ||
      (!storage) || (!block)) {
    SetError("Invalid block read request.", error);
    return false;
  }
  const BlockIndexEntry& entry = block_index_[block_idx];
  if (entry.byte_ct > std::numeric_limits<size_t>::max()) {
    SetError("Block exceeds platform memory limits.", error);
    return false;
  }
  storage->resize(static_cast<size_t>(entry.byte_ct));
  if (!ReadAt(entry.file_offset, storage->data(), storage->size(), error)) {
    return false;
  }
  return ParseBlockView(block_idx, storage->data(), storage->size(), block,
                        error);
}

bool ContainerReader::ParseBlockView(uint32_t block_idx,
                                     const uint8_t* data,
                                     size_t byte_ct,
                                     EncodedBlockView* block,
                                     std::string* error) const {
  if (((!file_) && (!read_at_)) || (block_idx >= block_index_.size()) ||
      (!data) || (!block)) {
    SetError("Invalid block parse request.", error);
    return false;
  }
  const BlockIndexEntry& entry = block_index_[block_idx];
  if (byte_ct != entry.byte_ct) {
    SetError("Conditional-rANS block byte count mismatch.", error);
    return false;
  }
  if (Crc32c(data, byte_ct) != entry.checksum) {
    SetError("Conditional-rANS block checksum mismatch.", error);
    return false;
  }
  size_t offset = 0;
  uint32_t magic;
  uint32_t first_variant;
  uint16_t variant_ct;
  uint16_t anchor_ct;
  uint16_t restart_variant_ct;
  uint16_t restart_offset_ct;
  if ((!ReadU32(data, byte_ct, &offset, &magic)) ||
      (!ReadU32(data, byte_ct, &offset, &first_variant)) ||
      (!ReadU16(data, byte_ct, &offset, &variant_ct)) ||
      (!ReadU16(data, byte_ct, &offset, &anchor_ct)) ||
      (!ReadU16(data, byte_ct, &offset,
                &restart_variant_ct)) ||
      (!ReadU16(data, byte_ct, &offset,
                &restart_offset_ct)) ||
      (magic != kBlockMagic) || (first_variant != entry.first_variant) ||
      (variant_ct != entry.variant_ct) ||
      (anchor_ct != std::min(params_.anchor_ct, entry.variant_ct)) ||
      (restart_variant_ct != params_.restart_variant_ct)) {
    SetError("Invalid conditional-rANS block header.", error);
    return false;
  }
  const uint32_t expected_restart_offset_ct =
      (variant_ct + restart_variant_ct - 1) / restart_variant_ct - 1;
  if (restart_offset_ct != expected_restart_offset_ct) {
    SetError("Invalid block restart-offset count.", error);
    return false;
  }
  std::vector<uint32_t> record_lengths(variant_ct);
  for (uint32_t& record_length : record_lengths) {
    if ((!ReadU24(data, byte_ct, &offset, &record_length)) ||
        (!record_length)) {
      SetError("Invalid or truncated block record-length table.", error);
      return false;
    }
  }
  std::vector<uint32_t> restart_offsets(restart_offset_ct);
  for (uint32_t& restart_offset : restart_offsets) {
    if (!ReadU32(data, byte_ct, &offset, &restart_offset)) {
      SetError("Truncated block restart-offset table.", error);
      return false;
    }
  }
  const size_t record_data_offset = offset;
  uint64_t cumulative_record_bytes = 0;
  uint32_t restart_idx = 0;
  for (uint32_t variant_offset = 0; variant_offset != variant_ct;
       ++variant_offset) {
    if (variant_offset &&
        (!(variant_offset % restart_variant_ct))) {
      if ((restart_idx == restart_offsets.size()) ||
          (restart_offsets[restart_idx] != cumulative_record_bytes)) {
        SetError("Block restart offset does not match record lengths.", error);
        return false;
      }
      ++restart_idx;
    }
    cumulative_record_bytes += record_lengths[variant_offset];
  }
  if ((restart_idx != restart_offsets.size()) ||
      (cumulative_record_bytes != byte_ct - record_data_offset) ||
      (cumulative_record_bytes > UINT32_MAX)) {
    SetError("Block record lengths do not span its payload.", error);
    return false;
  }
  block->record_data_ = data + record_data_offset;
  block->record_data_byte_ct_ =
      static_cast<size_t>(cumulative_record_bytes);
  block->first_variant_ = first_variant;
  block->anchor_ct_ = anchor_ct;
  block->record_offsets_.clear();
  block->record_offsets_.reserve(static_cast<size_t>(variant_ct) + 1);
  uint32_t record_offset = 0;
  for (const uint32_t record_length : record_lengths) {
    block->record_offsets_.push_back(record_offset);
    record_offset += record_length;
  }
  block->record_offsets_.push_back(record_offset);
  return true;
}

bool ContainerReader::ReadBlock(uint32_t block_idx, EncodedBlock* block,
                                std::string* error) {
  if (!block) {
    SetError("Invalid block read request.", error);
    return false;
  }
  std::vector<uint8_t> storage;
  EncodedBlockView view;
  if (!ReadBlockView(block_idx, &storage, &view, error)) {
    return false;
  }
  block->first_variant = view.first_variant();
  block->records.clear();
  block->records.reserve(view.variant_ct());
  for (uint32_t variant_offset = 0;
       variant_offset != view.variant_ct(); ++variant_offset) {
    const ByteSpan record = view.record(variant_offset);
    block->records.emplace_back(record.data, record.data + record.size);
  }
  return true;
}

bool ContainerReader::FindBlock(uint32_t variant_idx, uint32_t* block_idx,
                                std::string* error) const {
  if (((!file_) && (!read_at_)) || (!block_idx) ||
      (variant_idx >= params_.variant_ct)) {
    SetError("Variant index is outside the container.", error);
    return false;
  }
  const auto iter = std::upper_bound(
      block_index_.begin(), block_index_.end(), variant_idx,
      [](uint32_t value, const BlockIndexEntry& entry) {
        return value < entry.first_variant;
      });
  if (iter == block_index_.begin()) {
    SetError("Container block lookup failed.", error);
    return false;
  }
  *block_idx = static_cast<uint32_t>(
      std::distance(block_index_.begin(), iter - 1));
  return true;
}

bool ConcatenateContainers(const std::vector<std::string>& input_paths,
                           const std::string& output_path,
                           ContainerConcatStats* stats,
                           std::string* error) {
  if (stats) {
    *stats = {};
  }
  if (input_paths.empty() || output_path.empty()) {
    SetError("Conditional-rANS concatenation requires input and output paths.",
             error);
    return false;
  }
  for (const std::string& input_path : input_paths) {
    if (input_path == output_path) {
      SetError("Conditional-rANS concatenation output is also an input.",
               error);
      return false;
    }
  }
  if (PathExists(output_path)) {
    SetError("Conditional-rANS concatenation output already exists.", error);
    return false;
  }
  if (input_paths.size() > UINT32_MAX) {
    SetError("Too many conditional-rANS concatenation inputs.", error);
    return false;
  }

  ContainerParams output_params;
  ContainerMetadata output_metadata;
  uint64_t variant_ct = 0;
  uint64_t block_ct = 0;
  uint64_t copied_block_byte_ct = 0;
  bool saw_nonref = false;
  bool saw_ref = false;
  std::vector<ContainerParams> input_params;
  std::vector<ContainerMetadata> input_metadata;
  std::vector<std::vector<BlockIndexEntry>> input_block_indexes;
  input_params.reserve(input_paths.size());
  input_metadata.reserve(input_paths.size());
  input_block_indexes.reserve(input_paths.size());
  for (uint32_t input_idx = 0; input_idx != input_paths.size();
       ++input_idx) {
    ContainerReader reader;
    if (!reader.Open(input_paths[input_idx], error)) {
      return false;
    }
    const ContainerParams& params = reader.params();
    if (!input_idx) {
      output_params = params;
      output_params.variant_ct = 0;
      output_params.block_ct = 0;
    } else if ((params.sample_ct != output_params.sample_ct) ||
               (params.anchor_ct != output_params.anchor_ct) ||
               (params.state_ct != output_params.state_ct) ||
               (params.scale_bits != output_params.scale_bits) ||
               (params.restart_variant_ct !=
                output_params.restart_variant_ct)) {
      SetError("Conditional-rANS inputs have incompatible parameters.",
               error);
      return false;
    }
    output_params.block_variant_ct =
        std::max(output_params.block_variant_ct, params.block_variant_ct);
    output_params.max_allele_ct =
        std::max(output_params.max_allele_ct, params.max_allele_ct);
    if ((variant_ct + params.variant_ct > kMaximumVariantCt) ||
        (block_ct + params.block_ct > UINT32_MAX)) {
      SetError("Conditional-rANS concatenation count exceeds format limits.",
               error);
      return false;
    }
    const uint32_t global_first_variant =
        static_cast<uint32_t>(variant_ct);
    const ContainerMetadata& metadata = reader.metadata();
    input_params.push_back(params);
    input_metadata.push_back(metadata);
    input_block_indexes.push_back(reader.block_index());
    for (uint32_t local_variant_idx = 0;
         local_variant_idx != params.variant_ct; ++local_variant_idx) {
      const bool is_nonref =
          metadata.all_nonref ||
          ((!metadata.nonref_flags.empty()) &&
           ((metadata.nonref_flags[local_variant_idx / 8] >>
             (local_variant_idx % 8)) &
            1U));
      saw_nonref |= is_nonref;
      saw_ref |= !is_nonref;
      if (is_nonref) {
        const uint32_t global_variant_idx =
            global_first_variant + local_variant_idx;
        const size_t required_byte_ct =
            static_cast<size_t>(global_variant_idx / 8) + 1;
        if (output_metadata.nonref_flags.size() < required_byte_ct) {
          output_metadata.nonref_flags.resize(required_byte_ct, 0);
        }
        output_metadata.nonref_flags[global_variant_idx / 8] |=
            static_cast<uint8_t>(1U << (global_variant_idx % 8));
      }
    }
    for (const BlockIndexEntry& entry : reader.block_index()) {
      if (copied_block_byte_ct > UINT64_MAX - entry.byte_ct) {
        SetError("Conditional-rANS concatenation byte count overflow.",
                 error);
        return false;
      }
      copied_block_byte_ct += entry.byte_ct;
    }
    variant_ct += params.variant_ct;
    block_ct += params.block_ct;
  }
  output_params.variant_ct = static_cast<uint32_t>(variant_ct);
  output_params.block_ct = static_cast<uint32_t>(block_ct);
  if (!saw_nonref) {
    output_metadata.nonref_flags.clear();
  } else if (!saw_ref) {
    output_metadata.nonref_flags.clear();
    output_metadata.all_nonref = true;
  } else {
    output_metadata.nonref_flags.resize(
        (static_cast<size_t>(output_params.variant_ct) + 7) / 8, 0);
  }
  const uint64_t output_overhead_byte_ct =
      kFileHeaderByteCt +
      static_cast<uint64_t>(output_params.block_ct) * kBlockIndexByteCt +
      output_metadata.nonref_flags.size();
  if ((copied_block_byte_ct >
       UINT64_MAX - output_overhead_byte_ct) ||
      (copied_block_byte_ct + output_overhead_byte_ct >
       static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))) {
    SetError("Conditional-rANS concatenation output exceeds file limits.",
             error);
    return false;
  }

  ContainerWriter writer;
  if (!writer.Open(output_path, output_params, output_metadata, error)) {
    return false;
  }
  uint32_t global_first_variant = 0;
  bool success = true;
  std::vector<uint8_t> serialized;
  for (uint32_t input_idx = 0; input_idx != input_paths.size();
       ++input_idx) {
    const std::string& input_path = input_paths[input_idx];
    ContainerReader reader;
    if (!reader.Open(input_path, error)) {
      success = false;
      break;
    }
    const ContainerMetadata& metadata = reader.metadata();
    if ((!SameParams(reader.params(), input_params[input_idx])) ||
        (metadata.all_nonref != input_metadata[input_idx].all_nonref) ||
        (metadata.nonref_flags !=
         input_metadata[input_idx].nonref_flags) ||
        (!SameBlockIndex(reader.block_index(),
                         input_block_indexes[input_idx]))) {
      SetError(
          "Conditional-rANS input changed during concatenation.", error);
      success = false;
      break;
    }
    for (uint32_t block_idx = 0; block_idx != reader.params().block_ct;
         ++block_idx) {
      EncodedBlockView view;
      if ((!reader.ReadBlockView(block_idx, &serialized, &view, error)) ||
          (!writer.WriteSerializedBlock(global_first_variant,
                                        view.variant_ct(), &serialized,
                                        error))) {
        success = false;
        break;
      }
      global_first_variant += view.variant_ct();
    }
    if (!success) {
      break;
    }
  }
  if (success) {
    success = writer.Close(error);
  }
  if (!success) {
    std::string ignored_error;
    writer.Close(&ignored_error);
    RemovePartialOutput(output_path, error);
    return false;
  }

  if (stats) {
    stats->input_file_ct = static_cast<uint32_t>(input_paths.size());
    stats->sample_ct = output_params.sample_ct;
    stats->variant_ct = output_params.variant_ct;
    stats->block_ct = output_params.block_ct;
    stats->copied_block_byte_ct = copied_block_byte_ct;
    stats->output_byte_ct =
        output_overhead_byte_ct + copied_block_byte_ct;
  }
  return true;
}

}  // namespace pgen_rans
