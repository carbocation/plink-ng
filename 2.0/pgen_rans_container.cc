// SPDX-License-Identifier: GPL-3.0-or-later

#include "pgen_rans_container.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstddef>
#include <cstring>
#include <limits>

namespace pgen_rans {
namespace {

constexpr std::array<uint8_t, 8> kFileMagic = {
    'P', 'G', 'R', 'A', 'N', 'S', '1', '\0'};
constexpr uint32_t kFormatVersion = 1;
constexpr uint32_t kFileHeaderByteCt = 64;
constexpr uint32_t kBlockIndexByteCt = 32;
constexpr uint32_t kBlockHeaderByteCt = 16;
constexpr uint32_t kBlockMagic = 0x314b4c42U;
constexpr uint32_t kMaximumRecordByteCt = 0xffffffU;
constexpr uint32_t kCrc32cPolynomial = 0x82f63b78U;

void SetError(const std::string& message, std::string* error) {
  if (error) {
    *error = message;
  }
}

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

uint32_t Crc32c(const uint8_t* data, size_t byte_ct) {
  const std::array<uint32_t, 256>& table = Crc32cTable();
  uint32_t checksum = UINT32_MAX;
  for (size_t byte_idx = 0; byte_idx != byte_ct; ++byte_idx) {
    checksum =
        table[(checksum ^ data[byte_idx]) & 0xffU] ^ (checksum >> 8);
  }
  return ~checksum;
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

std::vector<uint8_t> SerializeHeader(const ContainerParams& params) {
  std::vector<uint8_t> output;
  output.insert(output.end(), kFileMagic.begin(), kFileMagic.end());
  AppendU32(kFormatVersion, &output);
  AppendU32(kFileHeaderByteCt, &output);
  AppendU32(params.sample_ct, &output);
  AppendU32(params.variant_ct, &output);
  AppendU32(params.block_variant_ct, &output);
  AppendU32(params.anchor_ct, &output);
  AppendU32(params.state_ct, &output);
  AppendU32(params.scale_bits, &output);
  AppendU32(params.restart_variant_ct, &output);
  AppendU32(params.block_ct, &output);
  AppendU64(kFileHeaderByteCt, &output);
  AppendU64(kFileHeaderByteCt +
            static_cast<uint64_t>(params.block_ct) * kBlockIndexByteCt,
            &output);
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
      (!params.block_variant_ct) || (params.block_variant_ct > UINT16_MAX) ||
      (!params.anchor_ct) || (params.anchor_ct > 256) ||
      (params.anchor_ct > params.block_variant_ct) || (!params.state_ct) ||
      (params.state_ct > 256) || (params.scale_bits < 8) ||
      (params.scale_bits > 16) || (!params.restart_variant_ct) ||
      (params.restart_variant_ct > UINT16_MAX) || (!params.block_ct) ||
      (params.block_ct > params.variant_ct)) {
    SetError("Invalid conditional-rANS container parameters.", error);
    return false;
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
                           std::string* error) {
  if (file_) {
    SetError("Container writer is already open.", error);
    return false;
  }
  if (!ValidateParams(params, error)) {
    return false;
  }
  file_ = fopen(path.c_str(), "w+b");
  if (!file_) {
    SetError(std::string("Could not create container: ") + strerror(errno),
             error);
    return false;
  }
  params_ = params;
  block_index_.reserve(params.block_ct);
  const uint64_t data_offset =
      kFileHeaderByteCt +
      static_cast<uint64_t>(params.block_ct) * kBlockIndexByteCt;
  if (!Seek(file_, data_offset, error)) {
    failed_ = true;
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
  const std::vector<uint8_t> header = SerializeHeader(params_);
  const std::vector<uint8_t> block_index =
      SerializeBlockIndex(block_index_);
  bool success = Seek(file_, 0, error) &&
                 WriteBytes(file_, header.data(), header.size(), error) &&
                 WriteBytes(file_, block_index.data(), block_index.size(),
                            error);
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
    SetError("Invalid conditional-rANS container magic.", error);
    Close();
    return false;
  }
  size_t offset = kFileMagic.size();
  uint32_t version;
  uint32_t header_byte_ct;
  uint64_t block_table_offset;
  uint64_t data_offset;
  if ((!ReadU32(header.data(), header.size(), &offset, &version)) ||
      (!ReadU32(header.data(), header.size(), &offset, &header_byte_ct)) ||
      (!ReadU32(header.data(), header.size(), &offset, &params_.sample_ct)) ||
      (!ReadU32(header.data(), header.size(), &offset, &params_.variant_ct)) ||
      (!ReadU32(header.data(), header.size(), &offset,
                &params_.block_variant_ct)) ||
      (!ReadU32(header.data(), header.size(), &offset, &params_.anchor_ct)) ||
      (!ReadU32(header.data(), header.size(), &offset, &params_.state_ct)) ||
      (!ReadU32(header.data(), header.size(), &offset, &params_.scale_bits)) ||
      (!ReadU32(header.data(), header.size(), &offset,
                &params_.restart_variant_ct)) ||
      (!ReadU32(header.data(), header.size(), &offset, &params_.block_ct)) ||
      (!ReadU64(header.data(), header.size(), &offset,
                &block_table_offset)) ||
      (!ReadU64(header.data(), header.size(), &offset, &data_offset))) {
    SetError("Truncated conditional-rANS container header.", error);
    Close();
    return false;
  }
  const uint64_t expected_data_offset =
      kFileHeaderByteCt +
      static_cast<uint64_t>(params_.block_ct) * kBlockIndexByteCt;
  if ((version != kFormatVersion) ||
      (header_byte_ct != kFileHeaderByteCt) ||
      (block_table_offset != kFileHeaderByteCt) ||
      (data_offset != expected_data_offset) || (data_offset > file_size_) ||
      (!ValidateParams(params_, error))) {
    if (error && error->empty()) {
      *error = "Invalid conditional-rANS container header.";
    }
    Close();
    return false;
  }
  std::vector<uint8_t> serialized_index(
      static_cast<size_t>(params_.block_ct) * kBlockIndexByteCt);
  if (!ReadAt(kFileHeaderByteCt, serialized_index.data(),
              serialized_index.size(), error)) {
    Close();
    return false;
  }
  block_index_.resize(params_.block_ct);
  offset = 0;
  uint32_t next_variant = 0;
  uint64_t next_file_offset = data_offset;
  for (BlockIndexEntry& entry : block_index_) {
    uint32_t checksum;
    uint32_t reserved;
    if ((!ReadU32(serialized_index.data(), serialized_index.size(), &offset,
                  &entry.first_variant)) ||
        (!ReadU32(serialized_index.data(), serialized_index.size(), &offset,
                  &entry.variant_ct)) ||
        (!ReadU64(serialized_index.data(), serialized_index.size(), &offset,
                  &entry.file_offset)) ||
        (!ReadU64(serialized_index.data(), serialized_index.size(), &offset,
                  &entry.byte_ct)) ||
        (!ReadU32(serialized_index.data(), serialized_index.size(), &offset,
                  &checksum)) ||
        (!ReadU32(serialized_index.data(), serialized_index.size(), &offset,
                  &reserved)) ||
        reserved || (entry.first_variant != next_variant) ||
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

}  // namespace pgen_rans
