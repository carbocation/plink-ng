// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef PGEN_RANS_CONTAINER_H_
#define PGEN_RANS_CONTAINER_H_

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace pgen_rans {

constexpr uint8_t kPgenRansStorageMode = 0x80;
constexpr uint32_t kPgenRansFormatVersion = 1;

struct ContainerParams {
  ContainerParams(uint32_t samples = 0, uint32_t variants = 0,
                  uint32_t block_variants = 128,
                  uint32_t anchors = 32, uint32_t states = 32,
                  uint32_t scale_bit_count = 12,
                  uint32_t restart_variants = 64,
                  uint32_t blocks = 0,
                  uint32_t maximum_allele_ct = 2)
      : sample_ct(samples),
        variant_ct(variants),
        block_variant_ct(block_variants),
        anchor_ct(anchors),
        state_ct(states),
        scale_bits(scale_bit_count),
        restart_variant_ct(restart_variants),
        block_ct(blocks),
        max_allele_ct(maximum_allele_ct) {}

  uint32_t sample_ct;
  uint32_t variant_ct;
  uint32_t block_variant_ct;
  uint32_t anchor_ct;
  uint32_t state_ct;
  uint32_t scale_bits;
  uint32_t restart_variant_ct;
  uint32_t block_ct;
  uint32_t max_allele_ct;
};

struct ContainerMetadata {
  // Empty means all REF alleles are trusted, unless all_nonref is true.
  // Otherwise this is an exact ceil(variant_count / 8)-byte bitmap in output
  // variant order.  A set bit marks a provisional REF allele.
  std::vector<uint8_t> nonref_flags;
  bool all_nonref = false;
};

struct EncodedBlock {
  uint32_t first_variant = 0;
  std::vector<std::vector<uint8_t>> records;
};

struct BlockIndexEntry {
  BlockIndexEntry(uint32_t first = 0, uint32_t variants = 0,
                  uint64_t offset = 0, uint64_t bytes = 0,
                  uint32_t crc = 0)
      : first_variant(first),
        variant_ct(variants),
        file_offset(offset),
        byte_ct(bytes),
        checksum(crc) {}

  uint32_t first_variant;
  uint32_t variant_ct;
  uint64_t file_offset;
  uint64_t byte_ct;
  uint32_t checksum;
};

struct ByteSpan {
  ByteSpan(const uint8_t* bytes = nullptr, size_t byte_ct = 0)
      : data(bytes), size(byte_ct) {}

  const uint8_t* data;
  size_t size;
};

class EncodedBlockView {
 public:
  uint32_t first_variant() const { return first_variant_; }
  uint32_t variant_ct() const {
    return record_offsets_.empty()
               ? 0
               : static_cast<uint32_t>(record_offsets_.size() - 1);
  }
  uint32_t anchor_ct() const { return anchor_ct_; }
  ByteSpan record(uint32_t variant_offset) const;

 private:
  friend class ContainerReader;

  const uint8_t* record_data_ = nullptr;
  size_t record_data_byte_ct_ = 0;
  uint32_t first_variant_ = 0;
  uint32_t anchor_ct_ = 0;
  std::vector<uint32_t> record_offsets_;
};

using ReadAtCallback = bool (*)(void* context, uint64_t offset,
                                uint8_t* destination, size_t byte_ct,
                                std::string* error);

std::vector<uint32_t> ScheduledAnchorOffsets(uint32_t block_variant_ct,
                                             uint32_t requested_anchor_ct);

class ContainerWriter {
 public:
  ContainerWriter() = default;
  ~ContainerWriter();
  ContainerWriter(const ContainerWriter&) = delete;
  ContainerWriter& operator=(const ContainerWriter&) = delete;

  bool Open(const std::string& path, const ContainerParams& params,
            const ContainerMetadata& metadata,
            std::string* error);
  bool WriteBlock(const EncodedBlock& block, std::string* error);
  bool Close(std::string* error);

 private:
  FILE* file_ = nullptr;
  ContainerParams params_;
  ContainerMetadata metadata_;
  std::vector<BlockIndexEntry> block_index_;
  uint32_t next_variant_ = 0;
  bool failed_ = false;
};

class ContainerReader {
 public:
  ContainerReader() = default;
  ~ContainerReader();
  ContainerReader(const ContainerReader&) = delete;
  ContainerReader& operator=(const ContainerReader&) = delete;

  bool Open(const std::string& path, std::string* error);
  bool OpenReadAt(uint64_t file_size, ReadAtCallback read_at,
                  void* read_at_context, std::string* error);
  void Close();
  bool ReadBlockView(uint32_t block_idx, std::vector<uint8_t>* storage,
                     EncodedBlockView* block, std::string* error);
  bool ParseBlockView(uint32_t block_idx, const uint8_t* data,
                      size_t byte_ct, EncodedBlockView* block,
                      std::string* error) const;
  bool ReadBlock(uint32_t block_idx, EncodedBlock* block,
                 std::string* error);
  bool FindBlock(uint32_t variant_idx, uint32_t* block_idx,
                 std::string* error) const;

  const ContainerParams& params() const { return params_; }
  const std::vector<BlockIndexEntry>& block_index() const {
    return block_index_;
  }
  const ContainerMetadata& metadata() const { return metadata_; }

 private:
  bool OpenIndex(std::string* error);
  bool ReadAt(uint64_t offset, uint8_t* destination, size_t byte_ct,
              std::string* error);

  FILE* file_ = nullptr;
  ReadAtCallback read_at_ = nullptr;
  void* read_at_context_ = nullptr;
  ContainerParams params_;
  ContainerMetadata metadata_;
  std::vector<BlockIndexEntry> block_index_;
  uint64_t file_size_ = 0;
};

}  // namespace pgen_rans

#endif  // PGEN_RANS_CONTAINER_H_
