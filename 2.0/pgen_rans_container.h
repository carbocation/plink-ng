// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef PGEN_RANS_CONTAINER_H_
#define PGEN_RANS_CONTAINER_H_

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace pgen_rans {

struct ContainerParams {
  uint32_t sample_ct = 0;
  uint32_t variant_ct = 0;
  uint32_t block_variant_ct = 128;
  uint32_t anchor_ct = 32;
  uint32_t state_ct = 32;
  uint32_t scale_bits = 12;
  uint32_t restart_variant_ct = 64;
  uint32_t block_ct = 0;
};

struct EncodedBlock {
  uint32_t first_variant = 0;
  std::vector<std::vector<uint8_t>> records;
};

struct BlockIndexEntry {
  uint32_t first_variant = 0;
  uint32_t variant_ct = 0;
  uint64_t file_offset = 0;
  uint64_t byte_ct = 0;
  uint32_t checksum = 0;
};

std::vector<uint32_t> ScheduledAnchorOffsets(uint32_t block_variant_ct,
                                             uint32_t requested_anchor_ct);

class ContainerWriter {
 public:
  ContainerWriter() = default;
  ~ContainerWriter();
  ContainerWriter(const ContainerWriter&) = delete;
  ContainerWriter& operator=(const ContainerWriter&) = delete;

  bool Open(const std::string& path, const ContainerParams& params,
            std::string* error);
  bool WriteBlock(const EncodedBlock& block, std::string* error);
  bool Close(std::string* error);

 private:
  FILE* file_ = nullptr;
  ContainerParams params_;
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
  void Close();
  bool ReadBlock(uint32_t block_idx, EncodedBlock* block,
                 std::string* error);
  bool FindBlock(uint32_t variant_idx, uint32_t* block_idx,
                 std::string* error) const;

  const ContainerParams& params() const { return params_; }
  const std::vector<BlockIndexEntry>& block_index() const {
    return block_index_;
  }

 private:
  FILE* file_ = nullptr;
  ContainerParams params_;
  std::vector<BlockIndexEntry> block_index_;
  uint64_t file_size_ = 0;
};

}  // namespace pgen_rans

#endif  // PGEN_RANS_CONTAINER_H_
