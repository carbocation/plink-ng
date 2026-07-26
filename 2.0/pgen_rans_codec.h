// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef PGEN_RANS_CODEC_H_
#define PGEN_RANS_CODEC_H_

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace pgen_rans {

enum class RecordMode : uint8_t {
  kMarginal = 0,
  kOneReference = 1,
  kTwoReference = 2,
};

struct CodecParams {
  CodecParams(uint32_t state_count = 32,
              uint32_t scale_bit_count = 12)
      : state_ct(state_count), scale_bits(scale_bit_count) {}

  uint32_t state_ct;
  uint32_t scale_bits;
};

struct RecordMetadata {
  RecordMode mode = RecordMode::kMarginal;
  uint8_t reference1 = 0;
  uint8_t reference2 = 0;
  bool has_entropy_payload = false;
  bool has_interleaved_payload = false;
};

uint32_t PackedWordCt(uint32_t sample_ct);
uint8_t GetPackedGenotype(const uint64_t* genotypes, uint32_t sample_idx);
void SetPackedGenotype(uint64_t* genotypes, uint32_t sample_idx,
                       uint8_t genotype);

bool EncodeRecord(const uint64_t* target, const uint64_t* reference1,
                  const uint64_t* reference2, uint32_t sample_ct,
                  RecordMode mode, uint8_t reference1_idx,
                  uint8_t reference2_idx, const CodecParams& params,
                  std::vector<uint8_t>* record, std::string* error);

bool EstimateRecordBytes(const uint32_t* context_symbol_counts,
                         RecordMode mode, const CodecParams& params,
                         uint64_t* record_bytes, std::string* error);

bool ParseRecordMetadata(const uint8_t* record, size_t record_size,
                         RecordMetadata* metadata, std::string* error);

bool DecodeRecordToBuffer(const uint8_t* record, size_t record_size,
                          const uint64_t* const* anchors,
                          uint32_t anchor_ct, uint32_t sample_ct,
                          const CodecParams& params, uint64_t* target,
                          size_t target_word_ct, RecordMetadata* metadata,
                          std::string* error);

// Container blocks have already passed their structural and checksum
// validation.  This entry point additionally assumes that references do
// not select model contexts that were absent during encoding.
bool DecodeRecordToBufferFromValidatedBlock(
    const uint8_t* record, size_t record_size,
    const uint64_t* const* anchors, uint32_t anchor_ct,
    uint32_t sample_ct, const CodecParams& params, uint64_t* target,
    size_t target_word_ct, RecordMetadata* metadata,
    std::string* error);

bool DecodeRecord(const uint8_t* record, size_t record_size,
                  const uint64_t* const* anchors, uint32_t anchor_ct,
                  uint32_t sample_ct, const CodecParams& params,
                  std::vector<uint64_t>* target, RecordMetadata* metadata,
                  std::string* error);

}  // namespace pgen_rans

#endif  // PGEN_RANS_CODEC_H_
