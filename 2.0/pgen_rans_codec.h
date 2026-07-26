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

// Decoder selection is automatic in production.  The override and
// observation hooks are thread-local and exist so each compiled SIMD path can
// be covered by correctness tests and benchmarked against the scalar decoder.
enum class DecodeKernel : uint8_t {
  kAuto = 0,
  kScalar = 1,
  kAvx2 = 2,
  kAvx512 = 3,
  kNeon = 4,
};

bool DecodeKernelSupported(DecodeKernel kernel);
void SetDecodeKernelForTesting(DecodeKernel kernel);
DecodeKernel LastDecodeKernelForTesting();

// Encoder selection is likewise automatic.  The SIMD encoders intentionally
// retain the scalar encoder's byte layout, so these hooks can verify exact
// record identity as well as round-trip correctness.
enum class EncodeKernel : uint8_t {
  kAuto = 0,
  kScalar = 1,
  kAvx512 = 2,
  kAvx2 = 3,
};

bool EncodeKernelSupported(EncodeKernel kernel);
void SetEncodeKernelForTesting(EncodeKernel kernel);
EncodeKernel LastEncodeKernelForTesting();

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
  bool has_multiallelic_patches = false;
  bool is_raw_packed = false;
  bool is_sparse_predictor = false;
};

// PGEN's base 2-bit hardcall stream collapses all alternate alleles to ALT1.
// These sparse patches restore exact multiallelic allele codes.  patch_01 has
// one value per sample ID; patch_10 has two consecutive values per sample ID.
struct MultiallelicPatches {
  uint16_t allele_ct = 2;
  std::vector<uint32_t> patch_01_sample_ids;
  std::vector<uint8_t> patch_01_values;
  std::vector<uint32_t> patch_10_sample_ids;
  std::vector<uint8_t> patch_10_values;
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

// Identical to EncodeRecord(), but reuses an exact row-major
// context-by-symbol contingency table instead of rescanning the genotypes to
// rebuild the entropy model.  The table must sum to sample_ct.
bool EncodeRecordFromCounts(
    const uint64_t* target, const uint64_t* reference1,
    const uint64_t* reference2, uint32_t sample_ct, RecordMode mode,
    uint8_t reference1_idx, uint8_t reference2_idx,
    const uint32_t* context_symbol_counts, const CodecParams& params,
    std::vector<uint8_t>* record, std::string* error);

bool EstimateRecordBytes(const uint32_t* context_symbol_counts,
                         RecordMode mode, const CodecParams& params,
                         uint64_t* record_bytes, std::string* error);

bool ParseRecordMetadata(const uint8_t* record, size_t record_size,
                         RecordMetadata* metadata, std::string* error);

bool AppendMultiallelicPatches(uint32_t sample_ct,
                               const MultiallelicPatches& patches,
                               std::vector<uint8_t>* record,
                               std::string* error);

bool DecodeMultiallelicPatches(const uint8_t* record, size_t record_size,
                               uint32_t sample_ct,
                               MultiallelicPatches* patches,
                               std::string* error);

// Returns the byte count of the ordinary 2-bit/rANS portion of a record.
// This lets collapsed hardcall decoders ignore the sparse allele-code suffix.
bool GetBaseRecordByteCt(const uint8_t* record, size_t record_size,
                         size_t* base_record_size, std::string* error);

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
