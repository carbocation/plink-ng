// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef PGEN_RANS_HYBRID_H_
#define PGEN_RANS_HYBRID_H_

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "pgen_rans_codec.h"

namespace pgen_rans {

// These flags extend the developer conditional-rANS record format without
// changing the low two reference-mode bits.
// Bit 4 belongs to the guardless-rANS extension in pgen_rans_codec.cc.
constexpr uint8_t kRawPackedRecordFlag = 0x20;
constexpr uint8_t kSparsePredictorRecordFlag = 0x40;
constexpr uint8_t kSparsePredictorBitmapFlag = 0x80;
constexpr uint8_t kAlternateRecordFlagMask =
    kRawPackedRecordFlag | kSparsePredictorRecordFlag |
    kSparsePredictorBitmapFlag;

bool IsAlternateRecord(uint8_t flags);

// Raw records cap worst-case expansion and decode with a bounded memcpy-like
// loop.  Trailing nyps are canonicalized to zero.
bool EncodeRawRecord(const uint64_t* target, uint32_t sample_ct,
                     std::vector<uint8_t>* record, std::string* error);

// Sparse records predict one target genotype for each populated marginal,
// one-reference, or two-reference context and patch only exceptions.
// Positions are represented as either increasing delta varints or a bitmap,
// whichever is smaller.
bool EncodeSparsePredictorRecord(
    const uint64_t* target, const uint64_t* reference1,
    const uint64_t* reference2, uint32_t sample_ct, RecordMode mode,
    uint8_t reference1_idx, uint8_t reference2_idx,
    std::vector<uint8_t>* record, std::string* error);

// Reuses an exact row-major context-by-symbol contingency table when it is
// already available from reference selection.
bool EncodeSparsePredictorRecordFromCounts(
    const uint64_t* target, const uint64_t* reference1,
    const uint64_t* reference2, uint32_t sample_ct, RecordMode mode,
    uint8_t reference1_idx, uint8_t reference2_idx,
    const uint32_t* context_symbol_counts, std::vector<uint8_t>* record,
    std::string* error);

// Decodes and canonically validates a raw or sparse record.  record_size is
// the base-record size, excluding any multiallelic patch suffix.
bool DecodeAlternateRecordToBuffer(
    const uint8_t* record, size_t record_size,
    const uint64_t* const* anchors, uint32_t anchor_ct,
    uint32_t sample_ct, uint64_t* target, size_t target_word_ct,
    RecordMetadata* metadata, std::string* error);

}  // namespace pgen_rans

#endif  // PGEN_RANS_HYBRID_H_
