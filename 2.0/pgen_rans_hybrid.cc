// SPDX-License-Identifier: GPL-3.0-or-later

#include "pgen_rans_hybrid.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>

namespace pgen_rans {
namespace {

constexpr uint8_t kModeMask = 0x03;
constexpr uint8_t kEntropyPayloadFlag = 0x04;
constexpr uint8_t kMultiallelicPatchFlag = 0x08;
constexpr uint8_t kKnownFlagMask =
    kModeMask | kEntropyPayloadFlag | kMultiallelicPatchFlag |
    kAlternateRecordFlagMask;

void SetError(const char* message, std::string* error) {
  if (error) {
    *error = message;
  }
}

uint32_t RowCt(RecordMode mode) {
  if (mode == RecordMode::kMarginal) {
    return 1;
  }
  return (mode == RecordMode::kOneReference) ? 4 : 16;
}

uint32_t ContextIndex(RecordMode mode, const uint64_t* reference1,
                      const uint64_t* reference2, uint32_t sample_idx) {
  if (mode == RecordMode::kMarginal) {
    return 0;
  }
  uint32_t context = GetPackedGenotype(reference1, sample_idx);
  if (mode == RecordMode::kTwoReference) {
    context =
        4 * context + GetPackedGenotype(reference2, sample_idx);
  }
  return context;
}

uint64_t ExpandContextMask(uint64_t context_mask, uint8_t symbol) {
  if (symbol == 1) {
    return context_mask;
  }
  if (symbol == 2) {
    return context_mask << 1;
  }
  if (symbol == 3) {
    return context_mask | (context_mask << 1);
  }
  return 0;
}

void BuildGenotypeMasks(uint64_t packed_word, uint64_t* masks) {
  constexpr uint64_t kEvenBitMask = 0x5555555555555555ULL;
  const uint64_t low = packed_word & kEvenBitMask;
  const uint64_t high = (packed_word >> 1) & kEvenBitMask;
  masks[0] = (~(low | high)) & kEvenBitMask;
  masks[1] = low & (~high);
  masks[2] = (~low) & high;
  masks[3] = low & high;
}

void AppendVarint(uint32_t value, std::vector<uint8_t>* output) {
  while (value >= 0x80U) {
    output->push_back(static_cast<uint8_t>(value | 0x80U));
    value >>= 7;
  }
  output->push_back(static_cast<uint8_t>(value));
}

size_t VarintByteCt(uint32_t value) {
  size_t result = 1;
  while (value >= 0x80U) {
    ++result;
    value >>= 7;
  }
  return result;
}

bool ReadVarint(const uint8_t* input, size_t input_size, size_t* offset,
                uint32_t* value) {
  uint32_t result = 0;
  for (uint32_t byte_idx = 0; byte_idx != 5; ++byte_idx) {
    if (*offset == input_size) {
      return false;
    }
    const uint8_t cur_byte = input[(*offset)++];
    if ((byte_idx == 4) && (cur_byte & 0xf0U)) {
      return false;
    }
    result |= static_cast<uint32_t>(cur_byte & 0x7fU)
              << (7 * byte_idx);
    if (!(cur_byte & 0x80U)) {
      *value = result;
      return true;
    }
  }
  return false;
}

size_t DeltaIdByteCt(const std::vector<uint32_t>& sample_ids) {
  size_t result = 0;
  uint32_t previous = 0;
  for (size_t idx = 0; idx != sample_ids.size(); ++idx) {
    const uint32_t delta =
        idx ? sample_ids[idx] - previous : sample_ids[idx];
    result += VarintByteCt(delta);
    previous = sample_ids[idx];
  }
  return result;
}

void AppendDeltaIds(const std::vector<uint32_t>& sample_ids,
                    std::vector<uint8_t>* output) {
  uint32_t previous = 0;
  for (size_t idx = 0; idx != sample_ids.size(); ++idx) {
    const uint32_t delta =
        idx ? sample_ids[idx] - previous : sample_ids[idx];
    AppendVarint(delta, output);
    previous = sample_ids[idx];
  }
}

bool ParseMode(uint8_t flags, RecordMode* mode, std::string* error) {
  if (flags & ~kKnownFlagMask) {
    SetError("Unknown alternate-record flags.", error);
    return false;
  }
  const uint8_t mode_code = flags & kModeMask;
  if (mode_code > static_cast<uint8_t>(RecordMode::kTwoReference)) {
    SetError("Unknown alternate-record reference mode.", error);
    return false;
  }
  *mode = static_cast<RecordMode>(mode_code);
  return true;
}

bool ParseSelectors(const uint8_t* record, size_t record_size,
                    RecordMode mode, size_t* offset,
                    RecordMetadata* metadata, std::string* error) {
  metadata->mode = mode;
  if (mode != RecordMode::kMarginal) {
    if (*offset == record_size) {
      SetError("Truncated sparse first reference selector.", error);
      return false;
    }
    metadata->reference1 = record[(*offset)++];
  }
  if (mode == RecordMode::kTwoReference) {
    if (*offset == record_size) {
      SetError("Truncated sparse second reference selector.", error);
      return false;
    }
    metadata->reference2 = record[(*offset)++];
    if (metadata->reference1 == metadata->reference2) {
      SetError("Sparse record repeats an anchor selector.", error);
      return false;
    }
  }
  return true;
}

}  // namespace

bool IsAlternateRecord(uint8_t flags) {
  return flags & (kRawPackedRecordFlag | kSparsePredictorRecordFlag);
}

bool EncodeRawRecord(const uint64_t* target, uint32_t sample_ct,
                     std::vector<uint8_t>* record, std::string* error) {
  if ((!target) || (!sample_ct) || (!record)) {
    SetError("Invalid raw-record encoder arguments.", error);
    return false;
  }
  record->clear();
  const size_t payload_byte_ct =
      (static_cast<size_t>(sample_ct) + 3) / 4;
  record->push_back(kRawPackedRecordFlag);
#if defined(_WIN32) || \
    (defined(__BYTE_ORDER__) && defined(__ORDER_LITTLE_ENDIAN__) && \
     (__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__))
  record->resize(1 + payload_byte_ct);
  memcpy(record->data() + 1, target, payload_byte_ct);
  if (sample_ct % 4) {
    record->back() &= static_cast<uint8_t>(
        (1U << (2 * (sample_ct % 4))) - 1);
  }
#else
  record->reserve(1 + payload_byte_ct);
  uint8_t packed_byte = 0;
  for (uint32_t sample_idx = 0; sample_idx != sample_ct; ++sample_idx) {
    packed_byte |= static_cast<uint8_t>(
        GetPackedGenotype(target, sample_idx)
        << (2 * (sample_idx % 4)));
    if ((sample_idx % 4 == 3) || (sample_idx + 1 == sample_ct)) {
      record->push_back(packed_byte);
      packed_byte = 0;
    }
  }
#endif
  return true;
}

static bool EncodeSparsePredictorRecordImpl(
    const uint64_t* target, const uint64_t* reference1,
    const uint64_t* reference2, uint32_t sample_ct, RecordMode mode,
    uint8_t reference1_idx, uint8_t reference2_idx,
    const uint32_t* context_symbol_counts,
    std::vector<uint8_t>* record, std::string* error) {
  if ((!target) || (!sample_ct) || (!record) ||
      ((mode != RecordMode::kMarginal) && (!reference1)) ||
      ((mode == RecordMode::kTwoReference) &&
       ((!reference2) || (reference1_idx == reference2_idx)))) {
    SetError("Invalid sparse-record encoder arguments.", error);
    return false;
  }
  const uint32_t row_ct = RowCt(mode);
  std::array<std::array<uint32_t, 4>, 16> counts = {};
  if (context_symbol_counts) {
    uint64_t counted_sample_ct = 0;
    for (uint32_t context = 0; context != row_ct; ++context) {
      for (uint32_t symbol = 0; symbol != 4; ++symbol) {
        const uint32_t count =
            context_symbol_counts[4 * context + symbol];
        counts[context][symbol] = count;
        counted_sample_ct += count;
      }
    }
    if (counted_sample_ct != sample_ct) {
      SetError("Supplied sparse model counts do not match sample count.",
               error);
      return false;
    }
  } else {
    for (uint32_t sample_idx = 0; sample_idx != sample_ct; ++sample_idx) {
      const uint32_t context =
          ContextIndex(mode, reference1, reference2, sample_idx);
      ++counts[context][GetPackedGenotype(target, sample_idx)];
    }
  }
  std::array<uint8_t, 16> predictions = {};
  uint32_t expected_exception_ct = 0;
  for (uint32_t context = 0; context != row_ct; ++context) {
    uint32_t row_total = counts[context][0];
    uint32_t best_count = counts[context][0];
    for (uint32_t symbol = 1; symbol != 4; ++symbol) {
      row_total += counts[context][symbol];
      if (counts[context][symbol] > best_count) {
        predictions[context] = static_cast<uint8_t>(symbol);
        best_count = counts[context][symbol];
      }
    }
    expected_exception_ct += row_total - best_count;
  }

  std::vector<uint32_t> exception_ids;
  std::vector<uint8_t> exception_values;
  exception_ids.reserve(expected_exception_ct);
  exception_values.reserve(expected_exception_ct);
  for (uint32_t sample_idx = 0; sample_idx != sample_ct; ++sample_idx) {
    const uint32_t context =
        ContextIndex(mode, reference1, reference2, sample_idx);
    const uint8_t value = GetPackedGenotype(target, sample_idx);
    if (value != predictions[context]) {
      exception_ids.push_back(sample_idx);
      exception_values.push_back(value);
    }
  }
  const size_t bitmap_byte_ct =
      (static_cast<size_t>(sample_ct) + 7) / 8;
  const bool use_bitmap =
      bitmap_byte_ct < DeltaIdByteCt(exception_ids);
  uint8_t flags =
      static_cast<uint8_t>(mode) | kSparsePredictorRecordFlag;
  if (use_bitmap) {
    flags |= kSparsePredictorBitmapFlag;
  }
  record->clear();
  record->push_back(flags);
  if (mode != RecordMode::kMarginal) {
    record->push_back(reference1_idx);
  }
  if (mode == RecordMode::kTwoReference) {
    record->push_back(reference2_idx);
  }
  const size_t mapping_offset = record->size();
  const size_t mapping_byte_ct = (row_ct + 3) / 4;
  record->resize(mapping_offset + mapping_byte_ct, 0);
  for (uint32_t context = 0; context != row_ct; ++context) {
    (*record)[mapping_offset + context / 4] |=
        static_cast<uint8_t>(
            predictions[context] << (2 * (context % 4)));
  }
  AppendVarint(static_cast<uint32_t>(exception_ids.size()), record);
  if (use_bitmap) {
    const size_t bitmap_offset = record->size();
    record->resize(bitmap_offset + bitmap_byte_ct, 0);
    for (const uint32_t sample_id : exception_ids) {
      (*record)[bitmap_offset + sample_id / 8] |=
          static_cast<uint8_t>(1U << (sample_id % 8));
    }
  } else {
    AppendDeltaIds(exception_ids, record);
  }
  const size_t value_offset = record->size();
  record->resize(
      value_offset + (exception_values.size() + 3) / 4, 0);
  for (size_t idx = 0; idx != exception_values.size(); ++idx) {
    (*record)[value_offset + idx / 4] |=
        static_cast<uint8_t>(
            exception_values[idx] << (2 * (idx % 4)));
  }
  return true;
}

bool EncodeSparsePredictorRecord(
    const uint64_t* target, const uint64_t* reference1,
    const uint64_t* reference2, uint32_t sample_ct, RecordMode mode,
    uint8_t reference1_idx, uint8_t reference2_idx,
    std::vector<uint8_t>* record, std::string* error) {
  return EncodeSparsePredictorRecordImpl(
      target, reference1, reference2, sample_ct, mode, reference1_idx,
      reference2_idx, nullptr, record, error);
}

bool EncodeSparsePredictorRecordFromCounts(
    const uint64_t* target, const uint64_t* reference1,
    const uint64_t* reference2, uint32_t sample_ct, RecordMode mode,
    uint8_t reference1_idx, uint8_t reference2_idx,
    const uint32_t* context_symbol_counts, std::vector<uint8_t>* record,
    std::string* error) {
  if (!context_symbol_counts) {
    SetError("Missing supplied sparse model counts.", error);
    return false;
  }
  return EncodeSparsePredictorRecordImpl(
      target, reference1, reference2, sample_ct, mode, reference1_idx,
      reference2_idx, context_symbol_counts, record, error);
}

bool DecodeAlternateRecordToBuffer(
    const uint8_t* record, size_t record_size,
    const uint64_t* const* anchors, uint32_t anchor_ct,
    uint32_t sample_ct, uint64_t* target, size_t target_word_ct,
    RecordMetadata* metadata, std::string* error) {
  if ((!record) || (!record_size) || (!sample_ct) || (!target) ||
      (target_word_ct < PackedWordCt(sample_ct))) {
    SetError("Invalid alternate-record decoder arguments.", error);
    return false;
  }
  const uint8_t flags = record[0];
  const bool is_raw = flags & kRawPackedRecordFlag;
  const bool is_sparse = flags & kSparsePredictorRecordFlag;
  if ((!is_raw && !is_sparse) || (is_raw && is_sparse) ||
      ((flags & kSparsePredictorBitmapFlag) && !is_sparse) ||
      (flags & kEntropyPayloadFlag)) {
    SetError("Invalid alternate-record flag combination.", error);
    return false;
  }
  RecordMode mode;
  if (!ParseMode(flags, &mode, error)) {
    return false;
  }
  RecordMetadata parsed_metadata;
  parsed_metadata.has_multiallelic_patches =
      flags & kMultiallelicPatchFlag;
  parsed_metadata.is_raw_packed = is_raw;
  parsed_metadata.is_sparse_predictor = is_sparse;
  size_t offset = 1;
  if (!ParseSelectors(record, record_size, mode, &offset,
                      &parsed_metadata, error)) {
    return false;
  }
  memset(target, 0, PackedWordCt(sample_ct) * sizeof(uint64_t));
  if (is_raw) {
    if (mode != RecordMode::kMarginal) {
      SetError("Raw record has a reference selector.", error);
      return false;
    }
    const size_t packed_byte_ct =
        (static_cast<size_t>(sample_ct) + 3) / 4;
    if ((offset > record_size) ||
        (record_size - offset != packed_byte_ct)) {
      SetError("Raw record has an invalid payload length.", error);
      return false;
    }
    if ((sample_ct % 4) &&
        (record[record_size - 1] >> (2 * (sample_ct % 4)))) {
      SetError("Raw record has nonzero trailing nyps.", error);
      return false;
    }
    memcpy(target, record + offset, packed_byte_ct);
    if (metadata) {
      *metadata = parsed_metadata;
    }
    return true;
  }

  const uint64_t* reference1 = nullptr;
  const uint64_t* reference2 = nullptr;
  if (mode != RecordMode::kMarginal) {
    if ((!anchors) || (parsed_metadata.reference1 >= anchor_ct) ||
        (!(reference1 = anchors[parsed_metadata.reference1]))) {
      SetError("Sparse first reference is unavailable.", error);
      return false;
    }
  }
  if (mode == RecordMode::kTwoReference) {
    if ((parsed_metadata.reference2 >= anchor_ct) ||
        (!(reference2 = anchors[parsed_metadata.reference2]))) {
      SetError("Sparse second reference is unavailable.", error);
      return false;
    }
  }
  const uint32_t row_ct = RowCt(mode);
  const size_t mapping_byte_ct = (row_ct + 3) / 4;
  if ((offset > record_size) ||
      (mapping_byte_ct > record_size - offset)) {
    SetError("Truncated sparse predictor mapping.", error);
    return false;
  }
  const uint8_t* const mapping = record + offset;
  offset += mapping_byte_ct;
  if ((row_ct % 4) &&
      (mapping[mapping_byte_ct - 1] >> (2 * (row_ct % 4)))) {
    SetError("Sparse predictor mapping has nonzero padding.", error);
    return false;
  }
  std::array<uint8_t, 16> predictions = {};
  for (uint32_t context = 0; context != row_ct; ++context) {
    predictions[context] = static_cast<uint8_t>(
        (mapping[context / 4] >> (2 * (context % 4))) & 3U);
  }
  const uint32_t packed_word_ct = PackedWordCt(sample_ct);
  for (uint32_t word_idx = 0; word_idx != packed_word_ct; ++word_idx) {
    uint64_t predicted_word = 0;
    if (mode == RecordMode::kMarginal) {
      predicted_word = ExpandContextMask(
          0x5555555555555555ULL, predictions[0]);
    } else if (mode == RecordMode::kOneReference) {
      uint64_t masks[4];
      BuildGenotypeMasks(reference1[word_idx], masks);
      for (uint32_t context = 0; context != 4; ++context) {
        predicted_word |=
            ExpandContextMask(masks[context], predictions[context]);
      }
    } else {
      uint64_t masks1[4];
      uint64_t masks2[4];
      BuildGenotypeMasks(reference1[word_idx], masks1);
      BuildGenotypeMasks(reference2[word_idx], masks2);
      for (uint32_t context1 = 0; context1 != 4; ++context1) {
        for (uint32_t context2 = 0; context2 != 4; ++context2) {
          const uint32_t context = 4 * context1 + context2;
          predicted_word |= ExpandContextMask(
              masks1[context1] & masks2[context2],
              predictions[context]);
        }
      }
    }
    if ((word_idx + 1 == packed_word_ct) && (sample_ct % 32)) {
      predicted_word &=
          (1ULL << (2 * (sample_ct % 32))) - 1;
    }
    target[word_idx] = predicted_word;
  }
  uint32_t exception_ct;
  if ((!ReadVarint(record, record_size, &offset, &exception_ct)) ||
      (exception_ct > sample_ct)) {
    SetError("Invalid sparse exception count.", error);
    return false;
  }
  std::vector<uint32_t> exception_ids(exception_ct);
  if (flags & kSparsePredictorBitmapFlag) {
    const size_t bitmap_byte_ct =
        (static_cast<size_t>(sample_ct) + 7) / 8;
    if ((offset > record_size) ||
        (bitmap_byte_ct > record_size - offset)) {
      SetError("Truncated sparse exception bitmap.", error);
      return false;
    }
    const uint8_t* const bitmap = record + offset;
    if ((sample_ct % 8) &&
        (bitmap[bitmap_byte_ct - 1] >> (sample_ct % 8))) {
      SetError("Sparse exception bitmap has nonzero padding.", error);
      return false;
    }
    uint32_t observed_ct = 0;
    for (uint32_t sample_idx = 0; sample_idx != sample_ct; ++sample_idx) {
      if (bitmap[sample_idx / 8] & (1U << (sample_idx % 8))) {
        if (observed_ct == exception_ct) {
          SetError("Sparse exception bitmap count mismatch.", error);
          return false;
        }
        exception_ids[observed_ct++] = sample_idx;
      }
    }
    if (observed_ct != exception_ct) {
      SetError("Sparse exception bitmap count mismatch.", error);
      return false;
    }
    offset += bitmap_byte_ct;
  } else {
    uint32_t previous = 0;
    for (uint32_t idx = 0; idx != exception_ct; ++idx) {
      uint32_t delta;
      if ((!ReadVarint(record, record_size, &offset, &delta)) ||
          (idx && !delta) ||
          (delta > std::numeric_limits<uint32_t>::max() - previous) ||
          (previous + delta >= sample_ct)) {
        SetError("Invalid sparse exception sample index.", error);
        return false;
      }
      previous += delta;
      exception_ids[idx] = previous;
    }
  }
  const size_t value_byte_ct =
      (static_cast<size_t>(exception_ct) + 3) / 4;
  if ((offset > record_size) ||
      (record_size - offset != value_byte_ct)) {
    SetError("Sparse exception values have an invalid length.", error);
    return false;
  }
  if ((exception_ct % 4) && value_byte_ct &&
      (record[record_size - 1] >> (2 * (exception_ct % 4)))) {
    SetError("Sparse exception values have nonzero padding.", error);
    return false;
  }
  for (uint32_t idx = 0; idx != exception_ct; ++idx) {
    const uint8_t value = static_cast<uint8_t>(
        (record[offset + idx / 4] >> (2 * (idx % 4))) & 3U);
    const uint32_t sample_idx = exception_ids[idx];
    if (GetPackedGenotype(target, sample_idx) == value) {
      SetError("Sparse exception repeats its predictor.", error);
      return false;
    }
    SetPackedGenotype(target, sample_idx, value);
  }
  if (metadata) {
    *metadata = parsed_metadata;
  }
  return true;
}

}  // namespace pgen_rans
