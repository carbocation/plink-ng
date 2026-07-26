// SPDX-License-Identifier: GPL-3.0-or-later

#include "pgen_rans_codec.h"

#include "pgen_rans_hybrid.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>

#if defined(__x86_64__) && \
    (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#define PGEN_RANS_X86_RUNTIME_DISPATCH 1
#define PGEN_RANS_TARGET_AVX512 \
  __attribute__((target( \
      "avx512f,avx512dq,avx512bw,avx512vl,bmi2")))
#else
#define PGEN_RANS_X86_RUNTIME_DISPATCH 0
#define PGEN_RANS_TARGET_AVX512
#endif

namespace pgen_rans {
namespace {

constexpr uint32_t kRansLowerBound = 1U << 23;
constexpr uint8_t kModeMask = 3;
constexpr uint8_t kEntropyPayloadFlag = 1U << 2;
constexpr uint8_t kMultiallelicPatchFlag = 1U << 3;
constexpr uint8_t kUnpaddedPayloadFlag = 1U << 4;
constexpr uint8_t kKnownFlagMask =
    kModeMask | kEntropyPayloadFlag | kMultiallelicPatchFlag |
    kUnpaddedPayloadFlag | kAlternateRecordFlagMask;
constexpr uint32_t kInterleavedPaddingByteCt = 15;
constexpr uint32_t kDefaultScaleBits = 12;
constexpr uint8_t kMultiallelicPatchVersion = 1;
constexpr uint8_t kPatch01BitmapFlag = 1U << 0;
constexpr uint8_t kPatch10BitmapFlag = 1U << 1;
constexpr uint8_t kKnownPatchFlagMask =
    kPatch01BitmapFlag | kPatch10BitmapFlag;
constexpr size_t kMultiallelicPatchHeaderByteCt = 11;
constexpr size_t kMultiallelicPatchFooterByteCt = 4;
#if PGEN_RANS_X86_RUNTIME_DISPATCH
constexpr uint32_t kDefaultSlotCt = 1U << kDefaultScaleBits;
constexpr uint32_t kAvx512MinimumSampleCt = 32768;
#endif

#if defined(_MSC_VER)
#define PGEN_RANS_ALWAYS_INLINE __forceinline
#define PGEN_RANS_NOINLINE __declspec(noinline)
#elif defined(__GNUC__) || defined(__clang__)
#define PGEN_RANS_ALWAYS_INLINE inline __attribute__((always_inline))
#define PGEN_RANS_NOINLINE __attribute__((noinline))
#else
#define PGEN_RANS_ALWAYS_INLINE inline
#define PGEN_RANS_NOINLINE
#endif

struct ModelRow {
  std::array<uint32_t, 4> frequencies = {};
  std::array<uint32_t, 4> cumulative = {};
};

struct Model {
  std::array<ModelRow, 16> rows;
  std::array<uint8_t, 16> symbol_masks = {};
  std::array<uint8_t, 16> deterministic_symbols = {};
  std::array<uint8_t, 16> active_symbol_cts = {};
  uint16_t context_mask = 0;
  uint16_t entropy_context_mask = 0;
  uint32_t row_ct = 0;
  bool has_entropy = false;
};

void SetError(const char* message, std::string* error) {
  if (error) {
    *error = message;
  }
}

void AppendU16(uint16_t value, std::vector<uint8_t>* output) {
  output->push_back(static_cast<uint8_t>(value));
  output->push_back(static_cast<uint8_t>(value >> 8));
}

void AppendU32(uint32_t value, std::vector<uint8_t>* output) {
  output->push_back(static_cast<uint8_t>(value));
  output->push_back(static_cast<uint8_t>(value >> 8));
  output->push_back(static_cast<uint8_t>(value >> 16));
  output->push_back(static_cast<uint8_t>(value >> 24));
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

uint32_t CeilLog2(uint32_t value_ct) {
  uint32_t result = 0;
  if (value_ct) {
    --value_ct;
  }
  while (value_ct) {
    ++result;
    value_ct >>= 1;
  }
  return result;
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

void AppendPackedValue(uint32_t value, uint32_t bit_width,
                       uint64_t* bit_buffer, uint32_t* bit_ct,
                       std::vector<uint8_t>* output) {
  if (!bit_width) {
    return;
  }
  *bit_buffer |= static_cast<uint64_t>(value) << *bit_ct;
  *bit_ct += bit_width;
  while (*bit_ct >= 8) {
    output->push_back(static_cast<uint8_t>(*bit_buffer));
    *bit_buffer >>= 8;
    *bit_ct -= 8;
  }
}

void FlushPackedValues(uint64_t* bit_buffer, uint32_t* bit_ct,
                       std::vector<uint8_t>* output) {
  if (*bit_ct) {
    output->push_back(static_cast<uint8_t>(*bit_buffer));
    *bit_buffer = 0;
    *bit_ct = 0;
  }
}

bool ReadPackedValue(const uint8_t* input, size_t input_size,
                     uint32_t bit_width, size_t* offset,
                     uint64_t* bit_buffer, uint32_t* bit_ct,
                     uint32_t* value) {
  if (!bit_width) {
    *value = 0;
    return true;
  }
  while (*bit_ct < bit_width) {
    if (*offset == input_size) {
      return false;
    }
    *bit_buffer |=
        static_cast<uint64_t>(input[(*offset)++]) << *bit_ct;
    *bit_ct += 8;
  }
  *value = static_cast<uint32_t>(
      *bit_buffer & ((1U << bit_width) - 1));
  *bit_buffer >>= bit_width;
  *bit_ct -= bit_width;
  return true;
}

bool ValidatePatchIds(const std::vector<uint32_t>& sample_ids,
                      uint32_t sample_ct, std::string* error) {
  for (size_t idx = 0; idx != sample_ids.size(); ++idx) {
    if (sample_ids[idx] >= sample_ct) {
      SetError("Multiallelic patch sample index is out of range.", error);
      return false;
    }
    if (idx && (sample_ids[idx] <= sample_ids[idx - 1])) {
      SetError("Multiallelic patch sample indices are not increasing.",
               error);
      return false;
    }
  }
  return true;
}

bool PatchIdSetsAreDisjoint(
    const std::vector<uint32_t>& first,
    const std::vector<uint32_t>& second) {
  size_t first_idx = 0;
  size_t second_idx = 0;
  while ((first_idx != first.size()) &&
         (second_idx != second.size())) {
    if (first[first_idx] == second[second_idx]) {
      return false;
    }
    if (first[first_idx] < second[second_idx]) {
      ++first_idx;
    } else {
      ++second_idx;
    }
  }
  return true;
}

size_t DeltaPatchIdByteCt(
    const std::vector<uint32_t>& sample_ids) {
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

bool PatchBitmapIsSmaller(
    const std::vector<uint32_t>& sample_ids, uint32_t sample_ct) {
  return (static_cast<size_t>(sample_ct) + 7) / 8 <
         DeltaPatchIdByteCt(sample_ids);
}

void AppendPatchIds(const std::vector<uint32_t>& sample_ids,
                    uint32_t sample_ct, bool use_bitmap,
                    std::vector<uint8_t>* output) {
  if (use_bitmap) {
    const size_t bitmap_offset = output->size();
    output->resize(
        bitmap_offset + (static_cast<size_t>(sample_ct) + 7) / 8, 0);
    for (const uint32_t sample_id : sample_ids) {
      (*output)[bitmap_offset + sample_id / 8] |=
          static_cast<uint8_t>(1U << (sample_id % 8));
    }
    return;
  }
  uint32_t previous = 0;
  for (size_t idx = 0; idx != sample_ids.size(); ++idx) {
    const uint32_t delta =
        idx ? sample_ids[idx] - previous : sample_ids[idx];
    AppendVarint(delta, output);
    previous = sample_ids[idx];
  }
}

bool ReadPatchIds(const uint8_t* input, size_t input_size,
                  uint32_t sample_ct, uint32_t id_ct,
                  bool use_bitmap, size_t* offset,
                  std::vector<uint32_t>* sample_ids,
                  std::string* error) {
  if (use_bitmap) {
    const size_t bitmap_byte_ct =
        (static_cast<size_t>(sample_ct) + 7) / 8;
    if ((*offset > input_size) ||
        (bitmap_byte_ct > input_size - *offset)) {
      SetError("Truncated multiallelic patch bitmap.", error);
      return false;
    }
    const uint8_t* const bitmap = input + *offset;
    if ((sample_ct % 8) &&
        (bitmap[bitmap_byte_ct - 1] >>
         (sample_ct % 8))) {
      SetError("Nonzero multiallelic patch bitmap padding bits.", error);
      return false;
    }
    sample_ids->clear();
    sample_ids->reserve(id_ct);
    for (uint32_t sample_idx = 0; sample_idx != sample_ct;
         ++sample_idx) {
      if (bitmap[sample_idx / 8] &
          (1U << (sample_idx % 8))) {
        sample_ids->push_back(sample_idx);
      }
    }
    *offset += bitmap_byte_ct;
    if (sample_ids->size() != id_ct) {
      SetError("Multiallelic patch bitmap count mismatch.", error);
      return false;
    }
    return true;
  }
  sample_ids->resize(id_ct);
  uint32_t previous = 0;
  for (uint32_t idx = 0; idx != id_ct; ++idx) {
    uint32_t delta;
    if (!ReadVarint(input, input_size, offset, &delta)) {
      SetError("Invalid or truncated multiallelic patch sample index.",
               error);
      return false;
    }
    if (idx && (!delta)) {
      SetError("Multiallelic patch sample indices are not increasing.",
               error);
      return false;
    }
    if ((delta > std::numeric_limits<uint32_t>::max() - previous) ||
        (previous + delta >= sample_ct)) {
      SetError("Multiallelic patch sample index is out of range.", error);
      return false;
    }
    previous += delta;
    (*sample_ids)[idx] = previous;
  }
  return true;
}

uint32_t RowCt(RecordMode mode) {
  if (mode == RecordMode::kMarginal) {
    return 1;
  }
  if (mode == RecordMode::kOneReference) {
    return 4;
  }
  return 16;
}

uint32_t ContextIndex(RecordMode mode, const uint64_t* reference1,
                      const uint64_t* reference2, uint32_t sample_idx) {
  if (mode == RecordMode::kMarginal) {
    return 0;
  }
  const uint32_t first = GetPackedGenotype(reference1, sample_idx);
  if (mode == RecordMode::kOneReference) {
    return first;
  }
  return 4 * first + GetPackedGenotype(reference2, sample_idx);
}

bool NormalizeRow(const uint32_t* counts, uint32_t scale_bits,
                  uint32_t context, Model* model, std::string* error) {
  ModelRow& row = model->rows[context];
  uint8_t& symbol_mask = model->symbol_masks[context];
  uint8_t& deterministic_symbol =
      model->deterministic_symbols[context];
  uint8_t& active_symbol_ct = model->active_symbol_cts[context];
  const uint32_t total_frequency = 1U << scale_bits;
  uint64_t total_count = 0;
  std::array<double, 4> raw_frequencies = {};
  std::array<uint32_t, 4> normalized = {};
  for (uint32_t symbol = 0; symbol != 4; ++symbol) {
    total_count += counts[symbol];
    if (counts[symbol]) {
      symbol_mask |= static_cast<uint8_t>(1U << symbol);
      deterministic_symbol = static_cast<uint8_t>(symbol);
      ++active_symbol_ct;
    }
  }
  if (!total_count) {
    SetError("Cannot normalize an empty model row.", error);
    return false;
  }
  if (active_symbol_ct == 1) {
    row.frequencies[deterministic_symbol] = total_frequency;
    return true;
  }
  uint32_t frequency_sum = 0;
  for (uint32_t symbol = 0; symbol != 4; ++symbol) {
    if (!counts[symbol]) {
      continue;
    }
    raw_frequencies[symbol] =
        static_cast<double>(counts[symbol]) * total_frequency /
        static_cast<double>(total_count);
    normalized[symbol] =
        std::max(1U, static_cast<uint32_t>(raw_frequencies[symbol]));
    frequency_sum += normalized[symbol];
  }
  while (frequency_sum < total_frequency) {
    uint32_t best_symbol = 0;
    double best_score = -std::numeric_limits<double>::infinity();
    for (uint32_t symbol = 0; symbol != 4; ++symbol) {
      if (!counts[symbol]) {
        continue;
      }
      const double score = raw_frequencies[symbol] - normalized[symbol];
      if (score > best_score) {
        best_score = score;
        best_symbol = symbol;
      }
    }
    ++normalized[best_symbol];
    ++frequency_sum;
  }
  while (frequency_sum > total_frequency) {
    uint32_t best_symbol = 0;
    double best_score = -std::numeric_limits<double>::infinity();
    for (uint32_t symbol = 0; symbol != 4; ++symbol) {
      if (normalized[symbol] <= 1) {
        continue;
      }
      const double score = normalized[symbol] - raw_frequencies[symbol];
      if (score > best_score) {
        best_score = score;
        best_symbol = symbol;
      }
    }
    --normalized[best_symbol];
    --frequency_sum;
  }
  uint32_t cumulative = 0;
  for (uint32_t symbol = 0; symbol != 4; ++symbol) {
    row.frequencies[symbol] = normalized[symbol];
    row.cumulative[symbol] = cumulative;
    cumulative += normalized[symbol];
  }
  if (cumulative != total_frequency) {
    SetError("Internal normalized-frequency sum mismatch.", error);
    return false;
  }
  return true;
}

bool BuildModelFromCounts(const uint32_t* counts, RecordMode mode,
                          uint32_t scale_bits, Model* model,
                          std::string* error) {
  model->row_ct = RowCt(mode);
  for (uint32_t context = 0; context != model->row_ct; ++context) {
    uint32_t row_total = 0;
    for (uint32_t symbol = 0; symbol != 4; ++symbol) {
      row_total += counts[4 * context + symbol];
    }
    if (!row_total) {
      continue;
    }
    model->context_mask |= static_cast<uint16_t>(1U << context);
    if (!NormalizeRow(&(counts[4 * context]), scale_bits, context,
                      model, error)) {
      return false;
    }
    if (model->active_symbol_cts[context] > 1) {
      model->entropy_context_mask |=
          static_cast<uint16_t>(1U << context);
      model->has_entropy = true;
    }
  }
  return true;
}

bool BuildModel(const uint64_t* target, const uint64_t* reference1,
                const uint64_t* reference2, uint32_t sample_ct,
                RecordMode mode, uint32_t scale_bits, Model* model,
                std::string* error) {
  std::array<uint32_t, 64> counts = {};
  for (uint32_t sample_idx = 0; sample_idx != sample_ct; ++sample_idx) {
    const uint32_t context =
        ContextIndex(mode, reference1, reference2, sample_idx);
    const uint32_t target_genotype =
        GetPackedGenotype(target, sample_idx);
    ++counts[4 * context + target_genotype];
  }
  return BuildModelFromCounts(counts.data(), mode, scale_bits, model, error);
}

void SerializeModel(const Model& model, RecordMode mode,
                    std::vector<uint8_t>* output) {
  if (mode == RecordMode::kOneReference) {
    output->push_back(static_cast<uint8_t>(model.context_mask));
  } else if (mode == RecordMode::kTwoReference) {
    AppendU16(model.context_mask, output);
  }
  for (uint32_t context = 0; context != model.row_ct; ++context) {
    if (!(model.context_mask & (1U << context))) {
      continue;
    }
    const ModelRow& row = model.rows[context];
    const uint8_t symbol_mask = model.symbol_masks[context];
    output->push_back(symbol_mask);
    uint32_t remaining = model.active_symbol_cts[context];
    for (uint32_t symbol = 0; symbol != 4; ++symbol) {
      if (!(symbol_mask & (1U << symbol))) {
        continue;
      }
      --remaining;
      if (!remaining) {
        break;
      }
      AppendU16(static_cast<uint16_t>(row.frequencies[symbol]), output);
    }
  }
}

bool ParseModel(const uint8_t* input, size_t input_size, RecordMode mode,
                uint32_t scale_bits, size_t* offset, Model* model,
                std::string* error) {
  model->row_ct = RowCt(mode);
  if (mode == RecordMode::kMarginal) {
    model->context_mask = 1;
  } else if (mode == RecordMode::kOneReference) {
    if (*offset == input_size) {
      SetError("Truncated one-reference context mask.", error);
      return false;
    }
    model->context_mask = input[(*offset)++];
    if (model->context_mask & 0xfff0U) {
      SetError("Invalid one-reference context mask.", error);
      return false;
    }
  } else {
    if (!ReadU16(input, input_size, offset, &(model->context_mask))) {
      SetError("Truncated two-reference context mask.", error);
      return false;
    }
  }
  if (!model->context_mask) {
    SetError("Model has no populated contexts.", error);
    return false;
  }
  const uint32_t total_frequency = 1U << scale_bits;
  for (uint32_t context = 0; context != model->row_ct; ++context) {
    if (!(model->context_mask & (1U << context))) {
      continue;
    }
    if (*offset == input_size) {
      SetError("Truncated model symbol mask.", error);
      return false;
    }
    ModelRow& row = model->rows[context];
    uint8_t& symbol_mask = model->symbol_masks[context];
    uint8_t& deterministic_symbol =
        model->deterministic_symbols[context];
    uint8_t& active_symbol_ct =
        model->active_symbol_cts[context];
    symbol_mask = input[(*offset)++];
    if ((!symbol_mask) || (symbol_mask & 0xf0U)) {
      SetError("Invalid model symbol mask.", error);
      return false;
    }
    for (uint32_t symbol = 0; symbol != 4; ++symbol) {
      if (symbol_mask & (1U << symbol)) {
        deterministic_symbol = static_cast<uint8_t>(symbol);
        ++active_symbol_ct;
      }
    }
    uint32_t frequency_sum = 0;
    uint32_t remaining = active_symbol_ct;
    for (uint32_t symbol = 0; symbol != 4; ++symbol) {
      if (!(symbol_mask & (1U << symbol))) {
        row.cumulative[symbol] = frequency_sum;
        continue;
      }
      --remaining;
      uint32_t frequency;
      if (remaining) {
        uint16_t stored_frequency;
        if (!ReadU16(input, input_size, offset, &stored_frequency) ||
            (!stored_frequency)) {
          SetError("Invalid or truncated model frequency.", error);
          return false;
        }
        frequency = stored_frequency;
      } else {
        if (frequency_sum >= total_frequency) {
          SetError("Model frequencies exceed normalization total.", error);
          return false;
        }
        frequency = total_frequency - frequency_sum;
      }
      row.cumulative[symbol] = frequency_sum;
      row.frequencies[symbol] = frequency;
      frequency_sum += frequency;
    }
    if (frequency_sum != total_frequency) {
      SetError("Model frequencies do not sum to normalization total.", error);
      return false;
    }
    if (active_symbol_ct > 1) {
      model->entropy_context_mask |=
          static_cast<uint16_t>(1U << context);
      model->has_entropy = true;
    }
  }
  return true;
}

#if defined(__aarch64__) || defined(__arm64__) || defined(_M_ARM64)
#define PGEN_RANS_ARM64_ENCODER 1
#else
#define PGEN_RANS_ARM64_ENCODER 0
#endif

#if PGEN_RANS_ARM64_ENCODER
struct RansEncoderSymbol {
  uint64_t maximum_state = 0;
  uint32_t frequency = 0;
  uint32_t cumulative = 0;
};
static_assert(sizeof(RansEncoderSymbol) == 16);
#else
struct RansEncoderSymbol {
  uint32_t division_multiplier = 0;
  uint32_t maximum_state = 0;
  uint32_t frequency_complement = 0;
  uint16_t frequency = 0;
  uint16_t cumulative = 0;
  uint8_t division_pre_shift = 0;
  uint8_t division_post_shift = 0;
  uint8_t division_increment = 0;
};
static_assert(sizeof(RansEncoderSymbol) == 20);

// Runtime-constant unsigned division from "Labor of Division (Episode III)".
// Model construction pays the hardware divides once; the sample loop then
// uses one multiply and shift for both quotient and remainder.
void BuildDivisionMagic(uint32_t divisor, uint32_t* multiplier,
                        uint32_t* pre_shift, uint32_t* post_shift,
                        uint32_t* increment) {
  if (!(divisor & (divisor - 1))) {
    uint32_t shift = 0;
    while (divisor > 1) {
      divisor >>= 1;
      ++shift;
    }
    *multiplier = 1;
    *pre_shift = 0;
    *post_shift = shift;
    *increment = 0;
    return;
  }
  uint32_t quotient = 0x80000000U / divisor;
  uint32_t remainder = 0x80000000U - quotient * divisor;
  uint32_t ceil_log_2_d = 0;
  for (uint32_t value = divisor - 1; value; value >>= 1) {
    ++ceil_log_2_d;
  }
  uint32_t down_multiplier = 0;
  uint32_t down_exponent = 0;
  bool has_magic_down = false;
  for (uint32_t exponent = 0;; ++exponent) {
    if (remainder >= divisor - remainder) {
      quotient = quotient * 2 + 1;
      remainder = remainder * 2 - divisor;
    } else {
      quotient *= 2;
      remainder *= 2;
    }
    if ((exponent >= ceil_log_2_d) ||
        (divisor - remainder <= (1U << exponent))) {
      if (exponent < ceil_log_2_d) {
        *multiplier = quotient + 1;
        *pre_shift = 0;
        *post_shift = 32 + exponent;
        *increment = 0;
        return;
      }
      break;
    }
    if ((!has_magic_down) && (remainder <= (1U << exponent))) {
      has_magic_down = true;
      down_multiplier = quotient;
      down_exponent = exponent;
    }
  }
  if (divisor & 1) {
    *multiplier = down_multiplier;
    *pre_shift = 0;
    *post_shift = 32 + down_exponent;
    *increment = 1;
    return;
  }
  uint32_t shift = 0;
  uint32_t odd_divisor = divisor;
  while (!(odd_divisor & 1)) {
    odd_divisor >>= 1;
    ++shift;
  }
  uint32_t ignored_pre_shift;
  BuildDivisionMagic(
      odd_divisor, multiplier, &ignored_pre_shift, post_shift, increment);
  *pre_shift = shift;
}
#endif

RansEncoderSymbol BuildRansEncoderSymbol(
    uint32_t cumulative, uint32_t frequency, uint32_t scale_bits) {
  RansEncoderSymbol result;
#if PGEN_RANS_ARM64_ENCODER
  result.maximum_state =
      ((static_cast<uint64_t>(kRansLowerBound) >> scale_bits) << 8) *
      frequency;
  result.frequency = frequency;
  result.cumulative = cumulative;
#else
  result.maximum_state = static_cast<uint32_t>(
      ((static_cast<uint64_t>(kRansLowerBound) >> scale_bits) << 8) *
      frequency);
  result.frequency_complement =
      (1U << scale_bits) - frequency;
  result.frequency = static_cast<uint16_t>(frequency);
  result.cumulative = static_cast<uint16_t>(cumulative);
  uint32_t division_pre_shift;
  uint32_t division_post_shift;
  uint32_t division_increment;
  BuildDivisionMagic(
      frequency, &result.division_multiplier,
      &division_pre_shift, &division_post_shift, &division_increment);
  result.division_pre_shift =
      static_cast<uint8_t>(division_pre_shift);
  result.division_post_shift =
      static_cast<uint8_t>(division_post_shift);
  result.division_increment =
      static_cast<uint8_t>(division_increment);
#endif
  return result;
}

PGEN_RANS_ALWAYS_INLINE uint32_t RansEncodeSymbolToBytes(
    const RansEncoderSymbol& encoder, uint32_t scale_bits,
    uint32_t* state, uint8_t* output) {
  uint32_t output_byte_ct = 0;
  while (*state >= encoder.maximum_state) {
    output[output_byte_ct++] = static_cast<uint8_t>(*state);
    *state >>= 8;
  }
#if PGEN_RANS_ARM64_ENCODER
  const uint32_t quotient = *state / encoder.frequency;
  const uint32_t remainder =
      *state - quotient * encoder.frequency;
  *state =
      (quotient << scale_bits) + remainder + encoder.cumulative;
#else
  (void)scale_bits;
  const uint32_t quotient = static_cast<uint32_t>(
      (static_cast<uint64_t>(encoder.division_multiplier) *
       ((*state >> encoder.division_pre_shift) +
        encoder.division_increment)) >>
      encoder.division_post_shift);
  *state += quotient * encoder.frequency_complement +
            encoder.cumulative;
#endif
  return output_byte_ct;
}

constexpr int kRuntimeRecordMode = -1;

template <int kMode, uint32_t kStateCt>
bool BuildInterleavedPayloadDirectMode(
    const uint64_t* target, const uint64_t* reference1,
    const uint64_t* reference2, uint32_t sample_ct,
    RecordMode runtime_mode,
    const Model& model, uint32_t scale_bits,
    const std::array<std::array<RansEncoderSymbol, 4>, 16>& encoders,
    std::vector<uint32_t>* states, std::vector<uint8_t>* reverse_payload,
    std::string* error) {
  const RecordMode mode =
      (kMode == kRuntimeRecordMode)
          ? runtime_mode
          : static_cast<RecordMode>(kMode);
  const uint32_t state_ct =
      kStateCt ? kStateCt : static_cast<uint32_t>(states->size());
  reverse_payload->clear();
  reverse_payload->reserve(sample_ct / 8);
  uint32_t first_sample =
      ((sample_ct - 1) / state_ct) * state_ct;
  while (true) {
    const uint32_t round_sample_ct =
        std::min(state_ct, sample_ct - first_sample);
    uint64_t target_word = 0;
    uint64_t reference1_word = 0;
    uint64_t reference2_word = 0;
    const bool packed_word_round =
        kStateCt ? true : (state_ct == 32);
    const uint32_t word_sample_offset =
        (kStateCt == 16) ? (first_sample % 32) : 0;
    if (packed_word_round) {
      const uint32_t word_idx = first_sample / 32;
      target_word = target[word_idx];
      if (mode != RecordMode::kMarginal) {
        reference1_word = reference1[word_idx];
      }
      if (mode == RecordMode::kTwoReference) {
        reference2_word = reference2[word_idx];
      }
    }
    uint32_t group_start =
        ((round_sample_ct - 1) / 16) * 16;
    while (true) {
      const uint32_t group_end =
          std::min(group_start + 16, round_sample_ct);
      std::array<std::array<uint8_t, 4>, 16> emitted_bytes = {};
      std::array<uint8_t, 16> emitted_byte_cts = {};
      uint32_t maximum_emitted_byte_ct = 0;
      uint64_t target_remaining = 0;
      uint64_t reference1_remaining = 0;
      uint64_t reference2_remaining = 0;
      if constexpr ((kStateCt == 16) || (kStateCt == 32)) {
        const uint32_t group_shift =
            2 * (word_sample_offset + group_start);
        target_remaining = target_word >> group_shift;
        if (mode != RecordMode::kMarginal) {
          reference1_remaining = reference1_word >> group_shift;
        }
        if (mode == RecordMode::kTwoReference) {
          reference2_remaining = reference2_word >> group_shift;
        }
      }
      for (uint32_t lane = group_start; lane != group_end;
           ++lane) {
        const uint32_t group_lane = lane - group_start;
        const uint32_t sample_idx = first_sample + lane;
        uint32_t context;
        uint32_t symbol;
        if (packed_word_round) {
          if constexpr ((kStateCt == 16) || (kStateCt == 32)) {
            symbol = static_cast<uint32_t>(
                target_remaining & 3U);
            target_remaining >>= 2;
            if (mode == RecordMode::kMarginal) {
              context = 0;
            } else {
              const uint32_t first_reference =
                  static_cast<uint32_t>(
                      reference1_remaining & 3U);
              reference1_remaining >>= 2;
              if (mode == RecordMode::kOneReference) {
                context = first_reference;
              } else {
                context =
                    4 * first_reference +
                    static_cast<uint32_t>(
                        reference2_remaining & 3U);
                reference2_remaining >>= 2;
              }
            }
          } else {
            const uint32_t shift = 2 * lane;
            symbol = static_cast<uint32_t>(
                (target_word >> shift) & 3U);
            if (mode == RecordMode::kMarginal) {
              context = 0;
            } else {
              const uint32_t first_reference =
                  static_cast<uint32_t>(
                      (reference1_word >> shift) & 3U);
              context =
                  (mode == RecordMode::kOneReference)
                      ? first_reference
                      : (4 * first_reference +
                         static_cast<uint32_t>(
                             (reference2_word >> shift) & 3U));
            }
          }
        } else {
          context = ContextIndex(
              mode, reference1, reference2, sample_idx);
          symbol = GetPackedGenotype(target, sample_idx);
        }
        const uint32_t active_symbol_ct =
            model.active_symbol_cts[context];
        if (active_symbol_ct > 1) {
          if (!encoders[context][symbol].frequency) {
            SetError("Supplied rANS model counts do not match genotypes.",
                     error);
            return false;
          }
          const uint32_t emitted_byte_ct = RansEncodeSymbolToBytes(
              encoders[context][symbol], scale_bits, &((*states)[lane]),
              emitted_bytes[group_lane].data());
          emitted_byte_cts[group_lane] =
              static_cast<uint8_t>(emitted_byte_ct);
          maximum_emitted_byte_ct =
              std::max(maximum_emitted_byte_ct, emitted_byte_ct);
        } else if ((!active_symbol_ct) ||
                   (symbol != model.deterministic_symbols[context])) {
          SetError("Supplied rANS model counts do not match genotypes.",
                   error);
          return false;
        }
      }

      // The decoder consumes refill layers from low to high and lanes
      // from low to high.  We are traversing symbols in reverse, so append
      // the exact reverse of that order and reverse the entire payload once
      // when it is copied to the record.
      for (uint32_t layer_plus_one = maximum_emitted_byte_ct;
           layer_plus_one; --layer_plus_one) {
        const uint32_t decoder_layer = layer_plus_one - 1;
        for (uint32_t lane = group_end; lane != group_start;) {
          --lane;
          const uint32_t group_lane = lane - group_start;
          const uint32_t emitted_byte_ct =
              emitted_byte_cts[group_lane];
          if (emitted_byte_ct <= decoder_layer) {
            continue;
          }
          reverse_payload->push_back(
              emitted_bytes[group_lane][
                  emitted_byte_ct - 1 - decoder_layer]);
        }
      }
      if (!group_start) {
        break;
      }
      group_start -= 16;
    }
    if (!first_sample) {
      break;
    }
    first_sample -= state_ct;
  }
  return true;
}

template <RecordMode kMode>
bool BuildInterleavedPayloadDirectForMode(
    const uint64_t* target, const uint64_t* reference1,
    const uint64_t* reference2, uint32_t sample_ct,
    const Model& model, uint32_t scale_bits,
    const std::array<std::array<RansEncoderSymbol, 4>, 16>& encoders,
    std::vector<uint32_t>* states, std::vector<uint8_t>* reverse_payload,
    std::string* error) {
  if (states->size() == 32) {
    return BuildInterleavedPayloadDirectMode<
        static_cast<int>(kMode), 32>(
        target, reference1, reference2, sample_ct, kMode, model,
        scale_bits, encoders, states, reverse_payload, error);
  }
  if (states->size() == 16) {
    return BuildInterleavedPayloadDirectMode<
        static_cast<int>(kMode), 16>(
        target, reference1, reference2, sample_ct, kMode, model,
        scale_bits, encoders, states, reverse_payload, error);
  }
  return BuildInterleavedPayloadDirectMode<
      static_cast<int>(kMode), 0>(
      target, reference1, reference2, sample_ct, kMode, model,
      scale_bits, encoders, states, reverse_payload, error);
}

bool BuildInterleavedPayloadDirect(
    const uint64_t* target, const uint64_t* reference1,
    const uint64_t* reference2, uint32_t sample_ct,
    RecordMode mode, const Model& model, uint32_t scale_bits,
    const std::array<std::array<RansEncoderSymbol, 4>, 16>& encoders,
    std::vector<uint32_t>* states, std::vector<uint8_t>* reverse_payload,
    std::string* error) {
#if PGEN_RANS_ARM64_ENCODER
  switch (mode) {
    case RecordMode::kMarginal:
      return BuildInterleavedPayloadDirectForMode<
          RecordMode::kMarginal>(
          target, reference1, reference2, sample_ct, model, scale_bits,
          encoders, states, reverse_payload, error);
    case RecordMode::kOneReference:
      return BuildInterleavedPayloadDirectForMode<
          RecordMode::kOneReference>(
          target, reference1, reference2, sample_ct, model, scale_bits,
          encoders, states, reverse_payload, error);
    case RecordMode::kTwoReference:
      return BuildInterleavedPayloadDirectForMode<
          RecordMode::kTwoReference>(
          target, reference1, reference2, sample_ct, model, scale_bits,
          encoders, states, reverse_payload, error);
  }
  SetError("Invalid conditional-rANS record mode.", error);
  return false;
#else
  return BuildInterleavedPayloadDirectMode<
      kRuntimeRecordMode, 0>(
      target, reference1, reference2, sample_ct, mode, model,
      scale_bits, encoders, states, reverse_payload, error);
#endif
}

#if PGEN_RANS_X86_RUNTIME_DISPATCH
constexpr uint32_t kDecodeTableAbsentFlag = 1U << 30;
constexpr uint32_t kDecodeTableDeterministicFlag = 1U << 31;

struct alignas(64) Avx512DecodeModel12 {
  // The three cumulative boundaries for all 16 contexts fit in three
  // zmm registers, and context flags fit in one more.  The selected
  // interval's cumulative and frequency are derived directly from
  // adjacent boundaries, avoiding both a random gather and a second
  // symbol-indexed lookup.
  std::array<std::array<uint32_t, 16>, 3> thresholds = {};
  std::array<uint32_t, 16> context_flags = {};
};

bool HasAvx512Decoder() {
  static const bool result = []() {
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx512f") &&
           __builtin_cpu_supports("avx512dq") &&
           __builtin_cpu_supports("avx512bw") &&
           __builtin_cpu_supports("avx512vl") &&
           __builtin_cpu_supports("bmi2");
  }();
  return result;
}

void BuildAvx512DecodeModel12(
    const Model& model, Avx512DecodeModel12* decode_model) {
  for (uint32_t context = 0; context != 16; ++context) {
    const uint32_t active_symbol_ct =
        (context < model.row_ct)
            ? model.active_symbol_cts[context]
            : 0;
    if (!active_symbol_ct) {
      for (uint32_t boundary = 0; boundary != 3; ++boundary) {
        decode_model->thresholds[boundary][context] =
            kDefaultSlotCt;
      }
      decode_model->context_flags[context] =
          kDecodeTableAbsentFlag |
          kDecodeTableDeterministicFlag;
      continue;
    }
    if (active_symbol_ct == 1) {
      const uint32_t deterministic_symbol =
          model.deterministic_symbols[context];
      for (uint32_t boundary = 1; boundary != 4; ++boundary) {
        decode_model->thresholds[boundary - 1][context] =
            (deterministic_symbol >= boundary)
                ? 0
                : kDefaultSlotCt;
      }
      decode_model->context_flags[context] =
          kDecodeTableDeterministicFlag;
      continue;
    }
    const ModelRow& row = model.rows[context];
    for (uint32_t boundary = 1; boundary != 4; ++boundary) {
      decode_model->thresholds[boundary - 1][context] =
          row.cumulative[boundary];
    }
    decode_model->context_flags[context] = 0;
  }
}

template <RecordMode kMode, uint32_t kGroup, bool kAllEntropy>
PGEN_RANS_TARGET_AVX512 PGEN_RANS_ALWAYS_INLINE
bool RansDecodeGroup16Avx512(
    const Avx512DecodeModel12& decode_model,
    uint64_t reference1_word,
    uint64_t reference2_word,
    __m512i* state, const uint8_t** interleaved_iter,
    const uint8_t* interleaved_end,
    uint32_t* packed_symbols, uint32_t* absent_context_mask,
    std::string* error) {
  const __m512i shifts = _mm512_setr_epi32(
      0, 2, 4, 6, 8, 10, 12, 14,
      16, 18, 20, 22, 24, 26, 28, 30);
  __m512i contexts = _mm512_setzero_si512();
  if (kMode != RecordMode::kMarginal) {
    const uint32_t reference1_chunk = static_cast<uint32_t>(
        reference1_word >> (32 * kGroup));
    contexts = _mm512_and_si512(
        _mm512_srlv_epi32(
            _mm512_set1_epi32(
                static_cast<int>(reference1_chunk)),
            shifts),
        _mm512_set1_epi32(3));
  }
  if (kMode == RecordMode::kTwoReference) {
    const uint32_t reference2_chunk = static_cast<uint32_t>(
        reference2_word >> (32 * kGroup));
    const __m512i second_contexts = _mm512_and_si512(
        _mm512_srlv_epi32(
            _mm512_set1_epi32(
                static_cast<int>(reference2_chunk)),
            shifts),
        _mm512_set1_epi32(3));
    contexts = _mm512_add_epi32(
        _mm512_slli_epi32(contexts, 2), second_contexts);
  }

  const __m512i slots = _mm512_and_si512(
      *state, _mm512_set1_epi32(kDefaultSlotCt - 1));
  __m512i threshold0;
  __m512i threshold1;
  __m512i threshold2;
  if (kMode == RecordMode::kMarginal) {
    threshold0 = _mm512_set1_epi32(
        static_cast<int>(decode_model.thresholds[0][0]));
    threshold1 = _mm512_set1_epi32(
        static_cast<int>(decode_model.thresholds[1][0]));
    threshold2 = _mm512_set1_epi32(
        static_cast<int>(decode_model.thresholds[2][0]));
  } else {
    threshold0 = _mm512_permutexvar_epi32(
        contexts,
        _mm512_load_si512(
            decode_model.thresholds[0].data()));
    threshold1 = _mm512_permutexvar_epi32(
        contexts,
        _mm512_load_si512(
            decode_model.thresholds[1].data()));
    threshold2 = _mm512_permutexvar_epi32(
        contexts,
        _mm512_load_si512(
            decode_model.thresholds[2].data()));
  }
  const __mmask16 above0 = _mm512_cmp_epu32_mask(
      slots, threshold0, _MM_CMPINT_GE);
  const __mmask16 above1 = _mm512_cmp_epu32_mask(
      slots, threshold1, _MM_CMPINT_GE);
  const __mmask16 above2 = _mm512_cmp_epu32_mask(
      slots, threshold2, _MM_CMPINT_GE);
  __mmask16 entropy_mask = 0xffffU;
  if ((kMode != RecordMode::kMarginal) && !kAllEntropy) {
    const __m512i context_flags =
        _mm512_permutexvar_epi32(
            contexts,
            _mm512_load_si512(
                decode_model.context_flags.data()));
    *absent_context_mask |= static_cast<uint32_t>(
        _mm512_movepi32_mask(
            _mm512_slli_epi32(context_flags, 1)));
    const __mmask16 deterministic_mask =
        _mm512_movepi32_mask(context_flags);
    entropy_mask =
        static_cast<__mmask16>(~deterministic_mask);
  }
  __m512i cumulatives = _mm512_maskz_mov_epi32(
      above0, threshold0);
  cumulatives = _mm512_mask_mov_epi32(
      cumulatives, above1, threshold1);
  cumulatives = _mm512_mask_mov_epi32(
      cumulatives, above2, threshold2);
  __m512i upper_bounds = _mm512_mask_mov_epi32(
      threshold0, above0, threshold1);
  upper_bounds = _mm512_mask_mov_epi32(
      upper_bounds, above1, threshold2);
  upper_bounds = _mm512_mask_set1_epi32(
      upper_bounds, above2, kDefaultSlotCt);
  const __m512i frequencies =
      _mm512_sub_epi32(upper_bounds, cumulatives);
  const __m512i updated_states = _mm512_add_epi32(
      _mm512_mullo_epi32(
          frequencies,
          _mm512_srli_epi32(*state, kDefaultScaleBits)),
      _mm512_sub_epi32(slots, cumulatives));
  *state = _mm512_mask_mov_epi32(
      *state, entropy_mask, updated_states);

  __mmask16 renormalization_mask =
      entropy_mask &
      _mm512_cmp_epu32_mask(
          *state, _mm512_set1_epi32(kRansLowerBound),
          _MM_CMPINT_LT);
  while (renormalization_mask) {
    const uint32_t refill_byte_ct =
        static_cast<uint32_t>(__builtin_popcount(
            static_cast<uint32_t>(renormalization_mask)));
    if (static_cast<size_t>(
            interleaved_end - *interleaved_iter) <
        refill_byte_ct) {
      SetError("Truncated interleaved rANS payload.", error);
      return false;
    }
    const size_t remaining_byte_ct =
        static_cast<size_t>(interleaved_end - *interleaved_iter);
    const __m128i packed_bytes =
        (remaining_byte_ct >= 16)
            ? _mm_loadu_si128(
                  reinterpret_cast<const __m128i*>(
                      *interleaved_iter))
            : _mm_maskz_loadu_epi8(
                  static_cast<__mmask16>(
                      (1U << remaining_byte_ct) - 1),
                  *interleaved_iter);
    const __m512i refill_values =
        _mm512_cvtepu8_epi32(packed_bytes);
    const __m512i expanded_values =
        _mm512_maskz_expand_epi32(
            renormalization_mask, refill_values);
    const __m512i renormalized_states = _mm512_or_si512(
        _mm512_slli_epi32(*state, 8), expanded_values);
    *state = _mm512_mask_mov_epi32(
        *state, renormalization_mask,
        renormalized_states);
    *interleaved_iter += refill_byte_ct;
    renormalization_mask =
        entropy_mask &
        _mm512_cmp_epu32_mask(
            *state, _mm512_set1_epi32(kRansLowerBound),
            _MM_CMPINT_LT);
  }

  // With monotonic threshold masks, symbol = above0 + above1 +
  // above2.  Its high bit is therefore above1 and its low bit is the
  // parity of the three masks.  Pack those bits directly, without
  // constructing a vector of symbols first.
  const uint32_t low_bits = static_cast<uint32_t>(
      above0 ^ above1 ^ above2);
  *packed_symbols = static_cast<uint32_t>(
      _pdep_u64(low_bits, 0x55555555ULL) |
      (_pdep_u64(
           static_cast<uint32_t>(above1), 0x55555555ULL)
       << 1));
  return true;
}

template <RecordMode kMode, bool kAllEntropy>
PGEN_RANS_TARGET_AVX512
bool DecodeEntropyWords32InterleavedAvx512(
    const Model& model, const uint64_t* reference1,
    const uint64_t* reference2, uint32_t sample_ct,
    const uint8_t* payload_begin, const uint8_t* payload_end,
    std::array<uint32_t, 256>* states, uint64_t* target,
    const uint8_t** payload_iter_out, std::string* error) {
  Avx512DecodeModel12 decode_model;
  BuildAvx512DecodeModel12(model, &decode_model);
  __m512i state0 = _mm512_loadu_si512(&(states->at(0)));
  __m512i state1 = _mm512_loadu_si512(&(states->at(16)));
  const uint8_t* payload_iter = payload_begin;
  uint32_t absent_context_mask = 0;
  const uint32_t full_word_ct = sample_ct / 32;
  for (uint32_t word_idx = 0; word_idx != full_word_ct;
       ++word_idx) {
    const uint64_t reference1_word =
        (kMode == RecordMode::kMarginal)
            ? 0
            : reference1[word_idx];
    const uint64_t reference2_word =
        (kMode == RecordMode::kTwoReference)
            ? reference2[word_idx]
            : 0;
    uint32_t packed0;
    uint32_t packed1;
    if (!RansDecodeGroup16Avx512<
            kMode, 0, kAllEntropy>(
            decode_model, reference1_word, reference2_word,
            &state0, &payload_iter, payload_end, &packed0,
            &absent_context_mask, error) ||
        !RansDecodeGroup16Avx512<
            kMode, 1, kAllEntropy>(
            decode_model, reference1_word, reference2_word,
            &state1, &payload_iter, payload_end, &packed1,
            &absent_context_mask, error)) {
      return false;
    }
    target[word_idx] =
        static_cast<uint64_t>(packed0) |
        (static_cast<uint64_t>(packed1) << 32);
  }
  _mm512_storeu_si512(&(states->at(0)), state0);
  _mm512_storeu_si512(&(states->at(16)), state1);

  const uint32_t tail_sample_ct = sample_ct % 32;
  if (tail_sample_ct) {
    const uint32_t word_idx = full_word_ct;
    const uint64_t reference1_word =
        (kMode == RecordMode::kMarginal)
            ? 0
            : reference1[word_idx];
    const uint64_t reference2_word =
        (kMode == RecordMode::kTwoReference)
            ? reference2[word_idx]
            : 0;
    uint64_t packed_word = 0;
    for (uint32_t group_start = 0;
         group_start < tail_sample_ct; group_start += 16) {
      const uint32_t group_end =
          std::min(group_start + 16, tail_sample_ct);
      uint32_t entropy_lane_mask = 0;
      for (uint32_t lane = group_start; lane != group_end;
           ++lane) {
        uint32_t context = 0;
        if (kMode != RecordMode::kMarginal) {
          context = static_cast<uint32_t>(
              (reference1_word >> (2 * lane)) & 3U);
        }
        if (kMode == RecordMode::kTwoReference) {
          context =
              4 * context +
              static_cast<uint32_t>(
                  (reference2_word >> (2 * lane)) & 3U);
        }
        const uint32_t active_symbol_ct =
            model.active_symbol_cts[context];
        if (!active_symbol_ct) {
          absent_context_mask = 1;
          continue;
        }
        uint8_t symbol = model.deterministic_symbols[context];
        if (active_symbol_ct > 1) {
          entropy_lane_mask |= 1U << (lane - group_start);
          const ModelRow& row = model.rows[context];
          const uint32_t slot =
              (*states)[lane] & (kDefaultSlotCt - 1);
          symbol = static_cast<uint8_t>(
              static_cast<uint32_t>(
                  slot >= row.cumulative[1]) +
              static_cast<uint32_t>(
                  slot >= row.cumulative[2]) +
              static_cast<uint32_t>(
                  slot >= row.cumulative[3]));
          (*states)[lane] =
              row.frequencies[symbol] *
                  ((*states)[lane] >> kDefaultScaleBits) +
              slot - row.cumulative[symbol];
        }
        packed_word |=
            static_cast<uint64_t>(symbol) << (2 * lane);
      }
      while (true) {
        uint32_t refill_mask = 0;
        for (uint32_t lane = group_start; lane != group_end;
             ++lane) {
          if ((entropy_lane_mask &
               (1U << (lane - group_start))) &&
              ((*states)[lane] < kRansLowerBound)) {
            refill_mask |= 1U << (lane - group_start);
          }
        }
        if (!refill_mask) {
          break;
        }
        for (uint32_t lane = group_start; lane != group_end;
             ++lane) {
          if (!(refill_mask & (1U << (lane - group_start)))) {
            continue;
          }
          if (payload_iter == payload_end) {
            SetError(
                "Truncated interleaved rANS payload.", error);
            return false;
          }
          (*states)[lane] =
              ((*states)[lane] << 8) | *payload_iter++;
        }
      }
    }
    target[word_idx] = packed_word;
  }
  if (absent_context_mask) {
    SetError("Reference selects an absent model context.", error);
    return false;
  }
  *payload_iter_out = payload_iter;
  return true;
}

template <bool kAllEntropy>
PGEN_RANS_TARGET_AVX512
bool DecodeEntropyRecordInterleavedAvx512(
    RecordMode mode, const Model& model, const uint64_t* reference1,
    const uint64_t* reference2, uint32_t sample_ct,
    const uint8_t* payload_begin, const uint8_t* payload_end,
    std::array<uint32_t, 256>* states, uint64_t* target,
    const uint8_t** payload_iter_out, std::string* error) {
  switch (mode) {
    case RecordMode::kMarginal:
      return DecodeEntropyWords32InterleavedAvx512<
          RecordMode::kMarginal, kAllEntropy>(
          model, reference1, reference2, sample_ct,
          payload_begin, payload_end, states, target,
          payload_iter_out, error);
    case RecordMode::kOneReference:
      return DecodeEntropyWords32InterleavedAvx512<
          RecordMode::kOneReference, kAllEntropy>(
          model, reference1, reference2, sample_ct,
          payload_begin, payload_end, states, target,
          payload_iter_out, error);
    case RecordMode::kTwoReference:
      return DecodeEntropyWords32InterleavedAvx512<
          RecordMode::kTwoReference, kAllEntropy>(
          model, reference1, reference2, sample_ct,
          payload_begin, payload_end, states, target,
          payload_iter_out, error);
  }
  SetError("Unknown rANS record mode.", error);
  return false;
}
#endif

template <RecordMode kMode, uint32_t kStateCt, uint32_t kScaleBits>
bool DecodeEntropyPackedWordsInterleavedScalar(
    const Model& model, const uint64_t* reference1,
    const uint64_t* reference2, uint32_t sample_ct,
    uint32_t runtime_scale_bits, const uint8_t* payload_begin,
    const uint8_t* payload_end, std::array<uint32_t, 256>* states,
    uint64_t* target, const uint8_t** payload_iter_out,
    std::string* error) {
  const uint32_t scale_bits =
      kScaleBits ? kScaleBits : runtime_scale_bits;
  const uint32_t slot_mask = (1U << scale_bits) - 1;
  const uint8_t* payload_iter = payload_begin;
  const uint32_t word_ct = (sample_ct + 31) / 32;
  for (uint32_t word_idx = 0; word_idx != word_ct; ++word_idx) {
    const uint32_t word_sample_ct =
        std::min(32U, sample_ct - 32 * word_idx);
    const uint64_t reference1_word =
        (kMode == RecordMode::kMarginal)
            ? 0
            : reference1[word_idx];
    const uint64_t reference2_word =
        (kMode == RecordMode::kTwoReference)
            ? reference2[word_idx]
            : 0;
    uint64_t packed_word = 0;
    for (uint32_t first_word_sample = 0;
         first_word_sample < word_sample_ct;
         first_word_sample += kStateCt) {
      const uint32_t round_sample_ct =
          std::min(kStateCt, word_sample_ct - first_word_sample);
      for (uint32_t group_start = 0;
           group_start < round_sample_ct; group_start += 16) {
        const uint32_t group_end =
            std::min(group_start + 16, round_sample_ct);
        uint32_t entropy_lane_mask = 0;
        for (uint32_t lane = group_start; lane != group_end;
             ++lane) {
          const uint32_t word_sample = first_word_sample + lane;
          uint32_t context = 0;
          if (kMode != RecordMode::kMarginal) {
            context = static_cast<uint32_t>(
                (reference1_word >> (2 * word_sample)) & 3U);
          }
          if (kMode == RecordMode::kTwoReference) {
            context =
                4 * context +
                static_cast<uint32_t>(
                    (reference2_word >> (2 * word_sample)) & 3U);
          }
          const uint32_t active_symbol_ct =
              model.active_symbol_cts[context];
          if (!active_symbol_ct) {
            SetError("Reference selects an absent model context.",
                     error);
            return false;
          }
          uint8_t symbol = model.deterministic_symbols[context];
          if (active_symbol_ct > 1) {
            entropy_lane_mask |= 1U << (lane - group_start);
            const ModelRow& row = model.rows[context];
            const uint32_t slot = (*states)[lane] & slot_mask;
            symbol = static_cast<uint8_t>(
                static_cast<uint32_t>(
                    slot >= row.cumulative[1]) +
                static_cast<uint32_t>(
                    slot >= row.cumulative[2]) +
                static_cast<uint32_t>(
                    slot >= row.cumulative[3]));
            (*states)[lane] =
                row.frequencies[symbol] *
                    ((*states)[lane] >> scale_bits) +
                slot - row.cumulative[symbol];
          }
          packed_word |=
              static_cast<uint64_t>(symbol) << (2 * word_sample);
        }
        while (true) {
          uint32_t refill_mask = 0;
          for (uint32_t lane = group_start; lane != group_end;
               ++lane) {
            if ((entropy_lane_mask &
                 (1U << (lane - group_start))) &&
                ((*states)[lane] < kRansLowerBound)) {
              refill_mask |= 1U << (lane - group_start);
            }
          }
          if (!refill_mask) {
            break;
          }
          for (uint32_t lane = group_start; lane != group_end;
               ++lane) {
            if (!(refill_mask & (1U << (lane - group_start)))) {
              continue;
            }
            if (payload_iter == payload_end) {
              SetError(
                  "Truncated interleaved rANS payload.", error);
              return false;
            }
            (*states)[lane] =
                ((*states)[lane] << 8) | *payload_iter++;
          }
        }
      }
    }
    target[word_idx] = packed_word;
  }
  *payload_iter_out = payload_iter;
  return true;
}

template <uint32_t kStateCt, uint32_t kScaleBits>
PGEN_RANS_NOINLINE
bool DecodeEntropyRecordPackedWordsInterleavedScalar(
    RecordMode mode, const Model& model, const uint64_t* reference1,
    const uint64_t* reference2, uint32_t sample_ct,
    uint32_t runtime_scale_bits, const uint8_t* payload_begin,
    const uint8_t* payload_end, std::array<uint32_t, 256>* states,
    uint64_t* target, const uint8_t** payload_iter_out,
    std::string* error) {
  switch (mode) {
    case RecordMode::kMarginal:
      return DecodeEntropyPackedWordsInterleavedScalar<
          RecordMode::kMarginal, kStateCt, kScaleBits>(
          model, reference1, reference2, sample_ct,
          runtime_scale_bits, payload_begin, payload_end, states,
          target, payload_iter_out, error);
    case RecordMode::kOneReference:
      return DecodeEntropyPackedWordsInterleavedScalar<
          RecordMode::kOneReference, kStateCt, kScaleBits>(
          model, reference1, reference2, sample_ct,
          runtime_scale_bits, payload_begin, payload_end, states,
          target, payload_iter_out, error);
    case RecordMode::kTwoReference:
      return DecodeEntropyPackedWordsInterleavedScalar<
          RecordMode::kTwoReference, kStateCt, kScaleBits>(
          model, reference1, reference2, sample_ct,
          runtime_scale_bits, payload_begin, payload_end, states,
          target, payload_iter_out, error);
  }
  SetError("Unknown rANS record mode.", error);
  return false;
}

template <uint32_t kScaleBits>
bool DecodeEntropyRecordPowerOfTwoInterleavedScalar(
    RecordMode mode, const Model& model, const uint64_t* reference1,
    const uint64_t* reference2, uint32_t sample_ct,
    uint32_t state_ct, uint32_t runtime_scale_bits,
    const uint8_t* payload_begin, const uint8_t* payload_end,
    std::array<uint32_t, 256>* states, uint64_t* target,
    const uint8_t** payload_iter_out, std::string* error) {
  switch (state_ct) {
    case 4:
      return DecodeEntropyRecordPackedWordsInterleavedScalar<
          4, kScaleBits>(
          mode, model, reference1, reference2, sample_ct,
          runtime_scale_bits, payload_begin, payload_end, states,
          target, payload_iter_out, error);
    case 8:
      return DecodeEntropyRecordPackedWordsInterleavedScalar<
          8, kScaleBits>(
          mode, model, reference1, reference2, sample_ct,
          runtime_scale_bits, payload_begin, payload_end, states,
          target, payload_iter_out, error);
    case 16:
      return DecodeEntropyRecordPackedWordsInterleavedScalar<
          16, kScaleBits>(
          mode, model, reference1, reference2, sample_ct,
          runtime_scale_bits, payload_begin, payload_end, states,
          target, payload_iter_out, error);
    case 32:
      return DecodeEntropyRecordPackedWordsInterleavedScalar<
          32, kScaleBits>(
          mode, model, reference1, reference2, sample_ct,
          runtime_scale_bits, payload_begin, payload_end, states,
          target, payload_iter_out, error);
  }
  SetError("Unsupported packed-word rANS state count.", error);
  return false;
}

template <RecordMode kMode, uint32_t kScaleBits>
bool DecodeEntropyInterleaved(
    const Model& model, const uint64_t* reference1,
    const uint64_t* reference2, uint32_t sample_ct,
    uint32_t state_ct, uint32_t runtime_scale_bits,
    const uint8_t* payload_begin, const uint8_t* payload_end,
    std::array<uint32_t, 256>* states, uint64_t* target,
    const uint8_t** payload_iter_out, std::string* error) {
  const uint32_t scale_bits =
      kScaleBits ? kScaleBits : runtime_scale_bits;
  const uint32_t slot_mask = (1U << scale_bits) - 1;
  memset(
      target, 0,
      static_cast<size_t>((sample_ct + 31) / 32) *
          sizeof(uint64_t));
  const uint8_t* payload_iter = payload_begin;
  for (uint32_t first_sample = 0; first_sample < sample_ct;
       first_sample += state_ct) {
    const uint32_t round_sample_ct =
        std::min(state_ct, sample_ct - first_sample);
    for (uint32_t group_start = 0;
         group_start < round_sample_ct; group_start += 16) {
      const uint32_t group_end =
          std::min(group_start + 16, round_sample_ct);
      uint32_t entropy_lane_mask = 0;
      for (uint32_t lane = group_start; lane != group_end;
           ++lane) {
        const uint32_t sample_idx = first_sample + lane;
        uint32_t context = 0;
        if (kMode != RecordMode::kMarginal) {
          context =
              GetPackedGenotype(reference1, sample_idx);
        }
        if (kMode == RecordMode::kTwoReference) {
          context =
              4 * context +
              GetPackedGenotype(reference2, sample_idx);
        }
        const uint32_t active_symbol_ct =
            model.active_symbol_cts[context];
        if (!active_symbol_ct) {
          SetError("Reference selects an absent model context.",
                   error);
          return false;
        }
        uint8_t symbol = model.deterministic_symbols[context];
        if (active_symbol_ct > 1) {
          entropy_lane_mask |= 1U << (lane - group_start);
          const ModelRow& row = model.rows[context];
          const uint32_t slot = (*states)[lane] & slot_mask;
          symbol = static_cast<uint8_t>(
              static_cast<uint32_t>(
                  slot >= row.cumulative[1]) +
              static_cast<uint32_t>(
                  slot >= row.cumulative[2]) +
              static_cast<uint32_t>(
                  slot >= row.cumulative[3]));
          (*states)[lane] =
              row.frequencies[symbol] *
                  ((*states)[lane] >> scale_bits) +
              slot - row.cumulative[symbol];
        }
        SetPackedGenotype(target, sample_idx, symbol);
      }
      while (true) {
        uint32_t refill_mask = 0;
        for (uint32_t lane = group_start; lane != group_end;
             ++lane) {
          if ((entropy_lane_mask &
               (1U << (lane - group_start))) &&
              ((*states)[lane] < kRansLowerBound)) {
            refill_mask |= 1U << (lane - group_start);
          }
        }
        if (!refill_mask) {
          break;
        }
        for (uint32_t lane = group_start; lane != group_end;
             ++lane) {
          if (!(refill_mask & (1U << (lane - group_start)))) {
            continue;
          }
          if (payload_iter == payload_end) {
            SetError(
                "Truncated interleaved rANS payload.", error);
            return false;
          }
          (*states)[lane] =
              ((*states)[lane] << 8) | *payload_iter++;
        }
      }
    }
  }
  *payload_iter_out = payload_iter;
  return true;
}

template <uint32_t kScaleBits>
bool DecodeEntropyRecordInterleaved(
    RecordMode mode, const Model& model, const uint64_t* reference1,
    const uint64_t* reference2, uint32_t sample_ct,
    uint32_t state_ct, uint32_t runtime_scale_bits,
    const uint8_t* payload_begin, const uint8_t* payload_end,
    std::array<uint32_t, 256>* states, uint64_t* target,
    const uint8_t** payload_iter_out, std::string* error) {
  switch (mode) {
    case RecordMode::kMarginal:
      return DecodeEntropyInterleaved<
          RecordMode::kMarginal, kScaleBits>(
          model, reference1, reference2, sample_ct, state_ct,
          runtime_scale_bits, payload_begin, payload_end, states,
          target, payload_iter_out, error);
    case RecordMode::kOneReference:
      return DecodeEntropyInterleaved<
          RecordMode::kOneReference, kScaleBits>(
          model, reference1, reference2, sample_ct, state_ct,
          runtime_scale_bits, payload_begin, payload_end, states,
          target, payload_iter_out, error);
    case RecordMode::kTwoReference:
      return DecodeEntropyInterleaved<
          RecordMode::kTwoReference, kScaleBits>(
          model, reference1, reference2, sample_ct, state_ct,
          runtime_scale_bits, payload_begin, payload_end, states,
          target, payload_iter_out, error);
  }
  SetError("Unknown rANS record mode.", error);
  return false;
}

bool ParsePrefix(const uint8_t* record, size_t record_size,
                 RecordMetadata* metadata, size_t* offset,
                 std::string* error) {
  if (!record_size) {
    SetError("Empty rANS record.", error);
    return false;
  }
  const uint8_t flags = record[0];
  if (flags & ~kKnownFlagMask) {
    SetError("Unknown rANS record flags.", error);
    return false;
  }
  if ((flags & kUnpaddedPayloadFlag) &&
      !(flags & kEntropyPayloadFlag)) {
    SetError("Unpadded-payload flag requires an entropy payload.",
             error);
    return false;
  }
  const uint8_t mode_code = flags & kModeMask;
  if (mode_code > static_cast<uint8_t>(RecordMode::kTwoReference)) {
    SetError("Unknown rANS record mode.", error);
    return false;
  }
  metadata->mode = static_cast<RecordMode>(mode_code);
  metadata->has_entropy_payload = (flags & kEntropyPayloadFlag);
  metadata->has_multiallelic_patches =
      (flags & kMultiallelicPatchFlag);
  metadata->is_raw_packed = flags & kRawPackedRecordFlag;
  metadata->is_sparse_predictor =
      flags & kSparsePredictorRecordFlag;
  if ((metadata->is_raw_packed && metadata->is_sparse_predictor) ||
      ((flags & kSparsePredictorBitmapFlag) &&
       !metadata->is_sparse_predictor) ||
      ((metadata->is_raw_packed || metadata->is_sparse_predictor) &&
       (flags & (kEntropyPayloadFlag | kUnpaddedPayloadFlag))) ||
      (metadata->is_raw_packed &&
       (metadata->mode != RecordMode::kMarginal))) {
    SetError("Invalid alternate-record flag combination.", error);
    return false;
  }
  *offset = 1;
  if (metadata->mode != RecordMode::kMarginal) {
    if (*offset == record_size) {
      SetError("Truncated first reference selector.", error);
      return false;
    }
    metadata->reference1 = record[(*offset)++];
  }
  if (metadata->mode == RecordMode::kTwoReference) {
    if (*offset == record_size) {
      SetError("Truncated second reference selector.", error);
      return false;
    }
    metadata->reference2 = record[(*offset)++];
    if (metadata->reference1 == metadata->reference2) {
      SetError("Two-reference record repeats an anchor.", error);
      return false;
    }
  }
  return true;
}

}  // namespace

uint32_t PackedWordCt(uint32_t sample_ct) {
  return (sample_ct + 31) / 32;
}

uint8_t GetPackedGenotype(const uint64_t* genotypes, uint32_t sample_idx) {
  return static_cast<uint8_t>(
      (genotypes[sample_idx / 32] >> (2 * (sample_idx % 32))) & 3U);
}

void SetPackedGenotype(uint64_t* genotypes, uint32_t sample_idx,
                       uint8_t genotype) {
  const uint32_t shift = 2 * (sample_idx % 32);
  uint64_t& word = genotypes[sample_idx / 32];
  word = (word & ~(3ULL << shift)) |
         (static_cast<uint64_t>(genotype & 3U) << shift);
}

static bool EncodeRecordImpl(
    const uint64_t* target, const uint64_t* reference1,
    const uint64_t* reference2, uint32_t sample_ct, RecordMode mode,
    uint8_t reference1_idx, uint8_t reference2_idx,
    const uint32_t* context_symbol_counts, const CodecParams& params,
    std::vector<uint8_t>* record, std::string* error) {
  record->clear();
  if ((!target) || (!sample_ct)) {
    SetError("A target and at least one sample are required.", error);
    return false;
  }
  if ((!params.state_ct) || (params.state_ct > 256) ||
      (params.scale_bits < 8) || (params.scale_bits > 16)) {
    SetError("Invalid codec parameters.", error);
    return false;
  }
  if ((mode != RecordMode::kMarginal) && (!reference1)) {
    SetError("Conditional record is missing its first reference.", error);
    return false;
  }
  if ((mode == RecordMode::kTwoReference) &&
      ((!reference2) || (reference1_idx == reference2_idx))) {
    SetError("Two-reference record requires two distinct references.", error);
    return false;
  }

  Model model;
  if (context_symbol_counts) {
    uint64_t counted_sample_ct = 0;
    const uint32_t count_ct = 4 * RowCt(mode);
    for (uint32_t count_idx = 0; count_idx != count_ct; ++count_idx) {
      counted_sample_ct += context_symbol_counts[count_idx];
    }
    if (counted_sample_ct != sample_ct) {
      SetError("Supplied rANS model counts do not match sample count.",
               error);
      return false;
    }
  }
  if (context_symbol_counts
          ? !BuildModelFromCounts(
                context_symbol_counts, mode, params.scale_bits, &model,
                error)
          : !BuildModel(
                target, reference1, reference2, sample_ct, mode,
                params.scale_bits, &model, error)) {
    return false;
  }
  uint8_t flags = static_cast<uint8_t>(mode);
  if (model.has_entropy) {
    flags |= kEntropyPayloadFlag | kUnpaddedPayloadFlag;
  }
  record->push_back(flags);
  if (mode != RecordMode::kMarginal) {
    record->push_back(reference1_idx);
  }
  if (mode == RecordMode::kTwoReference) {
    record->push_back(reference2_idx);
  }
  SerializeModel(model, mode, record);
  if (!model.has_entropy) {
    if (context_symbol_counts) {
      for (uint32_t sample_idx = 0; sample_idx != sample_ct;
           ++sample_idx) {
        const uint32_t context = ContextIndex(
            mode, reference1, reference2, sample_idx);
        if ((!model.active_symbol_cts[context]) ||
            (GetPackedGenotype(target, sample_idx) !=
             model.deterministic_symbols[context])) {
          SetError("Supplied rANS model counts do not match genotypes.",
                   error);
          record->clear();
          return false;
        }
      }
    }
    return true;
  }

  const uint32_t state_ct = std::min(params.state_ct, sample_ct);
  std::vector<uint32_t> states(state_ct, kRansLowerBound);
  std::array<std::array<RansEncoderSymbol, 4>, 16> encoders;
  for (uint32_t context = 0; context != model.row_ct; ++context) {
    if (model.active_symbol_cts[context] <= 1) {
      continue;
    }
    const ModelRow& row = model.rows[context];
    for (uint32_t symbol = 0; symbol != 4; ++symbol) {
      if (row.frequencies[symbol]) {
        encoders[context][symbol] = BuildRansEncoderSymbol(
            row.cumulative[symbol], row.frequencies[symbol],
            params.scale_bits);
      }
    }
  }
  std::vector<uint8_t> reverse_payload;
  if (!BuildInterleavedPayloadDirect(
          target, reference1, reference2, sample_ct, mode, model,
          params.scale_bits, encoders, &states, &reverse_payload, error)) {
    record->clear();
    return false;
  }
  for (const uint32_t state : states) {
    AppendU32(state, record);
  }
  record->insert(
      record->end(), reverse_payload.rbegin(), reverse_payload.rend());
  return true;
}

bool EncodeRecord(const uint64_t* target, const uint64_t* reference1,
                  const uint64_t* reference2, uint32_t sample_ct,
                  RecordMode mode, uint8_t reference1_idx,
                  uint8_t reference2_idx, const CodecParams& params,
                  std::vector<uint8_t>* record, std::string* error) {
  return EncodeRecordImpl(
      target, reference1, reference2, sample_ct, mode, reference1_idx,
      reference2_idx, nullptr, params, record, error);
}

bool EncodeRecordFromCounts(
    const uint64_t* target, const uint64_t* reference1,
    const uint64_t* reference2, uint32_t sample_ct, RecordMode mode,
    uint8_t reference1_idx, uint8_t reference2_idx,
    const uint32_t* context_symbol_counts, const CodecParams& params,
    std::vector<uint8_t>* record, std::string* error) {
  if (!context_symbol_counts) {
    SetError("Missing supplied rANS model counts.", error);
    return false;
  }
  return EncodeRecordImpl(
      target, reference1, reference2, sample_ct, mode, reference1_idx,
      reference2_idx, context_symbol_counts, params, record, error);
}

bool EstimateRecordBytes(const uint32_t* context_symbol_counts,
                         RecordMode mode, const CodecParams& params,
                         uint64_t* record_bytes, std::string* error) {
  if ((!context_symbol_counts) || (!record_bytes) || (!params.state_ct) ||
      (params.state_ct > 256) || (params.scale_bits < 8) ||
      (params.scale_bits > 16)) {
    SetError("Invalid record-size estimator arguments.", error);
    return false;
  }
  Model model;
  if (!BuildModelFromCounts(context_symbol_counts, mode, params.scale_bits,
                            &model, error)) {
    return false;
  }
  uint64_t result = 1;
  if (mode == RecordMode::kOneReference) {
    result += 2;
  } else if (mode == RecordMode::kTwoReference) {
    result += 4;
  }
  long double quantized_bits = 0.0;
  const uint32_t total_frequency = 1U << params.scale_bits;
  uint64_t sample_ct = 0;
  for (uint32_t context = 0; context != model.row_ct; ++context) {
    if (!(model.context_mask & (1U << context))) {
      continue;
    }
    const ModelRow& row = model.rows[context];
    result +=
        1 + 2 * (model.active_symbol_cts[context] - 1);
    for (uint32_t symbol = 0; symbol != 4; ++symbol) {
      const uint32_t count =
          context_symbol_counts[4 * context + symbol];
      sample_ct += count;
      if (count) {
        quantized_bits +=
            static_cast<long double>(count) *
            std::log2(static_cast<long double>(total_frequency) /
                      row.frequencies[symbol]);
      }
    }
  }
  if (model.has_entropy) {
    const uint32_t state_ct = static_cast<uint32_t>(
        std::min<uint64_t>(params.state_ct, sample_ct));
    result += 4LLU * state_ct;
    result += static_cast<uint64_t>(std::ceil(quantized_bits / 8.0L));
  }
  *record_bytes = result;
  return true;
}

bool ParseRecordMetadata(const uint8_t* record, size_t record_size,
                         RecordMetadata* metadata, std::string* error) {
  size_t base_record_size;
  if (!GetBaseRecordByteCt(
          record, record_size, &base_record_size, error)) {
    return false;
  }
  size_t offset;
  return ParsePrefix(
      record, base_record_size, metadata, &offset, error);
}

bool GetBaseRecordByteCt(const uint8_t* record, size_t record_size,
                         size_t* base_record_size, std::string* error) {
  if ((!record) || (!record_size) || (!base_record_size)) {
    SetError("Invalid rANS record-size arguments.", error);
    return false;
  }
  if (!(record[0] & kMultiallelicPatchFlag)) {
    *base_record_size = record_size;
    return true;
  }
  if (record_size <
      1 + kMultiallelicPatchHeaderByteCt +
          kMultiallelicPatchFooterByteCt) {
    SetError("Truncated multiallelic patch record.", error);
    return false;
  }
  size_t footer_offset =
      record_size - kMultiallelicPatchFooterByteCt;
  uint32_t patch_byte_ct;
  if (!ReadU32(
          record, record_size, &footer_offset, &patch_byte_ct) ||
      (patch_byte_ct < kMultiallelicPatchHeaderByteCt) ||
      (static_cast<size_t>(patch_byte_ct) >
       record_size - 1 - kMultiallelicPatchFooterByteCt)) {
    SetError("Invalid multiallelic patch length.", error);
    return false;
  }
  *base_record_size =
      record_size - kMultiallelicPatchFooterByteCt - patch_byte_ct;
  return true;
}

bool AppendMultiallelicPatches(uint32_t sample_ct,
                               const MultiallelicPatches& patches,
                               std::vector<uint8_t>* record,
                               std::string* error) {
  if ((!record) || record->empty() || (!sample_ct)) {
    SetError("Invalid multiallelic patch encoder arguments.", error);
    return false;
  }
  if (record->front() & kMultiallelicPatchFlag) {
    SetError("rANS record already has multiallelic patches.", error);
    return false;
  }
  if ((patches.allele_ct < 3) || (patches.allele_ct > 255)) {
    SetError("Multiallelic allele count is out of range.", error);
    return false;
  }
  if ((patches.patch_01_values.size() !=
       patches.patch_01_sample_ids.size()) ||
      (patches.patch_10_values.size() !=
       2 * patches.patch_10_sample_ids.size())) {
    SetError("Multiallelic patch value count mismatch.", error);
    return false;
  }
  if ((patches.patch_01_sample_ids.size() >
       std::numeric_limits<uint32_t>::max()) ||
      (patches.patch_10_sample_ids.size() >
       std::numeric_limits<uint32_t>::max())) {
    SetError("Multiallelic patch count exceeds format limits.", error);
    return false;
  }
  if (!ValidatePatchIds(
          patches.patch_01_sample_ids, sample_ct, error) ||
      !ValidatePatchIds(
          patches.patch_10_sample_ids, sample_ct, error)) {
    return false;
  }
  if (!PatchIdSetsAreDisjoint(
          patches.patch_01_sample_ids,
          patches.patch_10_sample_ids)) {
    SetError("Multiallelic patch classes overlap.", error);
    return false;
  }
  for (const uint8_t allele_code : patches.patch_01_values) {
    if ((allele_code < 2) || (allele_code >= patches.allele_ct)) {
      SetError("Invalid ref/ALT multiallelic patch allele code.", error);
      return false;
    }
  }
  for (const uint8_t allele_code : patches.patch_10_values) {
    if ((!allele_code) || (allele_code >= patches.allele_ct)) {
      SetError("Invalid ALT/ALT multiallelic patch allele code.", error);
      return false;
    }
  }

  std::vector<uint8_t> payload;
  payload.reserve(
      kMultiallelicPatchHeaderByteCt +
      2 * (patches.patch_01_sample_ids.size() +
           patches.patch_10_sample_ids.size()) +
      patches.patch_01_values.size() +
      patches.patch_10_values.size());
  payload.push_back(kMultiallelicPatchVersion);
  payload.push_back(static_cast<uint8_t>(patches.allele_ct));
  const bool patch_01_bitmap =
      PatchBitmapIsSmaller(
          patches.patch_01_sample_ids, sample_ct);
  const bool patch_10_bitmap =
      PatchBitmapIsSmaller(
          patches.patch_10_sample_ids, sample_ct);
  payload.push_back(
      static_cast<uint8_t>(
          (patch_01_bitmap ? kPatch01BitmapFlag : 0) |
          (patch_10_bitmap ? kPatch10BitmapFlag : 0)));
  AppendU32(
      static_cast<uint32_t>(patches.patch_01_sample_ids.size()),
      &payload);
  AppendU32(
      static_cast<uint32_t>(patches.patch_10_sample_ids.size()),
      &payload);
  AppendPatchIds(
      patches.patch_01_sample_ids, sample_ct,
      patch_01_bitmap, &payload);
  AppendPatchIds(
      patches.patch_10_sample_ids, sample_ct,
      patch_10_bitmap, &payload);

  uint64_t bit_buffer = 0;
  uint32_t bit_ct = 0;
  const uint32_t patch_01_bit_width =
      CeilLog2(patches.allele_ct - 2);
  for (const uint8_t allele_code : patches.patch_01_values) {
    AppendPackedValue(
        allele_code - 2, patch_01_bit_width,
        &bit_buffer, &bit_ct, &payload);
  }
  FlushPackedValues(&bit_buffer, &bit_ct, &payload);
  const uint32_t patch_10_bit_width =
      CeilLog2(patches.allele_ct - 1);
  for (const uint8_t allele_code : patches.patch_10_values) {
    AppendPackedValue(
        allele_code - 1, patch_10_bit_width,
        &bit_buffer, &bit_ct, &payload);
  }
  FlushPackedValues(&bit_buffer, &bit_ct, &payload);
  if (payload.size() > std::numeric_limits<uint32_t>::max()) {
    SetError("Multiallelic patch payload is too large.", error);
    return false;
  }
  record->front() |= kMultiallelicPatchFlag;
  record->insert(record->end(), payload.begin(), payload.end());
  AppendU32(static_cast<uint32_t>(payload.size()), record);
  return true;
}

bool DecodeMultiallelicPatches(const uint8_t* record, size_t record_size,
                               uint32_t sample_ct,
                               MultiallelicPatches* patches,
                               std::string* error) {
  if ((!record) || (!record_size) || (!sample_ct) || (!patches)) {
    SetError("Invalid multiallelic patch decoder arguments.", error);
    return false;
  }
  *patches = MultiallelicPatches();
  size_t base_record_size;
  if (!GetBaseRecordByteCt(
          record, record_size, &base_record_size, error)) {
    return false;
  }
  if (!(record[0] & kMultiallelicPatchFlag)) {
    return true;
  }
  const size_t payload_size =
      record_size - base_record_size - kMultiallelicPatchFooterByteCt;
  const uint8_t* const payload = record + base_record_size;
  size_t offset = 0;
  if ((payload_size < kMultiallelicPatchHeaderByteCt) ||
      (payload[offset++] != kMultiallelicPatchVersion)) {
    SetError("Unsupported multiallelic patch version.", error);
    return false;
  }
  patches->allele_ct = payload[offset++];
  if (patches->allele_ct < 3) {
    SetError("Invalid multiallelic patch allele count.", error);
    return false;
  }
  const uint8_t patch_flags = payload[offset++];
  if (patch_flags & ~kKnownPatchFlagMask) {
    SetError("Unknown multiallelic patch flags.", error);
    return false;
  }
  uint32_t patch_01_ct;
  uint32_t patch_10_ct;
  if (!ReadU32(payload, payload_size, &offset, &patch_01_ct) ||
      !ReadU32(payload, payload_size, &offset, &patch_10_ct)) {
    SetError("Truncated multiallelic patch header.", error);
    return false;
  }
  if ((patch_01_ct > sample_ct) || (patch_10_ct > sample_ct)) {
    SetError("Multiallelic patch count exceeds the sample count.", error);
    return false;
  }
  if (!ReadPatchIds(
          payload, payload_size, sample_ct, patch_01_ct,
          patch_flags & kPatch01BitmapFlag, &offset,
          &patches->patch_01_sample_ids, error) ||
      !ReadPatchIds(
          payload, payload_size, sample_ct, patch_10_ct,
          patch_flags & kPatch10BitmapFlag, &offset,
          &patches->patch_10_sample_ids, error)) {
    return false;
  }
  if (!PatchIdSetsAreDisjoint(
          patches->patch_01_sample_ids,
          patches->patch_10_sample_ids)) {
    SetError("Multiallelic patch classes overlap.", error);
    return false;
  }

  uint64_t bit_buffer = 0;
  uint32_t bit_ct = 0;
  const uint32_t patch_01_bit_width =
      CeilLog2(patches->allele_ct - 2);
  patches->patch_01_values.resize(patch_01_ct);
  for (uint32_t idx = 0; idx != patch_01_ct; ++idx) {
    uint32_t encoded_value;
    if (!ReadPackedValue(
            payload, payload_size, patch_01_bit_width, &offset,
            &bit_buffer, &bit_ct, &encoded_value) ||
        (encoded_value >=
         static_cast<uint32_t>(patches->allele_ct - 2))) {
      SetError("Invalid or truncated ref/ALT multiallelic patch values.",
               error);
      return false;
    }
    patches->patch_01_values[idx] =
        static_cast<uint8_t>(encoded_value + 2);
  }
  if (bit_buffer) {
    SetError("Nonzero ref/ALT multiallelic patch padding bits.", error);
    return false;
  }
  bit_buffer = 0;
  bit_ct = 0;
  const uint32_t patch_10_bit_width =
      CeilLog2(patches->allele_ct - 1);
  const size_t patch_10_value_ct =
      static_cast<size_t>(2) * patch_10_ct;
  patches->patch_10_values.resize(patch_10_value_ct);
  for (size_t idx = 0; idx != patch_10_value_ct; ++idx) {
    uint32_t encoded_value;
    if (!ReadPackedValue(
            payload, payload_size, patch_10_bit_width, &offset,
            &bit_buffer, &bit_ct, &encoded_value) ||
        (encoded_value >=
         static_cast<uint32_t>(patches->allele_ct - 1))) {
      SetError("Invalid or truncated ALT/ALT multiallelic patch values.",
               error);
      return false;
    }
    patches->patch_10_values[idx] =
        static_cast<uint8_t>(encoded_value + 1);
  }
  if (bit_buffer || (offset != payload_size)) {
    SetError("Invalid multiallelic patch padding or trailing bytes.",
             error);
    return false;
  }
  return true;
}

bool DecodeRecordToBufferImpl(
    const uint8_t* record, size_t record_size,
    const uint64_t* const* anchors, uint32_t anchor_ct,
    uint32_t sample_ct, const CodecParams& params, uint64_t* target,
    size_t target_word_ct, bool validated_contexts,
    RecordMetadata* metadata, std::string* error) {
  (void)validated_contexts;
  const uint32_t packed_word_ct = PackedWordCt(sample_ct);
  if ((!record) || (!sample_ct) || (!params.state_ct) ||
      (params.state_ct > 256) || (params.scale_bits < 8) ||
      (params.scale_bits > 16) || (!target) ||
      (target_word_ct < packed_word_ct)) {
    SetError("Invalid decoder arguments.", error);
    return false;
  }
  RecordMetadata parsed_metadata;
  size_t base_record_size;
  if (!GetBaseRecordByteCt(
          record, record_size, &base_record_size, error)) {
    return false;
  }
  if (IsAlternateRecord(record[0])) {
    return DecodeAlternateRecordToBuffer(
        record, base_record_size, anchors, anchor_ct, sample_ct,
        target, target_word_ct, metadata, error);
  }
  size_t offset;
  if (!ParsePrefix(
          record, base_record_size, &parsed_metadata, &offset, error)) {
    return false;
  }
  if ((parsed_metadata.mode != RecordMode::kMarginal) &&
      ((!anchors) || (parsed_metadata.reference1 >= anchor_ct))) {
    SetError("First reference selector is out of range.", error);
    return false;
  }
  if ((parsed_metadata.mode == RecordMode::kTwoReference) &&
      (parsed_metadata.reference2 >= anchor_ct)) {
    SetError("Second reference selector is out of range.", error);
    return false;
  }
  const uint64_t* reference1 =
      (parsed_metadata.mode == RecordMode::kMarginal)
          ? nullptr
          : anchors[parsed_metadata.reference1];
  const uint64_t* reference2 =
      (parsed_metadata.mode == RecordMode::kTwoReference)
          ? anchors[parsed_metadata.reference2]
          : nullptr;
  if ((parsed_metadata.mode != RecordMode::kMarginal) && (!reference1)) {
    SetError("First reference genotype vector is unavailable.", error);
    return false;
  }
  if ((parsed_metadata.mode == RecordMode::kTwoReference) && (!reference2)) {
    SetError("Second reference genotype vector is unavailable.", error);
    return false;
  }

  Model model;
  if (!ParseModel(record, base_record_size, parsed_metadata.mode,
                  params.scale_bits, &offset, &model, error)) {
    return false;
  }
  if (model.has_entropy != parsed_metadata.has_entropy_payload) {
    SetError("Entropy-payload flag does not match the decoded model.", error);
    return false;
  }
  if (!model.has_entropy) {
    if (offset != base_record_size) {
      SetError("Deterministic record has trailing payload.", error);
      return false;
    }
    for (uint32_t word_idx = 0; word_idx != packed_word_ct; ++word_idx) {
      const uint32_t first_sample = 32 * word_idx;
      const uint32_t word_sample_ct =
          std::min(32U, sample_ct - first_sample);
      uint64_t packed_word = 0;
      for (uint32_t word_offset = 0; word_offset != word_sample_ct;
           ++word_offset) {
        const uint32_t sample_idx = first_sample + word_offset;
        const uint32_t context = ContextIndex(
            parsed_metadata.mode, reference1, reference2, sample_idx);
        if (model.active_symbol_cts[context] != 1) {
          SetError("Reference selects an absent deterministic context.",
                   error);
          return false;
        }
        packed_word |=
                       static_cast<uint64_t>(
                           model.deterministic_symbols[context])
                       << (2 * word_offset);
      }
      target[word_idx] = packed_word;
    }
    if (metadata) {
      *metadata = parsed_metadata;
    }
    return true;
  }

  const uint32_t state_ct = std::min(params.state_ct, sample_ct);
  std::array<uint32_t, 256> states = {};
  for (uint32_t lane = 0; lane != state_ct; ++lane) {
    if (!ReadU32(record, base_record_size, &offset, &(states[lane])) ||
        (states[lane] < kRansLowerBound)) {
      SetError("Invalid or truncated rANS initial state.", error);
      return false;
    }
  }
  const uint8_t* const payload_begin = record + offset;
  const bool unpadded_payload = record[0] & kUnpaddedPayloadFlag;
  const uint8_t* payload_end = record + base_record_size;
  if (!unpadded_payload) {
    if (base_record_size - offset < kInterleavedPaddingByteCt) {
      SetError("Truncated interleaved rANS padding.", error);
      return false;
    }
    payload_end -= kInterleavedPaddingByteCt;
    for (const uint8_t* padding_iter = payload_end;
         padding_iter != record + base_record_size; ++padding_iter) {
      if (*padding_iter) {
        SetError("Interleaved rANS padding is nonzero.", error);
        return false;
      }
    }
  }
  const uint8_t* payload_iter = payload_begin;
  bool decode_ok = false;
  bool used_vector_decoder = false;
#if PGEN_RANS_X86_RUNTIME_DISPATCH
  if ((state_ct == 32) &&
      (params.scale_bits == kDefaultScaleBits) &&
      (sample_ct >= kAvx512MinimumSampleCt) &&
      HasAvx512Decoder()) {
    used_vector_decoder = true;
    const uint16_t full_context_mask = static_cast<uint16_t>(
        (1U << model.row_ct) - 1);
    decode_ok =
        (validated_contexts ||
         (model.entropy_context_mask == full_context_mask))
            ? DecodeEntropyRecordInterleavedAvx512<true>(
                  parsed_metadata.mode, model, reference1,
                  reference2, sample_ct, payload_begin,
                  payload_end, &states, target, &payload_iter,
                  error)
            : DecodeEntropyRecordInterleavedAvx512<false>(
                  parsed_metadata.mode, model, reference1,
                  reference2, sample_ct, payload_begin,
                  payload_end, &states, target, &payload_iter,
                  error);
  }
#endif
  if (!used_vector_decoder) {
    if ((state_ct == 4) || (state_ct == 8) ||
        (state_ct == 16) || (state_ct == 32)) {
      decode_ok =
          (params.scale_bits == kDefaultScaleBits)
              ? DecodeEntropyRecordPowerOfTwoInterleavedScalar<
                    kDefaultScaleBits>(
                    parsed_metadata.mode, model, reference1,
                    reference2, sample_ct, state_ct, params.scale_bits,
                    payload_begin, payload_end, &states, target,
                    &payload_iter, error)
              : DecodeEntropyRecordPowerOfTwoInterleavedScalar<0>(
                    parsed_metadata.mode, model, reference1,
                    reference2, sample_ct, state_ct, params.scale_bits,
                    payload_begin, payload_end, &states, target,
                    &payload_iter, error);
    } else {
      decode_ok =
          (params.scale_bits == kDefaultScaleBits)
              ? DecodeEntropyRecordInterleaved<
                    kDefaultScaleBits>(
                    parsed_metadata.mode, model, reference1,
                    reference2, sample_ct, state_ct,
                    params.scale_bits, payload_begin, payload_end,
                    &states, target, &payload_iter, error)
              : DecodeEntropyRecordInterleaved<0>(
                    parsed_metadata.mode, model, reference1,
                    reference2, sample_ct, state_ct,
                    params.scale_bits, payload_begin, payload_end,
                    &states, target, &payload_iter, error);
    }
  }
  if (!decode_ok) {
    return false;
  }
  if (payload_iter != payload_end) {
    SetError(
        "Interleaved rANS payload has trailing bytes.", error);
    return false;
  }
  for (uint32_t lane = 0; lane != state_ct; ++lane) {
    if (states[lane] != kRansLowerBound) {
      SetError(
          "Interleaved rANS state did not terminate canonically.",
          error);
      return false;
    }
  }
  if (metadata) {
    *metadata = parsed_metadata;
  }
  return true;
}

bool DecodeRecordToBuffer(const uint8_t* record, size_t record_size,
                          const uint64_t* const* anchors,
                          uint32_t anchor_ct, uint32_t sample_ct,
                          const CodecParams& params, uint64_t* target,
                          size_t target_word_ct, RecordMetadata* metadata,
                          std::string* error) {
  return DecodeRecordToBufferImpl(
      record, record_size, anchors, anchor_ct, sample_ct, params,
      target, target_word_ct, false, metadata, error);
}

bool DecodeRecordToBufferFromValidatedBlock(
    const uint8_t* record, size_t record_size,
    const uint64_t* const* anchors, uint32_t anchor_ct,
    uint32_t sample_ct, const CodecParams& params, uint64_t* target,
    size_t target_word_ct, RecordMetadata* metadata,
    std::string* error) {
  return DecodeRecordToBufferImpl(
      record, record_size, anchors, anchor_ct, sample_ct, params,
      target, target_word_ct, true, metadata, error);
}

bool DecodeRecord(const uint8_t* record, size_t record_size,
                  const uint64_t* const* anchors, uint32_t anchor_ct,
                  uint32_t sample_ct, const CodecParams& params,
                  std::vector<uint64_t>* target, RecordMetadata* metadata,
                  std::string* error) {
  if (!target) {
    SetError("Invalid decoder output.", error);
    return false;
  }
  target->assign(PackedWordCt(sample_ct), 0);
  if (!DecodeRecordToBuffer(
          record, record_size, anchors, anchor_ct, sample_ct, params,
          target->data(), target->size(), metadata, error)) {
    target->clear();
    return false;
  }
  return true;
}

}  // namespace pgen_rans
