// SPDX-License-Identifier: GPL-3.0-or-later

#include "pgen_rans_codec.h"

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
constexpr uint8_t kInterleavedPayloadFlag = 1U << 3;
constexpr uint8_t kKnownFlagMask =
    kModeMask | kEntropyPayloadFlag | kInterleavedPayloadFlag;
constexpr uint32_t kInterleavedPaddingByteCt = 15;
constexpr uint32_t kDefaultScaleBits = 12;
#if PGEN_RANS_X86_RUNTIME_DISPATCH
constexpr uint32_t kDefaultSlotCt = 1U << kDefaultScaleBits;
constexpr uint32_t kAvx512MinimumSampleCt = 32768;
#endif

#if defined(_MSC_VER)
#define PGEN_RANS_ALWAYS_INLINE __forceinline
#elif defined(__GNUC__) || defined(__clang__)
#define PGEN_RANS_ALWAYS_INLINE inline __attribute__((always_inline))
#else
#define PGEN_RANS_ALWAYS_INLINE inline
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

void RansEncodeSymbol(uint32_t cumulative, uint32_t frequency,
                      uint32_t scale_bits, uint32_t* state,
                      std::vector<uint8_t>* output) {
  const uint64_t maximum_state =
      ((static_cast<uint64_t>(kRansLowerBound) >> scale_bits) << 8) *
      frequency;
  while (*state >= maximum_state) {
    output->push_back(static_cast<uint8_t>(*state));
    *state >>= 8;
  }
  *state = ((*state / frequency) << scale_bits) +
           (*state % frequency) + cumulative;
}

bool BuildInterleavedPayload(
    const uint64_t* target, const uint64_t* reference1,
    const uint64_t* reference2, uint32_t sample_ct,
    RecordMode mode, const Model& model, uint32_t scale_bits,
    const std::vector<uint32_t>& initial_states,
    const std::vector<std::vector<uint8_t>>& lane_payloads,
    std::vector<uint8_t>* payload, std::string* error) {
  const uint32_t state_ct =
      static_cast<uint32_t>(initial_states.size());
  std::vector<uint32_t> states = initial_states;
  std::vector<size_t> lane_offsets(state_ct);
  size_t payload_byte_ct = 0;
  for (uint32_t lane = 0; lane != state_ct; ++lane) {
    lane_offsets[lane] = lane_payloads[lane].size();
    payload_byte_ct += lane_offsets[lane];
  }
  payload->clear();
  payload->reserve(payload_byte_ct);
  const uint32_t slot_mask = (1U << scale_bits) - 1;
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
        const uint32_t context = ContextIndex(
            mode, reference1, reference2, sample_idx);
        if (model.active_symbol_cts[context] <= 1) {
          continue;
        }
        entropy_lane_mask |= 1U << (lane - group_start);
        const ModelRow& row = model.rows[context];
        const uint32_t symbol =
            GetPackedGenotype(target, sample_idx);
        const uint32_t slot = states[lane] & slot_mask;
        states[lane] =
            row.frequencies[symbol] *
                (states[lane] >> scale_bits) +
            slot - row.cumulative[symbol];
      }
      while (true) {
        uint32_t refill_mask = 0;
        for (uint32_t lane = group_start; lane != group_end;
             ++lane) {
          if ((entropy_lane_mask &
               (1U << (lane - group_start))) &&
              (states[lane] < kRansLowerBound)) {
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
          if (!lane_offsets[lane]) {
            SetError(
                "Internal rANS lane merge underflow.", error);
            return false;
          }
          const uint8_t input_byte =
              lane_payloads[lane][--lane_offsets[lane]];
          payload->push_back(input_byte);
          states[lane] =
              (states[lane] << 8) | input_byte;
        }
      }
    }
  }
  for (uint32_t lane = 0; lane != state_ct; ++lane) {
    if (lane_offsets[lane] ||
        (states[lane] != kRansLowerBound)) {
      SetError("Internal rANS lane merge did not terminate.",
               error);
      return false;
    }
  }
  return true;
}

template <uint32_t kScaleBits>
PGEN_RANS_ALWAYS_INLINE bool RansDecodeSymbol(
    const ModelRow& row, uint32_t runtime_scale_bits,
    const uint8_t* lane_start, const uint8_t** lane_iter,
    uint32_t* state, uint8_t* symbol, std::string* error) {
  const uint32_t scale_bits =
      kScaleBits ? kScaleBits : runtime_scale_bits;
  const uint32_t slot = *state & ((1U << scale_bits) - 1);
  // ParseModel() guarantees monotonically increasing cumulative
  // frequencies ending at 1 << scale_bits.  Repeated boundaries skip
  // zero-frequency symbols, so these comparisons replace a branchy
  // four-symbol interval search.
  const uint32_t selected_symbol =
      static_cast<uint32_t>(slot >= row.cumulative[1]) +
      static_cast<uint32_t>(slot >= row.cumulative[2]) +
      static_cast<uint32_t>(slot >= row.cumulative[3]);
  const uint32_t frequency = row.frequencies[selected_symbol];
  *state = frequency * (*state >> scale_bits) + slot -
           row.cumulative[selected_symbol];
  while (*state < kRansLowerBound) {
    if (*lane_iter == lane_start) {
      SetError("Truncated rANS lane payload.", error);
      return false;
    }
    *state = (*state << 8) | *--(*lane_iter);
  }
  *symbol = static_cast<uint8_t>(selected_symbol);
  return true;
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

template <RecordMode kMode, uint32_t kGroup, bool kInterleaved,
          bool kAllEntropy>
PGEN_RANS_TARGET_AVX512 PGEN_RANS_ALWAYS_INLINE
bool RansDecodeGroup16Avx512(
    const Avx512DecodeModel12& decode_model,
    uint64_t reference1_word,
    uint64_t reference2_word,
    const uint8_t* lane_base, uint32_t lane_payload_size,
    const __m512i& lane_start_offsets,
    __m512i* lane_iter_offsets, __m512i* state,
    const uint8_t** interleaved_iter,
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
  if constexpr (kMode == RecordMode::kMarginal) {
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
  if constexpr (
      (kMode != RecordMode::kMarginal) && !kAllEntropy) {
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
  if constexpr (kInterleaved) {
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
      // Fifteen zero padding bytes at the end of an interleaved
      // record make this 16-byte load safe even for the final refill.
      const __m128i packed_bytes = _mm_loadu_si128(
          reinterpret_cast<const __m128i*>(*interleaved_iter));
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
  } else {
    // AVX-512 has no byte gather, but a dword gather whose low byte is
    // still substantially cheaper than spilling every state and
    // chasing individual lane pointers.
    while (renormalization_mask) {
      const __mmask16 available_mask =
          _mm512_cmp_epu32_mask(
              *lane_iter_offsets, lane_start_offsets,
              _MM_CMPINT_GT);
      if (renormalization_mask &
          static_cast<__mmask16>(~available_mask)) {
        SetError("Truncated rANS lane payload.", error);
        return false;
      }
      const __m512i next_offsets = _mm512_mask_sub_epi32(
          *lane_iter_offsets, renormalization_mask,
          *lane_iter_offsets, _mm512_set1_epi32(1));
      __mmask16 gather_mask = 0;
      if (lane_payload_size >= sizeof(uint32_t)) {
        gather_mask =
            renormalization_mask &
            _mm512_cmp_epu32_mask(
                next_offsets,
                _mm512_set1_epi32(
                    static_cast<int>(
                        lane_payload_size - sizeof(uint32_t))),
                _MM_CMPINT_LE);
      }
      const __m512i input_dwords = _mm512_mask_i32gather_epi32(
          _mm512_setzero_si512(), gather_mask,
          next_offsets, lane_base, 1);
      const __m512i renormalized_states = _mm512_or_si512(
          _mm512_slli_epi32(*state, 8),
          _mm512_and_si512(
              input_dwords, _mm512_set1_epi32(255)));
      *state = _mm512_mask_mov_epi32(
          *state, gather_mask,
          renormalized_states);
      *lane_iter_offsets = _mm512_mask_mov_epi32(
          *lane_iter_offsets, renormalization_mask,
          next_offsets);
      __mmask16 scalar_mask =
          renormalization_mask &
          static_cast<__mmask16>(~gather_mask);
      if (scalar_mask) {
        alignas(64) uint32_t state_lanes[16];
        alignas(64) uint32_t iter_offsets[16];
        _mm512_store_si512(state_lanes, *state);
        _mm512_store_si512(
            iter_offsets, *lane_iter_offsets);
        while (scalar_mask) {
          const uint32_t lane = static_cast<uint32_t>(
              __builtin_ctz(
                  static_cast<uint32_t>(scalar_mask)));
          scalar_mask = static_cast<__mmask16>(
              scalar_mask & (scalar_mask - 1));
          state_lanes[lane] =
              (state_lanes[lane] << 8) |
              lane_base[iter_offsets[lane]];
        }
        *state = _mm512_load_si512(state_lanes);
      }
      renormalization_mask =
          entropy_mask &
          _mm512_cmp_epu32_mask(
              *state, _mm512_set1_epi32(kRansLowerBound),
              _MM_CMPINT_LT);
    }
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

template <RecordMode kMode>
PGEN_RANS_TARGET_AVX512
bool DecodeEntropyWords32Avx512(
    const Model& model, const uint64_t* reference1,
    const uint64_t* reference2, uint32_t sample_ct,
    const std::array<const uint8_t*, 256>& lane_starts,
    std::array<const uint8_t*, 256>* lane_iters,
    std::array<uint32_t, 256>* states, uint64_t* target,
    std::string* error) {
  Avx512DecodeModel12 decode_model;
  BuildAvx512DecodeModel12(model, &decode_model);
  const uint8_t* const lane_base = lane_starts[0];
  alignas(64) uint32_t lane_start_offset_values[32];
  alignas(64) uint32_t lane_iter_offset_values[32];
  for (uint32_t lane = 0; lane != 32; ++lane) {
    lane_start_offset_values[lane] = static_cast<uint32_t>(
        lane_starts[lane] - lane_base);
    lane_iter_offset_values[lane] = static_cast<uint32_t>(
        (*lane_iters)[lane] - lane_base);
  }
  const uint32_t lane_payload_size =
      lane_iter_offset_values[31];
  const __m512i lane_start_offsets0 =
      _mm512_load_si512(lane_start_offset_values);
  const __m512i lane_start_offsets1 =
      _mm512_load_si512(lane_start_offset_values + 16);
  __m512i lane_iter_offsets0 =
      _mm512_load_si512(lane_iter_offset_values);
  __m512i lane_iter_offsets1 =
      _mm512_load_si512(lane_iter_offset_values + 16);
  __m512i state0 = _mm512_loadu_si512(&(states->at(0)));
  __m512i state1 = _mm512_loadu_si512(&(states->at(16)));
  uint32_t absent_context_mask = 0;
  const uint32_t full_word_ct = sample_ct / 32;
  for (uint32_t word_idx = 0; word_idx != full_word_ct; ++word_idx) {
    const uint64_t reference1_word =
        (kMode == RecordMode::kMarginal) ? 0 : reference1[word_idx];
    const uint64_t reference2_word =
        (kMode == RecordMode::kTwoReference) ? reference2[word_idx] : 0;
    uint32_t packed0;
    uint32_t packed1;
    if (!RansDecodeGroup16Avx512<kMode, 0, false, false>(
            decode_model, reference1_word, reference2_word,
            lane_base, lane_payload_size, lane_start_offsets0,
            &lane_iter_offsets0, &state0, nullptr, nullptr,
            &packed0,
            &absent_context_mask, error) ||
        !RansDecodeGroup16Avx512<kMode, 1, false, false>(
            decode_model, reference1_word, reference2_word,
            lane_base, lane_payload_size, lane_start_offsets1,
            &lane_iter_offsets1, &state1, nullptr, nullptr,
            &packed1,
            &absent_context_mask, error)) {
      return false;
    }
    target[word_idx] =
        static_cast<uint64_t>(packed0) |
        (static_cast<uint64_t>(packed1) << 32);
  }
  _mm512_storeu_si512(&(states->at(0)), state0);
  _mm512_storeu_si512(&(states->at(16)), state1);
  _mm512_store_si512(
      lane_iter_offset_values, lane_iter_offsets0);
  _mm512_store_si512(
      lane_iter_offset_values + 16, lane_iter_offsets1);
  for (uint32_t lane = 0; lane != 32; ++lane) {
    (*lane_iters)[lane] =
        lane_base + lane_iter_offset_values[lane];
  }

  const uint32_t tail_sample_ct = sample_ct % 32;
  if (tail_sample_ct) {
    const uint32_t word_idx = full_word_ct;
    const uint64_t reference1_word =
        (kMode == RecordMode::kMarginal) ? 0 : reference1[word_idx];
    const uint64_t reference2_word =
        (kMode == RecordMode::kTwoReference) ? reference2[word_idx] : 0;
    uint64_t packed_word = 0;
    for (uint32_t lane = 0; lane != tail_sample_ct; ++lane) {
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
      const uint32_t slot =
          (*states)[lane] & (kDefaultSlotCt - 1);
      const uint32_t active_symbol_ct =
          model.active_symbol_cts[context];
      if (!active_symbol_ct) {
        absent_context_mask |= kDecodeTableAbsentFlag;
      }
      uint8_t symbol = model.deterministic_symbols[context];
      if (active_symbol_ct > 1) {
        const ModelRow& row = model.rows[context];
        symbol = static_cast<uint8_t>(
            static_cast<uint32_t>(
                slot >= row.cumulative[1]) +
            static_cast<uint32_t>(
                slot >= row.cumulative[2]) +
            static_cast<uint32_t>(
                slot >= row.cumulative[3]));
        const uint32_t frequency = row.frequencies[symbol];
        const uint32_t cumulative = row.cumulative[symbol];
        (*states)[lane] =
            frequency *
                ((*states)[lane] >> kDefaultScaleBits) +
            slot - cumulative;
        while ((*states)[lane] < kRansLowerBound) {
          if ((*lane_iters)[lane] == lane_starts[lane]) {
            SetError("Truncated rANS lane payload.", error);
            return false;
          }
          (*states)[lane] =
              ((*states)[lane] << 8) |
              *--((*lane_iters)[lane]);
        }
      }
      packed_word |= static_cast<uint64_t>(symbol) << (2 * lane);
    }
    target[word_idx] = packed_word;
  }
  if (absent_context_mask) {
    SetError("Reference selects an absent model context.", error);
    return false;
  }
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
  const __m512i unused_offsets = _mm512_setzero_si512();
  __m512i unused_iter_offsets = _mm512_setzero_si512();
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
            kMode, 0, true, kAllEntropy>(
            decode_model, reference1_word, reference2_word,
            nullptr, 0, unused_offsets, &unused_iter_offsets,
            &state0, &payload_iter, payload_end, &packed0,
            &absent_context_mask, error) ||
        !RansDecodeGroup16Avx512<
            kMode, 1, true, kAllEntropy>(
            decode_model, reference1_word, reference2_word,
            nullptr, 0, unused_offsets, &unused_iter_offsets,
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

PGEN_RANS_TARGET_AVX512
bool DecodeEntropyRecord32Avx512(
    RecordMode mode, const Model& model, const uint64_t* reference1,
    const uint64_t* reference2, uint32_t sample_ct,
    const std::array<const uint8_t*, 256>& lane_starts,
    std::array<const uint8_t*, 256>* lane_iters,
    std::array<uint32_t, 256>* states, uint64_t* target,
    std::string* error) {
  switch (mode) {
    case RecordMode::kMarginal:
      return DecodeEntropyWords32Avx512<RecordMode::kMarginal>(
          model, reference1, reference2, sample_ct, lane_starts,
          lane_iters, states, target, error);
    case RecordMode::kOneReference:
      return DecodeEntropyWords32Avx512<RecordMode::kOneReference>(
          model, reference1, reference2, sample_ct, lane_starts,
          lane_iters, states, target, error);
    case RecordMode::kTwoReference:
      return DecodeEntropyWords32Avx512<RecordMode::kTwoReference>(
          model, reference1, reference2, sample_ct, lane_starts,
          lane_iters, states, target, error);
  }
  SetError("Unknown rANS record mode.", error);
  return false;
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
        if constexpr (kMode != RecordMode::kMarginal) {
          context =
              GetPackedGenotype(reference1, sample_idx);
        }
        if constexpr (kMode == RecordMode::kTwoReference) {
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

template <RecordMode kMode, uint32_t kScaleBits, bool kAllEntropy>
bool DecodeEntropyWords32(
    const Model& model, const uint64_t* reference1,
    const uint64_t* reference2, uint32_t sample_ct,
    uint32_t runtime_scale_bits,
    const std::array<const uint8_t*, 256>& lane_starts,
    std::array<const uint8_t*, 256>* lane_iters,
    std::array<uint32_t, 256>* states, uint64_t* target,
    std::string* error) {
  uint16_t observed_context_mask =
      (kMode == RecordMode::kMarginal) ? 1 : 0;
  const uint32_t full_word_ct = sample_ct / 32;
  for (uint32_t word_idx = 0; word_idx != full_word_ct; ++word_idx) {
    const uint64_t reference1_word =
        (kMode == RecordMode::kMarginal) ? 0 : reference1[word_idx];
    const uint64_t reference2_word =
        (kMode == RecordMode::kTwoReference) ? reference2[word_idx] : 0;
    uint64_t packed_word = 0;
    for (uint32_t lane = 0; lane != 32; ++lane) {
      uint32_t context = 0;
      if (kMode != RecordMode::kMarginal) {
        context =
            static_cast<uint32_t>((reference1_word >> (2 * lane)) & 3U);
      }
      if (kMode == RecordMode::kTwoReference) {
        context =
            4 * context +
            static_cast<uint32_t>(
                (reference2_word >> (2 * lane)) & 3U);
      }
      if (kMode != RecordMode::kMarginal) {
        observed_context_mask |=
            static_cast<uint16_t>(1U << context);
      }
      uint8_t symbol;
      if (kAllEntropy ||
          (model.entropy_context_mask & (1U << context))) {
        if (!RansDecodeSymbol<kScaleBits>(
                model.rows[context], runtime_scale_bits,
                lane_starts[lane], &((*lane_iters)[lane]),
                &((*states)[lane]), &symbol, error)) {
          return false;
        }
      } else {
        symbol = model.deterministic_symbols[context];
      }
      packed_word |= static_cast<uint64_t>(symbol) << (2 * lane);
    }
    target[word_idx] = packed_word;
  }
  const uint32_t tail_sample_ct = sample_ct % 32;
  if (tail_sample_ct) {
    const uint32_t word_idx = full_word_ct;
    const uint64_t reference1_word =
        (kMode == RecordMode::kMarginal) ? 0 : reference1[word_idx];
    const uint64_t reference2_word =
        (kMode == RecordMode::kTwoReference) ? reference2[word_idx] : 0;
    uint64_t packed_word = 0;
    for (uint32_t lane = 0; lane != tail_sample_ct; ++lane) {
      uint32_t context = 0;
      if (kMode != RecordMode::kMarginal) {
        context =
            static_cast<uint32_t>((reference1_word >> (2 * lane)) & 3U);
      }
      if (kMode == RecordMode::kTwoReference) {
        context =
            4 * context +
            static_cast<uint32_t>(
                (reference2_word >> (2 * lane)) & 3U);
      }
      if (kMode != RecordMode::kMarginal) {
        observed_context_mask |=
            static_cast<uint16_t>(1U << context);
      }
      uint8_t symbol;
      if (kAllEntropy ||
          (model.entropy_context_mask & (1U << context))) {
        if (!RansDecodeSymbol<kScaleBits>(
                model.rows[context], runtime_scale_bits,
                lane_starts[lane], &((*lane_iters)[lane]),
                &((*states)[lane]), &symbol, error)) {
          return false;
        }
      } else {
        symbol = model.deterministic_symbols[context];
      }
      packed_word |= static_cast<uint64_t>(symbol) << (2 * lane);
    }
    target[word_idx] = packed_word;
  }
  if (observed_context_mask & ~model.context_mask) {
    SetError("Reference selects an absent model context.", error);
    return false;
  }
  return true;
}

template <uint32_t kScaleBits, bool kAllEntropy>
bool DecodeEntropyRecord32(
    RecordMode mode, const Model& model, const uint64_t* reference1,
    const uint64_t* reference2, uint32_t sample_ct,
    uint32_t runtime_scale_bits,
    const std::array<const uint8_t*, 256>& lane_starts,
    std::array<const uint8_t*, 256>* lane_iters,
    std::array<uint32_t, 256>* states, uint64_t* target,
    std::string* error) {
  switch (mode) {
    case RecordMode::kMarginal:
      return DecodeEntropyWords32<
          RecordMode::kMarginal, kScaleBits, kAllEntropy>(
              model, reference1, reference2, sample_ct,
              runtime_scale_bits, lane_starts, lane_iters, states,
              target, error);
    case RecordMode::kOneReference:
      return DecodeEntropyWords32<
          RecordMode::kOneReference, kScaleBits, kAllEntropy>(
              model, reference1, reference2, sample_ct,
              runtime_scale_bits, lane_starts, lane_iters, states,
              target, error);
    case RecordMode::kTwoReference:
      return DecodeEntropyWords32<
          RecordMode::kTwoReference, kScaleBits, kAllEntropy>(
              model, reference1, reference2, sample_ct,
              runtime_scale_bits, lane_starts, lane_iters, states,
              target, error);
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
  const uint8_t mode_code = flags & kModeMask;
  if (mode_code > static_cast<uint8_t>(RecordMode::kTwoReference)) {
    SetError("Unknown rANS record mode.", error);
    return false;
  }
  metadata->mode = static_cast<RecordMode>(mode_code);
  metadata->has_entropy_payload = (flags & kEntropyPayloadFlag);
  metadata->has_interleaved_payload =
      (flags & kInterleavedPayloadFlag);
  if (metadata->has_interleaved_payload &&
      !metadata->has_entropy_payload) {
    SetError(
        "Interleaved record is missing the entropy-payload flag.",
        error);
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

bool EncodeRecord(const uint64_t* target, const uint64_t* reference1,
                  const uint64_t* reference2, uint32_t sample_ct,
                  RecordMode mode, uint8_t reference1_idx,
                  uint8_t reference2_idx, const CodecParams& params,
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
  if (!BuildModel(target, reference1, reference2, sample_ct, mode,
                  params.scale_bits, &model, error)) {
    return false;
  }
  uint8_t flags = static_cast<uint8_t>(mode);
  if (model.has_entropy) {
    flags |= kEntropyPayloadFlag | kInterleavedPayloadFlag;
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
    return true;
  }

  const uint32_t state_ct = std::min(params.state_ct, sample_ct);
  std::vector<uint32_t> states(state_ct, kRansLowerBound);
  std::vector<std::vector<uint8_t>> lane_payloads(state_ct);
  for (uint32_t lane = 0; lane != state_ct; ++lane) {
    uint32_t sample_idx =
        lane + ((sample_ct - 1 - lane) / state_ct) * state_ct;
    while (true) {
      const uint32_t context =
          ContextIndex(mode, reference1, reference2, sample_idx);
      const ModelRow& row = model.rows[context];
      if (model.active_symbol_cts[context] > 1) {
        const uint32_t symbol = GetPackedGenotype(target, sample_idx);
        RansEncodeSymbol(row.cumulative[symbol], row.frequencies[symbol],
                         params.scale_bits, &(states[lane]),
                         &(lane_payloads[lane]));
      }
      if (sample_idx < state_ct) {
        break;
      }
      sample_idx -= state_ct;
    }
  }
  for (const uint32_t state : states) {
    AppendU32(state, record);
  }
  std::vector<uint8_t> interleaved_payload;
  if (!BuildInterleavedPayload(
          target, reference1, reference2, sample_ct, mode, model,
          params.scale_bits, states, lane_payloads,
          &interleaved_payload, error)) {
    record->clear();
    return false;
  }
  record->insert(
      record->end(), interleaved_payload.begin(),
      interleaved_payload.end());
  record->insert(
      record->end(), kInterleavedPaddingByteCt, 0);
  return true;
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
    result +=
        4LLU * state_ct + kInterleavedPaddingByteCt;
    result += static_cast<uint64_t>(std::ceil(quantized_bits / 8.0L));
  }
  *record_bytes = result;
  return true;
}

bool ParseRecordMetadata(const uint8_t* record, size_t record_size,
                         RecordMetadata* metadata, std::string* error) {
  size_t offset;
  return ParsePrefix(record, record_size, metadata, &offset, error);
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
  size_t offset;
  if (!ParsePrefix(record, record_size, &parsed_metadata, &offset, error)) {
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
  if (!ParseModel(record, record_size, parsed_metadata.mode,
                  params.scale_bits, &offset, &model, error)) {
    return false;
  }
  if (model.has_entropy != parsed_metadata.has_entropy_payload) {
    SetError("Entropy-payload flag does not match the decoded model.", error);
    return false;
  }
  if (!model.has_entropy) {
    if (offset != record_size) {
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
    if (!ReadU32(record, record_size, &offset, &(states[lane])) ||
        (states[lane] < kRansLowerBound)) {
      SetError("Invalid or truncated rANS initial state.", error);
      return false;
    }
  }
  if (parsed_metadata.has_interleaved_payload) {
    if (record_size - offset < kInterleavedPaddingByteCt) {
      SetError("Truncated interleaved rANS padding.", error);
      return false;
    }
    const uint8_t* const payload_begin = record + offset;
    const uint8_t* const payload_end =
        record + record_size - kInterleavedPaddingByteCt;
    for (const uint8_t* padding_iter = payload_end;
         padding_iter != record + record_size; ++padding_iter) {
      if (*padding_iter) {
        SetError("Interleaved rANS padding is nonzero.", error);
        return false;
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
  std::array<uint32_t, 257> lane_boundaries = {};
  for (uint32_t lane = 0; lane + 1 != state_ct; ++lane) {
    if (!ReadU32(record, record_size, &offset,
                 &(lane_boundaries[lane + 1]))) {
      SetError("Truncated rANS lane boundary table.", error);
      return false;
    }
  }
  const size_t payload_size = record_size - offset;
  if (payload_size > UINT32_MAX) {
    SetError("rANS payload exceeds the v1 record limit.", error);
    return false;
  }
  lane_boundaries[state_ct] = static_cast<uint32_t>(payload_size);
  std::array<const uint8_t*, 256> lane_starts = {};
  std::array<const uint8_t*, 256> lane_iters = {};
  for (uint32_t lane = 0; lane != state_ct; ++lane) {
    if ((lane_boundaries[lane] > lane_boundaries[lane + 1]) ||
        (lane_boundaries[lane + 1] > payload_size)) {
      SetError("Invalid rANS lane boundary.", error);
      return false;
    }
    lane_starts[lane] = record + offset + lane_boundaries[lane];
    lane_iters[lane] =
        record + offset + lane_boundaries[lane + 1];
  }
  if (state_ct == 32) {
    bool used_vector_decoder = false;
    bool decode_ok = false;
#if PGEN_RANS_X86_RUNTIME_DISPATCH
    if ((params.scale_bits == kDefaultScaleBits) &&
        (sample_ct >= kAvx512MinimumSampleCt) &&
        HasAvx512Decoder()) {
      used_vector_decoder = true;
      decode_ok = DecodeEntropyRecord32Avx512(
          parsed_metadata.mode, model, reference1, reference2,
          sample_ct, lane_starts, &lane_iters, &states, target,
          error);
    }
#endif
    const bool all_contexts_have_entropy =
        model.entropy_context_mask == model.context_mask;
    if (!used_vector_decoder) {
      if (params.scale_bits == kDefaultScaleBits) {
        decode_ok =
            all_contexts_have_entropy
                ? DecodeEntropyRecord32<
                      kDefaultScaleBits, true>(
                      parsed_metadata.mode, model, reference1,
                      reference2, sample_ct, params.scale_bits,
                      lane_starts, &lane_iters, &states, target, error)
                : DecodeEntropyRecord32<
                      kDefaultScaleBits, false>(
                      parsed_metadata.mode, model, reference1,
                      reference2, sample_ct, params.scale_bits,
                      lane_starts, &lane_iters, &states, target, error);
      } else {
        decode_ok =
            all_contexts_have_entropy
                ? DecodeEntropyRecord32<0, true>(
                      parsed_metadata.mode, model, reference1,
                      reference2, sample_ct, params.scale_bits,
                      lane_starts, &lane_iters, &states, target, error)
                : DecodeEntropyRecord32<0, false>(
                      parsed_metadata.mode, model, reference1,
                      reference2, sample_ct, params.scale_bits,
                      lane_starts, &lane_iters, &states, target, error);
      }
    }
    if (!decode_ok) {
      return false;
    }
  } else {
    memset(target, 0,
           static_cast<size_t>(packed_word_ct) * sizeof(uint64_t));
    for (uint32_t first_sample = 0; first_sample < sample_ct;
         first_sample += state_ct) {
      const uint32_t round_sample_ct =
          std::min(state_ct, sample_ct - first_sample);
      for (uint32_t lane = 0; lane != round_sample_ct; ++lane) {
        const uint32_t sample_idx = first_sample + lane;
        const uint32_t context = ContextIndex(
            parsed_metadata.mode, reference1, reference2, sample_idx);
        const ModelRow& row = model.rows[context];
        const uint32_t active_symbol_ct =
            model.active_symbol_cts[context];
        if (!active_symbol_ct) {
          SetError("Reference selects an absent model context.", error);
          return false;
        }
        uint8_t symbol;
        if (active_symbol_ct == 1) {
          symbol = model.deterministic_symbols[context];
        } else if (!RansDecodeSymbol<0>(
                       row, params.scale_bits, lane_starts[lane],
                       &(lane_iters[lane]), &(states[lane]), &symbol,
                       error)) {
          return false;
        }
        SetPackedGenotype(target, sample_idx, symbol);
      }
    }
  }
  for (uint32_t lane = 0; lane != state_ct; ++lane) {
    if ((lane_iters[lane] != lane_starts[lane]) ||
        (states[lane] != kRansLowerBound)) {
      SetError("rANS lane did not terminate at its canonical state.", error);
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
