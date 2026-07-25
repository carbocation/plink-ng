// SPDX-License-Identifier: GPL-3.0-or-later

#include "pgen_rans_codec.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>

namespace pgen_rans {
namespace {

constexpr uint32_t kRansLowerBound = 1U << 23;
constexpr uint8_t kModeMask = 3;
constexpr uint8_t kEntropyPayloadFlag = 1U << 2;
constexpr uint8_t kKnownFlagMask = kModeMask | kEntropyPayloadFlag;

struct ModelRow {
  std::array<uint32_t, 4> frequencies = {};
  std::array<uint32_t, 4> cumulative = {};
  uint8_t symbol_mask = 0;
  uint8_t deterministic_symbol = 0;
  uint8_t active_symbol_ct = 0;
};

struct Model {
  std::array<ModelRow, 16> rows;
  uint16_t context_mask = 0;
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
                  ModelRow* row, std::string* error) {
  const uint32_t total_frequency = 1U << scale_bits;
  uint64_t total_count = 0;
  std::array<double, 4> raw_frequencies = {};
  std::array<uint32_t, 4> normalized = {};
  for (uint32_t symbol = 0; symbol != 4; ++symbol) {
    total_count += counts[symbol];
    if (counts[symbol]) {
      row->symbol_mask |= static_cast<uint8_t>(1U << symbol);
      row->deterministic_symbol = static_cast<uint8_t>(symbol);
      ++row->active_symbol_ct;
    }
  }
  if (!total_count) {
    SetError("Cannot normalize an empty model row.", error);
    return false;
  }
  if (row->active_symbol_ct == 1) {
    row->frequencies[row->deterministic_symbol] =
        total_frequency;
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
    row->frequencies[symbol] = normalized[symbol];
    row->cumulative[symbol] = cumulative;
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
    if (!NormalizeRow(&(counts[4 * context]), scale_bits,
                      &(model->rows[context]), error)) {
      return false;
    }
    model->has_entropy |=
        (model->rows[context].active_symbol_ct > 1);
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
    output->push_back(row.symbol_mask);
    uint32_t remaining = row.active_symbol_ct;
    for (uint32_t symbol = 0; symbol != 4; ++symbol) {
      if (!(row.symbol_mask & (1U << symbol))) {
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
    row.symbol_mask = input[(*offset)++];
    if ((!row.symbol_mask) || (row.symbol_mask & 0xf0U)) {
      SetError("Invalid model symbol mask.", error);
      return false;
    }
    for (uint32_t symbol = 0; symbol != 4; ++symbol) {
      if (row.symbol_mask & (1U << symbol)) {
        row.deterministic_symbol = static_cast<uint8_t>(symbol);
        ++row.active_symbol_ct;
      }
    }
    uint32_t frequency_sum = 0;
    uint32_t remaining = row.active_symbol_ct;
    for (uint32_t symbol = 0; symbol != 4; ++symbol) {
      if (!(row.symbol_mask & (1U << symbol))) {
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
    model->has_entropy |= (row.active_symbol_ct > 1);
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

bool RansDecodeSymbol(const ModelRow& row, uint32_t scale_bits,
                      const uint8_t* lane_start, const uint8_t** lane_iter,
                      uint32_t* state, uint8_t* symbol, std::string* error) {
  const uint32_t slot = *state & ((1U << scale_bits) - 1);
  uint32_t selected_symbol = 4;
  for (uint32_t candidate = 0; candidate != 4; ++candidate) {
    const uint32_t frequency = row.frequencies[candidate];
    if (frequency && (slot >= row.cumulative[candidate]) &&
        (slot < row.cumulative[candidate] + frequency)) {
      selected_symbol = candidate;
      break;
    }
  }
  if (selected_symbol == 4) {
    SetError("rANS slot is not covered by the decoded model.", error);
    return false;
  }
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
    flags |= kEntropyPayloadFlag;
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
      if (row.active_symbol_ct > 1) {
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
  uint64_t cumulative_payload_size = 0;
  for (uint32_t lane = 0; lane + 1 != state_ct; ++lane) {
    cumulative_payload_size += lane_payloads[lane].size();
    if (cumulative_payload_size > UINT32_MAX) {
      SetError("rANS lane payload exceeds the v1 record limit.", error);
      record->clear();
      return false;
    }
    AppendU32(static_cast<uint32_t>(cumulative_payload_size), record);
  }
  for (const std::vector<uint8_t>& lane_payload : lane_payloads) {
    record->insert(record->end(), lane_payload.begin(), lane_payload.end());
  }
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
    result += 1 + 2 * (row.active_symbol_ct - 1);
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
    result += 4LLU * state_ct + 4LLU * (state_ct - 1);
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

bool DecodeRecordToBuffer(const uint8_t* record, size_t record_size,
                          const uint64_t* const* anchors,
                          uint32_t anchor_ct, uint32_t sample_ct,
                          const CodecParams& params, uint64_t* target,
                          size_t target_word_ct, RecordMetadata* metadata,
                          std::string* error) {
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
  memset(target, 0, static_cast<size_t>(packed_word_ct) * sizeof(uint64_t));
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
        const ModelRow& row = model.rows[context];
        if (row.active_symbol_ct != 1) {
          SetError("Reference selects an absent deterministic context.",
                   error);
          return false;
        }
        packed_word |= static_cast<uint64_t>(row.deterministic_symbol)
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
    for (uint32_t word_idx = 0; word_idx != packed_word_ct; ++word_idx) {
      const uint32_t first_sample = 32 * word_idx;
      const uint32_t word_sample_ct =
          std::min(32U, sample_ct - first_sample);
      uint64_t packed_word = 0;
      for (uint32_t lane = 0; lane != word_sample_ct; ++lane) {
        const uint32_t sample_idx = first_sample + lane;
        const uint32_t context = ContextIndex(
            parsed_metadata.mode, reference1, reference2, sample_idx);
        const ModelRow& row = model.rows[context];
        if (!row.active_symbol_ct) {
          SetError("Reference selects an absent model context.", error);
          return false;
        }
        uint8_t symbol;
        if (row.active_symbol_ct == 1) {
          symbol = row.deterministic_symbol;
        } else if (!RansDecodeSymbol(
                       row, params.scale_bits, lane_starts[lane],
                       &(lane_iters[lane]), &(states[lane]), &symbol,
                       error)) {
          return false;
        }
        packed_word |= static_cast<uint64_t>(symbol) << (2 * lane);
      }
      target[word_idx] = packed_word;
    }
  } else {
    for (uint32_t first_sample = 0; first_sample < sample_ct;
         first_sample += state_ct) {
      const uint32_t round_sample_ct =
          std::min(state_ct, sample_ct - first_sample);
      for (uint32_t lane = 0; lane != round_sample_ct; ++lane) {
        const uint32_t sample_idx = first_sample + lane;
        const uint32_t context = ContextIndex(
            parsed_metadata.mode, reference1, reference2, sample_idx);
        const ModelRow& row = model.rows[context];
        if (!row.active_symbol_ct) {
          SetError("Reference selects an absent model context.", error);
          return false;
        }
        uint8_t symbol;
        if (row.active_symbol_ct == 1) {
          symbol = row.deterministic_symbol;
        } else if (!RansDecodeSymbol(
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
