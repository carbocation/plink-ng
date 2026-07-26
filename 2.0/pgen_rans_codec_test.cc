// SPDX-License-Identifier: GPL-3.0-or-later

#include "pgen_rans_codec.h"

#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

namespace {

using pgen_rans::CodecParams;
using pgen_rans::DecodeRecord;
using pgen_rans::DecodeRecordToBuffer;
using pgen_rans::DecodeRecordToBufferFromValidatedBlock;
using pgen_rans::EncodeRecord;
using pgen_rans::EstimateRecordBytes;
using pgen_rans::GetPackedGenotype;
using pgen_rans::PackedWordCt;
using pgen_rans::RecordMetadata;
using pgen_rans::RecordMode;
using pgen_rans::SetPackedGenotype;

[[noreturn]] void Fail(const std::string& message) {
  fprintf(stderr, "FAIL: %s\n", message.c_str());
  exit(1);
}

void Expect(bool condition, const std::string& message) {
  if (!condition) {
    Fail(message);
  }
}

std::vector<uint64_t> Pack(const std::vector<uint8_t>& genotypes) {
  const uint32_t sample_ct = static_cast<uint32_t>(genotypes.size());
  std::vector<uint64_t> packed(PackedWordCt(sample_ct), 0);
  for (uint32_t sample_idx = 0; sample_idx != genotypes.size();
       ++sample_idx) {
    SetPackedGenotype(packed.data(), sample_idx, genotypes[sample_idx]);
  }
  return packed;
}

void ExpectEqual(const std::vector<uint64_t>& packed,
                 const std::vector<uint8_t>& expected,
                 const std::string& label) {
  for (uint32_t sample_idx = 0; sample_idx != expected.size();
       ++sample_idx) {
    if (GetPackedGenotype(packed.data(), sample_idx) !=
        expected[sample_idx]) {
      Fail(label + " mismatch at sample " + std::to_string(sample_idx));
    }
  }
}

void RoundTrip(const std::vector<uint8_t>& target,
               const std::vector<uint8_t>& reference1,
               const std::vector<uint8_t>& reference2, RecordMode mode,
               const CodecParams& params) {
  const std::vector<uint64_t> packed_target = Pack(target);
  const std::vector<uint64_t> packed_reference1 = Pack(reference1);
  const std::vector<uint64_t> packed_reference2 = Pack(reference2);
  const uint32_t sample_ct = static_cast<uint32_t>(target.size());
  std::vector<uint8_t> record;
  std::string error;
  const uint64_t* reference1_ptr =
      (mode == RecordMode::kMarginal) ? nullptr : packed_reference1.data();
  const uint64_t* reference2_ptr =
      (mode == RecordMode::kTwoReference)
          ? packed_reference2.data()
          : nullptr;
  Expect(EncodeRecord(packed_target.data(), reference1_ptr, reference2_ptr,
                      sample_ct, mode, 3, 7, params, &record, &error),
         "encode failed: " + error);
  std::vector<uint8_t> second_record;
  Expect(EncodeRecord(packed_target.data(), reference1_ptr, reference2_ptr,
                      sample_ct, mode, 3, 7, params, &second_record,
                      &error),
         "second encode failed: " + error);
  Expect(record == second_record, "encoding is not deterministic");

  std::vector<uint32_t> counts(
      4 * ((mode == RecordMode::kMarginal)
               ? 1
               : ((mode == RecordMode::kOneReference) ? 4 : 16)),
      0);
  for (uint32_t sample_idx = 0; sample_idx != target.size(); ++sample_idx) {
    uint32_t context = 0;
    if (mode != RecordMode::kMarginal) {
      context = reference1[sample_idx];
    }
    if (mode == RecordMode::kTwoReference) {
      context = 4 * context + reference2[sample_idx];
    }
    ++counts[4 * context + target[sample_idx]];
  }
  uint64_t estimated_bytes;
  Expect(EstimateRecordBytes(counts.data(), mode, params, &estimated_bytes,
                             &error),
         "size estimate failed: " + error);
  const uint64_t size_delta =
      (estimated_bytes > record.size())
          ? (estimated_bytes - record.size())
          : (record.size() - estimated_bytes);
  Expect(size_delta <= params.state_ct,
         "size estimate differs from encoded record by too much");

  const uint64_t* anchors[8] = {};
  anchors[3] = packed_reference1.data();
  anchors[7] = packed_reference2.data();
  std::vector<uint64_t> decoded;
  RecordMetadata metadata;
  Expect(DecodeRecord(record.data(), record.size(), anchors, 8, sample_ct,
                      params, &decoded, &metadata, &error),
         "decode failed: " + error);
  Expect(metadata.mode == mode, "decoded mode mismatch");
  Expect(metadata.has_interleaved_payload ==
             metadata.has_entropy_payload,
         "interleaved-payload flag mismatch");
  ExpectEqual(decoded, target, "round-trip");

  const uint32_t packed_word_ct = PackedWordCt(sample_ct);
  std::vector<uint64_t> decoded_buffer(packed_word_ct + 1,
                                       0xdec0dedec0dedULL);
  Expect(DecodeRecordToBuffer(
             record.data(), record.size(), anchors, 8, sample_ct, params,
             decoded_buffer.data(), packed_word_ct, &metadata, &error),
         "buffer decode failed: " + error);
  ExpectEqual(decoded_buffer, target, "buffer round-trip");
  Expect(decoded_buffer.back() == 0xdec0dedec0dedULL,
         "buffer decoder wrote past its declared output");
  Expect(DecodeRecordToBufferFromValidatedBlock(
             record.data(), record.size(), anchors, 8, sample_ct,
             params, decoded_buffer.data(), packed_word_ct, &metadata,
             &error),
         "validated-block decode failed: " + error);
  ExpectEqual(decoded_buffer, target, "validated-block round-trip");
  if (packed_word_ct > 1) {
    Expect(!DecodeRecordToBuffer(
               record.data(), record.size(), anchors, 8, sample_ct, params,
               decoded_buffer.data(), packed_word_ct - 1, nullptr, &error),
           "undersized decode buffer was accepted");
  }

  if (record.size() > 1) {
    Expect(!DecodeRecord(record.data(), record.size() - 1, anchors, 8,
                         sample_ct, params, &decoded, nullptr, &error),
           "truncated record was accepted");
  }
  std::vector<uint8_t> invalid_flags = record;
  invalid_flags[0] |= 0x80;
  Expect(!DecodeRecord(invalid_flags.data(), invalid_flags.size(), anchors, 8,
                       sample_ct, params, &decoded, nullptr, &error),
         "unknown record flag was accepted");
  if (metadata.has_interleaved_payload) {
    std::vector<uint8_t> invalid_padding = record;
    invalid_padding.back() = 1;
    Expect(!DecodeRecord(invalid_padding.data(), invalid_padding.size(),
                         anchors, 8, sample_ct, params, &decoded,
                         nullptr, &error),
           "nonzero interleaved padding was accepted");
  }
}

void TestRandomRoundTrips() {
  std::mt19937_64 rng(0x7067656e72616e73ULL);
  const uint32_t sample_counts[] = {1, 2, 31, 32, 33, 257, 4099};
  const CodecParams parameter_sets[] = {
      {1, 8}, {7, 10}, {32, 12}, {32, 16}};
  for (const uint32_t sample_ct : sample_counts) {
    std::vector<uint8_t> reference1(sample_ct);
    std::vector<uint8_t> reference2(sample_ct);
    std::vector<uint8_t> marginal(sample_ct);
    std::vector<uint8_t> one_reference(sample_ct);
    std::vector<uint8_t> two_reference(sample_ct);
    for (uint32_t sample_idx = 0; sample_idx != sample_ct; ++sample_idx) {
      reference1[sample_idx] = rng() % 4;
      reference2[sample_idx] = rng() % 4;
      marginal[sample_idx] = rng() % 4;
      one_reference[sample_idx] =
          ((rng() % 8) == 0) ? (rng() % 4) : reference1[sample_idx];
      two_reference[sample_idx] =
          ((rng() % 10) == 0)
              ? (rng() % 4)
              : ((reference1[sample_idx] + reference2[sample_idx]) % 3);
    }
    for (const CodecParams& params : parameter_sets) {
      RoundTrip(marginal, reference1, reference2, RecordMode::kMarginal,
                params);
      RoundTrip(one_reference, reference1, reference2,
                RecordMode::kOneReference, params);
      RoundTrip(two_reference, reference1, reference2,
                RecordMode::kTwoReference, params);
    }
  }
}

void TestDeterministicRecords() {
  constexpr uint32_t kSampleCt = 1000;
  std::vector<uint8_t> target(kSampleCt, 0);
  std::vector<uint8_t> reference1(kSampleCt);
  std::vector<uint8_t> reference2(kSampleCt);
  for (uint32_t sample_idx = 0; sample_idx != kSampleCt; ++sample_idx) {
    reference1[sample_idx] = sample_idx % 4;
    reference2[sample_idx] = (sample_idx / 4) % 4;
  }
  const CodecParams params;
  RoundTrip(target, reference1, reference2, RecordMode::kMarginal, params);
  for (uint32_t sample_idx = 0; sample_idx != kSampleCt; ++sample_idx) {
    target[sample_idx] = reference1[sample_idx];
  }
  RoundTrip(target, reference1, reference2, RecordMode::kOneReference,
            params);
  for (uint32_t sample_idx = 0; sample_idx != kSampleCt; ++sample_idx) {
    target[sample_idx] =
        (reference1[sample_idx] + reference2[sample_idx]) % 4;
  }
  RoundTrip(target, reference1, reference2, RecordMode::kTwoReference,
            params);
}

void TestLargeDefaultRoundTrips() {
  constexpr uint32_t kSampleCt = 32769;
  std::mt19937_64 rng(0x6176783531327261ULL);
  std::vector<uint8_t> reference1(kSampleCt);
  std::vector<uint8_t> reference2(kSampleCt);
  std::vector<uint8_t> marginal(kSampleCt);
  std::vector<uint8_t> one_reference(kSampleCt);
  std::vector<uint8_t> two_reference(kSampleCt);
  for (uint32_t sample_idx = 0; sample_idx != kSampleCt; ++sample_idx) {
    reference1[sample_idx] = rng() % 4;
    reference2[sample_idx] = rng() % 4;
    marginal[sample_idx] = rng() % 4;
    one_reference[sample_idx] =
        (reference1[sample_idx] < 2)
            ? reference1[sample_idx]
            : (rng() % 4);
    const uint32_t context =
        4 * reference1[sample_idx] + reference2[sample_idx];
    two_reference[sample_idx] =
        (context & 1) ? (context % 4) : (rng() % 4);
  }
  const CodecParams params;
  RoundTrip(marginal, reference1, reference2, RecordMode::kMarginal,
            params);
  RoundTrip(one_reference, reference1, reference2,
            RecordMode::kOneReference, params);
  RoundTrip(two_reference, reference1, reference2,
            RecordMode::kTwoReference, params);
}

void TestInvalidArguments() {
  const std::vector<uint64_t> genotypes = Pack({0, 1, 2, 3});
  std::vector<uint8_t> record;
  std::string error;
  CodecParams params;
  Expect(!EncodeRecord(genotypes.data(), nullptr, nullptr, 4,
                       RecordMode::kOneReference, 0, 0, params, &record,
                       &error),
         "missing one-reference input was accepted");
  Expect(!EncodeRecord(genotypes.data(), genotypes.data(), genotypes.data(), 4,
                       RecordMode::kTwoReference, 1, 1, params, &record,
                       &error),
         "duplicate two-reference selectors were accepted");
  params.scale_bits = 7;
  Expect(!EncodeRecord(genotypes.data(), nullptr, nullptr, 4,
                       RecordMode::kMarginal, 0, 0, params, &record, &error),
         "invalid scale precision was accepted");
}

}  // namespace

int main() {
  TestRandomRoundTrips();
  TestDeterministicRecords();
  TestLargeDefaultRoundTrips();
  TestInvalidArguments();
  puts("pgen_rans_codec_test: PASS");
  return 0;
}
