// SPDX-License-Identifier: GPL-3.0-or-later

#include "pgen_rans_codec.h"

#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

namespace {

using pgen_rans::CodecParams;
using pgen_rans::AppendMultiallelicPatches;
using pgen_rans::DecodeMultiallelicPatches;
using pgen_rans::DecodeRecord;
using pgen_rans::DecodeRecordToBuffer;
using pgen_rans::DecodeRecordToBufferFromValidatedBlock;
using pgen_rans::EncodeRecord;
using pgen_rans::EstimateRecordBytes;
using pgen_rans::GetPackedGenotype;
using pgen_rans::GetBaseRecordByteCt;
using pgen_rans::MultiallelicPatches;
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
  invalid_flags = record;
  invalid_flags[0] |= 0x08;
  Expect(!DecodeRecord(invalid_flags.data(), invalid_flags.size(), anchors, 8,
                       sample_ct, params, &decoded, nullptr, &error),
         "multiallelic flag without a patch suffix was accepted");
  if (metadata.has_entropy_payload) {
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

void TestMultiallelicPatches() {
  constexpr uint32_t kSampleCt = 1003;
  std::vector<uint8_t> target(kSampleCt);
  for (uint32_t sample_idx = 0; sample_idx != kSampleCt; ++sample_idx) {
    target[sample_idx] = static_cast<uint8_t>(sample_idx % 4);
  }
  const std::vector<uint64_t> packed_target = Pack(target);
  const CodecParams params;
  std::vector<uint8_t> record;
  std::string error;
  Expect(EncodeRecord(
             packed_target.data(), nullptr, nullptr, kSampleCt,
             RecordMode::kMarginal, 0, 0, params, &record, &error),
         "multiallelic base encode failed: " + error);
  const size_t original_size = record.size();
  MultiallelicPatches expected;
  expected.allele_ct = 5;
  expected.patch_01_sample_ids = {0, 127, 1002};
  expected.patch_01_values = {2, 3, 4};
  expected.patch_10_sample_ids = {1, 128, 1001};
  expected.patch_10_values = {1, 2, 4, 4, 2, 3};
  Expect(AppendMultiallelicPatches(
             kSampleCt, expected, &record, &error),
         "multiallelic patch encode failed: " + error);
  size_t base_size;
  Expect(GetBaseRecordByteCt(
             record.data(), record.size(), &base_size, &error),
         "base-record length parse failed: " + error);
  Expect(base_size == original_size, "base-record length mismatch");
  RecordMetadata metadata;
  std::vector<uint64_t> decoded;
  Expect(DecodeRecord(
             record.data(), record.size(), nullptr, 0, kSampleCt,
             params, &decoded, &metadata, &error),
         "patched base record decode failed: " + error);
  Expect(metadata.has_multiallelic_patches,
         "multiallelic metadata flag was not exposed");
  ExpectEqual(decoded, target, "patched base round-trip");
  MultiallelicPatches observed;
  Expect(DecodeMultiallelicPatches(
             record.data(), record.size(), kSampleCt, &observed,
             &error),
         "multiallelic patch decode failed: " + error);
  Expect(
      (observed.allele_ct == expected.allele_ct) &&
          (observed.patch_01_sample_ids ==
           expected.patch_01_sample_ids) &&
          (observed.patch_01_values == expected.patch_01_values) &&
          (observed.patch_10_sample_ids ==
           expected.patch_10_sample_ids) &&
          (observed.patch_10_values == expected.patch_10_values),
      "multiallelic patch round-trip mismatch");

  std::vector<uint8_t> corrupt_footer = record;
  corrupt_footer.back() ^= 1;
  Expect(!DecodeMultiallelicPatches(
             corrupt_footer.data(), corrupt_footer.size(), kSampleCt,
             &observed, &error),
         "corrupt multiallelic footer was accepted");
  std::vector<uint8_t> corrupt_version = record;
  corrupt_version[base_size] = 2;
  Expect(!DecodeMultiallelicPatches(
             corrupt_version.data(), corrupt_version.size(), kSampleCt,
             &observed, &error),
         "unknown multiallelic patch version was accepted");
  std::vector<uint8_t> truncated = record;
  truncated.pop_back();
  Expect(!DecodeRecord(
             truncated.data(), truncated.size(), nullptr, 0, kSampleCt,
             params, &decoded, nullptr, &error),
         "truncated multiallelic record was accepted");

  std::vector<uint8_t> invalid_record;
  Expect(EncodeRecord(
             packed_target.data(), nullptr, nullptr, kSampleCt,
             RecordMode::kMarginal, 0, 0, params, &invalid_record,
             &error),
         "invalid-patch base encode failed: " + error);
  MultiallelicPatches invalid = expected;
  invalid.patch_01_sample_ids = {7, 7, 8};
  Expect(!AppendMultiallelicPatches(
             kSampleCt, invalid, &invalid_record, &error),
         "duplicate multiallelic patch sample ID was accepted");
  invalid = expected;
  invalid.patch_10_sample_ids[0] =
      invalid.patch_01_sample_ids[0];
  Expect(!AppendMultiallelicPatches(
             kSampleCt, invalid, &invalid_record, &error),
         "overlapping multiallelic patch classes were accepted");

  std::vector<uint8_t> empty_patch_record;
  Expect(EncodeRecord(
             packed_target.data(), nullptr, nullptr, kSampleCt,
             RecordMode::kMarginal, 0, 0, params, &empty_patch_record,
             &error),
         "empty-patch base encode failed: " + error);
  MultiallelicPatches empty_patches;
  empty_patches.allele_ct = 3;
  Expect(AppendMultiallelicPatches(
             kSampleCt, empty_patches, &empty_patch_record, &error),
         "empty multiallelic patch encode failed: " + error);
  Expect(DecodeMultiallelicPatches(
             empty_patch_record.data(), empty_patch_record.size(),
             kSampleCt, &observed, &error),
         "empty multiallelic patch decode failed: " + error);
  Expect((observed.allele_ct == 3) &&
             observed.patch_01_sample_ids.empty() &&
             observed.patch_10_sample_ids.empty(),
         "empty multiallelic patch schema was not retained");

  std::vector<uint8_t> dense_patch_record;
  Expect(EncodeRecord(
             packed_target.data(), nullptr, nullptr, kSampleCt,
             RecordMode::kMarginal, 0, 0, params,
             &dense_patch_record, &error),
         "dense-patch base encode failed: " + error);
  MultiallelicPatches dense_patches;
  dense_patches.allele_ct = 3;
  for (uint32_t sample_idx = 0; sample_idx != kSampleCt;
       ++sample_idx) {
    if (sample_idx % 3) {
      dense_patches.patch_10_sample_ids.push_back(sample_idx);
      dense_patches.patch_10_values.push_back(1);
      dense_patches.patch_10_values.push_back(2);
    }
  }
  Expect(AppendMultiallelicPatches(
             kSampleCt, dense_patches, &dense_patch_record, &error),
         "dense multiallelic patch encode failed: " + error);
  Expect(DecodeMultiallelicPatches(
             dense_patch_record.data(), dense_patch_record.size(),
             kSampleCt, &observed, &error),
         "dense multiallelic patch decode failed: " + error);
  Expect(
      (observed.patch_10_sample_ids ==
       dense_patches.patch_10_sample_ids) &&
          (observed.patch_10_values ==
           dense_patches.patch_10_values),
      "dense multiallelic bitmap round-trip mismatch");
}

}  // namespace

int main() {
  TestRandomRoundTrips();
  TestDeterministicRecords();
  TestLargeDefaultRoundTrips();
  TestInvalidArguments();
  TestMultiallelicPatches();
  puts("pgen_rans_codec_test: PASS");
  return 0;
}
