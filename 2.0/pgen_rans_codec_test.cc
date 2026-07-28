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
using pgen_rans::DecodeKernel;
using pgen_rans::DecodeKernelSupported;
using pgen_rans::DecodeRecord;
using pgen_rans::DecodeRecordToBuffer;
using pgen_rans::DecodeRecordToBufferFromValidatedBlock;
using pgen_rans::EncodeKernel;
using pgen_rans::EncodeKernelSupported;
using pgen_rans::EncodeRecord;
using pgen_rans::EncodeRecordFromCounts;
using pgen_rans::EstimateRecordBytes;
using pgen_rans::GetPackedGenotype;
using pgen_rans::GetBaseRecordByteCt;
using pgen_rans::LastEncodeKernelForTesting;
using pgen_rans::LastDecodeKernelForTesting;
using pgen_rans::MultiallelicPatches;
using pgen_rans::PackedWordCt;
using pgen_rans::RecordMetadata;
using pgen_rans::RecordMode;
using pgen_rans::SetDecodeKernelForTesting;
using pgen_rans::SetEncodeKernelForTesting;
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
  std::vector<uint8_t> count_reused_record;
  Expect(EncodeRecordFromCounts(
             packed_target.data(), reference1_ptr, reference2_ptr,
             sample_ct, mode, 3, 7, counts.data(), params,
             &count_reused_record, &error),
         "count-reusing encode failed: " + error);
  Expect(record == count_reused_record,
         "count-reusing encode changed the record");
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
    std::vector<uint8_t> legacy_padded = record;
    legacy_padded[0] &= static_cast<uint8_t>(~0x10U);
    legacy_padded.insert(legacy_padded.end(), 15, 0);
    Expect(DecodeRecord(legacy_padded.data(), legacy_padded.size(),
                        anchors, 8, sample_ct, params, &decoded,
                        nullptr, &error),
           "legacy interleaved padding was rejected");
    std::vector<uint8_t> invalid_padding = legacy_padded;
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
      {1, 8}, {4, 12}, {7, 10}, {8, 12},
      {16, 12}, {32, 12}, {32, 16}};
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

void TestForcedDefaultEncodeKernels() {
  const bool has_avx2 =
      EncodeKernelSupported(EncodeKernel::kAvx2);
  const bool has_avx512 =
      EncodeKernelSupported(EncodeKernel::kAvx512);
  if ((!has_avx2) && (!has_avx512)) {
    return;
  }
  constexpr uint32_t kSampleCt = 65567;
  std::mt19937_64 rng(0x617678353132656eULL);
  std::vector<uint8_t> reference1(kSampleCt);
  std::vector<uint8_t> reference2(kSampleCt);
  std::vector<uint8_t> marginal(kSampleCt);
  std::vector<uint8_t> one_reference(kSampleCt);
  std::vector<uint8_t> two_reference(kSampleCt);
  for (uint32_t sample_idx = 0; sample_idx != kSampleCt; ++sample_idx) {
    reference1[sample_idx] = rng() % 4;
    reference2[sample_idx] = rng() % 4;
    marginal[sample_idx] = rng() % 4;
    switch (reference1[sample_idx]) {
      case 0:
        one_reference[sample_idx] = 2;
        break;
      case 1:
        one_reference[sample_idx] = rng() & 1U;
        break;
      case 2: {
        constexpr uint8_t kThreeSymbolAlphabet[] = {0, 2, 3};
        one_reference[sample_idx] =
            kThreeSymbolAlphabet[rng() % 3];
        break;
      }
      default:
        // Exercise a skewed entropy row with one absent symbol.
        one_reference[sample_idx] =
            ((rng() & 15U) == 0) ? 3 : 1;
        break;
    }
    const uint32_t context =
        4 * reference1[sample_idx] + reference2[sample_idx];
    switch (context & 3U) {
      case 0:
        two_reference[sample_idx] =
            static_cast<uint8_t>(context >> 2);
        break;
      case 1:
        two_reference[sample_idx] =
            (rng() & 1U) ? 0 : 3;
        break;
      case 2: {
        constexpr uint8_t kThreeSymbolAlphabet[] = {0, 1, 3};
        two_reference[sample_idx] =
            kThreeSymbolAlphabet[rng() % 3];
        break;
      }
      default:
        two_reference[sample_idx] = rng() % 4;
        break;
    }
  }
  const std::vector<uint8_t>* const targets[] = {
      &marginal, &one_reference, &two_reference};
  const RecordMode modes[] = {
      RecordMode::kMarginal, RecordMode::kOneReference,
      RecordMode::kTwoReference};
  const std::vector<uint64_t> packed_reference1 = Pack(reference1);
  const std::vector<uint64_t> packed_reference2 = Pack(reference2);
  for (const uint32_t state_ct : {16U, 32U}) {
    const CodecParams params(state_ct, 12);
    for (uint32_t mode_idx = 0; mode_idx != 3; ++mode_idx) {
      const RecordMode mode = modes[mode_idx];
      const std::vector<uint64_t> packed_target =
          Pack(*targets[mode_idx]);
      const uint64_t* reference1_ptr =
          (mode == RecordMode::kMarginal)
              ? nullptr
              : packed_reference1.data();
      const uint64_t* reference2_ptr =
          (mode == RecordMode::kTwoReference)
              ? packed_reference2.data()
              : nullptr;
      std::vector<uint8_t> scalar_record;
      std::vector<uint8_t> vector_record;
      std::string error;
      SetEncodeKernelForTesting(EncodeKernel::kScalar);
      Expect(EncodeRecord(
                 packed_target.data(), reference1_ptr,
                 reference2_ptr, kSampleCt, mode, 3, 7, params,
                 &scalar_record, &error),
             "forced scalar encode failed: " + error);
      Expect(LastEncodeKernelForTesting() == EncodeKernel::kScalar,
             "forced scalar encoder kernel was not selected");
      for (const EncodeKernel kernel :
           {EncodeKernel::kAvx2, EncodeKernel::kAvx512}) {
        if (!EncodeKernelSupported(kernel)) {
          continue;
        }
        const char* const kernel_name =
            (kernel == EncodeKernel::kAvx2) ? "AVX2" : "AVX-512";
        SetEncodeKernelForTesting(kernel);
        Expect(EncodeRecord(
                   packed_target.data(), reference1_ptr,
                   reference2_ptr, kSampleCt, mode, 3, 7, params,
                   &vector_record, &error),
               std::string("forced ") + kernel_name +
                   " encode failed: " + error);
        Expect(LastEncodeKernelForTesting() == kernel,
               std::string("forced ") + kernel_name +
                   " encoder kernel was not selected");
        Expect(vector_record == scalar_record,
               std::string(kernel_name) +
                   " encoder changed serialized record bytes");
      }

      SetEncodeKernelForTesting(EncodeKernel::kAuto);
      Expect(EncodeRecord(
                 packed_target.data(), reference1_ptr,
                 reference2_ptr, kSampleCt, mode, 3, 7, params,
                 &vector_record, &error),
             "automatic SIMD encode failed: " + error);
      const EncodeKernel expected_auto_kernel =
          has_avx512 ? EncodeKernel::kAvx512 : EncodeKernel::kAvx2;
      Expect(LastEncodeKernelForTesting() == expected_auto_kernel,
             "automatic encoder did not select the best SIMD kernel");
      Expect(vector_record == scalar_record,
             "automatic SIMD encoder changed serialized record bytes");
    }
  }
  SetEncodeKernelForTesting(EncodeKernel::kAuto);
}

void TestAvx2EncodeTailsAndFallbacks() {
  if (!EncodeKernelSupported(EncodeKernel::kAvx2)) {
    return;
  }
  std::string error;
  for (const uint32_t state_ct : {16U, 32U}) {
    for (const uint32_t tail_ct : {8U, 9U, 16U, 17U}) {
      if (tail_ct > state_ct) {
        continue;
      }
      const uint32_t sample_ct = 2 * state_ct + tail_ct;
      std::vector<uint8_t> target(sample_ct);
      for (uint32_t sample_idx = 0; sample_idx != sample_ct;
           ++sample_idx) {
        target[sample_idx] =
            static_cast<uint8_t>((5 * sample_idx + sample_idx / 7) & 3U);
      }
      const std::vector<uint64_t> packed_target = Pack(target);
      const CodecParams params(state_ct, 12);
      std::vector<uint8_t> scalar_record;
      std::vector<uint8_t> avx2_record;
      SetEncodeKernelForTesting(EncodeKernel::kScalar);
      Expect(EncodeRecord(
                 packed_target.data(), nullptr, nullptr, sample_ct,
                 RecordMode::kMarginal, 0, 0, params,
                 &scalar_record, &error),
             "tail scalar encode failed: " + error);
      SetEncodeKernelForTesting(EncodeKernel::kAvx2);
      Expect(EncodeRecord(
                 packed_target.data(), nullptr, nullptr, sample_ct,
                 RecordMode::kMarginal, 0, 0, params,
                 &avx2_record, &error),
             "tail AVX2 encode failed: " + error);
      Expect(LastEncodeKernelForTesting() == EncodeKernel::kAvx2,
             "tail test did not reach the AVX2 encoder");
      Expect(avx2_record == scalar_record,
             "AVX2 encoder changed bytes for a partial tail");
    }
  }

  constexpr uint32_t kFallbackSampleCt = 97;
  std::vector<uint8_t> fallback_target(kFallbackSampleCt);
  for (uint32_t sample_idx = 0; sample_idx != kFallbackSampleCt;
       ++sample_idx) {
    fallback_target[sample_idx] =
        static_cast<uint8_t>((sample_idx + sample_idx / 3) & 3U);
  }
  const std::vector<uint64_t> packed_fallback_target =
      Pack(fallback_target);
  for (const CodecParams params :
       {CodecParams(16, 11), CodecParams(24, 12)}) {
    std::vector<uint8_t> record;
    SetEncodeKernelForTesting(EncodeKernel::kAvx2);
    Expect(EncodeRecord(
               packed_fallback_target.data(), nullptr, nullptr,
               kFallbackSampleCt, RecordMode::kMarginal, 0, 0, params,
               &record, &error),
           "ineligible AVX2 fallback encode failed: " + error);
    Expect(LastEncodeKernelForTesting() == EncodeKernel::kScalar,
           "ineligible AVX2 parameters did not fall back to scalar");
  }

  std::vector<uint64_t> deterministic(
      PackedWordCt(kFallbackSampleCt), 0);
  std::vector<uint8_t> deterministic_record;
  SetEncodeKernelForTesting(EncodeKernel::kAvx2);
  Expect(EncodeRecord(
             deterministic.data(), nullptr, nullptr, kFallbackSampleCt,
             RecordMode::kMarginal, 0, 0, CodecParams(),
             &deterministic_record, &error),
         "deterministic fallback encode failed: " + error);
  Expect(LastEncodeKernelForTesting() == EncodeKernel::kScalar,
         "deterministic encode retained a stale SIMD kernel");

  constexpr uint32_t kMismatchSampleCt = 96;
  uint32_t marginal_counts[4] = {48, 48, 0, 0};
  std::vector<uint8_t> mismatch_target(kMismatchSampleCt);
  for (uint32_t sample_idx = 0; sample_idx != kMismatchSampleCt;
       ++sample_idx) {
    mismatch_target[sample_idx] =
        static_cast<uint8_t>(sample_idx & 1U);
  }
  mismatch_target[9] = 2;
  std::vector<uint64_t> packed_mismatch_target =
      Pack(mismatch_target);
  for (const uint32_t state_ct : {16U, 32U}) {
    std::vector<uint8_t> record;
    SetEncodeKernelForTesting(EncodeKernel::kAvx2);
    Expect(!EncodeRecordFromCounts(
               packed_mismatch_target.data(), nullptr, nullptr,
               kMismatchSampleCt, RecordMode::kMarginal, 0, 0,
               marginal_counts, CodecParams(state_ct, 12),
               &record, &error),
           "AVX2 encoder accepted a zero-frequency symbol");
    Expect(LastEncodeKernelForTesting() == EncodeKernel::kAvx2,
           "zero-frequency mismatch did not reach the AVX2 encoder");
  }

  uint32_t conditional_counts[16] = {};
  conditional_counts[0] = 48;
  conditional_counts[4] = 24;
  conditional_counts[5] = 24;
  std::vector<uint8_t> reference(kMismatchSampleCt);
  mismatch_target.assign(kMismatchSampleCt, 0);
  for (uint32_t sample_idx = 48; sample_idx != kMismatchSampleCt;
       ++sample_idx) {
    reference[sample_idx] = 1;
    mismatch_target[sample_idx] =
        static_cast<uint8_t>(sample_idx & 1U);
  }
  mismatch_target[3] = 1;
  std::vector<uint64_t> packed_reference = Pack(reference);
  packed_mismatch_target = Pack(mismatch_target);
  std::vector<uint8_t> mismatch_record;
  SetEncodeKernelForTesting(EncodeKernel::kAvx2);
  Expect(!EncodeRecordFromCounts(
             packed_mismatch_target.data(), packed_reference.data(),
             nullptr, kMismatchSampleCt, RecordMode::kOneReference,
             1, 0, conditional_counts, CodecParams(16, 12),
             &mismatch_record, &error),
         "AVX2 encoder accepted a wrong deterministic symbol");
  Expect(LastEncodeKernelForTesting() == EncodeKernel::kAvx2,
         "deterministic mismatch did not reach the AVX2 encoder");

  mismatch_target[3] = 0;
  reference[3] = 2;
  packed_reference = Pack(reference);
  packed_mismatch_target = Pack(mismatch_target);
  SetEncodeKernelForTesting(EncodeKernel::kAvx2);
  Expect(!EncodeRecordFromCounts(
             packed_mismatch_target.data(), packed_reference.data(),
             nullptr, kMismatchSampleCt, RecordMode::kOneReference,
             1, 0, conditional_counts, CodecParams(32, 12),
             &mismatch_record, &error),
         "AVX2 encoder accepted an absent context");
  Expect(LastEncodeKernelForTesting() == EncodeKernel::kAvx2,
         "absent-context mismatch did not reach the AVX2 encoder");
  SetEncodeKernelForTesting(EncodeKernel::kAuto);
}

void TestForcedDefaultDecodeKernels() {
  // A 31-sample tail exercises state reuse in the state-16 NEON path.
  constexpr uint32_t kSampleCt = 4127;
  std::mt19937_64 rng(0x73696d646465636fULL);
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
  const std::vector<uint8_t>* const targets[] = {
      &marginal, &one_reference, &two_reference};
  const RecordMode modes[] = {
      RecordMode::kMarginal, RecordMode::kOneReference,
      RecordMode::kTwoReference};
  const DecodeKernel kernels[] = {
      DecodeKernel::kScalar, DecodeKernel::kAvx2,
      DecodeKernel::kAvx512, DecodeKernel::kNeon};
  const std::vector<uint64_t> packed_reference1 = Pack(reference1);
  const std::vector<uint64_t> packed_reference2 = Pack(reference2);
  const uint64_t* anchors[8] = {};
  anchors[3] = packed_reference1.data();
  anchors[7] = packed_reference2.data();
  const CodecParams params;
  for (uint32_t mode_idx = 0; mode_idx != 3; ++mode_idx) {
    const RecordMode mode = modes[mode_idx];
    const std::vector<uint64_t> packed_target =
        Pack(*targets[mode_idx]);
    const uint64_t* reference1_ptr =
        (mode == RecordMode::kMarginal)
            ? nullptr
            : packed_reference1.data();
    const uint64_t* reference2_ptr =
        (mode == RecordMode::kTwoReference)
            ? packed_reference2.data()
            : nullptr;
    std::vector<uint8_t> record;
    std::string error;
    Expect(EncodeRecord(
               packed_target.data(), reference1_ptr, reference2_ptr,
               kSampleCt, mode, 3, 7, params, &record, &error),
           "forced-kernel encode failed: " + error);
    for (const DecodeKernel kernel : kernels) {
      if (!DecodeKernelSupported(kernel)) {
        continue;
      }
      SetDecodeKernelForTesting(kernel);
      std::vector<uint64_t> decoded(PackedWordCt(kSampleCt));
      Expect(DecodeRecordToBuffer(
                 record.data(), record.size(), anchors, 8, kSampleCt,
                 params, decoded.data(), decoded.size(), nullptr,
                 &error),
             "forced-kernel decode failed: " + error);
      Expect(LastDecodeKernelForTesting() == kernel,
             "forced decoder kernel was not selected");
      Expect(decoded == packed_target,
             "forced decoder kernel changed the decoded genotypes");
      Expect(DecodeRecordToBufferFromValidatedBlock(
                 record.data(), record.size(), anchors, 8, kSampleCt,
                 params, decoded.data(), decoded.size(), nullptr,
                 &error),
             "forced validated-block decode failed: " + error);
      Expect(LastDecodeKernelForTesting() == kernel,
             "forced validated-block kernel was not selected");
      Expect(decoded == packed_target,
             "forced validated-block kernel changed the genotypes");
    }
  }

  if (DecodeKernelSupported(DecodeKernel::kNeon)) {
    const CodecParams state16_params(16, 12);
    for (uint32_t mode_idx = 0; mode_idx != 3; ++mode_idx) {
      const RecordMode mode = modes[mode_idx];
      const std::vector<uint64_t> packed_target =
          Pack(*targets[mode_idx]);
      const uint64_t* reference1_ptr =
          (mode == RecordMode::kMarginal)
              ? nullptr
              : packed_reference1.data();
      const uint64_t* reference2_ptr =
          (mode == RecordMode::kTwoReference)
              ? packed_reference2.data()
              : nullptr;
      std::vector<uint8_t> record;
      std::string error;
      Expect(EncodeRecord(
                 packed_target.data(), reference1_ptr, reference2_ptr,
                 kSampleCt, mode, 3, 7, state16_params, &record,
                 &error),
             "state-16 NEON test encode failed: " + error);
      std::vector<uint64_t> scalar_decoded(PackedWordCt(kSampleCt));
      SetDecodeKernelForTesting(DecodeKernel::kScalar);
      Expect(DecodeRecordToBuffer(
                 record.data(), record.size(), anchors, 8, kSampleCt,
                 state16_params, scalar_decoded.data(),
                 scalar_decoded.size(), nullptr, &error),
             "forced state-16 scalar decode failed: " + error);
      Expect(scalar_decoded == packed_target,
             "forced state-16 scalar decoder changed the genotypes");

      std::vector<uint64_t> neon_decoded(PackedWordCt(kSampleCt));
      SetDecodeKernelForTesting(DecodeKernel::kNeon);
      Expect(DecodeRecordToBuffer(
                 record.data(), record.size(), anchors, 8, kSampleCt,
                 state16_params, neon_decoded.data(),
                 neon_decoded.size(), nullptr, &error),
             "forced state-16 NEON decode failed: " + error);
      Expect(LastDecodeKernelForTesting() == DecodeKernel::kNeon,
             "forced state-16 NEON decoder was not selected");
      Expect(neon_decoded == scalar_decoded,
             "state-16 NEON decoder disagrees with scalar");
      Expect(DecodeRecordToBufferFromValidatedBlock(
                 record.data(), record.size(), anchors, 8, kSampleCt,
                 state16_params, neon_decoded.data(),
                 neon_decoded.size(), nullptr, &error),
             "forced validated state-16 NEON decode failed: " + error);
      Expect(LastDecodeKernelForTesting() == DecodeKernel::kNeon,
             "validated state-16 NEON decoder was not selected");
      Expect(neon_decoded == scalar_decoded,
             "validated state-16 NEON decoder disagrees with scalar");

      SetDecodeKernelForTesting(DecodeKernel::kAuto);
      Expect(DecodeRecordToBuffer(
                 record.data(), record.size(), anchors, 8, kSampleCt,
                 state16_params, neon_decoded.data(),
                 neon_decoded.size(), nullptr, &error),
             "automatic state-16 NEON decode failed: " + error);
      Expect(LastDecodeKernelForTesting() == DecodeKernel::kNeon,
             "automatic state-16 decode did not select NEON");
      Expect(neon_decoded == scalar_decoded,
             "automatic state-16 NEON decoder disagrees with scalar");
    }
  }

  // The non-validated entry point must retain the scalar decoder's absent
  // model-context rejection in every SIMD implementation.
  const std::vector<uint8_t> encoded_reference(kSampleCt, 0);
  const std::vector<uint8_t> invalid_reference(kSampleCt, 1);
  const std::vector<uint64_t> packed_encoded_reference =
      Pack(encoded_reference);
  const std::vector<uint64_t> packed_invalid_reference =
      Pack(invalid_reference);
  const std::vector<uint64_t> packed_absent_target = Pack(marginal);
  std::vector<uint8_t> absent_context_record;
  std::string error;
  Expect(EncodeRecord(
             packed_absent_target.data(),
             packed_encoded_reference.data(), nullptr, kSampleCt,
             RecordMode::kOneReference, 3, 0, params,
             &absent_context_record, &error),
         "absent-context test encode failed: " + error);
  const uint64_t* invalid_anchors[4] = {};
  invalid_anchors[3] = packed_invalid_reference.data();
  std::vector<uint64_t> decoded(PackedWordCt(kSampleCt));
  for (const DecodeKernel kernel : kernels) {
    if (!DecodeKernelSupported(kernel)) {
      continue;
    }
    SetDecodeKernelForTesting(kernel);
    Expect(!DecodeRecordToBuffer(
               absent_context_record.data(),
               absent_context_record.size(), invalid_anchors, 4,
               kSampleCt, params, decoded.data(), decoded.size(),
               nullptr, &error),
           "forced decoder accepted an absent model context");
    Expect(LastDecodeKernelForTesting() == kernel,
           "absent-context test did not reach the forced kernel");
  }
  SetDecodeKernelForTesting(DecodeKernel::kAuto);
}

void TestAvx2HighStateRejection() {
  if (!DecodeKernelSupported(DecodeKernel::kAvx2)) {
    return;
  }
  constexpr uint32_t kSampleCt = 4096;
  constexpr size_t kInitialStateOffset = 4;
  constexpr uint32_t kStateCt = 32;
  std::vector<uint64_t> packed(PackedWordCt(kSampleCt), 0);
  const CodecParams params(kStateCt, 12);
  uint32_t counts[4] = {kSampleCt - 1, 1, 0, 0};
  std::vector<uint8_t> record;
  std::string error;
  Expect(EncodeRecordFromCounts(
             packed.data(), nullptr, nullptr, kSampleCt,
             RecordMode::kMarginal, 0, 0, counts, params, &record,
             &error),
         "high-state regression encode failed: " + error);
  Expect(record.size() >= kInitialStateOffset + 4 * kStateCt,
         "high-state regression record is unexpectedly short");
  record.resize(kInitialStateOffset + 4 * kStateCt);
  for (uint32_t lane = 0; lane != kStateCt; ++lane) {
    const size_t offset = kInitialStateOffset + 4 * lane;
    record[offset] = 0;
    record[offset + 1] = 0xf0;
    record[offset + 2] = 0xff;
    record[offset + 3] = 0xff;
  }

  const uint64_t* no_anchors[1] = {};
  std::vector<uint64_t> decoded(PackedWordCt(kSampleCt));
  SetDecodeKernelForTesting(DecodeKernel::kScalar);
  Expect(!DecodeRecordToBuffer(
             record.data(), record.size(), no_anchors, 0, kSampleCt,
             params, decoded.data(), decoded.size(), nullptr, &error),
         "scalar decoder accepted malformed high initial states");
  const std::string scalar_error = error;

  SetDecodeKernelForTesting(DecodeKernel::kAvx2);
  Expect(!DecodeRecordToBuffer(
             record.data(), record.size(), no_anchors, 0, kSampleCt,
             params, decoded.data(), decoded.size(), nullptr, &error),
         "AVX2 decoder accepted malformed high initial states");
  Expect(LastDecodeKernelForTesting() == DecodeKernel::kAvx2,
         "high-state regression did not reach the AVX2 decoder");
  Expect(error == scalar_error,
         "AVX2 high-state rejection differs from scalar");
  SetDecodeKernelForTesting(DecodeKernel::kAuto);
}

void TestAvx512RefillBoundaries() {
  if (!DecodeKernelSupported(DecodeKernel::kAvx512)) {
    return;
  }
  constexpr uint32_t kSampleCt = 1U << 12;
  constexpr uint32_t kStateCt = 32;
  constexpr uint32_t kRansLowerBound = 1U << 23;
  constexpr size_t kInitialStateOffset = 4;
  constexpr size_t kPayloadOffset =
      kInitialStateOffset + 4 * kStateCt;
  constexpr size_t kUncheckedPayloadByteCt = 64;
  std::vector<uint64_t> packed_target(
      PackedWordCt(kSampleCt), 0x5555555555555555ULL);
  SetPackedGenotype(packed_target.data(), 0, 0);
  uint32_t counts[4] = {1, kSampleCt - 1, 0, 0};
  const CodecParams params(kStateCt, 12);
  std::vector<uint8_t> record;
  std::string error;
  Expect(EncodeRecordFromCounts(
             packed_target.data(), nullptr, nullptr, kSampleCt,
             RecordMode::kMarginal, 0, 0, counts, params, &record,
             &error),
         "AVX-512 refill-boundary encode failed: " + error);
  Expect(record.size() >= kPayloadOffset,
         "AVX-512 refill-boundary record is unexpectedly short");

  // A marginal [1, 4095] model maps slot zero to a frequency-one
  // symbol.  Starting every lane at 2^23 therefore transforms every
  // state to 2^11, forcing two refill bytes in all 32 lanes.  The
  // 64-byte record exercises unchecked loads at payload offsets
  // 0, 16, 32, and 48; the 63-byte record must take the checked path.
  for (uint32_t lane = 0; lane != kStateCt; ++lane) {
    const size_t offset = kInitialStateOffset + 4 * lane;
    record[offset] = static_cast<uint8_t>(kRansLowerBound);
    record[offset + 1] =
        static_cast<uint8_t>(kRansLowerBound >> 8);
    record[offset + 2] =
        static_cast<uint8_t>(kRansLowerBound >> 16);
    record[offset + 3] =
        static_cast<uint8_t>(kRansLowerBound >> 24);
  }
  record.resize(kPayloadOffset + kUncheckedPayloadByteCt);
  for (size_t offset = kPayloadOffset; offset != record.size();
       ++offset) {
    record[offset] = 0;
  }

  const uint64_t* no_anchors[1] = {};
  std::vector<uint64_t> decoded(PackedWordCt(kSampleCt));
  for (const size_t payload_byte_ct :
       {kUncheckedPayloadByteCt, kUncheckedPayloadByteCt - 1}) {
    const size_t record_size = kPayloadOffset + payload_byte_ct;
    SetDecodeKernelForTesting(DecodeKernel::kScalar);
    const bool scalar_ok = DecodeRecordToBuffer(
        record.data(), record_size, no_anchors, 0, kSampleCt,
        params, decoded.data(), decoded.size(), nullptr, &error);
    const std::string scalar_error = error;
    Expect(!scalar_ok,
           "scalar decoder accepted a malformed refill-boundary payload");

    SetDecodeKernelForTesting(DecodeKernel::kAvx512);
    const bool avx512_ok = DecodeRecordToBuffer(
        record.data(), record_size, no_anchors, 0, kSampleCt,
        params, decoded.data(), decoded.size(), nullptr, &error);
    Expect(avx512_ok == scalar_ok,
           "AVX-512 refill-boundary acceptance differs from scalar");
    Expect(LastDecodeKernelForTesting() == DecodeKernel::kAvx512,
           "refill-boundary test did not reach the AVX-512 decoder");
    Expect(error == scalar_error,
           "AVX-512 refill-boundary error differs from scalar");
  }
  SetDecodeKernelForTesting(DecodeKernel::kAuto);
}

void TestRuntimeDivisionFrequencies() {
  constexpr uint32_t kSampleCt = 1U << 12;
  std::vector<uint64_t> packed(
      PackedWordCt(kSampleCt), 0x5555555555555555ULL);
  const CodecParams params(16, 12);
  const uint64_t* no_anchors[1] = {};
  std::vector<uint8_t> record;
  std::vector<uint8_t> scalar_record;
  std::vector<uint64_t> decoded;
  std::string error;
  uint32_t counts[4] = {};
  for (uint32_t frequency = 1; frequency != kSampleCt; ++frequency) {
    SetPackedGenotype(packed.data(), frequency - 1, 0);
    counts[0] = frequency;
    counts[1] = kSampleCt - frequency;
    SetEncodeKernelForTesting(EncodeKernel::kScalar);
    Expect(EncodeRecordFromCounts(
               packed.data(), nullptr, nullptr, kSampleCt,
               RecordMode::kMarginal, 0, 0, counts, params, &record,
               &error),
           "runtime-division encode failed at frequency " +
               std::to_string(frequency) + ": " + error);
    scalar_record = record;
    for (const EncodeKernel kernel :
         {EncodeKernel::kAvx2, EncodeKernel::kAvx512}) {
      if (!EncodeKernelSupported(kernel)) {
        continue;
      }
      const char* const kernel_name =
          (kernel == EncodeKernel::kAvx2) ? "AVX2" : "AVX-512";
      SetEncodeKernelForTesting(kernel);
      Expect(EncodeRecordFromCounts(
                 packed.data(), nullptr, nullptr, kSampleCt,
                 RecordMode::kMarginal, 0, 0, counts, params,
                 &record, &error),
             std::string(kernel_name) +
                 " runtime-division encode failed at frequency " +
                 std::to_string(frequency) + ": " + error);
      Expect(LastEncodeKernelForTesting() == kernel,
             std::string("forced ") + kernel_name +
                 " runtime-division kernel was not selected");
      Expect(record == scalar_record,
             std::string(kernel_name) +
                 " runtime division changed bytes at frequency " +
                 std::to_string(frequency));
    }
    Expect(DecodeRecord(
               record.data(), record.size(), no_anchors, 0, kSampleCt,
               params, &decoded, nullptr, &error),
           "runtime-division decode failed at frequency " +
               std::to_string(frequency) + ": " + error);
    Expect(decoded == packed,
           "runtime-division round-trip mismatch at frequency " +
               std::to_string(frequency));
  }
  SetEncodeKernelForTesting(EncodeKernel::kAuto);
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
  params.scale_bits = 12;
  const uint32_t wrong_counts[4] = {1, 1, 1, 0};
  Expect(!EncodeRecordFromCounts(
             genotypes.data(), nullptr, nullptr, 4,
             RecordMode::kMarginal, 0, 0, wrong_counts, params,
             &record, &error),
         "model counts with the wrong sample total were accepted");
  const uint32_t wrong_support[4] = {4, 0, 0, 0};
  Expect(!EncodeRecordFromCounts(
             genotypes.data(), nullptr, nullptr, 4,
             RecordMode::kMarginal, 0, 0, wrong_support, params,
             &record, &error),
         "model counts with the wrong symbol support were accepted");
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
  TestForcedDefaultEncodeKernels();
  TestAvx2EncodeTailsAndFallbacks();
  TestForcedDefaultDecodeKernels();
  TestAvx2HighStateRejection();
  TestAvx512RefillBoundaries();
  TestRuntimeDivisionFrequencies();
  TestInvalidArguments();
  TestMultiallelicPatches();
  puts("pgen_rans_codec_test: PASS");
  return 0;
}
