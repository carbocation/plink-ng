// SPDX-License-Identifier: GPL-3.0-or-later

#include "pgen_rans_hybrid.h"

#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

namespace {

using pgen_rans::DecodeAlternateRecordToBuffer;
using pgen_rans::EncodeRawRecord;
using pgen_rans::EncodeSparsePredictorRecord;
using pgen_rans::EncodeSparsePredictorRecordFromCounts;
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
  std::vector<uint64_t> result(PackedWordCt(genotypes.size()), 0);
  for (uint32_t idx = 0; idx != genotypes.size(); ++idx) {
    SetPackedGenotype(result.data(), idx, genotypes[idx]);
  }
  return result;
}

void ExpectEqual(const std::vector<uint64_t>& packed,
                 const std::vector<uint8_t>& expected,
                 const std::string& label) {
  for (uint32_t idx = 0; idx != expected.size(); ++idx) {
    if (GetPackedGenotype(packed.data(), idx) != expected[idx]) {
      Fail(label + " mismatch at sample " + std::to_string(idx));
    }
  }
}

void RoundTrip(const std::vector<uint8_t>& target,
               const std::vector<uint8_t>& reference1,
               const std::vector<uint8_t>& reference2,
               RecordMode mode) {
  const std::vector<uint64_t> packed_target = Pack(target);
  const std::vector<uint64_t> packed_reference1 = Pack(reference1);
  const std::vector<uint64_t> packed_reference2 = Pack(reference2);
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
  Expect(EncodeSparsePredictorRecord(
             packed_target.data(), reference1_ptr, reference2_ptr,
             target.size(), mode, 2, 5, &record, &error),
         "sparse encode failed: " + error);
  const uint32_t row_ct =
      (mode == RecordMode::kMarginal)
          ? 1
          : ((mode == RecordMode::kOneReference) ? 4 : 16);
  std::vector<uint32_t> counts(4 * row_ct, 0);
  for (uint32_t sample_idx = 0; sample_idx != target.size();
       ++sample_idx) {
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
  Expect(EncodeSparsePredictorRecordFromCounts(
             packed_target.data(), reference1_ptr, reference2_ptr,
             target.size(), mode, 2, 5, counts.data(),
             &count_reused_record, &error),
         "count-reusing sparse encode failed: " + error);
  Expect(record == count_reused_record,
         "count-reusing sparse encode changed the record");
  const uint64_t* anchors[6] = {};
  anchors[2] = packed_reference1.data();
  anchors[5] = packed_reference2.data();
  std::vector<uint64_t> decoded(PackedWordCt(target.size()) + 1,
                                0xfacefeedULL);
  RecordMetadata metadata;
  Expect(DecodeAlternateRecordToBuffer(
             record.data(), record.size(), anchors, 6, target.size(),
             decoded.data(), PackedWordCt(target.size()), &metadata,
             &error),
         "sparse decode failed: " + error);
  Expect(metadata.mode == mode, "sparse mode mismatch");
  ExpectEqual(decoded, target, "sparse");
  Expect(decoded.back() == 0xfacefeedULL,
         "sparse decoder wrote past output");

  if (record.size() > 2) {
    std::vector<uint8_t> truncated = record;
    truncated.pop_back();
    Expect(!DecodeAlternateRecordToBuffer(
               truncated.data(), truncated.size(), anchors, 6,
               target.size(), decoded.data(), PackedWordCt(target.size()),
               nullptr, &error),
           "truncated sparse record was accepted");
  }
}

void TestSparseRoundTrips() {
  std::mt19937_64 rng(0xd5a24e69b37c18f1ULL);
  const uint32_t sample_counts[] = {1, 2, 31, 32, 33, 257, 1003, 4099};
  for (const uint32_t sample_ct : sample_counts) {
    std::vector<uint8_t> reference1(sample_ct);
    std::vector<uint8_t> reference2(sample_ct);
    std::vector<uint8_t> marginal(sample_ct);
    std::vector<uint8_t> one_reference(sample_ct);
    std::vector<uint8_t> two_reference(sample_ct);
    for (uint32_t idx = 0; idx != sample_ct; ++idx) {
      reference1[idx] = rng() % 4;
      reference2[idx] = rng() % 4;
      marginal[idx] = ((rng() % 127) == 0) ? (rng() % 4) : 0;
      one_reference[idx] =
          ((rng() % 97) == 0) ? (rng() % 4) : reference1[idx];
      two_reference[idx] =
          ((rng() % 89) == 0)
              ? (rng() % 4)
              : ((reference1[idx] + 2 * reference2[idx]) & 3);
    }
    RoundTrip(marginal, reference1, reference2,
              RecordMode::kMarginal);
    RoundTrip(one_reference, reference1, reference2,
              RecordMode::kOneReference);
    RoundTrip(two_reference, reference1, reference2,
              RecordMode::kTwoReference);
  }
}

void TestRawRoundTrips() {
  std::mt19937_64 rng(0x32a5d36c702eef33ULL);
  const uint32_t sample_counts[] = {1, 3, 4, 5, 31, 32, 33, 1003};
  for (const uint32_t sample_ct : sample_counts) {
    std::vector<uint8_t> target(sample_ct);
    for (uint8_t& value : target) {
      value = rng() % 4;
    }
    const std::vector<uint64_t> packed = Pack(target);
    std::vector<uint8_t> record;
    std::string error;
    Expect(EncodeRawRecord(
               packed.data(), sample_ct, &record, &error),
           "raw encode failed: " + error);
    std::vector<uint64_t> decoded(PackedWordCt(sample_ct) + 1,
                                  0xc001d00dULL);
    Expect(DecodeAlternateRecordToBuffer(
               record.data(), record.size(), nullptr, 0, sample_ct,
               decoded.data(), PackedWordCt(sample_ct), nullptr, &error),
           "raw decode failed: " + error);
    ExpectEqual(decoded, target, "raw");
    Expect(decoded.back() == 0xc001d00dULL,
           "raw decoder wrote past output");
    record.push_back(0);
    Expect(!DecodeAlternateRecordToBuffer(
               record.data(), record.size(), nullptr, 0, sample_ct,
               decoded.data(), PackedWordCt(sample_ct), nullptr, &error),
           "raw record with trailing byte was accepted");
  }
}

}  // namespace

int main() {
  TestRawRoundTrips();
  TestSparseRoundTrips();
  puts("pgen_rans_hybrid_test: PASS");
  return 0;
}
