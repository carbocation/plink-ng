// SPDX-License-Identifier: GPL-3.0-or-later

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <sys/stat.h>
#include <string>
#include <unistd.h>
#include <vector>

#include "pgen_rans_benchmark.h"
#include "pgen_rans_codec.h"
#include "pgen_rans_container.h"
#include "pgen_rans_size.h"

namespace {

bool Expect(bool condition, const char* message) {
  if (!condition) {
    fprintf(stderr, "FAIL: %s\n", message);
    return false;
  }
  return true;
}

bool TestContainerOverhead() {
  bool ok = true;
  ok &= Expect(
      pgen_rans::ContainerGlobalOverheadByteCt(2, 0) == 160,
      "two-block global overhead");
  ok &= Expect(
      pgen_rans::ContainerBlockOverheadByteCt(128, 64) == 404,
      "full block overhead with one restart");
  ok &= Expect(
      pgen_rans::ContainerBlockOverheadByteCt(2, 64) == 22,
      "short block overhead without a restart");
  ok &= Expect(
      pgen_rans::ContainerGlobalOverheadByteCt(2, 17) + 404 + 22 ==
          603,
      "mixed-nonref 130-variant container overhead");
  return ok;
}

bool TestContainerWriterSize() {
  char path[] = "/tmp/pgen-rans-size-test.XXXXXX";
  const int fd = mkstemp(path);
  if (fd < 0) {
    return Expect(false, "create temporary container path");
  }
  close(fd);
  pgen_rans::ContainerParams params(
      1, 130, 128, 1, 1, 12, 64, 2, 2);
  pgen_rans::ContainerMetadata metadata;
  metadata.nonref_flags.assign(17, 0);
  metadata.nonref_flags[0] = 1;
  pgen_rans::ContainerWriter writer;
  std::string error;
  if (!writer.Open(path, params, metadata, &error)) {
    fprintf(stderr, "FAIL: open size-test container: %s\n",
            error.c_str());
    remove(path);
    return false;
  }
  pgen_rans::EncodedBlock first_block;
  first_block.records.assign(128, std::vector<uint8_t>(1, 0));
  pgen_rans::EncodedBlock second_block;
  second_block.first_variant = 128;
  second_block.records.assign(2, std::vector<uint8_t>(1, 0));
  if ((!writer.WriteBlock(first_block, &error)) ||
      (!writer.WriteBlock(second_block, &error)) ||
      (!writer.Close(&error))) {
    fprintf(stderr, "FAIL: write size-test container: %s\n",
            error.c_str());
    remove(path);
    return false;
  }
  struct stat stat_buf;
  const bool stat_ok = !stat(path, &stat_buf);
  remove(path);
  return Expect(stat_ok, "stat size-test container") &&
         Expect(
             static_cast<uint64_t>(stat_buf.st_size) == 603 + 130,
             "formula matches serialized container size");
}

bool TestTimingSummary() {
  const pgen_rans::TimingSummary summary =
      pgen_rans::SummarizeTimings({9.0, 1.0, 5.0, 3.0, 7.0});
  return Expect(summary.median == 5.0, "timing median") &&
         Expect(summary.minimum == 1.0, "timing minimum") &&
         Expect(summary.maximum == 9.0, "timing maximum") &&
         Expect(summary.median_absolute_deviation == 2.0,
                "timing median absolute deviation");
}

bool CheckEstimateMatchesEncode(
    const std::vector<uint64_t>& target,
    const std::vector<uint64_t>* reference,
    const uint32_t* counts, pgen_rans::RecordMode mode,
    const char* label) {
  const pgen_rans::CodecParams params(32, 12);
  uint64_t estimate = 0;
  std::string error;
  if (!pgen_rans::EstimateRecordBytes(
          counts, mode, params, &estimate, &error)) {
    fprintf(stderr, "FAIL: %s estimate: %s\n", label, error.c_str());
    return false;
  }
  std::vector<uint8_t> record;
  if (!pgen_rans::EncodeRecord(
          target.data(), reference ? reference->data() : nullptr,
          nullptr, 256, mode, 0, 0, params, &record, &error)) {
    fprintf(stderr, "FAIL: %s encode: %s\n", label, error.c_str());
    return false;
  }
  if (estimate != record.size()) {
    fprintf(stderr, "FAIL: %s estimate %llu != encoded size %zu\n",
            label, static_cast<unsigned long long>(estimate),
            record.size());
    return false;
  }
  return true;
}

bool TestProductionEstimator() {
  std::vector<uint64_t> target(pgen_rans::PackedWordCt(256), 0);
  uint32_t marginal_counts[4] = {};
  for (uint32_t sample_idx = 0; sample_idx != 256; ++sample_idx) {
    const uint8_t genotype = static_cast<uint8_t>(sample_idx % 4);
    pgen_rans::SetPackedGenotype(
        target.data(), sample_idx, genotype);
    ++marginal_counts[genotype];
  }
  if (!CheckEstimateMatchesEncode(
          target, nullptr, marginal_counts,
          pgen_rans::RecordMode::kMarginal,
          "uniform marginal record")) {
    return false;
  }

  std::vector<uint64_t> reference(target);
  uint32_t conditional_counts[16] = {};
  for (uint32_t genotype = 0; genotype != 4; ++genotype) {
    conditional_counts[4 * genotype + genotype] = 64;
  }
  return CheckEstimateMatchesEncode(
      target, &reference, conditional_counts,
      pgen_rans::RecordMode::kOneReference,
      "deterministic conditional record");
}

}  // namespace

int main() {
  if ((!TestContainerOverhead()) || (!TestContainerWriterSize()) ||
      (!TestTimingSummary()) || (!TestProductionEstimator())) {
    return 1;
  }
  puts("pgen rANS size/benchmark tests passed");
  return 0;
}
