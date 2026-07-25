// SPDX-License-Identifier: GPL-3.0-or-later

#include "pgen_rans_codec.h"
#include "pgen_rans_container.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <unistd.h>
#include <vector>

namespace {

using pgen_rans::CodecParams;
using pgen_rans::ContainerParams;
using pgen_rans::ContainerReader;
using pgen_rans::ContainerWriter;
using pgen_rans::DecodeRecord;
using pgen_rans::EncodeRecord;
using pgen_rans::EncodedBlock;
using pgen_rans::GetPackedGenotype;
using pgen_rans::PackedWordCt;
using pgen_rans::RecordMetadata;
using pgen_rans::RecordMode;
using pgen_rans::ScheduledAnchorOffsets;
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

std::vector<uint64_t> MakeVariant(uint32_t variant_idx,
                                  uint32_t sample_ct) {
  std::vector<uint64_t> result(PackedWordCt(sample_ct), 0);
  for (uint32_t sample_idx = 0; sample_idx != sample_ct; ++sample_idx) {
    uint8_t genotype =
        static_cast<uint8_t>((sample_idx * 3 + variant_idx) % 3);
    if ((sample_idx + 5 * variant_idx) % 97 == 0) {
      genotype = 3;
    }
    SetPackedGenotype(result.data(), sample_idx, genotype);
  }
  return result;
}

std::string TemporaryPath() {
  char path[] = "/tmp/pgen_rans_container_test.XXXXXX";
  const int fd = mkstemp(path);
  if (fd < 0) {
    Fail("mkstemp failed");
  }
  close(fd);
  unlink(path);
  return path;
}

void ExpectEqual(const std::vector<uint64_t>& observed,
                 const std::vector<uint64_t>& expected,
                 uint32_t sample_ct) {
  for (uint32_t sample_idx = 0; sample_idx != sample_ct; ++sample_idx) {
    if (GetPackedGenotype(observed.data(), sample_idx) !=
        GetPackedGenotype(expected.data(), sample_idx)) {
      Fail("decoded container genotype mismatch");
    }
  }
}

}  // namespace

int main() {
  constexpr uint32_t kSampleCt = 1003;
  constexpr uint32_t kVariantCt = 19;
  constexpr uint32_t kBlockVariantCt = 10;
  constexpr uint32_t kAnchorCt = 3;
  const CodecParams codec_params;
  const ContainerParams container_params = {
      kSampleCt, kVariantCt, kBlockVariantCt, kAnchorCt,
      codec_params.state_ct, codec_params.scale_bits, 4, 2};
  std::vector<std::vector<uint64_t>> source;
  for (uint32_t variant_idx = 0; variant_idx != kVariantCt; ++variant_idx) {
    source.push_back(MakeVariant(variant_idx, kSampleCt));
  }

  const std::string path = TemporaryPath();
  std::string error;
  ContainerWriter writer;
  Expect(writer.Open(path, container_params, &error),
         "writer open failed: " + error);
  uint32_t first_variant = 0;
  for (uint32_t block_idx = 0; block_idx != 2; ++block_idx) {
    const uint32_t block_variant_ct =
        std::min(kBlockVariantCt, kVariantCt - first_variant);
    const std::vector<uint32_t> anchor_offsets =
        ScheduledAnchorOffsets(block_variant_ct, kAnchorCt);
    EncodedBlock block;
    block.first_variant = first_variant;
    block.records.resize(block_variant_ct);
    for (uint32_t variant_offset = 0; variant_offset != block_variant_ct;
         ++variant_offset) {
      RecordMode mode = RecordMode::kMarginal;
      uint8_t reference1 = 0;
      uint8_t reference2 = 0;
      const uint64_t* reference1_data = nullptr;
      const uint64_t* reference2_data = nullptr;
      const auto anchor_iter =
          std::find(anchor_offsets.begin(), anchor_offsets.end(),
                    variant_offset);
      if (anchor_iter == anchor_offsets.end()) {
        reference1 = static_cast<uint8_t>(
            variant_offset % anchor_offsets.size());
        reference1_data =
            source[first_variant + anchor_offsets[reference1]].data();
        if (variant_offset % 2) {
          mode = RecordMode::kOneReference;
        } else {
          mode = RecordMode::kTwoReference;
          reference2 = (reference1 + 1) % anchor_offsets.size();
          reference2_data =
              source[first_variant + anchor_offsets[reference2]].data();
        }
      }
      Expect(EncodeRecord(
                 source[first_variant + variant_offset].data(),
                 reference1_data, reference2_data, kSampleCt, mode,
                 reference1, reference2, codec_params,
                 &(block.records[variant_offset]), &error),
             "record encode failed: " + error);
    }
    Expect(writer.WriteBlock(block, &error),
           "block write failed: " + error);
    first_variant += block_variant_ct;
  }
  Expect(writer.Close(&error), "writer close failed: " + error);

  ContainerReader reader;
  Expect(reader.Open(path, &error), "reader open failed: " + error);
  Expect(reader.params().sample_ct == kSampleCt,
         "container sample count mismatch");
  Expect(reader.params().variant_ct == kVariantCt,
         "container variant count mismatch");
  for (uint32_t variant_idx = 0; variant_idx != kVariantCt; ++variant_idx) {
    uint32_t block_idx;
    Expect(reader.FindBlock(variant_idx, &block_idx, &error),
           "block lookup failed: " + error);
    const auto& entry = reader.block_index()[block_idx];
    Expect((variant_idx >= entry.first_variant) &&
               (variant_idx < entry.first_variant + entry.variant_ct),
           "block lookup returned the wrong range");
  }
  for (uint32_t block_idx = 0; block_idx != 2; ++block_idx) {
    EncodedBlock block;
    Expect(reader.ReadBlock(block_idx, &block, &error),
           "block read failed: " + error);
    const std::vector<uint32_t> anchor_offsets =
        ScheduledAnchorOffsets(
            static_cast<uint32_t>(block.records.size()), kAnchorCt);
    std::vector<std::vector<uint64_t>> anchors(anchor_offsets.size());
    std::vector<const uint64_t*> anchor_ptrs(anchor_offsets.size());
    for (uint32_t anchor_idx = 0; anchor_idx != anchor_offsets.size();
         ++anchor_idx) {
      RecordMetadata metadata;
      Expect(DecodeRecord(
                 block.records[anchor_offsets[anchor_idx]].data(),
                 block.records[anchor_offsets[anchor_idx]].size(), nullptr, 0,
                 kSampleCt, codec_params, &(anchors[anchor_idx]), &metadata,
                 &error),
             "anchor decode failed: " + error);
      Expect(metadata.mode == RecordMode::kMarginal,
             "anchor record is conditional");
      anchor_ptrs[anchor_idx] = anchors[anchor_idx].data();
    }
    for (uint32_t variant_offset = 0;
         variant_offset != block.records.size(); ++variant_offset) {
      std::vector<uint64_t> decoded;
      Expect(DecodeRecord(
                 block.records[variant_offset].data(),
                 block.records[variant_offset].size(), anchor_ptrs.data(),
                 static_cast<uint32_t>(anchor_ptrs.size()), kSampleCt,
                 codec_params, &decoded,
                 nullptr, &error),
             "target decode failed: " + error);
      ExpectEqual(decoded, source[block.first_variant + variant_offset],
                  kSampleCt);
    }
  }
  reader.Close();
  Expect(unlink(path.c_str()) == 0, "temporary file cleanup failed");
  puts("pgen_rans_container_test: PASS");
  return 0;
}
