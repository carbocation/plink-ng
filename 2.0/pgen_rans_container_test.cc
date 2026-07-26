// SPDX-License-Identifier: GPL-3.0-or-later

#include "pgen_rans_codec.h"
#include "pgen_rans_container.h"
#include "pgen_rans_cpu.h"
#include "pgen_rans_reader.h"

#include <algorithm>
#include <cerrno>
#include <cstring>
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
using pgen_rans::CpuBlockDecoder;
using pgen_rans::EncodedBlockView;
using pgen_rans::PackedReadStats;
using pgen_rans::PackedVariantReader;
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

struct MemoryReader {
  std::vector<uint8_t> bytes;
  std::vector<std::pair<uint64_t, size_t>> reads;
};

bool ReadMemory(void* context, uint64_t offset, uint8_t* destination,
                size_t byte_ct, std::string* error) {
  MemoryReader* reader = static_cast<MemoryReader*>(context);
  if ((offset > reader->bytes.size()) ||
      (byte_ct > reader->bytes.size() - offset)) {
    *error = "memory read is outside the source";
    return false;
  }
  memcpy(destination, reader->bytes.data() + offset, byte_ct);
  reader->reads.emplace_back(offset, byte_ct);
  return true;
}

MemoryReader LoadMemoryReader(const std::string& path) {
  FILE* input = fopen(path.c_str(), "rb");
  if (!input) {
    Fail(std::string("could not open temporary container: ") +
         strerror(errno));
  }
  Expect(fseeko(input, 0, SEEK_END) == 0,
         "could not query temporary container size");
  const off_t byte_ct = ftello(input);
  Expect(byte_ct >= 0, "could not read temporary container size");
  Expect(fseeko(input, 0, SEEK_SET) == 0,
         "could not rewind temporary container");
  MemoryReader result;
  result.bytes.resize(static_cast<size_t>(byte_ct));
  Expect(fread(result.bytes.data(), 1, result.bytes.size(), input) ==
             result.bytes.size(),
         "could not read temporary container");
  Expect(fclose(input) == 0, "could not close temporary container");
  return result;
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

  MemoryReader memory_reader = LoadMemoryReader(path);
  MemoryReader obsolete_reader = memory_reader;
  obsolete_reader.bytes[6] = '1';
  ContainerReader obsolete_container;
  Expect(!obsolete_container.OpenReadAt(
             obsolete_reader.bytes.size(), ReadMemory,
             &obsolete_reader, &error),
         "obsolete experimental container identity was accepted");
  Expect(reader.OpenReadAt(memory_reader.bytes.size(), ReadMemory,
                           &memory_reader, &error),
         "ranged reader open failed: " + error);
  Expect(memory_reader.reads.size() == 2,
         "ranged open did not issue exactly header and index reads");
  std::vector<uint8_t> block_storage;
  EncodedBlockView block_view;
  Expect(reader.ReadBlockView(1, &block_storage, &block_view, &error),
         "ranged block read failed: " + error);
  Expect(memory_reader.reads.size() == 3,
         "ranged block read issued more than one request");
  const auto& ranged_entry = reader.block_index()[1];
  Expect(memory_reader.reads.back() ==
             std::make_pair(ranged_entry.file_offset,
                            static_cast<size_t>(ranged_entry.byte_ct)),
         "ranged block request did not match the block index");
  Expect(block_view.first_variant() == kBlockVariantCt,
         "block view first variant mismatch");
  Expect(block_view.variant_ct() == kVariantCt - kBlockVariantCt,
         "block view variant count mismatch");
  for (uint32_t offset = 0; offset != block_view.variant_ct(); ++offset) {
    const pgen_rans::ByteSpan record = block_view.record(offset);
    Expect(record.data && record.size, "block view returned an empty record");
  }
  Expect(!block_view.record(block_view.variant_ct()).data,
         "block view accepted an out-of-range record");
  const uint32_t decoded_word_stride = PackedWordCt(kSampleCt);
  std::vector<uint64_t> decoded_block(
      static_cast<size_t>(block_view.variant_ct()) * decoded_word_stride);
  CpuBlockDecoder cpu_decoder(4);
  Expect(cpu_decoder.thread_ct() == 4,
         "CPU decoder did not start the requested worker count");
  Expect(cpu_decoder.Decode(
             block_view, kSampleCt, codec_params, decoded_block.data(),
             decoded_block.size(), &error),
         "CPU block decode failed: " + error);
  for (uint32_t offset = 0; offset != block_view.variant_ct(); ++offset) {
    std::vector<uint64_t> observed(
        decoded_block.begin() +
            static_cast<std::ptrdiff_t>(offset * decoded_word_stride),
        decoded_block.begin() +
            static_cast<std::ptrdiff_t>((offset + 1) *
                                        decoded_word_stride));
    ExpectEqual(observed, source[block_view.first_variant() + offset],
                kSampleCt);
  }
  const uint32_t projected_samples[] = {0, 2, 33, 999, 1002};
  const size_t projected_stride = 4;
  std::vector<uint8_t> projected_block(
      block_view.variant_ct() * projected_stride, 0xa5);
  Expect(cpu_decoder.ProjectSampleSubset(
             decoded_block.data(), block_view.variant_ct(), kSampleCt,
             projected_samples, 5, projected_block.data(),
             projected_stride, &error),
         "CPU sample projection failed: " + error);
  for (uint32_t offset = 0; offset != block_view.variant_ct(); ++offset) {
    for (uint32_t subset_idx = 0; subset_idx != 5; ++subset_idx) {
      const uint8_t observed =
          (projected_block[offset * projected_stride +
                           subset_idx / 4] >>
           (2 * (subset_idx % 4))) &
          3U;
      const uint8_t expected = GetPackedGenotype(
          source[block_view.first_variant() + offset].data(),
          projected_samples[subset_idx]);
      Expect(observed == expected,
             "CPU sample projection genotype mismatch");
    }
    Expect(
        !(projected_block[offset * projected_stride + 1] & 0xfcU),
        "CPU sample projection did not clear packed tail bits");
    Expect(projected_block[offset * projected_stride + 2] == 0xa5,
           "CPU sample projection overwrote output stride padding");
    Expect(projected_block[offset * projected_stride + 3] == 0xa5,
           "CPU sample projection overwrote output stride padding");
  }
  const uint32_t invalid_projected_samples[] = {2, 2};
  Expect(!cpu_decoder.ProjectSampleSubset(
             decoded_block.data(), block_view.variant_ct(), kSampleCt,
             invalid_projected_samples, 2, projected_block.data(),
             projected_stride, &error),
         "CPU sample projection accepted duplicate indices");
  Expect(!cpu_decoder.Decode(
             block_view, kSampleCt, codec_params, decoded_block.data(),
             decoded_block.size() - 1, &error),
         "CPU block decoder accepted an undersized output");
  reader.Close();

  PackedVariantReader packed_reader;
  Expect(packed_reader.Open(path, 4, &error),
         "packed reader open failed: " + error);
  Expect(packed_reader.sample_ct() == kSampleCt,
         "packed reader sample count mismatch");
  Expect(packed_reader.variant_ct() == kVariantCt,
         "packed reader variant count mismatch");
  const size_t packed_byte_ct = (kSampleCt + 3) / 4;
  const size_t output_stride = packed_byte_ct + 5;
  std::vector<uint8_t> packed_output(7 * output_stride, 0xa5);
  PackedReadStats packed_stats;
  Expect(packed_reader.ReadRange(
             7, 7, packed_output.data(), output_stride, &packed_stats,
             &error),
         "packed range read failed: " + error);
  for (uint32_t output_idx = 0; output_idx != 7; ++output_idx) {
    Expect(!memcmp(packed_output.data() + output_idx * output_stride,
                   source[7 + output_idx].data(), packed_byte_ct),
           "packed range genotype mismatch");
    for (size_t padding_idx = packed_byte_ct;
         padding_idx != output_stride; ++padding_idx) {
      Expect(packed_output[output_idx * output_stride + padding_idx] == 0xa5,
             "packed range read overwrote output padding");
    }
  }
  const uint32_t requested_variants[] = {18, 1, 11, 10, 1};
  packed_output.assign(5 * output_stride, 0xa5);
  Expect(packed_reader.ReadList(
             requested_variants, 5, packed_output.data(), output_stride,
             &packed_stats, &error),
         "packed list read failed: " + error);
  for (uint32_t output_idx = 0; output_idx != 5; ++output_idx) {
    Expect(!memcmp(packed_output.data() + output_idx * output_stride,
                   source[requested_variants[output_idx]].data(),
                   packed_byte_ct),
           "packed list genotype mismatch");
  }
  const uint64_t cached_read_ct = packed_stats.block_read_ct;
  packed_output.assign(2 * output_stride, 0xa5);
  Expect(packed_reader.ReadRange(
             10, 2, packed_output.data(), output_stride, &packed_stats,
             &error),
         "cached packed range read failed: " + error);
  Expect(packed_stats.block_read_ct == cached_read_ct,
         "packed reader did not reuse its decoded block cache");
  Expect(packed_stats.returned_variant_ct == 14,
         "packed reader returned-variant accounting mismatch");
  Expect(!packed_reader.ReadRange(
             kVariantCt, 1, packed_output.data(), output_stride,
             &packed_stats, &error),
         "packed reader accepted an out-of-range request");
  std::vector<uint8_t> single_output(packed_byte_ct, 0);
  Expect(packed_reader.ReadVariant(
             4, single_output.data(), single_output.size(),
             &packed_stats, &error),
         "packed single-variant read failed: " + error);
  Expect(!memcmp(single_output.data(), source[4].data(), packed_byte_ct),
         "packed single-variant genotype mismatch");
  Expect(packed_stats.returned_variant_ct == 15,
         "packed single-variant accounting mismatch");
  const uint32_t sample_subset[] = {0, 2, 33, 999, 1002};
  Expect(packed_reader.SetSampleSubset(
             sample_subset, 5, &error),
         "packed reader sample subset failed: " + error);
  Expect(packed_reader.raw_sample_ct() == kSampleCt,
         "packed reader raw sample count changed");
  Expect(packed_reader.sample_ct() == 5,
         "packed reader subset sample count mismatch");
  Expect(packed_reader.packed_variant_byte_ct() == 2,
         "packed reader subset byte count mismatch");
  std::vector<uint8_t> subset_output(5 * 4, 0xa5);
  Expect(packed_reader.ReadList(
             requested_variants, 5, subset_output.data(), 4,
             &packed_stats, &error),
         "packed subset read failed: " + error);
  for (uint32_t output_idx = 0; output_idx != 5; ++output_idx) {
    for (uint32_t subset_idx = 0; subset_idx != 5; ++subset_idx) {
      const uint8_t observed =
          static_cast<uint8_t>(
              (subset_output[4 * output_idx + subset_idx / 4] >>
               (2 * (subset_idx % 4))) &
              3U);
      const uint8_t expected = GetPackedGenotype(
          source[requested_variants[output_idx]].data(),
          sample_subset[subset_idx]);
      Expect(observed == expected, "packed subset genotype mismatch");
    }
    Expect((subset_output[4 * output_idx + 1] & 0xfcU) == 0,
           "packed subset trailing bits are nonzero");
    Expect((subset_output[4 * output_idx + 2] == 0xa5) &&
               (subset_output[4 * output_idx + 3] == 0xa5),
           "packed subset read overwrote output padding");
  }
  const uint32_t invalid_subset[] = {2, 1};
  Expect(!packed_reader.SetSampleSubset(
             invalid_subset, 2, &error),
         "packed reader accepted an unsorted sample subset");
  Expect(packed_reader.sample_ct() == 5,
         "invalid sample subset changed reader state");
  packed_reader.ClearSampleSubset();
  Expect(packed_reader.sample_ct() == kSampleCt,
         "packed reader did not clear its sample subset");
  packed_reader.Close();

  FILE* corrupt_file = fopen(path.c_str(), "r+b");
  Expect(corrupt_file != nullptr, "could not reopen temporary container");
  Expect(fseeko(corrupt_file, -1, SEEK_END) == 0,
         "could not seek to container payload");
  const int original_byte = fgetc(corrupt_file);
  Expect(original_byte != EOF, "could not read container payload byte");
  Expect(fseeko(corrupt_file, -1, SEEK_CUR) == 0,
         "could not rewind container payload byte");
  Expect(fputc(original_byte ^ 1, corrupt_file) != EOF,
         "could not corrupt container payload byte");
  Expect(fclose(corrupt_file) == 0, "could not close corrupted container");
  Expect(reader.Open(path, &error),
         "corrupted container index could not be opened: " + error);
  EncodedBlock corrupted_block;
  Expect(!reader.ReadBlock(1, &corrupted_block, &error),
         "block payload bit flip was not detected");
  reader.Close();

  Expect(unlink(path.c_str()) == 0, "temporary file cleanup failed");
  puts("pgen_rans_container_test: PASS");
  return 0;
}
