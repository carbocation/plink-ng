// SPDX-License-Identifier: GPL-3.0-or-later

#include "pgen_rans_pgenlib.h"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unistd.h>
#include <vector>

#include "include/plink2_bits.h"
#include "include/pgenlib_write.h"
#include "pgen_rans_container.h"
#include "pgen_rans_hybrid.h"

namespace {

using namespace plink2;
using pgen_rans::ContainerMetadata;
using pgen_rans::ContainerParams;
using pgen_rans::ContainerWriter;
using pgen_rans::EncodeRawRecord;
using pgen_rans::EncodeSparsePredictorRecord;
using pgen_rans::EncodedBlock;
using pgen_rans::GetPackedGenotype;
using pgen_rans::PackedWordCt;
using pgen_rans::PgenlibRansBackend;
using pgen_rans::PgenStorageMode;
using pgen_rans::RecordMode;
using pgen_rans::SetPackedGenotype;
using pgen_rans::UnifiedPgenOpenOptions;
using pgen_rans::UnifiedPgenReader;

constexpr uint32_t kSampleCt = 33;
constexpr uint32_t kVariantCt = 2;

static_assert(sizeof(uintptr_t) == sizeof(uint64_t),
              "The experimental codec currently requires 64-bit words.");

[[noreturn]] void Fail(const std::string& message) {
  fprintf(stderr, "FAIL: %s\n", message.c_str());
  exit(1);
}

void Expect(bool condition, const std::string& message) {
  if (!condition) {
    Fail(message);
  }
}

std::string TemporaryPath(const char* label) {
  std::string pattern =
      std::string("/tmp/pgen_rans_") + label + ".XXXXXX";
  std::vector<char> path(pattern.begin(), pattern.end());
  path.push_back('\0');
  const int fd = mkstemp(path.data());
  if (fd < 0) {
    Fail("mkstemp failed");
  }
  close(fd);
  return path.data();
}

std::vector<uint64_t> MakeVariant(uint32_t variant_idx) {
  std::vector<uint64_t> result(PackedWordCt(kSampleCt), 0);
  for (uint32_t sample_idx = 0; sample_idx != kSampleCt; ++sample_idx) {
    uint8_t genotype = 0;
    if (!variant_idx) {
      if (sample_idx == 3) {
        genotype = 1;
      } else if (sample_idx == 17) {
        genotype = 3;
      }
    } else {
      genotype = static_cast<uint8_t>((sample_idx + 2) % 3);
      if ((sample_idx + 1) % 17 == 0) {
        genotype = 3;
      }
    }
    SetPackedGenotype(result.data(), sample_idx, genotype);
  }
  return result;
}

void WriteStandardPgen(
    const std::string& path,
    const std::vector<std::vector<uint64_t>>& variants) {
  FILE* file = fopen(path.c_str(), "wb");
  if (!file) {
    Fail("could not create standard PGEN fixture");
  }
  unsigned char header[12] = {0x6c, 0x1b, 0x02};
  memcpy(&(header[3]), &kVariantCt, sizeof(kVariantCt));
  memcpy(&(header[7]), &kSampleCt, sizeof(kSampleCt));
  const size_t packed_byte_ct = (kSampleCt + 3) / 4;
  bool ok = fwrite(header, sizeof(header), 1, file) == 1;
  for (const std::vector<uint64_t>& variant : variants) {
    ok = ok &&
         (fwrite(variant.data(), packed_byte_ct, 1, file) == 1);
  }
  const bool close_ok = !fclose(file);
  ok = ok && close_ok;
  if (!ok) {
    Fail("could not write standard PGEN fixture");
  }
}

void WriteRansPgen(
    const std::string& path,
    const std::vector<std::vector<uint64_t>>& variants) {
  EncodedBlock block;
  block.first_variant = 0;
  block.records.resize(kVariantCt);
  std::string error;
  Expect(EncodeSparsePredictorRecord(
             variants[0].data(), nullptr, nullptr, kSampleCt,
             RecordMode::kMarginal, 0, 0, &block.records[0], &error),
         "could not encode sparse rANS fixture: " + error);
  Expect(EncodeRawRecord(
             variants[1].data(), kSampleCt, &block.records[1], &error),
         "could not encode raw rANS fixture: " + error);
  ContainerMetadata metadata;
  metadata.nonref_flags.assign(1, 1);
  const ContainerParams params(
      kSampleCt, kVariantCt, kVariantCt, 1, 4, 12, 1, 1, 2);
  ContainerWriter writer;
  Expect(writer.Open(path, params, metadata, &error),
         "could not create rANS fixture: " + error);
  Expect(writer.WriteBlock(block, &error),
         "could not write rANS fixture: " + error);
  Expect(writer.Close(&error),
         "could not close rANS fixture: " + error);
}

void WriteDosagePgen(const std::string& path,
                     const std::vector<uint64_t>& hardcalls) {
  STPgenWriter writer;
  PreinitSpgw(&writer);
  uintptr_t alloc_cacheline_ct = 0;
  uint32_t max_vrec_len = 0;
  Expect(SpgwInitPhase1(
             path.c_str(), nullptr, nullptr, 1, kSampleCt, 0,
             kPgenWriteBackwardSeek, kfPgenGlobalDosagePresent, 2,
             &writer, &alloc_cacheline_ct, &max_vrec_len) ==
             kPglRetSuccess,
         "standard dosage writer initialization failed");
  unsigned char* writer_alloc = nullptr;
  Expect(!cachealigned_malloc(alloc_cacheline_ct * kCacheline,
                              &writer_alloc),
         "standard dosage writer allocation failed");
  SpgwInitPhase2(max_vrec_len, &writer, writer_alloc);
  alignas(kCacheline) uintptr_t genovec[8] = {};
  Expect(hardcalls.size() <= (sizeof(genovec) / sizeof(*genovec)),
         "standard dosage hardcall buffer is too large");
  memcpy(genovec, hardcalls.data(),
         hardcalls.size() * sizeof(hardcalls[0]));
  alignas(kCacheline) uintptr_t dosage_present[8] = {};
  SetBit(5, dosage_present);
  const uint16_t dosage_main[1] = {8192};
  Expect(SpgwAppendBiallelicGenovecDosage16(
             genovec, dosage_present, dosage_main, 1, &writer) ==
             kPglRetSuccess,
         "standard dosage record write failed");
  Expect(SpgwFinish(&writer) == kPglRetSuccess,
         "standard dosage writer finish failed");
  aligned_free(writer_alloc);
}

void ExpectVariant(
    UnifiedPgenReader* reader, uint32_t variant_idx,
    const std::vector<uint64_t>& expected) {
  PgrSampleSubsetIndex pssi;
  PgrClearSampleSubsetIndex(reader->pgen_reader(), &pssi);
  alignas(kCacheline) uintptr_t genovec[8] = {};
  const PglErr reterr = PgrGet(
      nullptr, pssi, kSampleCt, variant_idx, reader->pgen_reader(),
      genovec);
  Expect(reterr == kPglRetSuccess, "unified PgrGet failed");
  for (uint32_t sample_idx = 0; sample_idx != kSampleCt; ++sample_idx) {
    Expect(GetNyparrEntry(genovec, sample_idx) ==
               GetPackedGenotype(expected.data(), sample_idx),
           "unified PGEN genotype mismatch");
  }
}

void ExpectSparseRans(UnifiedPgenReader* reader) {
  PgrSampleSubsetIndex pssi;
  PgrClearSampleSubsetIndex(reader->pgen_reader(), &pssi);
  alignas(kCacheline) uintptr_t genovec[8] = {};
  alignas(kCacheline) uintptr_t dosage_present[8] = {};
  alignas(kCacheline) uint16_t dosage_main[kSampleCt] = {};
  alignas(kCacheline) uint32_t sample_ids[kSampleCt] = {};
  uint32_t dosage_ct = 0;
  uint16_t common_dosage = 1;
  const PglErr reterr = PgrGetDMaybeSparse(
      nullptr, pssi, kSampleCt, 0, 4, reader->pgen_reader(),
      genovec, dosage_present, dosage_main, &dosage_ct,
      &common_dosage, sample_ids);
  Expect((reterr == kPglRetSuccess) && (common_dosage == 0) &&
             (dosage_ct == 2) && (sample_ids[0] == 3) &&
             (sample_ids[1] == 17) && (dosage_main[0] == 16384) &&
             (dosage_main[1] == 65535),
         "unified rANS reader returned an incorrect sparse dosage list");

  common_dosage = 0;
  dosage_ct = UINT32_MAX;
  Expect(PgrGetDMaybeSparse(
             nullptr, pssi, kSampleCt, 0, 1, reader->pgen_reader(),
             genovec, dosage_present, dosage_main, &dosage_ct,
             &common_dosage, sample_ids) == kPglRetSuccess,
         "unified rANS sparse threshold fallback failed");
  Expect((common_dosage == 1) && (!dosage_ct),
         "unified rANS threshold fallback did not return dense hardcalls");
  for (uint32_t sample_idx = 0; sample_idx != kSampleCt; ++sample_idx) {
    const uint32_t expected =
        (sample_idx == 3) ? 1 : ((sample_idx == 17) ? 3 : 0);
    Expect(GetNyparrEntry(genovec, sample_idx) == expected,
           "unified rANS threshold fallback genotype mismatch");
  }
}

void ExpectDenseStandard(UnifiedPgenReader* reader,
                         const std::vector<uint64_t>& expected) {
  PgrSampleSubsetIndex pssi;
  PgrClearSampleSubsetIndex(reader->pgen_reader(), &pssi);
  alignas(kCacheline) uintptr_t genovec[8] = {};
  alignas(kCacheline) uintptr_t dosage_present[8] = {};
  alignas(kCacheline) uint16_t dosage_main[kSampleCt] = {};
  alignas(kCacheline) uint32_t sample_ids[kSampleCt] = {};
  uint32_t dosage_ct = UINT32_MAX;
  uint16_t common_dosage = 0;
  Expect(PgrGetDMaybeSparse(
             nullptr, pssi, kSampleCt, 1, 4, reader->pgen_reader(),
             genovec, dosage_present, dosage_main, &dosage_ct,
             &common_dosage, sample_ids) == kPglRetSuccess,
         "standard dense PgrGetDMaybeSparse failed");
  Expect((common_dosage == 1) && (!dosage_ct),
         "standard dense PgrGetDMaybeSparse returned a sparse count");
  for (uint32_t sample_idx = 0; sample_idx != kSampleCt; ++sample_idx) {
    Expect(GetNyparrEntry(genovec, sample_idx) ==
               GetPackedGenotype(expected.data(), sample_idx),
           "standard dense PgrGetDMaybeSparse genotype mismatch");
  }
}

void ExpectDenseStandardWithDosage(UnifiedPgenReader* reader,
                                   const std::vector<uint64_t>& expected) {
  PgrSampleSubsetIndex pssi;
  PgrClearSampleSubsetIndex(reader->pgen_reader(), &pssi);
  alignas(kCacheline) uintptr_t genovec[8] = {};
  alignas(kCacheline) uintptr_t dosage_present[8] = {};
  alignas(kCacheline) uint16_t dosage_main[kSampleCt] = {};
  alignas(kCacheline) uint32_t sample_ids[kSampleCt] = {};
  uint32_t dosage_ct = UINT32_MAX;
  uint16_t common_dosage = 0;
  Expect(PgrGetDMaybeSparse(
             nullptr, pssi, kSampleCt, 0, 4, reader->pgen_reader(),
             genovec, dosage_present, dosage_main, &dosage_ct,
             &common_dosage, sample_ids) == kPglRetSuccess,
         "standard dense dosage PgrGetDMaybeSparse failed");
  Expect((common_dosage == 1) && (dosage_ct == 1) &&
             IsSet(dosage_present, 5) && (dosage_main[0] == 8192),
         "standard dense dosage PgrGetDMaybeSparse result mismatch");
  for (uint32_t sample_idx = 0; sample_idx != kSampleCt; ++sample_idx) {
    Expect(GetNyparrEntry(genovec, sample_idx) ==
               GetPackedGenotype(expected.data(), sample_idx),
           "standard dense dosage PgrGetDMaybeSparse genotype mismatch");
  }
}

void ExpectUnphasedAndCounts(UnifiedPgenReader* reader) {
  PgrSampleSubsetIndex pssi;
  PgrClearSampleSubsetIndex(reader->pgen_reader(), &pssi);
  alignas(kCacheline) uintptr_t genovec[8] = {};
  alignas(kCacheline) uintptr_t phasepresent[8] = {};
  alignas(kCacheline) uintptr_t phaseinfo[8] = {};
  uint32_t phasepresent_ct = UINT32_MAX;
  Expect(PgrGetP(
             nullptr, pssi, kSampleCt, 0, reader->pgen_reader(), genovec,
             phasepresent, phaseinfo, &phasepresent_ct) == kPglRetSuccess,
         "unified rANS unphased hardcall read failed");
  Expect(!phasepresent_ct,
         "unified rANS reader reported nonexistent hardcall phase");

  std::array<uint32_t, 4> genocounts = {};
  uint64_t all_dosages[2] = {};
  double imp_r2 = 0.0;
  Expect(PgrGetDCounts(
             nullptr, nullptr, pssi, kSampleCt, 0, 0,
             reader->pgen_reader(), &imp_r2, genocounts,
             all_dosages) == kPglRetSuccess,
         "unified rANS hardcall dosage counts failed");
  Expect((genocounts[0] == 31) && (genocounts[1] == 1) &&
             (!genocounts[2]) && (genocounts[3] == 1) &&
             (all_dosages[0] == 63 * 16384ULL) &&
             (all_dosages[1] == 16384),
         "unified rANS hardcall dosage counts are incorrect");
}

void ExpectBackendContract(const std::string& path) {
  PgenlibRansBackend backend;
  std::string error;
  Expect(!backend.Validate(&error),
         "closed conditional-rANS backend validated successfully");
  Expect(backend.Open(path, 1, &error),
         "conditional-rANS backend open failed: " + error);

  PgenFileInfo pgfi = {};
  PgenReader pgr = {};
  PreinitPgfi(&pgfi);
  PreinitPgr(&pgr);
  pgfi.raw_sample_ct = backend.sample_ct();
  pgfi.raw_variant_ct = backend.variant_ct();
  pgfi.max_allele_ct = backend.max_allele_ct();
  Expect(backend.Install(&pgfi, &pgr, &error),
         "conditional-rANS backend install failed: " + error);

  PgenFileInfo second_pgfi = {};
  PgenReader second_pgr = {};
  PreinitPgfi(&second_pgfi);
  PreinitPgr(&second_pgr);
  second_pgfi.raw_sample_ct = backend.sample_ct();
  second_pgfi.raw_variant_ct = backend.variant_ct();
  second_pgfi.max_allele_ct = backend.max_allele_ct();
  Expect(!backend.Install(&second_pgfi, &second_pgr, &error),
         "conditional-rANS backend installed on two readers");
  Expect(!backend.Open(path, 1, &error),
         "installed conditional-rANS backend reopened without Close()");
  backend.Close();
  Expect(!backend.Validate(&error),
         "closed conditional-rANS backend remained valid");
}

}  // namespace

int main() {
  const std::vector<std::vector<uint64_t>> variants = {
      MakeVariant(0), MakeVariant(1)};
  const std::string standard_path = TemporaryPath("standard_pgen");
  const std::string dosage_path = TemporaryPath("dosage_pgen");
  const std::string rans_path = TemporaryPath("unified_pgen");
  WriteStandardPgen(standard_path, variants);
  unlink(dosage_path.c_str());
  WriteDosagePgen(dosage_path, variants[1]);
  unlink(rans_path.c_str());
  WriteRansPgen(rans_path, variants);
  ExpectBackendContract(rans_path);

  UnifiedPgenReader reader;
  std::string error;
  Expect(reader.Open(standard_path, {}, &error),
         "unified standard open failed: " + error);
  Expect(reader.is_open() &&
             (reader.storage_mode() == PgenStorageMode::kStandard) &&
             (reader.sample_ct() == kSampleCt) &&
             (reader.variant_ct() == kVariantCt),
         "unified standard metadata mismatch");
  ExpectVariant(&reader, 1, variants[1]);
  ExpectDenseStandard(&reader, variants[1]);

  Expect(reader.Open(dosage_path, {}, &error),
         "unified standard dosage open failed: " + error);
  ExpectDenseStandardWithDosage(&reader, variants[1]);

  Expect(reader.Open(rans_path, UnifiedPgenOpenOptions{2}, &error),
         "unified rANS open failed: " + error);
  Expect(reader.is_open() &&
             (reader.storage_mode() ==
              PgenStorageMode::kConditionalRans) &&
             reader.file_info()->nonref_flags &&
             IsSet(reader.file_info()->nonref_flags, 0) &&
             !IsSet(reader.file_info()->nonref_flags, 1),
         "unified rANS metadata mismatch");
  ExpectVariant(&reader, 1, variants[1]);
  ExpectSparseRans(&reader);
  ExpectUnphasedAndCounts(&reader);

  Expect(reader.Open(standard_path, {}, &error),
         "unified standard reopen failed: " + error);
  ExpectVariant(&reader, 0, variants[0]);
  reader.Close();
  Expect((!reader.is_open()) && (!reader.sample_ct()) &&
             (!reader.variant_ct()),
         "unified reader did not close cleanly");
  Expect(!reader.Open(rans_path, UnifiedPgenOpenOptions{0}, &error),
         "unified reader accepted zero decoder threads");

  Expect(unlink(standard_path.c_str()) == 0,
         "standard fixture cleanup failed");
  Expect(unlink(dosage_path.c_str()) == 0,
         "standard dosage fixture cleanup failed");
  Expect(unlink(rans_path.c_str()) == 0,
         "rANS fixture cleanup failed");
  puts("pgen_rans_pgenlib_test: PASS");
  return 0;
}
