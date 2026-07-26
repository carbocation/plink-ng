// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <thread>
#include <unordered_map>
#include <vector>

#include "include/pgenlib_misc.h"
#include "include/pgenlib_read.h"
#include "include/plink2_base.h"
#include "include/plink2_bits.h"
#include "pgen_rans_codec.h"
#include "pgen_rans_benchmark.h"
#include "pgen_rans_container.h"
#include "pgen_rans_cpu.h"
#include "pgen_rans_encode.h"

namespace {

using namespace plink2;
using pgen_rans::CodecParams;
using pgen_rans::BlockIndexEntry;
using pgen_rans::ContainerParams;
using pgen_rans::ContainerReader;
using pgen_rans::CpuBlockDecoder;
using pgen_rans::DecodeMultiallelicPatches;
using pgen_rans::DecodeRecord;
using pgen_rans::EncodeInput;
using pgen_rans::EncodeParams;
using pgen_rans::EncodePgenRans;
using pgen_rans::EncodeStats;
using pgen_rans::EncodedBlock;
using pgen_rans::EncodedBlockView;
using pgen_rans::GetBaseRecordByteCt;
using pgen_rans::MultiallelicPatches;
using pgen_rans::ParseRecordMetadata;
using pgen_rans::RecordMetadata;
using pgen_rans::RecordMode;
using pgen_rans::ScheduledAnchorOffsets;
using pgen_rans::VariantMetadata;

static_assert(sizeof(uintptr_t) == sizeof(uint64_t),
              "The experimental codec currently requires 64-bit words.");

struct Options {
  std::string command;
  std::string input_fname;
  std::string output_fname;
  std::string pvar_fname;
  uint32_t block_variant_ct = 128;
  uint32_t anchor_ct = 32;
  uint32_t two_ref_shortlist = 4;
  uint64_t max_anchor_bp = 1000000;
  uint32_t restart_variant_ct = 64;
  uint32_t rans_state_ct = 0;
  uint32_t rans_scale_bits = 12;
  uint32_t thread_ct = 0;
  uint32_t variant_limit = UINT32_MAX;
  uint32_t benchmark_block_ct = 80;
  uint32_t benchmark_iteration_ct = 0;
  uint32_t benchmark_warmup_ct = 2;
  uint32_t benchmark_run_ct = 7;
  uint64_t benchmark_seed = 17817764168754780924ULL;
  bool enable_alternate_records = true;
};

struct PvarData {
  std::vector<VariantMetadata> variants;
};

void PrintUsage(FILE* out) {
  fputs(
      "Usage:\n"
      "  pgen_rans encode <source.pgen> <rans.pgen> [options]\n"
      "  pgen_rans verify <source.pgen> <rans.pgen> [options]\n"
      "  pgen_rans benchmark <source.pgen> <rans.pgen> [options]\n"
      "  pgen_rans inspect <rans.pgen>\n"
      "\n"
      "Encode options:\n"
      "  --pvar <file>              Required plain-text PVAR schema.\n"
      "  --block-variants <n>       Maximum variants per block (default 128).\n"
      "  --anchors <n>              Independent anchors per full block (default 32).\n"
      "  --two-ref-shortlist <n>    Pair the n best single anchors; 0 disables (default 4).\n"
      "  --max-anchor-bp <n>        Maximum reference distance (default 1000000).\n"
      "  --restart-variants <n>     Record-offset restart interval (default 64).\n"
      "  --rans-states <auto|n>     Independent rANS lanes (default auto: 16 on ARM64; x86 uses 16 small/32 large).\n"
      "  --rans-scale-bits <n>      Frequency precision, 8..16 (default 12).\n"
      "  --alternate-modes <on|off> Enable exact raw/sparse record selection (default on).\n"
      "  --threads <n>              Reference-selection threads (default hardware count).\n"
      "  --variant-limit <n>        Encode only the first n variants.\n"
      "\n"
      "Verify options:\n"
      "  --pvar <file>              Required plain-text PVAR schema.\n"
      "  --variant-limit <n>        Verify only the first n variants.\n"
      "\n"
      "Benchmark options:\n"
      "  --pvar <file>              Optional plain-text PVAR schema.\n"
      "  --threads <n>              Persistent CPU decoder workers.\n"
      "  --blocks <n>               Deterministic block strata (default 80).\n"
      "  --iterations <n>           Full passes/run (default auto, >=50 ms).\n"
      "  --warmup <n>               Untimed A/B passes (default 2).\n"
      "  --runs <n>                 Randomized timing runs (default 7).\n"
      "  --seed <n>                 Reproducible 64-bit order seed.\n",
      out);
}

bool ParseU32(const char* text, uint32_t min_value, uint32_t max_value,
              uint32_t* value) {
  if ((!text) || (!text[0])) {
    return false;
  }
  errno = 0;
  char* endp = nullptr;
  const unsigned long long parsed = strtoull(text, &endp, 10);
  if (errno || (!endp) || endp[0] || (parsed < min_value) ||
      (parsed > max_value)) {
    return false;
  }
  *value = static_cast<uint32_t>(parsed);
  return true;
}

bool ParseU64(const char* text, uint64_t min_value, uint64_t max_value,
              uint64_t* value) {
  if ((!text) || (!text[0])) {
    return false;
  }
  errno = 0;
  char* endp = nullptr;
  const unsigned long long parsed = strtoull(text, &endp, 10);
  if (errno || (!endp) || endp[0] || (parsed < min_value) ||
      (parsed > max_value)) {
    return false;
  }
  *value = static_cast<uint64_t>(parsed);
  return true;
}

bool ParseOptions(int argc, char** argv, Options* opts) {
  if ((argc < 2) || (!strcmp(argv[1], "--help"))) {
    PrintUsage((argc < 2) ? stderr : stdout);
    return argc >= 2;
  }
  opts->command = argv[1];
  if (opts->command == "inspect") {
    if (argc != 3) {
      PrintUsage(stderr);
      return false;
    }
    opts->input_fname = argv[2];
    return true;
  }
  if ((opts->command != "encode") && (opts->command != "verify") &&
      (opts->command != "benchmark")) {
    fprintf(stderr, "Error: Unknown command '%s'.\n", argv[1]);
    return false;
  }
  if (argc < 4) {
    PrintUsage(stderr);
    return false;
  }
  opts->input_fname = argv[2];
  opts->output_fname = argv[3];
  for (int arg_idx = 4; arg_idx < argc; ++arg_idx) {
    const char* arg = argv[arg_idx];
    if (!strcmp(arg, "--help")) {
      PrintUsage(stdout);
      exit(0);
    }
    if (arg_idx + 1 == argc) {
      fprintf(stderr, "Error: Missing value after %s.\n", arg);
      return false;
    }
    const char* value = argv[++arg_idx];
    if (!strcmp(arg, "--pvar")) {
      opts->pvar_fname = value;
    } else if (!strcmp(arg, "--block-variants")) {
      if (!ParseU32(value, 2, UINT16_MAX, &(opts->block_variant_ct))) {
        fprintf(stderr, "Error: Invalid --block-variants value '%s'.\n",
                value);
        return false;
      }
    } else if (!strcmp(arg, "--anchors")) {
      if (!ParseU32(value, 1, 256, &(opts->anchor_ct))) {
        fprintf(stderr, "Error: Invalid --anchors value '%s'.\n", value);
        return false;
      }
    } else if (!strcmp(arg, "--two-ref-shortlist")) {
      if (!ParseU32(value, 0, 256, &(opts->two_ref_shortlist))) {
        fprintf(stderr, "Error: Invalid --two-ref-shortlist value '%s'.\n",
                value);
        return false;
      }
    } else if (!strcmp(arg, "--max-anchor-bp")) {
      if (!ParseU64(value, 0, UINT64_MAX, &(opts->max_anchor_bp))) {
        fprintf(stderr, "Error: Invalid --max-anchor-bp value '%s'.\n",
                value);
        return false;
      }
    } else if (!strcmp(arg, "--restart-variants")) {
      if (!ParseU32(value, 1, UINT16_MAX,
                    &(opts->restart_variant_ct))) {
        fprintf(stderr, "Error: Invalid --restart-variants value '%s'.\n",
                value);
        return false;
      }
    } else if (!strcmp(arg, "--rans-states")) {
      if (!strcmp(value, "auto")) {
        opts->rans_state_ct = 0;
      } else if (!ParseU32(value, 1, 256, &(opts->rans_state_ct))) {
        fprintf(stderr, "Error: Invalid --rans-states value '%s'.\n", value);
        return false;
      }
    } else if (!strcmp(arg, "--rans-scale-bits")) {
      if (!ParseU32(value, 8, 16, &(opts->rans_scale_bits))) {
        fprintf(stderr, "Error: Invalid --rans-scale-bits value '%s'.\n",
                value);
        return false;
      }
    } else if (!strcmp(arg, "--alternate-modes")) {
      if (!strcmp(value, "on")) {
        opts->enable_alternate_records = true;
      } else if (!strcmp(value, "off")) {
        opts->enable_alternate_records = false;
      } else {
        fprintf(stderr,
                "Error: Invalid --alternate-modes value '%s'.\n", value);
        return false;
      }
    } else if (!strcmp(arg, "--threads")) {
      if (!ParseU32(value, 1, 1024, &(opts->thread_ct))) {
        fprintf(stderr, "Error: Invalid --threads value '%s'.\n", value);
        return false;
      }
    } else if (!strcmp(arg, "--variant-limit")) {
      if (!ParseU32(value, 1, UINT32_MAX, &(opts->variant_limit))) {
        fprintf(stderr, "Error: Invalid --variant-limit value '%s'.\n",
                value);
        return false;
      }
    } else if (!strcmp(arg, "--blocks")) {
      if ((opts->command != "benchmark") ||
          (!ParseU32(value, 1, UINT32_MAX,
                     &(opts->benchmark_block_ct)))) {
        fprintf(stderr, "Error: Invalid --blocks value '%s'.\n", value);
        return false;
      }
    } else if (!strcmp(arg, "--iterations")) {
      if ((opts->command != "benchmark") ||
          (!ParseU32(value, 1, 1000,
                     &(opts->benchmark_iteration_ct)))) {
        fprintf(stderr, "Error: Invalid --iterations value '%s'.\n", value);
        return false;
      }
    } else if (!strcmp(arg, "--warmup")) {
      if ((opts->command != "benchmark") ||
          (!ParseU32(value, 1, 100,
                     &(opts->benchmark_warmup_ct)))) {
        fprintf(stderr, "Error: Invalid --warmup value '%s'.\n", value);
        return false;
      }
    } else if (!strcmp(arg, "--runs")) {
      if ((opts->command != "benchmark") ||
          (!ParseU32(value, 1, 101,
                     &(opts->benchmark_run_ct)))) {
        fprintf(stderr, "Error: Invalid --runs value '%s'.\n", value);
        return false;
      }
    } else if (!strcmp(arg, "--seed")) {
      if ((opts->command != "benchmark") ||
          (!ParseU64(value, 0, UINT64_MAX,
                     &(opts->benchmark_seed)))) {
        fprintf(stderr, "Error: Invalid --seed value '%s'.\n", value);
        return false;
      }
    } else {
      fprintf(stderr, "Error: Unknown option '%s'.\n", arg);
      return false;
    }
  }
  if (opts->command == "verify") {
    if ((opts->block_variant_ct != 128) || (opts->anchor_ct != 32) ||
        (opts->two_ref_shortlist != 4) ||
        (opts->max_anchor_bp != 1000000) ||
        (opts->restart_variant_ct != 64) ||
        opts->rans_state_ct || (opts->rans_scale_bits != 12) ||
        opts->thread_ct || (!opts->enable_alternate_records)) {
      fputs(
          "Error: verify accepts only --pvar and --variant-limit.\n",
          stderr);
      return false;
    }
  } else if (opts->command == "benchmark") {
    if ((opts->block_variant_ct != 128) || (opts->anchor_ct != 32) ||
        (opts->two_ref_shortlist != 4) ||
        (opts->max_anchor_bp != 1000000) ||
        (opts->restart_variant_ct != 64) ||
        opts->rans_state_ct || (opts->rans_scale_bits != 12) ||
        (opts->variant_limit != UINT32_MAX) ||
        (!opts->enable_alternate_records)) {
      fputs(
          "Error: benchmark accepts only --pvar, --threads, --blocks, "
          "--iterations, --warmup, --runs, and --seed.\n",
          stderr);
      return false;
    }
  }
  if (((opts->command == "encode") || (opts->command == "verify")) &&
      opts->pvar_fname.empty()) {
    fprintf(
        stderr,
        "Error: %s requires --pvar so multiallelic allele counts are "
        "known.\n",
        opts->command.c_str());
    return false;
  }
  opts->anchor_ct = std::min(opts->anchor_ct, opts->block_variant_ct);
  if (opts->two_ref_shortlist == 1) {
    fputs("Error: --two-ref-shortlist must be 0 or at least 2.\n", stderr);
    return false;
  }
  opts->two_ref_shortlist =
      std::min(opts->two_ref_shortlist, opts->anchor_ct);
  if (opts->anchor_ct < 2) {
    opts->two_ref_shortlist = 0;
  }
  if (!opts->thread_ct) {
    opts->thread_ct =
        std::max(1U, std::thread::hardware_concurrency());
  }
  return true;
}

std::vector<std::string> SplitWhitespace(const std::string& line) {
  std::istringstream line_stream(line);
  std::vector<std::string> fields;
  std::string field;
  while (line_stream >> field) {
    fields.push_back(field);
  }
  return fields;
}

bool LoadPvar(const std::string& fname, PvarData* pvar,
              std::string* error) {
  if ((fname.size() >= 4) && (fname.substr(fname.size() - 4) == ".zst")) {
    *error = "The prototype requires a plain-text PVAR.";
    return false;
  }
  std::ifstream input(fname);
  if (!input) {
    *error = "Could not open PVAR '" + fname + "'.";
    return false;
  }
  pvar->variants.clear();
  std::unordered_map<std::string, uint32_t> chrom_codes;
  int32_t chrom_col = 0;
  int32_t pos_col = 1;
  int32_t alt_col = 4;
  std::string line;
  while (std::getline(input, line)) {
    if (line.empty()) {
      continue;
    }
    if (line[0] == '#') {
      if ((line.size() >= 6) &&
          ((line.compare(0, 6, "#CHROM") == 0) ||
           (line.compare(0, 6, "#chrom") == 0))) {
        const std::vector<std::string> fields = SplitWhitespace(line);
        chrom_col = -1;
        pos_col = -1;
        alt_col = -1;
        for (uint32_t col = 0; col != fields.size(); ++col) {
          std::string name = fields[col];
          if ((!name.empty()) && (name[0] == '#')) {
            name.erase(0, 1);
          }
          if ((name == "CHROM") || (name == "chrom")) {
            chrom_col = static_cast<int32_t>(col);
          } else if ((name == "POS") || (name == "pos")) {
            pos_col = static_cast<int32_t>(col);
          } else if ((name == "ALT") || (name == "alt")) {
            alt_col = static_cast<int32_t>(col);
          }
        }
      }
      continue;
    }
    const std::vector<std::string> fields = SplitWhitespace(line);
    const int32_t required_col =
        std::max(chrom_col, std::max(pos_col, alt_col));
    if ((chrom_col < 0) || (pos_col < 0) ||
        (required_col >= static_cast<int32_t>(fields.size()))) {
      *error = "Malformed PVAR data line.";
      return false;
    }
    uint64_t bp;
    if (!ParseU64(fields[pos_col].c_str(), 0, UINT64_MAX, &bp)) {
      *error = "Invalid PVAR position.";
      return false;
    }
    const std::string& chrom = fields[chrom_col];
    const auto inserted = chrom_codes.emplace(
        chrom, static_cast<uint32_t>(chrom_codes.size()));
    VariantMetadata meta;
    meta.chrom_code = inserted.first->second;
    meta.bp = bp;
    uint32_t allele_ct = 2;
    if ((alt_col >= 0) && (fields[alt_col] != ".")) {
      allele_ct += static_cast<uint32_t>(
          std::count(
              fields[alt_col].begin(), fields[alt_col].end(), ','));
    }
    if (allele_ct > 255) {
      *error = "PVAR variant has more than 255 alleles.";
      return false;
    }
    meta.allele_ct = static_cast<uint16_t>(allele_ct);
    pvar->variants.push_back(meta);
  }
  return true;
}

uint64_t FileSize(const std::string& fname) {
  struct stat stat_buf;
  if (stat(fname.c_str(), &stat_buf)) {
    return 0;
  }
  return static_cast<uint64_t>(stat_buf.st_size);
}

class PgenInput {
 public:
  PgenInput() {
    PreinitPgfi(&pgfi_);
    PreinitPgr(&pgr_);
  }

  ~PgenInput() {
    Close();
  }

  bool Open(const std::string& fname, const PvarData* pvar,
            std::string* error) {
    char errstr_buf[kPglErrstrBufBlen];
    PgenHeaderCtrl header_ctrl;
    uintptr_t pgfi_alloc_cacheline_ct;
    PglErr pgl_error = PgfiInitPhase1(
        fname.c_str(), nullptr, UINT32_MAX, UINT32_MAX, &header_ctrl, &pgfi_,
        &pgfi_alloc_cacheline_ct, errstr_buf);
    if (pgl_error) {
      *error = errstr_buf;
      return false;
    }
    if (pvar) {
      if (pvar->variants.size() != pgfi_.raw_variant_ct) {
        *error = "PVAR/PGEN variant-count mismatch.";
        return false;
      }
      allele_idx_offsets_.resize(
          static_cast<size_t>(pgfi_.raw_variant_ct) + 1);
      uintptr_t allele_idx_offset = 0;
      uint32_t max_allele_ct = 2;
      for (uint32_t variant_idx = 0;
           variant_idx != pgfi_.raw_variant_ct; ++variant_idx) {
        allele_idx_offsets_[variant_idx] = allele_idx_offset;
        const uint32_t cur_allele_ct =
            pvar->variants[variant_idx].allele_ct;
        allele_idx_offset += cur_allele_ct;
        max_allele_ct = std::max(max_allele_ct, cur_allele_ct);
      }
      allele_idx_offsets_[pgfi_.raw_variant_ct] = allele_idx_offset;
      pgfi_.allele_idx_offsets = allele_idx_offsets_.data();
      pgfi_.max_allele_ct = max_allele_ct;
    }
    if (cachealigned_malloc(pgfi_alloc_cacheline_ct * kCacheline,
                            &pgfi_alloc_)) {
      *error = "Out of memory allocating the PGEN index.";
      return false;
    }
    uint32_t maximum_record_width;
    uintptr_t pgr_alloc_cacheline_ct;
    pgl_error = PgfiInitPhase2(
        header_ctrl, pvar ? 1 : 0, 0, 0, 0, pgfi_.raw_variant_ct,
        &maximum_record_width, &pgfi_, pgfi_alloc_,
        &pgr_alloc_cacheline_ct, errstr_buf);
    if (pgl_error) {
      *error = errstr_buf;
      return false;
    }
    if (cachealigned_malloc(pgr_alloc_cacheline_ct * kCacheline,
                            &pgr_alloc_)) {
      *error = "Out of memory allocating the PGEN reader.";
      return false;
    }
    pgl_error = PgrInit(fname.c_str(), maximum_record_width, &pgfi_, &pgr_,
                        pgr_alloc_);
    if (pgl_error) {
      *error = "PgrInit failed with code " +
               std::to_string(static_cast<uint32_t>(pgl_error)) + ".";
      return false;
    }
    if (pgfi_.gflags & (kfPgenGlobalHardcallPhasePresent |
                        kfPgenGlobalDosagePresent |
                        kfPgenGlobalDosagePhasePresent)) {
      *error =
          "Only unphased hardcall PGENs without dosage are supported.";
      return false;
    }
    if (pgfi_.max_allele_ct > 255) {
      *error = "PGEN variants with more than 255 alleles are unsupported.";
      return false;
    }
    if ((!pvar) && (pgfi_.max_allele_ct > 2)) {
      *error =
          "A plain-text PVAR is required for multiallelic PGEN input.";
      return false;
    }
    if ((pgfi_.max_allele_ct > 2) &&
        !AllocatePatchBuffers(error)) {
      return false;
    }
    PgrClearSampleSubsetIndex(&pgr_, &pssi_);
    open_ = true;
    return true;
  }

  void Close() {
    PglErr cleanup_error = kPglRetSuccess;
    CleanupPgr(&pgr_, &cleanup_error);
    CleanupPgfi(&pgfi_, &cleanup_error);
    if (pgr_alloc_) {
      aligned_free(pgr_alloc_);
      pgr_alloc_ = nullptr;
    }
    if (pgfi_alloc_) {
      aligned_free(pgfi_alloc_);
      pgfi_alloc_ = nullptr;
    }
    aligned_free_cond(patch_10_vals_);
    aligned_free_cond(patch_10_set_);
    aligned_free_cond(patch_01_vals_);
    aligned_free_cond(patch_01_set_);
    patch_10_vals_ = nullptr;
    patch_10_set_ = nullptr;
    patch_01_vals_ = nullptr;
    patch_01_set_ = nullptr;
    allele_idx_offsets_.clear();
    open_ = false;
  }

  bool Read(uint32_t variant_idx, uintptr_t* genovec,
            std::string* error) {
    if (!open_) {
      *error = "PGEN reader is not open.";
      return false;
    }
    const PglErr pgl_error =
        PgrGet(nullptr, pssi_, pgfi_.raw_sample_ct, variant_idx, &pgr_,
               genovec);
    if (pgl_error) {
      *error = "PgrGet failed at variant " +
               std::to_string(variant_idx) + " with code " +
               std::to_string(static_cast<uint32_t>(pgl_error)) + ".";
      return false;
    }
    ZeroTrailingNyps(pgfi_.raw_sample_ct, genovec);
    return true;
  }

  bool ReadExact(uint32_t variant_idx, uintptr_t* genovec,
                 MultiallelicPatches* patches, std::string* error) {
    if ((!patches) || (!open_)) {
      *error = "Invalid exact PGEN read request.";
      return false;
    }
    *patches = MultiallelicPatches();
    const uint16_t cur_allele_ct = allele_ct(variant_idx);
    if (cur_allele_ct == 2) {
      return Read(variant_idx, genovec, error);
    }
    PgenVariant pgv = {};
    pgv.genovec = genovec;
    pgv.patch_01_set = patch_01_set_;
    pgv.patch_01_vals = patch_01_vals_;
    pgv.patch_10_set = patch_10_set_;
    pgv.patch_10_vals = patch_10_vals_;
    const PglErr pgl_error =
        PgrGetM(
            nullptr, pssi_, pgfi_.raw_sample_ct, variant_idx,
            &pgr_, &pgv);
    if (pgl_error) {
      *error = "PgrGetM failed at variant " +
               std::to_string(variant_idx) + " with code " +
               std::to_string(static_cast<uint32_t>(pgl_error)) + ".";
      return false;
    }
    ZeroTrailingNyps(pgfi_.raw_sample_ct, genovec);
    patches->allele_ct = cur_allele_ct;
    patches->patch_01_sample_ids.resize(pgv.patch_01_ct);
    patches->patch_01_values.assign(
        pgv.patch_01_vals, pgv.patch_01_vals + pgv.patch_01_ct);
    if (pgv.patch_01_ct) {
      uintptr_t sample_idx_base = 0;
      uintptr_t cur_bits = pgv.patch_01_set[0];
      for (uint32_t patch_idx = 0; patch_idx != pgv.patch_01_ct;
           ++patch_idx) {
        patches->patch_01_sample_ids[patch_idx] =
            static_cast<uint32_t>(
                BitIter1(
                    pgv.patch_01_set, &sample_idx_base, &cur_bits));
      }
    }
    patches->patch_10_sample_ids.resize(pgv.patch_10_ct);
    patches->patch_10_values.assign(
        pgv.patch_10_vals, pgv.patch_10_vals + 2 * pgv.patch_10_ct);
    if (pgv.patch_10_ct) {
      uintptr_t sample_idx_base = 0;
      uintptr_t cur_bits = pgv.patch_10_set[0];
      for (uint32_t patch_idx = 0; patch_idx != pgv.patch_10_ct;
           ++patch_idx) {
        patches->patch_10_sample_ids[patch_idx] =
            static_cast<uint32_t>(
                BitIter1(
                    pgv.patch_10_set, &sample_idx_base, &cur_bits));
      }
    }
    return true;
  }

  uint32_t sample_ct() const { return pgfi_.raw_sample_ct; }
  uint32_t variant_ct() const { return pgfi_.raw_variant_ct; }
  uint16_t allele_ct(uint32_t variant_idx) const {
    return pgfi_.allele_idx_offsets
               ? static_cast<uint16_t>(
                     pgfi_.allele_idx_offsets[variant_idx + 1] -
                     pgfi_.allele_idx_offsets[variant_idx])
               : 2;
  }
  uint32_t record_byte_ct(uint32_t variant_idx) const {
    return GetPgfiVrecWidth(&pgfi_, variant_idx);
  }
  PgenFileInfo* pgfi() { return &pgfi_; }
  PgenReader* reader() { return &pgr_; }

 private:
  bool AllocatePatchBuffers(std::string* error) {
    const uint32_t patch_set_word_ct =
        BitCtToVecCt(pgfi_.raw_sample_ct) * kWordsPerVec;
    if (cachealigned_malloc(
            static_cast<uintptr_t>(patch_set_word_ct) *
                sizeof(uintptr_t),
            &patch_01_set_) ||
        cachealigned_malloc(
            static_cast<uintptr_t>(pgfi_.raw_sample_ct) *
                sizeof(AlleleCode),
            &patch_01_vals_) ||
        cachealigned_malloc(
            static_cast<uintptr_t>(patch_set_word_ct) *
                sizeof(uintptr_t),
            &patch_10_set_) ||
        cachealigned_malloc(
            static_cast<uintptr_t>(2) * pgfi_.raw_sample_ct *
                sizeof(AlleleCode),
            &patch_10_vals_)) {
      *error = "Out of memory allocating PGEN multiallelic patches.";
      return false;
    }
    return true;
  }

  PgenFileInfo pgfi_;
  PgenReader pgr_;
  PgrSampleSubsetIndex pssi_;
  unsigned char* pgfi_alloc_ = nullptr;
  unsigned char* pgr_alloc_ = nullptr;
  std::vector<uintptr_t> allele_idx_offsets_;
  uintptr_t* patch_01_set_ = nullptr;
  AlleleCode* patch_01_vals_ = nullptr;
  uintptr_t* patch_10_set_ = nullptr;
  AlleleCode* patch_10_vals_ = nullptr;
  bool open_ = false;
};

int Encode(const Options& opts) {
  std::string error;
  PvarData pvar;
  const PvarData* pvar_ptr = nullptr;
  if (!opts.pvar_fname.empty()) {
    if (!LoadPvar(opts.pvar_fname, &pvar, &error)) {
      fprintf(stderr, "Error: %s\n", error.c_str());
      return 1;
    }
    pvar_ptr = &pvar;
  }
  PgenInput pgen;
  if (!pgen.Open(opts.input_fname, pvar_ptr, &error)) {
    fprintf(stderr, "Error: %s\n", error.c_str());
    return 1;
  }
  const uint32_t variant_ct =
      std::min(pgen.variant_ct(), opts.variant_limit);

  EncodeInput input;
  input.raw_sample_ct = pgen.sample_ct();
  input.sample_ct = pgen.sample_ct();
  input.raw_variant_ct = pgen.variant_ct();
  input.variant_ct = variant_ct;
  input.variant_metadata = pvar.variants.data();
  input.pgfi = pgen.pgfi();
  input.pgen_reader = pgen.reader();
  EncodeParams params;
  params.block_variant_ct = opts.block_variant_ct;
  params.anchor_ct = opts.anchor_ct;
  params.two_ref_shortlist = opts.two_ref_shortlist;
  params.max_anchor_bp = opts.max_anchor_bp;
  params.restart_variant_ct = opts.restart_variant_ct;
  params.rans_state_ct = opts.rans_state_ct;
  params.rans_scale_bits = opts.rans_scale_bits;
  params.thread_ct = opts.thread_ct;
  params.enable_alternate_records = opts.enable_alternate_records;
  EncodeStats stats;
  printf("Encoding %s ... ", opts.output_fname.c_str());
  fflush(stdout);
  if (EncodePgenRans(opts.output_fname, input, params, &stats, &error)) {
    fprintf(stderr, "\nError: %s\n", error.c_str());
    return 1;
  }
  printf("done.\n\nConditional-rANS encode complete\n");
  printf("  samples:                 %u\n", pgen.sample_ct());
  printf("  variants:                %u\n", stats.variant_ct);
  printf("  blocks:                  %u\n", stats.block_ct);
  printf("  anchors:                 %llu\n",
         static_cast<unsigned long long>(stats.anchor_ct));
  printf("  marginal records:        %llu\n",
         static_cast<unsigned long long>(stats.marginal_ct));
  printf("  one-reference records:   %llu\n",
         static_cast<unsigned long long>(stats.one_reference_ct));
  printf("  two-reference records:   %llu\n",
         static_cast<unsigned long long>(stats.two_reference_ct));
  printf("  entropy-rANS records:    %llu\n",
         static_cast<unsigned long long>(stats.entropy_rans_ct));
  printf("  deterministic records:   %llu\n",
         static_cast<unsigned long long>(stats.deterministic_rans_ct));
  printf("  raw-packed records:      %llu\n",
         static_cast<unsigned long long>(stats.raw_packed_ct));
  printf("  sparse-predictor records: %llu\n",
         static_cast<unsigned long long>(stats.sparse_predictor_ct));
  printf("  multiallelic variants:   %llu\n",
         static_cast<unsigned long long>(stats.multiallelic_ct));
  printf("  ref/ALT patches:         %llu\n",
         static_cast<unsigned long long>(stats.patch_01_ct));
  printf("  ALT/ALT patches:         %llu\n",
         static_cast<unsigned long long>(stats.patch_10_ct));
  printf("  multiallelic bytes:      %llu\n",
         static_cast<unsigned long long>(
             stats.multiallelic_patch_bytes));
  printf("  PGEN payload bytes:      %llu\n",
         static_cast<unsigned long long>(stats.pgen_payload_bytes));
  printf("  output bytes:            %llu\n",
         static_cast<unsigned long long>(stats.output_bytes));
  printf("  payload/output ratio:    %.3f\n",
         stats.output_bytes
             ? static_cast<double>(stats.pgen_payload_bytes) /
                   stats.output_bytes
             : 0.0);
  printf("  elapsed seconds:         %.3f\n", stats.elapsed_seconds);
  return 0;
}

bool GenotypesEqual(const uintptr_t* expected,
                    const std::vector<uint64_t>& observed,
                    uint32_t sample_ct) {
  const uint32_t word_ct = NypCtToWordCt(sample_ct);
  return !memcmp(expected, observed.data(),
                 static_cast<size_t>(word_ct) * sizeof(uintptr_t));
}

bool PatchesEqual(const MultiallelicPatches& expected,
                  const MultiallelicPatches& observed) {
  return (expected.allele_ct == observed.allele_ct) &&
         (expected.patch_01_sample_ids ==
          observed.patch_01_sample_ids) &&
         (expected.patch_01_values == observed.patch_01_values) &&
         (expected.patch_10_sample_ids ==
          observed.patch_10_sample_ids) &&
         (expected.patch_10_values == observed.patch_10_values);
}

int Verify(const Options& opts) {
  std::string error;
  PvarData pvar;
  const PvarData* pvar_ptr = nullptr;
  if (!opts.pvar_fname.empty()) {
    if (!LoadPvar(opts.pvar_fname, &pvar, &error)) {
      fprintf(stderr, "Error: %s\n", error.c_str());
      return 1;
    }
    pvar_ptr = &pvar;
  }
  PgenInput pgen;
  if (!pgen.Open(opts.input_fname, pvar_ptr, &error)) {
    fprintf(stderr, "Error: %s\n", error.c_str());
    return 1;
  }
  ContainerReader reader;
  if (!reader.Open(opts.output_fname, &error)) {
    fprintf(stderr, "Error: %s\n", error.c_str());
    return 1;
  }
  const ContainerParams& params = reader.params();
  if ((params.sample_ct != pgen.sample_ct()) ||
      (params.variant_ct > pgen.variant_ct())) {
    fputs("Error: Source/conditional-PGEN dimension mismatch.\n", stderr);
    return 1;
  }
  const uint32_t verify_variant_ct =
      std::min(params.variant_ct, opts.variant_limit);
  const CodecParams codec_params = {
      params.state_ct, params.scale_bits};
  const uint32_t genovec_word_stride =
      NypCtToVecCt(pgen.sample_ct()) * kWordsPerVec;
  uintptr_t* expected = nullptr;
  if (cachealigned_malloc(genovec_word_stride * sizeof(uintptr_t),
                          &expected)) {
    fputs("Error: Out of memory allocating verification genotypes.\n",
          stderr);
    return 1;
  }
  uint64_t marginal_ct = 0;
  uint64_t one_reference_ct = 0;
  uint64_t two_reference_ct = 0;
  uint64_t entropy_rans_ct = 0;
  uint64_t deterministic_rans_ct = 0;
  uint64_t raw_packed_ct = 0;
  uint64_t sparse_predictor_ct = 0;
  uint64_t multiallelic_ct = 0;
  uint32_t verified_ct = 0;
  const auto start_time = std::chrono::steady_clock::now();
  int return_code = 1;
  for (uint32_t block_idx = 0;
       (block_idx != params.block_ct) &&
       (verified_ct != verify_variant_ct);
       ++block_idx) {
    EncodedBlock block;
    if (!reader.ReadBlock(block_idx, &block, &error)) {
      fprintf(stderr, "\nError: %s\n", error.c_str());
      goto cleanup;
    }
    const std::vector<uint32_t> anchor_offsets =
        ScheduledAnchorOffsets(
            static_cast<uint32_t>(block.records.size()), params.anchor_ct);
    std::vector<std::vector<uint64_t>> anchors(anchor_offsets.size());
    std::vector<const uint64_t*> anchor_ptrs(anchor_offsets.size());
    for (uint32_t anchor_idx = 0; anchor_idx != anchor_offsets.size();
         ++anchor_idx) {
      const uint32_t anchor_offset = anchor_offsets[anchor_idx];
      RecordMetadata metadata;
      if (!DecodeRecord(
              block.records[anchor_offset].data(),
              block.records[anchor_offset].size(), nullptr, 0,
              params.sample_ct, codec_params, &(anchors[anchor_idx]),
              &metadata, &error)) {
        fprintf(stderr, "\nError decoding anchor: %s\n", error.c_str());
        goto cleanup;
      }
      if (metadata.mode != RecordMode::kMarginal) {
        fputs("\nError: Anchor record is conditional.\n", stderr);
        goto cleanup;
      }
      anchor_ptrs[anchor_idx] = anchors[anchor_idx].data();
    }
    for (uint32_t offset = 0;
         (offset != block.records.size()) &&
         (verified_ct != verify_variant_ct);
         ++offset) {
      std::vector<uint64_t> decoded;
      RecordMetadata metadata;
      if (!DecodeRecord(
              block.records[offset].data(), block.records[offset].size(),
              anchor_ptrs.data(), static_cast<uint32_t>(anchor_ptrs.size()),
              params.sample_ct, codec_params, &decoded, &metadata, &error)) {
        fprintf(stderr, "\nError decoding variant %u: %s\n",
                block.first_variant + offset, error.c_str());
        goto cleanup;
      }
      MultiallelicPatches expected_patches;
      if (!pgen.ReadExact(
              block.first_variant + offset, expected,
              &expected_patches, &error)) {
        fprintf(stderr, "\nError: %s\n", error.c_str());
        goto cleanup;
      }
      if (!GenotypesEqual(expected, decoded, params.sample_ct)) {
        fprintf(stderr, "\nError: Genotype mismatch at variant %u.\n",
                block.first_variant + offset);
        goto cleanup;
      }
      MultiallelicPatches decoded_patches;
      if (!DecodeMultiallelicPatches(
              block.records[offset].data(),
              block.records[offset].size(), params.sample_ct,
              &decoded_patches, &error)) {
        fprintf(
            stderr,
            "\nError decoding multiallelic patches at variant %u: %s\n",
            block.first_variant + offset, error.c_str());
        goto cleanup;
      }
      if (!PatchesEqual(expected_patches, decoded_patches)) {
        fprintf(
            stderr,
            "\nError: Multiallelic patch mismatch at variant %u.\n",
            block.first_variant + offset);
        goto cleanup;
      }
      multiallelic_ct += decoded_patches.allele_ct > 2;
      if (metadata.mode == RecordMode::kMarginal) {
        ++marginal_ct;
      } else if (metadata.mode == RecordMode::kOneReference) {
        ++one_reference_ct;
      } else {
        ++two_reference_ct;
      }
      if (metadata.is_raw_packed) {
        ++raw_packed_ct;
      } else if (metadata.is_sparse_predictor) {
        ++sparse_predictor_ct;
      } else if (metadata.has_entropy_payload) {
        ++entropy_rans_ct;
      } else {
        ++deterministic_rans_ct;
      }
      ++verified_ct;
    }
    if (!(verified_ct % 10000) || (verified_ct == verify_variant_ct)) {
      fprintf(stderr, "\rVerified %u/%u variants (%.1f%%).", verified_ct,
              verify_variant_ct, 100.0 * verified_ct / verify_variant_ct);
      fflush(stderr);
    }
  }
  {
    const double elapsed_seconds =
        std::chrono::duration<double>(
            std::chrono::steady_clock::now() - start_time)
            .count();
    printf("\nConditional-rANS verification passed\n");
    printf("  variants:                %u\n", verified_ct);
    printf("  marginal records:        %llu\n",
           static_cast<unsigned long long>(marginal_ct));
    printf("  one-reference records:   %llu\n",
           static_cast<unsigned long long>(one_reference_ct));
    printf("  two-reference records:   %llu\n",
           static_cast<unsigned long long>(two_reference_ct));
    printf("  entropy-rANS records:    %llu\n",
           static_cast<unsigned long long>(entropy_rans_ct));
    printf("  deterministic records:   %llu\n",
           static_cast<unsigned long long>(deterministic_rans_ct));
    printf("  raw-packed records:      %llu\n",
           static_cast<unsigned long long>(raw_packed_ct));
    printf("  sparse-predictor records: %llu\n",
           static_cast<unsigned long long>(sparse_predictor_ct));
    printf("  multiallelic variants:   %llu\n",
           static_cast<unsigned long long>(multiallelic_ct));
    printf("  elapsed seconds:         %.3f\n", elapsed_seconds);
  }
  return_code = 0;

cleanup:
  aligned_free(expected);
  return return_code;
}

std::vector<uint32_t> SelectBenchmarkBlocks(uint32_t block_ct,
                                            uint32_t requested_block_ct) {
  const uint32_t selected_ct = std::min(block_ct, requested_block_ct);
  std::vector<uint32_t> result;
  result.reserve(selected_ct);
  for (uint32_t stratum_idx = 0; stratum_idx != selected_ct;
       ++stratum_idx) {
    const uint64_t numerator =
        static_cast<uint64_t>(2 * stratum_idx + 1) * block_ct;
    result.push_back(
        static_cast<uint32_t>(numerator / (2 * selected_ct)));
  }
  return result;
}

int Benchmark(const Options& opts) {
  std::string error;
  PvarData pvar;
  const PvarData* pvar_ptr = nullptr;
  if (!opts.pvar_fname.empty()) {
    if (!LoadPvar(opts.pvar_fname, &pvar, &error)) {
      fprintf(stderr, "Error: %s\n", error.c_str());
      return 1;
    }
    pvar_ptr = &pvar;
  }
  PgenInput pgen;
  if (!pgen.Open(opts.input_fname, pvar_ptr, &error)) {
    fprintf(stderr, "Error: %s\n", error.c_str());
    return 1;
  }
  ContainerReader reader;
  if (!reader.Open(opts.output_fname, &error)) {
    fprintf(stderr, "Error: %s\n", error.c_str());
    return 1;
  }
  const ContainerParams& params = reader.params();
  if ((params.sample_ct != pgen.sample_ct()) ||
      (params.variant_ct > pgen.variant_ct())) {
    fputs("Error: Source/conditional-PGEN dimension mismatch.\n", stderr);
    return 1;
  }
  const std::vector<uint32_t> selected_blocks =
      SelectBenchmarkBlocks(params.block_ct, opts.benchmark_block_ct);
  const CodecParams codec_params = {
      params.state_ct, params.scale_bits};
  CpuBlockDecoder cpu_decoder(opts.thread_ct);
  if (!cpu_decoder.thread_ct()) {
    fputs("Error: Could not start CPU decoder workers.\n", stderr);
    return 1;
  }
  const uint32_t packed_word_ct = pgen_rans::PackedWordCt(params.sample_ct);
  const uint32_t pgen_word_stride =
      NypCtToVecCt(params.sample_ct) * kWordsPerVec;
  uintptr_t* pgen_block_genovecs = nullptr;
  if (cachealigned_malloc(
          static_cast<uintptr_t>(params.block_variant_ct) *
              pgen_word_stride * sizeof(uintptr_t),
          &pgen_block_genovecs)) {
    fputs("Error: Out of memory allocating PGEN benchmark output.\n",
          stderr);
    return 1;
  }

  uint64_t pgen_payload_bytes = 0;
  uint64_t pgr_block_bytes = 0;
  uint64_t benchmark_variant_ct = 0;
  uint64_t checksum = 0;
  std::vector<uint8_t> block_storage;
  std::vector<uint64_t> decoded;
  std::vector<double> pgen_timing_samples;
  std::vector<double> pgr_read_timing_samples;
  std::vector<double> pgr_decode_timing_samples;
  std::vector<double> pgr_total_timing_samples;
  pgen_timing_samples.reserve(opts.benchmark_run_ct);
  pgr_read_timing_samples.reserve(opts.benchmark_run_ct);
  pgr_decode_timing_samples.reserve(opts.benchmark_run_ct);
  pgr_total_timing_samples.reserve(opts.benchmark_run_ct);
  uint32_t effective_iteration_ct = opts.benchmark_iteration_ct;
  double calibration_pgen_seconds = 0.0;
  double calibration_pgr_seconds = 0.0;
  int return_code = 1;

  for (const uint32_t block_idx : selected_blocks) {
    const BlockIndexEntry& entry = reader.block_index()[block_idx];
    for (uint32_t variant_offset = 0;
         variant_offset != entry.variant_ct; ++variant_offset) {
      pgen_payload_bytes +=
          pgen.record_byte_ct(entry.first_variant + variant_offset);
    }
    pgr_block_bytes += entry.byte_ct;
    benchmark_variant_ct += entry.variant_ct;
  }

  const auto run_pgen_block =
      [&](uint32_t block_idx, double* seconds) -> bool {
    const BlockIndexEntry& entry = reader.block_index()[block_idx];
    const auto start = std::chrono::steady_clock::now();
    for (uint32_t variant_offset = 0;
         variant_offset != entry.variant_ct; ++variant_offset) {
      uintptr_t* target =
          pgen_block_genovecs +
          static_cast<size_t>(variant_offset) * pgen_word_stride;
      if (!pgen.Read(entry.first_variant + variant_offset, target,
                     &error)) {
        return false;
      }
    }
    *seconds +=
        std::chrono::duration<double>(
            std::chrono::steady_clock::now() - start)
            .count();
    return true;
  };
  const auto run_pgr_block =
      [&](uint32_t block_idx, double* read_seconds,
          double* decode_seconds) -> bool {
    EncodedBlockView block;
    const auto read_start = std::chrono::steady_clock::now();
    if (!reader.ReadBlockView(
            block_idx, &block_storage, &block, &error)) {
      return false;
    }
    *read_seconds +=
        std::chrono::duration<double>(
            std::chrono::steady_clock::now() - read_start)
            .count();
    decoded.resize(
        static_cast<size_t>(block.variant_ct()) * packed_word_ct);
    const auto decode_start = std::chrono::steady_clock::now();
    if (!cpu_decoder.Decode(
            block, params.sample_ct, codec_params, decoded.data(),
            decoded.size(), &error)) {
      return false;
    }
    *decode_seconds +=
        std::chrono::duration<double>(
            std::chrono::steady_clock::now() - decode_start)
            .count();
    return true;
  };
  const auto verify_block = [&](uint32_t block_idx) -> bool {
    const BlockIndexEntry& entry = reader.block_index()[block_idx];
    for (uint32_t variant_offset = 0;
         variant_offset != entry.variant_ct; ++variant_offset) {
      const uint32_t variant_idx =
          entry.first_variant + variant_offset;
      const uintptr_t* expected =
          pgen_block_genovecs +
          static_cast<size_t>(variant_offset) * pgen_word_stride;
      const uint64_t* observed =
          decoded.data() +
          static_cast<size_t>(variant_offset) * packed_word_ct;
      if (memcmp(expected, observed,
                 static_cast<size_t>(packed_word_ct) *
                     sizeof(uint64_t))) {
        error = "CPU benchmark genotype mismatch at variant " +
                std::to_string(variant_idx) + ".";
        return false;
      }
    }
    return true;
  };

  std::mt19937_64 order_rng(opts.benchmark_seed);
  std::vector<uint32_t> run_order(selected_blocks);
  for (uint32_t warmup_idx = 0;
       warmup_idx != opts.benchmark_warmup_ct; ++warmup_idx) {
    std::shuffle(run_order.begin(), run_order.end(), order_rng);
    double warmup_pgen_seconds = 0.0;
    double warmup_read_seconds = 0.0;
    double warmup_decode_seconds = 0.0;
    uint32_t completed_block_ct = 0;
    for (const uint32_t block_idx : run_order) {
      const bool pgr_first = (order_rng() & 1U) != 0;
      const bool success =
          pgr_first
              ? (run_pgr_block(
                     block_idx, &warmup_read_seconds,
                     &warmup_decode_seconds) &&
                 run_pgen_block(block_idx, &warmup_pgen_seconds))
              : (run_pgen_block(block_idx, &warmup_pgen_seconds) &&
                 run_pgr_block(
                     block_idx, &warmup_read_seconds,
                     &warmup_decode_seconds));
      if ((!success) || (!verify_block(block_idx))) {
        fprintf(stderr, "\nError: %s\n", error.c_str());
        goto cleanup;
      }
      ++completed_block_ct;
      fprintf(stderr,
              "\rWarmup %u/%u: %u/%zu blocks.",
              warmup_idx + 1, opts.benchmark_warmup_ct,
              completed_block_ct, selected_blocks.size());
      fflush(stderr);
    }
    calibration_pgen_seconds = warmup_pgen_seconds;
    calibration_pgr_seconds =
        warmup_read_seconds + warmup_decode_seconds;
  }
  if (!effective_iteration_ct) {
    const double calibration_seconds =
        std::max(calibration_pgen_seconds, calibration_pgr_seconds);
    effective_iteration_ct =
        (calibration_seconds > 0.0)
            ? static_cast<uint32_t>(
                  std::min(
                      1000.0,
                      std::max(
                          1.0, std::ceil(0.05 / calibration_seconds))))
            : 1;
  }

  for (uint32_t run_idx = 0;
       run_idx != opts.benchmark_run_ct; ++run_idx) {
    double pgen_seconds = 0.0;
    double pgr_read_seconds = 0.0;
    double pgr_decode_seconds = 0.0;
    for (uint32_t iteration_idx = 0;
         iteration_idx != effective_iteration_ct; ++iteration_idx) {
      std::shuffle(run_order.begin(), run_order.end(), order_rng);
      for (const uint32_t block_idx : run_order) {
        const bool pgr_first = (order_rng() & 1U) != 0;
        const bool success =
            pgr_first
                ? (run_pgr_block(
                       block_idx, &pgr_read_seconds,
                       &pgr_decode_seconds) &&
                   run_pgen_block(block_idx, &pgen_seconds))
                : (run_pgen_block(block_idx, &pgen_seconds) &&
                   run_pgr_block(
                       block_idx, &pgr_read_seconds,
                       &pgr_decode_seconds));
        if (!success) {
          fprintf(stderr, "\nError: %s\n", error.c_str());
          goto cleanup;
        }
        const uint64_t observed =
            decoded[(static_cast<size_t>(run_idx) + iteration_idx +
                     block_idx) %
                    decoded.size()];
        checksum ^=
            observed + 0x9e3779b97f4a7c15ULL +
            (checksum << 6) + (checksum >> 2);
      }
    }
    pgen_timing_samples.push_back(pgen_seconds);
    pgr_read_timing_samples.push_back(pgr_read_seconds);
    pgr_decode_timing_samples.push_back(pgr_decode_seconds);
    pgr_total_timing_samples.push_back(
        pgr_read_seconds + pgr_decode_seconds);
    fprintf(stderr, "\rTimed run %u/%u.                    ",
            run_idx + 1, opts.benchmark_run_ct);
    fflush(stderr);
  }

  {
    const pgen_rans::TimingSummary pgen_summary =
        pgen_rans::SummarizeTimings(pgen_timing_samples);
    const pgen_rans::TimingSummary pgr_read_summary =
        pgen_rans::SummarizeTimings(pgr_read_timing_samples);
    const pgen_rans::TimingSummary pgr_decode_summary =
        pgen_rans::SummarizeTimings(pgr_decode_timing_samples);
    const pgen_rans::TimingSummary pgr_total_summary =
        pgen_rans::SummarizeTimings(pgr_total_timing_samples);
    const long double calls_per_run =
        static_cast<long double>(benchmark_variant_ct) *
        params.sample_ct * effective_iteration_ct;
    const uint64_t pgen_file_bytes = FileSize(opts.input_fname);
    const uint64_t pgr_file_bytes = FileSize(opts.output_fname);
    printf("\nConditional-rANS CPU benchmark\n");
    printf("  samples:                 %u\n", params.sample_ct);
    printf("  variants:                %llu\n",
           static_cast<unsigned long long>(benchmark_variant_ct));
    printf("  blocks:                  %zu / %u\n", selected_blocks.size(),
           params.block_ct);
    printf("  decoder threads:         %u\n", cpu_decoder.thread_ct());
    printf("  passes/timing run:       %u\n",
           effective_iteration_ct);
    printf("  pass calibration:        %s\n",
           opts.benchmark_iteration_ct ? "explicit" : "automatic");
    printf("  warmup/timed runs:       %u / %u\n",
           opts.benchmark_warmup_ct, opts.benchmark_run_ct);
    printf("  randomized-order seed:   %llu\n",
           static_cast<unsigned long long>(opts.benchmark_seed));
    printf("  checksum:                %016llx\n",
           static_cast<unsigned long long>(checksum));
    printf("\nSize accounting\n");
    printf("  selected source payload: %llu\n",
           static_cast<unsigned long long>(pgen_payload_bytes));
    printf("  selected rANS blocks:    %llu\n",
           static_cast<unsigned long long>(pgr_block_bytes));
    printf("  selected source/rANS:    %.6f\n",
           pgr_block_bytes
               ? static_cast<double>(pgen_payload_bytes) /
                     pgr_block_bytes
               : 0.0);
    printf("  full source file:        %llu\n",
           static_cast<unsigned long long>(pgen_file_bytes));
    printf("  full rANS file:          %llu\n",
           static_cast<unsigned long long>(pgr_file_bytes));
    printf("  full source/rANS:        %.6f\n",
           pgr_file_bytes
               ? static_cast<double>(pgen_file_bytes) / pgr_file_bytes
               : 0.0);
    printf("\nCurrent PGEN\n");
    printf("  read+decode median:      %.6f s\n",
           pgen_summary.median);
    printf("  min / max / MAD:         %.6f / %.6f / %.6f s\n",
           pgen_summary.minimum, pgen_summary.maximum,
           pgen_summary.median_absolute_deviation);
    printf("  billion calls/second:    %.3Lf\n",
           (pgen_summary.median > 0.0)
               ? calls_per_run / pgen_summary.median / 1.0e9L
               : 0.0L);
    printf("\nConditional rANS\n");
    printf("  block-read median:       %.6f s (MAD %.6f)\n",
           pgr_read_summary.median,
           pgr_read_summary.median_absolute_deviation);
    printf("  decode median:           %.6f s (MAD %.6f)\n",
           pgr_decode_summary.median,
           pgr_decode_summary.median_absolute_deviation);
    printf("  read+decode median:      %.6f s\n",
           pgr_total_summary.median);
    printf("  min / max / MAD:         %.6f / %.6f / %.6f s\n",
           pgr_total_summary.minimum, pgr_total_summary.maximum,
           pgr_total_summary.median_absolute_deviation);
    printf("  decode billion calls/s:  %.3Lf\n",
           (pgr_decode_summary.median > 0.0)
               ? calls_per_run / pgr_decode_summary.median / 1.0e9L
               : 0.0L);
    printf("  source/rANS-PGEN serial speedup: %.3f\n",
           (pgr_total_summary.median > 0.0)
               ? pgen_summary.median / pgr_total_summary.median
               : 0.0);
  }
  return_code = 0;

cleanup:
  aligned_free(pgen_block_genovecs);
  return return_code;
}

int Inspect(const Options& opts) {
  ContainerReader reader;
  std::string error;
  if (!reader.Open(opts.input_fname, &error)) {
    fprintf(stderr, "Error: %s\n", error.c_str());
    return 1;
  }
  const ContainerParams& params = reader.params();
  uint64_t marginal_ct = 0;
  uint64_t one_reference_ct = 0;
  uint64_t two_reference_ct = 0;
  uint64_t entropy_rans_ct = 0;
  uint64_t deterministic_rans_ct = 0;
  uint64_t raw_packed_ct = 0;
  uint64_t sparse_predictor_ct = 0;
  uint64_t multiallelic_ct = 0;
  uint64_t patch_01_ct = 0;
  uint64_t patch_10_ct = 0;
  uint64_t multiallelic_patch_bytes = 0;
  uint64_t record_bytes = 0;
  for (uint32_t block_idx = 0; block_idx != params.block_ct; ++block_idx) {
    EncodedBlock block;
    if (!reader.ReadBlock(block_idx, &block, &error)) {
      fprintf(stderr, "Error: %s\n", error.c_str());
      return 1;
    }
    for (const std::vector<uint8_t>& record : block.records) {
      RecordMetadata metadata;
      if (!ParseRecordMetadata(record.data(), record.size(), &metadata,
                               &error)) {
        fprintf(stderr, "Error: %s\n", error.c_str());
        return 1;
      }
      MultiallelicPatches patches;
      if (!DecodeMultiallelicPatches(
              record.data(), record.size(), params.sample_ct,
              &patches, &error)) {
        fprintf(stderr, "Error: %s\n", error.c_str());
        return 1;
      }
      if (patches.allele_ct > 2) {
        size_t base_record_size;
        if (!GetBaseRecordByteCt(
                record.data(), record.size(), &base_record_size,
                &error)) {
          fprintf(stderr, "Error: %s\n", error.c_str());
          return 1;
        }
        ++multiallelic_ct;
        patch_01_ct += patches.patch_01_sample_ids.size();
        patch_10_ct += patches.patch_10_sample_ids.size();
        multiallelic_patch_bytes += record.size() - base_record_size;
      }
      record_bytes += record.size();
      if (metadata.mode == RecordMode::kMarginal) {
        ++marginal_ct;
      } else if (metadata.mode == RecordMode::kOneReference) {
        ++one_reference_ct;
      } else {
        ++two_reference_ct;
      }
      if (metadata.is_raw_packed) {
        ++raw_packed_ct;
      } else if (metadata.is_sparse_predictor) {
        ++sparse_predictor_ct;
      } else if (metadata.has_entropy_payload) {
        ++entropy_rans_ct;
      } else {
        ++deterministic_rans_ct;
      }
    }
  }
  printf("Conditional-rANS PGEN\n");
  printf("  samples:                 %u\n", params.sample_ct);
  printf("  variants:                %u\n", params.variant_ct);
  printf("  blocks:                  %u\n", params.block_ct);
  printf("  block/anchors:           %u / %u\n", params.block_variant_ct,
         params.anchor_ct);
  printf("  rANS states/scale:       %u / %u bits\n", params.state_ct,
         params.scale_bits);
  printf("  restart interval:        %u\n", params.restart_variant_ct);
  printf("  marginal records:        %llu\n",
         static_cast<unsigned long long>(marginal_ct));
  printf("  one-reference records:   %llu\n",
         static_cast<unsigned long long>(one_reference_ct));
  printf("  two-reference records:   %llu\n",
         static_cast<unsigned long long>(two_reference_ct));
  printf("  entropy-rANS records:    %llu\n",
         static_cast<unsigned long long>(entropy_rans_ct));
  printf("  deterministic records:   %llu\n",
         static_cast<unsigned long long>(deterministic_rans_ct));
  printf("  raw-packed records:      %llu\n",
         static_cast<unsigned long long>(raw_packed_ct));
  printf("  sparse-predictor records: %llu\n",
         static_cast<unsigned long long>(sparse_predictor_ct));
  printf("  multiallelic variants:   %llu\n",
         static_cast<unsigned long long>(multiallelic_ct));
  printf("  ref/ALT patches:         %llu\n",
         static_cast<unsigned long long>(patch_01_ct));
  printf("  ALT/ALT patches:         %llu\n",
         static_cast<unsigned long long>(patch_10_ct));
  printf("  multiallelic bytes:      %llu\n",
         static_cast<unsigned long long>(multiallelic_patch_bytes));
  printf("  record payload bytes:    %llu\n",
         static_cast<unsigned long long>(record_bytes));
  printf("  file bytes:              %llu\n",
         static_cast<unsigned long long>(FileSize(opts.input_fname)));
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  Options opts;
  if (!ParseOptions(argc, argv, &opts)) {
    return 2;
  }
  if ((argc >= 2) && (!strcmp(argv[1], "--help"))) {
    return 0;
  }
  if (opts.command == "encode") {
    return Encode(opts);
  }
  if (opts.command == "verify") {
    return Verify(opts);
  }
  if (opts.command == "benchmark") {
    return Benchmark(opts);
  }
  return Inspect(opts);
}
