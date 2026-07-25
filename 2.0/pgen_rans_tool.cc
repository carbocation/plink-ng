// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <mutex>
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
#include "pgen_rans_container.h"
#include "pgen_rans_cpu.h"

namespace {

using namespace plink2;
using pgen_rans::CodecParams;
using pgen_rans::ContainerParams;
using pgen_rans::ContainerReader;
using pgen_rans::ContainerWriter;
using pgen_rans::CpuBlockDecoder;
using pgen_rans::DecodeRecord;
using pgen_rans::EncodeRecord;
using pgen_rans::EncodedBlock;
using pgen_rans::EncodedBlockView;
using pgen_rans::EstimateRecordBytes;
using pgen_rans::ParseRecordMetadata;
using pgen_rans::RecordMetadata;
using pgen_rans::RecordMode;
using pgen_rans::ScheduledAnchorOffsets;

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
  uint32_t rans_state_ct = 32;
  uint32_t rans_scale_bits = 12;
  uint32_t thread_ct = 0;
  uint32_t variant_limit = UINT32_MAX;
  uint32_t benchmark_block_ct = 80;
  uint32_t benchmark_iteration_ct = 1;
};

struct VariantMeta {
  uint32_t chrom_code = 0;
  uint64_t bp = 0;
  bool multiallelic = false;
};

struct PvarData {
  std::vector<VariantMeta> variants;
  uint32_t multiallelic_ct = 0;
};

struct VariantBlock {
  uint32_t start = 0;
  uint32_t len = 0;
};

struct AnchorCandidate {
  uint64_t estimated_bytes = 0;
  uint32_t ordinal = 0;
  uint32_t offset = 0;
};

struct EncodeTotals {
  uint64_t pgen_payload_bytes = 0;
  uint64_t marginal_ct = 0;
  uint64_t one_reference_ct = 0;
  uint64_t two_reference_ct = 0;
  uint64_t anchor_ct = 0;
};

void PrintUsage(FILE* out) {
  fputs(
      "Usage:\n"
      "  pgen_rans encode <input.pgen> <output.pgr> [options]\n"
      "  pgen_rans verify <input.pgen> <input.pgr> [options]\n"
      "  pgen_rans benchmark <input.pgen> <input.pgr> [options]\n"
      "  pgen_rans inspect <input.pgr>\n"
      "\n"
      "Encode options:\n"
      "  --pvar <file>              Plain-text PVAR for chromosome/distance bounds.\n"
      "  --block-variants <n>       Maximum variants per block (default 128).\n"
      "  --anchors <n>              Independent anchors per full block (default 32).\n"
      "  --two-ref-shortlist <n>    Pair the n best single anchors; 0 disables (default 4).\n"
      "  --max-anchor-bp <n>        Maximum reference distance (default 1000000).\n"
      "  --restart-variants <n>     Record-offset restart interval (default 64).\n"
      "  --rans-states <n>          Independent rANS lanes (default 32).\n"
      "  --rans-scale-bits <n>      Frequency precision, 8..16 (default 12).\n"
      "  --threads <n>              Reference-selection threads (default hardware count).\n"
      "  --variant-limit <n>        Encode only the first n variants.\n"
      "\n"
      "Verify options:\n"
      "  --variant-limit <n>        Verify only the first n variants.\n"
      "\n"
      "Benchmark options:\n"
      "  --threads <n>              Persistent CPU decoder workers.\n"
      "  --blocks <n>               Deterministic block strata (default 80).\n"
      "  --iterations <n>           Decode passes per fetched block (default 1).\n",
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
      if (!ParseU32(value, 1, 256, &(opts->rans_state_ct))) {
        fprintf(stderr, "Error: Invalid --rans-states value '%s'.\n", value);
        return false;
      }
    } else if (!strcmp(arg, "--rans-scale-bits")) {
      if (!ParseU32(value, 8, 16, &(opts->rans_scale_bits))) {
        fprintf(stderr, "Error: Invalid --rans-scale-bits value '%s'.\n",
                value);
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
    } else {
      fprintf(stderr, "Error: Unknown option '%s'.\n", arg);
      return false;
    }
  }
  if (opts->command == "verify") {
    if (!opts->pvar_fname.empty() ||
        (opts->block_variant_ct != 128) || (opts->anchor_ct != 32) ||
        (opts->two_ref_shortlist != 4) ||
        (opts->max_anchor_bp != 1000000) ||
        (opts->restart_variant_ct != 64) ||
        (opts->rans_state_ct != 32) || (opts->rans_scale_bits != 12) ||
        opts->thread_ct) {
      fputs("Error: verify accepts only --variant-limit.\n", stderr);
      return false;
    }
  } else if (opts->command == "benchmark") {
    if (!opts->pvar_fname.empty() ||
        (opts->block_variant_ct != 128) || (opts->anchor_ct != 32) ||
        (opts->two_ref_shortlist != 4) ||
        (opts->max_anchor_bp != 1000000) ||
        (opts->restart_variant_ct != 64) ||
        (opts->rans_state_ct != 32) || (opts->rans_scale_bits != 12) ||
        (opts->variant_limit != UINT32_MAX)) {
      fputs(
          "Error: benchmark accepts only --threads, --blocks, and "
          "--iterations.\n",
          stderr);
      return false;
    }
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

bool LoadPvar(const std::string& fname, uint32_t expected_variant_ct,
              PvarData* pvar, std::string* error) {
  if ((fname.size() >= 4) && (fname.substr(fname.size() - 4) == ".zst")) {
    *error = "The prototype requires a plain-text PVAR.";
    return false;
  }
  std::ifstream input(fname);
  if (!input) {
    *error = "Could not open PVAR '" + fname + "'.";
    return false;
  }
  pvar->variants.reserve(expected_variant_ct);
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
    VariantMeta meta;
    meta.chrom_code = inserted.first->second;
    meta.bp = bp;
    meta.multiallelic =
        (alt_col >= 0) &&
        (fields[alt_col].find(',') != std::string::npos);
    pvar->multiallelic_ct += meta.multiallelic;
    pvar->variants.push_back(meta);
    if (pvar->variants.size() > expected_variant_ct) {
      *error = "PVAR has more variants than the PGEN header.";
      return false;
    }
  }
  if (pvar->variants.size() != expected_variant_ct) {
    *error = "PVAR/PGEN variant-count mismatch.";
    return false;
  }
  return true;
}

std::vector<VariantBlock> BuildBlocks(
    uint32_t variant_ct, uint32_t maximum_block_variant_ct,
    const std::vector<VariantMeta>* metadata) {
  std::vector<VariantBlock> blocks;
  uint32_t block_start = 0;
  while (block_start != variant_ct) {
    uint32_t block_len =
        std::min(maximum_block_variant_ct, variant_ct - block_start);
    if (metadata) {
      const uint32_t chrom_code = (*metadata)[block_start].chrom_code;
      for (uint32_t offset = 1; offset != block_len; ++offset) {
        if ((*metadata)[block_start + offset].chrom_code != chrom_code) {
          block_len = offset;
          break;
        }
      }
    }
    blocks.push_back({block_start, block_len});
    block_start += block_len;
  }
  return blocks;
}

bool AnchorEligible(uint32_t target_vidx, uint32_t anchor_vidx,
                    const std::vector<VariantMeta>* metadata,
                    uint64_t maximum_distance) {
  if (!metadata) {
    return true;
  }
  const VariantMeta& target = (*metadata)[target_vidx];
  const VariantMeta& anchor = (*metadata)[anchor_vidx];
  if (target.chrom_code != anchor.chrom_code) {
    return false;
  }
  if (!maximum_distance) {
    return true;
  }
  const uint64_t distance = (target.bp >= anchor.bp)
                                ? (target.bp - anchor.bp)
                                : (anchor.bp - target.bp);
  return distance <= maximum_distance;
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

  bool Open(const std::string& fname, std::string* error) {
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
    if (cachealigned_malloc(pgfi_alloc_cacheline_ct * kCacheline,
                            &pgfi_alloc_)) {
      *error = "Out of memory allocating the PGEN index.";
      return false;
    }
    uint32_t maximum_record_width;
    uintptr_t pgr_alloc_cacheline_ct;
    pgl_error = PgfiInitPhase2(
        header_ctrl, 0, 0, 0, 0, pgfi_.raw_variant_ct,
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
    if (pgfi_.max_allele_ct > 2) {
      *error = "Only biallelic PGENs are supported.";
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

  uint32_t sample_ct() const { return pgfi_.raw_sample_ct; }
  uint32_t variant_ct() const { return pgfi_.raw_variant_ct; }
  uint32_t record_byte_ct(uint32_t variant_idx) const {
    return GetPgfiVrecWidth(&pgfi_, variant_idx);
  }

 private:
  PgenFileInfo pgfi_;
  PgenReader pgr_;
  PgrSampleSubsetIndex pssi_;
  unsigned char* pgfi_alloc_ = nullptr;
  unsigned char* pgr_alloc_ = nullptr;
  bool open_ = false;
};

void CountJointGenotypes(const uintptr_t* anchor, const uintptr_t* target,
                         uint32_t sample_ct, const uint32_t* anchor_counts,
                         const uint32_t* target_counts,
                         uint32_t* joint_counts) {
  std::fill(joint_counts, &(joint_counts[16]), 0U);
  const uint32_t word_ct = NypCtToWordCt(sample_ct);
  const uint32_t trailing_sample_ct = sample_ct % kBitsPerWordD2;
  for (uint32_t word_idx = 0; word_idx != word_ct; ++word_idx) {
    uintptr_t valid_mask = kMask5555;
    if (trailing_sample_ct && (word_idx + 1 == word_ct)) {
      valid_mask = bzhi(kMask5555, 2 * trailing_sample_ct);
    }
    const uintptr_t anchor_word = anchor[word_idx];
    const uintptr_t target_word = target[word_idx];
    const uintptr_t anchor_low = anchor_word & valid_mask;
    const uintptr_t anchor_high = (anchor_word >> 1) & valid_mask;
    const uintptr_t target_low = target_word & valid_mask;
    const uintptr_t target_high = (target_word >> 1) & valid_mask;
    const uintptr_t anchor_masks[3] = {
        (~(anchor_low | anchor_high)) & valid_mask,
        anchor_low & (~anchor_high) & valid_mask,
        (~anchor_low) & anchor_high & valid_mask};
    const uintptr_t target_masks[3] = {
        (~(target_low | target_high)) & valid_mask,
        target_low & (~target_high) & valid_mask,
        (~target_low) & target_high & valid_mask};
    for (uint32_t anchor_genotype = 0; anchor_genotype != 3;
         ++anchor_genotype) {
      for (uint32_t target_genotype = 0; target_genotype != 3;
           ++target_genotype) {
        joint_counts[4 * anchor_genotype + target_genotype] +=
            PopcountWord(anchor_masks[anchor_genotype] &
                         target_masks[target_genotype]);
      }
    }
  }
  for (uint32_t anchor_genotype = 0; anchor_genotype != 3;
       ++anchor_genotype) {
    uint32_t known = 0;
    for (uint32_t target_genotype = 0; target_genotype != 3;
         ++target_genotype) {
      known += joint_counts[4 * anchor_genotype + target_genotype];
    }
    joint_counts[4 * anchor_genotype + 3] =
        anchor_counts[anchor_genotype] - known;
  }
  for (uint32_t target_genotype = 0; target_genotype != 3;
       ++target_genotype) {
    uint32_t known = 0;
    for (uint32_t anchor_genotype = 0; anchor_genotype != 3;
         ++anchor_genotype) {
      known += joint_counts[4 * anchor_genotype + target_genotype];
    }
    joint_counts[12 + target_genotype] =
        target_counts[target_genotype] - known;
  }
  joint_counts[15] =
      anchor_counts[3] - joint_counts[12] - joint_counts[13] -
      joint_counts[14];
}

void CountTripleGenotypes(const uintptr_t* anchor1,
                          const uintptr_t* anchor2,
                          const uintptr_t* target, uint32_t sample_ct,
                          uint32_t* triple_counts) {
  std::fill(triple_counts, &(triple_counts[64]), 0U);
  const uint32_t word_ct = NypCtToWordCt(sample_ct);
  const uint32_t trailing_sample_ct = sample_ct % kBitsPerWordD2;
  for (uint32_t word_idx = 0; word_idx != word_ct; ++word_idx) {
    uintptr_t valid_mask = kMask5555;
    if (trailing_sample_ct && (word_idx + 1 == word_ct)) {
      valid_mask = bzhi(kMask5555, 2 * trailing_sample_ct);
    }
    const uintptr_t words[3] = {
        anchor1[word_idx], anchor2[word_idx], target[word_idx]};
    uintptr_t masks[3][4];
    for (uint32_t variant_idx = 0; variant_idx != 3; ++variant_idx) {
      const uintptr_t low = words[variant_idx] & valid_mask;
      const uintptr_t high = (words[variant_idx] >> 1) & valid_mask;
      masks[variant_idx][0] = (~(low | high)) & valid_mask;
      masks[variant_idx][1] = low & (~high) & valid_mask;
      masks[variant_idx][2] = (~low) & high & valid_mask;
      masks[variant_idx][3] = low & high & valid_mask;
    }
    for (uint32_t genotype1 = 0; genotype1 != 4; ++genotype1) {
      for (uint32_t genotype2 = 0; genotype2 != 4; ++genotype2) {
        const uintptr_t references =
            masks[0][genotype1] & masks[1][genotype2];
        for (uint32_t target_genotype = 0; target_genotype != 4;
             ++target_genotype) {
          triple_counts[16 * genotype1 + 4 * genotype2 + target_genotype] +=
              PopcountWord(references & masks[2][target_genotype]);
        }
      }
    }
  }
}

bool EncodeVariant(const uintptr_t* target,
                   const std::array<uint32_t, 4>& target_counts,
                   uint32_t target_vidx, uint32_t block_start,
                   const uintptr_t* block_genovecs,
                   uint32_t genovec_word_stride, uint32_t sample_ct,
                   const std::vector<uint32_t>& anchor_offsets,
                   const std::vector<std::array<uint32_t, 4>>& counts,
                   const std::vector<VariantMeta>* metadata,
                   const Options& opts, const CodecParams& codec_params,
                   bool is_anchor, std::vector<uint8_t>* output,
                   std::string* error) {
  const auto* target64 = reinterpret_cast<const uint64_t*>(target);
  if (!EncodeRecord(target64, nullptr, nullptr, sample_ct,
                    RecordMode::kMarginal, 0, 0, codec_params, output,
                    error)) {
    return false;
  }
  if (is_anchor) {
    return true;
  }
  std::vector<AnchorCandidate> candidates;
  candidates.reserve(anchor_offsets.size());
  for (uint32_t anchor_ordinal = 0;
       anchor_ordinal != anchor_offsets.size(); ++anchor_ordinal) {
    const uint32_t anchor_offset = anchor_offsets[anchor_ordinal];
    const uint32_t anchor_vidx = block_start + anchor_offset;
    if (!AnchorEligible(target_vidx, anchor_vidx, metadata,
                        opts.max_anchor_bp)) {
      continue;
    }
    const uintptr_t* anchor =
        &(block_genovecs[static_cast<uintptr_t>(anchor_offset) *
                          genovec_word_stride]);
    uint32_t joint_counts[16];
    CountJointGenotypes(anchor, target, sample_ct,
                        counts[anchor_offset].data(), target_counts.data(),
                        joint_counts);
    uint64_t estimated_bytes;
    if (!EstimateRecordBytes(joint_counts, RecordMode::kOneReference,
                             codec_params, &estimated_bytes, error)) {
      return false;
    }
    candidates.push_back({estimated_bytes, anchor_ordinal, anchor_offset});
  }
  if (candidates.empty()) {
    return true;
  }
  std::sort(candidates.begin(), candidates.end(),
            [](const AnchorCandidate& lhs, const AnchorCandidate& rhs) {
              if (lhs.estimated_bytes != rhs.estimated_bytes) {
                return lhs.estimated_bytes < rhs.estimated_bytes;
              }
              return lhs.ordinal < rhs.ordinal;
            });
  const AnchorCandidate& best_single = candidates[0];
  const uintptr_t* single_anchor =
      &(block_genovecs[static_cast<uintptr_t>(best_single.offset) *
                        genovec_word_stride]);
  std::vector<uint8_t> single_record;
  if (!EncodeRecord(
          target64, reinterpret_cast<const uint64_t*>(single_anchor), nullptr,
          sample_ct, RecordMode::kOneReference,
          static_cast<uint8_t>(best_single.ordinal), 0, codec_params,
          &single_record, error)) {
    return false;
  }
  if (single_record.size() < output->size()) {
    *output = std::move(single_record);
  }

  if ((!opts.two_ref_shortlist) || (candidates.size() < 2)) {
    return true;
  }
  if (candidates.size() > opts.two_ref_shortlist) {
    candidates.resize(opts.two_ref_shortlist);
  }
  uint64_t best_pair_estimate = UINT64_MAX;
  uint32_t best_first_idx = 0;
  uint32_t best_second_idx = 0;
  for (uint32_t first_idx = 0; first_idx + 1 != candidates.size();
       ++first_idx) {
    const uintptr_t* anchor1 =
        &(block_genovecs[static_cast<uintptr_t>(
                              candidates[first_idx].offset) *
                          genovec_word_stride]);
    for (uint32_t second_idx = first_idx + 1;
         second_idx != candidates.size(); ++second_idx) {
      const uintptr_t* anchor2 =
          &(block_genovecs[static_cast<uintptr_t>(
                                candidates[second_idx].offset) *
                            genovec_word_stride]);
      uint32_t triple_counts[64];
      CountTripleGenotypes(anchor1, anchor2, target, sample_ct,
                           triple_counts);
      uint64_t estimated_bytes;
      if (!EstimateRecordBytes(triple_counts, RecordMode::kTwoReference,
                               codec_params, &estimated_bytes, error)) {
        return false;
      }
      if (estimated_bytes < best_pair_estimate) {
        best_pair_estimate = estimated_bytes;
        best_first_idx = first_idx;
        best_second_idx = second_idx;
      }
    }
  }
  const AnchorCandidate& first = candidates[best_first_idx];
  const AnchorCandidate& second = candidates[best_second_idx];
  const uintptr_t* anchor1 =
      &(block_genovecs[static_cast<uintptr_t>(first.offset) *
                        genovec_word_stride]);
  const uintptr_t* anchor2 =
      &(block_genovecs[static_cast<uintptr_t>(second.offset) *
                        genovec_word_stride]);
  std::vector<uint8_t> pair_record;
  if (!EncodeRecord(
          target64, reinterpret_cast<const uint64_t*>(anchor1),
          reinterpret_cast<const uint64_t*>(anchor2), sample_ct,
          RecordMode::kTwoReference, static_cast<uint8_t>(first.ordinal),
          static_cast<uint8_t>(second.ordinal), codec_params, &pair_record,
          error)) {
    return false;
  }
  if (pair_record.size() < output->size()) {
    *output = std::move(pair_record);
  }
  return true;
}

int Encode(const Options& opts) {
  PgenInput pgen;
  std::string error;
  if (!pgen.Open(opts.input_fname, &error)) {
    fprintf(stderr, "Error: %s\n", error.c_str());
    return 1;
  }
  const uint32_t variant_ct =
      std::min(pgen.variant_ct(), opts.variant_limit);
  PvarData pvar;
  const std::vector<VariantMeta>* metadata = nullptr;
  if (!opts.pvar_fname.empty()) {
    if (!LoadPvar(opts.pvar_fname, pgen.variant_ct(), &pvar, &error)) {
      fprintf(stderr, "Error: %s\n", error.c_str());
      return 1;
    }
    if (pvar.multiallelic_ct) {
      fprintf(stderr, "Error: PVAR contains %u multiallelic variants.\n",
              pvar.multiallelic_ct);
      return 1;
    }
    metadata = &pvar.variants;
  } else {
    fputs(
        "Warning: No PVAR supplied; chromosome and distance bounds are "
        "not enforced.\n",
        stderr);
  }
  const std::vector<VariantBlock> blocks =
      BuildBlocks(variant_ct, opts.block_variant_ct, metadata);
  const CodecParams codec_params = {
      opts.rans_state_ct, opts.rans_scale_bits};
  const ContainerParams container_params = {
      pgen.sample_ct(), variant_ct, opts.block_variant_ct, opts.anchor_ct,
      opts.rans_state_ct, opts.rans_scale_bits, opts.restart_variant_ct,
      static_cast<uint32_t>(blocks.size())};
  ContainerWriter writer;
  if (!writer.Open(opts.output_fname, container_params, &error)) {
    fprintf(stderr, "Error: %s\n", error.c_str());
    return 1;
  }
  const uint32_t genovec_word_stride =
      NypCtToVecCt(pgen.sample_ct()) * kWordsPerVec;
  uintptr_t* block_genovecs = nullptr;
  if (cachealigned_malloc(
          static_cast<uintptr_t>(opts.block_variant_ct) *
              genovec_word_stride * sizeof(uintptr_t),
          &block_genovecs)) {
    fputs("Error: Out of memory allocating the genotype block.\n", stderr);
    return 1;
  }

  EncodeTotals totals;
  uint32_t processed_variant_ct = 0;
  const auto start_time = std::chrono::steady_clock::now();
  int return_code = 1;
  for (const VariantBlock& block_range : blocks) {
    std::vector<std::array<uint32_t, 4>> counts(block_range.len);
    for (uint32_t offset = 0; offset != block_range.len; ++offset) {
      uintptr_t* genovec =
          &(block_genovecs[static_cast<uintptr_t>(offset) *
                            genovec_word_stride]);
      const uint32_t variant_idx = block_range.start + offset;
      if (!pgen.Read(variant_idx, genovec, &error)) {
        fprintf(stderr, "\nError: %s\n", error.c_str());
        goto cleanup;
      }
      GenoarrCountFreqsUnsafe(genovec, pgen.sample_ct(), counts[offset]);
      totals.pgen_payload_bytes += pgen.record_byte_ct(variant_idx);
    }
    const std::vector<uint32_t> anchor_offsets =
        ScheduledAnchorOffsets(block_range.len, opts.anchor_ct);
    std::vector<uint8_t> is_anchor(block_range.len, 0);
    for (const uint32_t anchor_offset : anchor_offsets) {
      is_anchor[anchor_offset] = 1;
    }
    EncodedBlock block;
    block.first_variant = block_range.start;
    block.records.resize(block_range.len);
    std::atomic<uint32_t> next_offset(0);
    std::atomic<bool> failed(false);
    std::mutex error_mutex;
    std::string worker_error;
    const uint32_t worker_ct =
        std::min(opts.thread_ct, block_range.len);
    std::vector<std::thread> workers;
    workers.reserve(worker_ct);
    for (uint32_t worker_idx = 0; worker_idx != worker_ct; ++worker_idx) {
      workers.emplace_back([&]() {
        uint32_t offset;
        while ((!failed.load(std::memory_order_relaxed)) &&
               ((offset = next_offset.fetch_add(
                     1, std::memory_order_relaxed)) < block_range.len)) {
          const uintptr_t* target =
              &(block_genovecs[static_cast<uintptr_t>(offset) *
                                genovec_word_stride]);
          std::string local_error;
          if (!EncodeVariant(
                  target, counts[offset], block_range.start + offset,
                  block_range.start, block_genovecs, genovec_word_stride,
                  pgen.sample_ct(), anchor_offsets, counts, metadata, opts,
                  codec_params, is_anchor[offset], &(block.records[offset]),
                  &local_error)) {
            failed.store(true, std::memory_order_relaxed);
            std::lock_guard<std::mutex> lock(error_mutex);
            if (worker_error.empty()) {
              worker_error = local_error;
            }
          }
        }
      });
    }
    for (std::thread& worker : workers) {
      worker.join();
    }
    if (failed) {
      fprintf(stderr, "\nError: %s\n", worker_error.c_str());
      goto cleanup;
    }
    for (uint32_t offset = 0; offset != block_range.len; ++offset) {
      RecordMetadata record_metadata;
      if (!ParseRecordMetadata(block.records[offset].data(),
                               block.records[offset].size(),
                               &record_metadata, &error)) {
        fprintf(stderr, "\nError: %s\n", error.c_str());
        goto cleanup;
      }
      totals.anchor_ct += is_anchor[offset];
      if (record_metadata.mode == RecordMode::kMarginal) {
        ++totals.marginal_ct;
      } else if (record_metadata.mode == RecordMode::kOneReference) {
        ++totals.one_reference_ct;
      } else {
        ++totals.two_reference_ct;
      }
    }
    if (!writer.WriteBlock(block, &error)) {
      fprintf(stderr, "\nError: %s\n", error.c_str());
      goto cleanup;
    }
    processed_variant_ct += block_range.len;
    if (!(processed_variant_ct % 10000) ||
        (processed_variant_ct == variant_ct)) {
      fprintf(stderr, "\rEncoded %u/%u variants (%.1f%%).",
              processed_variant_ct, variant_ct,
              100.0 * processed_variant_ct / variant_ct);
      fflush(stderr);
    }
  }
  if (!writer.Close(&error)) {
    fprintf(stderr, "\nError: %s\n", error.c_str());
    goto cleanup;
  }
  {
    const double elapsed_seconds =
        std::chrono::duration<double>(
            std::chrono::steady_clock::now() - start_time)
            .count();
    const uint64_t output_bytes = FileSize(opts.output_fname);
    printf("\nConditional-rANS encode complete\n");
    printf("  samples:                 %u\n", pgen.sample_ct());
    printf("  variants:                %u\n", variant_ct);
    printf("  blocks:                  %zu\n", blocks.size());
    printf("  anchors:                 %llu\n",
           static_cast<unsigned long long>(totals.anchor_ct));
    printf("  marginal records:        %llu\n",
           static_cast<unsigned long long>(totals.marginal_ct));
    printf("  one-reference records:   %llu\n",
           static_cast<unsigned long long>(totals.one_reference_ct));
    printf("  two-reference records:   %llu\n",
           static_cast<unsigned long long>(totals.two_reference_ct));
    printf("  PGEN payload bytes:      %llu\n",
           static_cast<unsigned long long>(totals.pgen_payload_bytes));
    printf("  output bytes:            %llu\n",
           static_cast<unsigned long long>(output_bytes));
    printf("  payload/output ratio:    %.3f\n",
           output_bytes
               ? static_cast<double>(totals.pgen_payload_bytes) /
                     output_bytes
               : 0.0);
    printf("  elapsed seconds:         %.3f\n", elapsed_seconds);
  }
  return_code = 0;

cleanup:
  aligned_free(block_genovecs);
  return return_code;
}

bool GenotypesEqual(const uintptr_t* expected,
                    const std::vector<uint64_t>& observed,
                    uint32_t sample_ct) {
  const uint32_t word_ct = NypCtToWordCt(sample_ct);
  return !memcmp(expected, observed.data(),
                 static_cast<size_t>(word_ct) * sizeof(uintptr_t));
}

int Verify(const Options& opts) {
  PgenInput pgen;
  std::string error;
  if (!pgen.Open(opts.input_fname, &error)) {
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
    fputs("Error: PGEN/container dimension mismatch.\n", stderr);
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
      if (!pgen.Read(block.first_variant + offset, expected, &error)) {
        fprintf(stderr, "\nError: %s\n", error.c_str());
        goto cleanup;
      }
      if (!GenotypesEqual(expected, decoded, params.sample_ct)) {
        fprintf(stderr, "\nError: Genotype mismatch at variant %u.\n",
                block.first_variant + offset);
        goto cleanup;
      }
      if (metadata.mode == RecordMode::kMarginal) {
        ++marginal_ct;
      } else if (metadata.mode == RecordMode::kOneReference) {
        ++one_reference_ct;
      } else {
        ++two_reference_ct;
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
  PgenInput pgen;
  std::string error;
  if (!pgen.Open(opts.input_fname, &error)) {
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
    fputs("Error: PGEN/container dimension mismatch.\n", stderr);
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
  double pgen_seconds = 0.0;
  double pgr_read_seconds = 0.0;
  double pgr_decode_seconds = 0.0;
  std::vector<uint8_t> block_storage;
  std::vector<uint64_t> decoded;
  uint32_t completed_block_ct = 0;
  int return_code = 1;
  for (const uint32_t block_idx : selected_blocks) {
    EncodedBlockView block;
    const auto read_start = std::chrono::steady_clock::now();
    if (!reader.ReadBlockView(block_idx, &block_storage, &block, &error)) {
      fprintf(stderr, "\nError: %s\n", error.c_str());
      goto cleanup;
    }
    pgr_read_seconds +=
        std::chrono::duration<double>(
            std::chrono::steady_clock::now() - read_start)
            .count();
    const size_t decoded_word_ct =
        static_cast<size_t>(block.variant_ct()) * packed_word_ct;
    decoded.resize(decoded_word_ct);
    for (uint32_t iteration_idx = 0;
         iteration_idx != opts.benchmark_iteration_ct; ++iteration_idx) {
      const auto decode_start = std::chrono::steady_clock::now();
      if (!cpu_decoder.Decode(block, params.sample_ct, codec_params,
                              decoded.data(), decoded.size(), &error)) {
        fprintf(stderr, "\nError: %s\n", error.c_str());
        goto cleanup;
      }
      pgr_decode_seconds +=
          std::chrono::duration<double>(
              std::chrono::steady_clock::now() - decode_start)
              .count();
      checksum ^=
          decoded[(static_cast<size_t>(iteration_idx) +
                   block.first_variant()) %
                  decoded.size()];
    }
    const auto pgen_start = std::chrono::steady_clock::now();
    for (uint32_t variant_offset = 0;
         variant_offset != block.variant_ct(); ++variant_offset) {
      const uint32_t variant_idx =
          block.first_variant() + variant_offset;
      uintptr_t* expected =
          pgen_block_genovecs +
          static_cast<size_t>(variant_offset) * pgen_word_stride;
      if (!pgen.Read(variant_idx, expected, &error)) {
        fprintf(stderr, "\nError: %s\n", error.c_str());
        goto cleanup;
      }
    }
    pgen_seconds +=
        std::chrono::duration<double>(
            std::chrono::steady_clock::now() - pgen_start)
            .count();
    for (uint32_t variant_offset = 0;
         variant_offset != block.variant_ct(); ++variant_offset) {
      const uint32_t variant_idx =
          block.first_variant() + variant_offset;
      const uintptr_t* expected =
          pgen_block_genovecs +
          static_cast<size_t>(variant_offset) * pgen_word_stride;
      const uint64_t* observed =
          decoded.data() +
          static_cast<size_t>(variant_offset) * packed_word_ct;
      if (memcmp(expected, observed,
                 static_cast<size_t>(packed_word_ct) * sizeof(uint64_t))) {
        fprintf(stderr,
                "\nError: CPU benchmark genotype mismatch at variant %u.\n",
                variant_idx);
        goto cleanup;
      }
      pgen_payload_bytes += pgen.record_byte_ct(variant_idx);
    }
    pgr_block_bytes += reader.block_index()[block_idx].byte_ct;
    benchmark_variant_ct += block.variant_ct();
    ++completed_block_ct;
    fprintf(stderr, "\rBenchmarked %u/%zu blocks.", completed_block_ct,
            selected_blocks.size());
    fflush(stderr);
  }
  {
    const long double calls =
        static_cast<long double>(benchmark_variant_ct) * params.sample_ct;
    const double average_decode_seconds =
        pgr_decode_seconds / opts.benchmark_iteration_ct;
    const double pgr_serial_seconds =
        pgr_read_seconds + average_decode_seconds;
    printf("\nConditional-rANS CPU benchmark\n");
    printf("  samples:                 %u\n", params.sample_ct);
    printf("  variants:                %llu\n",
           static_cast<unsigned long long>(benchmark_variant_ct));
    printf("  blocks:                  %zu / %u\n", selected_blocks.size(),
           params.block_ct);
    printf("  decoder threads:         %u\n", cpu_decoder.thread_ct());
    printf("  decode iterations:       %u\n",
           opts.benchmark_iteration_ct);
    printf("  checksum:                %016llx\n",
           static_cast<unsigned long long>(checksum));
    printf("\nCurrent PGEN\n");
    printf("  payload bytes:           %llu\n",
           static_cast<unsigned long long>(pgen_payload_bytes));
    printf("  read + decode seconds:   %.6f\n", pgen_seconds);
    printf("  billion calls/second:    %.3Lf\n",
           (pgen_seconds > 0.0)
               ? calls / pgen_seconds / 1.0e9L
               : 0.0L);
    printf("\nConditional rANS\n");
    printf("  block bytes read:        %llu\n",
           static_cast<unsigned long long>(pgr_block_bytes));
    printf("  block read seconds:      %.6f\n", pgr_read_seconds);
    printf("  decode seconds/pass:     %.6f\n", average_decode_seconds);
    printf("  billion calls/second:    %.3Lf\n",
           (average_decode_seconds > 0.0)
               ? calls / average_decode_seconds / 1.0e9L
               : 0.0L);
    printf("  serial read + decode:    %.6f\n", pgr_serial_seconds);
    printf("  PGEN/PGR serial speedup: %.3f\n",
           (pgr_serial_seconds > 0.0)
               ? pgen_seconds / pgr_serial_seconds
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
      record_bytes += record.size();
      if (metadata.mode == RecordMode::kMarginal) {
        ++marginal_ct;
      } else if (metadata.mode == RecordMode::kOneReference) {
        ++one_reference_ct;
      } else {
        ++two_reference_ct;
      }
    }
  }
  printf("Conditional-rANS container\n");
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
