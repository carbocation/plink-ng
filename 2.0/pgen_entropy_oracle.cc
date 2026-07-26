// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
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
#include "pgen_rans_size.h"

namespace {

using namespace plink2;

struct Options {
  std::string pgen_fname;
  std::string pvar_fname;
  std::string variant_out_fname;
  uint32_t block_variant_ct = 128;
  uint32_t anchor_ct = 8;
  uint32_t previous_window = 8;
  uint64_t max_anchor_bp = 1000000;
  uint32_t restart_variant_ct = 64;
  uint32_t rans_state_ct = 32;
  uint32_t rans_scale_bits = 12;
  uint32_t thread_ct = 0;
  uint32_t variant_limit = UINT32_MAX;
  uint32_t sample_block_ct = 0;
  uint32_t two_ref_shortlist = 0;
  uint32_t project_sample_ct = 0;
};

struct VariantMeta {
  uint32_t chrom_code = 0;
  uint64_t bp = 0;
  bool multiallelic = false;
};

struct VariantBlock {
  uint32_t start = 0;
  uint32_t len = 0;
};

struct PvarData {
  std::vector<VariantMeta> variants;
  std::vector<std::string> chrom_names;
  uint32_t multiallelic_ct = 0;
};

struct RansEstimate {
  double shannon_bits = 0.0;
  double quantized_bits = 0.0;
  uint64_t model_bytes = 0;
  uint64_t record_bytes = 0;
};

struct VariantResult {
  std::array<uint32_t, 4> counts = {};
  double maf = 0.0;
  uint32_t pgen_bytes = 0;
  RansEstimate marginal;
  RansEstimate previous_best;
  RansEstimate scheduled_best;
  RansEstimate scheduled_two_ref_best;
  RansEstimate projected_previous_best;
  RansEstimate projected_scheduled_best;
  RansEstimate projected_scheduled_two_ref_best;
  int32_t previous_ref = -1;
  int32_t scheduled_ref = -1;
  int32_t scheduled_two_ref1 = -1;
  int32_t scheduled_two_ref2 = -1;
  int32_t projected_previous_ref = -1;
  int32_t projected_scheduled_ref = -1;
  int32_t projected_scheduled_two_ref1 = -1;
  int32_t projected_scheduled_two_ref2 = -1;
  bool scheduled_anchor = false;
};

struct AnchorCandidate {
  uint64_t record_bytes = 0;
  uint32_t offset = 0;
};

struct Totals {
  uint64_t variant_ct = 0;
  uint64_t genotype_call_ct = 0;
  uint64_t pgen_payload_bytes = 0;
  long double marginal_shannon_bits = 0.0;
  long double previous_shannon_bits = 0.0;
  long double scheduled_shannon_bits = 0.0;
  uint64_t marginal_bytes = 0;
  uint64_t previous_bytes = 0;
  uint64_t scheduled_bytes = 0;
  uint64_t scheduled_two_ref_bytes = 0;
  uint64_t projected_marginal_bytes = 0;
  uint64_t projected_previous_bytes = 0;
  uint64_t projected_scheduled_bytes = 0;
  uint64_t projected_scheduled_two_ref_bytes = 0;
  uint64_t marginal_model_bytes = 0;
  uint64_t previous_model_bytes = 0;
  uint64_t scheduled_model_bytes = 0;
  uint64_t scheduled_two_ref_model_bytes = 0;
  uint64_t previous_conditional_ct = 0;
  uint64_t scheduled_conditional_ct = 0;
  uint64_t scheduled_two_ref_ct = 0;
  uint64_t projected_previous_conditional_ct = 0;
  uint64_t projected_scheduled_conditional_ct = 0;
  uint64_t projected_scheduled_two_ref_ct = 0;
  uint64_t scheduled_anchor_ct = 0;
  uint64_t nonref_ct = 0;
  uint64_t block_ct = 0;
  long double scheduled_two_ref_shannon_bits = 0.0;
};

void PrintUsage(FILE* out) {
  fputs(
      "Usage:\n"
      "  pgen_entropy_oracle <input.pgen> [options]\n"
      "\n"
      "Options:\n"
      "  --pvar <file>              Plain-text PVAR for chromosome/position bounds.\n"
      "  --out-variants <file>      Write per-variant estimates as TSV.\n"
      "  --block-variants <n>       Independent block size (default 128).\n"
      "  --anchors <n>              Evenly spaced independent anchors/block (default 8).\n"
      "  --previous-window <n>      Chained lower-bound lookback (default 8).\n"
      "  --max-anchor-bp <n>        Maximum anchor distance with --pvar (default 1000000).\n"
      "  --restart-variants <n>     Cumulative-offset restart interval (default 64).\n"
      "  --rans-states <n>          Independent GPU rANS states/record (default 32).\n"
      "  --rans-scale-bits <n>      rANS frequency precision, 8..16 (default 12).\n"
      "  --threads <n>              Pair-count worker threads (default hardware count).\n"
      "  --variant-limit <n>        Analyze only the first n variants.\n"
      "  --sample-blocks <n>        Analyze n blocks stratified across the PGEN.\n"
      "  --two-ref-shortlist <n>    Test all pairs among the n best single anchors.\n"
      "  --project-samples <n>      Project rANS records to a larger sample count.\n"
      "  --help                     Show this message.\n"
      "\n"
      "The scheduled estimate uses independently encoded, evenly spaced anchors and\n"
      "has dependency depth one.  The previous-window estimate permits chaining and\n"
      "is a lower bound, not an immediately implementable random-access schedule.\n",
      out);
}

bool ParseU32(const char* text, uint32_t min_value, uint32_t max_value,
              uint32_t* value_ptr) {
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
  *value_ptr = static_cast<uint32_t>(parsed);
  return true;
}

bool ParseU64(const char* text, uint64_t min_value, uint64_t max_value,
              uint64_t* value_ptr) {
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
  *value_ptr = static_cast<uint64_t>(parsed);
  return true;
}

bool ParseOptions(int argc, char** argv, Options* opts_ptr) {
  Options& opts = *opts_ptr;
  if (argc < 2) {
    PrintUsage(stderr);
    return false;
  }
  if (!strcmp(argv[1], "--help")) {
    PrintUsage(stdout);
    exit(0);
  }
  opts.pgen_fname = argv[1];
  for (int arg_idx = 2; arg_idx < argc; ++arg_idx) {
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
      opts.pvar_fname = value;
    } else if (!strcmp(arg, "--out-variants")) {
      opts.variant_out_fname = value;
    } else if (!strcmp(arg, "--block-variants")) {
      if (!ParseU32(value, 2, 65536, &opts.block_variant_ct)) {
        fprintf(stderr, "Error: Invalid --block-variants value '%s'.\n", value);
        return false;
      }
    } else if (!strcmp(arg, "--anchors")) {
      if (!ParseU32(value, 1, 1024, &opts.anchor_ct)) {
        fprintf(stderr, "Error: Invalid --anchors value '%s'.\n", value);
        return false;
      }
    } else if (!strcmp(arg, "--previous-window")) {
      if (!ParseU32(value, 1, 1024, &opts.previous_window)) {
        fprintf(stderr, "Error: Invalid --previous-window value '%s'.\n", value);
        return false;
      }
    } else if (!strcmp(arg, "--max-anchor-bp")) {
      if (!ParseU64(value, 0, UINT64_MAX, &opts.max_anchor_bp)) {
        fprintf(stderr, "Error: Invalid --max-anchor-bp value '%s'.\n", value);
        return false;
      }
    } else if (!strcmp(arg, "--restart-variants")) {
      if (!ParseU32(value, 1, 65536, &opts.restart_variant_ct)) {
        fprintf(stderr, "Error: Invalid --restart-variants value '%s'.\n", value);
        return false;
      }
    } else if (!strcmp(arg, "--rans-states")) {
      if (!ParseU32(value, 1, 256, &opts.rans_state_ct)) {
        fprintf(stderr, "Error: Invalid --rans-states value '%s'.\n", value);
        return false;
      }
    } else if (!strcmp(arg, "--rans-scale-bits")) {
      if (!ParseU32(value, 8, 16, &opts.rans_scale_bits)) {
        fprintf(stderr, "Error: Invalid --rans-scale-bits value '%s'.\n", value);
        return false;
      }
    } else if (!strcmp(arg, "--threads")) {
      if (!ParseU32(value, 1, 1024, &opts.thread_ct)) {
        fprintf(stderr, "Error: Invalid --threads value '%s'.\n", value);
        return false;
      }
    } else if (!strcmp(arg, "--variant-limit")) {
      if (!ParseU32(value, 1, UINT32_MAX, &opts.variant_limit)) {
        fprintf(stderr, "Error: Invalid --variant-limit value '%s'.\n", value);
        return false;
      }
    } else if (!strcmp(arg, "--sample-blocks")) {
      if (!ParseU32(value, 1, UINT32_MAX, &opts.sample_block_ct)) {
        fprintf(stderr, "Error: Invalid --sample-blocks value '%s'.\n",
                value);
        return false;
      }
    } else if (!strcmp(arg, "--two-ref-shortlist")) {
      if (!ParseU32(value, 2, 1024, &opts.two_ref_shortlist)) {
        fprintf(stderr, "Error: Invalid --two-ref-shortlist value '%s'.\n",
                value);
        return false;
      }
    } else if (!strcmp(arg, "--project-samples")) {
      if (!ParseU32(value, 1, UINT32_MAX, &opts.project_sample_ct)) {
        fprintf(stderr, "Error: Invalid --project-samples value '%s'.\n",
                value);
        return false;
      }
    } else {
      fprintf(stderr, "Error: Unrecognized option '%s'.\n", arg);
      return false;
    }
  }
  if (opts.anchor_ct > opts.block_variant_ct) {
    opts.anchor_ct = opts.block_variant_ct;
  }
  if (opts.two_ref_shortlist > opts.anchor_ct) {
    opts.two_ref_shortlist = opts.anchor_ct;
  }
  if (opts.sample_block_ct && (opts.variant_limit != UINT32_MAX)) {
    fputs("Error: --sample-blocks and --variant-limit cannot be combined.\n",
          stderr);
    return false;
  }
  if (!opts.thread_ct) {
    opts.thread_ct = std::max(1U, std::thread::hardware_concurrency());
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
              PvarData* pvar_ptr) {
  if ((fname.size() >= 4) && (fname.substr(fname.size() - 4) == ".zst")) {
    fprintf(stderr,
            "Error: This prototype only accepts plain-text PVAR files; "
            "decompress '%s' first.\n",
            fname.c_str());
    return false;
  }
  std::ifstream infile(fname);
  if (!infile) {
    fprintf(stderr, "Error: Could not open PVAR '%s'.\n", fname.c_str());
    return false;
  }
  PvarData& pvar = *pvar_ptr;
  pvar.variants.reserve(expected_variant_ct);
  std::unordered_map<std::string, uint32_t> chrom_codes;
  int32_t chrom_col = 0;
  int32_t pos_col = 1;
  int32_t alt_col = 4;
  bool header_seen = false;
  std::string line;
  while (std::getline(infile, line)) {
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
        for (uint32_t idx = 0; idx != fields.size(); ++idx) {
          std::string name = fields[idx];
          if ((!name.empty()) && (name[0] == '#')) {
            name.erase(0, 1);
          }
          if ((name == "CHROM") || (name == "chrom")) {
            chrom_col = static_cast<int32_t>(idx);
          } else if ((name == "POS") || (name == "pos")) {
            pos_col = static_cast<int32_t>(idx);
          } else if ((name == "ALT") || (name == "alt")) {
            alt_col = static_cast<int32_t>(idx);
          }
        }
        header_seen = true;
      }
      continue;
    }
    const std::vector<std::string> fields = SplitWhitespace(line);
    const int32_t required_col = std::max(chrom_col, std::max(pos_col, alt_col));
    if ((chrom_col < 0) || (pos_col < 0) ||
        (static_cast<int32_t>(fields.size()) <= required_col)) {
      fprintf(stderr, "Error: Malformed PVAR data line %zu in '%s'.\n",
              pvar.variants.size() + 1, fname.c_str());
      return false;
    }
    uint64_t bp;
    if (!ParseU64(fields[pos_col].c_str(), 0, UINT64_MAX, &bp)) {
      fprintf(stderr, "Error: Invalid PVAR position '%s'.\n",
              fields[pos_col].c_str());
      return false;
    }
    const std::string& chrom = fields[chrom_col];
    auto code_iter = chrom_codes.find(chrom);
    uint32_t chrom_code;
    if (code_iter == chrom_codes.end()) {
      chrom_code = static_cast<uint32_t>(pvar.chrom_names.size());
      chrom_codes.emplace(chrom, chrom_code);
      pvar.chrom_names.push_back(chrom);
    } else {
      chrom_code = code_iter->second;
    }
    VariantMeta meta;
    meta.chrom_code = chrom_code;
    meta.bp = bp;
    meta.multiallelic =
        (alt_col >= 0) && (fields[alt_col].find(',') != std::string::npos);
    pvar.multiallelic_ct += meta.multiallelic;
    pvar.variants.push_back(meta);
    if (pvar.variants.size() > expected_variant_ct) {
      fprintf(stderr,
              "Error: PVAR contains more variants than the PGEN header "
              "(%u).\n",
              expected_variant_ct);
      return false;
    }
  }
  if ((!header_seen) && pvar.variants.empty()) {
    fprintf(stderr, "Error: No variants found in PVAR '%s'.\n", fname.c_str());
    return false;
  }
  if (pvar.variants.size() != expected_variant_ct) {
    fprintf(stderr,
            "Error: PVAR/PGEN variant-count mismatch (%zu vs. %u).\n",
            pvar.variants.size(), expected_variant_ct);
    return false;
  }
  return true;
}

std::vector<VariantBlock> BuildVariantBlocks(
    uint32_t variant_ct, uint32_t block_variant_ct,
    const std::vector<VariantMeta>* meta_ptr) {
  std::vector<VariantBlock> blocks;
  blocks.reserve((variant_ct + block_variant_ct - 1) / block_variant_ct);
  uint32_t block_start = 0;
  while (block_start != variant_ct) {
    uint32_t block_len =
        std::min(block_variant_ct, variant_ct - block_start);
    if (meta_ptr) {
      const uint32_t chrom_code = (*meta_ptr)[block_start].chrom_code;
      for (uint32_t offset = 1; offset != block_len; ++offset) {
        if ((*meta_ptr)[block_start + offset].chrom_code != chrom_code) {
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

std::vector<VariantBlock> SelectStratifiedBlocks(
    const std::vector<VariantBlock>& blocks, uint32_t requested_block_ct) {
  if ((!requested_block_ct) || (requested_block_ct >= blocks.size())) {
    return blocks;
  }
  std::vector<VariantBlock> selected;
  selected.reserve(requested_block_ct);
  const uint64_t block_ct = blocks.size();
  for (uint32_t sample_idx = 0; sample_idx != requested_block_ct;
       ++sample_idx) {
    const uint64_t block_idx =
        ((2ULL * sample_idx + 1) * block_ct) /
        (2ULL * requested_block_ct);
    selected.push_back(blocks[block_idx]);
  }
  return selected;
}

uint64_t FileSize(const std::string& fname) {
  struct stat stat_buf;
  if (stat(fname.c_str(), &stat_buf)) {
    return 0;
  }
  return static_cast<uint64_t>(stat_buf.st_size);
}

double ShannonBits(const uint32_t* counts, uint32_t count_ct) {
  uint64_t total = 0;
  for (uint32_t idx = 0; idx != count_ct; ++idx) {
    total += counts[idx];
  }
  if (!total) {
    return 0.0;
  }
  const double total_d = static_cast<double>(total);
  double result = total_d * log2(total_d);
  for (uint32_t idx = 0; idx != count_ct; ++idx) {
    if (counts[idx]) {
      result -= static_cast<double>(counts[idx]) *
                log2(static_cast<double>(counts[idx]));
    }
  }
  return result;
}

double QuantizedRowBits(const uint32_t* counts, uint32_t scale_bits,
                        uint32_t* active_ct_ptr) {
  uint64_t total = 0;
  uint32_t active_ct = 0;
  for (uint32_t idx = 0; idx != 4; ++idx) {
    total += counts[idx];
    active_ct += (counts[idx] != 0);
  }
  *active_ct_ptr = active_ct;
  if (active_ct <= 1) {
    return 0.0;
  }
  const uint32_t total_freq = 1U << scale_bits;
  double raw_freq[4] = {};
  uint32_t norm_freq[4] = {};
  uint32_t norm_sum = 0;
  for (uint32_t idx = 0; idx != 4; ++idx) {
    if (!counts[idx]) {
      continue;
    }
    raw_freq[idx] =
        static_cast<double>(counts[idx]) * total_freq / static_cast<double>(total);
    norm_freq[idx] =
        std::max(1U, static_cast<uint32_t>(floor(raw_freq[idx])));
    norm_sum += norm_freq[idx];
  }
  while (norm_sum < total_freq) {
    uint32_t best_idx = 0;
    double best_score = -std::numeric_limits<double>::infinity();
    for (uint32_t idx = 0; idx != 4; ++idx) {
      if (!counts[idx]) {
        continue;
      }
      const double score = raw_freq[idx] - norm_freq[idx];
      if (score > best_score) {
        best_score = score;
        best_idx = idx;
      }
    }
    ++norm_freq[best_idx];
    ++norm_sum;
  }
  while (norm_sum > total_freq) {
    uint32_t best_idx = 0;
    double best_score = -std::numeric_limits<double>::infinity();
    for (uint32_t idx = 0; idx != 4; ++idx) {
      if (norm_freq[idx] <= 1) {
        continue;
      }
      const double score = norm_freq[idx] - raw_freq[idx];
      if (score > best_score) {
        best_score = score;
        best_idx = idx;
      }
    }
    --norm_freq[best_idx];
    --norm_sum;
  }
  double result = 0.0;
  for (uint32_t idx = 0; idx != 4; ++idx) {
    if (counts[idx]) {
      result += static_cast<double>(counts[idx]) *
                log2(static_cast<double>(total_freq) / norm_freq[idx]);
    }
  }
  return result;
}

RansEstimate EstimateRans(const uint32_t* counts, uint32_t row_ct,
                          uint32_t reference_ct, const Options& opts) {
  RansEstimate result;
  result.model_bytes = (reference_ct == 2) ? 2 : reference_ct;
  for (uint32_t row_idx = 0; row_idx != row_ct; ++row_idx) {
    const uint32_t* row = &(counts[row_idx * 4]);
    uint32_t row_total = 0;
    for (uint32_t col_idx = 0; col_idx != 4; ++col_idx) {
      row_total += row[col_idx];
    }
    if (!row_total) {
      continue;
    }
    result.shannon_bits += ShannonBits(row, 4);
    uint32_t active_ct;
    result.quantized_bits +=
        QuantizedRowBits(row, opts.rans_scale_bits, &active_ct);
    ++result.model_bytes;
    result.model_bytes += 2 * (active_ct - 1);
  }
  pgen_rans::RecordMode mode = pgen_rans::RecordMode::kMarginal;
  if (reference_ct == 1) {
    mode = pgen_rans::RecordMode::kOneReference;
  } else if (reference_ct == 2) {
    mode = pgen_rans::RecordMode::kTwoReference;
  }
  std::string error;
  if (!pgen_rans::EstimateRecordBytes(
          counts, mode,
          pgen_rans::CodecParams(
              opts.rans_state_ct, opts.rans_scale_bits),
          &result.record_bytes, &error)) {
    fprintf(stderr, "Error: Production rANS size estimator failed: %s\n",
            error.c_str());
    exit(1);
  }
  return result;
}

uint64_t ProjectRansRecordBytes(const RansEstimate& estimate,
                                const Options& opts,
                                uint32_t source_sample_ct,
                                uint32_t projected_sample_ct) {
  const uint64_t source_payload_bytes =
      static_cast<uint64_t>(ceil(estimate.quantized_bits / 8.0));
  uint64_t source_state_bytes = 0;
  uint64_t projected_state_bytes = 0;
  if (estimate.quantized_bits > 0.0) {
    const uint32_t source_state_ct =
        std::min(opts.rans_state_ct, source_sample_ct);
    const uint32_t projected_state_ct =
        std::min(opts.rans_state_ct, projected_sample_ct);
    source_state_bytes = 4LLU * source_state_ct;
    projected_state_bytes = 4LLU * projected_state_ct;
  }
  const uint64_t fixed_bytes =
      estimate.record_bytes - source_payload_bytes - source_state_bytes;
  const long double projected_bits =
      static_cast<long double>(estimate.quantized_bits) *
      projected_sample_ct / source_sample_ct;
  return fixed_bytes + projected_state_bytes +
         static_cast<uint64_t>(ceill(projected_bits / 8.0L));
}

void CountJointGenotypes(const uintptr_t* anchor,
                         const uintptr_t* target, uint32_t sample_ct,
                         const uint32_t* anchor_counts,
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
    for (uint32_t anchor_geno = 0; anchor_geno != 3; ++anchor_geno) {
      for (uint32_t target_geno = 0; target_geno != 3; ++target_geno) {
        joint_counts[4 * anchor_geno + target_geno] +=
            PopcountWord(anchor_masks[anchor_geno] &
                         target_masks[target_geno]);
      }
    }
  }
  for (uint32_t anchor_geno = 0; anchor_geno != 3; ++anchor_geno) {
    uint32_t known = 0;
    for (uint32_t target_geno = 0; target_geno != 3; ++target_geno) {
      known += joint_counts[4 * anchor_geno + target_geno];
    }
    joint_counts[4 * anchor_geno + 3] =
        anchor_counts[anchor_geno] - known;
  }
  for (uint32_t target_geno = 0; target_geno != 3; ++target_geno) {
    uint32_t known = 0;
    for (uint32_t anchor_geno = 0; anchor_geno != 3; ++anchor_geno) {
      known += joint_counts[4 * anchor_geno + target_geno];
    }
    joint_counts[12 + target_geno] = target_counts[target_geno] - known;
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
    for (uint32_t geno1 = 0; geno1 != 4; ++geno1) {
      for (uint32_t geno2 = 0; geno2 != 4; ++geno2) {
        const uintptr_t anchor_intersection =
            masks[0][geno1] & masks[1][geno2];
        for (uint32_t target_geno = 0; target_geno != 4; ++target_geno) {
          triple_counts[16 * geno1 + 4 * geno2 + target_geno] +=
              PopcountWord(anchor_intersection & masks[2][target_geno]);
        }
      }
    }
  }
}

bool AnchorEligible(uint32_t target_vidx, uint32_t anchor_vidx,
                    const std::vector<VariantMeta>* meta_ptr,
                    uint64_t max_anchor_bp) {
  if (!meta_ptr) {
    return true;
  }
  const VariantMeta& target = (*meta_ptr)[target_vidx];
  const VariantMeta& anchor = (*meta_ptr)[anchor_vidx];
  if (target.chrom_code != anchor.chrom_code) {
    return false;
  }
  if (!max_anchor_bp) {
    return true;
  }
  const uint64_t distance =
      (target.bp >= anchor.bp) ? (target.bp - anchor.bp)
                               : (anchor.bp - target.bp);
  return distance <= max_anchor_bp;
}

std::vector<uint32_t> ScheduledAnchors(uint32_t block_start,
                                       uint32_t block_len,
                                       uint32_t requested_anchor_ct) {
  const uint32_t anchor_ct = std::min(block_len, requested_anchor_ct);
  std::vector<uint32_t> result;
  result.reserve(anchor_ct);
  for (uint32_t anchor_idx = 0; anchor_idx != anchor_ct; ++anchor_idx) {
    const uint64_t numerator =
        static_cast<uint64_t>(2 * anchor_idx + 1) * block_len;
    const uint32_t block_offset =
        static_cast<uint32_t>(numerator / (2 * anchor_ct));
    result.push_back(block_start + block_offset);
  }
  return result;
}

double ComputeMaf(const uint32_t* counts) {
  const uint64_t nonmissing_ct =
      static_cast<uint64_t>(counts[0]) + counts[1] + counts[2];
  if (!nonmissing_ct) {
    return 0.0;
  }
  const uint64_t alt_ct =
      static_cast<uint64_t>(counts[1]) + 2LLU * counts[2];
  const uint64_t allele_ct = 2 * nonmissing_ct;
  const uint64_t minor_ct = std::min(alt_ct, allele_ct - alt_ct);
  return static_cast<double>(minor_ct) / allele_ct;
}

uint64_t ContainerOverheadByteCt(
    const Options& opts, const std::vector<VariantBlock>& blocks,
    uint64_t metadata_byte_ct) {
  uint64_t result = pgen_rans::ContainerGlobalOverheadByteCt(
      blocks.size(), metadata_byte_ct);
  for (const VariantBlock& block : blocks) {
    result += pgen_rans::ContainerBlockOverheadByteCt(
        block.len, opts.restart_variant_ct);
  }
  return result;
}

std::string FormatGb(long double byte_ct) {
  char buf[64];
  snprintf(buf, sizeof(buf), "%.3Lf", byte_ct / 1000000000.0L);
  return std::string(buf);
}

double BitsPerCall(long double byte_ct, uint64_t call_ct) {
  if (!call_ct) {
    return 0.0;
  }
  return static_cast<double>(8.0L * byte_ct / call_ct);
}

void PrintEstimateRow(const char* label, long double bytes,
                      uint64_t call_ct, uint64_t pgen_payload_bytes) {
  const double ratio =
      (bytes != 0.0L) ? static_cast<double>(pgen_payload_bytes / bytes) : 0.0;
  printf("%-30s %12s %12.5f %12.3f\n", label,
         FormatGb(bytes).c_str(), BitsPerCall(bytes, call_ct), ratio);
}

}  // namespace

int main(int argc, char** argv) {
  using namespace plink2;
  Options opts;
  if (!ParseOptions(argc, argv, &opts)) {
    return 2;
  }

  PglErr reterr = kPglRetSuccess;
  PgenFileInfo pgfi;
  PgenReader pgr;
  PreinitPgfi(&pgfi);
  PreinitPgr(&pgr);
  unsigned char* pgfi_alloc = nullptr;
  unsigned char* pgr_alloc = nullptr;
  uintptr_t* block_genovecs = nullptr;
  FILE* variant_out = nullptr;
  int return_code = 1;
  uint32_t variant_ct = 0;
  uint32_t sample_ct = 0;
  uint32_t genovec_word_stride = 0;
  PvarData pvar;
  const std::vector<VariantMeta>* meta_ptr = nullptr;
  Totals totals;
  PgrSampleSubsetIndex pssi;
  std::chrono::steady_clock::time_point start_time;
  uint32_t block_start = 0;
  uint32_t source_variant_ct = 0;
  uint32_t processed_variant_ct = 0;
  uint64_t metadata_byte_ct = 0;
  uint64_t container_overhead_bytes = 0;
  std::vector<VariantBlock> all_blocks;
  std::vector<VariantBlock> analysis_blocks;

  char errstr_buf[kPglErrstrBufBlen];
  PgenHeaderCtrl header_ctrl;
  uintptr_t pgfi_alloc_cacheline_ct;
  reterr = PgfiInitPhase1(
      opts.pgen_fname.c_str(), nullptr, UINT32_MAX, UINT32_MAX, &header_ctrl,
      &pgfi, &pgfi_alloc_cacheline_ct, errstr_buf);
  if (reterr) {
    fputs(errstr_buf, stderr);
    goto cleanup;
  }
  if (cachealigned_malloc(pgfi_alloc_cacheline_ct * kCacheline,
                          &pgfi_alloc)) {
    fputs("Error: Out of memory allocating the PGEN index.\n", stderr);
    goto cleanup;
  }
  {
    uint32_t max_vrec_width;
    uintptr_t pgr_alloc_cacheline_ct;
    reterr = PgfiInitPhase2(
        header_ctrl, 0, 0, 0, 0, pgfi.raw_variant_ct, &max_vrec_width, &pgfi,
        pgfi_alloc, &pgr_alloc_cacheline_ct, errstr_buf);
    if (reterr) {
      fputs(errstr_buf, stderr);
      goto cleanup;
    }
    if (cachealigned_malloc(pgr_alloc_cacheline_ct * kCacheline, &pgr_alloc)) {
      fputs("Error: Out of memory allocating the PGEN reader.\n", stderr);
      goto cleanup;
    }
    reterr = PgrInit(opts.pgen_fname.c_str(), max_vrec_width, &pgfi, &pgr,
                     pgr_alloc);
    if (reterr) {
      fprintf(stderr, "Error: PgrInit failed with code %u.\n",
              static_cast<uint32_t>(reterr));
      goto cleanup;
    }
  }

  if (pgfi.gflags & (kfPgenGlobalHardcallPhasePresent |
                     kfPgenGlobalDosagePresent |
                     kfPgenGlobalDosagePhasePresent)) {
    fputs(
        "Error: This first oracle prototype only supports unphased hardcall "
        "PGENs without dosage tracks.\n",
        stderr);
    goto cleanup;
  }
  if (pgfi.max_allele_ct > 2) {
    fputs(
        "Error: This first oracle prototype only supports biallelic PGENs; "
        "split or filter multiallelic variants first.\n",
        stderr);
    goto cleanup;
  }

  source_variant_ct = std::min(pgfi.raw_variant_ct, opts.variant_limit);
  sample_ct = pgfi.raw_sample_ct;
  genovec_word_stride = NypCtToVecCt(sample_ct) * kWordsPerVec;
  if (cachealigned_malloc(
          static_cast<uintptr_t>(opts.block_variant_ct) *
              genovec_word_stride * sizeof(uintptr_t),
          &block_genovecs)) {
    fputs("Error: Out of memory allocating the genotype block.\n", stderr);
    goto cleanup;
  }

  if (!opts.pvar_fname.empty()) {
    if (!LoadPvar(opts.pvar_fname, pgfi.raw_variant_ct, &pvar)) {
      goto cleanup;
    }
    if (pvar.multiallelic_ct) {
      fprintf(
          stderr,
          "Error: PVAR contains %u multiallelic variant%s; split or filter "
          "these before running the four-symbol oracle.\n",
          pvar.multiallelic_ct, (pvar.multiallelic_ct == 1) ? "" : "s");
      goto cleanup;
    }
    meta_ptr = &pvar.variants;
  } else {
    fputs(
        "Warning: No PVAR supplied; chromosome boundaries and --max-anchor-bp "
        "will not be enforced.\n",
        stderr);
  }

  all_blocks =
      BuildVariantBlocks(source_variant_ct, opts.block_variant_ct, meta_ptr);
  analysis_blocks =
      SelectStratifiedBlocks(all_blocks, opts.sample_block_ct);
  for (const VariantBlock& block : analysis_blocks) {
    variant_ct += block.len;
  }

  if (!opts.variant_out_fname.empty()) {
    variant_out = fopen(opts.variant_out_fname.c_str(), "w");
    if (!variant_out) {
      fprintf(stderr, "Error: Could not open '%s' for writing.\n",
              opts.variant_out_fname.c_str());
      goto cleanup;
    }
    fputs(
        "#VIDX\tCHROM\tPOS\tMAF\tPGEN_BYTES\tMARGINAL_SHANNON_BYTES\t"
        "MARGINAL_RANS_BYTES\tPREVIOUS_REF\tPREVIOUS_SHANNON_BYTES\t"
        "PREVIOUS_RANS_BYTES\tSCHEDULED_REF\tSCHEDULED_ANCHOR\t"
        "SCHEDULED_SHANNON_BYTES\tSCHEDULED_RANS_BYTES\t"
        "PROJECTION_PREVIOUS_REF\tPROJECTION_PREVIOUS_SHANNON_BYTES\t"
        "PROJECTION_PREVIOUS_RANS_BYTES\tPROJECTION_SCHEDULED_REF\t"
        "PROJECTION_SCHEDULED_SHANNON_BYTES\t"
        "PROJECTION_SCHEDULED_RANS_BYTES\tTWO_REF_REF1\tTWO_REF_REF2\t"
        "TWO_REF_SHANNON_BYTES\tTWO_REF_RANS_BYTES\t"
        "PROJECTION_TWO_REF_REF1\tPROJECTION_TWO_REF_REF2\t"
        "PROJECTION_TWO_REF_SHANNON_BYTES\t"
        "PROJECTION_TWO_REF_RANS_BYTES\n",
        variant_out);
  }

  printf("PGEN entropy oracle\n");
  printf("  input:              %s\n", opts.pgen_fname.c_str());
  printf("  variants analyzed:  %u / %u\n", variant_ct,
         pgfi.raw_variant_ct);
  if (opts.sample_block_ct) {
    printf("  blocks sampled:     %zu / %zu (deterministic strata)\n",
           analysis_blocks.size(), all_blocks.size());
  }
  printf("  samples:            %u\n", sample_ct);
  printf("  block/anchors:      %u / %u\n", opts.block_variant_ct,
         opts.anchor_ct);
  printf("  previous window:    %u\n", opts.previous_window);
  if (opts.two_ref_shortlist) {
    printf("  two-ref shortlist:  %u best single anchors\n",
           opts.two_ref_shortlist);
  }
  printf("  rANS states/scale:  %u / %u bits\n", opts.rans_state_ct,
         opts.rans_scale_bits);
  printf("  worker threads:     %u\n\n", opts.thread_ct);

  PgrClearSampleSubsetIndex(&pgr, &pssi);
  start_time = std::chrono::steady_clock::now();
  for (const VariantBlock& block : analysis_blocks) {
    block_start = block.start;
    const uint32_t block_len = block.len;
    std::vector<VariantResult> results(block_len);
    for (uint32_t offset = 0; offset != block_len; ++offset) {
      uintptr_t* genovec =
          &(block_genovecs[static_cast<uintptr_t>(offset) *
                            genovec_word_stride]);
      const uint32_t vidx = block_start + offset;
      reterr = PgrGet(nullptr, pssi, sample_ct, vidx, &pgr, genovec);
      if (reterr) {
        fprintf(stderr, "\nError: PgrGet failed with code %u at variant %u.\n",
                static_cast<uint32_t>(reterr), vidx);
        goto cleanup;
      }
      ZeroTrailingNyps(sample_ct, genovec);
      GenoarrCountFreqsUnsafe(genovec, sample_ct, results[offset].counts);
      results[offset].maf = ComputeMaf(results[offset].counts.data());
      results[offset].pgen_bytes = GetPgfiVrecWidth(&pgfi, vidx);
      results[offset].marginal =
          EstimateRans(results[offset].counts.data(), 1, 0, opts);
      results[offset].previous_best = results[offset].marginal;
      results[offset].scheduled_best = results[offset].marginal;
      results[offset].scheduled_two_ref_best = results[offset].marginal;
      results[offset].projected_previous_best = results[offset].marginal;
      results[offset].projected_scheduled_best = results[offset].marginal;
      results[offset].projected_scheduled_two_ref_best =
          results[offset].marginal;
    }

    const std::vector<uint32_t> scheduled_anchors =
        ScheduledAnchors(block_start, block_len, opts.anchor_ct);
    std::vector<uint8_t> is_scheduled_anchor(block_len, 0);
    for (const uint32_t anchor_vidx : scheduled_anchors) {
      is_scheduled_anchor[anchor_vidx - block_start] = 1;
      results[anchor_vidx - block_start].scheduled_anchor = true;
    }
    std::atomic<uint32_t> next_offset(0);
    const uint32_t worker_ct =
        std::min(opts.thread_ct, std::max(1U, block_len));
    std::vector<std::thread> workers;
    workers.reserve(worker_ct);
    for (uint32_t worker_idx = 0; worker_idx != worker_ct; ++worker_idx) {
      workers.emplace_back([&]() {
        uint32_t offset;
        while ((offset = next_offset.fetch_add(1, std::memory_order_relaxed)) <
               block_len) {
          VariantResult& target_result = results[offset];
          const uint32_t target_vidx = block_start + offset;
          const uintptr_t* target =
              &(block_genovecs[static_cast<uintptr_t>(offset) *
                                genovec_word_stride]);
          uint64_t projected_previous_bytes = 0;
          uint64_t projected_scheduled_bytes = 0;
          uint64_t projected_scheduled_two_ref_bytes = 0;
          if (opts.project_sample_ct) {
            projected_previous_bytes = ProjectRansRecordBytes(
                target_result.projected_previous_best, opts, sample_ct,
                opts.project_sample_ct);
            projected_scheduled_bytes = ProjectRansRecordBytes(
                target_result.projected_scheduled_best, opts, sample_ct,
                opts.project_sample_ct);
            projected_scheduled_two_ref_bytes = projected_scheduled_bytes;
          }
          const uint32_t previous_start =
              (offset > opts.previous_window)
                  ? (offset - opts.previous_window)
                  : 0;
          for (uint32_t anchor_offset = previous_start;
               anchor_offset != offset; ++anchor_offset) {
            const uint32_t anchor_vidx = block_start + anchor_offset;
            if (!AnchorEligible(target_vidx, anchor_vidx, meta_ptr,
                                opts.max_anchor_bp)) {
              continue;
            }
            uint32_t joint_counts[16];
            const uintptr_t* anchor =
                &(block_genovecs[static_cast<uintptr_t>(anchor_offset) *
                                  genovec_word_stride]);
            CountJointGenotypes(anchor, target, sample_ct,
                                results[anchor_offset].counts.data(),
                                target_result.counts.data(), joint_counts);
            const RansEstimate estimate =
                EstimateRans(joint_counts, 4, 1, opts);
            if (estimate.record_bytes <
                target_result.previous_best.record_bytes) {
              target_result.previous_best = estimate;
              target_result.previous_ref =
                  static_cast<int32_t>(anchor_vidx);
            }
            if (opts.project_sample_ct) {
              const uint64_t candidate_projected_bytes =
                  ProjectRansRecordBytes(estimate, opts, sample_ct,
                                         opts.project_sample_ct);
              if (candidate_projected_bytes < projected_previous_bytes) {
                projected_previous_bytes = candidate_projected_bytes;
                target_result.projected_previous_best = estimate;
                target_result.projected_previous_ref =
                    static_cast<int32_t>(anchor_vidx);
              }
            }
          }
          std::vector<AnchorCandidate> two_ref_candidates;
          if (opts.two_ref_shortlist && (!is_scheduled_anchor[offset])) {
            two_ref_candidates.reserve(scheduled_anchors.size());
          }
          if (!is_scheduled_anchor[offset]) {
            for (const uint32_t anchor_vidx : scheduled_anchors) {
              if ((anchor_vidx == target_vidx) ||
                  (!AnchorEligible(target_vidx, anchor_vidx, meta_ptr,
                                   opts.max_anchor_bp))) {
                continue;
              }
              const uint32_t anchor_offset = anchor_vidx - block_start;
              uint32_t joint_counts[16];
              const uintptr_t* anchor =
                  &(block_genovecs[static_cast<uintptr_t>(anchor_offset) *
                                    genovec_word_stride]);
              CountJointGenotypes(anchor, target, sample_ct,
                                  results[anchor_offset].counts.data(),
                                  target_result.counts.data(), joint_counts);
              const RansEstimate estimate =
                  EstimateRans(joint_counts, 4, 1, opts);
              if (opts.two_ref_shortlist) {
                two_ref_candidates.push_back(
                    {estimate.record_bytes, anchor_offset});
              }
              if (estimate.record_bytes <
                  target_result.scheduled_best.record_bytes) {
                target_result.scheduled_best = estimate;
                target_result.scheduled_ref =
                    static_cast<int32_t>(anchor_vidx);
              }
              if (opts.project_sample_ct) {
                const uint64_t candidate_projected_bytes =
                    ProjectRansRecordBytes(estimate, opts, sample_ct,
                                           opts.project_sample_ct);
                if (candidate_projected_bytes < projected_scheduled_bytes) {
                  projected_scheduled_bytes = candidate_projected_bytes;
                  target_result.projected_scheduled_best = estimate;
                  target_result.projected_scheduled_ref =
                      static_cast<int32_t>(anchor_vidx);
                }
              }
            }
          }
          target_result.scheduled_two_ref_best =
              target_result.scheduled_best;
          target_result.projected_scheduled_two_ref_best =
              target_result.projected_scheduled_best;
          if (opts.project_sample_ct) {
            projected_scheduled_two_ref_bytes = projected_scheduled_bytes;
          }
          if (two_ref_candidates.size() >= 2) {
            std::sort(
                two_ref_candidates.begin(), two_ref_candidates.end(),
                [](const AnchorCandidate& lhs, const AnchorCandidate& rhs) {
                  if (lhs.record_bytes != rhs.record_bytes) {
                    return lhs.record_bytes < rhs.record_bytes;
                  }
                  return lhs.offset < rhs.offset;
                });
            if (two_ref_candidates.size() > opts.two_ref_shortlist) {
              two_ref_candidates.resize(opts.two_ref_shortlist);
            }
            for (uint32_t first_idx = 0;
                 first_idx + 1 != two_ref_candidates.size(); ++first_idx) {
              const uint32_t first_offset =
                  two_ref_candidates[first_idx].offset;
              const uintptr_t* anchor1 =
                  &(block_genovecs[static_cast<uintptr_t>(first_offset) *
                                    genovec_word_stride]);
              for (uint32_t second_idx = first_idx + 1;
                   second_idx != two_ref_candidates.size(); ++second_idx) {
                const uint32_t second_offset =
                    two_ref_candidates[second_idx].offset;
                const uintptr_t* anchor2 =
                    &(block_genovecs[static_cast<uintptr_t>(second_offset) *
                                      genovec_word_stride]);
                uint32_t triple_counts[64];
                CountTripleGenotypes(anchor1, anchor2, target, sample_ct,
                                     triple_counts);
                const RansEstimate estimate =
                    EstimateRans(triple_counts, 16, 2, opts);
                if (estimate.record_bytes <
                    target_result.scheduled_two_ref_best.record_bytes) {
                  target_result.scheduled_two_ref_best = estimate;
                  target_result.scheduled_two_ref1 =
                      static_cast<int32_t>(block_start + first_offset);
                  target_result.scheduled_two_ref2 =
                      static_cast<int32_t>(block_start + second_offset);
                }
                if (opts.project_sample_ct) {
                  const uint64_t candidate_projected_bytes =
                      ProjectRansRecordBytes(estimate, opts, sample_ct,
                                             opts.project_sample_ct);
                  if (candidate_projected_bytes <
                      projected_scheduled_two_ref_bytes) {
                    projected_scheduled_two_ref_bytes =
                        candidate_projected_bytes;
                    target_result.projected_scheduled_two_ref_best = estimate;
                    target_result.projected_scheduled_two_ref1 =
                        static_cast<int32_t>(block_start + first_offset);
                    target_result.projected_scheduled_two_ref2 =
                        static_cast<int32_t>(block_start + second_offset);
                  }
                }
              }
            }
          }
        }
      });
    }
    for (std::thread& worker : workers) {
      worker.join();
    }

    ++totals.block_ct;
    for (uint32_t offset = 0; offset != block_len; ++offset) {
      const uint32_t vidx = block_start + offset;
      const VariantResult& result = results[offset];
      ++totals.variant_ct;
      totals.genotype_call_ct += sample_ct;
      totals.pgen_payload_bytes += result.pgen_bytes;
      totals.marginal_shannon_bits += result.marginal.shannon_bits;
      totals.previous_shannon_bits += result.previous_best.shannon_bits;
      totals.scheduled_shannon_bits += result.scheduled_best.shannon_bits;
      totals.scheduled_two_ref_shannon_bits +=
          result.scheduled_two_ref_best.shannon_bits;
      totals.marginal_bytes += result.marginal.record_bytes;
      totals.previous_bytes += result.previous_best.record_bytes;
      totals.scheduled_bytes += result.scheduled_best.record_bytes;
      totals.scheduled_two_ref_bytes +=
          result.scheduled_two_ref_best.record_bytes;
      if (opts.project_sample_ct) {
        totals.projected_marginal_bytes += ProjectRansRecordBytes(
            result.marginal, opts, sample_ct, opts.project_sample_ct);
        totals.projected_previous_bytes += ProjectRansRecordBytes(
            result.projected_previous_best, opts, sample_ct,
            opts.project_sample_ct);
        totals.projected_scheduled_bytes += ProjectRansRecordBytes(
            result.projected_scheduled_best, opts, sample_ct,
            opts.project_sample_ct);
        totals.projected_scheduled_two_ref_bytes += ProjectRansRecordBytes(
            result.projected_scheduled_two_ref_best, opts, sample_ct,
            opts.project_sample_ct);
      }
      totals.marginal_model_bytes += result.marginal.model_bytes;
      totals.previous_model_bytes += result.previous_best.model_bytes;
      totals.scheduled_model_bytes += result.scheduled_best.model_bytes;
      totals.scheduled_two_ref_model_bytes +=
          result.scheduled_two_ref_best.model_bytes;
      totals.previous_conditional_ct += (result.previous_ref >= 0);
      totals.scheduled_conditional_ct += (result.scheduled_ref >= 0);
      totals.scheduled_two_ref_ct +=
          (result.scheduled_two_ref1 >= 0);
      totals.projected_previous_conditional_ct +=
          (result.projected_previous_ref >= 0);
      totals.projected_scheduled_conditional_ct +=
          (result.projected_scheduled_ref >= 0);
      totals.projected_scheduled_two_ref_ct +=
          (result.projected_scheduled_two_ref1 >= 0);
      totals.scheduled_anchor_ct += result.scheduled_anchor;
      if (pgfi.nonref_flags && IsSet(pgfi.nonref_flags, vidx)) {
        ++totals.nonref_ct;
      }
      if (variant_out) {
        const char* chrom = ".";
        uint64_t bp = 0;
        if (meta_ptr) {
          chrom = pvar.chrom_names[(*meta_ptr)[vidx].chrom_code].c_str();
          bp = (*meta_ptr)[vidx].bp;
        }
        fprintf(
            variant_out,
            "%u\t%s\t%llu\t%.8g\t%u\t%.3f\t%llu\t%d\t%.3f\t%llu\t"
            "%d\t%u\t%.3f\t%llu\t%d\t%.3f\t%llu\t%d\t%.3f\t%llu\t"
            "%d\t%d\t%.3f\t%llu\t%d\t%d\t%.3f\t%llu\n",
            vidx, chrom, static_cast<unsigned long long>(bp), result.maf,
            result.pgen_bytes, result.marginal.shannon_bits / 8.0,
            static_cast<unsigned long long>(result.marginal.record_bytes),
            result.previous_ref, result.previous_best.shannon_bits / 8.0,
            static_cast<unsigned long long>(
                result.previous_best.record_bytes),
            result.scheduled_ref, result.scheduled_anchor ? 1U : 0U,
            result.scheduled_best.shannon_bits / 8.0,
            static_cast<unsigned long long>(
                result.scheduled_best.record_bytes),
            result.projected_previous_ref,
            result.projected_previous_best.shannon_bits / 8.0,
            static_cast<unsigned long long>(
                opts.project_sample_ct
                    ? ProjectRansRecordBytes(result.projected_previous_best,
                                             opts, sample_ct,
                                             opts.project_sample_ct)
                    : result.projected_previous_best.record_bytes),
            result.projected_scheduled_ref,
            result.projected_scheduled_best.shannon_bits / 8.0,
            static_cast<unsigned long long>(
                opts.project_sample_ct
                    ? ProjectRansRecordBytes(result.projected_scheduled_best,
                                             opts, sample_ct,
                                             opts.project_sample_ct)
                    : result.projected_scheduled_best.record_bytes),
            result.scheduled_two_ref1, result.scheduled_two_ref2,
            result.scheduled_two_ref_best.shannon_bits / 8.0,
            static_cast<unsigned long long>(
                result.scheduled_two_ref_best.record_bytes),
            result.projected_scheduled_two_ref1,
            result.projected_scheduled_two_ref2,
            result.projected_scheduled_two_ref_best.shannon_bits / 8.0,
            static_cast<unsigned long long>(
                opts.project_sample_ct
                    ? ProjectRansRecordBytes(
                          result.projected_scheduled_two_ref_best, opts,
                          sample_ct, opts.project_sample_ct)
                    : result.projected_scheduled_two_ref_best.record_bytes));
      }
    }
    processed_variant_ct += block_len;
    if (!(processed_variant_ct % 10000) ||
        (processed_variant_ct == variant_ct)) {
      const double pct = 100.0 * processed_variant_ct / variant_ct;
      fprintf(stderr, "\rAnalyzed %u/%u%s variants (%.1f%%).",
              processed_variant_ct, variant_ct,
              opts.sample_block_ct ? " sampled" : "", pct);
      fflush(stderr);
    }
  }
  fputc('\n', stderr);

  metadata_byte_ct =
      (totals.nonref_ct && (totals.nonref_ct != totals.variant_ct))
          ? ((totals.variant_ct + 7) / 8)
          : 0;
  container_overhead_bytes =
      ContainerOverheadByteCt(opts, analysis_blocks, metadata_byte_ct);
  totals.marginal_bytes += container_overhead_bytes;
  totals.previous_bytes += container_overhead_bytes;
  totals.scheduled_bytes += container_overhead_bytes;
  if (opts.two_ref_shortlist) {
    totals.scheduled_two_ref_bytes += container_overhead_bytes;
  }
  if (opts.project_sample_ct) {
    totals.projected_marginal_bytes += container_overhead_bytes;
    totals.projected_previous_bytes += container_overhead_bytes;
    totals.projected_scheduled_bytes += container_overhead_bytes;
    if (opts.two_ref_shortlist) {
      totals.projected_scheduled_two_ref_bytes +=
          container_overhead_bytes;
    }
  }

  {
    const auto stop_time = std::chrono::steady_clock::now();
    const double elapsed_seconds =
        std::chrono::duration<double>(stop_time - start_time).count();
    const uint64_t pgen_file_bytes = FileSize(opts.pgen_fname);
    const bool whole_pgen =
        (!opts.sample_block_ct) &&
        (source_variant_ct == pgfi.raw_variant_ct);
    const uint64_t comparison_pgen_bytes =
        (whole_pgen && pgen_file_bytes) ? pgen_file_bytes
                                       : totals.pgen_payload_bytes;
    printf("\nResults\n");
    printf("  elapsed seconds:             %.3f\n", elapsed_seconds);
    printf("  PGEN file bytes:             %llu\n",
           static_cast<unsigned long long>(pgen_file_bytes));
    printf("  PGEN analyzed payload bytes: %llu\n",
           static_cast<unsigned long long>(totals.pgen_payload_bytes));
    printf("  exact container overhead:    %llu\n",
           static_cast<unsigned long long>(container_overhead_bytes));
    printf("  scheduled anchors:           %llu\n",
           static_cast<unsigned long long>(totals.scheduled_anchor_ct));
    printf("  previous conditional calls:  %llu (%.2f%%)\n",
           static_cast<unsigned long long>(totals.previous_conditional_ct),
           100.0 * totals.previous_conditional_ct / totals.variant_ct);
    printf("  scheduled conditional calls: %llu (%.2f%%)\n",
           static_cast<unsigned long long>(totals.scheduled_conditional_ct),
           100.0 * totals.scheduled_conditional_ct / totals.variant_ct);
    if (opts.two_ref_shortlist) {
      printf("  scheduled two-reference calls: %llu (%.2f%%)\n",
             static_cast<unsigned long long>(totals.scheduled_two_ref_ct),
             100.0 * totals.scheduled_two_ref_ct / totals.variant_ct);
    }
    putchar('\n');
    printf("%-30s %12s %12s %12s\n", "Encoding", "GB", "bits/call",
           "PGEN/est.");
    PrintEstimateRow(whole_pgen ? "Current PGEN file"
                                : "PGEN analyzed payload",
                     comparison_pgen_bytes, totals.genotype_call_ct,
                     comparison_pgen_bytes);
    PrintEstimateRow("Marginal rANS estimate", totals.marginal_bytes,
                     totals.genotype_call_ct, comparison_pgen_bytes);
    PrintEstimateRow("Previous-window lower bound", totals.previous_bytes,
                     totals.genotype_call_ct, comparison_pgen_bytes);
    PrintEstimateRow("Scheduled depth-one estimate", totals.scheduled_bytes,
                     totals.genotype_call_ct, comparison_pgen_bytes);
    if (opts.two_ref_shortlist) {
      PrintEstimateRow("Scheduled up-to-two-ref est.",
                       totals.scheduled_two_ref_bytes,
                       totals.genotype_call_ct, comparison_pgen_bytes);
    }
    if (opts.sample_block_ct && pgen_file_bytes) {
      const long double full_scale =
          static_cast<long double>(pgfi.raw_variant_ct) / totals.variant_ct;
      const uint64_t full_call_ct =
          static_cast<uint64_t>(pgfi.raw_variant_ct) * sample_ct;
      printf(
          "\nStratified whole-file extrapolation "
          "(sampled probabilities held fixed):\n");
      printf("%-30s %12s %12s %12s\n", "Encoding", "GB", "bits/call",
             "PGEN/est.");
      PrintEstimateRow("Current PGEN file", pgen_file_bytes, full_call_ct,
                       pgen_file_bytes);
      PrintEstimateRow("Marginal rANS extrapolation",
                       totals.marginal_bytes * full_scale, full_call_ct,
                       pgen_file_bytes);
      PrintEstimateRow("Previous-window extrapolation",
                       totals.previous_bytes * full_scale, full_call_ct,
                       pgen_file_bytes);
      PrintEstimateRow("Scheduled depth-one extrap.",
                       totals.scheduled_bytes * full_scale, full_call_ct,
                       pgen_file_bytes);
      if (opts.two_ref_shortlist) {
        PrintEstimateRow("Scheduled up-to-two-ref extr.",
                         totals.scheduled_two_ref_bytes * full_scale,
                         full_call_ct, pgen_file_bytes);
      }
    }
    printf("\nShannon lower bounds (payload only, no model/state/index):\n");
    PrintEstimateRow("Marginal Shannon",
                     totals.marginal_shannon_bits / 8.0L,
                     totals.genotype_call_ct, comparison_pgen_bytes);
    PrintEstimateRow("Previous-window Shannon",
                     totals.previous_shannon_bits / 8.0L,
                     totals.genotype_call_ct, comparison_pgen_bytes);
    PrintEstimateRow("Scheduled Shannon",
                     totals.scheduled_shannon_bits / 8.0L,
                     totals.genotype_call_ct, comparison_pgen_bytes);
    if (opts.two_ref_shortlist) {
      PrintEstimateRow("Up-to-two-reference Shannon",
                       totals.scheduled_two_ref_shannon_bits / 8.0L,
                       totals.genotype_call_ct, comparison_pgen_bytes);
    }
    if (opts.project_sample_ct) {
      const uint64_t projected_call_ct =
          totals.variant_ct * opts.project_sample_ct;
      const long double sample_scale =
          static_cast<long double>(opts.project_sample_ct) / sample_ct;
      const long double projected_pgen_payload =
          totals.pgen_payload_bytes * sample_scale;
      printf(
          "\nProjection to %u samples (observed probabilities held fixed; "
          "PGEN payload scaled linearly):\n",
          opts.project_sample_ct);
      printf("  projected previous conditional calls:  %llu (%.2f%%)\n",
             static_cast<unsigned long long>(
                 totals.projected_previous_conditional_ct),
             100.0 * totals.projected_previous_conditional_ct /
                 totals.variant_ct);
      printf("  projected scheduled conditional calls: %llu (%.2f%%)\n",
             static_cast<unsigned long long>(
                 totals.projected_scheduled_conditional_ct),
             100.0 * totals.projected_scheduled_conditional_ct /
                 totals.variant_ct);
      if (opts.two_ref_shortlist) {
        printf("  projected scheduled two-reference calls: %llu (%.2f%%)\n",
               static_cast<unsigned long long>(
                   totals.projected_scheduled_two_ref_ct),
               100.0 * totals.projected_scheduled_two_ref_ct /
                   totals.variant_ct);
      }
      printf("\n%-30s %12s %12s %12s\n", "Encoding", "GB", "bits/call",
             "PGEN/est.");
      PrintEstimateRow("PGEN payload projection", projected_pgen_payload,
                       projected_call_ct,
                       static_cast<uint64_t>(projected_pgen_payload));
      PrintEstimateRow("Marginal rANS projection",
                       totals.projected_marginal_bytes, projected_call_ct,
                       static_cast<uint64_t>(projected_pgen_payload));
      PrintEstimateRow("Previous-window projection",
                       totals.projected_previous_bytes, projected_call_ct,
                       static_cast<uint64_t>(projected_pgen_payload));
      PrintEstimateRow("Scheduled depth-one projection",
                       totals.projected_scheduled_bytes, projected_call_ct,
                       static_cast<uint64_t>(projected_pgen_payload));
      if (opts.two_ref_shortlist) {
        PrintEstimateRow("Scheduled up-to-two-ref proj.",
                         totals.projected_scheduled_two_ref_bytes,
                         projected_call_ct,
                         static_cast<uint64_t>(projected_pgen_payload));
      }
    }
    printf("\nEstimated model bytes: marginal=%llu, previous=%llu, "
           "scheduled=%llu",
           static_cast<unsigned long long>(totals.marginal_model_bytes),
           static_cast<unsigned long long>(totals.previous_model_bytes),
           static_cast<unsigned long long>(totals.scheduled_model_bytes));
    if (opts.two_ref_shortlist) {
      printf(", up-to-two-ref=%llu",
             static_cast<unsigned long long>(
                 totals.scheduled_two_ref_model_bytes));
    }
    putchar('\n');
  }

  return_code = 0;

cleanup:
  if (variant_out) {
    fclose(variant_out);
  }
  CleanupPgr(&pgr, &reterr);
  CleanupPgfi(&pgfi, &reterr);
  if (block_genovecs) {
    aligned_free(block_genovecs);
  }
  if (pgr_alloc) {
    aligned_free(pgr_alloc);
  }
  if (pgfi_alloc) {
    aligned_free(pgfi_alloc);
  }
  return return_code;
}
