// SPDX-License-Identifier: GPL-3.0-or-later

#include "pgen_rans_plink.h"

#include <algorithm>
#include <array>
#include <cinttypes>
#include <cstring>
#include <new>
#include <vector>

#include "include/plink2_base.h"
#include "include/plink2_bits.h"
#include "pgen_rans_codec.h"
#include "pgen_rans_container.h"
#include "pgen_rans_encode.h"

namespace pgen_rans {
namespace {

using namespace plink2;

static_assert(
    kPgenRansStorageMode == kPgenDeveloperRansStorageMode,
    "Conditional-rANS PGEN storage-mode constants disagree.");

PglErr ReadFailure(const std::string& message, std::string* last_error) {
  *last_error = message;
  return kPglRetReadFail;
}

}  // namespace

PglErr PrintPgenInfo(const char* pgenname) {
  ContainerReader reader;
  std::string error;
  if (!reader.Open(pgenname, &error)) {
    logerrprintfww("Error: Failed to inspect %s: %s\n", pgenname,
                   error.c_str());
    return kPglRetMalformedInput;
  }
  const ContainerParams& params = reader.params();
  logprintf("PGEN file: %s\n", pgenname);
  logputs("  Storage mode: 0x80 (block-local conditional-rANS hardcalls)\n");
  logprintf("  Variants: %u\n", params.variant_ct);
  logprintf("  Samples: %u\n", params.sample_ct);
  logprintf("  Block variants / anchors: %u / %u\n",
            params.block_variant_ct, params.anchor_ct);
  logprintf("  rANS states / scale bits: %u / %u\n",
            params.state_ct, params.scale_bits);
  logprintf("  Cumulative-offset restart interval: %u variants\n",
            params.restart_variant_ct);
  logprintf("  Maximum allele count for a single variant: %u\n",
            params.max_allele_ct);
  if (reader.metadata().all_nonref) {
    logputs("  REF alleles are all provisional\n");
  } else if (reader.metadata().nonref_flags.empty()) {
    logputs("  REF alleles are all known\n");
  } else {
    logputs("  REF alleles are a mix of known and provisional\n");
  }
  logputs("  No hardcalls are explicitly phased\n");
  logputs("  No dosages present\n");
  return kPglRetSuccess;
}

PglErr MakePgen(
    const uintptr_t* sample_include, const uintptr_t* variant_include,
    const ChrInfo* cip, const uint32_t* variant_bps,
    const uintptr_t* allele_idx_offsets, uint32_t raw_sample_ct,
    uint32_t sample_ct, uint32_t raw_variant_ct, uint32_t variant_ct,
    uint32_t max_thread_ct, uint32_t discard_phase,
    uint32_t discard_dosage, PgenFileInfo* pgfip, PgenReader* pgrp,
    char* outname, char* outname_end) {
  std::vector<uint32_t> variant_uidxs(variant_ct);
  std::vector<VariantMetadata> metadata(variant_ct);
  uintptr_t variant_uidx_base = 0;
  uintptr_t variant_include_bits = variant_include[0];
  for (uint32_t variant_idx = 0; variant_idx != variant_ct; ++variant_idx) {
    const uint32_t variant_uidx =
        BitIter1(variant_include, &variant_uidx_base, &variant_include_bits);
    const uintptr_t allele_ct =
        allele_idx_offsets
            ? allele_idx_offsets[variant_uidx + 1] -
                  allele_idx_offsets[variant_uidx]
            : 2;
    if ((allele_ct < 2) || (allele_ct > 255)) {
      logerrputs(
          "Error: Conditional-rANS PGEN encountered an unsupported allele "
          "count.\n");
      return kPglRetInconsistentInput;
    }
    variant_uidxs[variant_idx] = variant_uidx;
    metadata[variant_idx].chrom_code = GetVariantChr(cip, variant_uidx);
    metadata[variant_idx].bp = variant_bps[variant_uidx];
    metadata[variant_idx].allele_ct =
        static_cast<uint16_t>(allele_ct);
  }

  EncodeInput input;
  input.raw_sample_ct = raw_sample_ct;
  input.sample_ct = sample_ct;
  input.raw_variant_ct = raw_variant_ct;
  input.variant_ct = variant_ct;
  input.sample_include =
      (sample_ct == raw_sample_ct) ? nullptr : sample_include;
  input.variant_uidxs = variant_uidxs.data();
  input.variant_metadata = metadata.data();
  input.pgfi = pgfip;
  input.pgen_reader = pgrp;
  input.discard_phase = discard_phase;
  input.discard_dosage = discard_dosage;

  EncodeParams params;
  params.thread_ct = max_thread_ct;
  EncodeStats stats;
  std::string error;
  snprintf(outname_end, kMaxOutfnameExtBlen, ".pgen");
  logprintfww5("--make-pgen format=rans to %s ... ", outname);
  const PglErr reterr =
      EncodePgenRans(outname, input, params, &stats, &error);
  *outname_end = '\0';
  if (unlikely(reterr)) {
    logputs("\n");
    logerrprintfww(
        "Error: --make-pgen format=rans failed: %s\n", error.c_str());
    return reterr;
  }
  logputs("done.\n");
  logprintf(
      "Conditional-rANS encode complete: %u sample%s, %u variant%s, "
      "%u block%s.\n",
      sample_ct, (sample_ct == 1) ? "" : "s", stats.variant_ct,
      (stats.variant_ct == 1) ? "" : "s", stats.block_ct,
      (stats.block_ct == 1) ? "" : "s");
  logprintf(
      "  Marginal/one-reference/two-reference records: %" PRIu64
      "/%" PRIu64 "/%" PRIu64 "\n",
      stats.marginal_ct, stats.one_reference_ct,
      stats.two_reference_ct);
  if (stats.multiallelic_ct) {
    logprintf("  Multiallelic variants: %" PRIu64
              "; ref/ALT patches: %" PRIu64
              "; ALT/ALT patches: %" PRIu64
              "; patch bytes: %" PRIu64 ".\n",
              stats.multiallelic_ct, stats.patch_01_ct,
              stats.patch_10_ct, stats.multiallelic_patch_bytes);
  }
  logprintf("  Source PGEN payload bytes: %" PRIu64
            "; conditional PGEN bytes: %" PRIu64
            "; ratio: %.3f; elapsed: %.3f seconds.\n",
            stats.pgen_payload_bytes, stats.output_bytes,
            stats.output_bytes
                ? static_cast<double>(stats.pgen_payload_bytes) /
                      stats.output_bytes
                : 0.0,
            stats.elapsed_seconds);
  return kPglRetSuccess;
}

PglErr LoadPgen(
    const PlinkLoadOptions& options, uintptr_t** nonref_flags_ptr,
    PgenFileInfo* pgfi, PgenReader* pgr,
    uintptr_t* pgr_alloc_cacheline_ct_ptr,
    PlinkRansAdapter** adapter_ptr) {
  PlinkRansAdapter* adapter = new (std::nothrow) PlinkRansAdapter();
  *adapter_ptr = adapter;
  if (unlikely(!adapter)) {
    return kPglRetNomem;
  }
  std::string error;
  if (unlikely(!adapter->Open(
          options.pgenname, options.thread_ct, &error))) {
    logerrprintfww("Error: Failed to open %s: %s\n", options.pgenname,
                   error.c_str());
    return kPglRetOpenFail;
  }
  if (unlikely(
          (adapter->variant_ct() != options.raw_variant_ct) ||
          (adapter->sample_ct() != options.raw_sample_ct))) {
    logerrprintfww(
        "Error: %s contains %u variants and %u samples, while its "
        "companion metadata contains %u variants and %u samples.\n",
        options.pgenname, adapter->variant_ct(), adapter->sample_ct(),
        options.raw_variant_ct, options.raw_sample_ct);
    return kPglRetInconsistentInput;
  }
  if (unlikely(adapter->max_allele_ct() != options.max_allele_ct)) {
    logerrprintfww(
        "Error: Maximum allele count mismatch between %s (%u) and "
        "%s (%u).\n",
        options.pgenname, adapter->max_allele_ct(), options.pvarname,
        options.max_allele_ct);
    return kPglRetInconsistentInput;
  }
  if (unlikely(options.unsupported_command)) {
    logerrputs(
        "Error: This conditional-rANS PGEN operation is not yet "
        "supported by the\nalternate hardcall backend.\n");
    return kPglRetInvalidCmdline;
  }
  if (unlikely(options.unsupported_export)) {
    logerrputs(
        "Error: Conditional-rANS PGEN currently supports only "
        "--export A and\n--export Av.\n");
    return kPglRetInvalidCmdline;
  }
  if (unlikely(options.phased_ld)) {
    logerrputs(
        "Error: Conditional-rANS PGEN stores unphased hardcalls; use "
        "--indep-pairwise\nor --r-unphased instead of a phased LD "
        "command.\n");
    return kPglRetInvalidCmdline;
  }
  if (unlikely(options.unsupported_multiallelic_pca)) {
    logerrputs(
        "Error: --pca is currently limited to biallelic "
        "conditional-rANS PGEN\nfilesets.\n");
    return kPglRetInconsistentInput;
  }
  if (unlikely(options.unsupported_multiallelic_scan)) {
    logerrputs(
        "Error: This conditional-rANS PGEN operation requires an "
        "allele-frequency\nscan which is currently limited to "
        "biallelic filesets.  For multiallelic\nscoring, provide "
        "--read-freq; otherwise apply genotype-frequency filters "
        "while\ncreating the conditional-rANS PGEN fileset.\n");
    return kPglRetInconsistentInput;
  }

  uintptr_t* nonref_flags = *nonref_flags_ptr;
  if (nonref_flags) {
    for (uint32_t variant_uidx = 0;
         variant_uidx != options.raw_variant_ct; ++variant_uidx) {
      if (unlikely(
              IsSet(nonref_flags, variant_uidx) !=
              adapter->variant_is_nonref(variant_uidx))) {
        logerrprintfww(
            "Error: Provisional-REF metadata mismatch between %s and "
            "%s at variant index %u.\n",
            options.pgenname, options.pvarname, variant_uidx);
        return kPglRetInconsistentInput;
      }
    }
  } else if (adapter->has_mixed_nonref_flags()) {
    if (unlikely(bigstack_calloc_w(
            BitCtToWordCt(options.raw_variant_ct), &nonref_flags))) {
      return kPglRetNomem;
    }
    for (uint32_t variant_uidx = 0;
         variant_uidx != options.raw_variant_ct; ++variant_uidx) {
      if (adapter->variant_is_nonref(variant_uidx)) {
        SetBit(variant_uidx, nonref_flags);
      }
    }
    *nonref_flags_ptr = nonref_flags;
  }

  pgfi->raw_variant_ct = options.raw_variant_ct;
  pgfi->raw_sample_ct = options.raw_sample_ct;
  pgfi->const_fpos_offset = 0;
  pgfi->const_vrec_width = DivUp(options.raw_sample_ct, 4);
  pgfi->const_vrtype = 0;
  pgfi->var_fpos = nullptr;
  pgfi->vrtypes = nullptr;
  pgfi->allele_idx_offsets = options.allele_idx_offsets;
  pgfi->nonref_flags = nonref_flags;
  pgfi->gflags =
      (options.max_allele_ct > 2)
          ? kfPgenGlobalMultiallelicHardcallFound
          : kfPgenGlobal0;
  if (adapter->all_nonref()) {
    pgfi->gflags |= kfPgenGlobalAllNonref;
  }
  pgfi->max_allele_ct = options.max_allele_ct;
  pgfi->extensions_present = 0;
  fclose(pgfi->shared_ff);
  pgfi->shared_ff = nullptr;
  pgfi->pgi_ff = nullptr;
  pgfi->block_base = nullptr;
  pgfi->block_offset = 0;
  *pgr_alloc_cacheline_ct_ptr = CountPgrAllocCachelinesRequired(
      options.raw_sample_ct, pgfi->gflags, options.max_allele_ct, 0);
  adapter->Install(pgfi, pgr);
  logprintfww("%u variants and %u samples loaded from %s.\n",
              options.raw_variant_ct, options.raw_sample_ct,
              options.pgenname);

  if (options.validate) {
    logprintfww5("Validating %s... ", options.pgenname);
    fflush(stdout);
    if (unlikely(!adapter->Validate(&error))) {
      logputs("\n");
      logerrprintfww(
          "Error: Conditional-rANS PGEN validation failed: %s\n",
          error.c_str());
      return kPglRetMalformedInput;
    }
    logputs("done.\n");
  }
  if (options.print_info) {
    return PrintPgenInfo(options.pgenname);
  }
  return kPglRetSuccess;
}

PlinkRansAdapter::PlinkRansAdapter() : backend_{} {
  backend_.context = this;
  backend_.get = &Get;
  backend_.get_allele = &GetAllele;
  backend_.get_counts = &GetCounts;
  backend_.get_difflist_or_genovec = &GetDifflistOrGenovec;
  backend_.get_packed = &GetPacked;
  backend_.get_packed_batch = &GetPackedBatch;
  backend_.get_m = &GetM;
  backend_.get_raw = &GetRaw;
}

PlinkRansAdapter::~PlinkRansAdapter() {
  Close();
}

bool PlinkRansAdapter::Open(const std::string& path, uint32_t thread_ct,
                            std::string* error) {
  Close();
  if (!reader_.Open(path, thread_ct, error)) {
    return false;
  }
  raw_word_ct_ =
      NypCtToVecCt(reader_.raw_sample_ct()) * kWordsPerVec;
  if (cachealigned_malloc(
          static_cast<uintptr_t>(raw_word_ct_) * sizeof(uintptr_t),
          &raw_genovec_) ||
      cachealigned_malloc(
          static_cast<uintptr_t>(raw_word_ct_) * sizeof(uintptr_t),
          &count_genovec_)) {
    if (error) {
      *error =
          "Out of memory allocating the conditional-rANS PGEN adapter.";
    }
    Close();
    return false;
  }
  return true;
}

bool PlinkRansAdapter::Validate(std::string* error) {
  std::vector<uint8_t> packed(reader_.packed_variant_byte_ct());
  MultiallelicPatches patches;
  for (uint32_t vidx = 0; vidx != reader_.variant_ct(); ++vidx) {
    if (!reader_.ReadVariant(
            vidx, packed.data(), packed.size(), nullptr, error) ||
        ((reader_.max_allele_ct() > 2) &&
         (!reader_.ReadVariantPatches(
             vidx, &patches, nullptr, error)))) {
      return false;
    }
  }
  return true;
}

void PlinkRansAdapter::Close() {
  aligned_free_cond(count_genovec_);
  aligned_free_cond(raw_genovec_);
  count_genovec_ = nullptr;
  raw_genovec_ = nullptr;
  raw_word_ct_ = 0;
  reader_.Close();
  last_error_.clear();
}

void PlinkRansAdapter::Install(PgenFileInfo* pgfi, PgenReader* pgr) {
  pgfi->multiread_backend = &backend_;
  PgenReaderMain* pgrp = &GET_PRIVATE(*pgr, m);
  pgrp->fi = *pgfi;
  pgrp->hardcall_backend = &backend_;
}

uint32_t PlinkRansAdapter::sample_ct() const {
  return reader_.raw_sample_ct();
}

uint32_t PlinkRansAdapter::variant_ct() const {
  return reader_.variant_ct();
}

uint32_t PlinkRansAdapter::max_allele_ct() const {
  return reader_.max_allele_ct();
}

bool PlinkRansAdapter::all_nonref() const {
  return reader_.all_nonref();
}

bool PlinkRansAdapter::has_mixed_nonref_flags() const {
  return reader_.has_mixed_nonref_flags();
}

bool PlinkRansAdapter::variant_is_nonref(uint32_t variant) const {
  return reader_.variant_is_nonref(variant);
}

const std::string& PlinkRansAdapter::last_error() const {
  return last_error_;
}

bool PlinkRansAdapter::ReadRaw(uint32_t vidx) {
  std::fill(raw_genovec_, raw_genovec_ + raw_word_ct_, 0);
  std::string error;
  if (!reader_.ReadVariant(
          vidx, reinterpret_cast<uint8_t*>(raw_genovec_),
          reader_.packed_variant_byte_ct(), nullptr, &error)) {
    last_error_ = error;
    return false;
  }
  ZeroTrailingNyps(reader_.raw_sample_ct(), raw_genovec_);
  return true;
}

void PlinkRansAdapter::CopySubset(
    const uintptr_t* sample_include, uint32_t sample_ct,
    uintptr_t* destination) const {
  if (sample_ct == reader_.raw_sample_ct()) {
    memcpy(destination, raw_genovec_,
           NypCtToWordCt(sample_ct) * sizeof(uintptr_t));
  } else {
    CopyNyparrNonemptySubset(
        raw_genovec_, sample_include, reader_.raw_sample_ct(), sample_ct,
        destination);
  }
}

PglErr PlinkRansAdapter::ReadBase(
    const uintptr_t* sample_include, uint32_t sample_ct,
    uint32_t vidx, uintptr_t* genovec) {
  if ((!sample_ct) || (!genovec) ||
      (sample_ct > reader_.raw_sample_ct()) ||
      ((sample_ct != reader_.raw_sample_ct()) && (!sample_include))) {
    return kPglRetImproperFunctionCall;
  }
  if (!ReadRaw(vidx)) {
    return kPglRetReadFail;
  }
  CopySubset(sample_include, sample_ct, genovec);
  return kPglRetSuccess;
}

PglErr PlinkRansAdapter::ReadDifflistOrGenovec(
    const uintptr_t* sample_include,
    const uint32_t* sample_include_cumulative_popcounts,
    uint32_t sample_ct, uint32_t vidx, uint32_t max_difflist_len,
    uintptr_t* genovec, uint32_t* difflist_common_geno_ptr,
    uintptr_t* main_raregeno, uint32_t* difflist_sample_ids,
    uint32_t* difflist_len_ptr) {
  *difflist_common_geno_ptr = UINT32_MAX;
  *difflist_len_ptr = 0;
  const uint32_t raw_sample_ct = reader_.raw_sample_ct();
  if ((!sample_ct) || (sample_ct > raw_sample_ct) ||
      (max_difflist_len > sample_ct) || (!genovec) || (!main_raregeno) ||
      (!difflist_sample_ids) ||
      ((sample_ct != raw_sample_ct) &&
       ((!sample_include) ||
        (!sample_include_cumulative_popcounts)))) {
    return kPglRetImproperFunctionCall;
  }

  std::string error;
  if (!reader_.ReadVariantMaybeSparse(
          vidx, raw_sample_ct, reinterpret_cast<uint8_t*>(raw_genovec_),
          reader_.packed_variant_byte_ct(), &sparse_result_, nullptr,
          &error)) {
    return ReadFailure(error, &last_error_);
  }
  if (sparse_result_.common_genotype == UINT32_MAX) {
    ZeroTrailingNyps(raw_sample_ct, raw_genovec_);
    CopySubset(sample_include, sample_ct, genovec);
    return kPglRetSuccess;
  }

  uint32_t output_len = 0;
  for (size_t raw_idx = 0; raw_idx != sparse_result_.sample_ids.size();
       ++raw_idx) {
    const uint32_t raw_sample_idx = sparse_result_.sample_ids[raw_idx];
    if (raw_sample_idx >= raw_sample_ct) {
      last_error_ =
          "Conditional-rANS sparse sample index is out of range.";
      return kPglRetMalformedInput;
    }
    if (sample_ct != raw_sample_ct) {
      if (!IsSet(sample_include, raw_sample_idx)) {
        continue;
      }
    }
    if (output_len == max_difflist_len) {
      // The raw record still gives us an exact, inexpensive way to materialize
      // dense hardcalls.  Avoid invoking the entropy decoder on threshold
      // fallback.
      const uintptr_t common_word =
          sparse_result_.common_genotype * kMask5555;
      std::fill(raw_genovec_, raw_genovec_ + raw_word_ct_, common_word);
      for (size_t sparse_idx = 0;
           sparse_idx != sparse_result_.sample_ids.size(); ++sparse_idx) {
        AssignNyparrEntry(
            sparse_result_.sample_ids[sparse_idx],
            sparse_result_.genotypes[sparse_idx], raw_genovec_);
      }
      ZeroTrailingNyps(raw_sample_ct, raw_genovec_);
      CopySubset(sample_include, sample_ct, genovec);
      return kPglRetSuccess;
    }
    const uint32_t output_sample_idx =
        (sample_ct == raw_sample_ct)
            ? raw_sample_idx
            : RawToSubsettedPos(
                  sample_include, sample_include_cumulative_popcounts,
                  raw_sample_idx);
    difflist_sample_ids[output_len] = output_sample_idx;
    AssignNyparrEntry(
        output_len, sparse_result_.genotypes[raw_idx], main_raregeno);
    ++output_len;
  }
  *difflist_common_geno_ptr = sparse_result_.common_genotype;
  *difflist_len_ptr = output_len;
  return kPglRetSuccess;
}

PglErr PlinkRansAdapter::ReadAllele(
    const uintptr_t* sample_include, uint32_t sample_ct,
    uint32_t vidx, uint32_t allele_idx, uintptr_t* allele_countvec) {
  if ((!sample_ct) || (!allele_countvec) ||
      (sample_ct > reader_.raw_sample_ct()) ||
      ((sample_ct != reader_.raw_sample_ct()) && (!sample_include))) {
    return kPglRetImproperFunctionCall;
  }
  if (!ReadRaw(vidx)) {
    return kPglRetReadFail;
  }
  MultiallelicPatches patches;
  if (reader_.max_allele_ct() > 2) {
    std::string error;
    if (!reader_.ReadVariantPatches(vidx, &patches, nullptr, &error)) {
      return ReadFailure(error, &last_error_);
    }
  }
  if (allele_idx >= patches.allele_ct) {
    last_error_ =
        "Requested allele is outside the conditional-rANS PGEN variant.";
    return kPglRetInconsistentInput;
  }
  const uint32_t raw_sample_ct = reader_.raw_sample_ct();
  if (!allele_idx) {
    GenovecInvertUnsafe(raw_sample_ct, raw_genovec_);
  } else if (patches.allele_ct > 2) {
    if (allele_idx == 1) {
      for (const uint32_t sample_idx : patches.patch_01_sample_ids) {
        SetPackedGenotype(
            reinterpret_cast<uint64_t*>(raw_genovec_), sample_idx, 0);
      }
      for (size_t patch_idx = 0;
           patch_idx != patches.patch_10_sample_ids.size(); ++patch_idx) {
        const uint8_t count =
            static_cast<uint8_t>(
                (patches.patch_10_values[2 * patch_idx] == 1) +
                (patches.patch_10_values[2 * patch_idx + 1] == 1));
        SetPackedGenotype(
            reinterpret_cast<uint64_t*>(raw_genovec_),
            patches.patch_10_sample_ids[patch_idx], count);
      }
    } else {
      for (uint32_t sample_idx = 0; sample_idx != raw_sample_ct;
           ++sample_idx) {
        if (GetPackedGenotype(
                reinterpret_cast<const uint64_t*>(raw_genovec_),
                sample_idx) != 3) {
          SetPackedGenotype(
              reinterpret_cast<uint64_t*>(raw_genovec_), sample_idx, 0);
        }
      }
      for (size_t patch_idx = 0;
           patch_idx != patches.patch_01_sample_ids.size(); ++patch_idx) {
        if (patches.patch_01_values[patch_idx] == allele_idx) {
          SetPackedGenotype(
              reinterpret_cast<uint64_t*>(raw_genovec_),
              patches.patch_01_sample_ids[patch_idx], 1);
        }
      }
      for (size_t patch_idx = 0;
           patch_idx != patches.patch_10_sample_ids.size(); ++patch_idx) {
        const uint8_t count =
            static_cast<uint8_t>(
                (patches.patch_10_values[2 * patch_idx] == allele_idx) +
                (patches.patch_10_values[2 * patch_idx + 1] == allele_idx));
        SetPackedGenotype(
            reinterpret_cast<uint64_t*>(raw_genovec_),
            patches.patch_10_sample_ids[patch_idx], count);
      }
    }
  }
  CopySubset(sample_include, sample_ct, allele_countvec);
  return kPglRetSuccess;
}

PglErr PlinkRansAdapter::ReadMultiallelic(
    const uintptr_t* sample_include,
    const uint32_t* sample_include_cumulative_popcounts,
    uint32_t sample_ct, uint32_t vidx, PgenVariant* pgv) {
  if ((!sample_ct) || (!pgv) || (!pgv->genovec) ||
      (!pgv->patch_01_set) || (!pgv->patch_01_vals) ||
      (!pgv->patch_10_set) || (!pgv->patch_10_vals) ||
      (sample_ct > reader_.raw_sample_ct()) ||
      ((sample_ct != reader_.raw_sample_ct()) &&
       ((!sample_include) ||
        (!sample_include_cumulative_popcounts)))) {
    return kPglRetImproperFunctionCall;
  }
  if (!ReadRaw(vidx)) {
    return kPglRetReadFail;
  }
  CopySubset(sample_include, sample_ct, pgv->genovec);
  pgv->patch_01_ct = 0;
  pgv->patch_10_ct = 0;
  ZeroWArr(BitCtToWordCt(sample_ct), pgv->patch_01_set);
  ZeroWArr(BitCtToWordCt(sample_ct), pgv->patch_10_set);

  MultiallelicPatches patches;
  std::string error;
  if (!reader_.ReadVariantPatches(
          vidx, &patches, nullptr, &error)) {
    return ReadFailure(error, &last_error_);
  }
  for (size_t patch_idx = 0;
       patch_idx != patches.patch_01_sample_ids.size(); ++patch_idx) {
    const uint32_t raw_sample_idx =
        patches.patch_01_sample_ids[patch_idx];
    if (sample_include && (!IsSet(sample_include, raw_sample_idx))) {
      continue;
    }
    const uint32_t sample_idx =
        sample_include
            ? RawToSubsettedPos(
                  sample_include, sample_include_cumulative_popcounts,
                  raw_sample_idx)
            : raw_sample_idx;
    SetBit(sample_idx, pgv->patch_01_set);
    pgv->patch_01_vals[pgv->patch_01_ct++] =
        patches.patch_01_values[patch_idx];
  }
  for (size_t patch_idx = 0;
       patch_idx != patches.patch_10_sample_ids.size(); ++patch_idx) {
    const uint32_t raw_sample_idx =
        patches.patch_10_sample_ids[patch_idx];
    if (sample_include && (!IsSet(sample_include, raw_sample_idx))) {
      continue;
    }
    const uint32_t sample_idx =
        sample_include
            ? RawToSubsettedPos(
                  sample_include, sample_include_cumulative_popcounts,
                  raw_sample_idx)
            : raw_sample_idx;
    SetBit(sample_idx, pgv->patch_10_set);
    pgv->patch_10_vals[2 * pgv->patch_10_ct] =
        patches.patch_10_values[2 * patch_idx];
    pgv->patch_10_vals[2 * pgv->patch_10_ct + 1] =
        patches.patch_10_values[2 * patch_idx + 1];
    ++pgv->patch_10_ct;
  }
  return kPglRetSuccess;
}

PglErr PlinkRansAdapter::ReadRawRecord(
    uint32_t vidx, PgenGlobalFlags read_gflags,
    uintptr_t** loadbuf_iter_ptr,
    unsigned char* loaded_vrtype_ptr) {
  if ((!loadbuf_iter_ptr) || (!*loadbuf_iter_ptr) ||
      (vidx >= reader_.variant_ct())) {
    return kPglRetImproperFunctionCall;
  }
  const uint32_t raw_sample_ct = reader_.raw_sample_ct();
  uintptr_t* genovec = *loadbuf_iter_ptr;
  const uint32_t aligned_genovec_word_ct =
      NypCtToAlignedWordCt(raw_sample_ct);
  ZeroWArr(aligned_genovec_word_ct, genovec);
  std::string error;
  if (!reader_.ReadVariant(
          vidx, reinterpret_cast<uint8_t*>(genovec),
          reader_.packed_variant_byte_ct(), nullptr, &error)) {
    return ReadFailure(error, &last_error_);
  }
  ZeroTrailingNyps(raw_sample_ct, genovec);
  uintptr_t* loadbuf_iter =
      &(genovec[aligned_genovec_word_ct]);
  if (loaded_vrtype_ptr) {
    *loaded_vrtype_ptr = 0;
  }
  if ((reader_.max_allele_ct() == 2) ||
      (!(read_gflags & kfPgenGlobalMultiallelicHardcallFound))) {
    *loadbuf_iter_ptr = loadbuf_iter;
    return kPglRetSuccess;
  }

  MultiallelicPatches patches;
  if (!reader_.ReadVariantPatches(
          vidx, &patches, nullptr, &error)) {
    return ReadFailure(error, &last_error_);
  }
  const uint32_t rare01_ct =
      static_cast<uint32_t>(patches.patch_01_sample_ids.size());
  const uint32_t rare10_ct =
      static_cast<uint32_t>(patches.patch_10_sample_ids.size());
  if ((!rare01_ct) && (!rare10_ct)) {
    *loadbuf_iter_ptr = loadbuf_iter;
    return kPglRetSuccess;
  }
  if ((patches.patch_01_values.size() != rare01_ct) ||
      (patches.patch_10_values.size() != 2 * rare10_ct)) {
    last_error_ =
        "Malformed conditional-rANS PGEN multiallelic patch vectors.";
    return kPglRetMalformedInput;
  }
  if (loaded_vrtype_ptr) {
    *loaded_vrtype_ptr = 8;
  }
  loadbuf_iter[0] = rare01_ct;
  loadbuf_iter[1] = rare10_ct;
  loadbuf_iter = &(loadbuf_iter[RoundUpPow2(2, kWordsPerVec)]);
  const uint32_t raw_sample_ctl = BitCtToWordCt(raw_sample_ct);
  if (rare01_ct) {
    uintptr_t* patch_set = loadbuf_iter;
    ZeroWArr(raw_sample_ctl, patch_set);
    loadbuf_iter = &(loadbuf_iter[raw_sample_ctl]);
    AlleleCode* patch_values =
        reinterpret_cast<AlleleCode*>(loadbuf_iter);
    for (uint32_t patch_idx = 0; patch_idx != rare01_ct;
         ++patch_idx) {
      const uint32_t sample_idx =
          patches.patch_01_sample_ids[patch_idx];
      if (sample_idx >= raw_sample_ct) {
        last_error_ =
            "Conditional-rANS PGEN multiallelic patch sample is out of range.";
        return kPglRetMalformedInput;
      }
      SetBit(sample_idx, patch_set);
      patch_values[patch_idx] =
          patches.patch_01_values[patch_idx];
    }
    loadbuf_iter = &(loadbuf_iter[
        DivUp(rare01_ct,
              kBytesPerWord / sizeof(AlleleCode))]);
    AlignWToVec(&loadbuf_iter);
  }
  if (rare10_ct) {
    uintptr_t* patch_set = loadbuf_iter;
    ZeroWArr(raw_sample_ctl, patch_set);
    loadbuf_iter = &(loadbuf_iter[raw_sample_ctl]);
    AlleleCode* patch_values =
        reinterpret_cast<AlleleCode*>(loadbuf_iter);
    for (uint32_t patch_idx = 0; patch_idx != rare10_ct;
         ++patch_idx) {
      const uint32_t sample_idx =
          patches.patch_10_sample_ids[patch_idx];
      if (sample_idx >= raw_sample_ct) {
        last_error_ =
            "Conditional-rANS PGEN multiallelic patch sample is out of range.";
        return kPglRetMalformedInput;
      }
      SetBit(sample_idx, patch_set);
      patch_values[2 * patch_idx] =
          patches.patch_10_values[2 * patch_idx];
      patch_values[2 * patch_idx + 1] =
          patches.patch_10_values[2 * patch_idx + 1];
    }
    loadbuf_iter = &(loadbuf_iter[
        DivUp(rare10_ct,
              kBytesPerWord /
                  (2 * sizeof(AlleleCode)))]);
    AlignWToVec(&loadbuf_iter);
  }
  *loadbuf_iter_ptr = loadbuf_iter;
  return kPglRetSuccess;
}

PglErr PlinkRansAdapter::Get(
    void* context, const uintptr_t* sample_include,
    const uint32_t*, uint32_t sample_ct, uint32_t vidx,
    uintptr_t* genovec) {
  return static_cast<PlinkRansAdapter*>(context)->ReadBase(
      sample_include, sample_ct, vidx, genovec);
}

PglErr PlinkRansAdapter::GetAllele(
    void* context, const uintptr_t* sample_include,
    const uint32_t*, uint32_t sample_ct, uint32_t vidx,
    uint32_t allele_idx, uintptr_t* allele_countvec) {
  return static_cast<PlinkRansAdapter*>(context)->ReadAllele(
      sample_include, sample_ct, vidx, allele_idx, allele_countvec);
}

PglErr PlinkRansAdapter::GetCounts(
    void* context, const uintptr_t* sample_include,
    const uint32_t*, uint32_t sample_ct, uint32_t vidx,
    uint32_t* genocounts) {
  PlinkRansAdapter* adapter =
      static_cast<PlinkRansAdapter*>(context);
  const PglErr reterr = adapter->ReadBase(
      sample_include, sample_ct, vidx, adapter->count_genovec_);
  if (reterr) {
    return reterr;
  }
  std::array<uint32_t, 4> count_buffer;
  GenoarrCountFreqsUnsafe(
      adapter->count_genovec_, sample_ct, count_buffer);
  memcpy(genocounts, count_buffer.data(),
         count_buffer.size() * sizeof(uint32_t));
  return kPglRetSuccess;
}

PglErr PlinkRansAdapter::GetDifflistOrGenovec(
    void* context, const uintptr_t* sample_include,
    const uint32_t* sample_include_cumulative_popcounts,
    uint32_t sample_ct, uint32_t vidx, uint32_t max_difflist_len,
    uintptr_t* genovec, uint32_t* difflist_common_geno_ptr,
    uintptr_t* main_raregeno, uint32_t* difflist_sample_ids,
    uint32_t* difflist_len_ptr) {
  return static_cast<PlinkRansAdapter*>(context)->ReadDifflistOrGenovec(
      sample_include, sample_include_cumulative_popcounts, sample_ct, vidx,
      max_difflist_len, genovec, difflist_common_geno_ptr, main_raregeno,
      difflist_sample_ids, difflist_len_ptr);
}

PglErr PlinkRansAdapter::GetPacked(
    void* context, uint32_t vidx, unsigned char* packed_genotypes,
    uint32_t packed_byte_ct) {
  PlinkRansAdapter* adapter =
      static_cast<PlinkRansAdapter*>(context);
  std::string error;
  if (!adapter->reader_.ReadVariant(
          vidx, packed_genotypes, packed_byte_ct, nullptr, &error)) {
    fprintf(
        stderr,
        "Error: Conditional-rANS PGEN block materialization failed at "
        "variant %u: %s\n",
        vidx, error.c_str());
    return ReadFailure(error, &adapter->last_error_);
  }
  return kPglRetSuccess;
}

PglErr PlinkRansAdapter::GetPackedBatch(
    void* context, const uintptr_t* variant_include,
    uint32_t variant_uidx_start, uint32_t variant_uidx_end,
    uint32_t load_variant_ct, unsigned char* output,
    uint32_t raw_variant_stride) {
  PlinkRansAdapter* adapter =
      static_cast<PlinkRansAdapter*>(context);
  const uint32_t variant_span =
      variant_uidx_end - variant_uidx_start;
  const size_t packed_byte_ct =
      adapter->reader_.packed_variant_byte_ct();
  if ((!output) || (!load_variant_ct) ||
      (variant_uidx_start >= variant_uidx_end) ||
      (variant_uidx_end > adapter->reader_.variant_ct()) ||
      (raw_variant_stride < packed_byte_ct)) {
    return kPglRetImproperFunctionCall;
  }

  std::vector<uint32_t> variants;
  if (variant_include) {
    variants.reserve(load_variant_ct);
    for (uint32_t vidx = variant_uidx_start;
         vidx != variant_uidx_end; ++vidx) {
      if (IsSet(variant_include, vidx)) {
        variants.push_back(vidx);
      }
    }
    if (variants.size() != load_variant_ct) {
      return kPglRetImproperFunctionCall;
    }
  } else if (load_variant_ct != variant_span) {
    return kPglRetImproperFunctionCall;
  }

  std::string error;
  const bool dense =
      (!variant_include) || (load_variant_ct == variant_span);
  if (dense) {
    if (!adapter->reader_.ReadRange(
            variant_uidx_start, variant_span, output,
            raw_variant_stride, nullptr, &error)) {
      fprintf(
          stderr,
          "Error: Conditional-rANS PGEN batch materialization failed at "
          "variants %u-%u: %s\n",
          variant_uidx_start, variant_uidx_end - 1, error.c_str());
      return ReadFailure(error, &adapter->last_error_);
    }
    return kPglRetSuccess;
  }

  // ReadList() writes compactly at the start of output.  Scatter in reverse
  // raw-variant order so destinations can only overwrite compact records
  // which have already been moved.  This preserves PgfiMultiread()'s holes
  // without allocating a second potentially very large genotype buffer.
  if (!adapter->reader_.ReadList(
          variants.data(), load_variant_ct, output, packed_byte_ct,
          nullptr, &error)) {
    fprintf(
        stderr,
        "Error: Conditional-rANS PGEN sparse batch materialization failed "
        "at variants %u-%u: %s\n",
        variant_uidx_start, variant_uidx_end - 1, error.c_str());
    return ReadFailure(error, &adapter->last_error_);
  }
  for (uint32_t selected_idx = load_variant_ct;
       selected_idx; --selected_idx) {
    const uint32_t compact_idx = selected_idx - 1;
    const size_t destination_offset =
        static_cast<size_t>(
            variants[compact_idx] - variant_uidx_start) *
        raw_variant_stride;
    const size_t source_offset =
        static_cast<size_t>(compact_idx) * packed_byte_ct;
    memmove(output + destination_offset, output + source_offset,
            packed_byte_ct);
  }
  return kPglRetSuccess;
}

PglErr PlinkRansAdapter::GetM(
    void* context, const uintptr_t* sample_include,
    const uint32_t* sample_include_cumulative_popcounts,
    uint32_t sample_ct, uint32_t vidx, PgenVariant* pgv) {
  return static_cast<PlinkRansAdapter*>(context)->ReadMultiallelic(
      sample_include, sample_include_cumulative_popcounts, sample_ct,
      vidx, pgv);
}

PglErr PlinkRansAdapter::GetRaw(
    void* context, uint32_t vidx,
    PgenGlobalFlags read_gflags,
    uintptr_t** loadbuf_iter_ptr,
    unsigned char* loaded_vrtype_ptr) {
  return static_cast<PlinkRansAdapter*>(context)->ReadRawRecord(
      vidx, read_gflags, loadbuf_iter_ptr, loaded_vrtype_ptr);
}

}  // namespace pgen_rans
