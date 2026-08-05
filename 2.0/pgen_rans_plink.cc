// SPDX-License-Identifier: GPL-3.0-or-later

#include "pgen_rans_plink.h"

#include <cinttypes>
#include <new>
#include <vector>

#include "include/plink2_base.h"
#include "include/plink2_bits.h"
#include "pgen_rans_container.h"
#include "pgen_rans_encode.h"

namespace pgen_rans {
namespace {

using namespace plink2;

static_assert(
    kPgenRansStorageMode == kPgenDeveloperRansStorageMode,
    "Conditional-rANS PGEN storage-mode constants disagree.");

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
  if (unlikely(!adapter->Install(pgfi, pgr, &error))) {
    logerrprintfww(
        "Error: Failed to install conditional-rANS PGEN backend: %s\n",
        error.c_str());
    return kPglRetImproperFunctionCall;
  }
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

}  // namespace pgen_rans
