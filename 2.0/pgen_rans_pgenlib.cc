// SPDX-License-Identifier: GPL-3.0-or-later

#include "pgen_rans_pgenlib.h"

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstring>
#include <vector>

#include "include/plink2_base.h"
#include "include/plink2_bits.h"
#include "pgen_rans_container.h"

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

void SetError(const char* message, std::string* error) {
  if (error) {
    *error = message;
  }
}

void SetPgenlibError(const char* operation, const char* detail,
                     std::string* error) {
  if (!error) {
    return;
  }
  *error = operation;
  if (detail && detail[0]) {
    *error += ": ";
    *error += detail;
  }
}

}  // namespace

PgenlibRansBackend::PgenlibRansBackend() : backend_{} {
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

PgenlibRansBackend::~PgenlibRansBackend() {
  Close();
}

bool PgenlibRansBackend::Open(const std::string& path, uint32_t thread_ct,
                            std::string* error) {
  if (installed_pgr_) {
    SetError(
        "Close the installed conditional-rANS backend before reopening it.",
        error);
    return false;
  }
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
  is_open_ = true;
  return true;
}

bool PgenlibRansBackend::Validate(std::string* error) {
  if (!is_open_) {
    SetError("Conditional-rANS backend is not open.", error);
    return false;
  }
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

void PgenlibRansBackend::Close() {
  if (installed_pgfi_ &&
      (installed_pgfi_->multiread_backend == &backend_)) {
    installed_pgfi_->multiread_backend = nullptr;
  }
  if (installed_pgr_) {
    PgenReaderMain* pgrp = &GET_PRIVATE(*installed_pgr_, m);
    if (pgrp->hardcall_backend == &backend_) {
      pgrp->hardcall_backend = nullptr;
    }
  }
  installed_pgfi_ = nullptr;
  installed_pgr_ = nullptr;
  aligned_free_cond(count_genovec_);
  aligned_free_cond(raw_genovec_);
  count_genovec_ = nullptr;
  raw_genovec_ = nullptr;
  raw_word_ct_ = 0;
  reader_.Close();
  is_open_ = false;
  last_error_.clear();
}

bool PgenlibRansBackend::Install(
    PgenFileInfo* pgfi, PgenReader* pgr, std::string* error) {
  if ((!is_open_) || (!pgfi) || (!pgr)) {
    SetError("Conditional-rANS backend installation is invalid.", error);
    return false;
  }
  if (installed_pgr_) {
    if ((installed_pgfi_ == pgfi) && (installed_pgr_ == pgr)) {
      return true;
    }
    SetError("Conditional-rANS backend is already installed.", error);
    return false;
  }
  if ((pgfi->raw_sample_ct != sample_ct()) ||
      (pgfi->raw_variant_ct != variant_ct()) ||
      (pgfi->max_allele_ct != max_allele_ct())) {
    SetError("Conditional-rANS backend metadata does not match pgenlib.",
             error);
    return false;
  }
  pgfi->multiread_backend = &backend_;
  PgenReaderMain* pgrp = &GET_PRIVATE(*pgr, m);
  pgrp->fi = *pgfi;
  pgrp->hardcall_backend = &backend_;
  installed_pgfi_ = pgfi;
  installed_pgr_ = pgr;
  return true;
}

uint32_t PgenlibRansBackend::sample_ct() const {
  return reader_.raw_sample_ct();
}

uint32_t PgenlibRansBackend::variant_ct() const {
  return reader_.variant_ct();
}

uint32_t PgenlibRansBackend::max_allele_ct() const {
  return reader_.max_allele_ct();
}

bool PgenlibRansBackend::all_nonref() const {
  return reader_.all_nonref();
}

bool PgenlibRansBackend::has_mixed_nonref_flags() const {
  return reader_.has_mixed_nonref_flags();
}

bool PgenlibRansBackend::variant_is_nonref(uint32_t variant) const {
  return reader_.variant_is_nonref(variant);
}

const std::string& PgenlibRansBackend::last_error() const {
  return last_error_;
}

bool PgenlibRansBackend::ReadRaw(uint32_t vidx) {
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

void PgenlibRansBackend::CopySubset(
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

PglErr PgenlibRansBackend::ReadBase(
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

PglErr PgenlibRansBackend::ReadDifflistOrGenovec(
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
  const uint32_t raw_max_difflist_len =
      (sample_ct == raw_sample_ct) ? max_difflist_len : raw_sample_ct;
  if (!reader_.ReadVariantMaybeSparse(
          vidx, raw_max_difflist_len,
          reinterpret_cast<uint8_t*>(raw_genovec_),
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

PglErr PgenlibRansBackend::ReadAllele(
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

PglErr PgenlibRansBackend::ReadMultiallelic(
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

PglErr PgenlibRansBackend::ReadRawRecord(
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

PglErr PgenlibRansBackend::Get(
    void* context, const uintptr_t* sample_include,
    const uint32_t*, uint32_t sample_ct, uint32_t vidx,
    uintptr_t* genovec) {
  return static_cast<PgenlibRansBackend*>(context)->ReadBase(
      sample_include, sample_ct, vidx, genovec);
}

PglErr PgenlibRansBackend::GetAllele(
    void* context, const uintptr_t* sample_include,
    const uint32_t*, uint32_t sample_ct, uint32_t vidx,
    uint32_t allele_idx, uintptr_t* allele_countvec) {
  return static_cast<PgenlibRansBackend*>(context)->ReadAllele(
      sample_include, sample_ct, vidx, allele_idx, allele_countvec);
}

PglErr PgenlibRansBackend::GetCounts(
    void* context, const uintptr_t* sample_include,
    const uint32_t*, uint32_t sample_ct, uint32_t vidx,
    uint32_t* genocounts) {
  PgenlibRansBackend* adapter =
      static_cast<PgenlibRansBackend*>(context);
  if ((!sample_ct) || (!genocounts) ||
      (sample_ct > adapter->reader_.raw_sample_ct()) ||
      ((sample_ct != adapter->reader_.raw_sample_ct()) &&
       (!sample_include))) {
    return kPglRetImproperFunctionCall;
  }
  std::string error;
  if (!adapter->reader_.ReadVariantMaybeSparse(
          vidx, adapter->reader_.raw_sample_ct(),
          reinterpret_cast<uint8_t*>(adapter->raw_genovec_),
          adapter->reader_.packed_variant_byte_ct(),
          &adapter->sparse_result_, nullptr, &error)) {
    return ReadFailure(error, &adapter->last_error_);
  }
  if (adapter->sparse_result_.common_genotype != UINT32_MAX) {
    std::array<uint32_t, 4> count_buffer = {};
    const uint32_t common_genotype =
        adapter->sparse_result_.common_genotype;
    if (common_genotype > 3) {
      adapter->last_error_ =
          "Conditional-rANS sparse common genotype is invalid.";
      return kPglRetMalformedInput;
    }
    count_buffer[common_genotype] = sample_ct;
    for (size_t sparse_idx = 0;
         sparse_idx != adapter->sparse_result_.sample_ids.size();
         ++sparse_idx) {
      const uint32_t sample_idx =
          adapter->sparse_result_.sample_ids[sparse_idx];
      const uint32_t genotype =
          adapter->sparse_result_.genotypes[sparse_idx];
      if ((sample_idx >= adapter->reader_.raw_sample_ct()) ||
          (genotype > 3)) {
        adapter->last_error_ =
            "Conditional-rANS sparse count entry is invalid.";
        return kPglRetMalformedInput;
      }
      if ((sample_ct != adapter->reader_.raw_sample_ct()) &&
          (!IsSet(sample_include, sample_idx))) {
        continue;
      }
      --count_buffer[common_genotype];
      ++count_buffer[genotype];
    }
    memcpy(genocounts, count_buffer.data(),
           count_buffer.size() * sizeof(uint32_t));
    return kPglRetSuccess;
  }
  ZeroTrailingNyps(adapter->reader_.raw_sample_ct(),
                   adapter->raw_genovec_);
  adapter->CopySubset(
      sample_include, sample_ct, adapter->count_genovec_);
  std::array<uint32_t, 4> count_buffer;
  GenoarrCountFreqsUnsafe(
      adapter->count_genovec_, sample_ct, count_buffer);
  memcpy(genocounts, count_buffer.data(),
         count_buffer.size() * sizeof(uint32_t));
  return kPglRetSuccess;
}

PglErr PgenlibRansBackend::GetDifflistOrGenovec(
    void* context, const uintptr_t* sample_include,
    const uint32_t* sample_include_cumulative_popcounts,
    uint32_t sample_ct, uint32_t vidx, uint32_t max_difflist_len,
    uintptr_t* genovec, uint32_t* difflist_common_geno_ptr,
    uintptr_t* main_raregeno, uint32_t* difflist_sample_ids,
    uint32_t* difflist_len_ptr) {
  return static_cast<PgenlibRansBackend*>(context)->ReadDifflistOrGenovec(
      sample_include, sample_include_cumulative_popcounts, sample_ct, vidx,
      max_difflist_len, genovec, difflist_common_geno_ptr, main_raregeno,
      difflist_sample_ids, difflist_len_ptr);
}

PglErr PgenlibRansBackend::GetPacked(
    void* context, uint32_t vidx, unsigned char* packed_genotypes,
    uint32_t packed_byte_ct) {
  PgenlibRansBackend* adapter =
      static_cast<PgenlibRansBackend*>(context);
  std::string error;
  if (!adapter->reader_.ReadVariant(
          vidx, packed_genotypes, packed_byte_ct, nullptr, &error)) {
    return ReadFailure(error, &adapter->last_error_);
  }
  return kPglRetSuccess;
}

PglErr PgenlibRansBackend::GetPackedBatch(
    void* context, const uintptr_t* variant_include,
    uint32_t variant_uidx_start, uint32_t variant_uidx_end,
    uint32_t load_variant_ct, unsigned char* output,
    uint32_t raw_variant_stride) {
  PgenlibRansBackend* adapter =
      static_cast<PgenlibRansBackend*>(context);
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

PglErr PgenlibRansBackend::GetM(
    void* context, const uintptr_t* sample_include,
    const uint32_t* sample_include_cumulative_popcounts,
    uint32_t sample_ct, uint32_t vidx, PgenVariant* pgv) {
  return static_cast<PgenlibRansBackend*>(context)->ReadMultiallelic(
      sample_include, sample_include_cumulative_popcounts, sample_ct,
      vidx, pgv);
}

PglErr PgenlibRansBackend::GetRaw(
    void* context, uint32_t vidx,
    PgenGlobalFlags read_gflags,
    uintptr_t** loadbuf_iter_ptr,
    unsigned char* loaded_vrtype_ptr) {
  return static_cast<PgenlibRansBackend*>(context)->ReadRawRecord(
      vidx, read_gflags, loadbuf_iter_ptr, loaded_vrtype_ptr);
}

UnifiedPgenReader::UnifiedPgenReader() {
  PreinitPgfi(&pgfi_);
  PreinitPgr(&pgr_);
}

UnifiedPgenReader::~UnifiedPgenReader() {
  Close();
}

bool UnifiedPgenReader::Open(
    const std::string& path, const UnifiedPgenOpenOptions& options,
    std::string* error) {
  Close();
  if (path.empty() || !options.thread_ct) {
    SetError("Invalid unified PGEN open request.", error);
    return false;
  }

  PgenHeaderCtrl header_ctrl = 0;
  uintptr_t pgfi_alloc_cacheline_ct = 0;
  char error_buffer[kPglErrstrBufBlen] = {};
  const PglErr reterr = PgfiInitPhase1(
      path.c_str(), nullptr, UINT32_MAX, UINT32_MAX, &header_ctrl,
      &pgfi_, &pgfi_alloc_cacheline_ct, error_buffer);
  if (reterr != kPglRetSuccess) {
    SetPgenlibError("PgfiInitPhase1", error_buffer, error);
    Close();
    return false;
  }
  if (pgfi_.const_fpos_offset == UINT64_MAX) {
    if (!OpenRans(path, options, error)) {
      Close();
      return false;
    }
    return true;
  }
  if (!OpenStandard(
          path, header_ctrl, pgfi_alloc_cacheline_ct, error)) {
    Close();
    return false;
  }
  return true;
}

bool UnifiedPgenReader::OpenStandard(
    const std::string& path, PgenHeaderCtrl header_ctrl,
    uintptr_t pgfi_alloc_cacheline_ct, std::string* error) {
  if (pgfi_alloc_cacheline_ct &&
      cachealigned_malloc(pgfi_alloc_cacheline_ct * kCacheline,
                          &pgfi_alloc_)) {
    SetError("Out of memory allocating the PGEN file index.", error);
    return false;
  }
  uint32_t max_vrec_width = 0;
  uintptr_t pgr_alloc_cacheline_ct = 0;
  char error_buffer[kPglErrstrBufBlen] = {};
  PglErr reterr = PgfiInitPhase2(
      header_ctrl, 0, 0, 0, 0, pgfi_.raw_variant_ct, &max_vrec_width,
      &pgfi_, pgfi_alloc_, &pgr_alloc_cacheline_ct, error_buffer);
  if (reterr != kPglRetSuccess) {
    SetPgenlibError("PgfiInitPhase2", error_buffer, error);
    return false;
  }
  if (pgr_alloc_cacheline_ct &&
      cachealigned_malloc(pgr_alloc_cacheline_ct * kCacheline,
                          &pgr_alloc_)) {
    SetError("Out of memory allocating the PGEN reader.", error);
    return false;
  }
  reterr = PgrInit(
      path.c_str(), max_vrec_width, &pgfi_, &pgr_, pgr_alloc_);
  if (reterr != kPglRetSuccess) {
    SetPgenlibError("PgrInit", nullptr, error);
    return false;
  }
  storage_mode_ = PgenStorageMode::kStandard;
  is_open_ = true;
  return true;
}

bool UnifiedPgenReader::OpenRans(
    const std::string& path, const UnifiedPgenOpenOptions& options,
    std::string* error) {
  if (pgfi_.shared_ff) {
    if (fclose(pgfi_.shared_ff)) {
      pgfi_.shared_ff = nullptr;
      SetError("Failed to close the conditional-rANS header stream.", error);
      return false;
    }
    pgfi_.shared_ff = nullptr;
  }
  pgfi_.pgi_ff = nullptr;
  const uint32_t phase1_sample_ct = pgfi_.raw_sample_ct;
  const uint32_t phase1_variant_ct = pgfi_.raw_variant_ct;
  if (!rans_backend_.Open(path, options.thread_ct, error)) {
    return false;
  }
  if ((rans_backend_.sample_ct() != phase1_sample_ct) ||
      (rans_backend_.variant_ct() != phase1_variant_ct)) {
    SetError("Conditional-rANS header counts disagree.", error);
    return false;
  }

  const uint32_t sample_ct = rans_backend_.sample_ct();
  const uint32_t variant_ct = rans_backend_.variant_ct();
  const uint32_t max_allele_ct = rans_backend_.max_allele_ct();
  allele_idx_offsets_.clear();
  if (max_allele_ct > 2) {
    if (!options.allele_idx_offsets) {
      SetError(
          "Multiallelic conditional-rANS input requires PVAR allele "
          "offsets.", error);
      return false;
    }
    allele_idx_offsets_.assign(
        options.allele_idx_offsets,
        options.allele_idx_offsets + variant_ct + 1);
    uint32_t observed_max_allele_ct = 0;
    for (uint32_t vidx = 0; vidx != variant_ct; ++vidx) {
      const uintptr_t begin = allele_idx_offsets_[vidx];
      const uintptr_t end = allele_idx_offsets_[vidx + 1];
      if (end < begin) {
        SetError("Conditional-rANS PVAR allele offsets are invalid.", error);
        return false;
      }
      const uintptr_t allele_ct = end - begin;
      if ((allele_ct < 2) || (allele_ct > 255)) {
        SetError("Conditional-rANS PVAR allele offsets are invalid.", error);
        return false;
      }
      observed_max_allele_ct = std::max<uint32_t>(
          observed_max_allele_ct, static_cast<uint32_t>(allele_ct));
    }
    if (observed_max_allele_ct != max_allele_ct) {
      SetError(
          "Conditional-rANS PVAR allele counts disagree with the container.",
          error);
      return false;
    }
  }
  nonref_flags_.clear();
  if (rans_backend_.has_mixed_nonref_flags()) {
    nonref_flags_.assign(BitCtToWordCt(variant_ct), 0);
    for (uint32_t vidx = 0; vidx != variant_ct; ++vidx) {
      if (rans_backend_.variant_is_nonref(vidx)) {
        SetBit(vidx, nonref_flags_.data());
      }
    }
  }

  pgfi_.raw_variant_ct = variant_ct;
  pgfi_.raw_sample_ct = sample_ct;
  pgfi_.const_fpos_offset = 0;
  pgfi_.const_vrec_width = DivUp(sample_ct, 4);
  pgfi_.const_vrtype = 0;
  pgfi_.var_fpos = nullptr;
  pgfi_.vrtypes = nullptr;
  pgfi_.allele_idx_offsets =
      allele_idx_offsets_.empty() ? nullptr : allele_idx_offsets_.data();
  pgfi_.nonref_flags =
      nonref_flags_.empty() ? nullptr : nonref_flags_.data();
  pgfi_.gflags =
      (max_allele_ct > 2)
          ? kfPgenGlobalMultiallelicHardcallFound
          : kfPgenGlobal0;
  if (rans_backend_.all_nonref()) {
    pgfi_.gflags |= kfPgenGlobalAllNonref;
  }
  pgfi_.max_allele_ct = max_allele_ct;
  pgfi_.extensions_present = 0;
  pgfi_.block_base = nullptr;
  pgfi_.block_offset = 0;
  if (!rans_backend_.Install(&pgfi_, &pgr_, error)) {
    return false;
  }
  storage_mode_ = PgenStorageMode::kConditionalRans;
  is_open_ = true;
  return true;
}

void UnifiedPgenReader::Close() {
  PglErr cleanup_error = kPglRetSuccess;
  CleanupPgr(&pgr_, &cleanup_error);
  CleanupPgfi(&pgfi_, &cleanup_error);
  aligned_free_cond(pgr_alloc_);
  aligned_free_cond(pgfi_alloc_);
  pgr_alloc_ = nullptr;
  pgfi_alloc_ = nullptr;
  rans_backend_.Close();
  allele_idx_offsets_.clear();
  nonref_flags_.clear();
  PreinitPgfi(&pgfi_);
  PreinitPgr(&pgr_);
  pgfi_.raw_sample_ct = 0;
  pgfi_.raw_variant_ct = 0;
  pgfi_.max_allele_ct = 0;
  storage_mode_ = PgenStorageMode::kStandard;
  is_open_ = false;
}

}  // namespace pgen_rans
