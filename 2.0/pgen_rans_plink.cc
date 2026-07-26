// SPDX-License-Identifier: GPL-3.0-or-later

#include "pgen_rans_plink.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <vector>

#include "include/plink2_base.h"
#include "include/plink2_bits.h"
#include "pgen_rans_codec.h"

namespace pgen_rans {
namespace {

using namespace plink2;

PglErr ReadFailure(const std::string& message, std::string* last_error) {
  *last_error = message;
  return kPglRetReadFail;
}

}  // namespace

PlinkRansAdapter::PlinkRansAdapter() {
  backend_.context = this;
  backend_.get = &Get;
  backend_.get_allele = &GetAllele;
  backend_.get_counts = &GetCounts;
  backend_.get_packed = &GetPacked;
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
