// SPDX-License-Identifier: GPL-3.0-or-later

#include "pgen_rans_plink.h"

#include <algorithm>
#include <array>
#include <cstring>

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

PlinkPgrAdapter::PlinkPgrAdapter() {
  backend_.context = this;
  backend_.get = &Get;
  backend_.get_allele = &GetAllele;
  backend_.get_counts = &GetCounts;
  backend_.get_packed = &GetPacked;
}

PlinkPgrAdapter::~PlinkPgrAdapter() {
  Close();
}

bool PlinkPgrAdapter::Open(const std::string& path, uint32_t thread_ct,
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
      *error = "Out of memory allocating the PGR PLINK adapter.";
    }
    Close();
    return false;
  }
  return true;
}

void PlinkPgrAdapter::Close() {
  aligned_free_cond(count_genovec_);
  aligned_free_cond(raw_genovec_);
  count_genovec_ = nullptr;
  raw_genovec_ = nullptr;
  raw_word_ct_ = 0;
  reader_.Close();
  last_error_.clear();
}

void PlinkPgrAdapter::Install(PgenFileInfo* pgfi, PgenReader* pgr) {
  pgfi->multiread_backend = &backend_;
  PgenReaderMain* pgrp = &GET_PRIVATE(*pgr, m);
  pgrp->fi = *pgfi;
  pgrp->hardcall_backend = &backend_;
}

uint32_t PlinkPgrAdapter::sample_ct() const {
  return reader_.raw_sample_ct();
}

uint32_t PlinkPgrAdapter::variant_ct() const {
  return reader_.variant_ct();
}

uint32_t PlinkPgrAdapter::max_allele_ct() const {
  return reader_.max_allele_ct();
}

bool PlinkPgrAdapter::all_nonref() const {
  return reader_.all_nonref();
}

bool PlinkPgrAdapter::has_mixed_nonref_flags() const {
  return reader_.has_mixed_nonref_flags();
}

bool PlinkPgrAdapter::variant_is_nonref(uint32_t variant) const {
  return reader_.variant_is_nonref(variant);
}

const std::string& PlinkPgrAdapter::last_error() const {
  return last_error_;
}

bool PlinkPgrAdapter::ReadRaw(uint32_t vidx) {
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

void PlinkPgrAdapter::CopySubset(
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

PglErr PlinkPgrAdapter::ReadBase(
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

PglErr PlinkPgrAdapter::ReadAllele(
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
    last_error_ = "Requested allele is outside the PGR variant.";
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

PglErr PlinkPgrAdapter::Get(
    void* context, const uintptr_t* sample_include,
    const uint32_t*, uint32_t sample_ct, uint32_t vidx,
    uintptr_t* genovec) {
  return static_cast<PlinkPgrAdapter*>(context)->ReadBase(
      sample_include, sample_ct, vidx, genovec);
}

PglErr PlinkPgrAdapter::GetAllele(
    void* context, const uintptr_t* sample_include,
    const uint32_t*, uint32_t sample_ct, uint32_t vidx,
    uint32_t allele_idx, uintptr_t* allele_countvec) {
  return static_cast<PlinkPgrAdapter*>(context)->ReadAllele(
      sample_include, sample_ct, vidx, allele_idx, allele_countvec);
}

PglErr PlinkPgrAdapter::GetCounts(
    void* context, const uintptr_t* sample_include,
    const uint32_t*, uint32_t sample_ct, uint32_t vidx,
    uint32_t* genocounts) {
  PlinkPgrAdapter* adapter =
      static_cast<PlinkPgrAdapter*>(context);
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

PglErr PlinkPgrAdapter::GetPacked(
    void* context, uint32_t vidx, unsigned char* packed_genotypes,
    uint32_t packed_byte_ct) {
  PlinkPgrAdapter* adapter =
      static_cast<PlinkPgrAdapter*>(context);
  std::string error;
  if (!adapter->reader_.ReadVariant(
          vidx, packed_genotypes, packed_byte_ct, nullptr, &error)) {
    fprintf(stderr, "Error: PGR block materialization failed at variant %u: %s\n",
            vidx, error.c_str());
    return ReadFailure(error, &adapter->last_error_);
  }
  return kPglRetSuccess;
}

}  // namespace pgen_rans
