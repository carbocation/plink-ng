// SPDX-License-Identifier: GPL-3.0-or-later

#include "pgen_rans_encode.h"

#include <algorithm>
#include <array>
#ifdef PGEN_RANS_VERIFY_FAST_COUNTS
#include <cassert>
#endif
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <limits>
#include <mutex>
#include <string>
#include <sys/stat.h>
#include <thread>
#include <vector>

#include "include/plink2_base.h"
#include "include/plink2_bits.h"
#include "include/pgenlib_misc.h"
#include "pgen_rans_codec.h"
#include "pgen_rans_container.h"
#include "pgen_rans_hybrid.h"

namespace pgen_rans {
namespace {

using namespace plink2;

static_assert(sizeof(uintptr_t) == sizeof(uint64_t),
              "The experimental codec currently requires 64-bit words.");

struct VariantBlock {
  uint32_t start = 0;
  uint32_t len = 0;
};

struct AnchorCandidate {
  uint64_t estimated_bytes = 0;
  uint64_t sparse_exception_ct = 0;
  uint32_t ordinal = 0;
  uint32_t offset = 0;
  std::array<uint32_t, 16> joint_counts = {};
};

struct PackedGenotypePlanes {
  uint32_t word_ct = 0;
  uint8_t baseline = 0;
  uint8_t independent_ct = 0;
  std::array<uint8_t, 3> independent = {};
  std::vector<uintptr_t> storage;

  const uintptr_t* Plane(uint32_t plane_idx) const {
    return &(storage[static_cast<size_t>(plane_idx) * word_ct]);
  }
};

std::vector<VariantBlock> BuildBlocks(
    uint32_t variant_ct, uint32_t maximum_block_variant_ct,
    const VariantMetadata* metadata) {
  std::vector<VariantBlock> blocks;
  uint32_t block_start = 0;
  while (block_start != variant_ct) {
    uint32_t block_len =
        std::min(maximum_block_variant_ct, variant_ct - block_start);
    if (metadata) {
      const uint32_t chrom_code = metadata[block_start].chrom_code;
      for (uint32_t offset = 1; offset != block_len; ++offset) {
        if (metadata[block_start + offset].chrom_code != chrom_code) {
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
                    const VariantMetadata* metadata,
                    uint64_t maximum_distance) {
  if (!metadata) {
    return true;
  }
  const VariantMetadata& target = metadata[target_vidx];
  const VariantMetadata& anchor = metadata[anchor_vidx];
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

void CopyPgenPatches(const PgenVariant& pgv, uint16_t allele_ct,
                     MultiallelicPatches* patches) {
  patches->allele_ct = allele_ct;
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
}

void BuildGenotypeMasks(uintptr_t genotype_word, uintptr_t valid_mask,
                        uintptr_t* masks) {
  const uintptr_t low = genotype_word & valid_mask;
  const uintptr_t high = (genotype_word >> 1) & valid_mask;
  masks[0] = (~(low | high)) & valid_mask;
  masks[1] = low & (~high) & valid_mask;
  masks[2] = (~low) & high & valid_mask;
  masks[3] = low & high;
}

uint32_t MostFrequentGenotype(const uint32_t* counts) {
  uint32_t result = 0;
  for (uint32_t genotype = 1; genotype != 4; ++genotype) {
    if (counts[genotype] > counts[result]) {
      result = genotype;
    }
  }
  return result;
}

uint32_t IndependentGenotypes(const uint32_t* counts,
                              uint32_t baseline,
                              uint8_t* independent) {
  uint32_t independent_ct = 0;
  for (uint32_t genotype = 0; genotype != 4; ++genotype) {
    if ((genotype != baseline) && counts[genotype]) {
      independent[independent_ct++] =
          static_cast<uint8_t>(genotype);
    }
  }
  return independent_ct;
}

uintptr_t InverseMatchWord(uint32_t genotype) {
  if (!genotype) {
    return ~static_cast<uintptr_t>(0);
  }
  if (genotype == 1) {
    return kMaskAAAA;
  }
  if (genotype == 2) {
    return kMask5555;
  }
  return 0;
}

void BuildPackedGenotypePlanes(
    const uintptr_t* genovec, const uint32_t* counts,
    uint32_t sample_ct, PackedGenotypePlanes* planes) {
  planes->baseline = static_cast<uint8_t>(
      MostFrequentGenotype(counts));
  planes->independent_ct = static_cast<uint8_t>(
      IndependentGenotypes(
          counts, planes->baseline, planes->independent.data()));
  planes->word_ct = BitCtToWordCt(sample_ct);
  planes->storage.resize(
      static_cast<size_t>(planes->independent_ct) * planes->word_ct);
  const uint32_t genovec_word_ct = NypCtToWordCt(sample_ct);
  for (uint32_t plane_idx = 0;
       plane_idx != planes->independent_ct; ++plane_idx) {
    uintptr_t* plane =
        &(planes->storage[
            static_cast<size_t>(plane_idx) * planes->word_ct]);
    std::fill(plane, &(plane[planes->word_ct]), 0);
    PackWordsToHalfwordsInvmatch(
        genovec,
        InverseMatchWord(planes->independent[plane_idx]),
        genovec_word_ct, plane);
    ZeroTrailingBits(sample_ct, plane);
  }
}

void ReconstructJointGenotypes(
    const uint32_t* anchor_counts, const uint32_t* target_counts,
    uint32_t anchor_baseline, uint32_t target_baseline,
    uint32_t* joint_counts) {
  for (uint32_t anchor_genotype = 0; anchor_genotype != 4;
       ++anchor_genotype) {
    if (anchor_genotype == anchor_baseline) {
      continue;
    }
    uint32_t known = 0;
    for (uint32_t target_genotype = 0; target_genotype != 4;
         ++target_genotype) {
      if (target_genotype != target_baseline) {
        known +=
            joint_counts[4 * anchor_genotype + target_genotype];
      }
    }
    joint_counts[4 * anchor_genotype + target_baseline] =
        anchor_counts[anchor_genotype] - known;
  }
  for (uint32_t target_genotype = 0; target_genotype != 4;
       ++target_genotype) {
    if (target_genotype == target_baseline) {
      continue;
    }
    uint32_t known = 0;
    for (uint32_t anchor_genotype = 0; anchor_genotype != 4;
         ++anchor_genotype) {
      if (anchor_genotype != anchor_baseline) {
        known +=
            joint_counts[4 * anchor_genotype + target_genotype];
      }
    }
    joint_counts[4 * anchor_baseline + target_genotype] =
        target_counts[target_genotype] - known;
  }
  uint32_t known = 0;
  for (uint32_t target_genotype = 0; target_genotype != 4;
       ++target_genotype) {
    if (target_genotype != target_baseline) {
      known +=
          joint_counts[4 * anchor_baseline + target_genotype];
    }
  }
  joint_counts[4 * anchor_baseline + target_baseline] =
      anchor_counts[anchor_baseline] - known;
}

#ifdef PGEN_RANS_VERIFY_FAST_COUNTS
void CountJointGenotypesReference(
    const uintptr_t* anchor, const uintptr_t* target,
    uint32_t sample_ct, const uint32_t* anchor_counts,
    const uint32_t* target_counts, uint32_t* joint_counts) {
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

void CountTripleGenotypesReference(
    const uintptr_t* anchor1, const uintptr_t* anchor2,
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
#endif

void CountJointGenotypes(const uintptr_t* anchor, const uintptr_t* target,
                         uint32_t sample_ct, const uint32_t* anchor_counts,
                         const uint32_t* target_counts,
                         uint32_t* joint_counts) {
  std::fill(joint_counts, &(joint_counts[16]), 0U);
  const uint32_t anchor_baseline =
      MostFrequentGenotype(anchor_counts);
  const uint32_t target_baseline =
      MostFrequentGenotype(target_counts);
  uint8_t independent_anchors[3];
  uint8_t independent_targets[3];
  const uint32_t independent_anchor_ct =
      IndependentGenotypes(anchor_counts, anchor_baseline,
                           independent_anchors);
  const uint32_t independent_target_ct =
      IndependentGenotypes(target_counts, target_baseline,
                           independent_targets);
  const uint32_t word_ct = NypCtToWordCt(sample_ct);
  const uint32_t trailing_sample_ct = sample_ct % kBitsPerWordD2;
  for (uint32_t word_idx = 0; word_idx != word_ct; ++word_idx) {
    uintptr_t valid_mask = kMask5555;
    if (trailing_sample_ct && (word_idx + 1 == word_ct)) {
      valid_mask = bzhi(kMask5555, 2 * trailing_sample_ct);
    }
    uintptr_t anchor_masks[4];
    uintptr_t target_masks[4];
    BuildGenotypeMasks(anchor[word_idx], valid_mask, anchor_masks);
    BuildGenotypeMasks(target[word_idx], valid_mask, target_masks);
    for (uint32_t anchor_idx = 0;
         anchor_idx != independent_anchor_ct; ++anchor_idx) {
      const uint32_t anchor_genotype =
          independent_anchors[anchor_idx];
      for (uint32_t target_idx = 0;
           target_idx != independent_target_ct; ++target_idx) {
        const uint32_t target_genotype =
            independent_targets[target_idx];
        joint_counts[4 * anchor_genotype + target_genotype] +=
            PopcountWord(anchor_masks[anchor_genotype] &
                         target_masks[target_genotype]);
      }
    }
  }
  ReconstructJointGenotypes(
      anchor_counts, target_counts, anchor_baseline, target_baseline,
      joint_counts);
#ifdef PGEN_RANS_VERIFY_FAST_COUNTS
  uint32_t reference_counts[16];
  CountJointGenotypesReference(
      anchor, target, sample_ct, anchor_counts, target_counts,
      reference_counts);
  assert(std::equal(
      joint_counts, &(joint_counts[16]), reference_counts));
#endif
}

void CountJointGenotypesPacked(
    const PackedGenotypePlanes& anchor,
    const PackedGenotypePlanes& target,
    const uint32_t* anchor_counts, const uint32_t* target_counts,
    uint32_t* joint_counts) {
  std::fill(joint_counts, &(joint_counts[16]), 0U);
  for (uint32_t anchor_idx = 0;
       anchor_idx != anchor.independent_ct; ++anchor_idx) {
    const uintptr_t* anchor_plane = anchor.Plane(anchor_idx);
    const uint32_t anchor_genotype =
        anchor.independent[anchor_idx];
    for (uint32_t target_idx = 0;
         target_idx != target.independent_ct; ++target_idx) {
      const uintptr_t* target_plane = target.Plane(target_idx);
      uint32_t count = 0;
      for (uint32_t word_idx = 0; word_idx != anchor.word_ct;
           ++word_idx) {
        count += PopcountWord(
            anchor_plane[word_idx] & target_plane[word_idx]);
      }
      joint_counts[
          4 * anchor_genotype + target.independent[target_idx]] =
          count;
    }
  }
  ReconstructJointGenotypes(
      anchor_counts, target_counts, anchor.baseline, target.baseline,
      joint_counts);
}

void ReconstructTripleGenotypes(
    const uint32_t* anchor_pair_counts,
    const uint32_t* anchor1_target_counts,
    const uint32_t* anchor2_target_counts,
    uint32_t anchor1_baseline, uint32_t anchor2_baseline,
    uint32_t target_baseline, uint32_t* triple_counts) {
  for (uint32_t anchor1_genotype = 0; anchor1_genotype != 4;
       ++anchor1_genotype) {
    if (anchor1_genotype == anchor1_baseline) {
      continue;
    }
    for (uint32_t anchor2_genotype = 0; anchor2_genotype != 4;
         ++anchor2_genotype) {
      if (anchor2_genotype == anchor2_baseline) {
        continue;
      }
      uint32_t known = 0;
      for (uint32_t target_genotype = 0; target_genotype != 4;
           ++target_genotype) {
        if (target_genotype != target_baseline) {
          known += triple_counts[
              16 * anchor1_genotype + 4 * anchor2_genotype +
              target_genotype];
        }
      }
      triple_counts[
          16 * anchor1_genotype + 4 * anchor2_genotype +
          target_baseline] =
          anchor_pair_counts[
              4 * anchor1_genotype + anchor2_genotype] -
          known;
    }
  }
  for (uint32_t anchor1_genotype = 0; anchor1_genotype != 4;
       ++anchor1_genotype) {
    if (anchor1_genotype == anchor1_baseline) {
      continue;
    }
    for (uint32_t target_genotype = 0; target_genotype != 4;
         ++target_genotype) {
      if (target_genotype == target_baseline) {
        continue;
      }
      uint32_t known = 0;
      for (uint32_t anchor2_genotype = 0; anchor2_genotype != 4;
           ++anchor2_genotype) {
        if (anchor2_genotype != anchor2_baseline) {
          known += triple_counts[
              16 * anchor1_genotype + 4 * anchor2_genotype +
              target_genotype];
        }
      }
      triple_counts[
          16 * anchor1_genotype + 4 * anchor2_baseline +
          target_genotype] =
          anchor1_target_counts[
              4 * anchor1_genotype + target_genotype] -
          known;
    }
  }
  for (uint32_t anchor2_genotype = 0; anchor2_genotype != 4;
       ++anchor2_genotype) {
    if (anchor2_genotype == anchor2_baseline) {
      continue;
    }
    for (uint32_t target_genotype = 0; target_genotype != 4;
         ++target_genotype) {
      if (target_genotype == target_baseline) {
        continue;
      }
      uint32_t known = 0;
      for (uint32_t anchor1_genotype = 0; anchor1_genotype != 4;
           ++anchor1_genotype) {
        if (anchor1_genotype != anchor1_baseline) {
          known += triple_counts[
              16 * anchor1_genotype + 4 * anchor2_genotype +
              target_genotype];
        }
      }
      triple_counts[
          16 * anchor1_baseline + 4 * anchor2_genotype +
          target_genotype] =
          anchor2_target_counts[
              4 * anchor2_genotype + target_genotype] -
          known;
    }
  }
  for (uint32_t anchor1_genotype = 0; anchor1_genotype != 4;
       ++anchor1_genotype) {
    if (anchor1_genotype == anchor1_baseline) {
      continue;
    }
    uint32_t known = 0;
    for (uint32_t target_genotype = 0; target_genotype != 4;
         ++target_genotype) {
      if (target_genotype != target_baseline) {
        known += triple_counts[
            16 * anchor1_genotype + 4 * anchor2_baseline +
            target_genotype];
      }
    }
    triple_counts[
        16 * anchor1_genotype + 4 * anchor2_baseline +
        target_baseline] =
        anchor_pair_counts[
            4 * anchor1_genotype + anchor2_baseline] -
        known;
  }
  for (uint32_t anchor2_genotype = 0; anchor2_genotype != 4;
       ++anchor2_genotype) {
    if (anchor2_genotype == anchor2_baseline) {
      continue;
    }
    uint32_t known = 0;
    for (uint32_t target_genotype = 0; target_genotype != 4;
         ++target_genotype) {
      if (target_genotype != target_baseline) {
        known += triple_counts[
            16 * anchor1_baseline + 4 * anchor2_genotype +
            target_genotype];
      }
    }
    triple_counts[
        16 * anchor1_baseline + 4 * anchor2_genotype +
        target_baseline] =
        anchor_pair_counts[
            4 * anchor1_baseline + anchor2_genotype] -
        known;
  }
  for (uint32_t target_genotype = 0; target_genotype != 4;
       ++target_genotype) {
    if (target_genotype == target_baseline) {
      continue;
    }
    uint32_t known = 0;
    for (uint32_t anchor2_genotype = 0; anchor2_genotype != 4;
         ++anchor2_genotype) {
      if (anchor2_genotype != anchor2_baseline) {
        known += triple_counts[
            16 * anchor1_baseline + 4 * anchor2_genotype +
            target_genotype];
      }
    }
    triple_counts[
        16 * anchor1_baseline + 4 * anchor2_baseline +
        target_genotype] =
        anchor1_target_counts[
            4 * anchor1_baseline + target_genotype] -
        known;
  }
  uint32_t known = 0;
  for (uint32_t target_genotype = 0; target_genotype != 4;
       ++target_genotype) {
    if (target_genotype != target_baseline) {
      known += triple_counts[
          16 * anchor1_baseline + 4 * anchor2_baseline +
          target_genotype];
    }
  }
  triple_counts[
      16 * anchor1_baseline + 4 * anchor2_baseline +
      target_baseline] =
      anchor_pair_counts[
          4 * anchor1_baseline + anchor2_baseline] -
      known;
}

void CountTripleGenotypes(
    const uintptr_t* anchor1, const uintptr_t* anchor2,
    const uintptr_t* target, uint32_t sample_ct,
    const uint32_t* anchor1_counts, const uint32_t* anchor2_counts,
    const uint32_t* target_counts, const uint32_t* anchor_pair_counts,
    const uint32_t* anchor1_target_counts,
    const uint32_t* anchor2_target_counts, uint32_t* triple_counts) {
  std::fill(triple_counts, &(triple_counts[64]), 0U);
  const uint32_t anchor1_baseline =
      MostFrequentGenotype(anchor1_counts);
  const uint32_t anchor2_baseline =
      MostFrequentGenotype(anchor2_counts);
  const uint32_t target_baseline =
      MostFrequentGenotype(target_counts);
  uint8_t independent_anchor1[3];
  uint8_t independent_anchor2[3];
  uint8_t independent_targets[3];
  const uint32_t independent_anchor1_ct =
      IndependentGenotypes(anchor1_counts, anchor1_baseline,
                           independent_anchor1);
  const uint32_t independent_anchor2_ct =
      IndependentGenotypes(anchor2_counts, anchor2_baseline,
                           independent_anchor2);
  const uint32_t independent_target_ct =
      IndependentGenotypes(target_counts, target_baseline,
                           independent_targets);
  const uint32_t word_ct = NypCtToWordCt(sample_ct);
  const uint32_t trailing_sample_ct = sample_ct % kBitsPerWordD2;
  for (uint32_t word_idx = 0; word_idx != word_ct; ++word_idx) {
    uintptr_t valid_mask = kMask5555;
    if (trailing_sample_ct && (word_idx + 1 == word_ct)) {
      valid_mask = bzhi(kMask5555, 2 * trailing_sample_ct);
    }
    uintptr_t masks[3][4];
    BuildGenotypeMasks(anchor1[word_idx], valid_mask, masks[0]);
    BuildGenotypeMasks(anchor2[word_idx], valid_mask, masks[1]);
    BuildGenotypeMasks(target[word_idx], valid_mask, masks[2]);
    for (uint32_t anchor1_idx = 0;
         anchor1_idx != independent_anchor1_ct; ++anchor1_idx) {
      const uint32_t anchor1_genotype =
          independent_anchor1[anchor1_idx];
      for (uint32_t anchor2_idx = 0;
           anchor2_idx != independent_anchor2_ct; ++anchor2_idx) {
        const uint32_t anchor2_genotype =
            independent_anchor2[anchor2_idx];
        const uintptr_t references =
            masks[0][anchor1_genotype] &
            masks[1][anchor2_genotype];
        for (uint32_t target_idx = 0;
             target_idx != independent_target_ct; ++target_idx) {
          const uint32_t target_genotype =
              independent_targets[target_idx];
          triple_counts[
              16 * anchor1_genotype + 4 * anchor2_genotype +
              target_genotype] +=
              PopcountWord(references & masks[2][target_genotype]);
        }
      }
    }
  }
  ReconstructTripleGenotypes(
      anchor_pair_counts, anchor1_target_counts,
      anchor2_target_counts, anchor1_baseline, anchor2_baseline,
      target_baseline, triple_counts);
#ifdef PGEN_RANS_VERIFY_FAST_COUNTS
  uint32_t reference_counts[64];
  CountTripleGenotypesReference(
      anchor1, anchor2, target, sample_ct, reference_counts);
  assert(std::equal(
      triple_counts, &(triple_counts[64]), reference_counts));
#endif
}

void CountTripleGenotypesPacked(
    const PackedGenotypePlanes& anchor1,
    const PackedGenotypePlanes& anchor2,
    const PackedGenotypePlanes& target,
    const uint32_t* anchor_pair_counts,
    const uint32_t* anchor1_target_counts,
    const uint32_t* anchor2_target_counts,
    uint32_t* triple_counts) {
  std::fill(triple_counts, &(triple_counts[64]), 0U);
  const uintptr_t* anchor1_planes[3] = {};
  const uintptr_t* anchor2_planes[3] = {};
  const uintptr_t* target_planes[3] = {};
  for (uint32_t anchor1_idx = 0;
       anchor1_idx != anchor1.independent_ct; ++anchor1_idx) {
    anchor1_planes[anchor1_idx] = anchor1.Plane(anchor1_idx);
  }
  for (uint32_t anchor2_idx = 0;
       anchor2_idx != anchor2.independent_ct; ++anchor2_idx) {
    anchor2_planes[anchor2_idx] = anchor2.Plane(anchor2_idx);
  }
  for (uint32_t target_idx = 0;
       target_idx != target.independent_ct; ++target_idx) {
    target_planes[target_idx] = target.Plane(target_idx);
  }
  uint32_t core_counts[27] = {};
  for (uint32_t word_idx = 0; word_idx != anchor1.word_ct;
       ++word_idx) {
    uintptr_t anchor1_words[3];
    uintptr_t anchor2_words[3];
    uintptr_t target_words[3];
    for (uint32_t anchor1_idx = 0;
         anchor1_idx != anchor1.independent_ct; ++anchor1_idx) {
      anchor1_words[anchor1_idx] =
          anchor1_planes[anchor1_idx][word_idx];
    }
    for (uint32_t anchor2_idx = 0;
         anchor2_idx != anchor2.independent_ct; ++anchor2_idx) {
      anchor2_words[anchor2_idx] =
          anchor2_planes[anchor2_idx][word_idx];
    }
    for (uint32_t target_idx = 0;
         target_idx != target.independent_ct; ++target_idx) {
      target_words[target_idx] = target_planes[target_idx][word_idx];
    }
    for (uint32_t anchor1_idx = 0;
         anchor1_idx != anchor1.independent_ct; ++anchor1_idx) {
      for (uint32_t anchor2_idx = 0;
           anchor2_idx != anchor2.independent_ct; ++anchor2_idx) {
        const uintptr_t references =
            anchor1_words[anchor1_idx] & anchor2_words[anchor2_idx];
        for (uint32_t target_idx = 0;
             target_idx != target.independent_ct; ++target_idx) {
          core_counts[
              9 * anchor1_idx + 3 * anchor2_idx + target_idx] +=
              PopcountWord(references & target_words[target_idx]);
        }
      }
    }
  }
  for (uint32_t anchor1_idx = 0;
       anchor1_idx != anchor1.independent_ct; ++anchor1_idx) {
    const uint32_t anchor1_genotype =
        anchor1.independent[anchor1_idx];
    for (uint32_t anchor2_idx = 0;
         anchor2_idx != anchor2.independent_ct; ++anchor2_idx) {
      const uint32_t anchor2_genotype =
          anchor2.independent[anchor2_idx];
      for (uint32_t target_idx = 0;
           target_idx != target.independent_ct; ++target_idx) {
        triple_counts[
            16 * anchor1_genotype + 4 * anchor2_genotype +
            target.independent[target_idx]] =
            core_counts[
                9 * anchor1_idx + 3 * anchor2_idx + target_idx];
      }
    }
  }
  ReconstructTripleGenotypes(
      anchor_pair_counts, anchor1_target_counts,
      anchor2_target_counts, anchor1.baseline, anchor2.baseline,
      target.baseline, triple_counts);
}

uint64_t SparseExceptionCt(const uint32_t* counts, RecordMode mode) {
  const uint32_t row_ct =
      (mode == RecordMode::kMarginal)
          ? 1
          : ((mode == RecordMode::kOneReference) ? 4 : 16);
  uint64_t exception_ct = 0;
  for (uint32_t context = 0; context != row_ct; ++context) {
    uint32_t row_total = 0;
    uint32_t largest_count = 0;
    for (uint32_t symbol = 0; symbol != 4; ++symbol) {
      const uint32_t count = counts[4 * context + symbol];
      row_total += count;
      largest_count = std::max(largest_count, count);
    }
    exception_ct += row_total - largest_count;
  }
  return exception_ct;
}

uint32_t VarintByteCt(uint64_t value) {
  uint32_t byte_ct = 1;
  while (value >= 0x80U) {
    ++byte_ct;
    value >>= 7;
  }
  return byte_ct;
}

uint64_t SparseRecordLowerBoundByteCt(RecordMode mode,
                                      uint64_t exception_ct,
                                      uint32_t sample_ct) {
  const uint32_t row_ct =
      (mode == RecordMode::kMarginal)
          ? 1
          : ((mode == RecordMode::kOneReference) ? 4 : 16);
  const uint32_t selector_byte_ct =
      static_cast<uint32_t>(mode);
  const uint64_t minimum_id_byte_ct =
      std::min<uint64_t>(
          exception_ct, (static_cast<uint64_t>(sample_ct) + 7) / 8);
  return 1 + selector_byte_ct + (row_ct + 3) / 4 +
         VarintByteCt(exception_ct) + minimum_id_byte_ct +
         (exception_ct + 3) / 4;
}

void KeepSmallerRecord(std::vector<uint8_t>* candidate,
                       std::vector<uint8_t>* best) {
  if (candidate->size() < best->size()) {
    *best = std::move(*candidate);
  }
}

bool EncodeVariant(const uintptr_t* target,
                   const std::array<uint32_t, 4>& target_counts,
                   uint32_t target_vidx, uint32_t block_start,
                   const uintptr_t* block_genovecs,
                   uint32_t genovec_word_stride, uint32_t sample_ct,
                   const std::vector<uint32_t>& anchor_offsets,
                   const std::vector<std::array<uint32_t, 4>>& counts,
                   const std::vector<PackedGenotypePlanes>*
                       anchor_plane_cache,
                   PackedGenotypePlanes* target_plane_scratch,
                   const VariantMetadata* metadata,
                   const EncodeParams& params,
                   const CodecParams& codec_params, bool is_anchor,
                   std::vector<uint8_t>* output, std::string* error) {
  const auto* target64 = reinterpret_cast<const uint64_t*>(target);
  uint64_t marginal_estimate;
  if (!EstimateRecordBytes(
          target_counts.data(), RecordMode::kMarginal, codec_params,
          &marginal_estimate, error)) {
    return false;
  }

  const bool use_packed_planes =
      (!is_anchor) && anchor_plane_cache && target_plane_scratch;
  std::vector<AnchorCandidate> candidates;
  if (!is_anchor) {
    if (use_packed_planes) {
      BuildPackedGenotypePlanes(
          target, target_counts.data(), sample_ct,
          target_plane_scratch);
    }
    candidates.reserve(anchor_offsets.size());
    for (uint32_t anchor_ordinal = 0;
         anchor_ordinal != anchor_offsets.size(); ++anchor_ordinal) {
      const uint32_t anchor_offset = anchor_offsets[anchor_ordinal];
      const uint32_t anchor_vidx = block_start + anchor_offset;
      if (!AnchorEligible(target_vidx, anchor_vidx, metadata,
                          params.max_anchor_bp)) {
        continue;
      }
      const uintptr_t* anchor =
          &(block_genovecs[static_cast<uintptr_t>(anchor_offset) *
                            genovec_word_stride]);
      uint32_t joint_counts[16];
      if (use_packed_planes) {
        CountJointGenotypesPacked(
            (*anchor_plane_cache)[anchor_ordinal],
            *target_plane_scratch, counts[anchor_offset].data(),
            target_counts.data(), joint_counts);
      } else {
        CountJointGenotypes(
            anchor, target, sample_ct, counts[anchor_offset].data(),
            target_counts.data(), joint_counts);
      }
      uint64_t estimated_bytes;
      if (!EstimateRecordBytes(
              joint_counts, RecordMode::kOneReference, codec_params,
              &estimated_bytes, error)) {
        return false;
      }
      AnchorCandidate candidate;
      candidate.estimated_bytes = estimated_bytes;
      candidate.sparse_exception_ct =
          SparseExceptionCt(joint_counts, RecordMode::kOneReference);
      candidate.ordinal = anchor_ordinal;
      candidate.offset = anchor_offset;
      std::copy(joint_counts, &(joint_counts[16]),
                candidate.joint_counts.begin());
      candidates.push_back(std::move(candidate));
    }
    std::sort(
        candidates.begin(), candidates.end(),
        [](const AnchorCandidate& lhs, const AnchorCandidate& rhs) {
          if (lhs.estimated_bytes != rhs.estimated_bytes) {
            return lhs.estimated_bytes < rhs.estimated_bytes;
          }
          return lhs.ordinal < rhs.ordinal;
        });
  }

  std::vector<AnchorCandidate> sparse_candidates;
  if (params.enable_alternate_records && (!candidates.empty())) {
    sparse_candidates = candidates;
    std::sort(
        sparse_candidates.begin(), sparse_candidates.end(),
        [](const AnchorCandidate& lhs, const AnchorCandidate& rhs) {
          if (lhs.sparse_exception_ct != rhs.sparse_exception_ct) {
            return lhs.sparse_exception_ct < rhs.sparse_exception_ct;
          }
          return lhs.ordinal < rhs.ordinal;
        });
  }

  const bool has_single = !candidates.empty();
  AnchorCandidate best_single;
  if (has_single) {
    best_single = candidates[0];
  }
  bool has_pair = false;
  uint64_t best_pair_estimate = std::numeric_limits<uint64_t>::max();
  uint64_t best_sparse_exception_ct =
      std::numeric_limits<uint64_t>::max();
  uint32_t best_first_idx = 0;
  uint32_t best_second_idx = 0;
  uint32_t best_sparse_first_idx = 0;
  uint32_t best_sparse_second_idx = 0;
  std::array<uint32_t, 64> best_pair_counts = {};
  std::array<uint32_t, 64> best_sparse_pair_counts = {};
  if (params.two_ref_shortlist && (candidates.size() >= 2)) {
    if (candidates.size() > params.two_ref_shortlist) {
      candidates.resize(params.two_ref_shortlist);
    }
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
        uint32_t anchor_pair_counts[16];
        if (use_packed_planes) {
          CountJointGenotypesPacked(
              (*anchor_plane_cache)[candidates[first_idx].ordinal],
              (*anchor_plane_cache)[candidates[second_idx].ordinal],
              counts[candidates[first_idx].offset].data(),
              counts[candidates[second_idx].offset].data(),
              anchor_pair_counts);
        } else {
          CountJointGenotypes(
              anchor1, anchor2, sample_ct,
              counts[candidates[first_idx].offset].data(),
              counts[candidates[second_idx].offset].data(),
              anchor_pair_counts);
        }
        uint32_t triple_counts[64];
        if (use_packed_planes) {
          CountTripleGenotypesPacked(
              (*anchor_plane_cache)[candidates[first_idx].ordinal],
              (*anchor_plane_cache)[candidates[second_idx].ordinal],
              *target_plane_scratch, anchor_pair_counts,
              candidates[first_idx].joint_counts.data(),
              candidates[second_idx].joint_counts.data(), triple_counts);
#ifdef PGEN_RANS_VERIFY_FAST_COUNTS
          uint32_t reference_counts[64];
          CountTripleGenotypesReference(
              anchor1, anchor2, target, sample_ct, reference_counts);
          assert(std::equal(
              triple_counts, &(triple_counts[64]), reference_counts));
#endif
        } else {
          CountTripleGenotypes(
              anchor1, anchor2, target, sample_ct,
              counts[candidates[first_idx].offset].data(),
              counts[candidates[second_idx].offset].data(),
              target_counts.data(), anchor_pair_counts,
              candidates[first_idx].joint_counts.data(),
              candidates[second_idx].joint_counts.data(), triple_counts);
        }
        uint64_t estimated_bytes;
        if (!EstimateRecordBytes(
                triple_counts, RecordMode::kTwoReference, codec_params,
                &estimated_bytes, error)) {
          return false;
        }
        if (estimated_bytes < best_pair_estimate) {
          best_pair_estimate = estimated_bytes;
          best_first_idx = first_idx;
          best_second_idx = second_idx;
          std::copy(
              triple_counts, &(triple_counts[64]),
              best_pair_counts.begin());
        }
        if (params.enable_alternate_records) {
          const uint64_t sparse_exception_ct =
              SparseExceptionCt(triple_counts, RecordMode::kTwoReference);
          if (sparse_exception_ct < best_sparse_exception_ct) {
            best_sparse_exception_ct = sparse_exception_ct;
            best_sparse_first_idx = first_idx;
            best_sparse_second_idx = second_idx;
            std::copy(
                triple_counts, &(triple_counts[64]),
                best_sparse_pair_counts.begin());
          }
        }
      }
    }
    has_pair = true;
  }

  uint64_t best_rans_estimate = marginal_estimate;
  if (has_single) {
    best_rans_estimate =
        std::min(best_rans_estimate, best_single.estimated_bytes);
  }
  if (has_pair) {
    best_rans_estimate =
        std::min(best_rans_estimate, best_pair_estimate);
  }

  // For large cohorts the count-based byte estimate is accurate to much
  // less than a percent, while building every losing rANS payload costs a
  // complete reverse encode and forward interleave.  Materialize the
  // predicted winner and only genuinely close alternatives.  Retain the
  // exhaustive behavior for small cohorts, where fixed-size coder effects
  // are a larger fraction of each record and throughput is less important.
  // The estimator's finite-state error is bounded by one byte per lane, so
  // two estimates farther apart than twice the lane count cannot reverse
  // order after materialization.
  constexpr uint32_t kEstimateGatedSampleCt = 32768;
  const uint64_t estimate_slack = 2 * codec_params.state_ct;
  const auto should_materialize =
      [&](uint64_t estimate) {
        return (sample_ct < kEstimateGatedSampleCt) ||
               (estimate - best_rans_estimate <= estimate_slack);
      };
  output->clear();
  std::vector<uint8_t> candidate_record;
  const auto materialize =
      [&](const uint64_t* reference1, const uint64_t* reference2,
          RecordMode mode, uint8_t reference1_idx,
          uint8_t reference2_idx, uint64_t estimate,
          const uint32_t* context_symbol_counts) {
        if (!should_materialize(estimate)) {
          return true;
        }
        if (!EncodeRecordFromCounts(
                target64, reference1, reference2, sample_ct, mode,
                reference1_idx, reference2_idx, context_symbol_counts,
                codec_params,
                &candidate_record, error)) {
          return false;
        }
        if (output->empty() ||
            (candidate_record.size() < output->size())) {
          *output = std::move(candidate_record);
        }
        return true;
      };
  if (!materialize(
          nullptr, nullptr, RecordMode::kMarginal, 0, 0,
          marginal_estimate, target_counts.data())) {
    return false;
  }
  if (has_single) {
    const uintptr_t* single_anchor =
        &(block_genovecs[static_cast<uintptr_t>(best_single.offset) *
                          genovec_word_stride]);
    if (!materialize(
            reinterpret_cast<const uint64_t*>(single_anchor), nullptr,
            RecordMode::kOneReference,
            static_cast<uint8_t>(best_single.ordinal), 0,
            best_single.estimated_bytes,
            best_single.joint_counts.data())) {
      return false;
    }
  }
  if (has_pair) {
    const AnchorCandidate& first = candidates[best_first_idx];
    const AnchorCandidate& second = candidates[best_second_idx];
    const uintptr_t* anchor1 =
        &(block_genovecs[static_cast<uintptr_t>(first.offset) *
                          genovec_word_stride]);
    const uintptr_t* anchor2 =
        &(block_genovecs[static_cast<uintptr_t>(second.offset) *
                          genovec_word_stride]);
    if (!materialize(
            reinterpret_cast<const uint64_t*>(anchor1),
            reinterpret_cast<const uint64_t*>(anchor2),
            RecordMode::kTwoReference,
            static_cast<uint8_t>(first.ordinal),
            static_cast<uint8_t>(second.ordinal), best_pair_estimate,
            best_pair_counts.data())) {
      return false;
    }
  }
  if (output->empty()) {
    *error = "Internal rANS record selection produced no candidate.";
    return false;
  }

  if (params.enable_alternate_records) {
    std::vector<uint8_t> alternate_record;
    const uint64_t raw_record_byte_ct =
        1 + (static_cast<uint64_t>(sample_ct) + 3) / 4;
    if (raw_record_byte_ct < output->size()) {
      if (!EncodeRawRecord(
              target64, sample_ct, &alternate_record, error)) {
        return false;
      }
      KeepSmallerRecord(&alternate_record, output);
    }
    const uint64_t marginal_sparse_exception_ct =
        SparseExceptionCt(
            target_counts.data(), RecordMode::kMarginal);
    if (SparseRecordLowerBoundByteCt(
            RecordMode::kMarginal, marginal_sparse_exception_ct,
            sample_ct) < output->size()) {
      if (!EncodeSparsePredictorRecordFromCounts(
              target64, nullptr, nullptr, sample_ct,
              RecordMode::kMarginal, 0, 0, target_counts.data(),
              &alternate_record, error)) {
        return false;
      }
      KeepSmallerRecord(&alternate_record, output);
    }
    const size_t sparse_candidate_ct =
        std::min<size_t>(4, sparse_candidates.size());
    for (size_t candidate_idx = 0;
         candidate_idx != sparse_candidate_ct; ++candidate_idx) {
      const AnchorCandidate& candidate =
          sparse_candidates[candidate_idx];
      if (SparseRecordLowerBoundByteCt(
              RecordMode::kOneReference,
              candidate.sparse_exception_ct, sample_ct) >=
          output->size()) {
        continue;
      }
      const uintptr_t* sparse_anchor =
          &(block_genovecs[static_cast<uintptr_t>(candidate.offset) *
                            genovec_word_stride]);
      if (!EncodeSparsePredictorRecordFromCounts(
              target64,
              reinterpret_cast<const uint64_t*>(sparse_anchor), nullptr,
              sample_ct, RecordMode::kOneReference,
              static_cast<uint8_t>(candidate.ordinal), 0,
              candidate.joint_counts.data(),
              &alternate_record, error)) {
        return false;
      }
      KeepSmallerRecord(&alternate_record, output);
    }
    if (has_pair &&
        (SparseRecordLowerBoundByteCt(
             RecordMode::kTwoReference, best_sparse_exception_ct,
             sample_ct) < output->size())) {
      const AnchorCandidate& sparse_first =
          candidates[best_sparse_first_idx];
      const AnchorCandidate& sparse_second =
          candidates[best_sparse_second_idx];
      const uintptr_t* sparse_anchor1 =
          &(block_genovecs[static_cast<uintptr_t>(
                                sparse_first.offset) *
                            genovec_word_stride]);
      const uintptr_t* sparse_anchor2 =
          &(block_genovecs[static_cast<uintptr_t>(
                                sparse_second.offset) *
                            genovec_word_stride]);
      if (!EncodeSparsePredictorRecordFromCounts(
              target64, reinterpret_cast<const uint64_t*>(sparse_anchor1),
              reinterpret_cast<const uint64_t*>(sparse_anchor2), sample_ct,
              RecordMode::kTwoReference,
              static_cast<uint8_t>(sparse_first.ordinal),
              static_cast<uint8_t>(sparse_second.ordinal),
              best_sparse_pair_counts.data(),
              &alternate_record, error)) {
        return false;
      }
      KeepSmallerRecord(&alternate_record, output);
    }
  }
  return true;
}

bool ValidateInputs(const EncodeInput& input, EncodeParams* params,
                    std::string* error) {
  if ((!input.sample_ct) || (!input.variant_ct) || (!input.pgfi) ||
      (!input.pgen_reader)) {
    *error = "Invalid conditional-rANS encoder input.";
    return false;
  }
  if ((input.sample_ct > input.raw_sample_ct) ||
      (input.variant_ct > input.raw_variant_ct)) {
    *error = "Conditional-rANS subset counts exceed raw input counts.";
    return false;
  }
  if ((input.pgfi->raw_sample_ct != input.raw_sample_ct) ||
      (input.pgfi->raw_variant_ct != input.raw_variant_ct)) {
    *error = "Conditional-rANS input dimensions do not match the PGEN reader.";
    return false;
  }
  if ((input.sample_ct != input.raw_sample_ct) && (!input.sample_include)) {
    *error = "Conditional-rANS sample subset mask is missing.";
    return false;
  }
  if (input.variant_uidxs) {
    uint32_t previous_uidx = 0;
    for (uint32_t variant_idx = 0; variant_idx != input.variant_ct;
         ++variant_idx) {
      const uint32_t variant_uidx = input.variant_uidxs[variant_idx];
      if ((variant_uidx >= input.raw_variant_ct) ||
          (variant_idx && (variant_uidx <= previous_uidx))) {
        *error =
            "Conditional-rANS variant subset indices are invalid or unsorted.";
        return false;
      }
      previous_uidx = variant_uidx;
    }
  }
  if ((params->block_variant_ct < 2) ||
      (params->block_variant_ct > UINT16_MAX) || (!params->anchor_ct) ||
      (params->anchor_ct > 256) || (!params->thread_ct) ||
      (params->rans_state_ct > 256) ||
      (params->rans_scale_bits < 8) ||
      (params->rans_scale_bits > 16) ||
      (params->restart_variant_ct == 0) ||
      (params->restart_variant_ct > UINT16_MAX) ||
      (params->two_ref_shortlist == 1)) {
    *error = "Invalid conditional-rANS encoder parameters.";
    return false;
  }
  params->anchor_ct =
      std::min(params->anchor_ct, params->block_variant_ct);
  params->two_ref_shortlist =
      std::min(params->two_ref_shortlist, params->anchor_ct);
  if (params->anchor_ct < 2) {
    params->two_ref_shortlist = 0;
  }
  if (((input.pgfi->gflags & kfPgenGlobalHardcallPhasePresent) &&
       (!input.discard_phase)) ||
      ((input.pgfi->gflags &
        (kfPgenGlobalDosagePresent | kfPgenGlobalDosagePhasePresent)) &&
       (!input.discard_dosage))) {
    *error =
        "Conditional-rANS output requires explicit phase/dosage erasure.";
    return false;
  }
  if ((input.pgfi->max_allele_ct > 2) && (!input.variant_metadata)) {
    *error =
        "Conditional-rANS multiallelic output requires variant allele counts.";
    return false;
  }
  if (input.variant_metadata) {
    for (uint32_t variant_idx = 0; variant_idx != input.variant_ct;
         ++variant_idx) {
      const uint16_t allele_ct =
          input.variant_metadata[variant_idx].allele_ct;
      if ((allele_ct < 2) || (allele_ct > 255)) {
        *error = "Conditional-rANS variant allele count is out of range.";
        return false;
      }
    }
  }
  return true;
}

uint32_t AutomaticRansStateCt(uint32_t sample_ct) {
#if defined(__aarch64__) || defined(_M_ARM64)
  // Sixteen packed scalar lanes are both smaller and faster on ARM64.
  (void)sample_ct;
  return 16;
#else
  // Keep 32 lanes for large x86 cohorts so AVX-512 and CUDA consumers can
  // use their natural warp/register width.
  return (sample_ct < 32768) ? 16 : 32;
#endif
}

}  // namespace

PglErr EncodePgenRans(const std::string& output_path,
                      const EncodeInput& input,
                      const EncodeParams& requested_params,
                      EncodeStats* stats, std::string* error) {
  EncodeParams params = requested_params;
  if (!ValidateInputs(input, &params, error)) {
    return kPglRetInconsistentInput;
  }
  if (!params.rans_state_ct) {
    params.rans_state_ct = AutomaticRansStateCt(input.sample_ct);
  }
  *stats = EncodeStats();
  stats->variant_ct = input.variant_ct;
  const std::vector<VariantBlock> blocks =
      BuildBlocks(input.variant_ct, params.block_variant_ct,
                  input.variant_metadata);
  stats->block_ct = static_cast<uint32_t>(blocks.size());
  const CodecParams codec_params = {
      params.rans_state_ct, params.rans_scale_bits};
  uint32_t max_allele_ct = 2;
  if (input.variant_metadata) {
    for (uint32_t variant_idx = 0; variant_idx != input.variant_ct;
         ++variant_idx) {
      max_allele_ct =
          std::max<uint32_t>(
              max_allele_ct,
              input.variant_metadata[variant_idx].allele_ct);
    }
  }
  const ContainerParams container_params = {
      input.sample_ct, input.variant_ct, params.block_variant_ct,
      params.anchor_ct, params.rans_state_ct, params.rans_scale_bits,
      params.restart_variant_ct, static_cast<uint32_t>(blocks.size()),
      max_allele_ct};
  ContainerMetadata container_metadata;
  const uintptr_t* input_nonref_flags = input.pgfi->nonref_flags;
  if (input_nonref_flags) {
    container_metadata.nonref_flags.assign(
        (static_cast<size_t>(input.variant_ct) + 7) / 8, 0);
    uint32_t nonref_ct = 0;
    for (uint32_t variant_idx = 0; variant_idx != input.variant_ct;
         ++variant_idx) {
      const uint32_t variant_uidx =
          input.variant_uidxs ? input.variant_uidxs[variant_idx]
                              : variant_idx;
      if (IsSet(input_nonref_flags, variant_uidx)) {
        container_metadata.nonref_flags[variant_idx / 8] |=
            static_cast<uint8_t>(1U << (variant_idx % 8));
        ++nonref_ct;
      }
    }
    if (!nonref_ct) {
      container_metadata.nonref_flags.clear();
    } else if (nonref_ct == input.variant_ct) {
      container_metadata.nonref_flags.clear();
      container_metadata.all_nonref = true;
    }
  } else if (input.pgfi->gflags & kfPgenGlobalAllNonref) {
    container_metadata.all_nonref = true;
  }
  ContainerWriter writer;
  if (!writer.Open(
          output_path, container_params, container_metadata, error)) {
    return kPglRetOpenFail;
  }

  PgrSampleSubsetIndex pssi;
  std::vector<uint32_t> sample_include_cumulative_popcounts;
  if (input.sample_include) {
    sample_include_cumulative_popcounts.resize(
        BitCtToWordCt(input.raw_sample_ct));
    FillCumulativePopcounts(input.sample_include,
                            BitCtToWordCt(input.raw_sample_ct),
                            sample_include_cumulative_popcounts.data());
    PgrSetSampleSubsetIndex(sample_include_cumulative_popcounts.data(),
                            input.pgen_reader, &pssi);
  } else {
    PgrClearSampleSubsetIndex(input.pgen_reader, &pssi);
  }

  const uint32_t genovec_word_stride =
      NypCtToVecCt(input.sample_ct) * kWordsPerVec;
  uintptr_t* block_genovecs = nullptr;
  if (cachealigned_malloc(
          static_cast<uintptr_t>(params.block_variant_ct) *
              genovec_word_stride * sizeof(uintptr_t),
          &block_genovecs)) {
    *error = "Out of memory allocating the conditional-rANS genotype block.";
    return kPglRetNomem;
  }
  uintptr_t* patch_01_set = nullptr;
  AlleleCode* patch_01_vals = nullptr;
  uintptr_t* patch_10_set = nullptr;
  AlleleCode* patch_10_vals = nullptr;
  const uint32_t patch_set_word_ct =
      BitCtToVecCt(input.sample_ct) * kWordsPerVec;
  if (cachealigned_malloc(
          static_cast<uintptr_t>(patch_set_word_ct) *
              sizeof(uintptr_t),
          &patch_01_set) ||
      cachealigned_malloc(
          static_cast<uintptr_t>(input.sample_ct) *
              sizeof(AlleleCode),
          &patch_01_vals) ||
      cachealigned_malloc(
          static_cast<uintptr_t>(patch_set_word_ct) *
              sizeof(uintptr_t),
          &patch_10_set) ||
      cachealigned_malloc(
          static_cast<uintptr_t>(2) * input.sample_ct *
              sizeof(AlleleCode),
          &patch_10_vals)) {
    aligned_free_cond(patch_10_vals);
    aligned_free_cond(patch_10_set);
    aligned_free_cond(patch_01_vals);
    aligned_free_cond(patch_01_set);
    aligned_free(block_genovecs);
    *error =
        "Out of memory allocating conditional-rANS multiallelic patches.";
    return kPglRetNomem;
  }

  fputs("0%", stdout);
  fflush(stdout);
  uint32_t progress_char_ct = 2;
  uint32_t next_progress_variant_ct = static_cast<uint32_t>(
      (static_cast<uint64_t>(input.variant_ct) + 99) / 100);
  uint32_t processed_variant_ct = 0;
  const auto start_time = std::chrono::steady_clock::now();
  PglErr reterr = kPglRetSuccess;
  for (const VariantBlock& block_range : blocks) {
    std::vector<std::array<uint32_t, 4>> counts(block_range.len);
    std::vector<MultiallelicPatches> multiallelic_patches(
        block_range.len);
    for (uint32_t offset = 0; offset != block_range.len; ++offset) {
      uintptr_t* genovec =
          &(block_genovecs[static_cast<uintptr_t>(offset) *
                            genovec_word_stride]);
      const uint32_t variant_idx = block_range.start + offset;
      const uint32_t variant_uidx =
          input.variant_uidxs ? input.variant_uidxs[variant_idx] : variant_idx;
      const uint16_t allele_ct = input.variant_metadata
                                     ? input.variant_metadata[variant_idx]
                                           .allele_ct
                                     : 2;
      PglErr pgl_error;
      if (allele_ct > 2) {
        PgenVariant pgv = {};
        pgv.genovec = genovec;
        pgv.patch_01_set = patch_01_set;
        pgv.patch_01_vals = patch_01_vals;
        pgv.patch_10_set = patch_10_set;
        pgv.patch_10_vals = patch_10_vals;
        pgl_error =
            PgrGetM(
                input.sample_include, pssi, input.sample_ct,
                variant_uidx, input.pgen_reader, &pgv);
        if (!pgl_error) {
          CopyPgenPatches(
              pgv, allele_ct, &multiallelic_patches[offset]);
        }
      } else {
        pgl_error =
            PgrGet(
                input.sample_include, pssi, input.sample_ct,
                variant_uidx, input.pgen_reader, genovec);
      }
      if (pgl_error) {
        *error = "PGEN hardcall load failed at variant " +
                 std::to_string(variant_uidx) + " with code " +
                 std::to_string(static_cast<uint32_t>(pgl_error)) + ".";
        reterr = pgl_error;
        goto cleanup;
      }
      ZeroTrailingNyps(input.sample_ct, genovec);
      GenoarrCountFreqsUnsafe(genovec, input.sample_ct, counts[offset]);
      stats->pgen_payload_bytes +=
          GetPgfiVrecWidth(input.pgfi, variant_uidx);
    }
    const std::vector<uint32_t> anchor_offsets =
        ScheduledAnchorOffsets(block_range.len, params.anchor_ct);
    std::vector<uint8_t> is_anchor(block_range.len, 0);
    for (const uint32_t anchor_offset : anchor_offsets) {
      is_anchor[anchor_offset] = 1;
    }
    const bool use_packed_planes = input.sample_ct >= 4096;
    std::vector<PackedGenotypePlanes> anchor_plane_cache;
    if (use_packed_planes) {
      anchor_plane_cache.resize(anchor_offsets.size());
      for (uint32_t anchor_ordinal = 0;
           anchor_ordinal != anchor_offsets.size(); ++anchor_ordinal) {
        const uint32_t anchor_offset =
            anchor_offsets[anchor_ordinal];
        BuildPackedGenotypePlanes(
            &(block_genovecs[
                static_cast<uintptr_t>(anchor_offset) *
                genovec_word_stride]),
            counts[anchor_offset].data(), input.sample_ct,
            &(anchor_plane_cache[anchor_ordinal]));
      }
    }
    EncodedBlock block;
    block.first_variant = block_range.start;
    block.records.resize(block_range.len);
    std::atomic<uint32_t> next_offset(0);
    std::atomic<bool> failed(false);
    std::mutex error_mutex;
    std::string worker_error;
    const uint32_t worker_ct =
        std::min(params.thread_ct, block_range.len);
    std::vector<std::thread> workers;
    workers.reserve(worker_ct);
    for (uint32_t worker_idx = 0; worker_idx != worker_ct; ++worker_idx) {
      workers.emplace_back([&]() {
        PackedGenotypePlanes target_plane_scratch;
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
                  input.sample_ct, anchor_offsets, counts,
                  use_packed_planes ? &anchor_plane_cache : nullptr,
                  use_packed_planes ? &target_plane_scratch : nullptr,
                  input.variant_metadata, params, codec_params,
                  is_anchor[offset], &(block.records[offset]),
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
      *error = worker_error;
      reterr = kPglRetInconsistentInput;
      goto cleanup;
    }
    for (uint32_t offset = 0; offset != block_range.len; ++offset) {
      if (multiallelic_patches[offset].allele_ct > 2) {
        const size_t base_size = block.records[offset].size();
        if (!AppendMultiallelicPatches(
                input.sample_ct, multiallelic_patches[offset],
                &block.records[offset], error)) {
          reterr = kPglRetInconsistentInput;
          goto cleanup;
        }
        ++stats->multiallelic_ct;
        stats->patch_01_ct +=
            multiallelic_patches[offset].patch_01_sample_ids.size();
        stats->patch_10_ct +=
            multiallelic_patches[offset].patch_10_sample_ids.size();
        stats->multiallelic_patch_bytes +=
            block.records[offset].size() - base_size;
      }
      RecordMetadata record_metadata;
      if (!ParseRecordMetadata(block.records[offset].data(),
                               block.records[offset].size(),
                               &record_metadata, error)) {
        reterr = kPglRetMalformedInput;
        goto cleanup;
      }
      stats->anchor_ct += is_anchor[offset];
      if (record_metadata.mode == RecordMode::kMarginal) {
        ++stats->marginal_ct;
      } else if (record_metadata.mode == RecordMode::kOneReference) {
        ++stats->one_reference_ct;
      } else {
        ++stats->two_reference_ct;
      }
      if (record_metadata.is_raw_packed) {
        ++stats->raw_packed_ct;
      } else if (record_metadata.is_sparse_predictor) {
        ++stats->sparse_predictor_ct;
      } else if (record_metadata.has_entropy_payload) {
        ++stats->entropy_rans_ct;
      } else {
        ++stats->deterministic_rans_ct;
      }
    }
    if (!writer.WriteBlock(block, error)) {
      reterr = kPglRetWriteFail;
      goto cleanup;
    }
    processed_variant_ct += block_range.len;
    if ((processed_variant_ct != input.variant_ct) &&
        (processed_variant_ct >= next_progress_variant_ct)) {
      const uint32_t progress_pct = static_cast<uint32_t>(
          (static_cast<uint64_t>(processed_variant_ct) * 100) /
          input.variant_ct);
      for (uint32_t char_idx = 0; char_idx != progress_char_ct;
           ++char_idx) {
        fputc('\b', stdout);
      }
      progress_char_ct =
          static_cast<uint32_t>(printf("%u%%", progress_pct));
      fflush(stdout);
      next_progress_variant_ct = static_cast<uint32_t>(
          (static_cast<uint64_t>(progress_pct + 1) *
               input.variant_ct +
           99) /
          100);
    }
  }
  if (!writer.Close(error)) {
    reterr = kPglRetWriteFail;
    goto cleanup;
  }
  stats->elapsed_seconds =
      std::chrono::duration<double>(
          std::chrono::steady_clock::now() - start_time)
          .count();
  stats->output_bytes = FileSize(output_path);
  for (uint32_t char_idx = 0; char_idx != progress_char_ct;
       ++char_idx) {
    fputc('\b', stdout);
  }
  fflush(stdout);

cleanup:
  aligned_free(patch_10_vals);
  aligned_free(patch_10_set);
  aligned_free(patch_01_vals);
  aligned_free(patch_01_set);
  aligned_free(block_genovecs);
  return reterr;
}

}  // namespace pgen_rans
