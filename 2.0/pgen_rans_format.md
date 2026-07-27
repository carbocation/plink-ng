# Conditional-rANS genotype record format

This document defines developer PGEN storage mode `0x80`, used by this fork
for block-local conditional-rANS hardcalls.  The mode number is not an
upstream assignment; an accepted PGEN revision must replace it with an
officially reserved storage-mode value.

All integers are little-endian. Genotypes use the PGEN two-bit hardcall
alphabet: `0`, `1`, `2`, and `3` (missing). For multiallelic variants this
base stream has PGEN's usual ALT-collapsed meaning; a sparse suffix restores
the exact allele codes.

## Decode graph

A block contains evenly spaced, independently encoded anchor variants.
Every other record is encoded as one of:

- a marginal four-symbol model;
- `P(g_target | g_anchor)`;
- `P(g_target | g_anchor1, g_anchor2)`.

Both references must name independent anchors in the same block. A decoder
therefore performs two breadth-first steps: decode the anchor slab, then decode
all remaining records independently. Two-reference records do not add another
dependency level.

## Record

The first byte contains:

- bits 0-1: mode (`0` marginal, `1` one reference, `2` two references);
- bit 2: rANS state and byte-renormalization payload are present;
- bit 3: a multiallelic patch suffix is present;
- bit 4: the rANS payload has no legacy 15-byte zero guard;
- bit 5: raw packed record;
- bit 6: sparse predictor record;
- bit 7: sparse predictor exception IDs use a bitmap.

Bits 5 and 6 are mutually exclusive. Bit 7 requires bit 6. Alternate raw
and sparse records do not set bits 2 or 4; the patch-suffix bit remains
independent.

One-reference records next store one byte containing the anchor ordinal.
Two-reference records store two distinct anchor ordinals. The current format
supports at most 256 anchors per block.

The probability model follows:

- marginal: one model row;
- one reference: a one-byte mask of populated reference-genotype contexts,
  followed by up to four model rows;
- two references: a two-byte mask of populated ordered genotype-pair
  contexts, followed by up to sixteen model rows.

Each model row stores a four-bit active-symbol mask followed by a 16-bit
normalized frequency for every active symbol except the last. The final
frequency is the normalization total minus the preceding frequencies.
Contexts and symbols are serialized in numeric order.

The current normalization precision is configurable from 8 through 16 bits;
12 bits is the intended format default.

## rANS payload

When any model row has more than one active symbol, the record stores:

1. one 32-bit initial state per lane;
2. the byte-renormalization payload described below.

Sample `i` belongs to lane `i mod lane_count`. Each lane is an independent
bytewise-rANS state. The encoder API's automatic policy uses 16 lanes on
ARM64, where the packed scalar decoder is faster as well as smaller. On x86
it uses 16 lanes below 32768 samples and 32 lanes for larger cohorts so the
AVX-512 path remains available. Callers can still request an explicit count;
the CUDA decoder currently requires 32.

There is no lane-boundary table. Refill bytes are merged in forward decoder
order. Each `lane_count`-sample round visits consecutive groups of up to 16
lanes. For each group, the decoder computes the mask of states below the rANS
lower bound and consumes one byte for every set lane, in ascending lane order;
it repeats until no lane in that group needs another byte. New encoders set
record flag bit 4 and end the record at the coded stream. AVX-512 decoders use
a masked load for the final short refill. Decoders also accept the legacy
representation, where bit 4 is clear and fifteen validated zero guard bytes
follow the payload.

AVX-512 can load each refill layer contiguously and expand it under the state
mask, while CUDA lanes read a coalesced span in warp order.

If every populated row is deterministic, the state table and refill payload
are omitted.

## Raw and sparse alternate records

The encoder compares complete serialized record sizes and can replace rANS
with one of two exactly decoded alternatives. This is a per-record decision,
so these modes cannot make the base record larger.

A raw packed record sets bit 5, has marginal reference mode, and stores
exactly `ceil(sample_count / 4)` bytes in ordinary two-bit sample order.
Unused high nyps of the last byte are zero. This both caps worst-case
expansion and provides a copy-like decode path for high-entropy records.

A sparse predictor record sets bit 6 and may use marginal, one-reference, or
two-reference mode. After any reference selectors it stores:

1. one two-bit modal prediction for every context, four predictions per byte;
2. the exception count as an unsigned base-128 varint;
3. exception sample IDs, as either strictly increasing delta varints or a
   `ceil(sample_count / 8)` bitmap (bit 7), whichever is smaller;
4. the actual exception genotypes, four two-bit values per byte.

Predictor mappings cover all 1, 4, or 16 contexts, including contexts absent
from the input. Ties choose the lowest genotype. Mapping, bitmap, and value
padding bits are zero. The CPU decoder expands predictors a packed word at a
time; the CUDA decoder expands them warp-wide and applies sparse exceptions
afterward.

## Multiallelic patch suffix

Every variant with three or more alleles has a suffix, including variants
where the selected samples happen to contain no non-ALT1 calls. The suffix
follows the complete base record and ends with a four-byte suffix length, so a
collapsed decoder can find the base-record boundary in constant time.

The suffix contains:

1. one-byte patch format version (`1`);
2. one-byte allele count (`3` through `255`);
3. one-byte patch flags selecting a sample bitmap for `patch_01` and/or
   `patch_10`;
4. 32-bit `patch_01` and `patch_10` call counts;
5. sample IDs for each patch class, represented independently as either
   strictly increasing delta varints or a `ceil(sample_count / 8)` bitmap,
   whichever is smaller;
6. bit-packed allele codes.

`patch_01` identifies ref/ALT calls whose ALT is not ALT1 and stores one allele
code per sample. Its stored value is `allele_code - 2`, using
`ceil(log2(allele_count - 2))` bits. `patch_10` identifies two-ALT calls other
than ALT1/ALT1 and stores two allele codes per sample. Each stored value is
`allele_code - 1`, using `ceil(log2(allele_count - 1))` bits. These are the
same sparse semantics exposed by PGEN's `PgrGetM()`.

CPU and CUDA block decoders intentionally ignore this suffix when emitting
the standard packed 2-bit analysis matrix. Allele-aware callers decode the
sparse suffix separately; `PackedVariantReader::ReadVariantPatches()` exposes
it in the stored sample order.

## Canonical validation

A conforming decoder rejects:

- unknown flag bits or modes;
- invalid raw/sparse flag combinations, lengths, padding, or exceptions;
- truncated selectors, models, states, or refill payloads;
- duplicate two-reference selectors;
- selectors outside the block anchor slab;
- invalid normalized-frequency totals;
- references selecting an absent model context;
- nonzero refill-interleaved padding;
- refill-interleaved payloads that are not consumed exactly;
- lanes that do not terminate at the rANS lower-bound state.
- invalid suffix versions, lengths, allele codes, sample ordering, or padding.

## PGEN storage-mode framing

The file begins with the standard PGEN magic bytes `6c 1b`, followed by
developer storage mode `80`.  Bytes 3-10 contain the standard little-endian
variant and sample counts, and byte 11 uses PGEN's normal provisional-REF
summary bits.  The storage-mode-specific header continues through byte 95:

| Offset | Width | Value |
| --- | ---: | --- |
| 12 | 4 | conditional-rANS format version (`2`; readers also accept legacy v1) |
| 16 | 4 | header size (`96`) |
| 20 | 4 | variants per block |
| 24 | 4 | anchor count |
| 28 | 4 | rANS state count |
| 32 | 4 | rANS scale bits |
| 36 | 4 | cumulative-offset restart interval |
| 40 | 4 | block count |
| 44 | 4 | maximum allele count |
| 48 | 4 | storage-mode flags |
| 52 | 8 | block-table offset |
| 60 | 8 | metadata offset |
| 68 | 8 | metadata byte count |
| 76 | 8 | block-data offset |
| 84 | 12 | reserved, zero |

The header is followed by a fixed 32-byte entry for every block containing
the first variant, variant count, file offset, and byte count. This permits a
client to issue one bounded ranged read for a block without scanning earlier
variant records. Each entry also stores a CRC32C checksum of the complete
serialized block.

The format preserves PGEN's provisional/nonreference REF status. Uniform
trusted and uniform provisional files consume only a header flag. Mixed files
store one bit per variant between the block table and block data. The bitmap is
small enough to fetch with the block table during open, so ranged readers still
need only two initial requests. PVAR/PSAM remain the authoritative variant and
sample metadata; the PGEN header's counts, maximum allele count, and REF-status
summary let readers reject mismatched companion files.

Each block starts with a 16-byte header followed by:

1. one 24-bit record length per variant;
2. one 32-bit cumulative payload offset at each restart after the first;
3. the concatenated records.

To find a record, a reader starts from the closest preceding cumulative
restart and sums at most `restart_interval - 1` 24-bit lengths. The default
restart interval is 64 variants.

Readers validate that block-table entries cover the variants and file exactly,
blocks are contiguous, restart offsets agree with record lengths, record
lengths span each block payload, and the block CRC32C matches.

The reference reader exposes the file header and block table through a
caller-supplied `read_at(offset, length)` callback. Opening a file requires one
fixed-header read and one block-table read; fetching a block then requires
exactly one bounded read matching its table entry. A validated block view
points directly into that caller-owned byte buffer, so CPU and GPU adapters do
not need to copy every record into a separate allocation.

The CPU block decoder emits the standard variant-major packed two-bit
hardcall layout. It first decodes the scheduled anchor slab and then dispatches
all remaining records independently across a persistent worker pool. The
packed scalar paths for 4, 8, 16, and 32 states advance lanes in sample order
and assemble complete 64-bit packed genotype words without a per-sample output
loop.

On x86-64 GCC and Clang builds, 32-state/12-bit records use a
runtime-dispatched AVX-512 decoder for cohorts with at least 32768 samples.
It keeps all three model boundaries in registers, derives packed genotype bits
directly from the threshold masks, and refills 16 states from one contiguous
load. The scalar path remains available for smaller cohorts and CPUs without
AVX-512, so the binary keeps its existing x86-64-v3 compatibility floor.
Validated container blocks use the fact that deterministic and unobserved
contexts are identity rANS intervals; the strict standalone record API
continues to reject a reference selecting an absent context.

CRC32C block validation is runtime-dispatched to SSE4.2 on x86-64 and the
CRC32 extension on AArch64, with a portable table fallback.

`PackedVariantReader` is the CPU-facing compatibility layer. It accepts
contiguous ranges or arbitrary variant-index lists, groups requests by block,
and copies exact `(sample_count + 3) / 4` byte packed hardcalls into the
caller's existing per-variant stride. Sparse requests decode only requested
records and their one or two scheduled marginal anchors; complete fresh-block
requests retain the parallel dense fallback. A byte-bounded cache retains
recent block bytes, decoded anchors, and decoded targets for adjacent or
repeated requests. Patch-only reads still fetch and checksum the complete
block but skip hardcall entropy decoding.

An optional sorted sample index projects each decoded requested record into a
smaller packed output. Since rANS states cannot jump over arbitrary samples,
this projection reduces downstream work and buffer size but not the
within-record entropy-decode work.

## Reference CLI

Build from `2.0/`:

```sh
make -f Makefile.pgen_rans pgen_rans
```

CPU consumers can instead build `build_pgen_rans/lib/libpgen_rans.a`:

```sh
make -f Makefile.pgen_rans libpgen_rans
```

The main PLINK 2 binary can encode the remaining variants and samples with its
standard filtering and threading options:

```sh
plink2 --pfile cohort \
  --chr 22 \
  --make-pgen format=rans \
  --threads 16 \
  --out cohort-chr22
```

This writes a normal `cohort-chr22.pgen/.pvar/.psam` fileset. The `.pgen`
header identifies the conditional-rANS storage mode; no alternate extension
or input flag is involved. The command intentionally has no benchmark-only
limit options; use normal PLINK selectors such as `--chr`,
`--from-bp`/`--to-bp`, and `--extract` to bound an encoding run.
When the input contains phase or dosage, add the corresponding `erase-phase`
or `erase-dosage` modifier explicitly; the writer refuses silent loss.

The main binary can read that fileset through the existing packed-hardcall
analysis seam:

```sh
plink2 --pfile cohort-chr22 \
  --score weights.txt \
  --out scores
```

The initial production surface includes `--score[-list]`, `--freq`,
`--export A/Av`, `--indep-pairwise`, `--r-unphased`, `--clump`, `--pca`,
`--make-pgen`, `--write-snplist`, and `--write-samples`. Sample, position, ID,
and genotype-frequency filters can be applied with these commands. Standard
`--make-pgen` without `format=rans` reconstructs an exact hardcall PGEN in the
upstream general-purpose storage mode; phase and dosage cannot be reconstructed
because the conditional-rANS mode does not store them. Conditional-rANS files
with more than two alleles at any variant currently require `--read-freq` for
scoring; multiallelic `--make-pgen`, `--r-unphased`, and allele-aware A/Av
exports are exact, while PCA, frequency scans, LD pruning, clumping, and
genotype-frequency filters are currently limited to biallelic filesets.
Other construction commands continue to use the upstream general-purpose
PGEN storage mode as their working format.

Nonoverlapping conditional-rANS shards can be merged without decoding and
re-encoding their genotype records:

```sh
plink2 --pmerge-list shards.txt pfile \
  --make-pgen format=rans \
  --indiv-sort none \
  --out cohort
```

The direct path preserves each compressed record byte-for-byte, rebases block
variant offsets, rebuilds the block index and checksums, and combines
nonreference metadata. All inputs must use compatible conditional-rANS codec
parameters, contain identical samples in identical order, and already be in
concatenable variant order without filtering, overlapping variant ranges, or
duplicate-variant merging. `--indiv-sort none` is only needed when the common
input order should be retained instead of PLINK's default sample sort. Mixing
ordinary and conditional-rANS PGEN inputs is rejected; ordinary-only merges
continue through the existing upstream merge and optional re-encode path.
When downstream filters, metadata updates, or nondefault PVAR/PSAM output
modifiers are present, PLINK first performs the block-copy merge to the normal
`-merge` intermediate and then runs the regular `--make-pgen` step. Thus those
options retain their usual semantics, while the no-reencode shortcut is
reserved for a plain final concatenation.

Encode an unphased hardcall PGEN, including exact multiallelic calls:

```sh
build_pgen_rans/bin/pgen_rans encode cohort.pgen cohort-rans.pgen \
  --pvar cohort.pvar \
  --block-variants 128 \
  --anchors 32 \
  --two-ref-shortlist 4 \
  --threads 8
```

The standalone encoder and exact verifier require the matching plain-text
PVAR so allele counts are available before PGEN reader initialization.

Then perform a byte-for-byte packed-hardcall and exact sparse-patch round trip:

```sh
build_pgen_rans/bin/pgen_rans verify \
  cohort.pgen cohort-rans.pgen --pvar cohort.pvar
```

`inspect` validates the PGEN and summarizes its record modes without
decoding genotypes:

```sh
build_pgen_rans/bin/pgen_rans inspect cohort-rans.pgen
```

`benchmark` deterministically samples block strata, decodes each block with a
persistent CPU worker pool, and compares every packed output word with
`PgrGet()` from the source PGEN:

```sh
build_pgen_rans/bin/pgen_rans benchmark cohort.pgen cohort-rans.pgen \
  --threads 16 \
  --blocks 80 \
  --iterations 3
```

The report separates bounded block-read time from warm decode time. The
reported serial total is intentionally conservative; a production caller can
double-buffer block reads and decoding.

## CUDA prototype

The CUDA decoder batches multiple container blocks, launches one warp per
record, and preserves the two-level dependency graph with separate anchor and
target kernels. Each lane owns one rANS state. Warp ballots and population
counts assign a contiguous refill span to the active lanes, and also assemble
the low and high genotype bits into the standard packed 64-bit output word.

On a CUDA machine, build and run the exact CPU/GPU comparison with:

```sh
make -f Makefile.pgen_rans_cuda CUDA_ARCH=80
build_pgen_rans/bin/pgen_rans_cuda_benchmark cohort-rans.pgen \
  --blocks 80 \
  --batch-blocks 8 \
  --iterations 3 \
  --cpu-threads 16
```

The benchmark reports block-read, host-to-device, anchor-kernel,
target-kernel, and complete decoder wall times separately. GPU output is
copied back once per batch and compared byte-for-byte with the persistent
CPU block decoder.

To build one fat binary for both T4 (`sm_75`) and A100 (`sm_80`), use:

```sh
make -f Makefile.pgen_rans_cuda CUDA_ARCHS="75 80"
```

`compression_builds.md` documents the Linux MKL, macOS Accelerate, and
multi-architecture CUDA artifact workflow.
