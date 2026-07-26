# Conditional-rANS genotype record format

This document defines the experimental record codec used by the standalone
PGEN compression prototype. It is not an assigned PGEN storage mode.

All integers are little-endian. Genotypes use the PGEN two-bit hardcall
alphabet: `0`, `1`, `2`, and `3` (missing).

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
- bits 3-7: reserved and zero.

One-reference records next store one byte containing the anchor ordinal.
Two-reference records store two distinct anchor ordinals. Version 2 therefore
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

The intended lane count is 32. Sample `i` belongs to lane `i mod lane_count`.
Each lane is an independent bytewise-rANS state.

There is no lane-boundary table. Refill bytes are merged in forward decoder
order. Each 32-sample round visits lanes 0-15 and then 16-31. For each group,
the decoder computes the mask of states below the rANS lower bound and consumes
one byte for every set lane, in ascending lane order; it repeats until no lane
in that group needs another byte. Fifteen zero bytes follow the payload. They
make a final unaligned 16-byte CPU load safe without becoming part of the coded
stream.

AVX-512 can load each refill layer contiguously and expand it under the state
mask, while CUDA lanes read a coalesced span in warp order.

If every populated row is deterministic, the state table and refill payload
are omitted.

## Canonical validation

A conforming decoder rejects:

- unknown flag bits or modes;
- truncated selectors, models, states, or refill payloads;
- duplicate two-reference selectors;
- selectors outside the block anchor slab;
- invalid normalized-frequency totals;
- references selecting an absent model context;
- nonzero refill-interleaved padding;
- refill-interleaved payloads that are not consumed exactly;
- lanes that do not terminate at the rANS lower-bound state.

## Standalone container

The reference implementation wraps records in an experimental `.pgr` file.
This is deliberately separate from PGEN storage-mode assignment.
Version 2 files begin with `PGRANS2\0` and carry format version 2 in the
container header. Earlier experimental container identities are not supported.

The 64-byte file header stores the version, sample and variant counts, block
and anchor sizes, rANS parameters, restart interval, block count, and offsets
of the block table and block-data region. It is followed by a fixed 32-byte
entry for every block containing the first variant, variant count, file
offset, and byte count. This permits a client to issue one bounded ranged read
for a block without scanning earlier variant records. Each entry also stores a
CRC32C checksum of the complete serialized block.

Each block starts with a 16-byte header followed by:

1. one 24-bit record length per variant;
2. one 32-bit cumulative payload offset at each restart after the first;
3. the concatenated records.

To find a record, a reader starts from the closest preceding cumulative
restart and sums at most `restart_interval - 1` 24-bit lengths. The default
restart interval is 64 variants.

Version 2 readers validate that block-table entries cover the variants and
file exactly, blocks are contiguous, restart offsets agree with record
lengths, record lengths span each block payload, and the block CRC32C matches.

The reference reader exposes the file header and block table through a
caller-supplied `read_at(offset, length)` callback. Opening a file requires one
fixed-header read and one block-table read; fetching a block then requires
exactly one bounded read matching its table entry. A validated block view
points directly into that caller-owned byte buffer, so CPU and GPU adapters do
not need to copy every record into a separate allocation.

The CPU block decoder emits the standard variant-major packed two-bit
hardcall layout. It first decodes the scheduled anchor slab and then dispatches
all remaining records independently across a persistent worker pool. The
32-state record path advances the independent states in sample order and emits
one complete 64-bit packed genotype word per round.

On x86-64 GCC and Clang builds, default 32-state/12-bit records use a
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
decodes each required block once, and copies exact `(sample_count + 3) / 4`
byte packed hardcalls into the caller's existing per-variant stride. The most
recent decoded block is retained for adjacent or repeated requests. An
optional sorted sample index projects the fully decoded record into a smaller
packed output. Since rANS states cannot jump over arbitrary samples, this
projection reduces downstream work and buffer size but not entropy-decode
work.

## Reference CLI

Build from `2.0/`:

```sh
make -f Makefile.pgen_rans pgen_rans
```

CPU consumers can instead build `bin/libpgen_rans.a`:

```sh
make -f Makefile.pgen_rans libpgen_rans
```

Encode an unphased, biallelic hardcall PGEN:

```sh
bin/pgen_rans encode cohort.pgen cohort.pgr \
  --pvar cohort.pvar \
  --block-variants 128 \
  --anchors 32 \
  --two-ref-shortlist 4 \
  --threads 8
```

Then perform a byte-for-byte packed-hardcall round trip:

```sh
bin/pgen_rans verify cohort.pgen cohort.pgr
```

`inspect` validates the container and summarizes its record modes without
decoding genotypes:

```sh
bin/pgen_rans inspect cohort.pgr
```

`benchmark` deterministically samples block strata, decodes each block with a
persistent CPU worker pool, and compares every packed output word with
`PgrGet()` from the source PGEN:

```sh
bin/pgen_rans benchmark cohort.pgen cohort.pgr \
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
bin/pgen_rans_cuda_benchmark cohort.pgr \
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
