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
- bit 2: rANS state and lane payload are present;
- bits 3-7: reserved and zero.

One-reference records next store one byte containing the anchor ordinal.
Two-reference records store two distinct anchor ordinals. Version 1 therefore
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

## Warp-interleaved rANS payload

When any model row has more than one active symbol, the record stores:

1. one 32-bit initial state per lane;
2. one 32-bit cumulative boundary after every lane except the last;
3. the concatenated byte-renormalization streams for all lanes.

The intended lane count is 32. Sample `i` belongs to lane `i mod lane_count`.
Each lane is an independent bytewise-rANS stream, permitting one GPU warp to
decode 32 samples concurrently without a shared mutable input pointer.

Lane byte streams are stored in encoder emission order and consumed backward.
The record length supplies the final lane boundary.

If every populated row is deterministic, the state table, boundary table, and
lane payload are omitted.

## Canonical validation

A conforming decoder rejects:

- unknown flag bits or modes;
- truncated selectors, models, states, boundaries, or lane payloads;
- duplicate two-reference selectors;
- selectors outside the block anchor slab;
- invalid normalized-frequency totals;
- references selecting an absent model context;
- nonmonotonic lane boundaries;
- lanes that do not consume exactly their payload or terminate at the rANS
  lower-bound state.

## Standalone container

The reference implementation wraps records in an experimental `.pgr` file.
This is deliberately separate from PGEN storage-mode assignment.

The 64-byte file header stores the version, sample and variant counts, block
and anchor sizes, rANS parameters, restart interval, block count, and offsets
of the block table and block-data region. It is followed by a fixed 32-byte
entry for every block containing the first variant, variant count, file
offset, and byte count. This permits a client to issue one bounded ranged read
for a block without scanning earlier variant records.

Each block starts with a 16-byte header followed by:

1. one 24-bit record length per variant;
2. one 32-bit cumulative payload offset at each restart after the first;
3. the concatenated records.

To find a record, a reader starts from the closest preceding cumulative
restart and sums at most `restart_interval - 1` 24-bit lengths. The default
restart interval is 64 variants.

Version 1 readers validate that block-table entries cover the variants and
file exactly, blocks are contiguous, restart offsets agree with record
lengths, and record lengths span each block payload.
