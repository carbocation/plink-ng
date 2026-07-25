# PGEN entropy oracle prototype

`pgen_entropy_oracle` estimates the storage payoff of a GPU-oriented
four-symbol rANS main track for unphased biallelic hardcalls.

This is an experimental diagnostic, not a PGEN writer.  It reads a PGEN,
calculates exact genotype counts and 4x4 transition tables, and compares:

1. the current PGEN variant-record payload;
2. marginal rANS;
3. a previous-variant-window conditional lower bound, where references may
   chain;
4. a legal depth-one schedule with independently encoded, evenly spaced
   anchors in every block.
5. optionally, a legal depth-one `4x4x4` model conditioned on two independent
   anchors.

The scheduled lines are the main decision numbers.  They model the proposed
GPU-friendly format: decode the anchor slab first, then decode all targets
independently under one- or two-anchor conditional models.  The
previous-window line shows how much headroom remains if anchor scheduling is
improved, but its dependency chains make it unsuitable as a file layout.

Pass `--two-ref-shortlist L` to retain the `L` best single-anchor candidates
for each target and test every pair among them under
`P(g_target | g_anchor1, g_anchor2)`.  Both references remain in the
independent anchor slab, so decode depth stays at one.  The estimate charges
for the larger 16-row model and for an anchor-pair selector.  `L` equal to the
block's anchor count is exhaustive; smaller values are explicitly a
shortlisted estimate, since exhaustive pair search grows quadratically.

The estimated rANS byte counts include:

- empirical payload cost after frequency quantization;
- a compact model description;
- 32-bit initial states and 32-bit per-lane offsets;
- three bytes of record-length index per variant;
- a 32-bit cumulative restart offset at the requested interval;
- a 64-bit block offset.

They do not include a working rANS encoder or GPU decoder.  Treat the estimates
as a way to decide whether those are worth implementing.  In particular, the
oracle does not predict kernel throughput; the go/no-go decision still needs
an end-to-end localization + GPU decode + analysis benchmark.

## Build

From `2.0/`:

```sh
make -f Makefile.pgen_entropy_oracle
```

The standalone makefile reuses PLINK 2's existing build definitions without
modifying the upstream makefiles or adding the oracle to `make all`.

## Example

```sh
bin/pgen_entropy_oracle cohort.pgen \
  --pvar cohort.pvar \
  --block-variants 128 \
  --anchors 8 \
  --previous-window 8 \
  --max-anchor-bp 1000000 \
  --rans-states 32 \
  --project-samples 500000 \
  --threads 32 \
  --out-variants cohort.oracle.tsv
```

The first prototype accepts plain-text PVAR only.  Without `--pvar`, it cannot
reset at chromosome boundaries or enforce physical-distance limits.

Use `--variant-limit` for fast smoke tests before running a complete biobank
file.  With this option, the current-PGEN baseline is the analyzed record
payload rather than the full file, while the candidate estimates still include
their restart/index allowance; full-file runs are the comparison to use for
decisions.

For a representative minutes-scale estimate on a large biobank PGEN, use
`--sample-blocks`.  It deterministically selects complete blocks from evenly
spaced strata across the file, so block-local LD is retained while most
genotype records are never read:

```sh
bin/pgen_entropy_oracle cohort.pgen \
  --pvar cohort.pvar \
  --block-variants 128 \
  --anchors 32 \
  --sample-blocks 128 \
  --two-ref-shortlist 4 \
  --threads 4
```

Sample mode reports both the sampled payload comparison and a whole-file
extrapolation.  It cannot be combined with `--variant-limit`.  The
extrapolation is an oracle estimate, not a substitute for an eventual
end-to-end full-file decoder benchmark.

For the initial 1000 Genomes experiments on GCP N2, build in the existing
PLINK 2 checkout and write only the input PGEN/PVAR and the optional TSV to
local scratch:

```sh
make -C 2.0 -f Makefile.pgen_entropy_oracle

2.0/bin/pgen_entropy_oracle /local/1kg-unphased.pgen \
  --pvar /local/1kg-unphased.pvar \
  --block-variants 128 \
  --anchors 8 \
  --previous-window 8 \
  --restart-variants 64 \
  --project-samples 500000 \
  --threads 32 \
  --out-variants /local/1kg.oracle.tsv
```

Useful first comparisons are 8, 16, and 32 anchors with 64- and 128-variant
blocks.  Input should already be restricted to the hardcall, biallelic MAF
range of interest.  A phased 1000 Genomes PGEN should first be converted to an
unphased scratch fileset with PLINK 2's `--make-pgen erase-phase
erase-dosage`; otherwise the current-PGEN baseline contains data the proposed
four-symbol codec does not represent.

`--project-samples` scales the observed genotype and transition probabilities
to another sample count while retaining per-record model, rANS-state, and
index overhead.  Reference selection and the marginal/conditional decision are
re-optimized for the projected record sizes rather than inherited from the
source sample count.  Its current-PGEN comparison is only a linear payload
projection; it does not rerun PGEN's codec selection on a resampled cohort.
This is useful for separating 1000 Genomes' small-N record overhead from
expected N=500k behavior, but a genuinely expanded real-LD fixture is the
stronger PGEN baseline.

## Interpretation

The previous-window result is an optimistic lower bound because a target may
reference another conditional target.  The scheduled result is directly
decodable in two breadth-first steps:

1. decode all independent anchors;
2. decode all dependent variants.

Scheduled anchors are currently selected at evenly spaced positions.  An
optimized anchor-selection pass may improve this result later.

The initial tool intentionally rejects dosage, phase, and multiallelic tracks;
otherwise current PGEN byte counts and the proposed four-symbol hardcall
representation would not be comparable.

The two-reference line selects the best of marginal, one-reference, and
two-reference encoding independently for every target.  A target counted as a
two-reference call therefore has enough payload savings to pay the estimated
larger model and selector cost.  The storage oracle still does not account for
the extra GPU lookup and larger decoder table; that belongs in the eventual
end-to-end benchmark.
