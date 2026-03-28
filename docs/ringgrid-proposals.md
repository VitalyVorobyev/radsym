# Ringgrid Proposal Search

This note describes the current proposal-search path used by
`crates/radsym-py/examples/detect_ringgrid.py`.

## Goal

Produce a high-recall set of center proposals for ring markers on the board.
At this stage we prefer recall over perfect precision because:

- the JSON fixture is partial ground truth
- extra valid seeds on the same hex lattice are useful for later releases
- ellipse fitting and future lattice reasoning can filter or promote candidates

## Current Search Path

1. Load grayscale image with `radsym.load_grayscale`.
2. Compute Sobel gradients with `radsym.sobel_gradient`.
3. Build an outer-radius band around the size prior.
   Current default band: `0.8x .. 1.16x` of the outer radius, sampled at 5 radii.
4. Run a radial symmetry response over that band.
   Default in the demo: `RSD`, optional `FRST`.
5. Extract local maxima with image-space NMS.
   Current demo default: NMS radius `0.55 * outer_radius`, threshold `0.01`.
6. Greedily suppress nearby proposals.
   Current demo default: keep strongest proposals separated by at least
   `1.25 * outer_radius`.

The result is a proposal set that covers all annotated ringgrid markers in the
fixture while still allowing additional unlabeled lattice-consistent seeds.

## Why Outer-Radius Search

The ring markers have stronger and more stable response at the outer boundary
than at the inner boundary. Searching directly at the outer radius gave the
best center recall on `testdata/ringgrid.png`.

## Performance Notes

The dominant costs today are:

- multi-radius response accumulation
- gradient computation
- optional ellipse refinement after proposal generation

Both `FRST` and `RSD` now parallelize across radii behind the `rayon` feature.
The Python demo prints a per-call timing table so proposal-stage changes can be
compared quickly.

## Next Optimization Targets

- reduce response-map passes or fuse work across radii
- revisit smoothing cost per radius
- avoid Python-side overhead in proposal-only visualization flows
- exploit the regular hex lattice to reduce redundant candidates earlier
