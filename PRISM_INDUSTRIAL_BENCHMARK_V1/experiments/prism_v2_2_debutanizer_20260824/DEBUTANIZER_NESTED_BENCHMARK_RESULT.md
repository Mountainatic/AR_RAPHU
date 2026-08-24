# RETRACTED — invalid cross-dataset Debutanizer comparison

Date retracted: 2026-08-24

## Status

**DO NOT USE THIS FILE AS BENCHMARK EVIDENCE.**

The earlier experiment in this path compared PRISM results computed on the Fortuna-style 2394×8 Debutanizer public sequence against published MSA-HDMDc numbers from a different Debutanizer data source/protocol (ERGMED 2004 monthly March/May/July/September data).

Although both are called “Debutanizer” in the literature, they are not established as the same raw sequence. Dividing the Fortuna sequence into four chronological quarters and naming them after the MSA-HDMDc months was therefore not a valid same-dataset reconstruction.

Consequently, all earlier statements in this file such as “PRISM exceeds MSA-HDMDc at 30/60 min” are **withdrawn** and must not be cited, summarized, or used in PRISM performance claims.

The historical content remains recoverable from Git history solely for audit purposes.

## Replacement benchmark

Use the protocol-matched RTA-TCN benchmark instead:

`../prism_v2_2_debutanizer_rta_tcn_20260824/PRISM_V22_VS_RTA_TCN_PROTOCOL_BENCHMARK.md`

That replacement uses:
- the exact Debutanizer file published by the RTA-TCN authors;
- the author-released model code;
- the same 40-sample history budget;
- the same seven process inputs;
- no target history for either model;
- the same chronological training boundary;
- an exact-author parser audit and a corrected-header parser audit.

## Isolation

This retraction and its replacement belong only to the isolated PRISM v2.2(beta) benchmark branch family. They do not redefine PRISM v2.1.1 or the frozen `prism-v2-2-beta-ct` theory branch.
