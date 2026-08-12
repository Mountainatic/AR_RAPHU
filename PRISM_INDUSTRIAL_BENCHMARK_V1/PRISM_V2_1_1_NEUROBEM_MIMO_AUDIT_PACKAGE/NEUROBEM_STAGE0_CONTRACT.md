# NeuroBEM Stage-0 contract

All material result-affecting choices for the first NeuroBEM PRISM run are
resolved by the user authorization to design the dataset-specific experiment
and frozen in `PRISM_V2_1_1_NEUROBEM_MIMO_CONFIG_FROZEN.json` before processed
numeric values are opened.

Stage N0 may inspect archive integrity, filenames, CSV headers, row counts,
finite values, timestamps, cadence and segment boundaries. It may compute
source hashes and immutable split/sample registries. It may not fit a model,
choose a threshold from target values, inspect official-test target statistics,
or change any registered candidate after seeing development performance.

Hard stops before fitting are:

- source hash or ZIP integrity failure;
- any mismatch between the 95 distributed `Flights.txt` parent IDs and the 95
  processed-archive parent IDs (the paper's reported count of 96 is retained as
  a documented source discrepancy, not silently rewritten);
- missing required columns;
- a segment with non-monotone time, nonfinite required values or unexplained
  large internal gaps;
- parent-flight overlap across train/validation/test;
- any sample whose feature/residual history crosses its segment;
- any test numeric access before an N6 freeze explicitly permits it.

The published processed segments are the minimum history entities. Treating only
parent flight IDs as history entities would incorrectly bridge removed
non-flight/Vicon-dropout regions and is forbidden.

The first metadata-only cadence audit found one distributed parent flight,
`2021-02-18-16-43-54`, whose four segments are uniformly sampled at about
164 Hz rather than the registered 400 Hz. It is excluded as one whole parent
before any model fit. It is not resampled or split, and it was assigned to train,
so the frozen validation and test parent sets are unchanged. The executable
universe is 63 train + 19 validation + 13 locked test parents; one additional
parent has status `EXCLUDED_INCOMPATIBLE_CADENCE`.
