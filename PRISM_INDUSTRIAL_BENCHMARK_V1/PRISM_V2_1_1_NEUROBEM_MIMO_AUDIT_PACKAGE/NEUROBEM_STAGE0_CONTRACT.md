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
- flight metadata count other than 96 without a documented publisher reason;
- missing required columns;
- a segment with non-monotone time, nonfinite required values or unexplained
  large internal gaps;
- parent-flight overlap across train/validation/test;
- any sample whose feature/residual history crosses its segment;
- any test numeric access before an N6 freeze explicitly permits it.

The published processed segments are the minimum history entities. Treating only
the 96 parent flights as history entities would incorrectly bridge removed
non-flight/Vicon-dropout regions and is forbidden.
