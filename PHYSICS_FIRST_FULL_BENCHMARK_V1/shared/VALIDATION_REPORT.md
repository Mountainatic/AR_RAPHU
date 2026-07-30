# Shared L6 Benchmark Dataset Validation

- Status: **PASS**
- Source data SHA256: `c3428966fe006572809156ee5e3f488264b8206b19b20887dcd00840bb26fbc3`
- Protocol SHA256: `357841c8cd64df2a70b605f212faa00afb78bdda7593fc01633094e058fcb9a8`
- Target: L6, 20-minute horizon, 2-minute output window, 40-minute history.
- All histories are causal and remain inside the frozen stable segment.
- PCA is fitted separately on each training rod and frozen for its test rod.
- The shared package contains no raw Excel workbook.

This package is immutable input for both CPU and GPU benchmark batches.
