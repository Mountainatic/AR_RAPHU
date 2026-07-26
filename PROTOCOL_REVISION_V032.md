# PROTOCOL REVISION V032

Revision:
Spectral PS-AR-RAPHU v0.3.1 -> v0.3.2

Frozen old result:
E2A stopped with 15/60 passing rows.
KKT residuals were approximately 1e-16.

Newly identified protocol problems:
1. The amplitude spline silently clipped values outside train Q01-Q99.
2. A majority of 64-step windows contained at least one clipped value.
3. Frozen AR-S4 is a conditional kernel K(tau,u;c), not the tested 2D K(tau,u).
4. Natural-input contribution prediction and full rectangular-surface recovery
   were incorrectly combined into one capacity gate.

Interpretation:
The old stop action remains valid under v0.3.1.
The old scientific label is not treated as a final rejection of the full-kernel method.
