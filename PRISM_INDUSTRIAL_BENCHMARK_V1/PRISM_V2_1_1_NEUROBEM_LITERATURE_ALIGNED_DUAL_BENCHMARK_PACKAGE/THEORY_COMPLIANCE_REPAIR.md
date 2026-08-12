# NeuroBEM theory-interface compliance repair

The canonical PRISM v2.1.1 theory is unchanged. This file repairs only the
classification and routing of the NeuroBEM implementation used by the new
dual benchmark.

Raw actuator history is registered as causal `K`. Linear/angular motion may be
used as predictive context but must carry the label
`PREDICTIVE_MOTION_CONTEXT`, never a causal actuator interpretation. Products
among motion components and products of motor-response latents with motion
context are `C` interactions. Canonical `W` is applied only to the frozen `C`
latent and may be identity, natural-cubic latent curvature, or signed-quadratic
latent curvature.

The already completed `SIGNED_QUADRATIC_AERO_CONTEXT` and
`NATURAL_CUBIC_SPEED_CONTEXT` experiments are neither deleted nor relabeled as
canonical `W`; they remain
`AERODYNAMIC_CONTEXT_W_EXTENSION_DIAGNOSTIC`. No historical result, prediction,
freeze, or generating commit is modified.

Machine-readable assertions: `THEORY_CHANGED=false`,
`NEUROBEM_IMPLEMENTATION_CLASSIFICATION_CHANGED=true`, and
`HISTORICAL_RESULTS_PRESERVED=true`.
