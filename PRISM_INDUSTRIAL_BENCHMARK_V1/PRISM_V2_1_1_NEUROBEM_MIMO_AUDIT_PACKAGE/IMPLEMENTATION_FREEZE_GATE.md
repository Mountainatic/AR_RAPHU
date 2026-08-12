# NeuroBEM implementation freeze gate

Status: `RESOLVED_BEFORE_MODEL_IMPLEMENTATION`.

The following result-sensitive choices are frozen in the machine-readable
configuration: flight-grouped split, segment history isolation, horizon,
target construction, motor encoding, K histories, K/W/A candidate families,
numerical certificates, selection guards, ERA order candidates, stability gate,
high-speed diagnostic threshold, bootstrap unit and runtime bounds.

No model-facing gate is presently unresolved. If Stage N0 finds that a frozen
choice is impossible under the published data, the experiment must stop with a
named protocol incompatibility. It must not silently substitute another choice.
