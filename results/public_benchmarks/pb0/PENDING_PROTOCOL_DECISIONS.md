# PB1 protocol decisions required before development

Status: `BLOCKED_PENDING_USER_FREEZE`

The source, loader, causal-window and smoke gates do not authorize scientific
development runs. The v1.1 plan requires validation-only selection but does
not specify how validation is carved from official estimation data.

Recommended deterministic rule:

- PWH: hold out the final 20% of whole multisine phase-realization records,
  stratified across all five amplitude levels; never split a record.
- WHPN: hold out the final 20% of the ten estimation realizations; never split
  a realization.
- Cascaded Tanks: use the final 20% of the single estimation record as
  validation, preserving chronology and allowing only causal left history.
- Silverbox: use the final 20% of the multisine estimation record as
  validation, preserving chronology and allowing only causal left history.

Additional decisions:

- WHPN: keep raw unshifted channels as the primary protocol. The archive only
  warns that a one-sample inter-channel shift can occur, without freezing a
  direction. Any `-1/0/+1` sensitivity must be preregistered and selected only
  from validation.
- Cascaded Tanks: the official PDF describes overflow but supplies no numeric
  sample-level threshold. Overall official-test metrics can run after freeze;
  an overflow/non-overflow partition remains
  `BLOCKED_BY_MISSING_METADATA`.
- Silverbox: the official benchmark page and archive do not state an explicit
  data license. Local analysis may be kept separate, but raw redistribution
  remains disabled and the PB0 license gate remains
  `BLOCKED_BY_MISSING_METADATA`.
