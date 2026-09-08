"""Convert a frozen stagewise checkpoint/inference pair into the E5 schema."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.tim_validation.common.structure_signature import StructureSignature
from prism_benchmark.cpu_data import input_columns


FULL_MODEL = "PRISM_V2_1_1_PHYSICS_FIRST"


def _identity(value: Any) -> bool:
    text = str(value).upper()
    return "IDENTITY" in text or "EXACT_ZERO" in text


def _full_record(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    matches = [
        record for record in value["records"]
        if record.get("model") == FULL_MODEL and record.get("status") == "PASS"
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected one full-model record, got {len(matches)}")
    return matches[0]


def convert(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint_dir = next((args.checkpoint_root.resolve(strict=True) / "prism").iterdir())
    state = json.loads((checkpoint_dir / "checkpoint.json").read_text(encoding="utf-8"))
    record = _full_record(args.inference_manifest.resolve(strict=True))
    admitted = [str(channel) for channel in state["physical"]["channels"]]
    channels = input_columns(args.shared.resolve(strict=True), state["task"], args.proxy_policy)
    contracts = {str(item["channel"]): item for item in state["physical"]["channel_contracts"]}
    histories = {channel: int(contracts[channel]["profile"][1]) for channel in admitted}
    unique = sorted(set(histories.values()))
    scale_classes = {}
    for channel, history in histories.items():
        if len(unique) <= 1:
            scale_classes[channel] = "single_scale"
        elif history == unique[0]:
            scale_classes[channel] = "fast_scale"
        elif history == unique[-1]:
            scale_classes[channel] = "slow_scale"
        else:
            scale_classes[channel] = "intermediate_scale"
    c_family = state["c_contract"].get("family")
    w_family = state["w_contract"].get("family")
    a_family = state["a_contract"].get("family")
    flags = [bool(admitted), not _identity(c_family), not _identity(w_family), not _identity(a_family)]
    signature = StructureSignature(
        task=args.task, head=args.head, H=args.H, W=args.W,
        outer_fold="registered_outer_test", run=args.run, rod=args.rod, seed=args.seed,
        candidate_universe="standard", model_variant=FULL_MODEL,
        admitted_channels=admitted, rejected_channels=sorted(set(channels) - set(admitted)),
        selected_profile_by_channel={channel: contracts[channel]["profile"] for channel in admitted},
        selected_history_by_channel=histories,
        selected_scale_class_by_channel=scale_classes,
        K_admitted=flags[0], C_admitted=flags[1], W_admitted=flags[2], A_admitted=flags[3],
        selected_K_candidate={channel: contracts[channel]["k_contract"].get("family") for channel in admitted},
        selected_C_candidate=c_family, selected_W_candidate=w_family, selected_A_candidate=a_family,
        parameter_count=record.get("parameter_count"), active_channel_count=len(admitted),
        active_stage_count=sum(flags), RMSE=float(record["rmse"]), MAE=float(record["mae"]),
        R2=float(record["r2_level_reconstructed"]), support_id=str(record["scoring_support_hash"]),
        config_hash=str(state.get("selection_hash", "")),
        information_set=args.information_set,
        availability_scenario=args.availability_scenario,
        proxy_policy=args.proxy_policy,
    )
    signature.write(args.output.resolve())
    return signature.to_dict()


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument("--shared", type=Path, required=True)
    value.add_argument("--checkpoint-root", type=Path, required=True)
    value.add_argument("--inference-manifest", type=Path, required=True)
    value.add_argument("--output", type=Path, required=True)
    value.add_argument("--task", required=True)
    value.add_argument("--head", required=True)
    value.add_argument("--H", type=int, required=True)
    value.add_argument("--W", type=int, required=True)
    value.add_argument("--run", default="E1_stagewise")
    value.add_argument("--rod")
    value.add_argument("--seed", type=int, default=0)
    value.add_argument("--information-set", default="dynamic")
    value.add_argument("--availability-scenario", default="record_time")
    value.add_argument("--proxy-policy", default="primary")
    return value


def main() -> int:
    args = parser().parse_args()
    print(json.dumps(convert(args), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
