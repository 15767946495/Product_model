"""Safe command-line queries for the experiment registry."""

import argparse
import json
import sys
from dataclasses import asdict
from typing import Sequence

from .protocol import DEFAULT_PROTOCOL
from .registry import EXECUTABLE_RUNNERS, EXPERIMENTS, get_experiment, validate_experiment


def _protocol_dict() -> dict[str, object]:
    return asdict(DEFAULT_PROTOCOL)


def _experiment_dict(experiment) -> dict[str, object]:
    return {
        "experiment_id": experiment.experiment_id,
        "group": experiment.group,
        "title": experiment.title,
        "research_question": experiment.research_question,
        "baseline_id": experiment.baseline_id,
        "changes": dict(experiment.changes),
        "fixed_factors": dict(experiment.fixed_factors),
        "required_capabilities": sorted(capability.value for capability in experiment.required_capabilities),
        "missing_capabilities": [capability.value for capability in experiment.missing_capabilities],
        "metrics": list(experiment.metrics),
        "output_name": experiment.output_name,
        "runner": experiment.runner,
        "declared_status": experiment.declared_status.value if experiment.declared_status else None,
        "status": experiment.effective_status.value,
    }


def _write_json(payload: dict[str, object]) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _unknown_experiment(experiment_id: str) -> int:
    print(json.dumps({"error": f"unknown experiment: {experiment_id}"}, ensure_ascii=False), file=sys.stderr)
    return 2


def _validate(experiment_id: str) -> int:
    try:
        experiment, errors = validate_experiment(experiment_id)
    except KeyError:
        return _unknown_experiment(experiment_id)
    payload = {
        "experiment_id": experiment.experiment_id,
        "status": experiment.effective_status.value,
        "missing_capabilities": [capability.value for capability in experiment.missing_capabilities],
        "errors": list(errors),
    }
    _write_json(payload)
    return 1 if experiment.effective_status.value == "blocked" or errors else 0


def _dry_run(experiment_id: str) -> int:
    try:
        experiment, validation_errors = validate_experiment(experiment_id)
    except KeyError:
        return _unknown_experiment(experiment_id)
    if validation_errors:
        _write_json({
            "experiment_id": experiment.experiment_id,
            "status": experiment.effective_status.value,
            "missing_capabilities": [capability.value for capability in experiment.missing_capabilities],
            "errors": list(validation_errors),
        })
        return 1
    command = EXECUTABLE_RUNNERS.get(experiment.runner or "")
    if experiment.effective_status.value not in {"ready", "external"} or command is None:
        _write_json({
            "experiment_id": experiment.experiment_id,
            "status": experiment.effective_status.value,
            "output_dir": experiment.output_name,
            "protocol": _protocol_dict(),
            "errors": ["unknown runner; no command generated"],
        })
        return 1
    _write_json({
        "experiment_id": experiment.experiment_id,
        "status": experiment.effective_status.value,
        "output_dir": experiment.output_name,
        "protocol": _protocol_dict(),
        "command": command,
    })
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="TFT-MMST experiment registry")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("list")
    for name in ("show", "validate", "dry-run"):
        child = subparsers.add_parser(name)
        child.add_argument("experiment_id")
    args = parser.parse_args(argv)

    if args.command == "list":
        print("\n".join(
            f"{experiment.experiment_id}\t{experiment.group}\t{experiment.title}\t{experiment.effective_status.value}"
            for experiment in EXPERIMENTS.values()
        ))
        return 0
    if args.command == "show":
        try:
            experiment = get_experiment(args.experiment_id)
        except KeyError:
            return _unknown_experiment(args.experiment_id)
        payload = _experiment_dict(experiment)
        payload["protocol"] = _protocol_dict()
        _write_json(payload)
        return 0
    if args.command == "validate":
        return _validate(args.experiment_id)
    return _dry_run(args.experiment_id)


if __name__ == "__main__":
    raise SystemExit(main())
