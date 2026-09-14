"""Generate and audit the isolated five-state, AG-only TFT manifests."""

from __future__ import annotations

import argparse
import hashlib
import h5py
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cropnet_protocol import CROPNET_FIVE_STATES
from mmst_vit.manifest import five_state_shared_rows, write_jsonl
from TFT_model.data import AG_DATES, build_ag_paths, resolve_ag_date_names

def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_sha256(audit_hashes: dict[str, str]) -> dict[str, str]:
    """Add code provenance without importing model or training modules."""
    source_paths = {
        "TFT_model/models.py": ROOT / "TFT_model" / "models.py",
        "TFT_model/train.py": ROOT / "TFT_model" / "train.py",
        "TFT_model/infer.py": ROOT / "TFT_model" / "infer.py",
        "TFT_model/data.py": ROOT / "TFT_model" / "data.py",
    }
    result = dict(audit_hashes)
    result.update({name: _sha256(path) for name, path in source_paths.items()})
    return dict(sorted(result.items()))


def _audit_file(path: Path, dates: list[str], fips: str, year: int, cache: dict) -> int:
    file_cache = cache.setdefault("files", {})
    if str(path) not in file_cache:
        result = {}
        if not path.is_file():
            file_cache[str(path)] = {"__file_error__": f"missing AG file {path}"}
        else:
            try:
                with h5py.File(path, "r") as handle:
                    for candidate_fips in handle.keys():
                        try:
                            county = handle[candidate_fips]
                            date_names = resolve_ag_date_names(county, dates, year)
                            local_grid = None
                            for date in dates:
                                date_group = county[date_names[date]]
                                if "data" not in date_group:
                                    raise ValueError(f"missing data dataset at {date}")
                                dataset = date_group["data"]
                                if dataset.dtype != "uint8" or dataset.ndim != 4 or dataset.shape[1:] != (224, 224, 3):
                                    raise ValueError(f"invalid data shape or dtype at {date}")
                                if local_grid is None:
                                    local_grid = int(dataset.shape[0])
                                elif local_grid != int(dataset.shape[0]):
                                    raise ValueError(f"inconsistent grid counts at {date}")
                            result[candidate_fips] = (None, local_grid)
                        except (KeyError, OSError, TypeError, ValueError) as error:
                            result[candidate_fips] = (str(error), None)
            except (OSError, TypeError, ValueError) as error:
                result = {"__file_error__": str(error)}
            file_cache[str(path)] = result
    entries = file_cache[str(path)]
    if "__file_error__" in entries:
        raise ValueError(entries["__file_error__"])
    error, grid_count = entries.get(fips, (f"missing FIPS group {fips}", None))
    if error:
        raise ValueError(error)
    return grid_count


def _audit_one(sample: dict, ag_root: Path, index: int, cache: dict) -> tuple[bool, list[Path], str | None]:
    paths: list[Path] = []
    try:
        paths = build_ag_paths(sample, ag_root)
        fips = str(sample["FIPS"]).strip()
        grid_count = None
        for path, dates in zip(paths, (AG_DATES[:6], AG_DATES[6:])):
            local_grid = _audit_file(path, dates, fips, int(sample["Year"]), cache)
            if grid_count is None:
                grid_count = local_grid
            elif grid_count != local_grid:
                raise ValueError(f"inconsistent grid counts at {path}")
        return True, paths, None
    except (OSError, TypeError, ValueError, KeyError) as error:
        return False, paths, f"sample {index}: {error}"


def _audit_ag_samples_with_results(samples, ag_root) -> tuple[dict, list]:
    root = Path(ag_root)
    invalid_samples = []
    valid_count = 0
    reasons = Counter()
    states = Counter()
    years = Counter()
    file_hashes = {}
    cache = {}
    results = []
    split_by_year = {year: split for split, years in {
        "train": (2017, 2018, 2019, 2020), "val": (2021,), "test": (2022,)
    }.items() for year in years}
    split_state_year = Counter()
    state_coverage = {
        state: Counter(input_count=0, valid_count=0, invalid_count=0)
        for state in CROPNET_FIVE_STATES
    }
    split_counts = Counter()
    invalid_split_count = 0
    for index, sample in enumerate(samples):
        original_sample = sample
        sample = sample if isinstance(sample, dict) else {}
        state = str(sample.get("State", "")).strip().lower()
        year = sample.get("Year")
        states[state] += 1
        years[str(year)] += 1
        if state in state_coverage:
            state_coverage[state]["input_count"] += 1
        ok, paths, error = _audit_one(sample, root, index, cache)
        split = split_by_year.get(year, "unknown") if isinstance(year, int) else "unknown"
        split_state_year[(split, state, str(year), "valid" if ok else "invalid")] += 1
        split_counts[(split, "input")] += 1
        split_counts[(split, "valid" if ok else "invalid")] += 1
        if split == "unknown":
            invalid_split_count += 1
        results.append((ok, paths, error))
        for path in paths:
            if path.is_file() and str(path) not in file_hashes:
                file_hashes[str(path)] = _sha256(path)
        if ok:
            valid_count += 1
            if state in state_coverage:
                state_coverage[state]["valid_count"] += 1
        else:
            if state in state_coverage:
                state_coverage[state]["invalid_count"] += 1
            reason = error or "unknown error"
            reasons[reason.split(": ", 1)[-1]] += 1
            invalid_samples.append({
                "FIPS": sample.get("FIPS"), "Year": year, "State": state,
                "raw_type": type(original_sample).__name__,
                "paths": [str(path) for path in paths], "error": reason,
            })
    audit = {
        "states": sorted(states), "years": dict(sorted(years.items())),
        "state_counts": dict(sorted(states.items())),
        "valid_count": valid_count, "invalid_count": len(invalid_samples),
        "invalid_reasons": dict(sorted(reasons.items())),
        "invalid_samples": invalid_samples, "sha256": dict(sorted(file_hashes.items())),
        "ag_dates": AG_DATES,
        "split_state_year_counts": {
            "/".join(key): value for key, value in sorted(split_state_year.items())
        },
        "split_counts": {
            split: {
                "input_count": split_counts[(split, "input")],
                "valid_manifest_count": split_counts[(split, "valid")],
                "invalid_count": split_counts[(split, "invalid")],
            }
            for split in ("train", "val", "test")
        },
        "invalid_split_count": invalid_split_count,
        "state_coverage": {
            state: dict(state_coverage[state]) for state in sorted(state_coverage)
        },
    }
    return audit, results


def audit_ag_samples(samples, ag_root) -> dict:
    """Audit every sample and return counts, reasons, identities, and file hashes."""
    audit, _ = _audit_ag_samples_with_results(samples, ag_root)
    return audit


def build_tft_ag_protocol(
    audit: dict,
    manifest_counts: dict[str, int],
    ag_root: str | Path,
    source_shared_jsonl: str | Path | None = None,
    representative: dict | None = None,
) -> dict:
    """Build the complete read-only protocol record from audit results."""
    representative = representative or {}
    grid_count = representative.get("grid_count")
    shape = [12, grid_count, 3, 224, 224] if grid_count is not None else None
    state_coverage = {
        state: audit.get("state_coverage", {}).get(
            state, {"input_count": 0, "valid_count": 0, "invalid_count": 0}
        )
        for state in sorted(CROPNET_FIVE_STATES)
    }
    verification = {
        "status": "data-only",
        "model_boundary": {
            "dataset_independent_of_model_construction": True,
            "tft_model_imported": False,
            "tft_forward_called": False,
            "training_run": False,
            "inference_run": False,
            "TFT_model/models.py_changed": False,
            "training_entrypoints_changed": False,
            "sentinel_original_files_changed": False,
            "task8_active_runtime_changed": False,
        },
        "representative_sample": {
            "FIPS": representative.get("FIPS"),
            "Year": representative.get("Year"),
            "State": representative.get("State"),
            "County": representative.get("County", ""),
            "ag_dates": AG_DATES,
            "ag_images_shape": shape,
            "dtype": "torch.float32",
            "grid_count": grid_count,
            "transform": "eval: CenterCrop(224) + Normalize([0.466, 0.471, 0.380], [0.195, 0.194, 0.192])",
            "seed": 0,
            "metadata_keys": ["FIPS", "Year", "State", "County", "grid_count", "ag_dates"],
            "excluded_keys": ["weather", "label"],
        },
        "source_sha256": _source_sha256(audit["sha256"]),
        "state_coverage": state_coverage,
    }
    return {
        "states": sorted(CROPNET_FIVE_STATES),
        "split": {"train": [2017, 2018, 2019, 2020], "val": [2021], "test": [2022]},
        "modality": "AG-only",
        "ag_dates": AG_DATES,
        "ag_path_count": 2,
        "ag_root": str(ag_root),
        "source_shared_jsonl": str(source_shared_jsonl) if source_shared_jsonl else None,
        "manifest_counts": manifest_counts,
        "split_counts": audit["split_counts"],
        "state_coverage": state_coverage,
        "model_boundary": verification["model_boundary"],
        "representative_sample": verification["representative_sample"],
        "source_sha256": verification["source_sha256"],
        "final_data_only_verification": verification,
    }


def build_tft_ag_manifest(shared_rows, ag_root, output_dir) -> dict[str, int]:
    """Write valid AG-only split rows and an audit report to ``output_dir``."""
    rows = five_state_shared_rows(shared_rows)
    audit, results = _audit_ag_samples_with_results(rows, ag_root)
    output = Path(output_dir)
    manifest_dir = output / "manifests"
    audit_dir = output / "audit"
    counts = {}
    split_indices = {"train": [], "val": [], "test": []}
    for index, sample in enumerate(rows):
        year = sample.get("Year") if isinstance(sample, dict) else None
        split = {2017: "train", 2018: "train", 2019: "train", 2020: "train", 2021: "val", 2022: "test"}.get(year)
        if split is not None:
            split_indices[split].append(index)
    for split, indices in split_indices.items():
        valid_rows = []
        for index in indices:
            sample = rows[index]
            ok, paths, _ = results[index]
            if ok:
                row = dict(sample)
                row["ag_paths"] = [str(path) for path in paths]
                valid_rows.append(row)
        write_jsonl(manifest_dir / f"{split}.jsonl", valid_rows)
        counts[split] = len(valid_rows)
    audit_dir.mkdir(parents=True, exist_ok=True)
    (audit_dir / "ag_integrity.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    representative = {}
    for index, sample in enumerate(rows):
        ok, paths, _ = results[index]
        if ok:
            representative = dict(sample)
            representative["grid_count"] = _audit_file(
                paths[0], AG_DATES[:6], str(sample["FIPS"]).strip(), int(sample["Year"]), {}
            )
            break
    protocol = build_tft_ag_protocol(audit, counts, ag_root, representative=representative)
    (audit_dir / "protocol.json").write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return counts


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shared-jsonl", type=Path, required=True)
    parser.add_argument("--ag-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    with args.shared_jsonl.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    counts = build_tft_ag_manifest(rows, args.ag_root, args.output_dir)
    audit = json.loads((args.output_dir / "audit" / "ag_integrity.json").read_text(encoding="utf-8"))
    representative = {}
    for split in ("train", "val", "test"):
        manifest = args.output_dir / "manifests" / f"{split}.jsonl"
        if manifest.exists() and manifest.read_text(encoding="utf-8").strip():
            representative = json.loads(manifest.read_text(encoding="utf-8").splitlines()[0])
            representative["grid_count"] = _audit_file(
                Path(representative["ag_paths"][0]), AG_DATES[:6], representative["FIPS"], representative["Year"], {}
            )
            break
    protocol = build_tft_ag_protocol(
        audit, counts, args.ag_root, source_shared_jsonl=args.shared_jsonl, representative=representative
    )
    audit_dir = args.output_dir / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    (audit_dir / "protocol.json").write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(counts, sort_keys=True))


if __name__ == "__main__":
    main()
