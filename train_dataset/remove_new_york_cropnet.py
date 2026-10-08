"""Audit and optionally remove New York data from the CropNet source tree."""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


MIN_YEAR = 2017
MAX_YEAR = 2022
CSV_PREFIX = "USDA_Corn_County_"


@dataclass(frozen=True)
class CleanupTarget:
    kind: str
    path: Path
    file_count: int
    byte_count: int


@dataclass(frozen=True)
class CleanupPlan:
    data_root: Path
    years: tuple[int, ...]
    targets: tuple[CleanupTarget, ...]
    csv_paths: tuple[Path, ...]
    audit_snapshot: tuple[tuple[str, str, int, int, str], ...] = ()


class CleanupExecutionError(RuntimeError):
    def __init__(
        self,
        message: str,
        completed_directories: tuple[Path, ...] = (),
        completed_csvs: tuple[Path, ...] = (),
    ):
        self.completed_directories = completed_directories
        self.completed_csvs = completed_csvs
        super().__init__(
            f"{message}; completed_csvs={list(completed_csvs)}; "
            f"completed_directories={list(completed_directories)}"
        )


@dataclass(frozen=True)
class _PreparedCSV:
    source: Path
    temporary: Path
    removed: int


def _canonical(path: Path) -> Path:
    return path.absolute()


def _reject_symlink_components(path: Path, root: Path) -> None:
    relative = path.relative_to(root)
    current = root
    for component in relative.parts:
        current /= component
        if current.is_symlink():
            raise ValueError(f"symlink is not allowed in cleanup path: {current}")


def _rooted(path: Path, root: Path) -> Path:
    normalized_root = _canonical(root)
    normalized = _canonical(path)
    try:
        normalized.relative_to(normalized_root)
    except ValueError as exc:
        raise ValueError(f"target escapes data_root: {normalized}") from exc
    _reject_symlink_components(normalized, normalized_root)
    return normalized


def _validate_years(years: Iterable[int]) -> tuple[int, ...]:
    normalized = tuple(int(year) for year in years)
    invalid = [year for year in normalized if not MIN_YEAR <= year <= MAX_YEAR]
    if invalid:
        raise ValueError(f"years must be between {MIN_YEAR} and {MAX_YEAR}: {invalid}")
    if len(set(normalized)) != len(normalized):
        raise ValueError("years must not contain duplicates")
    return normalized


def _count_files(path: Path) -> tuple[int, int]:
    if not path.is_dir():
        return 0, 0
    file_count = 0
    byte_count = 0
    for current, directories, files in os.walk(path, followlinks=False):
        current_path = Path(current)
        for directory in directories:
            if (current_path / directory).is_symlink():
                raise ValueError(f"symlink is not allowed in cleanup directory: {current_path / directory}")
        for filename in files:
            file_path = current_path / filename
            if file_path.is_symlink():
                raise ValueError(f"symlink is not allowed in cleanup directory: {file_path}")
            file_count += 1
            byte_count += file_path.stat().st_size
    return file_count, byte_count


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _audit_snapshot(
    root: Path,
    targets: tuple[CleanupTarget, ...],
    csv_paths: tuple[Path, ...],
) -> tuple[tuple[str, str, int, int, str], ...]:
    snapshot = []
    for target in targets:
        if not target.path.exists():
            snapshot.append(("target", str(target.path.relative_to(root)), 0, 0, "missing"))
            continue
        for current, directories, files in os.walk(target.path, followlinks=False):
            current_path = Path(current)
            for name in sorted(directories + files):
                item = current_path / name
                if item.is_symlink():
                    raise ValueError(f"symlink is not allowed in cleanup path: {item}")
                item_stat = item.stat()
                digest = _file_digest(item) if item.is_file() else "directory"
                snapshot.append(("target", str(item.relative_to(root)), item_stat.st_mode, item_stat.st_size, digest))
    for csv_path in csv_paths:
        item_stat = csv_path.stat()
        snapshot.append(("csv", str(csv_path.relative_to(root)), item_stat.st_mode, item_stat.st_size, _file_digest(csv_path)))
    return tuple(snapshot)


def build_cleanup_plan(data_root: Path | str, years: Iterable[int]) -> CleanupPlan:
    root = _canonical(Path(data_root))
    if root.is_symlink():
        raise ValueError(f"data_root must not be a symlink: {root}")
    if not root.is_dir():
        raise ValueError(f"data_root is not a directory: {root}")
    selected_years = _validate_years(years)
    targets = []
    csv_paths = []
    for year in selected_years:
        for kind, relative in (
            ("weather", Path("weather") / str(year) / "NY"),
            ("ag", Path("sentinel") / "source" / "AG" / str(year) / "NY"),
        ):
            target_path = _rooted(root / relative, root)
            file_count, byte_count = _count_files(target_path)
            targets.append(CleanupTarget(kind, target_path, file_count, byte_count))
        csv_path = _rooted(root / "label" / "hf" / f"{CSV_PREFIX}{year}.csv", root)
        if not csv_path.is_file() or csv_path.is_symlink():
            raise FileNotFoundError(csv_path)
        csv_paths.append(csv_path)
    target_tuple = tuple(targets)
    csv_tuple = tuple(csv_paths)
    return CleanupPlan(root, selected_years, target_tuple, csv_tuple, _audit_snapshot(root, target_tuple, csv_tuple))


def _new_york_mask(frame: pd.DataFrame) -> pd.Series:
    if "state_name" not in frame.columns and "state_ansi" not in frame.columns:
        raise ValueError("CSV must contain state_name or state_ansi")
    mask = pd.Series(False, index=frame.index)
    if "state_name" in frame.columns:
        mask |= frame["state_name"].fillna("").astype(str).str.strip().str.upper().eq("NEW YORK")
    if "state_ansi" in frame.columns:
        ansi = frame["state_ansi"].fillna("").astype(str).str.strip()
        mask |= ansi.str.replace(r"\.0+$", "", regex=True).str.zfill(2).eq("36")
    return mask


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def _prepare_csv(csv_path: Path) -> _PreparedCSV:
    path = _canonical(Path(csv_path))
    if path.is_symlink():
        raise ValueError(f"CSV must not be a symlink: {path}")
    frame = _read_csv(path)
    mask = _new_york_mask(frame)
    removed = int(mask.sum())
    filtered = frame.loc[~mask].copy()
    original_mode = stat.S_IMODE(path.stat().st_mode)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    temporary_path = Path(temporary_name)
    try:
        serialized = filtered.to_csv(index=False)
        temporary_path.write_text(serialized)
        os.chmod(temporary_path, original_mode)
        check = _read_csv(temporary_path)
        expected = _read_csv_from_text(serialized)
        pd.testing.assert_frame_equal(check, expected, check_dtype=True)
        if _new_york_mask(check).any():
            raise ValueError(f"atomic verification found New York rows for {path}")
        if stat.S_IMODE(temporary_path.stat().st_mode) != original_mode:
            raise ValueError(f"temporary permissions changed for {path}")
        return _PreparedCSV(path, temporary_path, removed)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def _read_csv_from_text(content: str) -> pd.DataFrame:
    from io import StringIO

    return pd.read_csv(StringIO(content), dtype=str, keep_default_na=False)


def _cleanup_prepared(prepared: Iterable[_PreparedCSV]) -> None:
    for item in prepared:
        item.temporary.unlink(missing_ok=True)


def _commit_prepared(prepared: tuple[_PreparedCSV, ...]) -> tuple[Path, ...]:
    completed = []
    current_source = None
    try:
        for item in prepared:
            current_source = item.source
            os.replace(item.temporary, item.source)
            completed.append(item.source)
    except OSError as exc:
        raise CleanupExecutionError(
            f"CSV replacement failed: {current_source}: {exc}",
            completed_csvs=tuple(completed),
        ) from exc
    finally:
        _cleanup_prepared(prepared)
    return tuple(completed)


def remove_new_york_rows(csv_path: Path | str, apply: bool = False) -> int:
    if not apply:
        path = _canonical(Path(csv_path))
        if path.is_symlink():
            raise ValueError(f"CSV must not be a symlink: {path}")
        return int(_new_york_mask(_read_csv(path)).sum())
    prepared = _prepare_csv(Path(csv_path))
    _commit_prepared((prepared,))
    return prepared.removed


def _print_plan(plan: CleanupPlan) -> None:
    for target in plan.targets:
        print(f"{target.kind}: {target.path} files={target.file_count} bytes={target.byte_count}")
    for csv_path in plan.csv_paths:
        frame = _read_csv(csv_path)
        removed = int(_new_york_mask(frame).sum())
        print(f"csv: {csv_path} before_rows={len(frame)} remove_rows={removed} after_rows={len(frame) - removed}")


def execute_cleanup(plan: CleanupPlan, apply: bool = False, data_root: Path | str | None = None) -> None:
    if not isinstance(plan, CleanupPlan):
        raise ValueError("plan must be a CleanupPlan")
    if data_root is not None and _canonical(Path(data_root)) != plan.data_root:
        raise ValueError("supplied data_root does not match the cleanup plan")
    current_plan = build_cleanup_plan(plan.data_root, plan.years)
    if current_plan != plan:
        raise ValueError("cleanup plan no longer matches the audited filesystem")
    _print_plan(plan)
    if not apply:
        print("mode=dry-run; no changes made")
        return
    prepared = []
    try:
        for csv_path in plan.csv_paths:
            prepared.append(_prepare_csv(csv_path))
    except BaseException:
        _cleanup_prepared(prepared)
        raise
    completed_csvs = _commit_prepared(tuple(prepared))
    for item in prepared:
        print(f"applied csv: {item.source} removed_rows={item.removed}")
    completed_directories = []
    for target in plan.targets:
        if target.path.is_dir():
            try:
                shutil.rmtree(target.path)
            except OSError as exc:
                raise CleanupExecutionError(
                    f"directory removal failed: {target.path}: {exc}",
                    tuple(completed_directories),
                    completed_csvs,
                ) from exc
            completed_directories.append(target.path)
            print(f"removed directory: {target.path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--years", required=True, nargs="+", type=int)
    parser.add_argument("--apply", action="store_true", help="modify files; default is dry-run")
    args = parser.parse_args()
    plan = build_cleanup_plan(args.data_root, args.years)
    execute_cleanup(plan, apply=args.apply)


if __name__ == "__main__":
    main()
