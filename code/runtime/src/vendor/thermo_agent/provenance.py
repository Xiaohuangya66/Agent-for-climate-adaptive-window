from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator


EXCLUDED_PARTS = {
    ".venv",
    ".venv1",
    ".git",
    ".pytest_cache",
    ".codex_qa",
    "thermochromic_AI_inverse_design_upload_2026-06-30",
}


@dataclass(frozen=True)
class AssetRecord:
    relative_path: str
    role: str
    kind: str
    city: str
    status: str
    size_bytes: int
    modified_utc: str
    sha256: str
    csv_rows: int | None = None
    csv_columns: int | None = None
    csv_header: str | None = None
    note: str = ""


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def csv_schema(path: Path) -> tuple[int, int, str]:
    """Return physical row count and first-row schema without loading the file.

    Project CSV files do not contain embedded newlines. Counting newline bytes is
    substantially faster for the 100+ MiB spectrum tables than parsing every cell.
    """
    newline_count = 0
    last_byte = b""
    first_line = b""
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            if not first_line:
                first_line = chunk.splitlines(keepends=False)[0]
            newline_count += chunk.count(b"\n")
            last_byte = chunk[-1:]
    row_count = newline_count + (1 if path.stat().st_size and last_byte != b"\n" else 0)
    decoded = first_line.decode("utf-8-sig", errors="replace")
    first_row = next(csv.reader([decoded]), [])
    return row_count, len(first_row), json.dumps(first_row, ensure_ascii=False)


def _city_for(path: Path) -> str:
    lower = path.as_posix().lower()
    if lower in {
        "energyplus/ml_dataset_final.csv",
        "energyplus/surrogate_model.pkl",
    }:
        return "Nanjing"
    if "beijing" in lower:
        return "Beijing"
    if "nanjing" in lower:
        return "Nanjing"
    if "singapore" in lower or "singpore" in lower:
        return "Singapore"
    if "losangeles" in lower or "los.angeles" in lower:
        return "LosAngeles"
    return ""


def _role_for(path: Path) -> tuple[str, str, str, str]:
    p = path.as_posix()
    lower = p.lower()
    suffix = path.suffix.lower()
    kind = suffix.lstrip(".") or "file"
    city = _city_for(path)
    status = "canonical"
    note = ""

    if lower.startswith("dataset/"):
        role = "transformer_dataset"
    elif lower.startswith("checkpoints/"):
        role = "transformer_checkpoint"
    elif lower.startswith("tmm/tmm_ml/"):
        role = "optical_constants"
    elif lower.startswith("tmm/"):
        role = "tmm_code_or_result"
    elif lower.startswith("transopto_trl/checkpoints/"):
        role, status = "runtime_checkpoint_copy", "mirror"
    elif lower.startswith("transopto_trl/dataset/udi_energy/") and suffix == ".pkl":
        role, status = "runtime_surrogate_copy", "mirror"
    elif lower.startswith("transopto_trl/dataset/"):
        role, status = "rl_runtime_data", "mirror"
    elif lower.startswith("transopto_trl/result/"):
        role = "historical_rl_result"
    elif lower.startswith("transopto_trl/"):
        role = "rl_code"
    elif lower.startswith("energyplus/sim_results/"):
        role, status = "energyplus_temporary_run", "temporary"
    elif lower.startswith("energyplus/baseline_clear glass/"):
        role = "clear_glass_baseline"
    elif suffix == ".epw":
        role = "weather"
    elif suffix == ".idf":
        role = "building_model"
    elif suffix == ".pkl" and lower.startswith("energyplus/"):
        role = "city_surrogate"
    elif lower.startswith("energyplus/"):
        role = "energyplus_code_or_data"
    elif suffix == ".ckpt":
        role = "checkpoint"
    elif suffix == ".csv":
        role = "root_data_or_result"
    elif suffix == ".py":
        role = "root_code"
    elif suffix in {".docx", ".xlsx", ".out"}:
        role = "documentation_or_log"
    else:
        role = "other"

    if path.name == "inference_1.py":
        status, note = "known_broken", "Syntax error and stale import; do not use as entry point."
    if path.name == "290552.out":
        status, note = "failed_training", "Large Transformer training reached persistent NaN."
    if p == "energyplus/surrogate_model.pkl":
        status, note = "mirror", "Byte-identical copy of surrogate_model_Nanjing.pkl."
    if p == "energyplus/sim_results/run_12/eplusout.err":
        status, note = "failed_simulation", "Known severe-error regression fixture."
    return role, kind, city, status + (f": {note}" if note else "")


def iter_asset_paths(root: Path) -> Iterator[Path]:
    """Yield research assets while excluding environments, archives, and generated Agent files."""
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if any(part in EXCLUDED_PARTS for part in relative.parts):
            continue
        if relative.parts and relative.parts[0] == "agent_project":
            continue
        if path.name.endswith(".zip") or path.name.endswith(".7z"):
            continue
        if relative.parts[:2] == ("energyplus", "sim_results") and p_as_posix(relative) != "energyplus/sim_results/run_12/eplusout.err":
            continue
        yield path


def p_as_posix(path: Path) -> str:
    return path.as_posix()


def build_records(root: Path) -> list[AssetRecord]:
    records: list[AssetRecord] = []
    for path in sorted(iter_asset_paths(root), key=lambda item: item.as_posix().lower()):
        relative = path.relative_to(root)
        role, kind, city, status_note = _role_for(relative)
        status, _, note = status_note.partition(": ")
        stat = path.stat()
        rows = columns = None
        header = None
        if path.suffix.lower() in {".csv", ".tsv"}:
            try:
                rows, columns, header = csv_schema(path)
            except (OSError, UnicodeError, csv.Error) as exc:
                note = f"{note}; schema_error={exc}".strip("; ")
        records.append(
            AssetRecord(
                relative_path=relative.as_posix(),
                role=role,
                kind=kind,
                city=city,
                status=status,
                size_bytes=stat.st_size,
                modified_utc=datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
                sha256=sha256_file(path),
                csv_rows=rows,
                csv_columns=columns,
                csv_header=header,
                note=note,
            )
        )
    return records


def write_manifest(records: Iterable[AssetRecord], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    records = list(records)
    fieldnames = list(asdict(records[0]).keys()) if records else list(AssetRecord.__annotations__)
    csv_path = output_dir / "asset_manifest.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(asdict(record) for record in records)

    with (output_dir / "asset_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump([asdict(record) for record in records], handle, ensure_ascii=False, indent=2)

    groups: dict[str, list[AssetRecord]] = {}
    for record in records:
        groups.setdefault(record.sha256, []).append(record)
    duplicates = [group for group in groups.values() if len(group) > 1]
    with (output_dir / "file_duplicates.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sha256", "copy_count", "relative_path", "role", "status"])
        for group in sorted(duplicates, key=lambda items: (-len(items), items[0].sha256)):
            for record in group:
                writer.writerow([record.sha256, len(group), record.relative_path, record.role, record.status])

    role_counts: dict[str, int] = {}
    for record in records:
        role_counts[record.role] = role_counts.get(record.role, 0) + 1
    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "asset_count": len(records),
        "total_size_bytes": sum(record.size_bytes for record in records),
        "duplicate_groups": len(duplicates),
        "role_counts": dict(sorted(role_counts.items())),
    }
    with (output_dir / "manifest_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
