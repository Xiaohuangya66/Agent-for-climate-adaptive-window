from __future__ import annotations

import argparse
import os
import hashlib
import json
import signal
import shutil
import sqlite3
import subprocess
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


_CODE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = Path(os.environ.get("FOUR_CITY_CAMPAIGN_ROOT", _CODE_DIR)).resolve()
WORKSPACE = PROJECT_DIR
sys.path.insert(0, str(_CODE_DIR / "vendor"))
from campaign_helpers import build_idf_text, validate_metrics, store_evidence, registry, export_lock, atomic_frame_csv
from contextlib import closing


from thermo_agent.config import load_project_config, workspace_path  # noqa: E402
from thermo_agent.energyplus import (  # noqa: E402
    GOAL2_AUDIT_OUTPUT_REQUESTS,
    file_sha256,
    glazing_spectral_block,
    parse_energyplus_outputs,
)
from thermo_agent.spectra import state_from_strings  # noqa: E402


WEATHER = registry()
SOURCE_DATASETS = {city: "shared_optical_panel.csv.gz" for city in WEATHER["cities"]}
SOURCE_CITIES = WEATHER["source_cities"]

SPECTRAL_COLUMNS = ["T_cold", "Rf_cold", "Rb_cold", "T_hot", "Rf_hot", "Rb_hot"]
RESULT_COLUMNS = [
    "task_key",
    "city",
    "structure_hash",
    "status",
    "UDI",
    "UDI_qualifying_hours",
    "UDI_denominator_hours",
    "ElectricityFacility_J",
    "NaturalGasFacility_J",
    "TotalEnergy",
    "TotalSiteEnergy_kWh",
    "floor_area_m2",
    "EUI_kWh_m2_yr",
    "input_signature",
    "return_code",
    "error_code",
    "error_message",
    "run_seconds",
    "completed_utc",
]

_WORKER: dict[str, Any] = {}
_STOP_REQUESTED = False


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _task_key(city: str, structure: str) -> str:
    return f"{city}:{_sha256_text(structure)}"


def _initialize_worker(payload: dict[str, Any]) -> None:
    global _WORKER
    _WORKER = payload
    _WORKER["base_idf_text"] = Path(payload["base_idf"]).read_text(
        encoding="utf-8", errors="strict"
    )
    _WORKER["city_idf_text"] = {city: Path(path).read_text(encoding="utf-8")
                                for city,path in payload.get("city_idfs",{}).items()}


def _spectrum_signature_for_payload(
    row: dict[str, Any], city: str, payload: dict[str, Any]
) -> str:
    digest = hashlib.sha256()
    digest.update(payload["protocol_version"].encode("utf-8"))
    digest.update(payload["base_idf_sha256"].encode("ascii"))
    digest.update(payload.get("city_idf_sha256",{}).get(city,"").encode("ascii"))
    digest.update(payload["epw_sha256"][city].encode("ascii"))
    digest.update(str(row["structure"]).encode("utf-8"))
    for column in SPECTRAL_COLUMNS:
        digest.update(str(row[column]).encode("ascii"))
    digest.update(b"daylight_hours|UDI100-2000|Electricity:Facility+NaturalGas:Facility")
    return digest.hexdigest()


def _spectrum_signature(row: dict[str, Any], city: str) -> str:
    return _spectrum_signature_for_payload(row, city, _WORKER)


def _request_stop(signum, _frame) -> None:
    global _STOP_REQUESTED
    if not _STOP_REQUESTED:
        print(
            f"received signal {signum}; cancelling queued work and checkpointing "
            "currently running results before exit",
            flush=True,
        )
    _STOP_REQUESTED = True


def _run_one(task: tuple[str, dict[str, Any]]) -> dict[str, Any]:
    city, row = task
    structure = str(row["structure"])
    structure_hash = _sha256_text(structure)
    task_key = f"{city}:{structure_hash}"
    signature = _spectrum_signature(row, city)
    started = time.perf_counter()
    run_parent = Path(_WORKER["work_root"]) / city / structure_hash[:2]
    run_directory = run_parent / f"{structure_hash[:16]}_{signature[:12]}"
    if run_directory.exists():
        run_directory = run_parent / (
            f"{structure_hash[:16]}_{signature[:12]}_"
            f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')}"
        )
    run_directory.mkdir(parents=True, exist_ok=False)

    result: dict[str, Any] = {
        "task_key": task_key,
        "city": city,
        "structure_hash": structure_hash,
        "status": "failed",
        "UDI": None,
        "UDI_qualifying_hours": None,
        "UDI_denominator_hours": None,
        "ElectricityFacility_J": None,
        "NaturalGasFacility_J": None,
        "TotalEnergy": None,
        "TotalSiteEnergy_kWh": None,
        "floor_area_m2": None,
        "EUI_kWh_m2_yr": None,
        "input_signature": signature,
        "return_code": None,
        "error_code": None,
        "error_message": None,
        "run_seconds": None,
        "completed_utc": None,
    }
    try:
        wavelengths = np.linspace(300.0, 2500.0, 221)
        cold = state_from_strings(
            wavelengths, row["T_cold"], row["Rf_cold"], row["Rb_cold"]
        )
        hot = state_from_strings(
            wavelengths, row["T_hot"], row["Rf_hot"], row["Rb_hot"]
        )
        idf_text = build_idf_text(_WORKER["city_idf_text"].get(city,_WORKER["base_idf_text"]), cold, hot)
        idf_path = run_directory / "input.idf"
        idf_path.write_text(idf_text, encoding="utf-8")
        completed = subprocess.run(
            [
                _WORKER["executable"],
                "-w",
                _WORKER["epw"][city],
                "-d",
                str(run_directory),
                "-r",
                str(idf_path),
            ],
            cwd=str(run_directory),
            capture_output=True,
            timeout=int(_WORKER["timeout_seconds"]),
            check=False,
        )
        result["return_code"] = completed.returncode
        (run_directory / "process.stdout.log").write_bytes(completed.stdout)
        (run_directory / "process.stderr.log").write_bytes(completed.stderr)
        metrics = parse_energyplus_outputs(
            run_directory,
            return_code=completed.returncode,
            udi_lower_lux=100.0,
            udi_upper_lux=2000.0,
            udi_denominator_mode="daylight_hours",
        )
        validate_metrics(run_directory, metrics)
        result.update(
            {
                "status": metrics.status,
                "UDI": metrics.udi,
                "UDI_qualifying_hours": metrics.udi_qualifying_hours,
                "UDI_denominator_hours": metrics.udi_denominator_hours,
                "ElectricityFacility_J": metrics.electricity_J,
                "NaturalGasFacility_J": metrics.gas_J,
                "TotalEnergy": metrics.total_energy_J,
                "TotalSiteEnergy_kWh": metrics.total_energy_kWh,
                "floor_area_m2": metrics.floor_area_m2,
                "EUI_kWh_m2_yr": metrics.eui_kWh_m2_yr,
                "error_code": metrics.error_code,
                "error_message": metrics.error_message,
            }
        )
    except subprocess.TimeoutExpired as exc:
        result["error_code"] = "TIMEOUT"
        result["error_message"] = f"EnergyPlus exceeded {_WORKER['timeout_seconds']} seconds."
        (run_directory / "process.stdout.log").write_bytes(exc.stdout or b"")
        (run_directory / "process.stderr.log").write_bytes(exc.stderr or b"")
    except Exception as exc:  # retained in checkpoint and failure directory for audit
        result["error_code"] = type(exc).__name__
        result["error_message"] = str(exc)
    finally:
        result["run_seconds"] = time.perf_counter() - started
        result["completed_utc"] = datetime.now(timezone.utc).isoformat()
        (run_directory / "campaign_result.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        result["run_directory"] = str(run_directory)
        # The parent commits compressed evidence before owned scratch cleanup.
    return result


from contextlib import contextmanager

@contextmanager
def _connect(path: Path):
    connection = sqlite3.connect(path, timeout=60)
    try:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=NORMAL")
        with connection: yield connection
    finally: connection.close()


def _initialize_database(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with _connect(path) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS site_energy_labels (
                task_key TEXT PRIMARY KEY,
                city TEXT NOT NULL,
                structure_hash TEXT NOT NULL,
                status TEXT NOT NULL,
                UDI REAL,
                UDI_qualifying_hours INTEGER,
                UDI_denominator_hours INTEGER,
                ElectricityFacility_J REAL,
                NaturalGasFacility_J REAL,
                TotalEnergy REAL,
                TotalSiteEnergy_kWh REAL,
                floor_area_m2 REAL,
                EUI_kWh_m2_yr REAL,
                input_signature TEXT NOT NULL,
                return_code INTEGER,
                error_code TEXT,
                error_message TEXT,
                run_seconds REAL,
                completed_utc TEXT NOT NULL
            )
            """
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_site_energy_city_status "
            "ON site_energy_labels(city, status)"
        )


def _upsert_result(database: Path, result: dict[str, Any]) -> None:
    placeholders = ",".join("?" for _ in RESULT_COLUMNS)
    columns = ",".join(RESULT_COLUMNS)
    assignments = ",".join(f"{column}=excluded.{column}" for column in RESULT_COLUMNS[1:])
    with _connect(database) as connection:
        connection.execute(
            f"INSERT INTO site_energy_labels ({columns}) VALUES ({placeholders}) "
            f"ON CONFLICT(task_key) DO UPDATE SET {assignments}",
            [result.get(column) for column in RESULT_COLUMNS],
        )


def _existing_task_keys(
    database: Path, *, retry_failed: bool
) -> set[str]:
    if not database.is_file():
        return set()
    query = "SELECT task_key FROM site_energy_labels"
    if retry_failed:
        query += " WHERE status = 'success'"
    with _connect(database) as connection:
        return {row[0] for row in connection.execute(query)}


def _database_row_count(database: Path) -> int:
    if not database.is_file():
        return 0
    with _connect(database) as connection:
        return int(connection.execute("SELECT COUNT(*) FROM site_energy_labels").fetchone()[0])


def _source_hashes_from_config(config: dict[str, Any]) -> dict[str, str]:
    hashes = dict(config.get("source_dataset_sha256", {}))
    for city, value in config.get("source_datasets", {}).items():
        if city in hashes:
            continue
        path = Path(value)
        if path.is_file():
            hashes[city] = file_sha256(path)
    return hashes


def _validate_resume_compatibility(
    database: Path,
    existing: dict[str, Any],
    current: dict[str, Any],
) -> None:
    """Reject reuse when any scientific input differs from the checkpoint."""
    if _database_row_count(database) == 0:
        return
    mismatches: list[str] = []
    for key in ("protocol_version", "base_idf_sha256", "city_idf_sha256", "target_definition"):
        if existing.get(key) != current.get(key):
            mismatches.append(key)
    existing_epw = existing.get("epw_sha256", {})
    current_epw = current.get("epw_sha256", {})
    for city in current.get("cities", []):
        if existing_epw.get(city) != current_epw.get(city):
            mismatches.append(f"epw_sha256.{city}")
    existing_sources = _source_hashes_from_config(existing)
    current_sources = current.get("source_dataset_sha256", {})
    for city in current.get("cities", []):
        if existing_sources.get(city) != current_sources.get(city):
            mismatches.append(f"source_dataset_sha256.{city}")
    if mismatches:
        raise RuntimeError(
            "Refusing to reuse the site-energy checkpoint because scientific inputs "
            f"changed: {', '.join(sorted(set(mismatches)))}"
        )


def _validate_completed_signatures(
    database: Path,
    sources: dict[str, pd.DataFrame],
    payload: dict[str, Any],
) -> None:
    """Prove that every persisted success belongs to the current IDF/EPW/spectrum protocol."""
    if not database.is_file():
        return
    with _connect(database) as connection:
        persisted = dict(
            connection.execute(
                "SELECT task_key, input_signature FROM site_energy_labels "
                "WHERE status = 'success'"
            )
        )
    if not persisted:
        return
    expected: dict[str, str] = {}
    for city, frame in sources.items():
        for row in frame.to_dict("records"):
            key = _task_key(city, str(row["structure"]))
            if key in persisted:
                expected[key] = _spectrum_signature_for_payload(row, city, payload)
    missing = sorted(set(persisted) - set(expected))
    mismatched = sorted(
        key for key, signature in persisted.items() if expected.get(key) != signature
    )
    if missing or mismatched:
        detail = {
            "persisted_successes": len(persisted),
            "missing_from_current_sources": missing[:10],
            "signature_mismatches": mismatched[:10],
        }
        raise RuntimeError(
            "Refusing to resume from incompatible persisted labels: "
            + json.dumps(detail, ensure_ascii=False)
        )


def _read_source(city: str) -> pd.DataFrame:
    path = WORKSPACE / "energyplus" / SOURCE_DATASETS[city]
    frame = pd.read_csv(path)
    frame = frame.drop_duplicates("structure", keep="first").reset_index(drop=True)
    frame["structure_hash"] = frame["structure"].map(_sha256_text)
    return frame


def _export_city_dataset(city: str, database: Path, output_root: Path) -> dict[str, Any]:
    source = _read_source(city)
    with closing(sqlite3.connect(database.resolve().as_uri()+"?mode=ro",uri=True)) as connection:
        labels = pd.read_sql_query(
            "SELECT * FROM site_energy_labels WHERE city = ?",
            connection,
            params=[city],
        )
    legacy = source.rename(
        columns={
            "UDI": "LegacyAllHoursUDI",
            "TotalEnergy": "LegacyElectricityOnly_J",
        }
    )
    merged = legacy.merge(labels, on=["structure_hash"], how="left", validate="one_to_one")
    successful = merged[merged["status"].eq("success")].copy()
    ordered_front = [
        "id",
        "structure",
        *SPECTRAL_COLUMNS,
        "T_lum_cold",
        "T_lum_hot",
        "T_sol_cold",
        "T_sol_hot",
        "dT_sol",
        "Rf_sol_cold",
        "Rb_sol_cold",
        "Rf_sol_hot",
        "Rb_sol_hot",
        "UDI",
        "TotalEnergy",
        "ElectricityFacility_J",
        "NaturalGasFacility_J",
        "TotalSiteEnergy_kWh",
        "UDI_qualifying_hours",
        "UDI_denominator_hours",
        "floor_area_m2",
        "EUI_kWh_m2_yr",
        "LegacyAllHoursUDI",
        "LegacyElectricityOnly_J",
        "structure_hash",
        "input_signature",
    ]
    successful = successful[[column for column in ordered_front if column in successful.columns]]
    output_root.mkdir(parents=True, exist_ok=True)
    dataset_path = output_root / f"ML_Dataset_SiteEnergy_{city}.csv"
    atomic_frame_csv(successful,dataset_path)
    failures = merged[merged["status"].notna() & ~merged["status"].eq("success")]
    failure_path = output_root / f"ML_Dataset_SiteEnergy_{city}_failures.csv"
    atomic_frame_csv(failures,failure_path)
    return {
        "city": city,
        "source_rows": int(len(source)),
        "successful_rows": int(len(successful)),
        "failed_rows": int(len(failures)),
        "pending_rows": int(merged["status"].isna().sum()),
        "dataset": str(dataset_path),
        "failures": str(failure_path),
    }


def _tasks(
    cities: Iterable[str],
    sources: dict[str, pd.DataFrame],
    completed: set[str],
    limit_per_city: int | None,
) -> Iterable[tuple[str, dict[str, Any]]]:
    for city in cities:
        submitted = 0
        panel = sources[city].head(limit_per_city) if limit_per_city is not None else sources[city]
        for row in panel.to_dict("records"):
            key = _task_key(city, str(row["structure"]))
            if key in completed:
                continue
            yield city, {column: row[column] for column in ["id", "structure", *SPECTRAL_COLUMNS]}
            submitted += 1
            if limit_per_city is not None and submitted >= limit_per_city:
                break


def _export_checkpoint(cities, database, output_root):
    with export_lock(output_root):
        summaries = [_export_city_dataset(city,database,output_root) for city in cities]
        destination=output_root / "dataset_rebuild_summary.json"
        temporary=destination.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(summaries,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
        temporary.replace(destination)
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rebuild formal daylight-hours UDI and total site-energy surrogate labels."
    )
    parser.add_argument("--cities", nargs="+", choices=sorted(SOURCE_DATASETS), default=SOURCE_CITIES)
    parser.add_argument("--max-workers", type=int, default=12)
    parser.add_argument("--max-pending", type=int, default=48)
    parser.add_argument("--timeout-seconds", type=int, default=900)
    parser.add_argument("--limit-per-city", type=int)
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--keep-success-runs", action="store_true")
    parser.add_argument("--export-only", action="store_true")
    parser.add_argument(
        "--stop-file",
        help=(
            "Optional sentinel path. When it exists, stop submitting work, cancel queued "
            "futures, checkpoint running results, export partial datasets and exit cleanly."
        ),
    )
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs without starting EnergyPlus")
    args = parser.parse_args()
    if args.limit_per_city is not None and args.limit_per_city < 1:
        parser.error("limit-per-city must be positive")
    if args.max_workers < 1 or args.max_pending < args.max_workers:
        parser.error("max-pending must be >= max-workers >= 1")
    if not args.export_only and args.max_workers > int(os.environ.get("SLURM_CPUS_PER_TASK", args.max_workers)):
        parser.error("max-workers exceeds allocated Slurm CPUs")
    output_root = args.output_root.resolve() if args.output_root else WORKSPACE / "results"
    if args.dry_run:
        project = load_project_config(Path(os.environ.get("FOUR_CITY_DATASET_CONFIG", PROJECT_DIR / "configs" / "project.json")))
        paths = [Path(project["energyplus"]["executable"]), workspace_path(project,project["energyplus"]["base_idf"])]
        paths += [workspace_path(project,project["cities"][city][key])
                  for city in args.cities for key in ["epw","base_idf","baseline_idf"]]
        for path in paths:
            if not path.is_file(): raise FileNotFoundError(path)
        panel = _read_source(args.cities[0])
        print(json.dumps({"mode":"dry_run", "cities":args.cities, "panel_rows":len(panel),
              "planned_per_city":min(args.limit_per_city or len(panel),len(panel)),
              "output_root":str(output_root), "energyplus_started":False},indent=2))
        return
    database = output_root / "site_energy_labels.sqlite3"
    if args.export_only:
        if not database.is_file():
            raise FileNotFoundError(f"No label database yet: {database}; no simulation was started")
        summaries=_export_checkpoint(args.cities,database,output_root)
        print(json.dumps(summaries,ensure_ascii=False,indent=2),flush=True)
        return
    work_root = output_root / "work"
    output_root.mkdir(parents=True, exist_ok=True)
    run_lock = (output_root / "dataset_run.lock").open("a")
    if os.name != "nt":
        import fcntl
        try: fcntl.flock(run_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError: raise RuntimeError("Another dataset job owns this output directory")
    _initialize_database(database)
    project = load_project_config(Path(os.environ.get("FOUR_CITY_DATASET_CONFIG", PROJECT_DIR / "configs" / "project.json")))
    base_idf = workspace_path(project, project["energyplus"]["base_idf"])
    executable = Path(project["energyplus"]["executable"]).resolve()
    epw = {city: workspace_path(project, project["cities"][city]["epw"]) for city in args.cities}
    worker_payload = {
        "protocol_version": "unified-tmyx-site-energy-city-sizing-v2",
        "executable": str(executable),
        "base_idf": str(base_idf),
        "base_idf_sha256": file_sha256(base_idf),
        "city_idfs": {city: str(workspace_path(project,project["cities"][city]["base_idf"])) for city in args.cities},
        "city_idf_sha256": {city: file_sha256(workspace_path(project,project["cities"][city]["base_idf"])) for city in args.cities},
        "epw": {city: str(path) for city, path in epw.items()},
        "epw_sha256": {city: file_sha256(path) for city, path in epw.items()},
        "work_root": str(work_root),
        "timeout_seconds": args.timeout_seconds,
        "keep_success_runs": args.keep_success_runs,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    source_paths = {
        city: (WORKSPACE / "energyplus" / SOURCE_DATASETS[city]).resolve()
        for city in args.cities
    }
    config_path = output_root / "campaign_config.json"
    existing_config = (
        json.loads(config_path.read_text(encoding="utf-8")) if config_path.is_file() else {}
    )
    campaign_config = {
        **{key: value for key, value in worker_payload.items() if key != "keep_success_runs"},
        "energyplus_version": project["energyplus"].get("version"),
        "cities": args.cities,
        "source_datasets": {city: str(path) for city, path in source_paths.items()},
        "source_dataset_sha256": {
            city: file_sha256(path) for city, path in source_paths.items()
        },
        "target_definition": {
            "UDI": "fraction of solar-altitude-positive hours with 100-2000 lux",
            "TotalEnergy": "Electricity:Facility + NaturalGas:Facility, annual RunPeriod joules",
        },
        "created_utc": existing_config.get(
            "created_utc", datetime.now(timezone.utc).isoformat()
        ),
        "last_started_utc": datetime.now(timezone.utc).isoformat(),
    }
    _validate_resume_compatibility(database, existing_config, campaign_config)
    config_path.write_text(
        json.dumps(campaign_config, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    try:
        if not args.export_only:
            sources = {city: _read_source(city) for city in args.cities}
            _validate_completed_signatures(database, sources, worker_payload)
            completed = _existing_task_keys(database, retry_failed=args.retry_failed)
            iterator = iter(_tasks(args.cities, sources, completed, args.limit_per_city))
            stop_file = (
                Path(args.stop_file).resolve()
                if args.stop_file
                else output_root / "STOP_AFTER_CURRENT_BATCH"
            )
            signal.signal(signal.SIGINT, _request_stop)
            if hasattr(signal, "SIGTERM"):
                signal.signal(signal.SIGTERM, _request_stop)
            start = time.perf_counter()
            launched = finished = successes = failures = 0
            with ProcessPoolExecutor(
                max_workers=args.max_workers,
                initializer=_initialize_worker,
                initargs=(worker_payload,),
            ) as executor:
                pending = set()
                while True:
                    stop_requested = _STOP_REQUESTED or stop_file.exists()
                    while len(pending) < args.max_pending and not stop_requested:
                        try:
                            task = next(iterator)
                        except StopIteration:
                            break
                        pending.add(executor.submit(_run_one, task))
                        launched += 1
                        stop_requested = _STOP_REQUESTED or stop_file.exists()
                    if stop_requested and pending:
                        cancelled = 0
                        for future in list(pending):
                            if future.cancel():
                                pending.remove(future)
                                cancelled += 1
                        print(
                            f"graceful stop requested; cancelled={cancelled} "
                            f"running_to_checkpoint={len(pending)}",
                            flush=True,
                        )
                    if not pending:
                        break
                    done, pending = wait(pending, return_when=FIRST_COMPLETED)
                    for future in done:
                        result = future.result()
                        store_evidence(database, result, work_root, args.keep_success_runs)
                        _upsert_result(database, result)
                        finished += 1
                        successes += int(result["status"] == "success")
                        failures += int(result["status"] != "success")
                        if finished % 10 == 0 or result["status"] != "success":
                            elapsed = time.perf_counter() - start
                            rate = finished / elapsed if elapsed else 0.0
                            print(
                                f"completed={finished} launched={launched} success={successes} "
                                f"failed={failures} rate={rate:.3f}/s last={result['task_key']} "
                                f"last_status={result['status']}",
                                flush=True,
                            )
                    if (_STOP_REQUESTED or stop_file.exists()) and not pending:
                        break

    finally:
        # A handled cancellation/worker exception still leaves a readable snapshot.
        summaries=_export_checkpoint(args.cities,database,output_root)
        print(json.dumps(summaries,ensure_ascii=False,indent=2),flush=True)


if __name__ == "__main__":
    main()
