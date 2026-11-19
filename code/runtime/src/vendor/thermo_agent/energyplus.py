from __future__ import annotations

import csv
import hashlib
import json
import re
import sqlite3
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .spectra import SpectralState, validate_spectral_state
from .structures import StructureValidation, validate_structure


SUCCESS_MARKER = "EnergyPlus Completed Successfully"
SEVERE_SUMMARY = re.compile(r"(\d+)\s+Severe Errors?", re.IGNORECASE)

GOAL2_AUDIT_OUTPUT_REQUESTS = """
Output:Variable,
    G SW Apartment,
    Zone People Occupant Count,
    Hourly;

Output:Variable,
    *,
    Site Solar Altitude Angle,
    Hourly;

Output:Meter,
    Heating:Electricity,
    RunPeriod;

Output:Meter,
    Heating:NaturalGas,
    RunPeriod;

Output:Meter,
    Cooling:Electricity,
    RunPeriod;

Output:Meter,
    InteriorLights:Electricity,
    RunPeriod;

Output:Meter,
    ExteriorLights:Electricity,
    RunPeriod;

Output:Meter,
    InteriorEquipment:Electricity,
    RunPeriod;

Output:Meter,
    Fans:Electricity,
    RunPeriod;

Output:Meter,
    Pumps:Electricity,
    RunPeriod;

Output:Meter,
    HeatRecovery:Electricity,
    RunPeriod;

Output:Meter,
    WaterSystems:Electricity,
    RunPeriod;

Output:Meter,
    Electricity:Facility,
    RunPeriod;

Output:Meter,
    NaturalGas:Facility,
    RunPeriod;
""".strip()


def file_sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class EnergyPlusMetrics:
    status: str
    return_code: int | None
    success_marker: bool
    severe_errors: int | None
    fatal_detected: bool
    udi: float | None
    udi_qualifying_hours: int | None
    udi_denominator_hours: int | None
    illuminance_column: str | None
    electricity_J: float | None
    gas_J: float | None
    total_energy_J: float | None
    total_energy_kWh: float | None
    floor_area_m2: float | None
    eui_kWh_m2_yr: float | None
    error_code: str | None
    error_message: str | None


@dataclass(frozen=True)
class EnergyPlusRunResult:
    run_id: str
    city: str
    candidate_id: str
    structure: str
    structure_hash: str
    run_directory: str
    input_signature: str
    started_utc: str
    completed_utc: str
    metrics: EnergyPlusMetrics
    cache_hit: bool = False
    cache_source_run_directory: str | None = None


def glazing_spectral_block(name: str, state: SpectralState) -> str:
    validation = validate_spectral_state(state, tolerance=1e-5)
    if not validation.valid:
        raise ValueError("Invalid EnergyPlus spectrum: " + "; ".join(validation.errors))
    lines = ["MaterialProperty:GlazingSpectralData,", f"  {name},"]
    for index, wavelength_nm in enumerate(state.wavelengths_nm):
        terminator = ";" if index == len(state.wavelengths_nm) - 1 else ","
        lines.append(
            "  "
            f"{wavelength_nm / 1000.0:.4f}, "
            f"{state.transmittance[index]:.8f}, "
            f"{state.front_reflectance[index]:.8f}, "
            f"{state.back_reflectance[index]:.8f}{terminator}"
        )
    return "\n".join(lines) + "\n"


def _last_numeric(frame: pd.DataFrame, columns: list[str]) -> float:
    total = 0.0
    for column in columns:
        values = pd.to_numeric(frame[column], errors="coerce").dropna()
        if not values.empty:
            total += float(values.iloc[-1])
    return total


def parse_floor_area_from_eio(path: Path) -> float | None:
    if not path.is_file():
        return None
    total = 0.0
    found = False
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        for row in csv.reader(handle):
            if not row or row[0].strip() != "Zone Information" or len(row) < 30:
                continue
            if row[-1].strip().lower() != "yes":
                continue
            try:
                zone_multiplier = float(row[9])
                list_multiplier = float(row[10])
                floor_area = float(row[22])
            except (ValueError, IndexError):
                continue
            total += zone_multiplier * list_multiplier * floor_area
            found = True
    return total if found and total > 0 else None


def parse_energyplus_outputs(
    run_directory: str | Path,
    *,
    return_code: int | None,
    udi_lower_lux: float = 100,
    udi_upper_lux: float = 2000,
    udi_denominator_mode: str = "all_hours",
) -> EnergyPlusMetrics:
    run_directory = Path(run_directory)
    end_path = run_directory / "eplusout.end"
    err_path = run_directory / "eplusout.err"
    csv_path = run_directory / "eplusout.csv"
    end_text = end_path.read_text(encoding="utf-8", errors="replace") if end_path.is_file() else ""
    err_text = err_path.read_text(encoding="utf-8", errors="replace") if err_path.is_file() else ""
    success_marker = SUCCESS_MARKER.lower() in end_text.lower()
    fatal_detected = "**  Fatal  **" in err_text or "Fatal Error" in err_text or "terminated--fatal" in err_text.lower()
    severe_matches = SEVERE_SUMMARY.findall(end_text + "\n" + err_text)
    severe_errors = int(severe_matches[-1]) if severe_matches else None

    if return_code not in (0, None):
        return EnergyPlusMetrics(
            "failed", return_code, success_marker, severe_errors, fatal_detected,
            None, None, None, None, None, None, None, None, None, None,
            "NONZERO_RETURN_CODE", f"EnergyPlus returned {return_code}.",
        )
    if not success_marker or fatal_detected or (severe_errors is not None and severe_errors > 0):
        return EnergyPlusMetrics(
            "failed", return_code, success_marker, severe_errors, fatal_detected,
            None, None, None, None, None, None, None, None, None, None,
            "ENERGYPLUS_NOT_SUCCESSFUL", "Missing success marker or severe/fatal errors detected.",
        )
    if not csv_path.is_file():
        return EnergyPlusMetrics(
            "failed", return_code, success_marker, severe_errors, fatal_detected,
            None, None, None, None, None, None, None, None, None, None,
            "MISSING_OUTPUT_CSV", "eplusout.csv was not produced.",
        )

    try:
        frame = pd.read_csv(csv_path)
    except Exception as exc:
        return EnergyPlusMetrics(
            "failed", return_code, success_marker, severe_errors, fatal_detected,
            None, None, None, None, None, None, None, None, None, None,
            "CSV_PARSE_ERROR", str(exc),
        )

    illuminance_columns = [
        column for column in frame
        if "Daylighting Reference Point" in column and "Illuminance" in column
    ]
    if not illuminance_columns:
        illuminance_columns = [column for column in frame if "Illuminance" in column]
    if not illuminance_columns:
        return EnergyPlusMetrics(
            "failed", return_code, success_marker, severe_errors, fatal_detected,
            None, None, None, None, None, None, None, None, None, None,
            "MISSING_ILLUMINANCE", "No illuminance output column was found.",
        )
    illuminance_column = illuminance_columns[0]
    lux = pd.to_numeric(frame[illuminance_column], errors="coerce")
    denominator_mask = lux.notna()
    valid_modes = {
        "all_hours", "daylight_hours", "occupied_hours", "occupied_daylight_hours"
    }
    if udi_denominator_mode not in valid_modes:
        raise ValueError(f"Unknown UDI denominator mode: {udi_denominator_mode}")
    if "daylight" in udi_denominator_mode:
        solar_columns = [column for column in frame if "Site Solar Altitude Angle" in column]
        if not solar_columns:
            return EnergyPlusMetrics(
                "failed", return_code, success_marker, severe_errors, fatal_detected,
                None, None, None, illuminance_column,
                None, None, None, None, None, None,
                "MISSING_SOLAR_ALTITUDE",
                f"UDI mode {udi_denominator_mode} requires Site Solar Altitude Angle.",
            )
        solar_altitude = pd.to_numeric(frame[solar_columns[0]], errors="coerce")
        denominator_mask &= solar_altitude.gt(0)
    if "occupied" in udi_denominator_mode:
        occupancy_columns = [column for column in frame if "Zone People Occupant Count" in column]
        if not occupancy_columns:
            return EnergyPlusMetrics(
                "failed", return_code, success_marker, severe_errors, fatal_detected,
                None, None, None, illuminance_column,
                None, None, None, None, None, None,
                "MISSING_OCCUPANCY",
                f"UDI mode {udi_denominator_mode} requires Zone People Occupant Count.",
            )
        occupancy = pd.to_numeric(frame[occupancy_columns[0]], errors="coerce")
        denominator_mask &= occupancy.gt(0)
    denominator = int(denominator_mask.sum())
    qualifying = int(
        (
            denominator_mask
            & lux.between(udi_lower_lux, udi_upper_lux, inclusive="both")
        ).sum()
    )
    udi = qualifying / denominator if denominator else None

    electricity_columns = [
        column for column in frame if "Electricity:Facility" in column and "RunPeriod" in column
    ]
    natural_gas_columns = [
        column
        for column in frame
        if "NaturalGas:Facility" in column and "RunPeriod" in column
    ]
    if not electricity_columns:
        return EnergyPlusMetrics(
            "failed", return_code, success_marker, severe_errors, fatal_detected,
            udi, qualifying, denominator, illuminance_column,
            None, None, None, None, None, None,
            "MISSING_ELECTRICITY_METER",
            "Electricity:Facility RunPeriod meter was not found.",
        )
    if len(electricity_columns) != 1:
        return EnergyPlusMetrics(
            "failed", return_code, success_marker, severe_errors, fatal_detected,
            udi, qualifying, denominator, illuminance_column,
            None, None, None, None, None, None,
            "AMBIGUOUS_ELECTRICITY_METER",
            f"Expected one Electricity:Facility RunPeriod column; found {electricity_columns}.",
        )
    if not natural_gas_columns:
        return EnergyPlusMetrics(
            "failed", return_code, success_marker, severe_errors, fatal_detected,
            udi, qualifying, denominator, illuminance_column,
            None, None, None, None, None, None,
            "MISSING_NATURAL_GAS_METER",
            "NaturalGas:Facility RunPeriod meter was not found. "
            "Total site energy cannot be computed from electricity alone.",
        )
    if len(natural_gas_columns) != 1:
        return EnergyPlusMetrics(
            "failed", return_code, success_marker, severe_errors, fatal_detected,
            udi, qualifying, denominator, illuminance_column,
            None, None, None, None, None, None,
            "AMBIGUOUS_NATURAL_GAS_METER",
            f"Expected one NaturalGas:Facility RunPeriod column; found {natural_gas_columns}.",
        )
    electricity = _last_numeric(frame, electricity_columns)
    gas = _last_numeric(frame, natural_gas_columns)
    total = electricity + gas
    if not np.isfinite(total) or total <= 0:
        return EnergyPlusMetrics(
            "failed", return_code, success_marker, severe_errors, fatal_detected,
            udi, qualifying, denominator, illuminance_column,
            electricity, gas, None, None, None, None,
            "NONPOSITIVE_ENERGY", f"Total facility energy is invalid: {total}.",
        )
    energy_kwh = total / 3_600_000.0
    floor_area = parse_floor_area_from_eio(run_directory / "eplusout.eio")
    eui = energy_kwh / floor_area if floor_area else None
    return EnergyPlusMetrics(
        "success", return_code, success_marker, severe_errors, fatal_detected,
        udi, qualifying, denominator, illuminance_column,
        electricity, gas, total, energy_kwh, floor_area, eui,
        None, None,
    )


class EnergyPlusRunner:
    def __init__(
        self,
        *,
        executable: str | Path,
        base_idf: str | Path,
        epw: str | Path,
        output_root: str | Path,
        city: str,
        timeout_seconds: int = 900,
        udi_lower_lux: float = 100,
        udi_upper_lux: float = 2000,
        additional_idf_text: str = "",
        udi_denominator_mode: str = "all_hours",
    ) -> None:
        self.executable = Path(executable).resolve()
        self.base_idf = Path(base_idf).resolve()
        self.epw = Path(epw).resolve()
        self.output_root = Path(output_root).resolve()
        self.city = city
        self.timeout_seconds = timeout_seconds
        self.udi_lower_lux = udi_lower_lux
        self.udi_upper_lux = udi_upper_lux
        self.additional_idf_text = additional_idf_text.strip()
        self.udi_denominator_mode = udi_denominator_mode
        for path in (self.executable, self.base_idf, self.epw):
            if not path.is_file():
                raise FileNotFoundError(path)
        self.cache_database: Path | None = None
        for parent in (self.output_root, *self.output_root.parents):
            if parent.name == "agent_project":
                self.cache_database = parent / "runs" / "energyplus_success_cache.sqlite3"
                self.cache_database.parent.mkdir(parents=True, exist_ok=True)
                self._initialize_cache()
                break

    def _initialize_cache(self) -> None:
        if self.cache_database is None:
            return
        with sqlite3.connect(self.cache_database, timeout=30) as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS successful_runs (
                    input_signature TEXT PRIMARY KEY,
                    city TEXT NOT NULL,
                    result_json TEXT NOT NULL,
                    canonical_run_directory TEXT NOT NULL,
                    indexed_utc TEXT NOT NULL
                )
                """
            )

    def _cached_result(
        self,
        signature: str,
        *,
        candidate_id: str | int,
        structure: str,
        structure_hash: str,
    ) -> EnergyPlusRunResult | None:
        if self.cache_database is None:
            return None
        with sqlite3.connect(self.cache_database, timeout=30) as connection:
            row = connection.execute(
                "SELECT result_json, canonical_run_directory FROM successful_runs "
                "WHERE input_signature = ? AND city = ?",
                (signature, self.city),
            ).fetchone()
        if row is None:
            return None
        payload = json.loads(row[0])
        metrics_payload = payload.get("metrics", {})
        source = Path(row[1])
        if metrics_payload.get("status") != "success" or not source.is_dir():
            return None
        return EnergyPlusRunResult(
            run_id=f"cache_{signature[:12]}",
            city=self.city,
            candidate_id=str(candidate_id),
            structure=structure,
            structure_hash=structure_hash,
            run_directory=str(source),
            input_signature=signature,
            started_utc=payload["started_utc"],
            completed_utc=payload["completed_utc"],
            metrics=EnergyPlusMetrics(**metrics_payload),
            cache_hit=True,
            cache_source_run_directory=str(source),
        )

    def _store_success(self, result: EnergyPlusRunResult) -> None:
        if self.cache_database is None or result.metrics.status != "success":
            return
        payload = json.dumps(asdict(result), ensure_ascii=False)
        with sqlite3.connect(self.cache_database, timeout=30) as connection:
            connection.execute(
                """
                INSERT OR IGNORE INTO successful_runs
                (input_signature, city, result_json, canonical_run_directory, indexed_utc)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    result.input_signature,
                    result.city,
                    payload,
                    result.run_directory,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )

    def _input_signature(self, structure_hash: str, cold: SpectralState, hot: SpectralState) -> str:
        digest = hashlib.sha256()
        for path in (self.executable, self.base_idf, self.epw):
            digest.update(str(path).encode("utf-8"))
            digest.update(str(path.stat().st_size).encode("ascii"))
            digest.update(str(path.stat().st_mtime_ns).encode("ascii"))
        digest.update(structure_hash.encode("ascii"))
        digest.update(self.additional_idf_text.encode("utf-8"))
        digest.update(self.udi_denominator_mode.encode("ascii"))
        for state in (cold, hot):
            for array in (
                state.wavelengths_nm,
                state.transmittance,
                state.front_reflectance,
                state.back_reflectance,
            ):
                digest.update(np.asarray(array, dtype=np.float64).tobytes())
        return digest.hexdigest()

    def run(
        self,
        *,
        candidate_id: str | int,
        structure: str,
        cold: SpectralState,
        hot: SpectralState,
        allowed_tokens: set[str] | None = None,
        wvo2_min_thickness_nm: float = 20.0,
        wvo2_max_thickness_nm: float = 40.0,
        reuse_success: bool = True,
    ) -> EnergyPlusRunResult:
        structure_validation: StructureValidation = validate_structure(
            structure,
            allowed_tokens=allowed_tokens,
            wvo2_min_thickness_nm=wvo2_min_thickness_nm,
            wvo2_max_thickness_nm=wvo2_max_thickness_nm,
        )
        if not structure_validation.valid:
            raise ValueError("Invalid structure: " + "; ".join(structure_validation.errors))
        for name, state in (("cold", cold), ("hot", hot)):
            validation = validate_spectral_state(state, tolerance=1e-5)
            if not validation.valid:
                raise ValueError(f"Invalid {name} spectrum: " + "; ".join(validation.errors))

        signature = self._input_signature(structure_validation.structure_hash, cold, hot)
        if reuse_success:
            cached = self._cached_result(
                signature,
                candidate_id=candidate_id,
                structure=structure_validation.normalized,
                structure_hash=structure_validation.structure_hash,
            )
            if cached is not None:
                return cached
        run_id = f"{candidate_id}_{structure_validation.structure_hash[:12]}_{signature[:12]}"
        run_directory = self.output_root / self.city / run_id
        result_path = run_directory / "result.json"
        if reuse_success and result_path.is_file():
            existing = json.loads(result_path.read_text(encoding="utf-8"))
            if existing.get("input_signature") == signature and existing.get("metrics", {}).get("status") == "success":
                metrics = EnergyPlusMetrics(**existing["metrics"])
                return EnergyPlusRunResult(**{**existing, "metrics": metrics})
        if run_directory.exists() and any(run_directory.iterdir()):
            attempt = 2
            while (run_directory.parent / f"{run_id}_attempt{attempt}").exists():
                attempt += 1
            run_directory = run_directory.parent / f"{run_id}_attempt{attempt}"
            result_path = run_directory / "result.json"
        run_directory.mkdir(parents=True, exist_ok=False)

        started = datetime.now(timezone.utc).isoformat()
        input_metadata: dict[str, Any] = {
            "run_id": run_directory.name,
            "city": self.city,
            "candidate_id": str(candidate_id),
            "structure": structure_validation.normalized,
            "structure_hash": structure_validation.structure_hash,
            "input_signature": signature,
            "energyplus_executable": str(self.executable),
            "base_idf": str(self.base_idf),
            "epw": str(self.epw),
            "udi_definition": {
                "lower_lux": self.udi_lower_lux,
                "upper_lux": self.udi_upper_lux,
                "denominator": {
                    "all_hours": "all finite hourly illuminance values",
                    "daylight_hours": "finite hourly illuminance values with Site Solar Altitude Angle > 0",
                    "occupied_hours": "finite hourly illuminance values with Zone People Occupant Count > 0",
                    "occupied_daylight_hours": "finite hourly illuminance values with occupancy > 0 and solar altitude > 0",
                }[self.udi_denominator_mode],
                "mode": self.udi_denominator_mode,
            },
            "energy_definition": {
                "type": "annual_site_energy",
                "formula": "Electricity:Facility + NaturalGas:Facility",
                "reporting_frequency": "RunPeriod",
                "unit": "J",
            },
            "additional_idf_text": self.additional_idf_text,
        }
        (run_directory / "input.json").write_text(
            json.dumps(input_metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        base_text = self.base_idf.read_text(encoding="utf-8", errors="strict")
        idf_text = (
            base_text
            + "\n\n"
            + glazing_spectral_block("VO2_Cold_Spec", cold)
            + "\n"
            + glazing_spectral_block("VO2_Hot_Spec", hot)
            + ("\n\n" + self.additional_idf_text if self.additional_idf_text else "")
        )
        idf_path = run_directory / "input.idf"
        idf_path.write_text(idf_text, encoding="utf-8")

        return_code: int | None = None
        try:
            completed = subprocess.run(
                [
                    str(self.executable),
                    "-w", str(self.epw),
                    "-d", str(run_directory),
                    "-r", str(idf_path),
                ],
                # EnergyPlus/ReadVarsESO creates auxiliary files such as
                # ``readvars.audit`` in the process working directory.  Give
                # every simulation its own cwd so parallel city runs cannot
                # race on those files.
                cwd=str(run_directory),
                capture_output=True,
                timeout=self.timeout_seconds,
                check=False,
            )
            return_code = completed.returncode
            (run_directory / "process.stdout.log").write_bytes(completed.stdout)
            (run_directory / "process.stderr.log").write_bytes(completed.stderr)
            metrics = parse_energyplus_outputs(
                run_directory,
                return_code=return_code,
                udi_lower_lux=self.udi_lower_lux,
                udi_upper_lux=self.udi_upper_lux,
                udi_denominator_mode=self.udi_denominator_mode,
            )
        except subprocess.TimeoutExpired as exc:
            (run_directory / "process.stdout.log").write_bytes(exc.stdout or b"")
            (run_directory / "process.stderr.log").write_bytes(exc.stderr or b"")
            metrics = EnergyPlusMetrics(
                "failed", None, False, None, False,
                None, None, None, None, None, None, None, None, None, None,
                "TIMEOUT", f"EnergyPlus exceeded {self.timeout_seconds} seconds.",
            )

        result = EnergyPlusRunResult(
            run_directory.name,
            self.city,
            str(candidate_id),
            structure_validation.normalized,
            structure_validation.structure_hash,
            str(run_directory),
            signature,
            started,
            datetime.now(timezone.utc).isoformat(),
            metrics,
        )
        result_path.write_text(json.dumps(asdict(result), ensure_ascii=False, indent=2), encoding="utf-8")
        self._store_success(result)
        return result

    def run_static_idf(
        self,
        *,
        candidate_id: str,
        label: str,
        reuse_success: bool = True,
    ) -> EnergyPlusRunResult:
        """Run a prepared IDF such as the clear-glass baseline without spectral injection."""
        digest = hashlib.sha256()
        for path in (self.executable, self.base_idf, self.epw):
            digest.update(str(path).encode("utf-8"))
            digest.update(file_sha256(path).encode("ascii"))
        digest.update(self.additional_idf_text.encode("utf-8"))
        digest.update(self.udi_denominator_mode.encode("ascii"))
        signature = digest.hexdigest()
        if reuse_success:
            cached = self._cached_result(
                signature,
                candidate_id=candidate_id,
                structure=label,
                structure_hash=hashlib.sha256(label.encode("utf-8")).hexdigest(),
            )
            if cached is not None:
                return cached
        run_id = f"{candidate_id}_{signature[:12]}"
        run_directory = self.output_root / self.city / run_id
        result_path = run_directory / "result.json"
        if reuse_success and result_path.is_file():
            existing = json.loads(result_path.read_text(encoding="utf-8"))
            if existing.get("input_signature") == signature and existing.get("metrics", {}).get("status") == "success":
                return EnergyPlusRunResult(
                    **{**existing, "metrics": EnergyPlusMetrics(**existing["metrics"])}
                )
        if run_directory.exists() and any(run_directory.iterdir()):
            attempt = 2
            while (run_directory.parent / f"{run_id}_attempt{attempt}").exists():
                attempt += 1
            run_directory = run_directory.parent / f"{run_id}_attempt{attempt}"
            result_path = run_directory / "result.json"
        run_directory.mkdir(parents=True, exist_ok=False)
        started = datetime.now(timezone.utc).isoformat()
        (run_directory / "input.json").write_text(
            json.dumps(
                {
                    "run_id": run_directory.name,
                    "city": self.city,
                    "candidate_id": candidate_id,
                    "label": label,
                    "input_signature": signature,
                    "energyplus_executable": str(self.executable),
                    "energyplus_executable_sha256": file_sha256(self.executable),
                    "idf": str(self.base_idf),
                    "idf_sha256": file_sha256(self.base_idf),
                    "epw": str(self.epw),
                    "epw_sha256": file_sha256(self.epw),
                    "additional_idf_text": self.additional_idf_text,
                    "udi_definition": {
                        "lower_lux": self.udi_lower_lux,
                        "upper_lux": self.udi_upper_lux,
                        "denominator": "all finite hourly values in the first illuminance output",
                        "mode": self.udi_denominator_mode,
                    },
                    "energy_definition": {
                        "type": "annual_site_energy",
                        "formula": "Electricity:Facility + NaturalGas:Facility",
                        "reporting_frequency": "RunPeriod",
                        "unit": "J",
                    },
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        idf_path = self.base_idf
        if self.additional_idf_text:
            idf_path = run_directory / "input.idf"
            idf_path.write_text(
                self.base_idf.read_text(encoding="utf-8", errors="strict")
                + "\n\n"
                + self.additional_idf_text
                + "\n",
                encoding="utf-8",
            )
        try:
            completed = subprocess.run(
                [
                    str(self.executable),
                    "-w", str(self.epw),
                    "-d", str(run_directory),
                    "-r", str(idf_path),
                ],
                cwd=str(run_directory),
                capture_output=True,
                timeout=self.timeout_seconds,
                check=False,
            )
            (run_directory / "process.stdout.log").write_bytes(completed.stdout)
            (run_directory / "process.stderr.log").write_bytes(completed.stderr)
            metrics = parse_energyplus_outputs(
                run_directory,
                return_code=completed.returncode,
                udi_lower_lux=self.udi_lower_lux,
                udi_upper_lux=self.udi_upper_lux,
                udi_denominator_mode=self.udi_denominator_mode,
            )
        except subprocess.TimeoutExpired as exc:
            (run_directory / "process.stdout.log").write_bytes(exc.stdout or b"")
            (run_directory / "process.stderr.log").write_bytes(exc.stderr or b"")
            metrics = EnergyPlusMetrics(
                "failed", None, False, None, False,
                None, None, None, None, None, None, None, None, None, None,
                "TIMEOUT", f"EnergyPlus exceeded {self.timeout_seconds} seconds.",
            )
        result = EnergyPlusRunResult(
            run_directory.name,
            self.city,
            candidate_id,
            label,
            hashlib.sha256(label.encode("utf-8")).hexdigest(),
            str(run_directory),
            signature,
            started,
            datetime.now(timezone.utc).isoformat(),
            metrics,
        )
        result_path.write_text(json.dumps(asdict(result), ensure_ascii=False, indent=2), encoding="utf-8")
        self._store_success(result)
        return result
