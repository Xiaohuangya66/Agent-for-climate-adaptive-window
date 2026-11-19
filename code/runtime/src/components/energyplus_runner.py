"""Server-compatible EnergyPlus adapter with persistent, isolated run artifacts."""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from typing import Dict, Optional

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / 'src' / 'vendor'))
from thermo_agent.energyplus import EnergyPlusMetrics, glazing_spectral_block, parse_energyplus_outputs
from thermo_agent.spectra import SpectralState
from src.config import CITY_EPW_MAP, ENERGYPLUS_EXE, EPW_DIR, IDF_DIR, TIMEOUT_ENERGYPLUS


def _absolute(path: str | Path) -> Path:
    path = Path(path).expanduser()
    return (path if path.is_absolute() else _ROOT / path).resolve()


class EnergyPlusRunner:
    """Keep run(structure_str, city, output_dir=None); never fabricate metrics."""

    def __init__(self, idf_path: Optional[Path] = None,
                 epw_path: Optional[Path] = None,
                 energyplus_exe: Optional[str] = None):
        self._explicit_idf = idf_path is not None
        self.idf_path = _absolute(idf_path or IDF_DIR / 'Midrise_ReadyForLoop.idf')
        self.epw_path = _absolute(epw_path) if epw_path is not None else None
        self.energyplus_exe = str(_absolute(energyplus_exe or ENERGYPLUS_EXE))
        for path in [Path(self.energyplus_exe), self.idf_path]:
            if not path.is_file():
                raise FileNotFoundError(path)
        print('✓ EnergyPlusRunner initialized (reference parser, persistent outputs)')

    def _work_directory(self, city: str, output_dir: Optional[Path]) -> Path:
        if output_dir is None:
            parent = _ROOT / 'results' / 'energyplus_runs' / city
            parent.mkdir(parents=True, exist_ok=True)
            return Path(tempfile.mkdtemp(prefix='run_', dir=parent)).resolve()
        requested = _absolute(output_dir)
        requested.mkdir(parents=True, exist_ok=True)
        if any(requested.iterdir()):
            # Never parse stale output or overwrite an earlier simulation.
            return Path(tempfile.mkdtemp(prefix=requested.name + '_attempt_',
                                         dir=requested.parent)).resolve()
        return requested

    @staticmethod
    def _diagnostic(work_dir: Path) -> str:
        evidence = []
        for name in ['stdout.log', 'stderr.log', 'eplusout.err', 'eplusout.end']:
            path = work_dir / name
            if not path.is_file():
                continue
            lines = path.read_text(encoding='utf-8', errors='replace').splitlines()
            important = [line.strip() for line in lines
                         if re.search(r'\*\*\s*(Severe|Fatal)\b|\bERROR\b|Could not|not found|'
                                      r'Permission denied|Disk quota|No space|terminate', line, re.I)]
            if important:
                evidence.append(name + ': ' + ' | '.join(important[:4]))
        return '\n'.join(evidence)[:2400]

    def run(self, structure_str: str, city: str,
            output_dir: Optional[Path] = None) -> Dict:
        # Validate city before using it as a directory component.
        if city not in CITY_EPW_MAP:
            return {'status': 'failed', 'UDI': None, 'TotalEnergy': None,
                    'error_code': 'UNKNOWN_CITY', 'error_message': f'Unknown city: {city}',
                    'run_directory': None, 'return_code': None, 'process_started': False}
        if self._explicit_idf:
            base_idf_path=self.idf_path
        else:
            from src.runtime_profile import city_assets
            base_idf_path=Path(city_assets(city)['idf'])
        work_dir = self._work_directory(city, output_dir)
        epw_file = self.epw_path or _absolute(EPW_DIR / CITY_EPW_MAP[city])
        metadata = {'structure': structure_str, 'city': city,
                    'base_idf': str(base_idf_path), 'epw': str(epw_file),
                    'energyplus_executable': self.energyplus_exe,
                    'run_directory': str(work_dir),
                    'udi_definition': {'mode': 'daylight_hours', 'lower_lux': 100,
                                       'upper_lux': 2000},
                    'energy_definition': 'Electricity:Facility + NaturalGas:Facility (RunPeriod, J)',
                    'started_utc': datetime.now(timezone.utc).isoformat()}
        return_code = None
        process_started = False

        def finish(payload: Dict) -> Dict:
            payload.update(run_directory=str(work_dir), return_code=return_code,
                           process_started=process_started, structure=structure_str,
                           udi_denominator_mode='daylight_hours')
            metadata['completed_utc'] = datetime.now(timezone.utc).isoformat()
            (work_dir / 'input.json').write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding='utf-8')
            (work_dir / 'result.json').write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
            return payload

        def failed(code: str, message: str) -> Dict:
            diagnostic = self._diagnostic(work_dir)
            return finish({'status': 'failed', 'UDI': None, 'TotalEnergy': None,
                           'error_code': code,
                           'error_message': message + ('\n' + diagnostic if diagnostic else ''),
                           'diagnostic': diagnostic})

        try:
            if not epw_file.is_file():
                return failed('EPW_NOT_FOUND', f'EPW not found: {epw_file}')
            from src.components.tmm import TMM
            tmm_result = TMM(_ROOT / 'runtime_assets' / 'tmm' / 'TMM_ML').compute(structure_str)

            def state(name: str) -> SpectralState:
                values = tmm_result['spectrum_' + name]
                return SpectralState(tmm_result['wavelengths_nm'], values['T'], values['Rf'], values['Rb'])

            idf_file = work_dir / 'modified.idf'
            idf_file.write_text(self._create_modified_idf(state('cold'), state('hot'), base_idf_path), encoding='utf-8')
            command = [self.energyplus_exe, '-w', str(epw_file), '-d', str(work_dir), '-r', str(idf_file)]
            metadata['command'] = command
            creationflags = (subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP) if os.name == 'nt' else 0
            metadata['process_creationflags'] = creationflags
            (work_dir / 'input.json').write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding='utf-8')
            environment = os.environ.copy()
            environment.pop('DMX_API_KEY', None)
            environment.setdefault('OMP_NUM_THREADS', '1')
            # Stream logs to disk, including partial logs when the process fails or times out.
            with (work_dir / 'stdout.log').open('wb') as stdout, (work_dir / 'stderr.log').open('wb') as stderr:
                process_started = True
                import logging
                logging.getLogger(__name__).info('ENERGYPLUS_STARTED city=%s run=%s idf=%s epw=%s',city,work_dir,base_idf_path,epw_file)
                completed = subprocess.run(command, cwd=str(work_dir), stdout=stdout, stderr=stderr,
                                           timeout=TIMEOUT_ENERGYPLUS, env=environment,
                                           creationflags=creationflags, check=False)
                return_code = completed.returncode
            metrics = parse_energyplus_outputs(work_dir, return_code=return_code,
                                               udi_denominator_mode='daylight_hours')
            if metrics.status != 'success':
                return failed(metrics.error_code, metrics.error_message)
            error = self._validate_parsed_metrics(work_dir, metrics)
            if error:
                return failed(*error)
            (work_dir / 'metrics.json').write_text(json.dumps(asdict(metrics), indent=2, allow_nan=False), encoding='utf-8')
            return finish(self._convert_to_server_format(metrics, work_dir, structure_str))
        except subprocess.TimeoutExpired:
            return failed('TIMEOUT', f'EnergyPlus timeout after {TIMEOUT_ENERGYPLUS}s')
        except Exception as exc:
            return failed('EXCEPTION', f'{type(exc).__name__}: {exc}')

    def _create_modified_idf(self, cold_state: SpectralState, hot_state: SpectralState,
                             base_idf_path=None) -> str:
        from src.components.idf_builder import build_idf_text
        base=Path(base_idf_path or self.idf_path).read_text(encoding='utf-8')
        return build_idf_text(base,cold_state,hot_state)

    @staticmethod
    def _validate_parsed_metrics(work_dir: Path, metrics: EnergyPlusMetrics):
        if metrics.udi_denominator_hours is None or metrics.udi_denominator_hours <= 0 or metrics.udi is None or not np.isfinite(metrics.udi):
            return 'INVALID_UDI', 'UDI requires a positive daylight denominator and a finite result'
        frame = pd.read_csv(work_dir / 'eplusout.csv')
        for name in ['Electricity:Facility', 'NaturalGas:Facility']:
            columns = [c for c in frame if name in c and 'RunPeriod' in c]
            if len(columns) != 1:
                return 'INVALID_METER', f'Expected one {name} RunPeriod column; got {columns}'
            values = pd.to_numeric(frame[columns[0]], errors='coerce').dropna().to_numpy()
            if len(values) != 1 or not np.isfinite(values).all() or (values < 0).any():
                return 'INVALID_METER_VALUES', f'{name} needs exactly one finite nonnegative annual value; found {values.tolist()}'
        return None

    def _convert_to_server_format(self, metrics: EnergyPlusMetrics,
                                  work_dir: Path, structure: str) -> Dict:
        if metrics.status != 'success':
            return {'status': 'failed', 'UDI': None, 'TotalEnergy': None,
                    'error_code': metrics.error_code, 'error_message': metrics.error_message,
                    'run_directory': str(work_dir)}
        return {'status': 'success', 'UDI': float(metrics.udi),
                'TotalEnergy': float(metrics.total_energy_J),
                'ElectricityEnergy': float(metrics.electricity_J),
                'NaturalGasEnergy': float(metrics.gas_J),
                'udi_qualifying_hours': int(metrics.udi_qualifying_hours),
                'udi_denominator_hours': int(metrics.udi_denominator_hours),
                'success_marker': metrics.success_marker, 'severe_errors': metrics.severe_errors,
                'fatal_detected': metrics.fatal_detected,
                'run_directory': str(work_dir), 'structure': structure,
                'error_code': None, 'error_message': None}
