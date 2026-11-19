"""Discover arbitrary target cities and provision a run-local physical protocol."""
import copy
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unicodedata

from llm_controller.api import redact

PATH_KEYS = ('project_root', 'campaign_root', 'baseline_table', 'reference_assets', 'energyplus_exe',
             'model_file', 'checkpoint', 'vocab_file', 'source_anchors', 'model_run')


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(redact(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)), encoding='utf-8')
    temporary.replace(path)


def normalize(value):
    return ''.join(c for c in unicodedata.normalize('NFKC', value).casefold() if c.isalnum())


def safe_city(value):
    value = unicodedata.normalize('NFKC', value).strip()
    if (not value or len(value) > 80 or value in {'.', '..'} or '..' in value or
        not all(c.isalnum() or c in " .'-_" for c in value) or not any(c.isalpha() for c in value)):
        raise ValueError('City must be a name, not a path or command')
    return value


def design_spec(context, *, budget=None, generation_batch=None, candidate_budget=384):
    source = context['role'] == 'source'
    budget = budget if budget is not None else 40
    batch = generation_batch if generation_batch is not None else (64 if source else 128)
    if (type(budget) is not int or type(batch) is not int or type(candidate_budget) is not int or
        budget < (1 if source else 17) or batch < 1 or candidate_budget < batch):
        raise ValueError('Source runs need a positive budget; transfer runs need 16 probes plus online queries; candidate budget must cover one generation round')
    return {'budget': budget, 'batch_size': batch, 'candidate_budget': candidate_budget}


class CityAssetService:
    def __init__(self, root, output, *, epw=None, ddy=None, weather_dirs=(), catalog=None, runtime_settings=None):
        self.root = Path(root).resolve(); self.output = Path(output).resolve()
        self.settings_path = Path(runtime_settings).resolve() if runtime_settings else self.root/'configs/runtime.json'
        self.config = json.loads(self.settings_path.read_text(encoding='utf-8'))
        for key in PATH_KEYS:
            if key in self.config:self.config[key] = self.absolute(self.config[key])
        for record in self.config.get('targets', {}).values():
            for key in ('epw', 'idf', 'baseline_idf'):
                if key in record:record[key] = self.absolute(record[key])
        for section in ('transfer', 'tree_adaptation'):
            for key in ('selection_file', 'source_selection_file'):
                if key in self.config.get(section, {}):
                    self.config[section][key] = self.absolute(self.config[section][key])
        self.config['bundle_relative_paths'] = False
        if os.environ.get('ENERGYPLUS_EXE'):
            self.config['energyplus_exe'] = str(Path(os.environ['ENERGYPLUS_EXE']).resolve())
        campaign = Path(self.config['campaign_root'])
        self.registry = json.loads((campaign/'weather_registry.json').read_text(encoding='utf-8'))
        self.project = json.loads((campaign/'configs/project.json').read_text(encoding='utf-8'))
        with Path(self.config['baseline_table']).open(encoding='utf-8-sig') as stream:
            self.baselines = list(csv.DictReader(stream))
        path = self.root/'configs/city_catalog.json'
        self.catalog = json.loads(path.read_text(encoding='utf-8')) if path.is_file() else {}
        if catalog:
            extra = json.loads(Path(catalog).resolve().read_text(encoding='utf-8'))
            self.catalog = {**self.catalog, **extra,
                'aliases': {**self.catalog.get('aliases', {}), **extra.get('aliases', {})},
                'cities': {**self.catalog.get('cities', {}), **extra.get('cities', {})},
                'weather_roots': list(dict.fromkeys([*self.catalog.get('weather_roots', []), *extra.get('weather_roots', [])]))}
        if self.catalog.get('schema_version', 1) != 1:raise ValueError('Unsupported city catalog schema')
        self.sources = list(self.registry['source_cities'])
        self.explicit_epw = str(Path(epw).resolve()) if epw else None
        self.explicit_ddy = str(Path(ddy).resolve()) if ddy else None
        self.weather_roots = [Path(self.absolute(p)) for p in
            [*self.catalog.get('weather_roots', ['inputs/weather']), *weather_dirs]]
        self.context = None

    def absolute(self, value):
        path = Path(value).expanduser()
        return str((path if path.is_absolute() else self.root/path).resolve())

    def canonical_city(self, value):
        value = safe_city(value)
        names = {normalize(c): c for c in [*self.registry['cities'], *self.config.get('targets', {}),
                                           *self.catalog.get('cities', {})]}
        for alias, city in self.catalog.get('aliases', {}).items():names[normalize(alias)] = city
        for city, record in self.catalog.get('cities', {}).items():
            for alias in record.get('aliases', []):names[normalize(alias)] = city
        for city in self.sources:names[normalize(city)] = city
        return safe_city(names.get(normalize(value), value.replace(' ', '')))

    @staticmethod
    def epw_header(path):
        with Path(path).open(encoding='utf-8-sig') as stream:
            lines = [stream.readline() for _ in range(8)]
        fields = next(csv.reader([lines[0]]))
        if len(fields) < 10 or fields[0].upper() != 'LOCATION':raise ValueError('Invalid EPW LOCATION header')
        lat, lon, tz, elevation = map(float, fields[6:10])
        if not all(math.isfinite(v) for v in (lat, lon, tz, elevation)) or not -90 <= lat <= 90 or not -180 <= lon <= 180 or not -12 <= tz <= 14:
            raise ValueError('Invalid EPW location coordinates')
        return {'location': fields[1], 'station': fields[5], 'latitude': lat, 'longitude': lon,
                'time_zone': tz, 'elevation_m': elevation, 'protocol_text': '\n'.join(lines)}

    def inspect(self, requested):
        city = self.canonical_city(requested)
        source = city in self.sources
        registered = copy.deepcopy(self.config.get('targets', {}).get(city, {}))
        record = copy.deepcopy(self.catalog.get('cities', {}).get(city, {}))
        if source:
            campaign = Path(self.config['campaign_root'])
            weather = self.registry['cities'][city]
            registered = {'epw': str((campaign/weather['epw']).resolve()),
                'idf': str((campaign/self.project['cities'][city]['base_idf']).resolve()),
                'baseline_idf': str((campaign/self.project['cities'][city]['baseline_idf']).resolve()),
                'weather_dataset': self.registry['dataset'], 'weather_period': self.registry['period']}
        explicit = self.explicit_epw or (self.absolute(record['epw']) if record.get('epw') else None)
        if source and explicit and Path(explicit) != Path(registered['epw']):
            raise ValueError('A frozen source city cannot switch weather while reusing its trained model')
        candidates = []
        if explicit:candidates = [Path(explicit)]
        elif registered.get('epw'):candidates = [Path(registered['epw'])]
        else:
            needles = [normalize(city), *[normalize(x) for x in record.get('aliases', [])]]
            for directory in self.weather_roots:
                if not directory.is_dir():continue
                for path in directory.rglob('*'):
                    if not path.is_file() or path.suffix.casefold() != '.epw':continue
                    name = normalize(path.stem)
                    try:location = normalize(self.epw_header(path)['location'])
                    except (ValueError, OSError):continue
                    if any(len(n) >= 3 and (n in name or n in location) for n in needles):candidates.append(path.resolve())
        candidates = sorted(set(candidates))
        exists = [p for p in candidates if p.is_file()]
        context = {'city': city, 'requested_city': requested, 'source_cities': self.sources,
            'role': 'source' if source else 'target', 'registration': 'existing' if registered else 'new',
            'support_mode': 'native' if source else 'transfer', 'status': 'needs_assets', 'missing_assets': [],
            'searched_weather_roots': [str(p) for p in self.weather_roots],
            'weather_dataset': self.registry['dataset'], 'weather_period': self.registry['period'],
            'reference_baseline_budget': 'separate from design queries; at most one dispatch in this run',
            'new_target_database_required': False}
        if not exists:
            context['missing_assets'] = ['EPW']
            context['message'] = f'未找到 {city} 的天气文件。请下载该城市的 {self.registry["dataset"]} {self.registry["period"]} 天气包（EPW及同站DDY），放入 inputs/weather，或通过 --epw/--ddy 指定。'
            return self._remember(context)
        if len(exists) != 1:
            context.update(status='needs_weather_selection', weather_candidates=[str(p) for p in exists],
                message='找到多个天气站，不能自动选第一个；请用 --epw 指定本次天气文件。')
            return self._remember(context)
        epw = exists[0]
        header = self.epw_header(epw)
        # Existing registrations are explicit protocol metadata, not guessed names.
        inherited = registered if registered.get('epw') and Path(registered['epw']) == epw else {}
        metadata = {**inherited, **record}
        expected = self.registry['dataset'], self.registry['period']
        declared = metadata.get('weather_dataset'), metadata.get('weather_period')
        protocol_text = (epw.name+'\n'+header['protocol_text']).casefold()
        recorded_periods = re.findall(r'(?:tmyx[._ -]|period of record\s*[=:]\s*)(\d{4}-\d{4})', protocol_text)
        if any(period != expected[1] for period in recorded_periods):
            context.update(status='weather_protocol_mismatch', missing_assets=['matching_weather_protocol'],
                           message='天气文件声明的时段与源模型协议不一致，不能复用旧城市的元数据。')
            return self._remember(context)
        if declared != expected and not all(v.casefold() in protocol_text for v in expected):
            context.update(status='weather_protocol_mismatch', missing_assets=['matching_weather_protocol'],
                message=f'天气数据集/时段未证实匹配 {expected[0]} {expected[1]}，请提供匹配文件或在城市目录声明经确认的元数据。')
            return self._remember(context)
        if any(value and value != required for value, required in zip(declared, expected)):
            context.update(status='weather_protocol_mismatch', missing_assets=['matching_weather_protocol'],
                           message='城市目录中的天气协议与源模型协议不一致，不能静默混用。')
            return self._remember(context)
        from src.tools.city_support import climate_features
        climate = climate_features(str(epw))
        context.update(epw=str(epw), station={k: v for k, v in header.items() if k != 'protocol_text'},
                       climate=climate, weather_match='explicit' if explicit else 'registered_or_city_name')
        baseline = next((r for r in self.baselines if r['city'] == city), None)
        if source and not baseline:
            context.update(missing_assets=['source_baseline'], message='源城市的冻结基线缺失，请恢复原协议文件。')
            return self._remember(context)
        same_weather = bool(registered and Path(registered.get('epw', '')) == epw)
        if registered and same_weather and baseline:
            if city in self.config.get('targets', {}):
                recorded_epw = Path(self.absolute(baseline['epw']))
                if recorded_epw != epw:raise ValueError('Registered target baseline/weather mismatch')
            if baseline['weather_dataset'] != expected[0] or baseline['weather_period'] != expected[1]:
                raise ValueError('Registered baseline protocol mismatch')
            for key in ['UDI', 'TotalEnergy_J']:
                if not math.isfinite(float(baseline[key])) or float(baseline[key]) <= 0:raise ValueError('Invalid baseline')
            if (float(baseline['UDI']) > 1 or baseline['formal_udi_definition'] != 'daylight_hours' or
                'NaturalGas:Facility' not in baseline['energy_definition'] or
                not math.isclose(float(baseline['UDI_threshold_95pct']), .95*float(baseline['UDI']), rel_tol=1e-10)):
                raise ValueError('Registered baseline must satisfy the formal daylight/site-energy protocol')
            missing = [key for key in ['idf', 'baseline_idf'] if not Path(registered.get(key, '')).is_file()]
            if source and not (Path(self.config['model_run'])/'models'/city/'metadata.json').is_file():missing.append('source_model')
            if missing:
                context.update(missing_assets=missing, message='已注册城市资产缺失，请恢复所列资产。')
            else:
                context.update(status='ready', assets=registered, baseline=baseline, baseline_reused=True,
                               next_action='initialize_agent')
            return self._remember(context)
        ddy = Path(self.explicit_ddy or self.absolute(record['ddy'])) if (self.explicit_ddy or record.get('ddy')) else epw.with_suffix('.ddy')
        template_city = record.get('template_city', self.catalog.get('default_template_city', 'Shenzhen'))
        template_city = safe_city(template_city)
        directory = self.root/'inputs/buildings'/template_city
        design = Path(self.absolute(record['idf_template'])) if record.get('idf_template') else directory/'Midrise_ReadyForLoop.idf'
        clear = Path(self.absolute(record['baseline_idf_template'])) if record.get('baseline_idf_template') else directory/'Midrise_ClearGlass3mm_Baseline.idf'
        missing = [key for key, path in [('DDY', ddy), ('design_idf_template', design), ('clear_idf_template', clear)] if not path.is_file()]
        context.update(ddy=str(ddy), design_template=str(design), clear_template=str(clear), template_city=template_city,
                       baseline_reused=False, missing_assets=missing)
        if missing:
            context['message'] = '天气已找到；请补充同站 DDY 和所列建筑模板。不能沿用深圳的设计日或清玻璃数值。'
        else:
            # Check before advertising that preparation can proceed.
            self._localized(design.read_text(encoding='utf-8-sig'), context)
            context.update(status='needs_preparation', next_action='prepare_city_assets')
        return self._remember(context)

    def _remember(self, context):
        self.context = context
        return copy.deepcopy(context)

    def _helpers(self):
        path = str(self.root/'code/data_generation')
        if path not in sys.path:sys.path.insert(0, path)
        import campaign_helpers
        return campaign_helpers

    def _localized(self, base, context):
        helper = self._helpers()
        objects = list(helper.idf_objects(Path(context['ddy']).read_text(encoding='utf-8-sig')))
        station = context['station']
        for _, fields in objects:
            if fields[0].casefold() == 'site:location':
                if abs(float(fields[2])-station['latitude']) > .25 or abs(float(fields[3])-station['longitude']) > .25:
                    raise ValueError('DDY location differs from EPW station')
        selections = []
        for condition in [r'ann\s+htg\s+99\.6%\s+condns\s+db', r'ann\s+clg\s+0?\.4%\s+condns\s+db\s*=>\s*mwb']:
            matches = [(raw, fields) for raw, fields in objects if fields[0].casefold() == 'sizingperiod:designday'
                       and len(fields) > 5 and re.search(condition, fields[1], flags=re.I)]
            if len(matches) != 1:raise ValueError('DDY must have exactly one heating99.6% and cooling0.4% DB=>MWB condition')
            selections.append(matches[0])
        location_name = re.sub(r'[,;\r\n]', ' ', station['location'])
        location = '\nSite:Location,'+','.join([location_name, str(station['latitude']), str(station['longitude']),
                                               str(station['time_zone']), str(station['elevation_m'])])+';\n'
        output = [raw for raw, fields in helper.idf_objects(base) if fields[0].casefold() != 'site:location'
                  and not fields[0].casefold().startswith('sizingperiod:')]
        text = ''.join(output)+location+''.join('\n'+raw for raw, _ in selections)+'\n'
        return text, {'station': station['station'], 'latitude': station['latitude'], 'longitude': station['longitude'],
                      'ddy': context['ddy'], 'heating_design_day': selections[0][1][1], 'cooling_design_day': selections[1][1][1]}

    def prepare(self):
        c = self.context
        if not c or c['status'] != 'needs_preparation':raise ValueError('City is not ready for preparation')
        directory = self.output/'city_support/assets'; directory.mkdir(parents=True, exist_ok=False)
        for source, name in [('design_template', 'Midrise_ReadyForLoop.idf'),
                             ('clear_template', 'Midrise_ClearGlass3mm_Baseline.idf')]:
            text, sizing = self._localized(Path(c[source]).read_text(encoding='utf-8-sig'), c)
            # Preserve design spectra; ensure shared parser's audit output requests.
            if source == 'clear_template':text = self._helpers().build_idf_text(text)
            (directory/name).write_text(text, encoding='utf-8')
        c.update(idf=str(directory/'Midrise_ReadyForLoop.idf'),
                 baseline_idf=str(directory/'Midrise_ClearGlass3mm_Baseline.idf'), sizing=sizing,
                 status='needs_baseline', next_action='run_clear_glass_baseline')
        write_json(directory/'city_asset_provenance.json', c)
        return copy.deepcopy(c)

    def run_baseline(self):
        c = self.context
        if not c or c['status'] != 'needs_baseline':raise ValueError('City assets must be prepared first')
        directory = self.output/'city_support/clear_glass'
        directory.mkdir(parents=True, exist_ok=False)
        executable = Path(self.config['energyplus_exe'])
        if not executable.is_file():raise FileNotFoundError(executable)
        input_file = directory/'input.idf'
        input_file.write_text(Path(c['baseline_idf']).read_text(encoding='utf-8'), encoding='utf-8')
        command = [str(executable), '-w', c['epw'], '-d', str(directory), '-r', str(input_file)]
        dispatch = {'city': c['city'], 'role': 'clear_glass_reference', 'command': command,
                    'accounting': 'one reference process, excluded from probe/search budget and learning labels',
                    'started_utc': datetime.now(timezone.utc).isoformat()}
        write_json(directory/'baseline_dispatch.json', dispatch)
        c['reference_baseline_attempts'] = 1
        environment = os.environ.copy(); environment.pop('DMX_API_KEY', None)
        try:
            with (directory/'stdout.log').open('wb') as out, (directory/'stderr.log').open('wb') as err:
                completed = subprocess.run(command, cwd=directory, stdout=out, stderr=err,
                    env=environment, timeout=1800, check=False)
            from thermo_agent.energyplus import parse_energyplus_outputs
            metrics = parse_energyplus_outputs(directory, return_code=completed.returncode,
                                               udi_denominator_mode='daylight_hours')
            write_json(directory/'metrics.json', asdict(metrics))
            if metrics.status != 'success':raise RuntimeError(f'{metrics.error_code}: {metrics.error_message}')
            self._helpers().validate_metrics(directory, metrics)
            if metrics.udi <= 0 or metrics.total_energy_J <= 0:raise ValueError('Baseline must have positive real UDI and energy')
            c['baseline'] = {'city': c['city'], 'UDI': metrics.udi, 'UDI_threshold_95pct': .95*metrics.udi,
                'TotalEnergy_J': metrics.total_energy_J, 'TotalEnergy_kWh': metrics.total_energy_kWh,
                'formal_udi_definition': 'daylight_hours', 'energy_definition': 'Electricity:Facility + NaturalGas:Facility',
                'epw': c['epw'], 'weather_dataset': c['weather_dataset'], 'weather_period': c['weather_period'],
                'baseline_run_directory': str(directory), 'completed_utc': datetime.now(timezone.utc).isoformat()}
            write_json(directory/'baseline_row.json', c['baseline'])
            c.update(status='ready', next_action='initialize_agent', reference_baseline_processes=1)
            write_json(directory/'baseline_result.json', {'status': 'success', 'baseline': c['baseline']})
        except Exception as exc:
            c.update(status='baseline_failed', message=redact(f'{type(exc).__name__}: {exc}'), next_action='inspect_baseline_evidence')
            write_json(directory/'baseline_result.json', {'status': 'failed', 'error': c['message']})
        return copy.deepcopy(c)

    def create_runtime_overlay(self):
        c = self.context
        if not c or c['status'] != 'ready':raise ValueError('A verified baseline is required before Agent initialization')
        if c.get('baseline_reused'):return self.settings_path
        cfg = copy.deepcopy(self.config)
        record = {key: c[key] for key in ['epw', 'idf', 'baseline_idf', 'weather_dataset', 'weather_period', 'sizing']}
        cfg.setdefault('targets', {})[c['city']] = record
        rows = [copy.deepcopy(r) for r in self.baselines if r['city'] != c['city']]
        for row in rows:
            if row['city'] in cfg['targets']:row['epw'] = self.absolute(row['epw'])
        rows.append(c['baseline'])
        directory = self.output/'city_support/protocol'; directory.mkdir(parents=True, exist_ok=True)
        table = directory/'city_baselines_goal2.csv'
        fields = list(dict.fromkeys(k for row in rows for k in row))
        with table.open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
        cfg['baseline_table'] = str(table)
        overlay = directory/'runtime.json'; write_json(overlay, cfg)
        write_json(directory/'city_resolution.json', c)
        return overlay

    def activate(self):
        path = self.create_runtime_overlay()
        previous = os.environ.get('CITY_SUPPORT_SETTINGS')
        if (not previous or Path(previous).resolve() != path.resolve()) and any(k in sys.modules for k in ['src.config', 'src.runtime_profile']):
            raise RuntimeError('Activate new-city protocol before importing numerical runtime; use a fresh process')
        os.environ['CITY_SUPPORT_SETTINGS'] = str(path)
        return path
