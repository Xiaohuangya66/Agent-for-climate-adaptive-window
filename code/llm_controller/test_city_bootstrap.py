"""Synthetic fixtures and mocked dispatch only; no API or EnergyPlus execution."""
from dataclasses import dataclass
from datetime import datetime, timedelta
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

from llm_controller.city_assets import CityAssetService, design_spec, safe_city
from llm_controller.city_bootstrap import CityBootstrapSession, resolve_city_request
from llm_controller.controller import ToolRejected

ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Metrics:
    status: str = 'success'
    udi: float = .8
    total_energy_J: float = 900.
    total_energy_kWh: float = .00025
    electricity_J: float = 800.
    gas_J: float = 100.
    error_code: str = ''
    error_message: str = ''


class CityFixtures(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.weather = self.directory/'weather'; self.weather.mkdir()
        self.output = self.directory/'run'
        self.epw = self.weather/'TST_Testville.123456_TMYx.2011-2025.epw'

    def service(self, **kwargs):
        return CityAssetService(ROOT, self.output, weather_dirs=[self.weather], **kwargs)

    def weather_pair(self):
        rows = ['LOCATION,Testville,AA,TST,TMYx,123456,20,110,8,10\n']
        rows += ['COMMENTS 1,Period of Record=2011-2025\n']*7
        for i in range(8760):
            date = datetime(2017, 1, 1)+timedelta(hours=i)
            fields = ['0']*16
            fields[0:5] = ['2017', str(date.month), str(date.day), str(date.hour+1), '60']
            for index, value in [(6, '25'), (8, '65'), (13, '250'), (15, '100')]:fields[index] = value
            rows.append(','.join(fields)+'\n')
        self.epw.write_text(''.join(rows), encoding='utf-8')
        self.epw.with_suffix('.ddy').write_text(
            'Site:Location,Testville,20,110,8,10;\n'
            'SizingPeriod:DesignDay,Testville Ann Htg 99.6% Condns DB,1,21,WinterDesignDay,7;\n'
            'SizingPeriod:DesignDay,Testville Ann Clg .4% Condns DB=>MWB,7,21,SummerDesignDay,34;\n', encoding='utf-8')

    def prepared(self):
        self.weather_pair(); service = self.service()
        context = service.inspect('Testville')
        self.assertEqual(context['status'], 'needs_preparation')
        service.prepare()
        return service

    def test_unknown_city_is_transfer_and_missing_weather_blocks(self):
        service = self.service()
        context = service.inspect('CompletelyNewCity')
        self.assertEqual(context['role'], 'target')
        self.assertEqual(context['support_mode'], 'transfer')
        self.assertEqual(context['missing_assets'], ['EPW'])
        self.assertIn('下载', context['message'])
        self.assertFalse(self.output.exists())

    def test_source_and_registered_target_are_manifest_driven(self):
        service = self.service()
        source = service.inspect('新加坡')
        target = service.inspect('深圳')
        self.assertEqual((source['city'], source['role'], source['status']), ('Singapore', 'source', 'ready'))
        self.assertEqual((target['city'], target['role'], target['status']), ('Shenzhen', 'target', 'ready'))
        self.assertTrue(target['baseline_reused'])

    def test_city_is_not_a_filesystem_path(self):
        for value in ['../Paris', 'C:\\City', 'A;B', 'A*B']:
            with self.subTest(value=value), self.assertRaises(ValueError):safe_city(value)

    def test_discovery_does_not_require_city_in_catalog(self):
        self.weather_pair()
        context = self.service().inspect('Testville')
        self.assertEqual(context['status'], 'needs_preparation')
        self.assertEqual(context['registration'], 'new')
        self.assertEqual(context['station']['station'], '123456')

    def test_multiple_stations_require_selection(self):
        self.weather_pair()
        copy = self.weather/'TST_Testville-Airport_TMYx.2011-2025.epw'
        copy.write_bytes(self.epw.read_bytes())
        context = self.service().inspect('Testville')
        self.assertEqual(context['status'], 'needs_weather_selection')
        self.assertEqual(len(context['weather_candidates']), 2)

    def test_ddy_is_required_and_checked_against_station(self):
        self.weather_pair(); self.epw.with_suffix('.ddy').unlink()
        context = self.service().inspect('Testville')
        self.assertIn('DDY', context['missing_assets'])
        self.weather_pair()
        text = self.epw.with_suffix('.ddy').read_text().replace('Testville,20,110', 'Testville,30,110')
        self.epw.with_suffix('.ddy').write_text(text)
        with self.assertRaisesRegex(ValueError, 'DDY location'):self.service().inspect('Testville')

    def test_weather_protocol_mismatch_blocks(self):
        self.weather_pair()
        path = self.weather/'TST_Testville_TMYx.2009-2023.epw'
        path.write_text(self.epw.read_text().replace('2011-2025', '2009-2023'))
        context = self.service(epw=path).inspect('Testville')
        self.assertEqual(context['status'], 'weather_protocol_mismatch')

    def test_preparation_localizes_both_idfs_without_running(self):
        with patch('llm_controller.city_assets.subprocess.run') as run:
            service = self.prepared()
        run.assert_not_called()
        helper = service._helpers()
        for key in ['idf', 'baseline_idf']:
            objects = list(helper.idf_objects(Path(service.context[key]).read_text(encoding='utf-8')))
            locations = [fields for _, fields in objects if fields[0].casefold() == 'site:location']
            sizing = [fields[1] for _, fields in objects if fields[0].casefold() == 'sizingperiod:designday']
            self.assertEqual(locations[0][1:4], ['Testville', '20.0', '110.0'])
            self.assertEqual(len(sizing), 2)
            self.assertTrue(all('Testville' in name for name in sizing))
        self.assertEqual(service.context['status'], 'needs_baseline')
        self.assertFalse((self.output/'city_support/protocol/runtime.json').exists())

    def test_baseline_mocked_dispatch_is_separate_and_cannot_repeat(self):
        service = self.prepared()
        helper = service._helpers()
        with patch('llm_controller.city_assets.subprocess.run', return_value=Mock(returncode=0)) as run, \
             patch('thermo_agent.energyplus.parse_energyplus_outputs', return_value=Metrics()), \
             patch.object(helper, 'validate_metrics') as validate:
            result = service.run_baseline()
        run.assert_called_once(); validate.assert_called_once()
        self.assertEqual(result['status'], 'ready')
        self.assertEqual(result['baseline']['city'], 'Testville')
        self.assertEqual(result['baseline']['TotalEnergy_J'], 900.)
        self.assertFalse((self.output/'queries.jsonl').exists())
        with self.assertRaises(ValueError):service.run_baseline()

    def test_failed_baseline_cannot_publish_runtime_overlay(self):
        service = self.prepared()
        with patch('llm_controller.city_assets.subprocess.run', return_value=Mock(returncode=1)), \
             patch('thermo_agent.energyplus.parse_energyplus_outputs', return_value=Metrics(status='failed')):
            result = service.run_baseline()
        self.assertEqual(result['status'], 'baseline_failed')
        with self.assertRaises(ValueError):service.create_runtime_overlay()

    def test_overlay_resolves_unknown_target_through_existing_runtime(self):
        service = self.prepared()
        with patch('llm_controller.city_assets.subprocess.run', return_value=Mock(returncode=0)), \
             patch('thermo_agent.energyplus.parse_energyplus_outputs', return_value=Metrics()), \
             patch.object(service._helpers(), 'validate_metrics'):
            service.run_baseline()
        overlay = service.create_runtime_overlay()
        config = json.loads(overlay.read_text())
        self.assertIn('Testville', config['targets'])
        self.assertFalse(config['bundle_relative_paths'])
        environment = os.environ.copy(); environment['CITY_SUPPORT_SETTINGS'] = str(overlay)
        environment['PYTHONPATH'] = str(ROOT/'code/runtime')
        # Only instantiate the asset/protocol readers, never Agent/EnergyPlus.
        completed = subprocess.run([sys.executable, '-X', 'utf8', '-c',
            "from src.runtime_profile import city_assets; from src.agent.formal_protocol import FormalProtocol,configure_udi_ratio; "
            "configure_udi_ratio(.9); a=city_assets('Testville'); p=FormalProtocol('Testville'); "
            "assert a['support_mode']=='transfer'; assert abs(p.threshold-.72)<1e-12; "
            "from src.config import CITY_EPW_MAP; assert 'Testville' in CITY_EPW_MAP; print('PASS')"],
            env=environment, capture_output=True, text=True, check=True)
        self.assertIn('PASS', completed.stdout)
        self.assertNotEqual(config['baseline_table'], str(ROOT/'protocol/city_baselines_goal2.csv'))
        reused = self.service(runtime_settings=overlay).inspect('Testville')
        self.assertEqual(reused['status'], 'ready')
        self.assertTrue(reused['baseline_reused'])

    def test_bootstrap_roles_version_and_idempotency(self):
        service = self.service(); session = CityBootstrapSession(service, '请优化新加坡', '新加坡')
        args = {'state_version': 0, 'rationale': '先核对城市注册信息', 'city': 'Singapore'}
        first = session.call('resolve_city', args, 'resolve-1')
        self.assertEqual(session.phase, 'ready')
        self.assertEqual(first, session.call('resolve_city', args, 'resolve-1'))
        self.assertEqual(session.version, 1)
        self.assertEqual(first['state']['design_spec']['budget'], 40)

    def test_new_city_is_blocked_before_any_numerical_agent(self):
        service = self.service(); session = CityBootstrapSession(service, '优化Testville')
        result = session.call('resolve_city', dict(state_version=0, rationale='核验资产', city='Testville'), 'a')
        self.assertEqual(session.phase, 'blocked')
        self.assertFalse(result['state']['numerical_agent_initialized'])
        with self.assertRaises(ToolRejected):session.call('run_clear_glass_baseline', dict(state_version=1, rationale='越过天气'), 'b')

    def test_invalid_transfer_budget_stops_before_baseline(self):
        self.weather_pair(); service = self.service()
        session = CityBootstrapSession(service, '优化Testville', run_options={'budget': 10})
        session.call('resolve_city', dict(state_version=0, rationale='核验预算', city='Testville'), 'a')
        self.assertEqual(session.phase, 'blocked')
        self.assertFalse((self.output/'city_support/clear_glass').exists())

    def test_arbitrary_target_has_transfer_defaults(self):
        self.assertEqual(design_spec({'role': 'target'})['budget'], 40)
        self.assertEqual(design_spec({'role': 'target'})['batch_size'], 128)

    def test_single_conversation_reaches_new_city_readiness_with_mock_tools(self):
        self.weather_pair(); service = self.service()
        helper = service._helpers()
        calls = []
        def complete(messages, offered):
            state = json.loads(messages[-1]['content'])['state']
            name = {'city_request': 'resolve_city', 'prepare': 'prepare_city_assets',
                    'baseline': 'run_clear_glass_baseline'}[state['phase']]
            args = {'state_version': state['state_version'], 'rationale': '按城市资产状态选择工具'}
            if name == 'resolve_city':args['city'] = 'Testville'
            calls.append(name)
            return {'role': 'assistant', 'content': None, 'tool_calls': [{
                'id': f'call-{len(calls)}', 'type': 'function',
                'function': {'name': name, 'arguments': json.dumps(args)}}]}, {'usage': {}}
        client = Mock(); client.complete.side_effect = complete
        with patch('llm_controller.city_assets.subprocess.run', return_value=Mock(returncode=0)), \
             patch('thermo_agent.energyplus.parse_energyplus_outputs', return_value=Metrics()), \
             patch.object(helper, 'validate_metrics'):
            outcome, messages = resolve_city_request(service, client, '请为Testville优化节能窗')
        self.assertEqual(outcome['status'], 'ready')
        self.assertEqual(outcome['version'], 3)
        self.assertEqual(calls, ['resolve_city', 'prepare_city_assets', 'run_clear_glass_baseline'])
        self.assertEqual(len([m for m in messages if m['role'] == 'system']), 1)
        self.assertEqual(client.complete.call_count, 3)

    def test_ambiguous_request_can_return_clarification_without_simulation(self):
        service = self.service(); session = CityBootstrapSession(service, '请对两个城市进行优化')
        result = session.call('clarify_city', dict(state_version=0, rationale='需要明确本次城市',
            question='请指定本次需要优化的一个城市。'), 'clarify-1')
        self.assertEqual(session.phase, 'blocked')
        self.assertEqual(result['result']['status'], 'needs_city_clarification')
        self.assertIsNone(service.context)


if __name__ == '__main__':unittest.main()
