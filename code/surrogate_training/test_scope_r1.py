import importlib.util
import json
from pathlib import Path
import sqlite3
import tempfile
from types import SimpleNamespace
import unittest
import pandas as pd
from training_common import FEATURES,PROTOCOL,LEGACY_PROTOCOL,SOURCE_CITIES,TARGET_CITIES,make_split,read_config
from train_source_surrogates import prepare


class ScopeTests(unittest.TestCase):
    def fixture(self,root):
        campaign=root/'campaign';campaign.mkdir();(campaign/'protocol').mkdir()
        structures=[f'Mat{i}_10 SiO2_{5+j}' for i in range(400) for j in range(4)]
        weather={'dataset':'TMYx','period':'2011-2025','source_cities':list(SOURCE_CITIES),
            'transfer_target_cities':list(TARGET_CITIES),'cities':{c:{'epw':c+'.epw'} for c in SOURCE_CITIES}}
        (campaign/'weather_registry.json').write_text(json.dumps(weather))
        baselines=[]
        for city in [*SOURCE_CITIES,*TARGET_CITIES]:
            baselines.append({'city':city,'UDI':.9,'TotalEnergy_J':1000.,'UDI_threshold_95pct':.855,
                'weather_dataset':'TMYx','weather_period':'2011-2025','epw':city+'.epw',
                'energy_definition':'Electricity:Facility + NaturalGas:Facility','formal_udi_definition':'daylight_hours'})
        pd.DataFrame(baselines).to_csv(campaign/'protocol/city_baselines_goal2.csv',index=False)
        for city in SOURCE_CITIES:
            directory=campaign/'results'/city;directory.mkdir(parents=True)
            frame=pd.DataFrame({'id':range(len(structures)),'structure':structures,'UDI':.8,'TotalEnergy':1000.,
                'ElectricityFacility_J':800.,'NaturalGasFacility_J':200.,'UDI_qualifying_hours':3200,
                'UDI_denominator_hours':4000,'T_cold':','.join(['0.3']*221),'T_hot':','.join(['0.3']*221),**{f:.3 for f in FEATURES}})
            frame.to_csv(directory/f'ML_Dataset_SiteEnergy_{city}.csv',index=False)
            with sqlite3.connect(directory/'site_energy_labels.sqlite3') as connection:
                connection.execute('CREATE TABLE site_energy_labels(city TEXT,status TEXT,structure_hash TEXT)')
                connection.executemany('INSERT INTO site_energy_labels VALUES(?,?,?)',[(city,'success',str(i)) for i in range(len(structures))])
            connection.close()
        fixed=root/'fixed.csv';make_split(structures[:-1]).to_csv(fixed,index=False)
        return campaign,fixed

    def test_prepare_four_sources_filters_targets_and_preserves_frozen_panel(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);campaign,fixed=self.fixture(root)
            config=prepare(campaign,root/'out',split_manifest=fixed)
            self.assertEqual(config['protocol'],PROTOCOL);self.assertEqual(config['common_structures'],1599)
            self.assertEqual(config['available_common_structures'],1600)
            self.assertEqual(config['held_out_climate_cities'],list(TARGET_CITIES))
            self.assertEqual(set(pd.read_csv(root/'out/city_baselines.csv').city),set(SOURCE_CITIES))
            pd.testing.assert_frame_equal(pd.read_csv(fixed),pd.read_csv(root/'out/grouped_split.csv'))
            self.assertFalse(any('Shenzhen' in p.name for p in (root/'out/datasets').iterdir()))

    def test_target_cannot_be_a_source(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);campaign,fixed=self.fixture(root)
            path=campaign/'weather_registry.json';weather=json.loads(path.read_text())
            weather['source_cities']=['Singapore','LosAngeles','Shenzhen'];path.write_text(json.dumps(weather))
            with self.assertRaises(ValueError):prepare(campaign,root/'bad',split_manifest=fixed)
            self.assertFalse((root/'bad').exists())

    def test_frozen_panel_missing_structure_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);campaign,fixed=self.fixture(root)
            panel=make_split([f'Other{i}_5 SiO2_10' for i in range(400)])
            panel.to_csv(fixed,index=False)
            with self.assertRaises(ValueError):prepare(campaign,root/'bad',split_manifest=fixed)

    def test_protocol_compatibility_keeps_history_but_rejects_wrong_new_scope(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);path=root/'run_config.json'
            path.write_text(json.dumps({'protocol':LEGACY_PROTOCOL}))
            self.assertEqual(read_config(root)['protocol'],LEGACY_PROTOCOL)
            path.write_text(json.dumps({'protocol':PROTOCOL,'cities':['Shenzhen'],'held_out_climate_cities':[]}))
            with self.assertRaises(ValueError):read_config(root)

    def test_batch_scope_budget_and_wrapper(self):
        root=Path(__file__).resolve().parents[2]
        spec=importlib.util.spec_from_file_location('batch_scope',root/'code/runtime/experiments/run_batch.py')
        batch=importlib.util.module_from_spec(spec);spec.loader.exec_module(batch)
        self.assertEqual(set(batch.AVAILABLE_CITIES),{*SOURCE_CITIES,*TARGET_CITIES})
        tasks=batch.generate_tasks(SimpleNamespace(cities=list(batch.AVAILABLE_CITIES),methods=['agent_full','nsga2'],
            seeds=None,budget=None,output_dir=None))
        self.assertEqual(len(tasks),48);self.assertEqual(sum(t['budget'] for t in tasks),1920)
        self.assertTrue(all(t['budget']==40 for t in tasks))
        self.assertTrue(batch.task_command(tasks[0])[1].endswith('run_experiment.py'))

if __name__=='__main__':unittest.main()
