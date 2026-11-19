import os
from pathlib import Path
import sys


def runtime_environment(root):
    root=Path(root).resolve();environment=os.environ.copy()
    environment.setdefault('CITY_SUPPORT_SETTINGS',str(root/'configs/runtime.json'))
    environment['PYTHONPATH']=str(root/'code/runtime')
    environment['PYTHONUTF8']='1'
    environment['PYTHONIOENCODING']='utf-8'
    for stream in [sys.stdout,sys.stderr]:
        if hasattr(stream,'reconfigure'):stream.reconfigure(encoding='utf-8',errors='backslashreplace')
    for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:environment[key]='1'
    if os.name=='nt' and not environment.get('ENERGYPLUS_EXE'):
        candidates=[root/'software/EnergyPlus23.1.0/energyplus.exe',
                    Path('C:/EnergyPlusV23-1-0/energyplus.exe'),Path('D:/EnergyPlusV23-1-0/energyplus.exe')]
        for path in candidates:
            if path.is_file():environment['ENERGYPLUS_EXE']=str(path.resolve());break
    return environment
