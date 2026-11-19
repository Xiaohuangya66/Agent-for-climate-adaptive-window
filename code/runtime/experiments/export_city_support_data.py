"""Rebuild views from the single query ledger without running simulations."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',type=Path,required=True)
    args=parser.parse_args()
    from src.components.tmm import TMM
    from src.tools.query_service import export_support_data
    export_support_data(args.run,tmm=TMM(Path('.')))
    print('EXPORTED',args.run/'city_support')
