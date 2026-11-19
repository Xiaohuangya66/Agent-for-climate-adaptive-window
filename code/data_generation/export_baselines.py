"""Collect available per-city baseline rows; never modify the active Agent protocol."""
import csv
import json
from pathlib import Path
from campaign_helpers import registry

ROOT=Path(__file__).resolve().parent


def main():
    lock_path=ROOT/'protocol/baseline_export.lock'
    lock_path.parent.mkdir(exist_ok=True)
    lock=lock_path.open('a')
    import os
    if os.name!='nt':
        import fcntl
        fcntl.flock(lock,fcntl.LOCK_EX)
    rows=[];pending=[]
    for city in registry()['cities']:
        path=ROOT/'baselines'/city/'baseline_row.json'
        if path.exists():rows.append(json.loads(path.read_text()))
        else:pending.append(city)
    output=ROOT/'protocol/city_baselines_goal2.csv'
    output.parent.mkdir(exist_ok=True)
    if rows:
        # Concurrent city jobs can finish together; atomic rename prevents partial reads.
        import tempfile
        with tempfile.NamedTemporaryFile('w',dir=output.parent,delete=False,newline='',encoding='utf-8-sig') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
            name=stream.name
        os.replace(name,output)
    print(json.dumps({'available_baselines':len(rows),'pending':pending,'output':str(output)}))


if __name__=='__main__':main()
