"""Read label progress; missing baselines and zero failure counts are not errors."""
from contextlib import closing
import csv
from pathlib import Path
import sqlite3
import sys
from campaign_helpers import registry

ROOT=Path(__file__).resolve().parent


def lock_is_held(path):
    if sys.platform=='win32' or not path.exists():return False
    import fcntl
    with path.open('r+') as stream:
        try:fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return True
        fcntl.flock(stream,fcntl.LOCK_UN)
    return False


def main():
    print('城市\t用途\t清玻璃基线\t成功样本数\t失败样本数\t证据份数\tCSV样本数\t当前状态')
    for city,weather in registry()['cities'].items():
        directory=ROOT/'results'/city
        database=directory/'site_energy_labels.sqlite3'
        counts={};evidence=0;csv_rows=None
        if database.exists():
            with closing(sqlite3.connect(database.resolve().as_uri()+'?mode=ro',uri=True)) as connection:
                if connection.execute("SELECT 1 FROM sqlite_master WHERE name='site_energy_labels'").fetchone():
                    counts=dict(connection.execute('SELECT status,COUNT(*) FROM site_energy_labels GROUP BY status'))
                if connection.execute("SELECT 1 FROM sqlite_master WHERE name='run_evidence'").fetchone():
                    evidence=connection.execute('SELECT COUNT(*) FROM run_evidence').fetchone()[0]
        csv_path=directory/f'ML_Dataset_SiteEnergy_{city}.csv'
        if csv_path.exists():
            with csv_path.open(encoding='utf-8-sig',newline='') as stream:
                reader=csv.reader(stream);next(reader,None);csv_rows=sum(1 for _ in reader)
        baseline=(ROOT/'baselines'/city/'baseline_row.json').exists()
        running=any(lock_is_held(directory/name) for name in ['dataset_run.lock','city_job.lock'])
        if running:state='运行中'
        elif counts:state='已有结果，当前未运行'
        elif baseline:state='已有基线，尚无样本'
        elif weather['role']=='target':state='预留迁移，尚无结果'
        else:state='尚无基线或样本'
        print('\t'.join(map(str,[city,'源城市' if weather['role']=='source' else '迁移测试',
              '已完成' if baseline else '尚未生成',counts.get('success',0),counts.get('failed',0),
              evidence,csv_rows if csv_rows is not None else '尚未导出',state])))
    print('\n说明：失败样本数为 0 表示没有失败；尚未生成基线不等于失败。')
    print('本表统计标签数据库；Slurm 作业是否失败请另查 sacct。CSV 是导出时刻的快照。')


if __name__=='__main__':main()
