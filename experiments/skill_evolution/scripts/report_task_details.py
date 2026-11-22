"""Generate or watch one per-task report; no model calls or benchmark mutations."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'src'))
from lh_harness.skills.task_report import read_json, write_task_report

TERMINAL={'completed','completed_with_learning_errors','failed','interrupted','infrastructure_failure','final_evaluation_failed'}


def main(argv=None):
    parser=argparse.ArgumentParser(description='将逐题评分、Token、耗时和 Skill 使用/审核结果汇总到 tasks/task_summary.json')
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--watch',action='store_true',help='持续刷新已有运行，无需重启训练')
    parser.add_argument('--interval',type=float,default=15)
    parser.add_argument('--settle-seconds',type=float,default=30,help='终态稳定一段时间后退出，避免训练完成与最终复测启动间隙提前退出')
    parser.add_argument('--parent-pid',type=int,help='可选：训练主进程退出后，稳定刷新一段时间再退出')
    args=parser.parse_args(argv)
    if args.interval<=0 or args.settle_seconds<0:
        parser.error('刷新间隔必须为正数，终态稳定等待时间不能为负数')
    root=args.run_dir.expanduser().resolve()
    if not (root/'manifest.json').is_file():
        parser.error('运行目录不存在或缺少 manifest.json')
    identity=hashlib.sha256(str(root).encode()).hexdigest()[:20]
    (ROOT/'.cache').mkdir(exist_ok=True)
    with (ROOT/'.cache'/('task-report-watch-'+identity+'.lock')).open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            print('该运行已有汇总器，无需重复启动。',flush=True)
            return 0
        terminal_since=None;signature=None;announced=None
        while True:
            try:
                output=write_task_report(root)
                manifest=read_json(root/'manifest.json') or {}
                stat=(root/'manifest.json').stat();current=(stat.st_mtime_ns,stat.st_size)
                state=(manifest.get('status'),len(manifest.get('tasks',[])),len(manifest.get('batches',[])))
                if state!=announced:
                    print(json.dumps({'report':str(output),'run_status':state[0],'started_tasks':state[1]},ensure_ascii=False),flush=True)
                    announced=state
                if not args.watch:return 0
                parent_gone=False
                if args.parent_pid:
                    try:os.kill(args.parent_pid,0)
                    except ProcessLookupError:parent_gone=True
                    except PermissionError:pass
                if manifest.get('status') in TERMINAL or parent_gone:
                    if current!=signature or terminal_since is None:terminal_since=time.monotonic()
                    if terminal_since is not None and time.monotonic()-terminal_since>=args.settle_seconds:return 0
                else:terminal_since=None
                signature=current
            except (OSError,ValueError,KeyError,TypeError) as exc:
                print('汇总暂未刷新：'+str(exc),file=sys.stderr,flush=True)
                if not args.watch:return 1
            time.sleep(args.interval)


if __name__=='__main__':
    raise SystemExit(main())
