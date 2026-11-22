import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys


def run(arguments, directory):
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
    subprocess.run([sys.executable, *map(str, arguments)], cwd=directory, env=environment, check=True)


def main():
    parser = argparse.ArgumentParser(description='课程实验复现入口，结果写入work，不覆盖交付结果')
    parser.add_argument('stage', choices=['audit', 'smoke', 'verify-main', 'formal', 'rebuild-all'])
    arguments = parser.parse_args()
    root = Path(__file__).resolve().parent
    if arguments.stage == 'verify-main':
        run([root / 'verify_main.py'], root)
        return
    if arguments.stage == 'audit':
        run([root / 'outputs/hapt_exploration_v2/audit_results.py'], root)
        print('保存预测的指标复算通过')
        return
    target = root / 'work' / ('reproduce_' + arguments.stage)
    if target.exists():
        raise FileExistsError(f'请保留已有结果；需要重新执行时将该目录移走或重命名：{target}')
    target.mkdir(parents=True)
    for project in ['hapt_project', 'hapt_exploration_v2']:
        exclude_results = project == 'hapt_exploration_v2' or arguments.stage == 'rebuild-all'
        ignored = ['__pycache__', '*.pyc', 'DELIVERY_MANIFEST.json']
        if exclude_results:
            ignored.append('results')
        shutil.copytree(root / 'outputs' / project, target / 'outputs' / project,
                        ignore=shutil.ignore_patterns(*ignored))
    archive = root / 'work/hapt-341.zip'
    if archive.exists():
        (target / 'work').mkdir(exist_ok=True)
        shutil.copy2(archive, target / 'work/hapt-341.zip')
    original = target / 'outputs/hapt_project/run.py'
    exploration = target / 'outputs/hapt_exploration_v2/run_exploration.py'
    if arguments.stage == 'rebuild-all':
        run([original, 'all'], target)
    else:
        run([original, 'check'], target)
    run([exploration, 'smoke'], target)
    if arguments.stage != 'smoke':
        run([exploration, 'formal'], target)
        run([exploration.parent / 'audit_results.py'], target)
    print(f'复现完成：{target}')


if __name__ == '__main__':
    main()
