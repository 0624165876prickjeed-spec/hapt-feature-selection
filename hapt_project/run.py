import argparse
import json
import os
import sys
from pathlib import Path


def main():
    sys.dont_write_bytecode = True
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['prepare', 'check', 'smoke', 'train', 'evaluate', 'plot', 'all'])
    parser.add_argument('--config', type=Path, default=Path(__file__).parent / 'configs' / 'formal.json')
    parser.add_argument('--workspace', type=Path, default=Path(__file__).resolve().parents[2])
    arguments = parser.parse_args()
    layer = json.loads(arguments.config.read_text(encoding='utf-8'))
    configuration = json.loads((arguments.config.parent / layer.pop('base_config')).read_text(encoding='utf-8'))
    configuration.update(layer)
    workspace = arguments.workspace.resolve()
    os.environ['MPLCONFIGDIR'] = str(workspace / configuration['resources']['matplotlib_config_dir'])
    for name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
        os.environ[name] = str(configuration['resources']['numeric_threads'])
    from hapt.experiment import execute
    execute(arguments.command, configuration, workspace, Path(__file__).resolve().parent)


if __name__ == '__main__':
    main()
