"""Run configured local services/checks, or compare summaries of the same case."""
import argparse
import json
from pathlib import Path

from UAMSim.experiments import compare_runs, run_experiment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--config', type=Path)
    mode.add_argument('--compare', nargs='+', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.config:
        result = run_experiment(json.loads(args.config.read_text(encoding='utf-8')),
                                args.output, config_dir=args.config.parent)
    else:
        result = compare_runs(args.compare)
        with args.output.open('x', encoding='utf-8') as target:
            json.dump(result, target, indent=2, allow_nan=False)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
