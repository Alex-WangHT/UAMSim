"""Offline saved-state replay to MP4 and a local browser playback page."""
import argparse
import json
from pathlib import Path

from UAMSim.recording import export_video


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('recording', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--fps', type=int, default=20)
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--speed', type=float, default=1.)
    parser.add_argument('--start', type=float)
    parser.add_argument('--end', type=float)
    parser.add_argument('--camera', default='overview', help='overview or a camera name stored in the model')
    parser.add_argument('--allow-incomplete', action='store_true')
    args = parser.parse_args()
    print(json.dumps(export_video(**vars(args))), flush=True)


if __name__ == '__main__':
    main()
