"""Run reproducible task episodes: python -m examples.benchmark --seeds 0 1 2."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import uuid

import UAMSim as ua
from UAMSim.evaluation import aggregate, replay_episode, run_episode, write_json
from UAMSim.tasks import TaskConfig
from UAMSim.disturbances import DisturbanceConfig, EndEffectorWrench, WindConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", choices=("hover", "tracking", "arm_motion"),
                        default=["hover", "tracking", "arm_motion"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--duration", type=float, default=6.0)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--replay", type=Path, help="verify an episode directory using its batch's model snapshot")
    parser.add_argument("--disturbances", nargs="+", choices=("none", "wind", "wrench", "combined"), default=["none"])
    parser.add_argument("--controllers", nargs="+", choices=("pid", "geometric"), default=["pid"])
    args = parser.parse_args()
    if args.replay:
        manifest = json.loads((args.replay / "manifest.json").read_text(encoding="utf-8"))
        models = args.replay.parent / "models"
        for name, expected_hash in manifest["provenance"]["model_sha256"].items():
            if hashlib.sha256((models / name).read_bytes()).hexdigest() != expected_hash:
                raise ValueError(f"model snapshot changed: {name}")
        sim = ua.ControlledSimulation(
            ua.Environment(str(models / "environment.xml")), ua.Multirotor(str(models / "multirotor.xml")),
            ua.Manipulator(str(models / "Manipulator.xml")), control_steps=manifest["control_steps"],
            disturbance=DisturbanceConfig.from_dict(manifest["disturbance"]) if manifest.get("disturbance") else None,
        )
        try:
            print(json.dumps(replay_episode(sim, args.replay), indent=2, allow_nan=False))
        finally:
            sim.close()
        return
    if any(seed < 0 for seed in args.seeds):
        parser.error("seeds must be nonnegative")
    configs = [TaskConfig(name=name, duration=args.duration) for name in args.tasks]
    root = Path(__file__).resolve().parents[1]
    output = args.output or root / "logs" / f"benchmark-{uuid.uuid4().hex[:12]}"
    output.mkdir(parents=True, exist_ok=False)
    model_dir = output / "models"
    model_dir.mkdir()
    model_hashes = {}
    for filename in ("environment.xml", "multirotor.xml", "Manipulator.xml"):
        source = Path(__file__).with_name("models") / filename
        destination = model_dir / filename
        shutil.copyfile(source, destination)
        model_hashes[filename] = hashlib.sha256(destination.read_bytes()).hexdigest()
    source_hashes = {}
    for folder in ("UAMSim", "examples"):
        for source in sorted((root / folder).rglob("*.py")):
            source_hashes[source.relative_to(root).as_posix()] = hashlib.sha256(source.read_bytes()).hexdigest()
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True))
    except (OSError, subprocess.CalledProcessError):
        revision, dirty = None, None
    provenance = dict(git_revision=revision, git_dirty=dirty, source_sha256=source_hashes, model_sha256=model_hashes)
    write_json(output / "provenance.json", provenance)
    sim = ua.ControlledSimulation(
        ua.Environment(str(model_dir / "environment.xml")),
        ua.Multirotor(str(model_dir / "multirotor.xml")),
        ua.Manipulator(str(model_dir / "Manipulator.xml")),
    )
    results = []
    try:
        for algorithm in args.controllers:
            sim.reset()
            controller_type = ua.CascadedPID if algorithm == "pid" else ua.GeometricController
            controller = controller_type.from_robot(sim.robot)
            for scenario in args.disturbances:
                for config in configs:
                    for seed in args.seeds:
                        disturbance = None if scenario == "none" else DisturbanceConfig(
                            wind=WindConfig(velocity_world=(1, 0, 0), gust_amplitude=(0.3, 0.2, 0.0))
                            if scenario in ("wind", "combined") else None,
                            end_effector=EndEffectorWrench(force=(0.5, 0, 0), start=1, end=3)
                            if scenario in ("wrench", "combined") else None, seed=seed,
                        )
                        sim.configure_disturbance(disturbance)
                        episode = output / f"{len(results):03d}-{algorithm}-{scenario}-{config.name}-seed-{seed}"
                        result = run_episode(sim, controller, config, seed=seed, output=episode, provenance=provenance)
                        result.update(scenario=scenario, algorithm=algorithm)
                        results.append(result)
                        print(f"{algorithm}/{scenario}/{config.name} seed={seed}: {result['status']} ({result['reason']})")
        summary = {algorithm: {scenario: aggregate([row for row in results
                   if row["scenario"] == scenario and row["algorithm"] == algorithm])
                   for scenario in args.disturbances} for algorithm in args.controllers}
        write_json(output / "summary.json", summary)
        print(json.dumps(summary, indent=2, allow_nan=False))
        print(f"Results: {output.resolve()}")
    finally:
        sim.close()


if __name__ == "__main__":
    main()
