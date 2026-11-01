"""Fixed-step flight and optional arm control: python -m examples.demo_controlled --arm."""

import argparse
import json
from pathlib import Path

import numpy as np

import UAMSim as ua


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", action="store_true")
    parser.add_argument("--steps", type=int, default=1200, help="number of 5 ms control updates")
    parser.add_argument("--real-time-factor", type=float, default=0, help="zero disables wall-clock pacing")
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    models = Path(__file__).with_name("models")
    sim = ua.ControlledSimulation(
        ua.Environment(str(models / "environment.xml")),
        ua.Multirotor(str(models / "multirotor.xml")),
        ua.Manipulator(str(models / "Manipulator.xml")) if args.arm else None,
        real_time_factor=args.real_time_factor,
    )
    controller = ua.CascadedPID.from_robot(sim.robot)
    sim.reset(controller)
    reference = np.array([0.2, 0.1, 2.0])
    errors = []
    saturated_steps = 0
    try:
        for _ in range(args.steps):
            joints = None
            if args.arm:
                angle = -0.5 * (1 - np.cos(2*np.pi*sim.state.time/3))
                joints = ua.JointPositionCommand(sim.state.joints.names, [0, angle, 0])
            state = sim.step(ua.ControlTarget(reference, joints=joints))
            errors.append(np.linalg.norm(state.base.position_world - reference))
            saturated_steps += int(controller.saturated)
        print(json.dumps({
            "control_steps": sim.frame,
            "physics_steps": sim.frame * sim.control_steps,
            "simulation_seconds": state.time,
            "with_arm": args.arm,
            "final_position_m": state.base.position_world.tolist(),
            "final_position_error_m": errors[-1],
            "position_rmse_m": float(np.sqrt(np.mean(np.square(errors)))),
            "saturated_control_steps": saturated_steps,
        }, indent=2, allow_nan=False))
    finally:
        sim.close()


if __name__ == "__main__":
    main()
