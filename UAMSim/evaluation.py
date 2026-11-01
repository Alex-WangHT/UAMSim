"""Bounded episodes with streaming traces and JSON summaries."""

from dataclasses import asdict, is_dataclass
import json
from pathlib import Path
import platform

import mujoco
import numpy as np

from .tasks import FlightTask, TaskConfig
from .interfaces import JointPositionCommand, RobotCommand, RotorCommand


def json_value(value):
    if is_dataclass(value):
        return json_value(asdict(value))
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(item) for item in value]
    return value


def write_json(path, value):
    Path(path).write_text(json.dumps(json_value(value), indent=2, allow_nan=False), encoding="utf-8")


def run_episode(sim, controller, config: TaskConfig, *, seed=0, output, provenance=None):
    """Run an integer number of control periods, with reset on every episode.

    Input configuration errors raise. Runtime/controller failures become failed
    episode records. Collision and limits are checked at control sample times.
    Existing output directories are never overwritten.
    """
    periods = config.duration / sim.control_dt
    if not np.isclose(periods, round(periods), atol=1e-9, rtol=0):
        raise ValueError("task duration must be an integer number of control periods")
    if config.hold_seconds < sim.control_dt:
        raise ValueError("hold_seconds must be at least one control period")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    manifest = dict(schema_version=1, task=config, seed=seed,
                    controller=f"{type(controller).__module__}.{type(controller).__qualname__}",
                    controller_gains=getattr(controller, "gains", None),
                    physics_dt=float(sim.env.model.opt.timestep), control_dt=sim.control_dt,
                    control_steps=sim.control_steps, real_time_factor=sim.real_time_factor,
                    disturbance=sim.disturbance.config if sim.disturbance is not None else None,
                    python=platform.python_version(), mujoco=mujoco.__version__, numpy=np.__version__,
                    provenance=provenance or {})
    write_json(output / "manifest.json", manifest)
    errors, velocity_errors, joint_errors, tilts, durations, saturation = [], [], [], [], [], []
    status, reason, exception = "failed", "runtime_error", None
    completed = 0
    with (output / "trace.jsonl").open("w", encoding="utf-8") as trace:
        try:
            state = sim.reset(controller)
            task = FlightTask(config, state, seed)
            manifest["resolved_center"] = task.center
            manifest["initial_state"] = state
            write_json(output / "manifest.json", manifest)
            for _ in range(round(periods)):
                command_target = task.target(sim.state.time)
                state = sim.step(command_target)
                completed += 1
                contact = bool(sim.env.get_robot_contacts(sim.robot.prefix))
                status, reason, metrics = task.observe(state, sim.control_dt, contact)
                saturated = getattr(controller, "saturated", None)
                record = dict(frame=sim.frame, state=state, command_target=command_target,
                              reference=task.target(state.time), command=sim.last_command,
                              metrics=metrics, status=status, reason=reason,
                              disturbance_sample=sim.disturbance.last_sample if sim.disturbance is not None else None,
                              saturated=saturated, control_seconds=sim.last_control_seconds)
                trace.write(json.dumps(json_value(record), allow_nan=False) + "\n")
                errors.append(metrics["position_error_m"])
                velocity_errors.append(metrics["velocity_error_m_s"])
                joint_errors.append(metrics["joint_error_rad"])
                tilts.append(metrics["tilt_rad"])
                durations.append(sim.last_control_seconds)
                if saturated is not None:
                    saturation.append(bool(saturated))
                if status != "running":
                    break
        except Exception as exc:
            status, reason = "failed", "runtime_error"
            exception = f"{type(exc).__name__}: {exc}"
    result = dict(schema_version=1, task=config.name, seed=seed, status=status, reason=reason,
                  exception=exception, completed_steps=completed, simulation_seconds=float(sim.env.data.time),
                  position_rmse_m=float(np.sqrt(np.mean(np.square(errors)))) if errors else None,
                  final_position_error_m=errors[-1] if errors else None,
                  max_position_error_m=max(errors) if errors else None,
                  velocity_rmse_m_s=float(np.sqrt(np.mean(np.square(velocity_errors)))) if velocity_errors else None,
                  max_joint_error_rad=max(joint_errors) if joint_errors else None,
                  max_tilt_rad=max(tilts) if tilts else None,
                  saturated_fraction=float(np.mean(saturation)) if saturation else None,
                  mean_control_seconds=float(np.mean(durations)) if durations else None,
                  max_control_seconds=max(durations) if durations else None)
    write_json(output / "summary.json", result)
    return result


def aggregate(results):
    """Aggregate by task; keep failed/time-out episodes in the denominator."""
    grouped = {}
    for row in results:
        grouped.setdefault(row["task"], []).append(row)
    summary = {}
    for name, rows in grouped.items():
        measured = [row["position_rmse_m"] for row in rows if row["position_rmse_m"] is not None]
        summary[name] = dict(episodes=len(rows), succeeded=sum(row["status"] == "succeeded" for row in rows),
                             failed=sum(row["status"] == "failed" for row in rows),
                             timeout=sum(row["status"] == "timeout" for row in rows),
                             success_rate=sum(row["status"] == "succeeded" for row in rows)/len(rows),
                             mean_episode_position_rmse_m=float(np.mean(measured)) if measured else None,
                             episodes_with_metrics=len(measured))
    return summary


def replay_episode(sim, episode, *, atol=1e-9):
    """Replay recorded low-level commands and verify all logged pose/velocity fields.

    Caller supplies the same model snapshot. Require the recorded MuJoCo version
    and timing; mismatches raise instead of being reported as a successful replay.
    """
    if not np.isfinite(atol) or atol < 0:
        raise ValueError("atol must be finite and nonnegative")
    episode = Path(episode)
    manifest = json.loads((episode / "manifest.json").read_text(encoding="utf-8"))
    actual_disturbance = json_value(sim.disturbance.config) if sim.disturbance is not None else None
    if actual_disturbance != manifest.get("disturbance"):
        raise ValueError("replay requires matching disturbance configuration and seed")
    if (manifest["mujoco"] != mujoco.__version__ or manifest["control_steps"] != sim.control_steps
            or not np.isclose(manifest["physics_dt"], sim.env.model.opt.timestep, atol=1e-15, rtol=0)):
        raise ValueError("replay requires matching engine version and simulation timing")
    sim.reset()
    count, max_error = 0, 0.0
    with (episode / "trace.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            raw = record["command"]
            rotors = RotorCommand(raw["rotors"]["thrust"]) if raw["rotors"] is not None else None
            joints = JointPositionCommand(**raw["joints"]) if raw["joints"] is not None else None
            actual = json_value(sim.step_command(RobotCommand(rotors, joints)))
            expected = record["state"]
            if record["frame"] != sim.frame or actual["joints"]["names"] != expected["joints"]["names"]:
                raise ValueError("replay frame sequence or joint mapping mismatch")
            pairs = [(actual["time"], expected["time"])]
            for field in ("linear_velocity_world", "angular_velocity_body"):
                pairs.append((actual[field], expected[field]))
            for part in ("base", "end_effector"):
                if (actual[part] is None) != (expected[part] is None):
                    raise ValueError("replay morphology mismatch")
                if actual[part] is not None:
                    for field in ("position_world", "quaternion_wxyz"):
                        pairs.append((actual[part][field], expected[part][field]))
            for field in ("position", "velocity"):
                pairs.append((actual["joints"][field], expected["joints"][field]))
            for observed, reference in pairs:
                observed, reference = np.asarray(observed), np.asarray(reference)
                if observed.shape != reference.shape or not np.isfinite(reference).all():
                    raise ValueError("invalid replay reference")
                error = float(np.max(np.abs(observed - reference))) if observed.size else 0.0
                max_error = max(max_error, error)
                if error > atol:
                    raise AssertionError(f"replay diverged at frame {sim.frame}: absolute error {error}")
            count += 1
    if not count:
        raise ValueError("episode contains no replayable steps")
    summary = json.loads((episode / "summary.json").read_text(encoding="utf-8"))
    if count != summary["completed_steps"]:
        raise ValueError("trace is incomplete")
    return dict(verified_steps=count, max_absolute_state_error=max_error)
