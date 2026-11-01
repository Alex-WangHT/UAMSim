"""Configurable local experiment runner, independent of flight and ROS projects.

Commands are argv arrays, never shell strings. Checks own their acceptance criteria;
this runner records their results and treats nonzero exits as failures.
"""

import json
import math
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import time


def validate_config(config):
    if config.get('version') != 1 or not config.get('case_id') or not config.get('label'):
        raise ValueError('version=1, case_id and label are required')
    if not config.get('steps'):
        raise ValueError('at least one step is required')
    names, services = set(), set()
    for step in config['steps']:
        name, action = step.get('name', ''), step.get('action')
        timeout = step.get('timeout', 60)
        if not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError('step timeout must be finite and positive')
        if action == 'wait':
            if name not in services:
                raise ValueError('wait must refer to a started service')
            services.remove(name)
            continue
        if action not in ('start', 'check') or not re.fullmatch(r'[A-Za-z0-9_-]+', name) or name in names:
            raise ValueError('steps need unique safe names and start/check/wait actions')
        command = step.get('command')
        if not isinstance(command, list) or not command or not all(isinstance(x, str) and x for x in command):
            raise ValueError('command must be a nonempty argv array of strings')
        names.add(name)
        if action == 'start':
            services.add(name)


def run_experiment(config, output, *, framework=None, config_dir=None):
    validate_config(config)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    root = Path(framework or Path(__file__).resolve().parents[1]).resolve()
    variables = {'framework': str(root), 'output': str(output), 'python': sys.executable,
                 'config': str(Path(config_dir or root).resolve())}

    def expand(value):
        for key, replacement in variables.items():
            value = value.replace('${'+key+'}', replacement)
        return value

    env = os.environ.copy()
    env.update({k: expand(v) for k, v in config.get('env', {}).items()})
    processes, handles, services = {}, [], set()
    namespace = os.readlink('/proc/self/ns/net') if sys.platform == 'linux' else None
    manifest = dict(config=config, hostname=socket.gethostname(), net_namespace=namespace, processes=[])
    result = dict(version=1, case_id=config['case_id'], label=config['label'], passed=False,
                  checks={}, metrics={}, artifacts=config.get('artifacts', {}))
    start = time.monotonic()

    def save_manifest():
        (output/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')

    def health(exclude=None):
        for name in services - {exclude}:
            if processes[name].poll() is not None:
                raise RuntimeError(f'service {name} exited unexpectedly ({processes[name].returncode})')

    def wait_until(process, name, timeout, ready=None):
        deadline = time.monotonic()+timeout
        while time.monotonic() < deadline:
            health(exclude=name)
            code = process.poll()
            if ready is None and code is not None:
                if code:
                    raise RuntimeError(f'{name} failed ({code}); inspect {name}.log')
                return
            if code is not None:
                raise RuntimeError(f'{name} exited before readiness ({code})')
            if ready is not None and ready in (output/f'{name}.log').read_text(encoding='utf-8', errors='replace'):
                return
            time.sleep(.05)
        raise TimeoutError(f'{name} timed out after {timeout}s')

    save_manifest()
    try:
        for step in config['steps']:
            name, action = step['name'], step['action']
            timeout = step.get('timeout', 60)
            health(exclude=name if action == 'wait' else None)
            if action == 'wait':
                wait_until(processes[name], name, timeout)
                services.remove(name)
                continue
            handle = (output/f'{name}.log').open('w', encoding='utf-8')
            handles.append(handle)
            command = [expand(arg) for arg in step['command']]
            cwd = expand(step.get('cwd', '${framework}'))
            process = subprocess.Popen(command, cwd=cwd, env=env | {k: expand(v) for k, v in step.get('env', {}).items()},
                                       stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
            processes[name] = process
            child_ns = os.readlink(f'/proc/{process.pid}/ns/net') if namespace else None
            manifest['processes'].append(dict(name=name, pid=process.pid, command=command, cwd=cwd,
                                              net_namespace=child_ns))
            save_manifest()
            if namespace and child_ns != namespace:
                raise RuntimeError('experiment children must share the local network namespace')
            if action == 'start':
                services.add(name)
                if step.get('ready_text'):
                    wait_until(process, name, timeout, step['ready_text'])
            else:
                wait_until(process, name, timeout)
                rows = []
                for line in (output/f'{name}.log').read_text(encoding='utf-8', errors='replace').splitlines():
                    try:
                        row = json.loads(line)
                    except ValueError:
                        continue
                    if isinstance(row, dict):
                        rows.append(row)
                result['checks'][name] = dict(passed=True, records=rows)
        health()
        for metric, source in config.get('metrics', {}).items():
            rows = result['checks'][source['check']]['records']
            values = [row[source['key']] for row in rows if source['key'] in row]
            if not values or isinstance(values[-1], bool) or not isinstance(values[-1], (int, float)) or not math.isfinite(values[-1]):
                raise ValueError(f'missing or nonfinite metric: {metric}')
            result['metrics'][metric] = values[-1]
        result['passed'] = True
    except Exception as error:
        result['error'] = str(error)
        raise
    finally:
        for process in reversed(list(processes.values())):
            # A command can leave descendants behind after its leader exits.
            try:
                if os.name == 'posix':
                    os.killpg(process.pid, signal.SIGINT)
                elif process.poll() is None:
                    process.terminate()
            except ProcessLookupError:
                pass
        deadline = time.monotonic()+5
        while any(p.poll() is None for p in processes.values()) and time.monotonic() < deadline:
            time.sleep(.05)
        for process in processes.values():
            try:
                if os.name == 'posix':
                    os.killpg(process.pid, signal.SIGKILL)
                elif process.poll() is None:
                    process.kill()
            except ProcessLookupError:
                pass
            process.wait()
        for handle in handles:
            handle.close()
        result['wall_seconds'] = time.monotonic()-start
        (output/'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    return result


def compare_runs(paths):
    """Compare labeled runs of one explicitly shared case; keep failed runs visible."""
    runs = [json.loads(Path(path).read_text(encoding='utf-8')) for path in paths]
    if not runs or len({run['case_id'] for run in runs}) != 1:
        raise ValueError('comparison requires runs with the same case_id')
    if len({run['label'] for run in runs}) != len(runs):
        raise ValueError('comparison labels must be unique')
    return dict(case_id=runs[0]['case_id'], runs=[dict(label=run['label'], passed=run['passed'],
                 metrics=run['metrics'], error=run.get('error')) for run in runs])
