"""Self-contained simulation recording and offline MP4 rendering.

Capture is cheap: save integration-state snapshots, not rendered images, while
PX4 runs. Replay restores saved states without running controllers or dynamics.
"""

import hashlib
import html
import json
import math
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading
from urllib.parse import quote

import mujoco
import numpy as np


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


class StateRecorder:
    def __init__(self, model, directory):
        self.model = model
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.owner = threading.get_ident()
        self.spec = int(mujoco.mjtState.mjSTATE_INTEGRATION)
        self.size = mujoco.mj_stateSize(model, self.spec)
        self.row = np.empty(self.size+1, dtype='<f8')
        self.count, self.first_time, self.last_time = 0, None, None
        self.closed = False
        model_path = self.directory/'model.mjb'
        mujoco.mj_saveModel(model, str(model_path))
        self.model_hash = _sha256(model_path)
        self.state_hash = hashlib.sha256()
        self.stream = (self.directory/'states.bin').open('xb')
        self._manifest(False)

    def _manifest(self, completed):
        manifest = dict(format='uamsim-state-v1', mujoco_version=mujoco.__version__,
                        state_spec=self.spec, state_size=self.size, row_width=self.size+1,
                        dtype='<f8', rows=self.count, start_time=self.first_time, end_time=self.last_time,
                        model_sha256=self.model_hash, completed=bool(completed),
                        states_sha256=self.state_hash.hexdigest(),
                        physics_timestep=float(self.model.opt.timestep),
                        replay='saved-state sample-and-hold; no physics re-simulation')
        (self.directory/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')

    def capture(self, data):
        if self.closed or threading.get_ident() != self.owner:
            raise RuntimeError('capture requires the live recorder physics owner')
        if not math.isfinite(data.time) or (self.last_time is not None and data.time <= self.last_time):
            raise ValueError('recording timestamps must strictly increase; use a new recording after reset')
        self.row[0] = data.time
        mujoco.mj_getState(self.model, data, self.row[1:], self.spec)
        if not np.isfinite(self.row).all():
            raise ValueError('cannot record non-finite physics state')
        payload = self.row.tobytes()
        self.stream.write(payload)
        self.state_hash.update(payload)
        self.count += 1
        self.first_time = float(data.time) if self.first_time is None else self.first_time
        self.last_time = float(data.time)

    def close(self, *, completed=False):
        if threading.get_ident() != self.owner:
            raise RuntimeError('close recorder on its owner thread')
        if not self.closed:
            self.stream.close()
            self.closed = True
            self._manifest(completed)


class StateReplay:
    def __init__(self, directory, *, allow_incomplete=False):
        self.directory = Path(directory)
        meta = json.loads((self.directory/'manifest.json').read_text(encoding='utf-8'))
        if meta.get('format') != 'uamsim-state-v1' or meta.get('dtype') != '<f8':
            raise ValueError('unsupported recording format')
        if meta.get('mujoco_version') != mujoco.__version__:
            raise ValueError('binary model replay requires the recorded MuJoCo version')
        if not meta.get('completed') and not allow_incomplete:
            raise ValueError('recording did not finish; explicitly allow incomplete replay if intended')
        model_path = self.directory/'model.mjb'
        if _sha256(model_path) != meta['model_sha256']:
            raise ValueError('recorded model checksum mismatch')
        self.model = mujoco.MjModel.from_binary_path(str(model_path))
        self.data = mujoco.MjData(self.model)
        self.spec = int(mujoco.mjtState.mjSTATE_INTEGRATION)
        size = mujoco.mj_stateSize(self.model, self.spec)
        rows = meta['rows']
        if (type(rows) is not int or rows < 1 or meta['state_spec'] != self.spec or
                meta['state_size'] != size or meta['row_width'] != size+1):
            raise ValueError('recording dimensions do not match the saved model')
        if (self.directory/'states.bin').stat().st_size != rows*(size+1)*8:
            raise ValueError('truncated or oversized recording')
        if _sha256(self.directory/'states.bin') != meta['states_sha256']:
            raise ValueError('recorded state checksum mismatch')
        self.states = np.memmap(self.directory/'states.bin', dtype='<f8', mode='r', shape=(rows, size+1))
        self.times = self.states[:, 0]
        if not np.isfinite(self.states).all() or np.any(np.diff(self.times) <= 0):
            raise ValueError('invalid state values or timestamps')
        self.start_time, self.end_time = float(self.times[0]), float(self.times[-1])
        self.seek(self.start_time)

    def seek(self, timestamp):
        if self.states is None:
            raise RuntimeError('replay is closed')
        if not math.isfinite(timestamp) or not self.start_time <= timestamp <= self.end_time:
            raise ValueError('requested time is outside the recording')
        index = int(np.searchsorted(self.times, timestamp, side='right')-1)
        mujoco.mj_setState(self.model, self.data, self.states[index, 1:], self.spec)
        mujoco.mj_forward(self.model, self.data)
        return float(self.times[index])

    def close(self):
        if self.states is not None:
            self.states._mmap.close()
            self.times = self.states = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def export_video(recording, output, *, fps=20, width=640, height=480, speed=1., start=None, end=None,
                 camera='overview', allow_incomplete=False):
    """Render a clip on its simulation-time axis, with no PX4/ROS/network needed."""
    from PIL import Image, ImageDraw

    if (type(fps) is not int or not 1 <= fps <= 120 or
            any(type(v) is not int or v < 32 or v > 3840 or v % 2 for v in (width, height)) or
            not math.isfinite(speed) or speed <= 0):
        raise ValueError('invalid frame rate, even resolution or playback speed')
    executable = shutil.which('ffmpeg')
    if executable is None:
        raise RuntimeError('ffmpeg is required to export MP4')
    output = Path(output)
    if output.suffix.lower() != '.mp4':
        raise ValueError('video output must use .mp4')
    partial = output.with_name(output.stem+'.partial.mp4')
    player = output.with_suffix('.html')
    if output.exists() or partial.exists() or player.exists():
        raise FileExistsError('video/player output already exists; choose another name')
    output.parent.mkdir(parents=True, exist_ok=True)
    replay = StateReplay(recording, allow_incomplete=allow_incomplete)
    start = replay.start_time if start is None else start
    end = replay.end_time if end is None else end
    if not np.isfinite([start, end]).all() or not replay.start_time <= start < end <= replay.end_time:
        replay.close()
        raise ValueError('clip bounds must lie within the recording and have positive duration')
    replay.model.vis.global_.offwidth = max(width, replay.model.vis.global_.offwidth)
    replay.model.vis.global_.offheight = max(height, replay.model.vis.global_.offheight)
    if camera == 'overview':
        view = mujoco.MjvCamera()
        view.type = mujoco.mjtCamera.mjCAMERA_FREE
        view.lookat[:] = [0., 0., 1.]
        view.distance, view.azimuth, view.elevation = 4.5, 135., -25.
    else:
        if mujoco.mj_name2id(replay.model, mujoco.mjtObj.mjOBJ_CAMERA, camera) < 0:
            replay.close()
            raise ValueError(f'unknown recorded camera: {camera}')
        view = camera
    frames = max(1, math.ceil((end-start)*fps/speed-1e-6))
    try:
        renderer = mujoco.Renderer(replay.model, height=height, width=width)
    except Exception:
        replay.close()
        raise
    process = None
    try:
        with tempfile.TemporaryFile() as errors:
            process = subprocess.Popen([executable, '-v', 'error', '-n', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
                '-s', f'{width}x{height}', '-r', str(fps), '-i', 'pipe:0', '-an', '-c:v', 'libx264',
                '-threads', '4', '-preset', 'fast', '-crf', '20', '-pix_fmt', 'yuv420p',
                '-movflags', '+faststart', str(partial)], stdin=subprocess.PIPE, stderr=errors)
            try:
                for frame in range(frames):
                    requested = min(start+frame*speed/fps, end)
                    sampled = replay.seek(requested)
                    renderer.update_scene(replay.data, camera=view)
                    rgb = Image.fromarray(renderer.render())
                    draw = ImageDraw.Draw(rgb)
                    draw.rectangle((0, 0, min(width, 300), 25), fill=(15, 20, 28))
                    draw.text((8, 7), f'Sim {sampled:8.3f} s    Playback {speed:g}x', fill='white')
                    process.stdin.write(rgb.tobytes())
                process.stdin.close()
                code = process.wait(timeout=60)
                if code:
                    errors.seek(0)
                    raise RuntimeError(f'ffmpeg failed ({code}): {errors.read().decode(errors="replace")}')
            except Exception:
                if process.poll() is None:
                    process.kill()
                    process.wait()
                try:
                    process.stdin.close()
                except OSError:
                    pass
                raise
        partial.replace(output)
        player.write_text(f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<title>UAM-Sim 录像回放</title><style>body{{background:#111827;color:#eee;font-family:system-ui;
max-width:1000px;margin:30px auto;padding:16px}}video{{width:100%;background:#000}}button{{margin:8px;padding:8px 16px}}</style>
<h1>UAM-Sim 录像回放</h1><p>仿真时间 {start:.3f}–{end:.3f} 秒 · {fps} fps · 导出速度 {speed:g}×</p>
<video id="video" controls preload="metadata" src="{html.escape(quote(output.name))}"></video>
<p><button onclick="video.playbackRate=.5">0.5×</button><button onclick="video.playbackRate=1">1×</button>
<button onclick="video.playbackRate=2">2×</button><button onclick="video.currentTime=0">回到开头</button></p>
<p>使用播放器拖动时间轴、暂停或全屏。状态回放可重新选择视角、截取片段和导出速度。</p></html>''', encoding='utf-8')
        return dict(video=str(output), player=str(player), frames=frames, fps=fps, duration=frames/fps,
                    simulation_start=start, simulation_end=end, speed=speed, width=width, height=height)
    finally:
        renderer.close()
        replay.close()
