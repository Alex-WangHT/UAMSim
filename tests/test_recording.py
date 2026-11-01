import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

import numpy as np

import UAMSim as ua
from UAMSim.recording import StateRecorder, StateReplay, export_video
from test_controlled_simulation import runner


def record(directory, steps=100):
    sim = runner(True)
    recorder = StateRecorder(sim.env.model, directory)
    recorder.capture(sim.env.data)
    snapshots = [(sim.env.data.time, sim.env.data.qpos.copy(), sim.env.data.qvel.copy(), sim.env.data.ctrl.copy())]
    for _ in range(steps):
        sim.step_command(ua.RobotCommand(ua.RotorCommand(np.full(6, sim.robot.hover_thrust()))))
        recorder.capture(sim.env.data)
        snapshots.append((sim.env.data.time, sim.env.data.qpos.copy(), sim.env.data.qvel.copy(), sim.env.data.ctrl.copy()))
    recorder.close(completed=True)
    return snapshots


class RecordingTests(unittest.TestCase):
    def test_exact_state_seek_and_original_model(self):
        with tempfile.TemporaryDirectory() as root:
            folder = Path(root)/'episode'
            states = record(folder, 10)
            replay = StateReplay(folder)
            for time, qpos, qvel, ctrl in states[::3]:
                self.assertEqual(replay.seek(time), time)
                np.testing.assert_array_equal(replay.data.qpos, qpos)
                np.testing.assert_array_equal(replay.data.qvel, qvel)
                np.testing.assert_array_equal(replay.data.ctrl, ctrl)
            self.assertEqual(replay.seek(states[2][0]+.001), states[2][0])
            with self.assertRaises(ValueError):
                replay.seek(-1)
            replay.close()

    def test_truncated_state_or_wrong_model_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            folder = Path(root)/'episode'
            record(folder, 2)
            model = folder/'model.mjb'
            contents = model.read_bytes()
            model.write_bytes(contents+b'bad')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                StateReplay(folder)
            model.write_bytes(contents)
            states = folder/'states.bin'
            states.write_bytes(states.read_bytes()[:-1])
            with self.assertRaisesRegex(ValueError, 'truncated'):
                StateReplay(folder)

    def test_reset_and_incomplete_recording_are_explicit(self):
        with tempfile.TemporaryDirectory() as root:
            folder = Path(root)/'episode'
            sim = runner()
            recorder = StateRecorder(sim.env.model, folder)
            recorder.capture(sim.env.data)
            with self.assertRaises(ValueError):
                recorder.capture(sim.env.data)
            recorder.close()
            with self.assertRaisesRegex(ValueError, 'did not finish'):
                StateReplay(folder)
            with StateReplay(folder, allow_incomplete=True) as replay:
                self.assertEqual(replay.start_time, 0.)

    @unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'), 'video tools required')
    def test_mp4_frame_count_duration_and_playback_page(self):
        with tempfile.TemporaryDirectory() as root:
            folder, output = Path(root)/'episode', Path(root)/'video.mp4'
            record(folder)
            result = export_video(folder, output, fps=10, width=64, height=48, speed=.5, start=.1, end=.3)
            info = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                             '-show_entries', 'stream=codec_name,width,height,nb_frames,duration', '-of', 'json', str(output)]))['streams'][0]
            self.assertEqual(info['codec_name'], 'h264')
            self.assertEqual(int(info['nb_frames']), 4)
            self.assertAlmostEqual(float(info['duration']), .4)
            self.assertEqual((info['width'], info['height']), (64, 48))
            self.assertEqual(result['frames'], 4)
            self.assertIn('controls', output.with_suffix('.html').read_text())
            with self.assertRaises(FileExistsError):
                export_video(folder, output)
