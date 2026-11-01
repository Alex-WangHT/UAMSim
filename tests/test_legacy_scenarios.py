"""Make the original integration scripts discoverable by unittest and pytest."""

from pathlib import Path
import runpy
import unittest


class LegacyScenarios(unittest.TestCase):
    def run_scenario(self, filename):
        runpy.run_path(str(Path(__file__).with_name(filename)), run_name="__main__")

    def test_environment(self):
        self.run_scenario("test_environment.py")

    def test_robot(self):
        self.run_scenario("test_robot.py")

    def test_threaded_simulation(self):
        self.run_scenario("test_simulation.py")
