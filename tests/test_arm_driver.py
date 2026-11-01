import unittest


class SimulationDriverTests(unittest.TestCase):
    def test_simulation_driver_uses_same_named_position_contract(self):
        from test_controlled_simulation import runner
        from UAMSim.simulation.arm_driver import SimulationArmDriver
        sim = runner(True)
        arm = SimulationArmDriver(sim.robot)
        targets = arm.read_positions()
        targets[sim.robot.manipulator.joint_names[1]] = -.5
        arm.write_positions(targets)
        actuator = sim.robot.manipulator.servos[sim.robot.manipulator.joint_names[1]]
        self.assertEqual(sim.env.data.ctrl[actuator], -.5)
        arm.stop()
        self.assertEqual(sim.env.data.ctrl[actuator], arm.read_positions()[sim.robot.manipulator.joint_names[1]])
