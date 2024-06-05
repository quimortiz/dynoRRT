# this tests the inverse kinematics solver

import unittest
import pydynorrt as pyrrt
import numpy as np


class TestIK(unittest.TestCase):
    """ """

    def test_1(self):
        """ """
        base_path = pyrrt.DATADIR
        urdf = base_path + "models/iiwa.urdf"
        srdf = base_path + "models/kuka.srdf"

        target = np.array([0.4, 0.0, 0.1])

        q_lim = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1e-5])
        q0 = np.array([0.0, 1.0, 0.0, -1.4, -0.7, 0.0, 0.0])
        lb = q0 - q_lim
        ub = q0 + q_lim

        options = [
            {
                "gradient_descent": False,
                "finite_diff": False,
            },
            {
                "gradient_descent": True,
                "finite_diff": False,
            },
            {
                "gradient_descent": False,
                "finite_diff": True,
            },
            {
                "gradient_descent": True,
                "finite_diff": True,
            },
        ]

        max_solutions = 4

        for option in options:
            solver = pyrrt.Pin_ik_solver()
            solver.set_urdf_filename(urdf)
            solver.set_srdf_filename(srdf)
            solver.build()
            solver.set_p_des([target])
            solver.set_bounds(lb, ub)
            solver.set_max_num_attempts(1000)
            solver.set_frame_name(["contact"])
            solver.set_max_time_ms(10000)
            solver.set_max_solutions(max_solutions)
            solver.set_max_it(1000)
            solver.set_use_gradient_descent(option["gradient_descent"])
            solver.set_use_finite_diff(option["finite_diff"])
            out = solver.solve_ik()
            self.assertEqual(out, pyrrt.IKStatus.SUCCESS)
            ik_solutions = solver.get_ik_solutions()
            self.assertEqual(len(ik_solutions), max_solutions)


if __name__ == "__main__":
    unittest.main()
