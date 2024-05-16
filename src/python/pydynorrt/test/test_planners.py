# this test checks all planners in Python

import unittest
import numpy as np
import pydynorrt as pyrrt


class Test_planners(unittest.TestCase):
    def test_planners(self):

        xlim = [0, 3]
        ylim = [0, 3]
        start = np.array([0.1, 0.1])
        goal = np.array([2.0, 0.2])

        class Obstacle:
            def __init__(self, center: np.ndarray, radius: float):
                self.center = center
                self.radius = radius

        obstacles = [
            Obstacle(center=np.array([1, 0.4]), radius=0.5),
            Obstacle(center=np.array([1, 2]), radius=0.5),
        ]

        def is_collision_free(x: np.ndarray) -> bool:
            """
            x: 2D vector (x, y)

            """
            for obs in obstacles:
                if np.linalg.norm(x - obs.center) < obs.radius:
                    return False
            return True

        options_rrt = """
        [RRT_options]
        max_step = 0.2
        """

        options_rrtstar = """
        [RRTStar_options]
        max_step = 0.2
        max_it = 500
        """

        options_rrtconnect = """
        [RRTConnect_options]
        max_step = 0.2
        """

        options_birrt = """
        [BiRRT_options]
        max_step = 0.2
        """

        options_lazyprm = """
        [LazyPRM_options]
        num_vertices_0 = 20
        k_near = 10
        """

        options_prm = """
        [PRM_options]
        num_vertices_0 = 20
        k_near = 10
        """

        planners = [
            {"class": pyrrt.PlannerRRT_Rn, "options": options_rrt},
            {"class": pyrrt.PlannerRRTStar_Rn, "options": options_rrtstar},
            {"class": pyrrt.PlannerBiRRT_Rn, "options": options_birrt},
            {"class": pyrrt.PlannerRRTConnect_Rn, "options": options_rrtconnect},
            {"class": pyrrt.PlannerPRM_Rn, "options": options_prm},
            {"class": pyrrt.PlannerLazyPRM_Rn, "options": options_lazyprm},
        ]

        for planner in planners:
            p = planner["class"]()
            o = planner["options"]

            p.set_start(start)
            p.set_goal(goal)
            p.init(2)
            p.set_is_collision_free_fun(is_collision_free)
            p.set_bounds_to_state([xlim[0], ylim[0]], [xlim[1], ylim[1]])

            p.read_cfg_string(o)
            out = p.plan()
            path = p.get_path()
            fine_path = p.get_fine_path(0.05)

            if planner["class"] == pyrrt.PlannerRRTStar_Rn:
                self.assertEqual(out, pyrrt.TerminationCondition.MAX_IT_GOAL_REACHED)
            else:
                self.assertEqual(out, pyrrt.TerminationCondition.GOAL_REACHED)


if __name__ == "__main__":
    unittest.main()
