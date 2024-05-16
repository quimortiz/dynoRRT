import unittest
import pydynorrt as pr
import numpy as np


class TestBasic(unittest.TestCase):

    def test_planners(self):
        """ """
        pr.PlannerBase_Rn()
        pr.PlannerRRT_Rn()
        pr.PlannerRRTStar_Rn()
        pr.PlannerBiRRT_Rn()
        pr.PlannerRRTConnect_Rn()
        pr.PlannerPRM_Rn()
        pr.PlannerLazyPRM_Rn()
        pr.PlannerKinoRRT_Rn()
        pr.PlannerSSTstar_Rn()

    def test_cm(self):
        pr.Collision_manager_pinocchio()
        pr.FrameBounds()
        pr.BallObs2(np.ones(2), 1)
        pr.CM2()
        pr.BallObsX(np.ones(3), 1)
        p = pr.CMX()

    def test_ik(self):
        pr.Pin_ik_solver()

    def test_utils(self):
        pr.srand(0)
        pr.srandtime()
        pr.rand()
        pr.rand01()


if __name__ == "__main__":
    unittest.main()
