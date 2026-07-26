"""ClutchRetargeter: releasing the clutch must not leave a motion tail."""

import numpy as np
import pytest

from so_snake.config import SoSnakeConfig
from so_snake.m3_safety.ik5d import TaskIK5D
from so_snake.m3_safety.projection import OrientationProjector
from so_snake.teleop.clutch import ClutchRetargeter
from so_snake.teleop.sources import NintendoProSample


def _sample(clutch: bool, lx: float = 0.0, ly: float = 0.0) -> NintendoProSample:
    return NintendoProSample(
        t=0.0,
        left_stick=np.array([lx, ly]),
        right_stick=np.zeros(2),
        imu_quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
        clutch=clutch,
    )


@pytest.fixture
def retargeter():
    cfg = SoSnakeConfig()
    ik = TaskIK5D(arm=cfg.arm, teleop=cfg.teleop, ik=cfg.ik)
    proj = OrientationProjector(ik.chain, ik.frame)
    measured = ik.task_pose(np.array(cfg.teleop.home_joints_deg, float)).pose
    return ClutchRetargeter(projector=proj, teleop=cfg.teleop), measured


def test_release_snaps_target_to_measured(retargeter):
    rt, measured = retargeter
    # Hold the clutch and drive the stick: the target integrates open-loop and
    # leads the (fixed) measured pose -- the lead that would become the tail.
    for _ in range(10):
        rt.update(_sample(True, ly=1.0), measured)
    assert rt.target.x - measured.x > 0.02  # a real lead accumulated

    # Release: the frozen target must snap back to where the arm actually is,
    # so the arm stops instead of coasting into the leading target.
    rt.update(_sample(False), measured)
    assert np.allclose(rt.target.as_array(), measured.as_array(), atol=1e-9)


def test_release_without_engage_leaves_target_at_measured(retargeter):
    rt, measured = retargeter
    # Never engaged: still well-defined, and no snap side effects.
    rt.update(_sample(False, ly=1.0), measured)
    assert np.allclose(rt.target.as_array(), measured.as_array(), atol=1e-9)
