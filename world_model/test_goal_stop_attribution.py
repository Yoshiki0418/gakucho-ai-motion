from types import SimpleNamespace

import numpy as np

from .goal_stop_attribution import LatchedStopGeometryController


def test_goal_stop_is_triggered_by_actual_xy_distance_and_stays_latched_after_drift():
    body=SimpleNamespace(goal=np.asarray([0.,0.,0.]),pelvis_body_id=0,
        data=SimpleNamespace(qpos=np.asarray([0.,.3,1.]),time=1.,xmat=np.eye(3).reshape(1,9)))
    env=SimpleNamespace(unwrapped=body,scene={'spawn':[0,.3,0],'goals':[{'position':[0,0,0]}],'geoms':[]})
    teacher=LatchedStopGeometryController(env)
    assert teacher.action()[1]==np.float32(-.6)
    assert not teacher.stop_latched
    body.data.qpos[1]=.2;body.data.time=2.
    before=body.data.qpos.copy()
    np.testing.assert_array_equal(teacher.action(),[0.,-1.])
    np.testing.assert_array_equal(before,body.data.qpos)
    assert teacher.stop_latched and teacher.stop_trigger_time==2.
    body.data.qpos[1]=.4;body.data.time=3.
    np.testing.assert_array_equal(teacher.action(),[0.,-1.])
    assert teacher.stop_trigger_time==2.


def test_goal_stop_uses_horizontal_distance_independent_of_pelvis_height():
    body=SimpleNamespace(goal=np.zeros(3),pelvis_body_id=0,
        data=SimpleNamespace(qpos=np.asarray([.1,.1,1.1]),time=.8,xmat=np.eye(3).reshape(1,9)))
    env=SimpleNamespace(unwrapped=body,scene={'spawn':[.1,.1,0],'goals':[{'position':[0,0,0]}],'geoms':[]})
    teacher=LatchedStopGeometryController(env)
    np.testing.assert_array_equal(teacher.action(),[0.,-1.])
    assert teacher.stop_latched
