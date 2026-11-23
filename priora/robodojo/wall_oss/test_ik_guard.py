from pathlib import Path
import json
import numpy as np
import pytest
from .ik_guard import JointPathGuard

import os
URDF = Path(os.getenv('ROBODOJO_X5_URDF', 'Assets/Robots/x5/X5A.urdf'))
@pytest.fixture
def guard():
    if not URDF.exists():pytest.skip('X5 asset not installed')
    return JointPathGuard(URDF,['joint'+str(i) for i in range(1,7)])

def q(elbow):
    value=np.zeros(6);value[2]=elbow;return value

def test_known_penetration_and_home(guard):
    for value in [0,-.003,3.047,3.852]:
        assert guard.check_path(q(value),q(value))['passed']
    for value in [5.112,5.469,5.878]:
        assert not guard.check_path(q(value),q(value))['passed']
    # Endpoints clear; crossing the forbidden region is not.
    assert not guard.check_path(q(3.8),q(6.3))['passed']

def test_wrap_continuity_rate_limit_and_fallback(guard):
    current=q(2.)
    target=q(2.04);target[0]=2*np.pi+.02
    selected,info=guard.select(current,[target],[True],.16)
    assert np.allclose(selected,[.02,0,2.04,0,0,0])
    assert not info['blocked'] and not info['rate_limited']
    selected,info=guard.select(q(0),[q(3.85)],[True],.16)
    assert selected[2]>0 and np.abs(selected).max()<=.16+1e-9
    selected,info=guard.select(q(3.85),[q(5.469)],[True],.16)
    assert info['blocked'] and np.array_equal(selected,q(3.85))
    selected,info=guard.select(q(3.85),[q(5.469),q(3.9)],[True,True],.16)
    assert not info['blocked'] and np.allclose(selected,q(3.9))

def test_failure_nonfinite_and_joint_limits(guard):
    result,info=guard.select(q(1),[q(2)],[False],.16)
    assert info['blocked'] and np.array_equal(result,q(1))
    bad=q(1);bad[5]=np.nan
    with pytest.raises(ValueError):guard.normalize(bad,q(1))
    bad=q(1);bad[5]=10
    assert not guard.check_path(q(1),bad)['passed']


def test_collision_hold_interrupt_is_reported(tmp_path,monkeypatch):
    from .test_wall import Student,WallSim
    from .client import WallTools
    from ..test_support import invoke,response
    class GuardedSim(WallSim):
        def request(self,op,**args):
            if op=='wall_eef_step':
                result=super().request(op,**args)
                result['motion_guard']=dict(blocked=True,reason='mesh_interference')
                return result
            return super().request(op,**args)
    monkeypatch.setenv('ROLLOUT_MAIN_CAMERA_INPUT','rgb')
    monkeypatch.setenv('WALL_MIN_FREE_BYTES','0')
    r=WallTools(tmp_path,'arrange_largest_number',Student(),rpc_factory=GuardedSim,max_decisions=0)
    invoke(r);invoke(r);packet=invoke(r,response=response(r.request))
    assert packet['previous_result']['motion_guard']['blocked']
    assert json.loads((tmp_path/'history.json').read_text())[0]['motion_guard']['blocked']
