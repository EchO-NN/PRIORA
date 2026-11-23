"""Measured grasp feedback, using the pinned JustRUA X5 sensor implementation."""
import json

from .justrua_sensors import finger_state
from .justrua_sensors.readings import GRIPPER_GUIDE, render_fingers

VERSION = 'justrua_measured_grasp_v1'


def install():
    result = finger_state.install_before_create()
    if result['status'] != 'available':
        raise RuntimeError('Cannot enable measured finger contacts: ' + str(result))
    return result


def capture(env, *, episode_id, step_id, control_dt):
    """Read only; do not step physics, change commands or read scoring/object state."""
    sim = getattr(env, 'sim', None)
    native = getattr(sim, 'unwrapped', sim)
    before = getattr(native, '_sim_step_counter', None)
    raw = finger_state.sample(env)
    after = getattr(native, '_sim_step_counter', None)
    if after != before:
        raise RuntimeError('Grasp feedback crossed a physics step')
    raw.update(episode_id=episode_id, step_id=step_id,
               simulation_time_s=round(step_id * control_dt, 9), version=VERSION,
               physics_step=None if before is None else int(before),
               capture_extra_physics_steps=0 if before is not None else None)
    raw['text'] = render_fingers(raw)
    # JSON keeps absent readings null, not NaN, fabricated zero or object arrays.
    return json.dumps(raw, ensure_ascii=False, allow_nan=False, separators=(',', ':'))


def teacher_packet(observation):
    value = observation.get('grasp_feedback_json')
    if value is None:
        return dict(status='unavailable', text='Measured gripper feedback: unknown (older simulator).')
    raw = json.loads(str(value))
    return {key: raw[key] for key in ('version', 'status', 'episode_id', 'step_id',
                                     'simulation_time_s', 'text')}
