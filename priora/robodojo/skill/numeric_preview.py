"""Compact robot numbers/orientations in GPT-visible copies, not execution data."""
from scipy.spatial.transform import Rotation

ROBOT_FIELDS = frozenset(('current_state', 'current_eef', 'student_eef_trajectory'))
DECIMAL_PLACES = 3


def robot_numeric_preview(value, *, robot_field=False):
    """Also handle robot fields nested in previous observations/tool results.

    Preserve identifiers, counters, flags, camera calibration, tolerances and
    other numeric data. Full-precision packets/NPZ remain the control authority.
    """
    if isinstance(value, dict):
        if robot_field and 'quaternion_wxyz' in value:
            # Current packets already supply Euler angles. For older replies,
            # derive them from full precision using the controller's convention
            # before rounding; never discard the only orientation description.
            rpy = value.get('rpy_deg')
            if rpy is None:
                w, x, y, z = value['quaternion_wxyz']
                rpy = Rotation.from_quat([x, y, z, w]).as_euler('xyz', degrees=True).tolist()
            value = {key: item for key, item in value.items() if key != 'quaternion_wxyz'}
            value['rpy_deg'] = rpy
        return {key: robot_numeric_preview(item,
                    robot_field=robot_field or key in ROBOT_FIELDS)
                for key, item in value.items()}
    if isinstance(value, list):
        return [robot_numeric_preview(item, robot_field=robot_field) for item in value]
    if robot_field and isinstance(value, float):
        rounded = round(value, DECIMAL_PLACES)
        return 0.0 if rounded == 0 else rounded
    return value
