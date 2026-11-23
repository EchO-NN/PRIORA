"""X5 distal finger landmarks from robot assets, for prediction overlays only."""
import hashlib
from pathlib import Path
import struct
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

from .action_edit_kinematics import transform


def load_fingertip_geometry(robot):
    """Use each finger's distal mesh face, not the whole-finger bounding box."""
    path = Path(robot.urdf_path)
    root = ET.parse(path).getroot()
    scale = np.asarray(robot.gripper_scale, float)
    move = robot.gripper_move
    if (scale.shape != (2,) or not np.isfinite(scale).all() or scale[0] >= scale[1]
            or move['sign'] not in (-1, 1)):
        raise ValueError('Invalid native gripper mapping')
    names = [move['base'], move['mimic'][0]]
    if len(set(names)) != 2 or set(names) != set(robot.gripper_joints_name):
        raise ValueError('Expected exactly two configured finger joints')
    fingers = {}
    for name in names:
        joint = root.find(f"joint[@name='{name}']")
        if (joint is None or joint.get('type') != 'prismatic'
                or joint.find('parent').get('link') != robot.ee_link_name):
            raise ValueError('Unsupported finger attachment')
        link_name = joint.find('child').get('link')
        link = root.find(f"link[@name='{link_name}']")
        collisions = link.findall('collision')
        if len(collisions) != 1 or collisions[0].find('geometry/mesh') is None:
            raise ValueError('Expected one finger collision mesh')
        collision = collisions[0]
        mesh = collision.find('geometry/mesh')
        mesh_path = path.parent / mesh.get('filename')
        content = mesh_path.read_bytes()
        if len(content) < 84:
            raise ValueError('Expected binary finger STL')
        count = struct.unpack_from('<I', content, 80)[0]
        if not count or len(content) != 84 + 50 * count:
            raise ValueError('Expected binary finger STL')
        dtype = np.dtype([('normal', '<f4', (3,)), ('vertices', '<f4', (3, 3)), ('attr', '<u2')])
        vertices = np.frombuffer(content, offset=84, count=count, dtype=dtype)['vertices'].reshape(-1, 3)
        vertices = vertices * np.fromstring(mesh.get('scale', '1 1 1'), sep=' ')
        origin = collision.find('origin')
        if origin is not None:
            vertices = (Rotation.from_euler('xyz', np.fromstring(origin.get('rpy', '0 0 0'), sep=' '))
                        .apply(vertices) + np.fromstring(origin.get('xyz', '0 0 0'), sep=' '))
        lo, hi = vertices.min(0), vertices.max(0)
        if not np.isfinite(vertices).all() or np.argmax(hi-lo) != 0 or hi[0] <= abs(lo[0]):
            raise ValueError('Finger mesh does not establish a distal +X extension')
        distal = vertices[np.isclose(vertices[:, 0], hi[0], rtol=0, atol=1e-6)]
        tip = (distal.min(0) + distal.max(0)) / 2
        origin = joint.find('origin')
        rpy = np.fromstring(origin.get('rpy', '0 0 0'), sep=' ')
        offset = np.fromstring(origin.get('xyz', '0 0 0'), sep=' ')
        rotation = Rotation.from_euler('xyz', rpy).as_matrix()
        axis = np.fromstring(joint.find('axis').get('xyz'), sep=' ')
        if axis.shape != (3,) or not np.isclose(np.linalg.norm(axis), 1.):
            raise ValueError('Expected unit finger joint axis')
        limit = joint.find('limit')
        limits = [float(limit.get('lower')), float(limit.get('upper'))]
        if not np.isfinite(limits).all() or limits[0] > limits[1]:
            raise ValueError('Invalid finger joint limits')
        fingers[link_name] = dict(joint=name, tip_at_zero_eef=(offset+rotation@tip).tolist(),
            axis_eef=(rotation@axis).tolist(), joint_limits_m=limits,
            tip_in_finger=tip.tolist(), mesh_sha256=hashlib.sha256(content).hexdigest())
    return dict(schema='x5_distal_fingertips_v1', frame=robot.ee_link_name,
        landmark='distal_collision_mesh_face_center', fingers=fingers,
        gripper_scale=scale.tolist(), gripper_sign=move['sign'], base_joint=move['base'],
        mimic=list(move['mimic']), urdf_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        assumption='Effective commanded openings mapped to URDF-limited joint targets; no contact or tracking prediction.')


def predict_fingertips(eef, geometry):
    """Two world/environment landmarks, including EEF rotation and opening."""
    opening = float(eef['gripper_opening'])
    if not np.isfinite(opening) or not 0 <= opening <= 1:
        raise ValueError('Invalid effective gripper opening')
    low, high = geometry['gripper_scale']
    value = low + opening*(high-low) if geometry['gripper_sign'] == 1 else high-opening*(high-low)
    mimic, multiplier, offset = geometry['mimic']
    joints = {geometry['base_joint']: value, mimic: value*multiplier+offset}
    t = transform(eef['position'], eef['quaternion_wxyz'])
    if not np.isfinite(t).all():
        raise ValueError('Nonfinite EEF pose')
    result = {}
    for name, finger in geometry['fingers'].items():
        q = np.clip(joints[finger['joint']], *finger['joint_limits_m'])
        point = np.asarray(finger['tip_at_zero_eef']) + q*np.asarray(finger['axis_eef'])
        result[name] = (t @ np.r_[point, 1.])[:3].tolist()
    return result
