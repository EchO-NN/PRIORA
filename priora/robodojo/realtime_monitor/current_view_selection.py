"""Current candidate wrists only. Never sample or filter actual-history images."""
import math

import numpy as np

from .projection import CAMERAS

VERSION = 'current_wrist_path_grasp_exceptions_v1'
RULES = (
    '当前候选视角经过筛选：主视角始终保留；腕部按对应机械臂的预测指尖路径判断。'
    '基本静止按环境坐标系下整段 EEF 位姿变化判断，不使用投影像素位移。'
    '路径出画或基本静止时省略该腕部，但当前夹爪处于闭合侧状态、或本段在画面内存在'
    '张开到闭合的转变时保留。固定说明中的三视角是可选来源；本次只提供下方 '
    'camera_order 列出的视角，不要假设仍有三张。'
    '省略不表示失败；保留也不证明抓稳或无遮挡。本规则不作用于任何历史实际图像。')


def _opening(pose):
    value = pose.get('gripper_opening_command', pose.get('gripper_opening',pose.get('opening')))
    if value is None:
        return None
    value = float(value)
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError('Expected a finite current gripper opening in [0,1]')
    return value


def spatial_motion(trajectory,current_pose,arm):
    """Maximum pairwise pose change, including actual start; camera independent."""
    if not trajectory:
        return None
    poses=[current_pose]+[node[arm]for node in trajectory]
    if any('position' not in p or 'quaternion_wxyz' not in p for p in poses):
        return None
    if any(p.get('frame','environment_origin')!='environment_origin' for p in poses):
        raise ValueError('Spatial motion requires a shared environment coordinate frame')
    points=np.asarray([p['position']for p in poses],dtype=float)
    qs=np.asarray([p['quaternion_wxyz']for p in poses],dtype=float)
    if (points.shape!=(len(poses),3) or qs.shape!=(len(poses),4)
            or not np.isfinite(points).all() or not np.isfinite(qs).all()
            or np.any(np.linalg.norm(qs,axis=1)<1e-12)):
        raise ValueError('Finite XYZ and nonzero wxyz quaternions required')
    qs=qs/np.linalg.norm(qs,axis=1)[:,None]
    displacement=float(np.linalg.norm(points[:,None,:]-points[None,:,:],axis=2).max())
    # Absolute dot accounts for q/-q equivalence, including wraparound.
    angle=float(np.degrees(2*np.arccos(np.clip(np.abs(qs@qs.T),0,1))).max())
    return dict(max_position_span_m=displacement,max_rotation_span_deg=angle)


def select_current_views(projection, current_eef, trajectory=None, *,
                         stationary_translation_m=.005,stationary_rotation_deg=3.):
    """Measure the attached arm, not the other arm that happens to cross a view.

    Drawing uses snapped open/closed fingertip geometry. A close event is in
    frame when the midpoint of its two rendered fingertips is in front of the
    camera and within the image; raw fingertip visibility is still audited.
    First candidate step is compared with the actual current gripper state.
    Unknown/missing geometry retains the view explicitly rather than guessing.
    """
    translation,rotation=float(stationary_translation_m),float(stationary_rotation_deg)
    if any(not math.isfinite(v) or v<0 for v in [translation,rotation]):
        raise ValueError('Nonnegative spatial stationarity thresholds required')
    current_eef = current_eef or {}
    current_eef = current_eef.get('current_eef', current_eef)
    result = dict(version=VERSION, scope='current_candidate_only',
        stationary_translation_m=translation,stationary_rotation_deg=rotation,
        motion_frame='environment_origin',motion_landmark='EEF link6 pose',
        stationary_rule='maximum pairwise XYZ span <= translation AND rotation span <= angle; include actual start',
        arm_scope='camera-attached arm only', closed_rule='opening < 0.5; not proof of grasp',
        close_event_visibility='rendered fingertip midpoint in image; not an occlusion test',
        selected_cameras=['cam_high'], views={'cam_high':dict(keep=True,reason='main_always')})
    views = (projection or {}).get('views',{})
    for camera,arm in [('cam_left_wrist','left'),('cam_right_wrist','right')]:
        view = views.get(camera,{})
        paths = view.get('paths',[])
        opening = _opening(current_eef.get(arm,{}))
        metrics = dict(arm=arm,current_opening=opening,visible_close_steps=[])
        if not paths or opening is None:
            metrics.update(keep=True,reason='missing_geometry_or_current_opening_keep')
        else:
            if trajectory is not None and len(trajectory)!=len(paths):
                raise ValueError('Use the complete same-prefix spatial trajectory, not sparse key nodes')
            motion=spatial_motion(trajectory,current_eef.get(arm,{}),arm)
            w,h = view['original_size_wh']
            if min(w,h)<=0:
                raise ValueError('Positive image dimensions required')
            names = sorted(paths[0].get('display_arms',paths[0]['arms'])[arm])
            if len(names)!=2:
                raise ValueError('Two rendered fingertip paths required')
            outside=0;previous_closed=opening<.5
            for step,node in enumerate(paths,1):
                fingers=node.get('display_arms',node['arms'])[arm]
                xy=[]
                for name in names:
                    value=fingers[name].get('pixel')
                    p=np.asarray(value,dtype=float) if value is not None else None
                    finite=p is not None and p.shape==(2,) and np.isfinite(p).all()
                    in_frame=bool(finite and 0<=p[0]<w and 0<=p[1]<h)
                    outside+=not in_frame
                    if finite:
                        xy.append(p)
                now=_opening(node['grippers'][arm])<.5
                center=np.mean(xy,axis=0) if len(xy)==2 else None
                if not previous_closed and now and center is not None and 0<=center[0]<w and 0<=center[1]<h:
                    metrics['visible_close_steps'].append(step)
                previous_closed=now
            still=None if motion is None else (motion['max_position_span_m']<=translation+1e-12
                and motion['max_rotation_span_deg']<=rotation+1e-9)
            metrics.update(out_of_frame_points=outside,path_exits_frame=bool(outside),
                spatial_motion=motion,stationary=still,
                current_closed=opening<.5)
            if opening<.5:
                metrics.update(keep=True,reason='current_gripper_closed_exception')
            elif metrics['visible_close_steps']:
                metrics.update(keep=True,reason='visible_open_to_closed_exception')
            elif outside:
                metrics.update(keep=False,reason='path_outside_image')
            elif still:
                metrics.update(keep=False,reason='stationary_spatial_pose')
            elif motion is None:
                metrics.update(keep=True,reason='missing_spatial_trajectory_keep')
            else:
                metrics.update(keep=True,reason='moving_path_inside_image')
        result['views'][camera]=metrics
        if metrics['keep']:
            result['selected_cameras'].append(camera)
    return result


def selected_camera_order(selection):
    """Validate a bound selection receipt; absent receipt means legacy all views."""
    if selection is None:
        return list(CAMERAS)
    cameras=selection.get('selected_cameras')
    if (selection.get('version')!=VERSION or selection.get('scope')!='current_candidate_only'
            or not isinstance(cameras,list) or not cameras or cameras[0]!='cam_high'
            or cameras!=[c for c in CAMERAS if c in cameras]
            or set(selection.get('views',{}))!=set(CAMERAS)
            or any(type(selection['views'][c].get('keep')) is not bool for c in CAMERAS)
            or cameras!=[c for c in CAMERAS if selection['views'][c]['keep']]):
        raise ValueError('Invalid current candidate camera selection')
    return cameras
