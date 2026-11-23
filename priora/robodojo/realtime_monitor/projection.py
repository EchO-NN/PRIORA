"""Project candidate fingertips into each CURRENT camera frame, including wrists."""
import numpy as np
from PIL import Image, ImageDraw

from ..robodojo_server.fingertips import predict_fingertips

CAMERAS = ('cam_high', 'cam_left_wrist', 'cam_right_wrist')
GRIPPER_COLORS = {'closed': '#00b8ff', 'open': '#ff9a2e'}
GRIPPER_COLOR_ENCODING = dict(
    version='highcontrast_directional_fingertips_v5', colors=GRIPPER_COLORS,
    opening_scale='0 closed, 1 open; effective command, not measured contact',
    closed_rule='gripper_opening < 0.5', open_rule='gripper_opening >= 0.5',
    arm_styles={'left': 'solid', 'right': 'dashed'},
    image_content='native RGB with 4px colored fingertip paths, dark/light outlines and sparse direction chevrons; orange dots only at closed-to-open transitions; no text or timeline',
    direction_marker='at most one chevron per fingertip on a sufficiently long contiguous visible command phase; follows increasing candidate step; not a path endpoint, opening event or grasp claim',
    stroke_semantics='line width and outlines improve visibility only; not fingertip size, tolerance or collision clearance',
    opening_marker='two orange disks at the projected fingertips when the command switches closed-to-open; not proof of release',
    segment_rule='connect within each open/closed phase only; no connector across a state change',
    display_geometry='snap opening to 0 closed or 1 open for drawing only; exact FK remains in arms',
    warning='Colors do not prove grasp success, object motion or release.')

# Opt-in ablation: keep the archived thin lines and opening disks unchanged.
THIN_ARROW_ENCODING_VERSION = 'thin_directional_fingertips_v1'


def _gripper_state(opening):
    value = float(opening)
    if not np.isfinite(value) or not 0 <= value <= 1:
        raise ValueError('Expected finite effective gripper opening in [0,1]')
    return dict(opening=value, state='closed' if value < .5 else 'open')


def gripper_phase_summary(openings):
    """Compact command phases in text, never painted over camera evidence."""
    summary = {}
    for arm in ('left', 'right'):
        runs = []
        for step, opening in enumerate(openings[arm], 1):
            state = _gripper_state(opening)['state']
            if not runs or runs[-1]['command'] != state:
                runs.append(dict(first_step=step, last_step=step, command=state))
            else:
                runs[-1]['last_step'] = step
        summary[arm] = runs
    return summary


def with_gripper_commands(projection, openings):
    """Attach recorded effective openings without changing archived pixel paths."""
    from copy import deepcopy
    result = deepcopy(projection)
    views = result['views'].values() if result.get('schema') == 'predicted_fingertip_multiview_v1' else [result]
    for view in views:
        if set(openings) != {'left', 'right'} or any(
                len(openings[arm]) != len(view['paths']) for arm in ('left', 'right')):
            raise ValueError('Effective opening timeline must match both full projected paths')
        states = [{arm: _gripper_state(openings[arm][i]) for arm in ('left', 'right')}
                  for i in range(len(view['paths']))]
        for node in view['paths'] + view['nodes']:
            node['grippers'] = states[node['index']]
            # This compatibility helper only has saved pixels, not 3D geometry.
            # Never keep stale binary geometry after attaching another command timeline.
            node.pop('display_arms', None)
        view['gripper_color_encoding'] = deepcopy(GRIPPER_COLOR_ENCODING)
        view['gripper_color_encoding']['display_geometry'] = 'archived pixels retained; no binary geometry available'
    return result


def project_candidate(observation, trajectory, fingertip_geometry, camera='cam_high'):
    if camera not in CAMERAS:
        raise ValueError('Unknown projection camera')
    k = np.asarray(observation[camera+'_intrinsics_K'], dtype=float)
    environment_from_optical = np.asarray(observation[camera+'_T_environment_camera'], dtype=float)
    if (k.shape != (3, 3) or environment_from_optical.shape != (4, 4)
            or not np.isfinite(k).all() or not np.isfinite(environment_from_optical).all()):
        raise ValueError('Finite current intrinsics and optical-to-environment transform required')
    optical_from_environment = np.linalg.inv(environment_from_optical)
    h, w = np.asarray(observation[camera]).shape[:2]
    if not trajectory or set(fingertip_geometry) != {'left', 'right'}:
        raise ValueError('Two-arm fingertip geometry and a nonempty trajectory are required')
    nodes = []
    for index, state in enumerate(trajectory):
        row = dict(index=index, arms={}, display_arms={}, grippers={})
        for arm in ('left', 'right'):
            row['grippers'][arm] = _gripper_state(state[arm]['gripper_opening'])
            display_eef = dict(state[arm], gripper_opening=(
                0. if row['grippers'][arm]['state']=='closed' else 1.))
            for field, eef in (('arms', state[arm]), ('display_arms', display_eef)):
                fingers = {}
                for name, point in predict_fingertips(eef, fingertip_geometry[arm]).items():
                    xyz = optical_from_environment @ np.r_[point, 1.]
                    pixel = None
                    if np.isfinite(xyz).all() and xyz[2] > 0:
                        uvw = k @ xyz[:3]
                        pixel = (uvw[:2]/uvw[2]).tolist()
                    fingers[name] = dict(pixel=pixel,
                        in_frame=bool(pixel is not None and 0<=pixel[0]<w and 0<=pixel[1]<h))
                row[field][arm] = fingers
        nodes.append(row)
    indices = sorted(set(np.linspace(0, len(nodes)-1, min(4, len(nodes)), dtype=int).tolist()))
    return dict(schema='predicted_fingertip_overlay_v1', camera_id=camera,
        original_size_wh=[w, h], nodes=[nodes[i] for i in indices], paths=nodes,
        rendered_steps=len(nodes), transform_source='current_observation_optical_camera_to_environment',
        source_type='predicted_kinematics', landmark='two_distal_collision_mesh_face_centers_per_gripper',
        gripper_source='effective_command_with_native_scale_mimic_and_urdf_limits',
        gripper_color_encoding=GRIPPER_COLOR_ENCODING,
        current_camera_frame_frozen=True,
        camera_motion_compensated=False, occlusion_checked=False,
        warning='Predicted targets in the CURRENT camera pose; wrist camera will move during execution. '
                'Paths may be partial: out-of-frame and behind-camera points are not drawn. '
                'A missing path endpoint is not action end, release or failure; check other views, '
                'the textual command phases and complete numeric candidate. '
                'Not a future image, measured motion, visibility proof or guaranteed contact.')


def project_candidate_views(observation, trajectory, fingertip_geometry):
    return dict(schema='predicted_fingertip_multiview_v1',
        views={camera: project_candidate(observation, trajectory, fingertip_geometry, camera)
               for camera in CAMERAS})


def compact_projection_views(projection):
    """Images carry the paths; keep per-view semantics in the prompt, full pixels in audit."""
    if projection.get('schema') != 'predicted_fingertip_multiview_v1':
        raise ValueError('Expected three-view fingertip projection')
    return dict(schema=projection['schema'], pixel_paths_in_full_evidence=True,
        views={camera: {**{key: value for key, value in view.items() if key not in ('nodes', 'paths')},
                        'labeled_steps': [] if view.get('gripper_color_encoding') else [node['index']+1 for node in view['nodes']],
                        'key_node_steps': [node['index']+1 for node in view['nodes']]}
               for camera, view in projection['views'].items()})


def annotated_views(observation, projection):
    views = {camera: np.array(observation[camera], copy=True)
        for camera in CAMERAS}
    if not projection:
        return views
    if projection.get('schema') == 'predicted_fingertip_multiview_v1':
        if set(projection['views']) != set(CAMERAS):
            raise ValueError('Expected projections for all three cameras')
        for camera, view in projection['views'].items():
            if view.get('camera_id') != camera:
                raise ValueError('Projection camera identity mismatch')
            views[camera] = _draw_view(views[camera], view)
    else:
        # Explicit compatibility for the preceding main-camera-only revision.
        camera = projection.get('camera_id')
        if camera not in CAMERAS:
            raise ValueError('Unknown projection camera')
        views[camera] = _draw_view(views[camera], projection)
    return views


def _draw_view(frame, projection):
    if projection.get('schema') != 'predicted_fingertip_overlay_v1':
        raise ValueError('Expected explicitly identified fingertip projection')
    if projection.get('gripper_color_encoding'):
        return _draw_gripper_view(frame, projection)
    # Old archived projections have no command timeline: keep their original legend.
    image = Image.fromarray(frame); draw = ImageDraw.Draw(image)
    labels = {node['index'] for node in projection['nodes']}
    for arm, colors in [('left', ('#00e1ff', '#a0f4ff')), ('right', ('#ffa94a', '#ffe0a3'))]:
        names = sorted(projection['paths'][0]['arms'][arm])
        if len(names) != 2:
            raise ValueError('Expected two projected fingers per gripper')
        for finger_index, (name, color) in enumerate(zip(names, colors)):
            previous = None
            for node in projection['paths']:
                value = node['arms'][arm][name]
                if not value['in_frame']:
                    previous = None
                    continue
                x, y = value['pixel']
                if previous is not None:
                    draw.line([previous, (x, y)], fill=color, width=2)
                if node['index'] in labels:
                    marker = draw.ellipse if finger_index == 0 else draw.rectangle
                    marker((x-3, y-3, x+3, y+3), outline=color, width=2)
                    draw.text((x+4, y+(2 if finger_index == 0 else -12)), str(node['index']+1), fill=color)
                previous = (x, y)
    draw.rectangle((0, 0, image.width, 30), fill='#101c2d')
    draw.text((3, 2), 'PREDICTED fingertips: L cyan / R orange; 7 circle / 8 square; not executed', fill='white')
    draw.text((3, 16), 'CURRENT camera pose held fixed; visibility/contact not verified', fill='white')
    return np.asarray(image)


def _stroke(draw, a, b, color, dashed, phase, width=2):
    delta = np.asarray(b) - np.asarray(a)
    length = float(np.linalg.norm(delta))
    if length == 0:
        draw.point(tuple(a), fill=color)
        return phase
    intervals = [(0., length)]
    if dashed:
        intervals = []
        start = -phase
        while start < length:
            if start+6. > 0:
                intervals.append((max(0., start), min(length, start+6.)))
            start += 10.
    for start, end in intervals:
        points = [tuple(np.asarray(a) + delta*t/length) for t in (start, end)]
        draw.line(points, fill=color, width=width)
    return (phase + length) % 10.


def _draw_gripper_view(frame, projection):
    if projection['gripper_color_encoding'].get('version') == 'highcontrast_directional_fingertips_v5':
        return _draw_contrast_gripper_view(frame, projection)
    # All semantics and command timing travel as text; leave only thin paths in RGB.
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    paths = projection['paths']
    opening_points = []
    for arm in ('left', 'right'):
        names = sorted(paths[0]['arms'][arm])
        if len(names) != 2:
            raise ValueError('Expected two projected fingers per gripper')
        for name in names:
            previous, previous_state, phase = None, None, 0.
            for node in paths:
                command = _gripper_state(node['grippers'][arm]['opening'])
                color = GRIPPER_COLORS[command['state']]
                just_opened = previous_state == 'closed' and command['state'] == 'open'
                if command['state'] != previous_state:
                    previous, phase = None, 0.
                previous_state = command['state']
                point = node.get('display_arms', node['arms'])[arm][name]
                if not point['in_frame']:
                    previous, phase = None, 0.
                    continue
                xy = tuple(point['pixel'])
                if just_opened:
                    opening_points.append(xy)
                if previous is not None:
                    phase = _stroke(draw, previous, xy, color, arm == 'right', phase)
                else:
                    draw.point(xy, fill=color)
                previous = xy
    if projection['gripper_color_encoding'].get('version') == THIN_ARROW_ENCODING_VERSION:
        for marker in thin_direction_markers(projection):
            draw.line(marker['points'], fill=GRIPPER_COLORS[marker['state']],
                      width=2, joint='curve')
    # Show the opening event once per fingertip, without drawing the opening sweep.
    for x,y in opening_points:
        draw.ellipse((x-5,y-5,x+5,y+5), fill='#101c2d')
        draw.ellipse((x-4,y-4,x+4,y+4), fill=GRIPPER_COLORS['open'])
    return np.asarray(image)


def _visible_command_runs(paths, arm, name):
    """Never bridge a command change, a camera-plane break or an offscreen gap."""
    runs, current, opening_points = [], [], []
    previous_state = None
    for node in paths:
        state = _gripper_state(node['grippers'][arm]['opening'])['state']
        just_opened = previous_state == 'closed' and state == 'open'
        point = node.get('display_arms', node['arms'])[arm][name]
        if state != previous_state or not point['in_frame']:
            if current:
                runs.append((previous_state, current))
                current = []
        previous_state = state
        if not point['in_frame']:
            continue
        xy = tuple(point['pixel'])
        if just_opened:
            opening_points.append(xy)
        current.append(xy)
    if current:
        runs.append((previous_state, current))
    return runs, opening_points


def _direction_chevron(points, image_size):
    """One local time-direction cue inside a visible run, never at a clipped end."""
    points = np.asarray(points, dtype=float)
    if len(points) < 2:
        return None
    distances = np.r_[0., np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    if distances[-1] < 35.:
        return None
    def at(distance):
        i = min(int(np.searchsorted(distances, distance, side='right')), len(points)-1)
        return points[i-1] + (points[i]-points[i-1]) * (
            (distance-distances[i-1]) / max(distances[i]-distances[i-1], 1e-9))
    distance = .65 * distances[-1]
    tip = at(distance)
    direction = at(distance+6.) - at(distance-6.)
    length = np.linalg.norm(direction)
    if length < 1.:
        return None
    direction /= length
    perpendicular = np.array([-direction[1], direction[0]])
    legs = [tip-10.*direction+5.*perpendicular, tip, tip-10.*direction-5.*perpendicular]
    # Keep the cue complete; a view boundary is not a true action endpoint.
    w,h = image_size
    if any(not (4 <= p[0] < w-4 and 4 <= p[1] < h-4) for p in legs):
        return None
    return [tuple(p) for p in legs]


def thin_direction_markers(projection):
    """Sparse increasing-time arrows; do not bridge visibility or command gaps."""
    markers = []
    for arm in ('left', 'right'):
        names = sorted(projection['paths'][0]['arms'][arm])
        if len(names) != 2:
            raise ValueError('Expected two projected fingers per gripper')
        for name in names:
            runs, _ = _visible_command_runs(projection['paths'], arm, name)
            runs.sort(key=lambda run: sum(np.linalg.norm(np.asarray(b)-a)
                for a,b in zip(run[1],run[1][1:])), reverse=True)
            for state, points in runs:
                arrow = _direction_chevron(points, projection['original_size_wh'])
                if arrow is not None:
                    markers.append(dict(arm=arm, finger=name, state=state, points=arrow))
                    break
    return markers


def _draw_contrast_gripper_view(frame, projection):
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    runs, arrows, openings = [], [], []
    for arm in ('left', 'right'):
        names = sorted(projection['paths'][0]['arms'][arm])
        if len(names) != 2:
            raise ValueError('Expected two projected fingers per gripper')
        for name in names:
            finger_runs, events = _visible_command_runs(projection['paths'], arm, name)
            openings.extend(events)
            runs.extend((state, points, arm == 'right') for state,points in finger_runs)
            # Sparse cues: do not add an arrow at every sampled node or every phase.
            candidates = sorted(finger_runs, key=lambda r: sum(
                np.linalg.norm(np.asarray(b)-a) for a,b in zip(r[1],r[1][1:])), reverse=True)
            for state, points in candidates:
                arrow = _direction_chevron(points, image.size)
                if arrow is not None:
                    arrows.append((state, arrow))
                    break
    # Render all outlines before colored cores so crossing paths retain both colors.
    for width, outline in ((8, '#ffffff'), (6, '#101c2d'), (4, None)):
        for state, points, dashed in runs:
            color = outline or GRIPPER_COLORS[state]
            phase = 0.
            if len(points) == 1:
                draw.point(points[0], fill=color)
            for a,b in zip(points,points[1:]):
                phase = _stroke(draw, a, b, color, dashed, phase, width=width)
    for state, arrow in arrows:
        for width, color in ((7,'#ffffff'),(5,'#101c2d'),(3,GRIPPER_COLORS[state])):
            draw.line(arrow, fill=color, width=width, joint='curve')
    for x,y in openings:
        draw.ellipse((x-6,y-6,x+6,y+6), fill='#ffffff')
        draw.ellipse((x-5,y-5,x+5,y+5), fill='#101c2d')
        draw.ellipse((x-4,y-4,x+4,y+4), fill=GRIPPER_COLORS['open'])
    return np.asarray(image)
