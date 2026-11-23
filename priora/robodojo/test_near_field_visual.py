import numpy as np

from .skill.near_field_visual import compare_patch


def fixture(depth=1.):
    rng = np.random.default_rng(22)
    rgb = rng.integers(0, 256, (7, 7, 3), dtype=np.uint8)
    k = np.array([[100., 0., 12.], [0., 100., 12.], [0., 0., 1.]])
    v, u = np.mgrid[9:16, 9:16]
    xyz = np.stack(((u-12)*depth/100, (v-12)*depth/100, np.full_like(u, depth, dtype=float)), axis=-1)
    current = np.zeros((25, 25, 3), dtype=np.uint8)
    current[9:16, 9:16] = rgb
    return rgb, xyz, np.ones((7, 7), bool), dict(current_rgb=current,
        current_depth_m=np.full((25, 25), depth), K=k, T_environment_camera=np.eye(4),
        source_step=0, current_step=30)


def test_identical_appearance_is_historical_hint_only():
    rgb, xyz, valid, args = fixture()
    result = compare_patch(rgb, xyz, valid, **args)
    assert result['status'] == 'consistent' and result['compared_pixels'] == 49
    assert result['mean_absolute_rgb_difference'] == 0
    assert not result['current_measurement'] and result['historical_only']


def test_camera_translation_compensated_before_rgb_comparison():
    rgb, xyz, valid, args = fixture()
    args['T_environment_camera'][0, 3] = .03
    args['current_rgb'][:] = 0
    args['current_rgb'][9:16, 6:13] = rgb
    result = compare_patch(rgb, xyz, valid, **args)
    assert result['status'] == 'consistent'
    assert result['mean_absolute_rgb_difference'] == 0


def test_camera_rotation_compensated_before_rgb_comparison():
    rgb, xyz, valid, args = fixture()
    args['T_environment_camera'][:3, :3] = [[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]]
    camera = xyz @ args['T_environment_camera'][:3, :3]
    uvz = camera @ args['K'].T
    uv = np.rint(uvz[..., :2]/uvz[..., 2:3]).astype(int)
    args['current_rgb'][:] = 0
    args['current_rgb'][uv[..., 1], uv[..., 0]] = rgb
    assert compare_patch(rgb, xyz, valid, **args)['status'] == 'consistent'


def test_subpixel_camera_translation_does_not_retire_static_texture():
    rgb, xyz, valid, args = fixture()
    original = args['current_rgb'].astype(float)
    args['T_environment_camera'][0, 3] = .005  # half a pixel at Z=1 m
    args['current_rgb'][:, :-1] = np.rint((original[:, :-1]+original[:, 1:])/2).astype(np.uint8)
    result = compare_patch(rgb, xyz, valid, **args)
    assert result['status'] == 'consistent'
    assert result['compared_pixels'] == 49


def test_large_rgb_change_invalidates():
    rgb, xyz, valid, args = fixture()
    rgb[:] = 30
    args['current_rgb'][:] = 220
    result = compare_patch(rgb, xyz, valid, **args)
    assert result['status'] == 'invalidated_rgb_change'
    assert result['changed_pixel_fraction'] == 1


def test_small_rgb_change_is_not_geometry_verification():
    rgb, xyz, valid, args = fixture()
    rgb[:] = 100
    args['current_rgb'][:] = 105
    result = compare_patch(rgb, xyz, valid, **args)
    assert result['status'] == 'consistent'
    assert 'does not prove' in result['limitations']


def test_occluding_surface_is_unverified_not_claimed_motion():
    rgb, xyz, valid, args = fixture()
    args['current_depth_m'][:] = .5
    args['current_rgb'][:] = 255
    result = compare_patch(rgb, xyz, valid, **args)
    assert result['status'] == 'unverified'
    assert result['occluded_pixels'] == 49


def test_out_of_view_is_unverified():
    rgb, xyz, valid, args = fixture()
    args['T_environment_camera'][0, 3] = 100
    assert compare_patch(rgb, xyz, valid, **args)['status'] == 'unverified'


def test_rgb_comparison_still_possible_in_configured_near_depth_zone():
    rgb, xyz, valid, args = fixture(depth=.15)
    rgb[:] = 100
    args['T_environment_camera'][2, 3] = .071
    args['current_rgb'][:] = 100
    args['current_depth_m'][:] = .079
    result = compare_patch(rgb, xyz, valid, **args)
    assert result['status'] == 'consistent'
    assert result['current_valid_depth_pixels'] == 0
    assert result['current_measurement'] is False


def test_too_small_patch_and_same_source_identity():
    rgb, xyz, valid, args = fixture()
    valid[:] = False
    valid[3, 3] = True
    assert compare_patch(rgb, xyz, valid, **args)['status'] == 'unverified'
    args['current_step'] = 0
    assert compare_patch(rgb, xyz, valid, **args)['status'] == 'same_observation'
