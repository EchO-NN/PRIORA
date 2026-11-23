"""Check dated local RGB evidence after compensating known camera motion.

Reprojection assumes the remembered surface stayed still. This is a hypothesis
for an appearance check, not a tracker or a new depth measurement. A match on
plain cloth cannot prove that the cloth did not move or deform.
"""
import numpy as np

from ..camera_geometry import validate_transform
from ..camera_intrinsics import validate_k

VERSION = 'near_field_rgb_check_v1'
MIN_PIXELS = 9
MIN_COVERAGE = .6
OCCLUSION_MARGIN_M = .02
PIXEL_CHANGE_THRESHOLD = 40.  # Mean absolute RGB difference, on the 0..255 scale.
MEAN_CHANGE_THRESHOLD = 30.
CHANGED_FRACTION_THRESHOLD = .5


def _smooth_on_support(values, support):
    """Use the same local support on both images; never mix in occluded pixels."""
    kernel = np.array([1., 4., 6., 4., 1.])/16
    h, w = support.shape
    weighted = np.pad(values*support[..., None], ((2, 2), (2, 2), (0, 0)), mode='edge')
    weights = np.pad(support.astype(float), 2, mode='edge')
    total = np.zeros_like(values, dtype=float)
    denominator = np.zeros((h, w), dtype=float)
    for y in range(5):
        for x in range(5):
            weight = kernel[y]*kernel[x]
            total += weight*weighted[y:y+h, x:x+w]
            denominator += weight*weights[y:y+h, x:x+w]
    return total/np.maximum(denominator[..., None], 1e-12)


def compare_patch(rgb, xyz_environment, valid, *, current_rgb, current_depth_m,
                  K, T_environment_camera, source_step, current_step):
    """Return appearance evidence only; never mutate inputs or update old XYZ."""
    old = np.asarray(rgb)
    xyz = np.asarray(xyz_environment, dtype=float)
    mask = np.asarray(valid, dtype=bool)
    current = np.asarray(current_rgb)
    depth = np.asarray(current_depth_m, dtype=float)
    if (old.ndim != 3 or old.shape[-1] != 3 or old.dtype != np.uint8
            or xyz.shape != old.shape or mask.shape != old.shape[:2]
            or current.ndim != 3 or current.shape[-1] != 3 or current.dtype != np.uint8
            or depth.shape != current.shape[:2]
            or type(source_step) is not int or type(current_step) is not int
            or not 0 <= source_step <= current_step):
        raise ValueError('Invalid RGB memory comparison geometry or observation identity')
    k = validate_k(K)
    t = validate_transform(T_environment_camera)
    base = dict(version=VERSION, source_step=source_step, current_step=current_step,
        alignment='historical world XYZ projected into current camera using measured camera geometry',
        alignment_assumption='remembered scene surface stationary; a comparison hypothesis, not object tracking',
        current_measurement=False, historical_only=True,
        limitations='Similar RGB, especially uniform cloth, does not prove unchanged 3D geometry or grasp retention. '
                    'Lighting, material appearance and viewing angle can change RGB without object motion.',
        thresholds=dict(min_pixels=MIN_PIXELS, min_coverage=MIN_COVERAGE,
            occlusion_margin_m=OCCLUSION_MARGIN_M, pixel_mean_absolute_rgb=PIXEL_CHANGE_THRESHOLD,
            patch_mean_absolute_rgb=MEAN_CHANGE_THRESHOLD,
            changed_fraction=CHANGED_FRACTION_THRESHOLD))
    if current_step == source_step:
        return dict(base, status='same_observation', reason='same acknowledged source observation')
    selected = mask & np.isfinite(xyz).all(axis=-1)
    source_indices = np.flatnonzero(selected)
    points, colors = xyz[selected], old[selected].astype(float)
    n = len(points)
    if n < MIN_PIXELS:
        return dict(base, status='unverified', reason='too few valid historical surface pixels',
                    source_valid_pixels=n, compared_pixels=0)
    # Inverse rigid transform for row-vector points. Projection is permitted
    # below .1 m because it predicts an RGB location, not a valid depth sample.
    camera_points = (points-t[:3, 3]) @ t[:3, :3]
    positive = camera_points[:, 2] > 1e-6
    projected = camera_points @ k.T
    uv = np.full((n, 2), np.nan)
    uv[positive] = projected[positive, :2]/projected[positive, 2:3]
    height, width = depth.shape
    inside = (positive & np.isfinite(uv).all(axis=1)
              & (uv[:, 0] >= 0) & (uv[:, 0] <= width-1)
              & (uv[:, 1] >= 0) & (uv[:, 1] <= height-1))
    in_count = int(inside.sum())
    base.update(source_valid_pixels=n, projected_in_image_pixels=in_count,
                image_coverage=in_count/n)
    if in_count < MIN_PIXELS or in_count/n < MIN_COVERAGE:
        return dict(base, status='unverified', reason='historical patch mostly outside current view',
                    compared_pixels=0)
    sample_uv = uv[inside]
    source_indices = source_indices[inside]
    coords = np.floor(sample_uv).astype(int)
    depths = camera_points[inside, 2]
    colors = colors[inside]
    # Avoid counting repeated projections at one current pixel as independent
    # evidence; use the nearest historical sample if projection collides.
    order = np.argsort(depths, kind='stable')
    linear = coords[:, 1]*width+coords[:, 0]
    _, unique = np.unique(linear[order], return_index=True)
    chosen = order[unique]
    coords, depths, colors = coords[chosen], depths[chosen], colors[chosen]
    sample_uv, source_indices = sample_uv[chosen], source_indices[chosen]
    depth_coords = np.rint(sample_uv).astype(int)
    observed_depth = depth[depth_coords[:, 1], depth_coords[:, 0]]
    depth_valid = np.isfinite(observed_depth) & (observed_depth >= .1) & (observed_depth <= 10.)
    occluded = depth_valid & (observed_depth < depths-OCCLUSION_MARGIN_M)
    visible = ~occluded
    comparisons = int(visible.sum())
    base.update(unique_projected_pixels=len(chosen), occluded_pixels=int(occluded.sum()),
                current_valid_depth_pixels=int(depth_valid.sum()),
                unoccluded_fraction=comparisons/len(chosen), compared_pixels=comparisons,
                combined_coverage=in_count/n*comparisons/len(chosen))
    if (comparisons < MIN_PIXELS or comparisons/len(chosen) < MIN_COVERAGE
            or base['combined_coverage'] < MIN_COVERAGE):
        return dict(base, status='unverified', reason='historical surface is occluded or has insufficient visible samples')
    coords, sample_uv, source_indices = coords[visible], sample_uv[visible], source_indices[visible]
    # Bilinear sampling followed by shared-support smoothing avoids retiring a
    # static textured patch solely because camera motion shifts it half a pixel.
    x0, y0 = coords[:, 0], coords[:, 1]
    x1, y1 = np.minimum(x0+1, width-1), np.minimum(y0+1, height-1)
    dx, dy = (sample_uv-coords).T
    observed = ((1-dx)[:, None]*(1-dy)[:, None]*current[y0, x0]
                + dx[:, None]*(1-dy)[:, None]*current[y0, x1]
                + (1-dx)[:, None]*dy[:, None]*current[y1, x0]
                + dx[:, None]*dy[:, None]*current[y1, x1])
    support = np.zeros(mask.shape, dtype=bool)
    support.flat[source_indices] = True
    aligned = np.zeros(old.shape, dtype=float)
    aligned.reshape(-1, 3)[source_indices] = observed
    smoothed_old = _smooth_on_support(old.astype(float), support)
    smoothed_new = _smooth_on_support(aligned, support)
    difference = np.abs(smoothed_old[support]-smoothed_new[support]).mean(axis=1)
    mean = float(difference.mean())
    fraction = float((difference >= PIXEL_CHANGE_THRESHOLD).mean())
    changed = mean >= MEAN_CHANGE_THRESHOLD and fraction >= CHANGED_FRACTION_THRESHOLD
    borderline = not changed and (mean >= MEAN_CHANGE_THRESHOLD or fraction >= CHANGED_FRACTION_THRESHOLD)
    return dict(base, mean_absolute_rgb_difference=mean, changed_pixel_fraction=fraction,
                sampling='bilinear RGB with shared-support 5x5 binomial smoothing',
                status='invalidated_rgb_change' if changed else 'unverified' if borderline else 'consistent',
                reason='large aligned RGB change; discard this memory for planning' if changed else
                       'mixed RGB change evidence; require a new view' if borderline else
                       'below both RGB change thresholds; historical hint only, not confirmation of static geometry')
