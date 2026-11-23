"""Teacher-only JPEG attachments; never modify policy arrays or source images."""
from io import BytesIO
from pathlib import Path

from PIL import Image
from ..camera_intrinsics import preview_calibration

JPEG_QUALITY = 90
JPEG_SUBSAMPLING = 0  # Keep chroma detail for small colored objects and digits.
IMAGE_MIME_TYPE = 'image/jpeg'


def encode_jpeg(picture):
    """Encode an RGB preview; flatten transparency onto a white background."""
    if picture.mode in ('RGBA', 'LA') or 'transparency' in picture.info:
        rgba = picture.convert('RGBA')
        rgb = Image.new('RGB', picture.size, 'white')
        rgb.paste(rgba, mask=rgba.getchannel('A'))
    else:
        rgb = picture.convert('RGB')
    buffer = BytesIO()
    # The optimized encoder can exhaust Pillow's buffer on noisy 4:4:4 frames.
    # Streaming encoding keeps quality/chroma settings and handles those frames.
    rgb.save(buffer, format='JPEG', quality=JPEG_QUALITY,
             subsampling=JPEG_SUBSAMPLING, optimize=False)
    return buffer.getvalue()


def image_max_edge(value):
    if not str(value).isdigit():
        raise ValueError('CODEX_IMAGE_MAX_EDGE must be a non-negative integer (0 disables resizing)')
    return int(value)


def prepare_image(item, max_edge):
    """Return a copied descriptor and exact attachment bytes, saving any preview.

    ``path`` remains the full-resolution source for native view_image access.
    A separate preview is stored next to the immutable observation, so the
    existing artifact inventory and RPC log preserve what Codex actually saw.
    """
    max_edge = image_max_edge(max_edge)
    source = Path(item['path'])
    original = source.read_bytes()
    with Image.open(BytesIO(original)) as picture:
        original_size = list(picture.size)
        resized = max_edge > 0 and max(picture.size) > max_edge
        if resized:
            method = Image.Resampling.NEAREST if item.get('modality') == 'depth' else Image.Resampling.LANCZOS
            picture.thumbnail((max_edge, max_edge), method)
        payload = encode_jpeg(picture)
        size = list(picture.size)
    preview = source.parent/'codex_previews'/f'{source.name}.max{max_edge}.q{JPEG_QUALITY}.jpg'
    preview.parent.mkdir(exist_ok=True)
    try:
        with preview.open('xb') as stream:
            stream.write(payload)
    except FileExistsError:
        if preview.read_bytes() != payload:
            raise ValueError(f'Existing Codex preview differs; refusing to overwrite {preview}')
    result = dict(item, codex_preview=dict(path=str(preview), size=size,
        original_size=original_size, resampling=('NEAREST' if item.get('modality') == 'depth' else 'LANCZOS') if resized else None,
        format='JPEG', mime_type=IMAGE_MIME_TYPE, quality=JPEG_QUALITY,
        subsampling=JPEG_SUBSAMPLING))
    if item.get('camera_intrinsics'):
        result['codex_preview']['camera_intrinsics'] = preview_calibration(item['camera_intrinsics'], original_size, size)
    return result, payload
