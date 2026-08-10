from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from math import sqrt
from urllib.request import Request, urlopen

from PIL import Image


def _load_image(value) -> Image.Image:
    max_pixels = None
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, dict):
        max_pixels = value.get("max_pixels")
        value = value.get("image") or value.get("url") or value.get("path")
    if not isinstance(value, str):
        raise TypeError(f"Unsupported image reference: {type(value).__name__}")
    if value.startswith(("http://", "https://")):
        request = Request(value, headers={"User-Agent": "kdflow/1.0"})
        with urlopen(request, timeout=60) as response:
            image = Image.open(BytesIO(response.read())).convert("RGB")
    else:
        image = Image.open(value).convert("RGB")
    if max_pixels and image.width * image.height > max_pixels:
        scale = sqrt(max_pixels / (image.width * image.height))
        image = image.resize(
            (max(1, int(image.width * scale)), max(1, int(image.height * scale))),
            Image.Resampling.LANCZOS,
        )
    return image


def materialize_image_batch(image_batch):
    """Load a batch of image-reference lists while preserving sample grouping."""
    if image_batch is None:
        return None

    groups = [
        value if isinstance(value, list) else ([] if value is None else [value])
        for value in image_batch
    ]
    references = [reference for group in groups for reference in group]
    if not references:
        return [[] for _ in groups]

    with ThreadPoolExecutor(max_workers=min(32, len(references))) as pool:
        loaded = iter(pool.map(_load_image, references))
        return [[next(loaded) for _ in group] for group in groups]
