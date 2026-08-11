from concurrent.futures import ThreadPoolExecutor
from urllib.request import Request, urlopen

import torch
from torchvision.io import ImageReadMode, decode_image, read_image


def _load_image(value) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, dict):
        value = value.get("image") or value.get("url") or value.get("path")
    if not isinstance(value, str):
        raise TypeError(f"Unsupported image reference: {type(value).__name__}")
    if value.startswith(("http://", "https://")):
        request = Request(value, headers={"User-Agent": "kdflow/1.0"})
        with urlopen(request, timeout=60) as response:
            encoded = torch.frombuffer(bytearray(response.read()), dtype=torch.uint8)
        return decode_image(
            encoded,
            mode=ImageReadMode.RGB,
            apply_exif_orientation=True,
        )
    return read_image(value, mode=ImageReadMode.RGB, apply_exif_orientation=True)


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
