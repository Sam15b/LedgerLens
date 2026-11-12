"""Clean up a page image before OCR: grayscale, deskew, denoise, upscale, binarize.

Each step except upscaling is switched on or off by ``ocr.preprocess`` in
``configs/ingestion.yaml``. The order matters and was measured with real Tesseract on the
speckled receipt fixture (character error rate, lower is better):

==========================================  =====
deskew -> denoise -> upscale                0.053
denoise -> deskew -> upscale                0.064
... -> upscale -> binarize (Otsu)           0.117
upscale -> deskew -> denoise -> binarize    0.632
==========================================  =====

- **Deskew** finds the rotation that makes text rows sharpest: rotate a downscaled
  binary copy through small angles and keep the one whose row sums vary most (text rows
  dark, gaps light). This is the projection-profile method; it is robust to stamps and
  logos that fool line detection, and searching on a ~1000 px copy keeps it fast.
- **Denoise** is a 3x3 median filter, which removes scanner speckle without blurring
  stroke edges. It must run *before* upscaling: once a one-pixel speck is enlarged to a
  3x3 blob, a 3x3 median no longer removes it.
- **Upscale** (bicubic, whole factor) is requested by the engine for small images.
- **Binarize** uses Otsu's threshold. Off by default: Tesseract already binarizes
  internally, and thresholding the smooth upscaled image first loses stroke detail.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import numpy.typing as npt
from PIL import Image

from ledgerlens.core.config import PreprocessConfig, get_config

Gray = npt.NDArray[np.uint8]

MAX_SKEW_DEGREES = 5.0
SKEW_STEP_DEGREES = 0.25
_SEARCH_SIDE_PX = 1000  # longest side of the copy the skew search runs on


@dataclass(frozen=True)
class Preprocessed:
    image: Image.Image
    skew_degrees: float  # counter-clockwise rotation that was undone (0 if not deskewed)


def to_gray(image: Image.Image) -> Gray:
    return _u8(image.convert("L"))


def _u8(array: object) -> Gray:
    """View an OpenCV / Pillow result as a uint8 array (OpenCV's stubs return ``Mat``)."""
    return np.asarray(array, dtype=np.uint8)


def estimate_skew(gray: Gray, max_degrees: float = MAX_SKEW_DEGREES) -> float:
    """Angle in degrees (counter-clockwise positive) that the text is rotated by."""
    scale = min(1.0, _SEARCH_SIDE_PX / max(gray.shape))
    small = _u8(cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA))
    ink = _u8(cv2.threshold(small, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1])
    if not ink.any():
        return 0.0
    steps = round(max_degrees / SKEW_STEP_DEGREES)
    angles = [i * SKEW_STEP_DEGREES for i in range(-steps, steps + 1)]
    scores = [_row_profile_score(_rotate(ink, -angle, fill=0)) for angle in angles]
    best = max(range(len(angles)), key=lambda i: (scores[i], -abs(angles[i])))
    return angles[best]


def _row_profile_score(ink: Gray) -> float:
    rows = ink.sum(axis=1, dtype=np.float64)
    return float(np.var(rows))


def _rotate(gray: Gray, degrees: float, *, fill: int = 255) -> Gray:
    """Rotate counter-clockwise by ``degrees`` about the centre, keeping the size."""
    h, w = gray.shape
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), degrees, 1.0)
    return _u8(cv2.warpAffine(gray, matrix, (w, h), flags=cv2.INTER_LINEAR, borderValue=(fill,)))


def deskew(gray: Gray, max_degrees: float = MAX_SKEW_DEGREES) -> tuple[Gray, float]:
    angle = estimate_skew(gray, max_degrees)
    if angle == 0:
        return gray, 0.0
    return _rotate(gray, -angle), angle


def denoise(gray: Gray) -> Gray:
    return _u8(cv2.medianBlur(gray, 3))


def binarize(gray: Gray) -> Gray:
    return _u8(cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1])


def upscale(gray: Gray, factor: int) -> Gray:
    if factor <= 1:
        return gray
    return _u8(cv2.resize(gray, None, fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC))


def preprocess(
    image: Image.Image, config: PreprocessConfig | None = None, *, scale: int = 1
) -> Preprocessed:
    """Run the enabled steps on a grayscale copy: deskew, denoise, upscale x ``scale``, binarize.

    The result is ``scale`` times the size of ``image``.
    """
    config = config or get_config().ingestion.ocr.preprocess
    gray = to_gray(image)
    angle = 0.0
    if config.deskew:
        gray, angle = deskew(gray)
    if config.denoise:
        gray = denoise(gray)
    gray = upscale(gray, scale)
    if config.binarize:
        gray = binarize(gray)
    return Preprocessed(Image.fromarray(gray), angle)
