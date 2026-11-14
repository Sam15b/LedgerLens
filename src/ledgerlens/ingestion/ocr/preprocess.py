"""Clean up a page image before OCR: grayscale, denoise, deskew, upscale, binarize.

Each step except upscaling is switched on or off by ``ocr.preprocess`` in
``configs/ingestion.yaml``. Order and settings were measured with real Tesseract
(character error rate, lower is better). First on the speckled receipt fixture:

==========================================  =====
deskew -> denoise -> upscale                0.053
denoise -> deskew -> upscale                0.064
... -> upscale -> binarize (Otsu)           0.117
upscale -> deskew -> denoise -> binarize    0.632
==========================================  =====

then on 40 rendered synthetic invoices (150 dpi) and their classical degradations at
severity 0.25-1.0, 200 page images (``eval/ocr_robustness.py``; full 1,500-image
run in ``eval/reports/2026-11-14/ocr_degraded.md``):

==========================================  =====  =====  =====  =====
corpus CER at severity                      clean  0.50   0.75   1.00
==========================================  =====  =====  =====  =====
deskew -> denoise (always)                  0.047  0.134  0.252  0.442
deskew -> denoise (only if speckled)        0.008  0.134  0.252  0.442
denoise (only if speckled) -> deskew        0.008  0.072  0.200  0.320
==========================================  =====  =====  =====  =====

The last row is the pipeline: it costs the receipt fixture 0.011 CER (still inside its
0.10 budget) and halves the error on degraded pages.

- **Denoise** is a 3x3 median filter, which removes scanner speckle. It runs only when
  the page *has* speckle (:func:`speckle_fraction` at least ``denoise_min_speckle``): on
  a clean 150-dpi page the median eats thin strokes (clean CER 0.008 -> 0.047). It runs
  *before* deskew, because rotating with interpolation smears each one-pixel speck into a
  grey 2x2 blob that a 3x3 median no longer removes -- and before upscaling for the same
  reason.
- **Deskew** finds the rotation that makes text rows sharpest: rotate a downscaled
  binary copy through small angles and keep the one whose row sums vary most (text rows
  dark, gaps light). This is the projection-profile method; it is robust to stamps and
  logos that fool line detection, and searching on a ~1000 px copy keeps it fast.
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


_NEIGHBOURS = np.ones((3, 3), dtype=np.float32)
_NEIGHBOURS[1, 1] = 0


def speckle_fraction(gray: Gray) -> float:
    """Share of pixels that are isolated: dark with no dark neighbour, or light with only
    dark neighbours. Salt-and-pepper noise is made of them; anti-aliased print almost never
    is (clean 150-dpi renders: < 2e-5; 0.5 % pepper noise: > 8e-4)."""
    dark = (gray < 128).astype(np.uint8)
    neighbours = cv2.filter2D(dark, cv2.CV_16S, _NEIGHBOURS, borderType=cv2.BORDER_REPLICATE)
    lone_dark = (dark == 1) & (neighbours == 0)
    lone_light = (dark == 0) & (neighbours == 8)
    return float(lone_dark.mean() + lone_light.mean())


def binarize(gray: Gray) -> Gray:
    return _u8(cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1])


def upscale(gray: Gray, factor: int) -> Gray:
    if factor <= 1:
        return gray
    return _u8(cv2.resize(gray, None, fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC))


def preprocess(
    image: Image.Image, config: PreprocessConfig | None = None, *, scale: int = 1
) -> Preprocessed:
    """Run the enabled steps on a grayscale copy: denoise (if speckled), deskew, upscale x
    ``scale``, binarize.

    The result is ``scale`` times the size of ``image``.
    """
    config = config or get_config().ingestion.ocr.preprocess
    gray = to_gray(image)
    angle = 0.0
    if config.denoise and speckle_fraction(gray) >= config.denoise_min_speckle:
        gray = denoise(gray)
    if config.deskew:
        gray, angle = deskew(gray)
    gray = upscale(gray, scale)
    if config.binarize:
        gray = binarize(gray)
    return Preprocessed(Image.fromarray(gray), angle)
