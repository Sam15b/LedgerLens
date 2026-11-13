"""Classical scan degradations with ``albumentations``: skew, blur, JPEG artifacts, speckle.

    python -m ml.synthetic.diffusion.augment data/synthetic/pages/invoice \\
        --out data/synthetic/degraded/classical --severities 0.25 0.5 0.75 1.0

One knob, ``severity`` in ``[0, 1]``, scales every step together, so a run over several
severities gives a severity -> OCR error curve (Day 25):

==================  ===========================  =====================================
step                at severity 1                why
==================  ===========================  =====================================
rotate              up to +/-3 degrees            pages are fed into scanners crooked
Gaussian blur       sigma 1.5 px (at 150 dpi)    focus / motion in phone captures
JPEG compression    quality 25                   emailed scans are re-compressed
salt-and-pepper     2 % of pixels                dust and scanner noise
==================  ===========================  =====================================

Severity 0 leaves the image untouched. Every variant is seeded, so the same page,
severity and seed always give the same image. Images are grayscale ``uint8`` like the
rendered page images and what the OCR preprocessing expects.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import albumentations as A  # noqa: N812 -- the library's convention
import cv2
import numpy as np
import numpy.typing as npt
from PIL import Image

Gray = npt.NDArray[np.uint8]

MAX_ROTATE_DEGREES = 3.0
MAX_BLUR_SIGMA = 1.5
JPEG_QUALITY = (95, 25)  # at severity 0 and 1
MAX_NOISE = 0.02  # share of pixels flipped to black or white


@dataclass(frozen=True)
class Degradation:
    """The concrete settings for one severity."""

    severity: float
    rotate_degrees: float
    blur_sigma: float
    jpeg_quality: int
    noise_amount: float

    @classmethod
    def at(cls, severity: float) -> Degradation:
        if not 0.0 <= severity <= 1.0:
            raise ValueError(f"severity must be in [0, 1], got {severity}")
        best, worst = JPEG_QUALITY
        return cls(
            severity=severity,
            rotate_degrees=MAX_ROTATE_DEGREES * severity,
            blur_sigma=MAX_BLUR_SIGMA * severity,
            jpeg_quality=round(best - (best - worst) * severity),
            noise_amount=MAX_NOISE * severity,
        )


def build_pipeline(severity: float, *, seed: int = 0) -> A.Compose:
    """An ``albumentations`` pipeline applying every step at ``severity`` (always, p=1)."""
    d = Degradation.at(severity)
    steps: list[Any] = []
    if d.rotate_degrees > 0:
        steps.append(
            A.Rotate(
                limit=(-d.rotate_degrees, d.rotate_degrees),
                border_mode=cv2.BORDER_CONSTANT,
                fill=255,  # paper, not black corners
                p=1.0,
            )
        )
    if d.blur_sigma > 0:
        steps.append(A.GaussianBlur(blur_limit=0, sigma_limit=(d.blur_sigma, d.blur_sigma), p=1.0))
    if severity > 0:
        steps.append(A.ImageCompression(quality_range=(d.jpeg_quality, d.jpeg_quality), p=1.0))
    if d.noise_amount > 0:
        steps.append(A.SaltAndPepper(amount=(d.noise_amount, d.noise_amount), p=1.0))
    return A.Compose(steps, seed=seed)


def to_gray(image: Image.Image | npt.NDArray[Any]) -> Gray:
    if isinstance(image, Image.Image):
        return np.asarray(image.convert("L"), dtype=np.uint8)
    array = np.asarray(image)
    if array.ndim == 3:
        array = cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)
    return np.asarray(array, dtype=np.uint8)


def degrade(image: Image.Image | npt.NDArray[Any], severity: float, *, seed: int = 0) -> Gray:
    """``image`` degraded at ``severity`` (grayscale, same size)."""
    gray = to_gray(image)
    if severity == 0:
        return gray.copy()
    out = build_pipeline(severity, seed=seed)(image=gray)["image"]
    return np.asarray(out, dtype=np.uint8)


@dataclass
class Variant:
    source: str  # clean page image
    image: str  # degraded image
    method: str  # "classical" | "sd_img2img"
    strength: float  # severity (classical) or img2img strength (SD)
    seed: int
    prompt: str | None = None


def variant_name(source: Path, method: str, strength: float, seed: int) -> str:
    tag = "c" if method == "classical" else "sd"
    return f"{source.stem}-{tag}{strength:.2f}-s{seed}.png"


def degrade_files(
    sources: Sequence[Path],
    out: Path,
    severities: Sequence[float],
    *,
    seed: int = 0,
) -> list[Variant]:
    """Every source page at every severity, written under ``out``."""
    out.mkdir(parents=True, exist_ok=True)
    variants = []
    for i, source in enumerate(sources):
        with Image.open(source) as image:
            gray = to_gray(image)
        for severity in severities:
            variant_seed = seed + i
            path = out / variant_name(source, "classical", severity, variant_seed)
            Image.fromarray(degrade(gray, severity, seed=variant_seed)).save(path)
            variants.append(
                Variant(source.as_posix(), path.as_posix(), "classical", severity, variant_seed)
            )
    return variants


def write_manifest(variants: Sequence[Variant], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for variant in variants:
            fh.write(json.dumps(asdict(variant)) + "\n")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Classical scan degradations of page images")
    parser.add_argument("pages", type=Path, help="folder of clean page PNGs")
    parser.add_argument("--out", type=Path, default=Path("data/synthetic/degraded/classical"))
    parser.add_argument("--severities", type=float, nargs="+", default=[0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--limit", type=int, help="at most this many pages")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    sources = sorted(args.pages.glob("*.png"))[: args.limit]
    variants = degrade_files(sources, args.out, args.severities, seed=args.seed)
    write_manifest(variants, args.out / "manifest.jsonl")
    print(f"{len(variants)} degraded images from {len(sources)} pages -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
