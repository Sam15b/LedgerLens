import numpy as np
import pytest
from PIL import Image, ImageDraw

from ledgerlens.core.config import PreprocessConfig
from ledgerlens.ingestion.ocr.preprocess import (
    binarize,
    denoise,
    deskew,
    despeckle,
    estimate_skew,
    preprocess,
    speckle_fraction,
    to_gray,
)


def text_page(width: int = 600, height: int = 400) -> Image.Image:
    """White page with rows of black 'text' bars, like lines of print."""
    image = Image.new("L", (width, height), 255)
    draw = ImageDraw.Draw(image)
    for y in range(40, height - 40, 30):
        for x in range(40, width - 120, 70):
            draw.rectangle((x, y, x + 50, y + 10), fill=0)
    return image


@pytest.mark.parametrize("degrees", [-3.0, -1.5, 0.0, 2.0, 4.0])
def test_estimate_skew_finds_the_rotation(degrees: float) -> None:
    tilted = text_page().rotate(degrees, resample=Image.Resampling.BICUBIC, fillcolor=255)

    assert estimate_skew(to_gray(tilted)) == pytest.approx(degrees, abs=0.5)


def test_deskew_straightens_text_rows() -> None:
    tilted = to_gray(text_page().rotate(3.0, resample=Image.Resampling.BICUBIC, fillcolor=255))

    straight, angle = deskew(tilted)

    assert angle == pytest.approx(3.0, abs=0.5)
    assert abs(estimate_skew(straight)) <= 0.5
    assert straight.shape == tilted.shape


def test_blank_page_is_not_rotated() -> None:
    blank = np.full((100, 100), 255, dtype=np.uint8)

    assert deskew(blank) == (blank, 0.0)


def test_denoise_removes_isolated_speckle() -> None:
    page = np.full((50, 50), 255, dtype=np.uint8)
    page[10, 10] = page[30, 40] = 0

    assert int(np.min(denoise(page))) == 255


def test_binarize_returns_only_black_and_white() -> None:
    gradient = np.tile(np.arange(256, dtype=np.uint8), (8, 1))

    assert set(np.unique(binarize(gradient)).tolist()) == {0, 255}


def test_preprocess_runs_enabled_steps_only() -> None:
    tilted = text_page().rotate(2.0, resample=Image.Resampling.BICUBIC, fillcolor=255)

    everything = preprocess(tilted, PreprocessConfig(binarize=True), scale=2)
    nothing = preprocess(tilted, PreprocessConfig(deskew=False, denoise=False, binarize=False))

    assert everything.skew_degrees == pytest.approx(2.0, abs=0.5)
    assert everything.image.size == (1200, 800)
    assert set(np.unique(np.asarray(everything.image)).tolist()) <= {0, 255}
    assert nothing.skew_degrees == 0.0
    assert np.array_equal(np.asarray(nothing.image), to_gray(tilted))


def test_speckle_fraction_separates_noise_from_print() -> None:
    clean = to_gray(text_page())
    rng = np.random.default_rng(0)
    noisy = clean.copy()
    flip = rng.random(noisy.shape) < 0.005
    noisy[flip] = 255 - noisy[flip]

    assert speckle_fraction(clean) < 1e-4 <= speckle_fraction(noisy)


def test_clean_pages_are_not_median_filtered() -> None:
    page = np.full((60, 60), 255, dtype=np.uint8)
    page[10:50, 30] = 0  # a one-pixel-wide stroke, as small print renders at 150 dpi

    kept = preprocess(Image.fromarray(page), PreprocessConfig(deskew=False))
    forced = preprocess(
        Image.fromarray(page),
        PreprocessConfig(deskew=False, denoise_min_speckle=0, median_min_speckle=0),
    )

    assert np.array_equal(np.asarray(kept.image), page)  # stroke survives
    assert np.asarray(forced.image).min() == 255  # an unconditional median erases it


def test_speckle_is_removed_before_deskew_can_smear_it() -> None:
    tilted = np.asarray(text_page().rotate(2.0, fillcolor=255), dtype=np.uint8).copy()
    tilted[5:30:6, 5:590:12] = 0  # 250 isolated specks in the top margin

    cleaned = np.asarray(preprocess(Image.fromarray(tilted), PreprocessConfig()).image)

    assert cleaned[:30, :].min() == 255  # deskew-then-denoise leaves grey smudges


def test_despeckle_removes_isolated_pixels_and_leaves_strokes_alone() -> None:
    page = np.full((60, 60), 255, dtype=np.uint8)
    page[10:50, 30] = 0  # a one-pixel-wide stroke
    page[5, 5] = page[55, 50] = 0  # pepper
    page[20:23, 40:43] = 0
    page[21, 41] = 255  # salt inside a dark blob

    cleaned = despeckle(page)

    assert (cleaned[10:50, 30] == 0).all()  # the stroke is untouched
    assert cleaned[5, 5] == cleaned[55, 50] == 255
    assert cleaned[21, 41] == 0


def test_light_speckle_is_despeckled_heavy_speckle_gets_the_median() -> None:
    page = np.full((100, 100), 255, dtype=np.uint8)
    page[10:90, 50] = 0  # a thin stroke
    page[3, 3] = 0  # one speck: 1e-4 of the pixels

    def run(median_min_speckle: float) -> np.ndarray:
        config = PreprocessConfig(deskew=False, median_min_speckle=median_min_speckle)
        return np.asarray(preprocess(Image.fromarray(page), config).image)

    light, heavy = run(0.005), run(1e-4)

    assert light[3, 3] == 255 and (light[10:90, 50] == 0).all()
    assert heavy[3, 3] == 255 and (heavy[10:90, 50] == 255).all()  # the median erases it


def test_binarization_is_off_by_default() -> None:
    # Tesseract binarizes internally; thresholding first measured worse (see preprocess.py).
    assert not PreprocessConfig().binarize


def test_denoise_runs_before_upscaling() -> None:
    page = np.full((40, 40), 255, dtype=np.uint8)
    page[20, 20] = 0  # one-pixel speck

    cleaned = preprocess(Image.fromarray(page), PreprocessConfig(deskew=False), scale=3)

    assert np.asarray(cleaned.image).min() == 255  # removed, not enlarged into a blob
