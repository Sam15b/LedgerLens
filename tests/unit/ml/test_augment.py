"""Scan degradation: classical augmentations for real, Stable Diffusion with a fake pipeline
(no diffusers or GPU in CI), and the validity filter with a fake OCR engine."""

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from PIL import Image, ImageDraw

from ml.synthetic.diffusion import augment
from ml.synthetic.diffusion.augment import Degradation, degrade, degrade_files, to_gray
from ml.synthetic.diffusion.degrade_scans import (
    NEGATIVE_PROMPT,
    PROMPTS,
    check_variant,
    degrade_pages,
    filter_variants,
    key_fields,
    missing_fields,
    record_id_of,
    sd_size,
)


def page(width: int = 400, height: int = 300) -> Image.Image:
    """A white page with black text-like bars."""
    image = Image.new("L", (width, height), 255)
    draw = ImageDraw.Draw(image)
    for y in range(30, height - 30, 24):
        draw.rectangle((30, y, width - 60, y + 10), fill=0)
    return image


# --------------------------------------------------------------------------- classical


def test_severity_scales_every_step() -> None:
    zero, half, full = Degradation.at(0), Degradation.at(0.5), Degradation.at(1)

    assert (zero.rotate_degrees, zero.blur_sigma, zero.noise_amount) == (0, 0, 0)
    assert zero.jpeg_quality == 95 and full.jpeg_quality == 25
    assert half.rotate_degrees == pytest.approx(1.5) and full.rotate_degrees == 3.0
    assert full.blur_sigma == 1.5 and full.noise_amount == 0.02
    with pytest.raises(ValueError):
        Degradation.at(1.5)


def test_severity_zero_is_the_identity() -> None:
    gray = to_gray(page())

    assert np.array_equal(degrade(gray, 0.0), gray)


def test_degradation_keeps_size_and_type_and_is_seeded() -> None:
    gray = to_gray(page())

    first = degrade(gray, 0.7, seed=3)

    assert first.shape == gray.shape and first.dtype == np.uint8
    assert np.array_equal(degrade(gray, 0.7, seed=3), first)
    assert not np.array_equal(degrade(gray, 0.7, seed=4), first)
    assert not np.array_equal(first, gray)


def test_more_severity_means_more_damage() -> None:
    gray = to_gray(page()).astype(np.int16)

    damage = [np.abs(degrade(gray.astype(np.uint8), s, seed=1) - gray).mean() for s in (0.2, 1.0)]

    assert damage[0] < damage[1]


def test_full_severity_speckles_about_two_percent_of_a_blank_page() -> None:
    blank = np.full((200, 200), 255, dtype=np.uint8)

    noisy = degrade(blank, 1.0, seed=0)

    dark = (noisy < 100).mean()  # pepper on a white page; corners are filled white
    assert 0.003 < dark < 0.03


def test_to_gray_accepts_rgb_arrays_and_images() -> None:
    rgb = np.zeros((10, 12, 3), dtype=np.uint8)

    assert to_gray(rgb).shape == (10, 12)
    assert to_gray(Image.new("RGB", (12, 10))).shape == (10, 12)


def test_degrade_files_writes_every_severity_with_a_manifest(tmp_path: Path) -> None:
    sources = []
    for name in ("SINV-000001-p1.png", "SINV-000002-p1.png"):
        page().save(tmp_path / name)
        sources.append(tmp_path / name)

    variants = degrade_files(sources, tmp_path / "out", [0.25, 1.0], seed=10)
    augment.write_manifest(variants, tmp_path / "out" / "manifest.jsonl")

    assert [Path(v.image).name for v in variants] == [
        "SINV-000001-p1-c0.25-s10.png",
        "SINV-000001-p1-c1.00-s10.png",
        "SINV-000002-p1-c0.25-s11.png",
        "SINV-000002-p1-c1.00-s11.png",
    ]
    assert all(Image.open(v.image).size == (400, 300) for v in variants)
    lines = (tmp_path / "out" / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
    assert json.loads(lines[0])["method"] == "classical" and len(lines) == 4


def test_cli(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    page().save(tmp_path / "SINV-000001-p1.png")

    code = augment.main([str(tmp_path), "--out", str(tmp_path / "d"), "--severities", "0.5"])

    assert code == 0 and "1 degraded images from 1 pages" in capsys.readouterr().out


# --------------------------------------------------------------------------- Stable Diffusion


class FakeImg2Img:
    """Records the calls; returns the input darkened a little."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def __call__(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        image: Image.Image = kwargs["image"]
        return SimpleNamespace(images=[image.point(lambda v: max(0, v - 20))])


def test_sd_size_is_on_the_latent_grid() -> None:
    assert sd_size(1240, 1754) == (768, 1088)
    assert sd_size(400, 300) == (768, 576)
    assert all(v % 8 == 0 for v in sd_size(1001, 333))


def test_img2img_variants_keep_page_size_strength_and_seed(tmp_path: Path) -> None:
    sources = []
    for name in ("a-p1.png", "b-p1.png"):
        page(1240, 1754).save(tmp_path / name)
        sources.append(tmp_path / name)
    pipe = FakeImg2Img()

    variants = degrade_pages(
        pipe,
        sources,
        tmp_path / "sd",
        strengths=[0.2, 0.45],
        seed=5,
        make_generator=lambda seed: f"generator-{seed}",
    )

    assert len(variants) == 4
    assert [c["strength"] for c in pipe.calls] == [0.2, 0.45, 0.2, 0.45]
    assert [c["generator"] for c in pipe.calls] == ["generator-5"] * 2 + ["generator-6"] * 2
    assert [c["prompt"] for c in pipe.calls] == [PROMPTS[0]] * 2 + [PROMPTS[1]] * 2
    assert all(c["image"].size == (768, 1088) and c["image"].mode == "RGB" for c in pipe.calls)
    assert all(c["negative_prompt"] == NEGATIVE_PROMPT for c in pipe.calls)
    for variant in variants:
        with Image.open(variant.image) as out:
            assert out.size == (1240, 1754) and out.mode == "L"
    assert variants[1].method == "sd_img2img" and variants[1].prompt == PROMPTS[0]


def test_img2img_rejects_strengths_that_repaint_glyphs(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="repaint glyphs"):
        degrade_pages(FakeImg2Img(), [], tmp_path, strengths=[0.6], make_generator=str)


# --------------------------------------------------------------------------- validity filter


INVOICE: dict[str, Any] = {
    "record_type": "invoice",
    "id": "SINV-000001",
    "data": {
        "invoice_number": "INV-006031",
        "po_number": "PO-2024-00001",
        "currency": "SGD",
        "total": "1540.03",
    },
}
TEXT = "Tax Invoice\nInvoice # INV-006031\nYour order PO-2024-00001\nAmount due (SGD) 1,540.03"


def test_key_fields_are_formatted_as_printed() -> None:
    assert key_fields(INVOICE) == {
        "invoice_number": "INV-006031",
        "po_number": "PO-2024-00001",
        "total": "1,540.03",
    }
    payment = {
        "record_type": "payment",
        "data": {"payment_reference": "PAY-1", "amount": "1234500", "currency": "IDR"},
    }
    assert key_fields(payment) == {"payment_reference": "PAY-1", "amount": "1,234,500"}
    unknown = {"record_type": "invoice", "data": {**INVOICE["data"], "po_number": None}}
    assert "po_number" not in key_fields(unknown)


def test_fields_must_be_read_exactly_but_spacing_and_case_are_ignored() -> None:
    fields = key_fields(INVOICE)

    assert missing_fields(fields, TEXT) == []
    assert missing_fields(fields, TEXT.replace("1,540.03", "1, 540 .03").lower()) == []
    assert missing_fields(fields, TEXT.replace("1,540.03", "1,546.03")) == ["total"]
    assert missing_fields(fields, "") == ["invoice_number", "po_number", "total"]


def test_check_variant_keeps_readable_pages_and_caps_cer(tmp_path: Path) -> None:
    image = tmp_path / "x.png"
    fields = key_fields(INVOICE)

    clean = check_variant(image, reference_text=TEXT, fields=fields, ocr=lambda _: TEXT)
    wrong_digit = check_variant(
        image, reference_text=TEXT, fields=fields, ocr=lambda _: TEXT.replace("6031", "6081")
    )
    noisy = check_variant(
        image, reference_text=TEXT, fields=fields, ocr=lambda _: TEXT + " ~~~~", max_cer=0.01
    )

    assert clean.kept and clean.cer == 0.0
    assert not wrong_digit.kept and wrong_digit.missing == ["invoice_number"]
    assert wrong_digit.cer < 0.05  # one digit barely moves CER -- the reason for the filter
    assert not noisy.kept and noisy.missing == []


def test_filter_variants_reads_the_reference_from_the_rendered_pdf(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from ml.synthetic.diffusion import degrade_scans

    (tmp_path / "pdf" / "invoice").mkdir(parents=True)
    (tmp_path / "pdf" / "invoice" / "SINV-000001.pdf").write_bytes(b"%PDF")
    monkeypatch.setattr(degrade_scans, "pdf_text", lambda data: TEXT)
    variants = [
        {"source": "pages/invoice/SINV-000001-p1.png", "image": "a.png", "strength": 0.2},
        {"source": "pages/invoice/SINV-000001-p1.png", "image": "b.png", "strength": 0.45},
    ]
    ocr = {"a.png": TEXT, "b.png": "smudge"}

    checked = filter_variants(
        variants, {"SINV-000001": INVOICE}, root=tmp_path, ocr=lambda p: ocr[p.name]
    )

    assert [c["kept"] for c in checked] == [True, False]
    assert checked[1]["missing"] == ["invoice_number", "po_number", "total"]
    assert checked[0]["record_id"] == "SINV-000001" and checked[0]["strength"] == 0.2
    assert record_id_of(Path("SINV-000123-p2.png")) == "SINV-000123"
