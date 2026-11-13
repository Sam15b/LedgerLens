"""Degrade clean page images with Stable Diffusion img2img (GPU; run it on Colab).

    python -m ml.synthetic.diffusion.degrade_scans run data/synthetic/pages/invoice \\
        --out data/synthetic/degraded/sd --strengths 0.2 0.3 0.4 0.45 --limit 200

img2img starts from the clean page plus noise and denoises it toward a prompt such as
"old photocopied invoice, coffee stain, faded toner", so the page picks up paper texture,
stains, uneven toner and low contrast that classical augmentations (``augment.py``) do
not produce.

**Strength** (how much noise is added before denoising) is kept to **0.2-0.45**. Higher
values let the model repaint glyphs: it invents characters, so a "7" can turn into a "1"
and the ground truth is no longer the truth. Even inside the range it sometimes does,
which is why every variant goes through the validity filter (``filter`` below) before use.

Settings (``docs/layers/07-synthetic-data.md``):

* model ``stable-diffusion-v1-5/stable-diffusion-v1-5``, licensed CreativeML OpenRAIL-M,
  which allows generating images for research and commercial work outside its listed
  harmful uses; degrading synthetic invoices is fine. Check the licence before switching
  to another checkpoint (``--model``);
* fp16 on CUDA, 30 denoising steps, guidance 7.5, a negative prompt against new text;
* the page is resized to 768 px wide (a multiple of 8, about SD 1.5's native scale),
  run, and resized back, so outputs line up with the clean page;
* a fixed seed per variant (``torch.Generator``), so runs are reproducible;
* the pipeline's safety checker stays on; a page it blanks out fails the validity filter.

**Validity filter** (``filter``, for classical and SD variants alike): each variant is
OCR'd and kept only if every *key field* printed on the page -- document number, PO
reference, total -- can still be read exactly (case and spacing ignored). Exactly, not
fuzzily: a diffusion model that turns one digit of a total into another has made the label
wrong, and that is precisely the failure to catch. The character error rate against the
clean PDF's text layer is recorded too, and ``--max-cer`` can drop variants above a cap.
"""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

from PIL import Image

from eval.ocr_cer import cer
from ml.synthetic.diffusion.augment import Variant, variant_name, write_manifest
from ml.synthetic.render.render_documents import format_amount, pdf_text

MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"
PROMPTS = (
    "old photocopied invoice, coffee stain, faded toner",
    "crumpled paper scan, low contrast",
)
NEGATIVE_PROMPT = "new text, different words, handwriting, watermark, logo, colorful"
STRENGTHS = (0.2, 0.3, 0.4, 0.45)
STRENGTH_RANGE = (0.2, 0.45)
SD_WIDTH = 768


@dataclass(frozen=True)
class SDConfig:
    model_id: str = MODEL_ID
    steps: int = 30
    guidance: float = 7.5
    width: int = SD_WIDTH
    fp16: bool = True
    device: str = "cuda"


class Img2Img(Protocol):
    """The part of ``diffusers.StableDiffusionImg2ImgPipeline`` this module calls."""

    def __call__(
        self,
        *,
        prompt: str,
        image: Image.Image,
        strength: float,
        num_inference_steps: int,
        guidance_scale: float,
        negative_prompt: str,
        generator: Any,
    ) -> Any: ...


def load_pipeline(config: SDConfig | None = None) -> Img2Img:
    """The img2img pipeline on ``config.device`` (needs ``diffusers`` and a CUDA GPU)."""
    import torch

    config = config or SDConfig()
    from diffusers import StableDiffusionImg2ImgPipeline

    dtype = torch.float16 if config.fp16 else torch.float32
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(config.model_id, torch_dtype=dtype)
    pipe = pipe.to(config.device)
    pipe.set_progress_bar_config(disable=True)
    loaded: Img2Img = pipe
    return loaded


def sd_size(width: int, height: int, target_width: int = SD_WIDTH) -> tuple[int, int]:
    """``(width, height)`` scaled to ``target_width``, both multiples of 8 (SD's latent grid)."""
    scale = target_width / width
    return target_width // 8 * 8, max(8, round(height * scale / 8) * 8)


def check_strength(strength: float) -> float:
    low, high = STRENGTH_RANGE
    if not low <= strength <= high:
        raise ValueError(
            f"img2img strength {strength} is outside {low}-{high}: higher values repaint "
            "glyphs and break the ground truth"
        )
    return strength


def degrade_page(
    pipe: Img2Img,
    image: Image.Image,
    *,
    prompt: str,
    strength: float,
    generator: Any,
    config: SDConfig | None = None,
) -> Image.Image:
    """One img2img variant of ``image``: grayscale, same size as the input."""
    check_strength(strength)
    config = config or SDConfig()
    size = image.size
    rgb = image.convert("RGB").resize(sd_size(*size, config.width), Image.Resampling.LANCZOS)
    result = pipe(
        prompt=prompt,
        image=rgb,
        strength=strength,
        num_inference_steps=config.steps,
        guidance_scale=config.guidance,
        negative_prompt=NEGATIVE_PROMPT,
        generator=generator,
    )
    out: Image.Image = result.images[0]
    return out.convert("L").resize(size, Image.Resampling.LANCZOS)


def torch_generator(device: str) -> Callable[[int], Any]:
    """Seeded ``torch.Generator`` factory for ``device``."""
    import torch

    return lambda seed: torch.Generator(device=device).manual_seed(seed)


def degrade_pages(
    pipe: Img2Img,
    sources: Sequence[Path],
    out: Path,
    *,
    strengths: Sequence[float] = STRENGTHS,
    prompts: Sequence[str] = PROMPTS,
    seed: int = 0,
    make_generator: Callable[[int], Any],
    config: SDConfig | None = None,
) -> list[Variant]:
    """Every source page at every strength; prompts alternate per page."""
    for strength in strengths:
        check_strength(strength)
    out.mkdir(parents=True, exist_ok=True)
    variants = []
    for i, source in enumerate(sources):
        prompt = prompts[i % len(prompts)]
        with Image.open(source) as opened:
            image = opened.convert("L")
        for strength in strengths:
            variant_seed = seed + i
            degraded = degrade_page(
                pipe,
                image,
                prompt=prompt,
                strength=strength,
                generator=make_generator(variant_seed),
                config=config,
            )
            path = out / variant_name(source, "sd_img2img", strength, variant_seed)
            degraded.save(path)
            variants.append(
                Variant(
                    source.as_posix(), path.as_posix(), "sd_img2img", strength, variant_seed, prompt
                )
            )
    return variants


# --------------------------------------------------------------------------- validity filter


def key_fields(record: Mapping[str, Any]) -> dict[str, str]:
    """The values a degraded page must still show, as the templates print them."""
    data = record["data"]
    currency = str(data.get("currency", ""))
    kind = record["record_type"]
    fields: dict[str, str | None] = {}
    if kind == "invoice":
        fields = {
            "invoice_number": data["invoice_number"],
            "po_number": data.get("po_number"),
            "total": format_amount(data["total"], currency),
        }
    elif kind == "purchase_order":
        fields = {"po_number": data["po_number"], "total": format_amount(data["total"], currency)}
    elif kind == "payment":
        fields = {
            "payment_reference": data["payment_reference"],
            "amount": format_amount(data["amount"], currency),
        }
    elif kind == "contract":
        fields = {"contract_number": data.get("contract_number")}
    return {name: str(value) for name, value in fields.items() if value}


_SPACE = re.compile(r"\s+")


def _squash(text: str) -> str:
    return _SPACE.sub("", text).casefold()


def missing_fields(fields: Mapping[str, str], text: str) -> list[str]:
    """Names of ``fields`` whose value does not appear in ``text`` (spacing/case ignored)."""
    haystack = _squash(text)
    return [name for name, value in fields.items() if _squash(value) not in haystack]


@dataclass
class Validity:
    cer: float
    missing: list[str] = field(default_factory=list)
    kept: bool = True


def check_variant(
    image: Path,
    *,
    reference_text: str,
    fields: Mapping[str, str],
    ocr: Callable[[Path], str],
    max_cer: float | None = None,
) -> Validity:
    """OCR ``image`` and decide whether it is still a valid example of its record."""
    text = ocr(image)
    missing = missing_fields(fields, text)
    score = cer(reference_text, text)
    kept = not missing and (max_cer is None or score <= max_cer)
    return Validity(round(score, 4), missing, kept)


def record_id_of(page: Path) -> str:
    """``SINV-000001-p1.png`` -> ``SINV-000001``."""
    return page.stem.rsplit("-p", 1)[0]


def filter_variants(
    variants: Sequence[Mapping[str, Any]],
    records: Mapping[str, Mapping[str, Any]],
    *,
    root: Path,
    ocr: Callable[[Path], str],
    max_cer: float | None = None,
) -> list[dict[str, Any]]:
    """Each manifest entry with its validity (``cer``, ``missing``, ``kept``) added.

    The reference text is the text layer of the PDF the clean page was rendered from
    (``<root>/pdf/<doc type>/<record id>.pdf``).
    """
    references: dict[str, str] = {}
    out = []
    for variant in variants:
        source = Path(variant["source"])
        record_id = record_id_of(source)
        if record_id not in references:
            pdf = root / "pdf" / source.parent.name / f"{record_id}.pdf"
            references[record_id] = pdf_text(pdf.read_bytes())
        validity = check_variant(
            Path(variant["image"]),
            reference_text=references[record_id],
            fields=key_fields(records[record_id]),
            ocr=ocr,
            max_cer=max_cer,
        )
        out.append({**variant, **asdict(validity), "record_id": record_id})
    return out


def tesseract_ocr() -> Callable[[Path], str]:
    from eval.ocr_benchmark import ocr_text
    from ledgerlens.ingestion.ocr import get_engine

    engine = get_engine(name="tesseract")
    return lambda path: ocr_text(engine, path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Stable Diffusion img2img scan degradation")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="degrade page images on a GPU")
    run.add_argument("pages", type=Path, help="folder of clean page PNGs")
    run.add_argument("--out", type=Path, default=Path("data/synthetic/degraded/sd"))
    run.add_argument("--strengths", type=float, nargs="+", default=list(STRENGTHS))
    run.add_argument("--limit", type=int)
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--model", default=MODEL_ID)
    run.add_argument("--steps", type=int, default=SDConfig.steps)
    run.add_argument("--device", default="cuda")
    run.add_argument("--fp32", action="store_true", help="full precision (CPU / old GPUs)")
    check = commands.add_parser("filter", help="OCR variants and drop unreadable ones")
    check.add_argument("manifest", type=Path, help="a degraded set's manifest.jsonl")
    check.add_argument("--root", type=Path, default=Path("data/synthetic"))
    check.add_argument("--records", type=Path, help="default: <root>/records.jsonl")
    check.add_argument("--max-cer", type=float, help="also drop variants above this CER")
    args = parser.parse_args(argv)

    if args.command == "filter":
        return _filter(args)
    config = SDConfig(model_id=args.model, steps=args.steps, fp16=not args.fp32, device=args.device)
    sources = sorted(args.pages.glob("*.png"))[: args.limit]
    variants = degrade_pages(
        load_pipeline(config),
        sources,
        args.out,
        strengths=args.strengths,
        seed=args.seed,
        make_generator=torch_generator(config.device),
        config=config,
    )
    write_manifest(variants, args.out / "manifest.jsonl")
    print(f"{len(variants)} img2img variants from {len(sources)} pages -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


def _filter(args: argparse.Namespace) -> int:
    from ml.synthetic.build import read_records

    lines = args.manifest.read_text(encoding="utf-8").splitlines()
    variants = [json.loads(line) for line in lines if line]
    records = {r["id"]: r for r in read_records(args.records or args.root / "records.jsonl")}
    checked = filter_variants(
        variants, records, root=args.root, ocr=tesseract_ocr(), max_cer=args.max_cer
    )
    out = args.manifest.with_name("validity.jsonl")
    out.write_text("".join(json.dumps(v) + "\n" for v in checked), encoding="utf-8")
    kept = sum(v["kept"] for v in checked)
    print(f"kept {kept} / {len(checked)} variants -> {out}")
    return 0
