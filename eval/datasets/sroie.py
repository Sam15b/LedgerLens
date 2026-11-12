"""SROIE (ICDAR 2019 scanned receipts): images plus ground-truth text and key fields.

SROIE needs a manual download (https://rrc.cvc.uab.es/?ch=13, registration required).
Unpack it under ``data/raw/sroie`` in either common layout:

    data/raw/sroie/{train,test}/img/X51005200931.jpg
                               /box/X51005200931.txt        (task 1: text lines + boxes)
                               /entities/X51005200931.txt   (task 3: key fields, JSON)

or the competition's flat folders, with each image next to its ``.txt`` box file
(e.g. ``0325updated.task1train(626p)/``); point ``root`` at that folder.

Box files hold one line per text line: ``x1,y1,x2,y2,x3,y3,x4,y4,TEXT`` (the text may
itself contain commas). Entity files are JSON with ``company``, ``date``, ``address`` and
``total``. Some files in the release are not valid UTF-8, so they are read leniently.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")
FIELDS = ("company", "date", "address", "total")


@dataclass(frozen=True)
class SroieLine:
    text: str
    polygon: tuple[tuple[int, int], ...]  # four corners, clockwise from top-left


@dataclass(frozen=True)
class SroieSample:
    sample_id: str
    image_path: Path
    lines: list[SroieLine]
    fields: dict[str, str] = field(default_factory=dict)

    @property
    def text(self) -> str:
        """Ground-truth text, one line per annotated line, in annotation order."""
        return "\n".join(line.text for line in self.lines)


def read_text(path: Path) -> str:
    """Read a SROIE annotation file; a few are Latin-1 rather than UTF-8."""
    data = path.read_bytes()
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        return data.decode("latin-1")


def parse_box_file(text: str) -> list[SroieLine]:
    """Parse ``x1,y1,...,x4,y4,TEXT`` lines; malformed lines are skipped."""
    lines: list[SroieLine] = []
    for raw in text.splitlines():
        parts = raw.strip().split(",", 8)
        if len(parts) < 9:
            continue
        try:
            coords = [int(float(v)) for v in parts[:8]]
        except ValueError:
            continue
        polygon = tuple((coords[i], coords[i + 1]) for i in range(0, 8, 2))
        lines.append(SroieLine(text=parts[8].strip(), polygon=polygon))
    return lines


def parse_entities(text: str) -> dict[str, str]:
    """Key fields from an entities file; missing fields are left out."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return {}
    if not isinstance(data, dict):
        return {}
    return {k: str(data[k]).strip() for k in FIELDS if data.get(k) not in (None, "")}


def _split_dirs(root: Path, split: str) -> tuple[Path, Path, Path | None] | None:
    base = root / split
    if (base / "img").is_dir() and (base / "box").is_dir():
        entities = base / "entities"
        return base / "img", base / "box", entities if entities.is_dir() else None
    return None


def iter_sroie(
    root: Path | str, split: str = "test", limit: int | None = None
) -> Iterator[SroieSample]:
    """Samples from ``root`` (sorted by id, so ``limit`` always picks the same receipts).

    Images without a box file are skipped: they have no ground truth to score against.
    """
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(
            f"SROIE not found at {root}; download it from https://rrc.cvc.uab.es/?ch=13 "
            "(see eval/datasets/sroie.py for the expected layout)"
        )
    dirs = _split_dirs(root, split)
    if dirs is not None:
        image_dir, box_dir, entity_dir = dirs
    else:  # flat layout: images and box files side by side
        image_dir = box_dir = root
        entity_dir = None
    images = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    count = 0
    for image in images:
        if limit is not None and count >= limit:
            return
        box = box_dir / f"{image.stem}.txt"
        if not box.is_file():
            continue
        fields: dict[str, str] = {}
        if entity_dir is not None and (entity_dir / box.name).is_file():
            fields = parse_entities(read_text(entity_dir / box.name))
        yield SroieSample(image.stem, image, parse_box_file(read_text(box)), fields)
        count += 1
