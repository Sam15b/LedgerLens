"""Run the fine-tuned extractor: Llama 3 (4-bit) plus a LoRA adapter, text in, JSON out.

    python -m ml.finetune.infer --adapter $DRIVE/ledgerlens/finetune/run-1/adapter \\
        --doc-type receipt --text-file receipt.txt

The prompt is the one the adapter was trained on
(:func:`~ml.finetune.prepare_dataset.chat_messages`: the system rules, the task naming the
keys, the document text), decoding is greedy, and the reply is parsed as one JSON object
(:func:`parse_json_object`). Without ``--adapter`` it runs the base model with the same
prompt -- the "base Llama 3" row of the comparison.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ml.finetune.prepare_dataset import (
    CORD_KEYS,
    DEFAULT_BASE_MODEL,
    INVOICE_KEYS,
    RECEIPT_KEYS,
    Example,
    _ids,
    chat_messages,
)
from ml.finetune.train_lora import TrainConfig, load_tokenizer, quantization_config, torch_dtype

logger = logging.getLogger(__name__)

KEYS_BY_DOC_TYPE: dict[str, tuple[str, ...]] = {
    "receipt": (*RECEIPT_KEYS[:-1], *CORD_KEYS),  # vendor, address, date, lines, totals
    "invoice": INVOICE_KEYS,
}
DEFAULT_MAX_NEW_TOKENS = 768


def prompt_example(text: str, doc_type: str, keys: Sequence[str] | None = None) -> Example:
    """An unlabelled example for ``text``: the prompt asks for ``keys`` (default: every
    key the training data uses for ``doc_type``)."""
    wanted = list(keys or KEYS_BY_DOC_TYPE[doc_type])
    return Example("input", "input", "", doc_type, text, dict.fromkeys(wanted))


def parse_json_object(text: str) -> dict[str, Any] | None:
    """The first JSON object in a model's reply (code fences and chatter around it are
    ignored), or ``None`` if there is none."""
    decoder = json.JSONDecoder()
    start = text.find("{")
    while start != -1:
        try:
            value, _ = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            start = text.find("{", start + 1)
            continue
        return value if isinstance(value, dict) else None
    return None


@dataclass(frozen=True)
class Generation:
    """One reply and what it cost."""

    text: str
    prompt_tokens: int
    completion_tokens: int
    seconds: float

    @property
    def fields(self) -> dict[str, Any] | None:
        return parse_json_object(self.text)


class AdapterExtractor:
    """The base model with (or without) a LoRA adapter, loaded once on first use."""

    def __init__(
        self,
        adapter: str | Path | None = None,
        *,
        base_model: str = DEFAULT_BASE_MODEL,
        load_in_4bit: bool = True,
        compute_dtype: str = "float16",
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    ) -> None:
        self.adapter = str(adapter) if adapter else None
        self.config = TrainConfig(
            base_model=base_model, load_in_4bit=load_in_4bit, compute_dtype=compute_dtype
        )
        self.max_new_tokens = max_new_tokens
        self._model: Any = None
        self._tokenizer: Any = None

    @property
    def name(self) -> str:
        return f"{self.config.base_model}+lora" if self.adapter else self.config.base_model

    def load(self) -> tuple[Any, Any]:
        if self._model is None:
            from transformers import AutoModelForCausalLM

            self._tokenizer = load_tokenizer(self.config)
            model: Any = AutoModelForCausalLM.from_pretrained(
                self.config.base_model,
                quantization_config=quantization_config(self.config),
                torch_dtype=torch_dtype(self.config.compute_dtype),
                device_map="auto",
            )
            if self.adapter:
                from peft import PeftModel

                model = PeftModel.from_pretrained(model, self.adapter)
            model.eval()
            self._model = model
        return self._model, self._tokenizer

    def stop_ids(self, tokenizer: Any) -> list[int]:
        """End-of-sequence and Llama 3's end-of-turn token: a reply stops at either."""
        ids = {tokenizer.eos_token_id}
        eot = tokenizer.convert_tokens_to_ids("<|eot_id|>")
        if isinstance(eot, int) and eot != tokenizer.unk_token_id:
            ids.add(eot)
        return sorted(i for i in ids if i is not None)

    def generate(self, example: Example) -> Generation:
        """Greedy reply to ``example``'s prompt (its answer, if any, is not shown)."""
        import torch

        model, tokenizer = self.load()
        prompt = _ids(
            tokenizer.apply_chat_template(
                chat_messages(example, with_answer=False),
                tokenize=True,
                add_generation_prompt=True,
            )
        )
        inputs = torch.tensor([prompt], device=model.device)
        start = time.perf_counter()
        with torch.no_grad():
            output = model.generate(
                input_ids=inputs,
                attention_mask=torch.ones_like(inputs),
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                eos_token_id=self.stop_ids(tokenizer),
                pad_token_id=tokenizer.pad_token_id,
            )
        seconds = time.perf_counter() - start
        reply = output[0][len(prompt) :]
        return Generation(
            text=tokenizer.decode(reply, skip_special_tokens=True).strip(),
            prompt_tokens=len(prompt),
            completion_tokens=len(reply),
            seconds=seconds,
        )

    def extract(
        self, text: str, doc_type: str, keys: Sequence[str] | None = None
    ) -> tuple[dict[str, Any] | None, Generation]:
        """The fields in ``text`` (``None`` if the reply is not a JSON object)."""
        generation = self.generate(prompt_example(text, doc_type, keys))
        return generation.fields, generation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m ml.finetune.infer", description="Extract fields with base + adapter."
    )
    parser.add_argument("--adapter", default=None, help="adapter folder (omit: base model)")
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--doc-type", choices=sorted(KEYS_BY_DOC_TYPE), default="receipt")
    parser.add_argument("--text-file", type=Path, required=True)
    parser.add_argument("--no-4bit", action="store_true", help="load unquantized (CPU tests)")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    extractor = AdapterExtractor(
        args.adapter,
        base_model=args.base_model,
        load_in_4bit=not args.no_4bit,
        compute_dtype=args.dtype,
        max_new_tokens=args.max_new_tokens,
    )
    fields, generation = extractor.extract(
        args.text_file.read_text(encoding="utf-8"), args.doc_type
    )
    print(json.dumps(fields, indent=2) if fields is not None else generation.text)
    logger.info(
        "%d prompt + %d completion tokens in %.2f s",
        generation.prompt_tokens,
        generation.completion_tokens,
        generation.seconds,
    )
    return 0 if fields is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
