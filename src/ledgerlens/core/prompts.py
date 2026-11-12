"""Versioned prompt templates, stored as markdown in ``ledgerlens/agents/prompts/``.

Each file starts with a YAML front-matter block (``name``, ``version``, ``description``,
``variables``) followed by the template text. Placeholders are ``$name`` (Python
:class:`string.Template`), so JSON examples and braces in a prompt need no escaping; write
a literal dollar sign as ``$$``.

Bump ``version`` whenever the wording changes: it is logged with every call that uses the
prompt, so evaluation results can be tied to the exact prompt that produced them.
"""

from __future__ import annotations

import functools
import string
from dataclasses import dataclass, field
from importlib import resources

import yaml

PROMPTS_PACKAGE = "ledgerlens.agents"
PROMPTS_DIR = "prompts"


class PromptError(ValueError):
    pass


@dataclass(frozen=True)
class Prompt:
    name: str
    version: int
    template: str
    description: str = ""
    variables: tuple[str, ...] = field(default_factory=tuple)

    @property
    def id(self) -> str:
        """``name@vN``, for logs and eval reports."""
        return f"{self.name}@v{self.version}"

    def render(self, **values: object) -> str:
        """Fill every ``$variable``; a missing one is an error, not a silent blank."""
        missing = [name for name in self.variables if name not in values]
        if missing:
            raise PromptError(f"prompt {self.id} needs {', '.join(missing)}")
        try:
            return string.Template(self.template).substitute({k: str(v) for k, v in values.items()})
        except (KeyError, ValueError) as exc:
            raise PromptError(
                f"prompt {self.id} has an undeclared or malformed placeholder: {exc}"
            ) from None


def parse_prompt(text: str, *, source: str = "<prompt>") -> Prompt:
    """Split front matter from the template and validate it."""
    if not text.startswith("---"):
        raise PromptError(f"{source}: missing '---' front matter")
    try:
        _, header, body = text.split("---", 2)
    except ValueError:
        raise PromptError(f"{source}: unterminated front matter") from None
    meta = yaml.safe_load(header) or {}
    if not isinstance(meta, dict) or "name" not in meta or "version" not in meta:
        raise PromptError(f"{source}: front matter needs 'name' and 'version'")
    return Prompt(
        name=str(meta["name"]),
        version=int(meta["version"]),
        template=body.strip() + "\n",
        description=str(meta.get("description", "")),
        variables=tuple(str(v) for v in meta.get("variables", [])),
    )


@functools.lru_cache(maxsize=32)
def load_prompt(name: str) -> Prompt:
    """Load ``agents/prompts/<name>.md`` (cached)."""
    path = resources.files(PROMPTS_PACKAGE).joinpath(PROMPTS_DIR, f"{name}.md")
    if not path.is_file():
        raise PromptError(f"no prompt named {name!r} in {PROMPTS_PACKAGE}/{PROMPTS_DIR}")
    prompt = parse_prompt(path.read_text(encoding="utf-8"), source=f"{name}.md")
    if prompt.name != name:
        raise PromptError(f"{name}.md declares name {prompt.name!r}")
    return prompt
