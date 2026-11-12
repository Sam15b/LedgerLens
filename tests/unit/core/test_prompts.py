import pytest

from ledgerlens.core.prompts import PromptError, load_prompt, parse_prompt

TEMPLATE = """---
name: greet
version: 3
variables: [who]
---
Hello $who, that costs $$5.
"""


def test_extraction_prompt_is_versioned_and_renders() -> None:
    prompt = load_prompt("extract_invoice")

    text = prompt.render(doc_type="receipt", today="2026-11-12")

    assert prompt.id == "extract_invoice@v1"
    assert prompt.variables == ("doc_type", "today")
    assert "The document is a **receipt**" in text and "Today is 2026-11-12" in text
    assert 'with only "$" and nothing else' in text


def test_parse_and_render() -> None:
    prompt = parse_prompt(TEMPLATE)

    assert (prompt.name, prompt.version) == ("greet", 3)
    assert prompt.render(who="Ana") == "Hello Ana, that costs $5.\n"


@pytest.mark.parametrize(
    "text,message",
    [
        ("no front matter", "missing '---' front matter"),
        ("---\nname: x\n", "unterminated front matter"),
        ("---\nname: x\n---\nbody", "needs 'name' and 'version'"),
    ],
)
def test_malformed_prompt_files(text: str, message: str) -> None:
    with pytest.raises(PromptError, match=message):
        parse_prompt(text)


def test_missing_and_undeclared_variables() -> None:
    prompt = parse_prompt(TEMPLATE)
    undeclared = parse_prompt("---\nname: x\nversion: 1\n---\nHi $name")

    with pytest.raises(PromptError, match="needs who"):
        prompt.render()
    with pytest.raises(PromptError, match="undeclared or malformed placeholder"):
        undeclared.render()


def test_unknown_prompt() -> None:
    with pytest.raises(PromptError, match="no prompt named 'nope'"):
        load_prompt("nope")
