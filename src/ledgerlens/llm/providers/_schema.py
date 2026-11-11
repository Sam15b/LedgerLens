"""JSON-schema rewriting for providers with a strict structured-output mode.

Pydantic writes optional fields as non-required properties with a ``default`` and leaves
objects open. OpenAI, Mistral and Bedrock's strict modes want every object closed
(``additionalProperties: false``) with all of its properties required; an optional field
stays optional because Pydantic already writes ``X | None`` as ``anyOf [X, null]``.
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Collection
from typing import Any

SchemaNode = dict[str, Any]

# Keywords whose value is a single sub-schema / a list of them / a name -> schema mapping.
_SUBSCHEMA_KEYS = ("items", "additionalProperties", "not", "contains")
_SUBSCHEMA_LIST_KEYS = ("anyOf", "oneOf", "allOf", "prefixItems")
_SUBSCHEMA_MAP_KEYS = ("properties", "$defs", "definitions")


def walk_schema(node: SchemaNode, visit: Callable[[SchemaNode], None]) -> None:
    """Call ``visit`` on ``node`` and then on every sub-schema below it, in place.

    Property *names* are never visited, only the schemas they map to, so a field called
    ``pattern`` or ``default`` is safe.
    """
    visit(node)
    for key in _SUBSCHEMA_KEYS:
        if isinstance(node.get(key), dict):
            walk_schema(node[key], visit)
    for key in _SUBSCHEMA_LIST_KEYS:
        for sub in node.get(key) or []:
            if isinstance(sub, dict):
                walk_schema(sub, visit)
    for key in _SUBSCHEMA_MAP_KEYS:
        for sub in (node.get(key) or {}).values():
            if isinstance(sub, dict):
                walk_schema(sub, visit)


def strict_json_schema(schema: SchemaNode, *, drop_keywords: Collection[str] = ()) -> SchemaNode:
    """Return a closed, all-required copy of ``schema`` without ``default`` keywords.

    ``drop_keywords`` removes further keywords a provider rejects (e.g. ``minimum``).
    """
    dropped = {"default", *drop_keywords}

    def visit(node: SchemaNode) -> None:
        for keyword in dropped:
            node.pop(keyword, None)
        if node.get("type") == "object" and "properties" in node:
            node["additionalProperties"] = False
            node["required"] = list(node["properties"])

    strict = copy.deepcopy(schema)
    walk_schema(strict, visit)
    return strict
