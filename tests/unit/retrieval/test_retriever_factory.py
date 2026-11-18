"""The retriever factory: variants, sizes from the config, the default, shared parts."""

from typing import Any

import pytest

from ledgerlens.core.config import AppConfig, load_config
from ledgerlens.retrieval.factory import RetrieverFactory, build_retriever
from ledgerlens.retrieval.graphrag import GraphRAGRetriever, Text2CypherRetriever
from ledgerlens.retrieval.vector import DenseRetriever, HybridRetriever, RerankRetriever


def config(**retrieval: Any) -> AppConfig:
    return AppConfig.model_validate(
        {
            "pinecone": {"hybrid_alpha": 0.6, "query_top_k": 40},
            "retrieval": {"graph": {"max_hops": 1, "per_seed": 4, "limit": 12}} | retrieval,
        }
    )


def factory(cfg: AppConfig | None = None, **kwargs: Any) -> RetrieverFactory:
    """A factory whose live parts (store, models, Postgres, Neo4j) are stand-ins."""
    built = RetrieverFactory(config=cfg or config(), store=object(), **kwargs)  # type: ignore[arg-type]
    for part in ("embedder", "sparse", "texts", "driver", "analyzer"):
        built.__dict__[part] = object()
    return built


def test_variants_build_on_each_other_with_configured_sizes() -> None:
    f = factory()

    r1, r2, r3, r4 = (f.get(v) for v in ("R1", "R2", "R3", "R4"))

    assert type(r1) is DenseRetriever
    assert isinstance(r2, HybridRetriever) and r2.alpha == 0.6
    assert isinstance(r3, RerankRetriever) and r3.base is r2 and r3.candidates == 40
    assert isinstance(r4, GraphRAGRetriever) and r4.base is r3
    assert (r4.expander.max_hops, r4.expander.per_seed, r4.expander.limit) == (1, 4, 12)
    assert r4.analyzer is f.analyzer


def test_default_comes_from_the_config_and_variants_are_cached() -> None:
    f = factory(config(default="R3"))

    assert isinstance(f.get(), RerankRetriever)
    assert f.get() is f.get("r3")
    with pytest.raises(ValueError, match="unknown retriever"):
        f.get("R9")


def test_r5_wraps_r4_with_the_given_llm() -> None:
    llm: Any = object()
    f = factory(llm=llm)

    r5 = f.get("R5")

    assert isinstance(r5, Text2CypherRetriever)
    assert r5.graphrag is f.get("R4")
    assert r5.generator.llm is llm


def test_repo_default_retriever_is_graphrag() -> None:
    assert load_config().retrieval.default == "R4"


def test_build_retriever_passes_options(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    marker: Any = object()

    def fake_get(self: RetrieverFactory, variant: str | None = None) -> Any:
        seen.update(namespace=self.namespace, variant=variant)
        return marker

    monkeypatch.setattr(RetrieverFactory, "get", fake_get)

    assert build_retriever("R2", namespace="eval", config=config()) is marker
    assert seen == {"namespace": "eval", "variant": "R2"}
