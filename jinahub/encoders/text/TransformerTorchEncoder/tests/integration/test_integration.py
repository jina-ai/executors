__copyright__ = "Copyright (c) 2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

from typing import Callable, List

import pytest
from jina import DocumentArray, Flow
from ...transform_encoder import TransformerTorchEncoder


@pytest.mark.parametrize("request_size", [1, 10, 50, 100])
def test_integration(data_generator: Callable, request_size: int):
    with Flow(return_results=True).add(uses=TransformerTorchEncoder) as flow:
        resp = flow.post(
            on="/index", inputs=data_generator(), request_size=request_size, return_results=True
        )

    assert min(len(resp) * request_size, 50) == 50
    for r in resp:
        for doc in r.docs:
            assert doc.embedding is not None


def filter_none(elements):
    return list(filter(lambda e: e is not None, elements))


@pytest.mark.parametrize(
    ["docs", "docs_per_path", "traversal_path"],
    [
        (pytest.lazy_fixture("docs_with_text"), [["r", 10], ["c", 0], ["cc", 0]], "r"),
        (
            pytest.lazy_fixture("docs_with_chunk_text"),
            [["r", 0], ["c", 10], ["cc", 0]],
            "c",
        ),
        (
            pytest.lazy_fixture("docs_with_chunk_chunk_text"),
            [["r", 0], ["c", 0], ["cc", 10]],
            "cc",
        ),
    ],
)
def test_traversal_path(
    docs: DocumentArray, docs_per_path: List[List[str]], traversal_path: str
):
    def validate_traversal(expected_docs_per_path: List[List[str]]):
        def validate(res):
            for path, count in expected_docs_per_path:
                assert len(
                    filter_none(
                        DocumentArray(res[0].docs)
                        .traverse_flat([path])
                        .get_attributes("embedding")
                    )
                    ) == count
        return validate

    flow = Flow(return_results=True).add(uses=TransformerTorchEncoder)
    with flow:
        resp = flow.post(
            on="/test", inputs=docs, parameters={"traversal_paths": [traversal_path]}, return_results=True
        )

    validate_traversal(docs_per_path)(resp)
