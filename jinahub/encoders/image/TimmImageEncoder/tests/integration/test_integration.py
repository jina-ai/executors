import numpy as np
import pytest
import subprocess

from jina import Flow, Document, DocumentArray

from ...timm_encoder import TimmImageEncoder


@pytest.mark.parametrize(
    "arr_in",
    [
        (np.ones((224, 224, 3), dtype=np.uint8)),
        (np.ones((100, 100, 3), dtype=np.uint8)),
        (np.ones((50, 40, 3), dtype=np.uint8)),
    ],
)
def test_no_batch(arr_in: np.ndarray):
    flow = Flow().add(uses=TimmImageEncoder)
    with flow:
        resp = flow.post(
            on="/test", inputs=[Document(blob=arr_in)], return_results=True
        )

    results_arr = DocumentArray(resp[0].data.docs)
    assert len(results_arr) == 1
    assert results_arr[0].embedding is not None
    assert results_arr[0].embedding.shape == (512,)


def test_with_batch():
    flow = Flow().add(uses=TimmImageEncoder)

    with flow:
        resp = flow.post(
            on="/test",
            inputs=(
                Document(blob=np.ones((224, 224, 3), dtype=np.uint8)) for _ in range(25)
            ),
            return_results=True,
        )

    assert len(resp[0].docs.get_attributes("embedding")) == 25
    for r in resp:
        for doc in r.docs:
            assert doc.embedding is not None
            assert doc.embedding.shape == (512,)


@pytest.mark.docker
def test_docker_runtime():
    with pytest.raises(subprocess.TimeoutExpired):
        subprocess.run(
            ["jina", "pea", "--uses=docker://timmimageencoder"], timeout=30, check=True
        )
