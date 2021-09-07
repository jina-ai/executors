import subprocess

import numpy as np
import pytest
from jina import Document, Flow

from ...timm_encoder import TimmImageEncoder


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
def test_docker_runtime(build_docker_image: str):
    with pytest.raises(subprocess.TimeoutExpired):
        subprocess.run(
            ["jina", "executor", f"--uses=docker://{build_docker_image}"],
            timeout=30,
            check=True,
        )
