import os
import subprocess
import pytest

from jina import Flow, Document, DocumentArray
from ...tfidf_text_executor import TFIDFTextEncoder  # is implicitly required

cur_dir = os.path.dirname(os.path.abspath(__file__))


def test_flow_generates_embedding():
    doc = DocumentArray([Document(text='Han likes eating pizza')])

    with Flow.load_config(os.path.join(cur_dir, 'flow.yml')) as f:
        responses = f.index(inputs=doc, return_results=True)

    assert responses[0].docs[0].embedding is not None
    # input has 4 different words
    assert responses[0].docs[0].embedding.nnz == 4


@pytest.mark.docker
def test_docker_runtime(build_docker_image: str):
    with pytest.raises(subprocess.TimeoutExpired):
        subprocess.run(
            ['jina', 'executor', f'--uses=docker://{build_docker_image}'],
            timeout=30,
            check=True,
        )
