__copyright__ = "Copyright (c) 2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

import subprocess

import numpy as np
import pytest
from jina import Document, DocumentArray, Flow
from simpleranker import SimpleRanker


def test_integration():
    query = Document()
    chunks = [Document(), Document()]
    query.chunks = chunks

    query.chunks[0].matches = [
        Document(scores={'cosine': 0.1}, parent_id='1'),
        Document(scores={'cosine': 0.3}, parent_id='2'),
    ]
    query.chunks[1].matches = [
        Document(scores={'cosine': 0.2}, parent_id='1'),
        Document(scores={'cosine': 0.5}, parent_id='2'),
    ]

    with Flow(return_results=True).add(uses=SimpleRanker) as flow:
        resp = flow.post(
            on='/search',
            inputs=DocumentArray([query]),
            return_results=True,
        )

    for r in resp:
        for doc in r.docs:
            print(doc.matches)
            assert doc.matches[0].id == '1'
            np.testing.assert_almost_equal(doc.matches[0].scores['cosine'].value, 0.1)
            np.testing.assert_almost_equal(doc.matches[1].scores['cosine'].value, 0.3)


@pytest.mark.docker
def test_docker_runtime(build_docker_image: str):
    with pytest.raises(subprocess.TimeoutExpired):
        subprocess.run(
            [
                'jina',
                'executor',
                f'--uses=docker://{build_docker_image}',
            ],
            timeout=30,
            check=True,
        )
