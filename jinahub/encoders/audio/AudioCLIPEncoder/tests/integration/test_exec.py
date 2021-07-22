__copyright__ = "Copyright (c) 2020-2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

import os
import librosa

from jina import Flow, Document, DocumentArray

cur_dir = os.path.dirname(os.path.abspath(__file__))


def test_flow_from_yml():

    doc = DocumentArray([Document()])
    with Flow.load_config(os.path.join(cur_dir, 'flow.yml')) as f:
        resp = f.post(on='test', inputs=doc, return_results=True)

    assert resp is not None


def test_embedding_exists():

    x_audio, _ = librosa.load(os.path.join(cur_dir, '../data/sample.mp3'))
    doc = DocumentArray([Document(blob=x_audio)])

    with Flow.load_config(os.path.join(cur_dir, 'flow.yml')) as f:
        responses = f.post(on='index', inputs=doc, return_results=True)

    assert responses[0].docs[0].embedding is not None
    assert responses[0].docs[0].embedding.shape == (1024, )
