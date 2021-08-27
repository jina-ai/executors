__copyright__ = "Copyright (c) 2020-2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

from pathlib import Path
from typing import Dict

import numpy as np
import pytest
from jina import DocumentArray, Document, Executor

from ...image_tf_encoder import ImageTFEncoder


input_dim = 336
target_output_dim = 1280


def test_config():
    ex = Executor.load_config(str(Path(__file__).parents[2] / 'config.yml'))
    assert ex.model_name == 'MobileNetV2'


def test_encoding_results():
    num_doc = 2
    test_data = np.random.rand(num_doc, input_dim, input_dim, 3)
    doc = DocumentArray()
    for i in range(num_doc):
        doc.append(Document(blob=test_data[i]))

    encoder = ImageTFEncoder()
    encoder.encode(doc, parameters={})
    assert len(doc) == num_doc
    for i in range(num_doc):
        assert doc[i].embedding.shape == (target_output_dim,)


def test_image_results(test_images: Dict[str, np.array]):
    embeddings = {}
    encoder = ImageTFEncoder()
    for name, image_arr in test_images.items():
        docs = DocumentArray([Document(blob=image_arr)])
        encoder.encode(docs, parameters={})
        embeddings[name] = docs[0].embedding
        assert docs[0].embedding.shape == (target_output_dim,)

    def dist(a, b):
        a_embedding = embeddings[a]
        b_embedding = embeddings[b]
        return np.linalg.norm(a_embedding - b_embedding)

    small_distance = dist('banana1', 'banana2')
    assert small_distance < dist('banana1', 'airplane')
    assert small_distance < dist('banana1', 'satellite')
    assert small_distance < dist('banana1', 'studio')
    assert small_distance < dist('banana2', 'airplane')
    assert small_distance < dist('banana2', 'satellite')
    assert small_distance < dist('banana2', 'studio')
    assert small_distance < dist('airplane', 'studio')
    assert small_distance < dist('airplane', 'satellite')


@pytest.mark.gpu
def test_image_results_gpu(test_images: Dict[str, np.array]):
    num_doc = 2
    test_data = np.random.rand(num_doc, input_dim, input_dim, 3)
    doc = DocumentArray()
    for i in range(num_doc):
        doc.append(Document(blob=test_data[i]))

    encoder = ImageTFEncoder(device='cuda')
    encoder.encode(doc, parameters={})
    assert len(doc) == num_doc
    for i in range(num_doc):
        assert doc[i].embedding.shape == (target_output_dim,)
