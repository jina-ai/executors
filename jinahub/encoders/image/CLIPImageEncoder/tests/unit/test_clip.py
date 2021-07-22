import copy
import operator

import numpy as np

from jina import Flow, Document, DocumentArray
from jina.executors import BaseExecutor
from jinahub.encoder.clip_image import CLIPImageEncoder


def test_clip_any_image_shape():
    encoder = CLIPImageEncoder()
    docs = DocumentArray([Document(blob=np.ones((224, 224, 3), dtype=np.uint8))])

    encoder.encode(docs=docs, parameters={})
    assert len(docs.get_attributes('embedding')) == 1

    docs = DocumentArray([Document(blob=np.ones((100, 100, 3), dtype=np.uint8))])
    encoder.encode(docs=docs, parameters={})
    assert len(docs.get_attributes('embedding')) == 1


def test_clip_batch():
    encoder = CLIPImageEncoder(default_batch_size=10, model_name='ViT-B/32', device='cpu')
    docs = DocumentArray([Document(blob=np.ones((224, 224, 3), dtype=np.uint8)) for _ in range(25)])
    encoder.encode(docs, parameters={})
    assert len(docs.get_attributes('embedding')) == 25


def test_traversal_paths():
    blob = np.ones((224, 224, 3), dtype=np.uint8)
    docs = DocumentArray([Document(id='root1', blob=blob)])
    docs[0].chunks = [Document(id='chunk11', blob=blob),
                      Document(id='chunk12', blob=blob),
                      Document(id='chunk13', blob=blob)
                      ]
    docs[0].chunks[0].chunks = [
        Document(id='chunk111', blob=blob),
        Document(id='chunk112', blob=blob),
    ]

    original_docs = copy.deepcopy(docs)
    encoder = CLIPImageEncoder(default_traversal_paths=['c'], model_name='ViT-B/32', device='cpu')
    encoder.encode(docs, parameters={})
    for path, count in [['r', 0], ['c', 3], ['cc', 0]]:
        assert len(docs.traverse_flat([path]).get_attributes('embedding')) == count


    encoder = CLIPImageEncoder(default_traversal_paths=['cc'])
    encoder.encode(original_docs, parameters={})
    for path, count in [['r', 0], ['c', 0], ['cc', 2]]:
        assert len(original_docs.traverse_flat([path]).get_attributes('embedding')) == count


def test_custom_processing():
    encoder = CLIPImageEncoder()
    docs_one = DocumentArray([Document(blob=np.ones((224, 224, 3), dtype=np.uint8))])
    encoder.encode(docs=docs_one, parameters={})
    assert docs_one[0].embedding is not None

    encoder = CLIPImageEncoder(use_default_preprocessing=False)

    docs_two = DocumentArray([Document(blob=np.ones((224, 224, 3), dtype=np.float32))])
    encoder.encode(docs=docs_two, parameters={})
    assert docs_two[0].embedding is not None
    np.testing.assert_array_compare(operator.__ne__, docs_one[0].embedding, docs_two[0].embedding)


def test_no_documents():
    encoder = CLIPImageEncoder()
    docs = DocumentArray()
    encoder.encode(docs=docs, parameters={})
    assert len(docs) == 0  # SUCCESS

def test_none_docs():
    encoder = CLIPImageEncoder()
    docs = None
    encoder.encode(docs=docs, parameters={})

def test_clip():
    ex = BaseExecutor.load_config('../../config.yml')
    assert ex.default_batch_size == 32
    assert len(ex.default_traversal_paths) == 1
    assert ex.default_traversal_paths[0] == 'r'
    assert ex.device == 'cpu'
    assert ex.is_updated is False
