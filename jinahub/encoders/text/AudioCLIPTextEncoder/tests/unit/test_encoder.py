from typing import List

import numpy as np
import pytest
import torch
from jina import Document, DocumentArray
from jinahub.encoder.audioclip_text import AudioCLIPTextEncoder

_EMBEDDING_DIM = 1024


def test_encoding_cpu():
    enc = AudioCLIPTextEncoder(device='cpu')
    input_data = DocumentArray([Document(text='hello world')])

    enc.encode(docs=input_data, parameters={})

    assert input_data[0].embedding.shape == (_EMBEDDING_DIM,)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='GPU is needed for this test')
def test_encoding_gpu():
    enc = AudioCLIPTextEncoder(device='cuda')
    input_data = DocumentArray([Document(text='hello world')])

    enc.encode(docs=input_data, parameters={})

    assert input_data[0].embedding.shape == (_EMBEDDING_DIM,)


@pytest.mark.parametrize(
    ['docs', 'docs_per_path', 'traversal_path'],
    [
        (pytest.lazy_fixture('docs_with_text'), [['r', 10], ['c', 0], ['cc', 0]], 'r'),
        (
            pytest.lazy_fixture('docs_with_chunk_text'),
            [['r', 0], ['c', 10], ['cc', 0]],
            'c',
        ),
        (
            pytest.lazy_fixture('docs_with_chunk_chunk_text'),
            [['r', 0], ['c', 0], ['cc', 10]],
            'cc',
        ),
    ],
)
def test_traversal_path(
    docs: DocumentArray, docs_per_path: List[List[str]], traversal_path: str
):
    def validate_traversal(expected_docs_per_path: List[List[str]]):
        def validate(res):
            for path, count in expected_docs_per_path:
                embeddings = (
                    DocumentArray(res).traverse_flat([path]).get_attributes('embedding')
                )
                for emb in embeddings:
                    if emb is None:
                        return False
                return len(embeddings) == count

        return validate

    encoder = AudioCLIPTextEncoder(default_traversal_paths=[traversal_path])
    encoder.encode(docs, {'traversal_paths': [traversal_path]})

    assert validate_traversal(docs_per_path)(docs)


def test_encodes_semantic_meaning():
    sentences = dict()
    sentences["A"] = "Hello, my name is Michael."
    sentences["B"] = "Today we are going to Disney World."
    sentences["C"] = "There are animals on the road"
    sentences["D"] = "A dog is running down the road"

    encoder = AudioCLIPTextEncoder()

    embeddings = {}
    for id_, sentence in sentences.items():
        docs = DocumentArray([Document(text=sentence)])
        encoder.encode(docs, parameters={})
        embeddings[id_] = docs[0].embedding

    def dist(a, b):
        a_embedding = embeddings[a]
        b_embedding = embeddings[b]
        return np.linalg.norm(a_embedding - b_embedding)

    small_distance = dist("C", "D")
    assert small_distance < dist("C", "B")
    assert small_distance < dist("C", "A")
    assert small_distance < dist("B", "A")


def test_multiple_traversal_paths():
    sentences = list()
    sentences.append('Hello, my name is Michael.')
    sentences.append('Today we are going to Disney World.')
    sentences.append('There are animals on the road')
    sentences.append('A dog is running down the road')
    docs = DocumentArray([Document(text=sentence) for sentence in sentences])
    for index, sent in enumerate(sentences):
        docs[index].chunks.append(Document(text=sent))
        docs[index].chunks[0].chunks.append(Document(text=sentences[3 - index]))

    encoder = AudioCLIPTextEncoder(default_traversal_paths=['r', 'c', 'cc'])

    encoder.encode(docs, {})
    for doc in docs:
        assert doc.embedding.shape == (_EMBEDDING_DIM,)
        assert doc.chunks[0].embedding.shape == (_EMBEDDING_DIM,)
        assert doc.chunks[0].chunks[0].embedding.shape == (_EMBEDDING_DIM,)


def test_no_docs():
    encoder = AudioCLIPTextEncoder()
    encoder.encode(None, {})
    encoder.encode(DocumentArray(), {})
