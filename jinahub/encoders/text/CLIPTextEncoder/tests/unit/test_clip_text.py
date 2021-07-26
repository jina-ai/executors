import clip
import copy
import numpy as np
import torch
from jina import Document, DocumentArray, Executor
from jinahub.encoder.clip_text import CLIPTextEncoder


def test_clip_batch():
    test_docs = DocumentArray((Document(text='random text') for _ in range(30)))
    clip_text_encoder = CLIPTextEncoder()
    parameters = {'batch_size': 10}
    clip_text_encoder.encode(test_docs, parameters)
    assert 30 == len(test_docs.get_attributes('embedding'))


def test_clip_data():
    docs = []
    words = ['apple', 'banana1', 'banana2', 'studio', 'satelite', 'airplane']
    for word in words:
        docs.append(Document(text=word))

    sentences = [
        'Jina AI is lit',
        'Jina AI is great',
        'Jina AI is a cloud-native neural search company',
        'Jina AI is a github repo',
        'Jina AI is an open source neural search project',
    ]
    for sentence in sentences:
        docs.append(Document(text=sentence))

    clip_text_encoder = CLIPTextEncoder()
    clip_text_encoder.encode(DocumentArray(docs), {})

    txt_to_ndarray = {}
    for d in docs:
        txt_to_ndarray[d.text] = d.embedding

    def dist(a, b):
        nonlocal txt_to_ndarray
        a_embedding = txt_to_ndarray[a]
        b_embedding = txt_to_ndarray[b]
        return np.linalg.norm(a_embedding - b_embedding)

    # assert semantic meaning is captured in the encoding
    small_distance = dist('banana1', 'banana2')
    assert small_distance < dist('banana1', 'airplane')
    assert small_distance < dist('banana1', 'satelite')
    assert small_distance < dist('banana1', 'studio')
    assert small_distance < dist('banana2', 'airplane')
    small_distance = dist('Jina AI is lit', 'Jina AI is great')
    assert small_distance < dist(
        'Jina AI is a cloud-native neural search company', 'Jina AI is a github repo'
    )
    assert small_distance < dist(
        'Jina AI is a cloud-native neural search company',
        'Jina AI is an open source neural search project',
    )

    # assert same results like calculating it manually
    model, preprocess = clip.load('ViT-B/32', device='cpu')
    assert len(txt_to_ndarray) == 11
    for text, actual_embedding in txt_to_ndarray.items():
        with torch.no_grad():
            tokens = clip.tokenize(text)
            expected_embedding = model.encode_text(tokens).detach().numpy().flatten()

        np.testing.assert_almost_equal(actual_embedding, expected_embedding, 5)

def test_traversal_path():
    text = 'blah'
    docs = DocumentArray([Document(id='root1', text=text)])
    docs[0].chunks = [Document(id='chunk11', text=text),
                      Document(id='chunk12', text=text),
                      Document(id='chunk13', text=text)
                      ]
    docs[0].chunks[0].chunks = [
        Document(id='chunk111', text=text),
        Document(id='chunk112', text=text),
    ]

    encoder = CLIPTextEncoder(default_traversal_paths=['c'], model_name='ViT-B/32')

    original_docs = copy.deepcopy(docs)
    encoder.encode(docs=docs, parameters={}, return_results=True)
    for path, count in [['r', 0], ['c', 3], ['cc', 0]]:
        assert len(docs.traverse_flat([path]).get_attributes('embedding')) == count

    encoder.encode(docs=original_docs, parameters={'traversal_paths': ['cc']}, return_results=True)
    for path, count in [['r', 0], ['c', 0], ['cc', 2]]:
        assert len(original_docs.traverse_flat([path]).get_attributes('embedding')) == count