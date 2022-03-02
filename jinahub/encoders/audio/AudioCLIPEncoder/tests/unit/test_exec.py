__copyright__ = 'Copyright (c) 2020-2021 Jina AI Limited. All rights reserved.'
__license__ = 'Apache-2.0'

from pathlib import Path
from typing import Tuple

import librosa
import numpy as np
import pytest
from executor.audio_clip_encoder import AudioCLIPEncoder
from jina import Document, DocumentArray, Executor


@pytest.fixture(scope="module")
def encoder() -> AudioCLIPEncoder:
    return AudioCLIPEncoder()


@pytest.fixture(scope="module")
def gpu_encoder() -> AudioCLIPEncoder:
    return AudioCLIPEncoder(device='cuda', download_model=True)


@pytest.fixture(scope="function")
def nested_docs() -> DocumentArray:
    tensor, sample_rate = librosa.load(
        str(Path(__file__).parents[1] / 'test_data/sample.wav')
    )
    docs = DocumentArray(
        [Document(id="root1", tensor=tensor, tags={'sample_rate': sample_rate})]
    )
    docs[0].chunks = [
        Document(id="chunk11", tensor=tensor, tags={'sample_rate': sample_rate}),
        Document(id="chunk12", tensor=tensor, tags={'sample_rate': sample_rate}),
        Document(id="chunk13", tensor=tensor, tags={'sample_rate': sample_rate}),
    ]
    docs[0].chunks[0].chunks = [
        Document(id="chunk111", tensor=tensor, tags={'sample_rate': sample_rate}),
        Document(id="chunk112", tensor=tensor, tags={'sample_rate': sample_rate}),
    ]

    return docs


def test_config():
    ex = Executor.load_config(str(Path(__file__).parents[2] / 'config.yml'))
    assert ex.model_path.endswith('AudioCLIP-Full-Training.pt')


def test_no_documents(encoder: AudioCLIPEncoder):
    docs = DocumentArray()
    encoder.encode(docs=docs, parameters={})
    assert len(docs) == 0  # SUCCESS


def test_none_docs(encoder: AudioCLIPEncoder):
    encoder.encode(docs=None, parameters={})


def test_docs_no_tensors(encoder: AudioCLIPEncoder):
    docs = DocumentArray([Document()])
    encoder.encode(docs=DocumentArray(), parameters={})
    assert len(docs) == 1
    assert docs[0].embedding is None


def test_encode_single_document(encoder: AudioCLIPEncoder):
    x_audio, sample_rate = librosa.load(
        str(Path(__file__).parents[1] / 'test_data/sample.wav')
    )
    docs = DocumentArray([Document(tensor=x_audio, tags={'sample_rate': sample_rate})])
    encoder.encode(docs, parameters={})
    assert docs[0].embedding.shape == (1024,)
    assert docs[0].tags['sample_rate'] == AudioCLIPEncoder.TARGET_SAMPLE_RATE


def test_encode_multiple_documents():

    audio1, sample_rate1 = librosa.load(
        str(Path(__file__).parents[1] / 'test_data/sample.mp3')
    )
    audio2, sample_rate2 = librosa.load(
        str(Path(__file__).parents[1] / 'test_data/sample.wav')
    )
    docs = DocumentArray(
        [
            Document(tensor=audio1, tags={'sample_rate': sample_rate1}),
            Document(tensor=audio2, tags={'sample_rate': sample_rate2}),
        ]
    )

    encoder = AudioCLIPEncoder()
    encoder.encode(docs, parameters={})

    assert docs[0].embedding.shape == (1024,)
    assert docs[0].tags['sample_rate'] == AudioCLIPEncoder.TARGET_SAMPLE_RATE
    assert docs[1].embedding.shape == (1024,)
    assert docs[1].tags['sample_rate'] == AudioCLIPEncoder.TARGET_SAMPLE_RATE


@pytest.mark.gpu
def test_embedding_dimension_gpu(gpu_encoder: AudioCLIPEncoder):
    x_audio, sample_rate = librosa.load(
        str(Path(__file__).parents[1] / 'test_data/sample.wav')
    )
    docs = DocumentArray([Document(tensor=x_audio, tags={'sample_rate': sample_rate})])
    gpu_encoder.encode(docs, parameters={})
    assert docs[0].embedding.shape == (1024,)
    assert docs[0].tags['sample_rate'] == AudioCLIPEncoder.TARGET_SAMPLE_RATE


def test_traversal_paths():

    audio1, sample_rate1 = librosa.load(
        str(Path(__file__).parents[1] / 'test_data/sample.mp3')
    )
    audio2, sample_rate2 = librosa.load(
        str(Path(__file__).parents[1] / 'test_data/sample.wav')
    )

    audio1_chunks = np.split(audio1, 4)
    audio2_chunks = np.split(audio2, 2)

    docs = DocumentArray(
        [
            Document(
                id='root1',
                tensor=audio1,
                tags={'sample_rate': sample_rate1},
                chunks=[
                    Document(
                        id=f'chunk1{i}', tensor=chunk, tags={'sample_rate': sample_rate1}
                    )
                    for i, chunk in enumerate(audio1_chunks)
                ],
            ),
            Document(
                id='root2',
                tensor=audio2,
                tags={'sample_rate': sample_rate2},
                chunks=[
                    Document(
                        id='chunk21',
                        tensor=audio2_chunks[0],
                        tags={'sample_rate': sample_rate2},
                    ),
                    Document(
                        id='chunk22',
                        tensor=audio2_chunks[1],
                        tags={'sample_rate': sample_rate2},
                        chunks=[
                            Document(
                                id=f'chunk22{i}',
                                tensor=chunk,
                                tags={'sample_rate': sample_rate2},
                            )
                            for i, chunk in enumerate(np.split(audio2_chunks[1], 3))
                        ],
                    ),
                ],
            ),
        ]
    )

    encoder = AudioCLIPEncoder(traversal_paths=['@c'])
    encoder.encode(docs, parameters={})
    encoder.encode(docs, parameters={'traversal_paths': ['@cc']})
    for path, count in [['@r', 0], ['@c', 6], ['@cc', 3]]:
        embeddings = [
            embedding
            for embedding in DocumentArray(docs)[path].embeddings
            if embedding is not None
        ]

        sample_rates = {
            doc.tags['sample_rate'] for doc in DocumentArray(docs)[path]
        }

        assert all(embedding.shape == (1024,) for embedding in embeddings)
        assert len(embeddings) == count
        if path != 'r':
            assert (
                len(sample_rates) == 1
                and sample_rates.pop() == AudioCLIPEncoder.TARGET_SAMPLE_RATE
            )


def test_no_sample_rate():
    audio, sample_rate = librosa.load(
        str(Path(__file__).parents[1] / 'test_data/sample.mp3')
    )
    docs = DocumentArray([Document(tensor=audio)])
    encoder = AudioCLIPEncoder()
    with pytest.raises(
        NotImplementedError,
        match='sample rate is not given, please provide a valid sample rate',
    ):
        encoder.encode(docs, parameters={})


@pytest.mark.parametrize('batch_size', [1, 2, 4, 8])
def test_batch_size(encoder: AudioCLIPEncoder, batch_size: int):
    audio, sample_rate = librosa.load(
        str(Path(__file__).parents[1] / 'test_data/sample.mp3')
    )
    docs = DocumentArray(
        [Document(tensor=audio, tags={'sample_rate': sample_rate}) for _ in range(32)]
    )
    encoder.encode(docs, parameters={'batch_size': batch_size})

    for doc in docs:
        assert doc.embedding.shape == (1024,)


@pytest.mark.parametrize(
    "traversal_paths, counts",
    [
        [('@c',), (('@r', 0), ('@c', 3), ('@cc', 0))],
        [('@cc',), (('@r', 0), ('@c', 0), ('@cc', 2))],
        [('@r',), (('@r', 1), ('@c', 0), ('@cc', 0))],
        [('@cc,r'), (('@r', 1), ('@c', 0), ('@cc', 2))],
    ],
)
def test_traversal_path(
    traversal_paths: str,
    counts: Tuple[str, int],
    nested_docs: DocumentArray,
    encoder: AudioCLIPEncoder,
):
    encoder.encode(nested_docs, parameters={"traversal_paths": traversal_paths})
    for path, count in counts:
        embeddings = nested_docs[path].embeddings
        assert len([em for em in embeddings if em is not None]) == count
