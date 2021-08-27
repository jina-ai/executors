__copyright__ = 'Copyright (c) 2020-2021 Jina AI Limited. All rights reserved.'
__license__ = 'Apache-2.0'

from pathlib import Path

import librosa
import numpy as np
import pytest
from jina import Document, DocumentArray, Executor
from jina.excepts import BadDocType

from ...audio_clip_encoder import AudioCLIPEncoder


def test_config():
    ex = Executor.load_config(str(Path(__file__).parents[2] / 'config.yml'))
    assert ex.model_path.endswith('AudioCLIP-Full-Training.pt')


def test_embedding_dimension():
    x_audio, sample_rate = librosa.load(
        str(Path(__file__).parents[1] / 'test_data/sample.wav')
    )
    docs = DocumentArray([Document(blob=x_audio, tags={'sample_rate': sample_rate})])
    model = AudioCLIPEncoder()
    model.encode(docs, parameters={})
    assert docs[0].embedding.shape == (1024,)
    assert docs[0].tags['sample_rate'] == AudioCLIPEncoder.TARGET_SAMPLE_RATE


def test_many_documents():

    audio1, sample_rate1 = librosa.load(
        str(Path(__file__).parents[1] / 'test_data/sample.mp3')
    )
    audio2, sample_rate2 = librosa.load(
        str(Path(__file__).parents[1] / 'test_data/sample.wav')
    )
    docs = DocumentArray(
        [
            Document(blob=audio1, tags={'sample_rate': sample_rate1}),
            Document(blob=audio2, tags={'sample_rate': sample_rate2}),
        ]
    )

    encoder = AudioCLIPEncoder()
    encoder.encode(docs, parameters={})

    assert docs[0].embedding.shape == (1024,)
    assert docs[0].tags['sample_rate'] == AudioCLIPEncoder.TARGET_SAMPLE_RATE
    assert docs[1].embedding.shape == (1024,)
    assert docs[1].tags['sample_rate'] == AudioCLIPEncoder.TARGET_SAMPLE_RATE


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
                blob=audio1,
                tags={'sample_rate': sample_rate1},
                chunks=[
                    Document(
                        id=f'chunk1{i}', blob=chunk, tags={'sample_rate': sample_rate1}
                    )
                    for i, chunk in enumerate(audio1_chunks)
                ],
            ),
            Document(
                id='root2',
                blob=audio2,
                tags={'sample_rate': sample_rate2},
                chunks=[
                    Document(
                        id='chunk21',
                        blob=audio2_chunks[0],
                        tags={'sample_rate': sample_rate2},
                    ),
                    Document(
                        id='chunk22',
                        blob=audio2_chunks[1],
                        tags={'sample_rate': sample_rate2},
                        chunks=[
                            Document(
                                id=f'chunk22{i}',
                                blob=chunk,
                                tags={'sample_rate': sample_rate2},
                            )
                            for i, chunk in enumerate(np.split(audio2_chunks[1], 3))
                        ],
                    ),
                ],
            ),
        ]
    )

    encoder = AudioCLIPEncoder(default_traversal_paths=['c'])
    encoder.encode(docs, parameters={})
    encoder.encode(docs, parameters={'traversal_paths': ['cc']})
    for path, count in [['r', 0], ['c', 6], ['cc', 3]]:
        embeddings = [
            embedding
            for embedding in DocumentArray(docs)
            .traverse_flat([path])
            .get_attributes('embedding')
            if embedding is not None
        ]

        sample_rates = {
            doc.tags['sample_rate'] for doc in DocumentArray(docs).traverse_flat([path])
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
    docs = DocumentArray([Document(blob=audio)])
    encoder = AudioCLIPEncoder()
    with pytest.raises(
        BadDocType, match='sample rate is not given, please provide a valid sample rate'
    ):
        encoder.encode(docs, parameters={})
