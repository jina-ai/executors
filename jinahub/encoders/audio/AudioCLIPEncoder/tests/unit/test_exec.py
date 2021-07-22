__copyright__ = "Copyright (c) 2020-2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

import os
import librosa
import numpy as np

from jina import Executor, Document, DocumentArray

from audio_clip_encoder import AudioCLIPEncoder

cur_dir = os.path.dirname(os.path.abspath(__file__))


def test_load():
    encoder = Executor.load_config(os.path.join(cur_dir, '../../config.yml'))
    assert encoder.model_path.endswith('AudioCLIP-Full-Training.pt')


def test_embedding_dimension():
    x_audio, sample_rate = librosa.load(os.path.join(cur_dir, '../data/sample.wav'))
    docs = DocumentArray([Document(blob=x_audio)])
    model = AudioCLIPEncoder()
    model.encode(docs, parameters={})
    assert docs[0].embedding.shape == (1024, )


def test_many_documents():

    audio1, _ = librosa.load(os.path.join(cur_dir, '../data/sample.mp3'))
    audio2, _ = librosa.load(os.path.join(cur_dir, '../data/sample.wav'))
    docs = DocumentArray([Document(blob=audio1), Document(blob=audio2)])

    encoder = AudioCLIPEncoder()
    encoder.encode(docs, parameters={})

    assert docs[0].embedding.shape == (1024, )
    assert docs[1].embedding.shape == (1024, )


def test_traversal_paths():

    audio1, _ = librosa.load(os.path.join(cur_dir, '../data/sample.mp3'))
    audio2, _ = librosa.load(os.path.join(cur_dir, '../data/sample.wav'))

    audio1_chunks = np.split(audio1, 4)
    audio2_chunks = np.split(audio2, 2)

    docs = DocumentArray([
        Document(
            id='root1',
            blob=audio1,
            chunks=[
                Document(id=f'chunk1{i}', blob=chunk) for i, chunk in enumerate(audio1_chunks)
            ]
        ),
        Document(
            id='root2',
            blob=audio2,
            chunks=[
                Document(id='chunk21', blob=audio2_chunks[0]),
                Document(id='chunk22', blob=audio2_chunks[1], chunks=[
                    Document(id=f'chunk22{i}', blob=chunk) for i, chunk in enumerate(np.split(audio2_chunks[1], 3))
                ])
            ]
        )
    ])

    encoder = AudioCLIPEncoder(default_traversal_paths=['c'])
    encoder.encode(docs, parameters={})
    encoder.encode(docs, parameters={'traversal_paths': ['cc']})
    for path, count in [['r', 0], ['c', 6], ['cc', 3]]:
        embeddings = DocumentArray(docs).traverse_flat([path]).get_attributes('embedding')
        assert all(
            embedding.shape == (1024,)
            for embedding in embeddings
        )
        assert len(embeddings) == count
