__copyright__ = 'Copyright (c) 2020-2021 Jina AI Limited. All rights reserved.'
__license__ = 'Apache-2.0'

from pathlib import Path

import pytest
from executor import VideoLoader
from jina import Document, DocumentArray, Executor


def test_config():
    ex = Executor.load_config(str(Path(__file__).parents[2] / 'config.yml'))
    assert ex.fps == 1
    assert ex.max_num_frames == 50
    assert ex.debug == False


def test_no_docucments(encoder: VideoLoader):
    docs = DocumentArray()
    encoder.extract(docs=docs)
    assert len(docs) == 0  # SUCCESS


def test_none_docs(encoder: VideoLoader):
    encoder.extract(docs=None, parameters={})


def test_docs_no_uris(encoder: VideoLoader):
    docs = DocumentArray([Document()])

    with pytest.raises(ValueError, match='No uri'):
        encoder.extract(docs=docs, parameters={})

    assert len(docs) == 1
    assert len(docs[0].chunks) == 0


@pytest.mark.parametrize('batch_size', [1, 2, 4, 8])
def test_batch_encode(encoder: VideoLoader, batch_size: int):
    docs = DocumentArray(
        [
            Document(id=f'2c2OmN49cj8_{idx}.mp4', uri='tests/toy_data/2c2OmN49cj8.mp4')
            for idx in range(batch_size)
        ]
    )
    encoder.extract(docs=docs)
    for doc in docs:
        assert len(doc.chunks) == 16
        for image_chunk in filter(lambda x: x.modality == 'image', doc.chunks):
            assert len(image_chunk.blob.shape) == 3

        for audio_chunk in filter(lambda x: x.modality == 'audio', doc.chunks):
            assert audio_chunk.blob is not None
            assert audio_chunk.mime_type == 'audio/wav'
