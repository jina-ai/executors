import pytest
import os
from typing import Callable
from pathlib import Path

from jina import Document, DocumentArray
import numpy as np
import torchaudio

from ..vad_speech_segmenter import VADSpeechSegmenter


@pytest.fixture(scope='module')
def segmenter(tmpdir_factory) -> 'VADSpeechSegmenter':
    workspace = tmpdir_factory.mktemp('data') / 'segmenter'
    workspace.mkdir()
    return VADSpeechSegmenter(
        normalize=False, dump=True, metas={'workspace': str(workspace)}
    )


@pytest.fixture
def test_dir() -> str:
    return os.path.dirname(os.path.abspath(__file__))


@pytest.fixture
def build_da(test_dir) -> Callable[[str], 'Document']:
    def _build_da(_type):
        assert _type in {'wav', 'mp3', 'blob'}
        doc = Document(id=_type)
        extension = _type if _type != 'blob' else 'wav'
        path = str(Path(test_dir) / f'data/audio/test.{extension}')
        if _type == 'blob':
            data, sample_rate = torchaudio.load(path)
            data = np.mean(data.detach().cpu().numpy(), axis=0)
            doc.blob = data
            doc.tags['sample_rate'] = sample_rate
        else:
            doc.uri = path
        return DocumentArray(doc)

    return _build_da
