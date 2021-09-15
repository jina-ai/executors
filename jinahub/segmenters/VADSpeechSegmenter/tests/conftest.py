from pathlib import Path
from typing import Callable

import numpy as np
import pytest
import torchaudio
from jina import Document, DocumentArray
from vad_speech_segmenter import VADSpeechSegmenter


@pytest.fixture(scope='module')
def segmenter(tmpdir_factory) -> 'VADSpeechSegmenter':
    workspace = tmpdir_factory.mktemp('data')
    return VADSpeechSegmenter(
        normalize=False, dump=True, metas={'workspace': str(workspace)}
    )


@pytest.fixture
def build_da() -> Callable[[str], 'Document']:
    def _build_da(_type):
        assert _type in {'wav', 'mp3', 'blob'}
        doc = Document(id=_type)
        extension = _type if _type != 'blob' else 'wav'
        path = str(Path(__file__).parent / f'data/audio/test.{extension}')
        if _type == 'blob':
            data, sample_rate = torchaudio.load(path)
            data = np.mean(data.cpu().numpy(), axis=0)
            doc.blob = data
            doc.tags['sample_rate'] = sample_rate
        else:
            doc.uri = path
        return DocumentArray(doc)

    return _build_da
