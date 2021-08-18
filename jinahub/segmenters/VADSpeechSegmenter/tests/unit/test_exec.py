from pathlib import Path

import pytest
from jina import Executor
from jina.excepts import BadDocType

from ...vad_speech_segmenter import VADSpeechSegmenter


def test_load(test_dir):
    segmenter = Executor.load_config(str(Path(test_dir).parents[0] / 'config.yml'))
    assert type(segmenter).__name__ == 'VADSpeechSegmenter'


@pytest.mark.parametrize('_type', ['wav', 'mp3', 'blob'])
def test_segment(build_da, segmenter, _type):
    docs = build_da(_type)
    segmenter.segment(docs)

    # assert doc has 4 chunks
    for doc in docs:
        assert len(doc.chunks) == 4

    file_paths = [
        f'doc_{_type}_original.wav',
        f'chunk_{_type}_0_56500.wav',
        f'chunk_{_type}_69500_92000.wav',
        f'chunk_{_type}_94500_213000.wav',
        f'chunk_{_type}_223500_270500.wav',
    ]

    # assert dumped files exist
    for file_path in file_paths:
        assert (Path(segmenter.workspace) / f'audio/{file_path}').is_file()

    # assert exception is raised when doc blob is provided by sample rate is not
    if _type == 'blob':
        docs[0].tags.pop('sample_rate')
        with pytest.raises(BadDocType) as e:
            segmenter.segment(docs)
        assert str(e.value) == 'data is blob but sample rate is not provided'
