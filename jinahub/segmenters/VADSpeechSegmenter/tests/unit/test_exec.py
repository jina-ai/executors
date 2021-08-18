from pathlib import Path

import pytest
from jina import Executor, DocumentArray, Document
from jina.excepts import BadDocType

from ...vad_speech_segmenter import VADSpeechSegmenter


def test_load():
    segmenter = Executor.load_config(str(Path(__file__).parents[2] / 'config.yml'))
    assert type(segmenter).__name__ == 'VADSpeechSegmenter'


def test_init():
    with pytest.raises(ValueError, match='model and repo cannot be None'):
        segmenter = VADSpeechSegmenter(model=None)

    with pytest.raises(ValueError, match='model and repo cannot be None'):
        segmenter = VADSpeechSegmenter(repo=None)

    # default case
    segmenter = VADSpeechSegmenter()


@pytest.mark.parametrize('_type', ['wav', 'mp3', 'blob', '', None])
def test_segment(build_da, segmenter, _type):

    if _type == '':
        with pytest.raises(
            BadDocType, match='doc needs to have either a blob or a wav/mp3 uri'
        ):
            segmenter.segment(DocumentArray(Document()))
        return

    elif _type is None:
        segmenter.segment(DocumentArray())
        return

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
        with pytest.raises(
            BadDocType, match='data is blob but sample rate is not provided'
        ):
            segmenter.segment(docs)

        docs[0].tags['sample_rate'] = 0
        with pytest.raises(BadDocType, match='sample rate cannot be 0'):
            segmenter.segment(docs)
