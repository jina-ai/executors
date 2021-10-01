import pytest
from executor.audio_loader import AudioLoader
from jina import Document, DocumentArray


def test_init():
    audio_loader = AudioLoader()
    assert audio_loader.audio_types == ['mp3', 'wav']
    with pytest.raises(ValueError):
        _ = AudioLoader(audio_types=['aiff'])


def test_loading_mp3():
    audio_loader = AudioLoader(audio_types=['mp3'])
    docs = DocumentArray(
        [
            Document(uri='tests/test_data/example_wav.wav'),
            Document(uri='tests/test_data/example_mp3.mp3'),
        ]
    )
    audio_loader.load_audio(docs, {})
    assert docs[0].blob is None
    assert docs[1].blob is not None


def test_loading_wav():
    audio_loader = AudioLoader(audio_types=['wav'])
    docs = DocumentArray(
        [
            Document(uri='tests/test_data/example_wav.wav'),
            Document(uri='tests/test_data/example_mp3.mp3'),
        ]
    )
    audio_loader.load_audio(docs, {})
    assert docs[0].blob is not None
    assert docs[1].blob is None


def test_loading_mixed():
    audio_loader = AudioLoader()
    docs = DocumentArray(
        [
            Document(uri='tests/test_data/example_wav.wav'),
            Document(uri='tests/test_data/example_mp3.mp3'),
        ]
    )
    audio_loader.load_audio(docs, {})
    assert docs[0].blob is not None
    assert docs[1].blob is not None
