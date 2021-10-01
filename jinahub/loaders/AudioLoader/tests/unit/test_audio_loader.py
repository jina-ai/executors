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


def test_loading_wav():
    audio_loader = AudioLoader(audio_types=['wav'])


def test_loading_mixed():
    audio_loader = AudioLoader()
