from pathlib import Path

import librosa
import pytest
from executor.vggish import vggish_input
from executor.vggish_audio_encoder import VggishAudioEncoder
from jina import Document, DocumentArray, Executor
from tensorflow.python.framework import ops


@pytest.fixture(scope="module")
def encoder() -> VggishAudioEncoder:
    return VggishAudioEncoder()


@pytest.fixture(scope="module")
def gpu_encoder() -> VggishAudioEncoder:
    return VggishAudioEncoder(device='/GPU:0')


def test_config():
    ex = Executor.load_config(str(Path(__file__).parents[2] / 'config.yml'))
    assert str(ex.vgg_model_path).endswith('vggish_model.ckpt')
    assert str(ex.pca_model_path).endswith('vggish_pca_params.ckpt')


def test_no_documents(encoder: VggishAudioEncoder):
    docs = DocumentArray()
    encoder.encode(docs=docs, parameters={})
    assert len(docs) == 0  # SUCCESS


def test_embedding_dimension():
    x_audio, sample_rate = librosa.load(
        Path(__file__).parents[1] / 'test_data/sample.wav'
    )
    log_mel_examples = vggish_input.waveform_to_examples(x_audio, sample_rate)
    doc = DocumentArray([Document(blob=log_mel_examples)])
    ops.reset_default_graph()
    model = VggishAudioEncoder()
    model.encode(doc, parameters={})
    assert doc[0].embedding.shape == (128,)


@pytest.mark.gpu
def test_embedding_dimension_gpu():
    x_audio, sample_rate = librosa.load(
        Path(__file__).parents[1] / 'test_data/sample.wav'
    )
    log_mel_examples = vggish_input.waveform_to_examples(x_audio, sample_rate)
    doc = DocumentArray([Document(blob=log_mel_examples)])
    ops.reset_default_graph()
    model = VggishAudioEncoder(device='/GPU:0')
    model.encode(doc, parameters={})
    assert doc[0].embedding.shape == (128,)
