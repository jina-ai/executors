from pathlib import Path

import librosa
import pytest
from jina import Document, DocumentArray, Executor
from tensorflow.python.framework import ops

from ...vggish import vggish_input
from ...vggish_audio_encoder import VggishAudioEncoder


def test_config():
    ex = Executor.load_config(str(Path(__file__).parents[2] / 'config.yml'))
    assert str(ex.vgg_model_path).endswith('vggish_model.ckpt')
    assert str(ex.pca_model_path).endswith('vggish_pca_params.ckpt')


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
def test_embedding_dimension():
    x_audio, sample_rate = librosa.load(
        Path(__file__).parents[1] / 'test_data/sample.wav'
    )
    log_mel_examples = vggish_input.waveform_to_examples(x_audio, sample_rate)
    doc = DocumentArray([Document(blob=log_mel_examples)])
    ops.reset_default_graph()
    model = VggishAudioEncoder(device='cuda')
    model.encode(doc, parameters={})
    assert doc[0].embedding.shape == (128,)
