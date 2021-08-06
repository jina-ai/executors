import os
import librosa

from jina import Executor, Document, DocumentArray
from tensorflow.python.framework import ops

from ...vggish import vggish_input
from ...vggish_audio_encoder import VggishAudioEncoder

cur_dir = os.path.dirname(os.path.abspath(__file__))


def test_load():
    encoder = Executor.load_config(os.path.join(cur_dir, '../../config.yml'))
    assert str(encoder.vgg_model_path).endswith('vggish_model.ckpt')
    assert str(encoder.pca_model_path).endswith('vggish_pca_params.ckpt')


def test_embedding_dimension():
    x_audio, sample_rate = librosa.load(os.path.join(cur_dir, '../test_data/sample.wav'))
    log_mel_examples = vggish_input.waveform_to_examples(x_audio, sample_rate)
    doc = DocumentArray([Document(blob=log_mel_examples)])
    ops.reset_default_graph()
    model = VggishAudioEncoder()
    model.encode(doc, parameters={})
    assert doc[0].embedding.shape == (128,)
