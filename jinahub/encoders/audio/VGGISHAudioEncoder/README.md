
# VggishAudioEncoder

**VggishAudioEncoder** is a class that wraps the [VGGISH](https://github.com/tensorflow/models/tree/master/research/audioset/vggish) model for generating embeddings for audio data. 



## Prerequisites


> These are only needed if you download the source code and directly use the class. Not needed if you use the Jina Hub method below.

Run the provided bash script `download_model.sh` to download the pretrained model.



## Usage 

With fake data

```python
import numpy as np
from jina import Flow, Document, DocumentArray

f = Flow().add(uses='jinahub+docker://VGGishAudioEncoder', timeout_ready=3000)

fake_log_mel_examples = np.random.random((2,96,64))
doc_array = DocumentArray([Document(blob=fake_log_mel_examples)])

with f:
    resp = f.post(on='test', inputs=doc_array, return_results=True)
		print(f'{resp}')
```

Example with real data


```python
import librosa
from jina import Flow, Document, DocumentArray
from vggish import vggish_input

f = Flow().add(uses='jinahub+docker://VGGishAudioEncoder', timeout_ready=3000)

# Load data
x_audio, sample_rate = librosa.load('./data/sample.wav')
log_mel_examples = vggish_input.waveform_to_examples(x_audio, sample_rate)
doc_array = DocumentArray([Document(blob=log_mel_examples)])

with f:
    resp = f.post(on='test', inputs=doc_array, return_results=True)
    
print(f'{resp}')
```

#### GPU usage

You can use the GPU via the source code. Therefore, you need a matching CUDA version
and GPU drivers installed on your system. 
```yaml
jtype: Flow
pods:
  - name: encoder
    uses: 'jinahub://VGGISHAudioEncoder'
    uses_with:
      device: 'cuda'
```
Alternatively, use the jinahub gpu docker container. Therefore, you need GPU
drivers installed on your system and nvidia-docker installed.
```yaml
jtype: Flow
pods:
  - name: encoder
    uses: 'jinahub+docker://VGGISHAudioEncoder/gpu'
    gpus: all
    uses_with:
      device: 'cuda'
```


### Inputs 

`Document` with `blob` of containing loaded audio.

### Returns

`Document` with `embedding` fields filled with an `ndarray` of the shape `embedding_dim` with `dtype=nfloat32`.


## Reference
- [VGGISH paper](https://research.google/pubs/pub45611/)
- [VGGISH code](https://github.com/tensorflow/models/tree/master/research/audioset/vggish)
