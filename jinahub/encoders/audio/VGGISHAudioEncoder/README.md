
# ✨ VggishAudioEncoder 

**VggishAudioEncoder** is a class that wraps the [VGGISH](https://github.com/tensorflow/models/tree/master/research/audioset/vggish) model for generating embeddings for audio data. 


**Table of Contents**

- [✨ VggishAudioEncoder](#-vggishaudioencoder)
  - [🌱 Prerequisites](#-prerequisites)
  - [🚀 Usages](#-usages)
    - [🚚 Via JinaHub](#-via-jinahub)
      - [using docker images](#using-docker-images)
      - [using source code](#using-source-code)
  - [🎉️ Example](#️-example)
    - [Inputs](#inputs)
    - [Returns](#returns)
  - [🔍️ Reference](#️-reference)

## 🌱 Prerequisites


> These are only needed if you download the source code and directly use the class. Not needed if you use the Jina Hub method below.

Run the provided bash script `download_model.sh` to download the pretrained model.

To install the dependencies locally run 
```
pip install . 
pip install -r tests/requirements.txt
```
To verify the installation works:
```
pytest tests
```

## 🚀 Usages

### 🚚 Via JinaHub

#### using docker images
Use the prebuilt images from JinaHub in your Python code: 

```python
from jina import Flow
	
f = Flow().add(uses='jinahub+docker://VGGishAudioEncoder')
```

or in the `.yml` config.
```yaml
jtype: Flow
pods:
  - name: encoder
    uses: 'jinahub+docker://VGGishAudioEncoder'
```

#### using source code
Use the source code from JinaHub in your Python code:

```python
from jina import Flow
	
f = Flow().add(uses='jinahub://VGGishAudioEncoder')
```

or in the `.yml` config.

```yaml
jtype: Flow
pods:
  - name: encoder
    uses: 'jinahub://VGGishAudioEncoder'
```


## 🎉️ Example 

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


### Inputs 

`Document` with `blob` of containing loaded audio.

### Returns

`Document` with `embedding` fields filled with an `ndarray` of the shape `embedding_dim` with `dtype=nfloat32`.


## 🔍️ Reference
- [VGGISH paper](https://research.google/pubs/pub45611/)
- [VGGISH code](https://github.com/tensorflow/models/tree/master/research/audioset/vggish)
