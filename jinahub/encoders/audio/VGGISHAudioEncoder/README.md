
# ✨ VggishAudioEncoder

**VggishAudioEncoder** is a class that wraps the [VGGISH](https://github.com/tensorflow/models/tree/master/research/audioset/vggish) model for generating embeddings for audio data. 

<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**

- [🌱 Prerequisites](#-prerequisites)
- [🚀 Usages](#-usages)
- [🎉️ Example](#%EF%B8%8F-example)
- [🔍️ Reference](#%EF%B8%8F-reference)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

## 🌱 Prerequisites

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
Use the prebuilt images from JinaHub in your python codes, 

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

#### using source codes
Use the source codes from JinaHub in your python codes,

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


### 📦️ Via Pypi

1. Install the `jinahub-VGGishAudioEncoder` package.

	```bash
	pip install git+https://github.com/jina-ai/executor-audio-VGGishEncoder.git
	```

1. Use `jinahub-vggishaudio-encoder` in your code

	```python
	from jina import Flow
	from jinahub.encoder.jinahub-vggishaudio-encoder import VggishAudioEncoder
	
	f = Flow().add(uses='jinahub+docker://VggishAudioEncoder')
	```


### 🐳 Via Docker

1. Clone the repo and build the docker image

	```shell
	git clone https://github.com/jina-ai/executor-audio-VggishAudioEncoder.git
	cd executor-audio-VGGishEncoder
	docker build -t executor-audio-vggish-encoder-image .
	```

1. Use `executor-audio-vggish-encoder-image` in your codes

	```python
	from jina import Flow
	
	f = Flow().add(uses='docker://executor-audio-vggish-encoder-image:latest')
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

