
# ✨ AudioCLIPEncoder

**AudioCLIPEncoder** is a class that wraps the [AudioCLIP](https://github.com/AndreyGuzhov/AudioCLIP) model for generating embeddings for audio data. 

<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**

- [🌱 Prerequisites](#-prerequisites)
- [🚀 Usages](#-usages)
- [🎉️ Example](#%EF%B8%8F-example)
- [🔍️ Reference](#%EF%B8%8F-reference)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

## 🌱 Prerequisites

Run the provided bash script `scripts/download_model.sh` to download the pretrained model.

## 🚀 Usages

### 🚚 Via JinaHub

#### using docker images
Use the prebuilt images from JinaHub in your python codes, 

```python
from jina import Flow
	
f = Flow().add(uses='jinahub+docker://AudioCLIPEncoder')
```

or in the `.yml` config.
```yaml
jtype: Flow
pods:
  - name: encoder
    uses: 'jinahub+docker://AudioCLIPEncoder'
```

#### using source codes
Use the source codes from JinaHub in your python codes,

```python
from jina import Flow
	
f = Flow().add(uses='jinahub://AudioCLIPEncoder')
```

or in the `.yml` config.

```yaml
jtype: Flow
pods:
  - name: encoder
    uses: 'jinahub://AudioCLIPEncoder'
```


### 📦️ Via Pypi

1. Install the `jinahub-AudioCLIPEncoder` package.

	```bash
	pip install git+https://github.com/jina-ai/executor-audio-clip-encoder.git
	```

1. Use `jinahub-vggishaudio-encoder` in your code

	```python
	from jina import Flow
	from jinahub.encoder.audioclip import AudioCLIPEncoder
	
	f = Flow().add(uses='jinahub+docker://AudioCLIPEncoder')
	```


### 🐳 Via Docker

1. Clone the repo and build the docker image

	```shell
	git clone https://github.com/jina-ai/executor-audio-clip-encoder.git
	cd executor-audio-clip-encoder
	docker build -t executor-audio-clip-encoder .
	```

1. Use `executor-audio-clip-encoder` in your codes

	```python
	from jina import Flow
	
	f = Flow().add(uses='docker://executor-audio-clip-encoder:latest')
	```

## 🎉️ Example 

With fake data

```python
import numpy as np
from jina import Flow, Document, DocumentArray

f = Flow().add(uses='jinahub+docker://AudioCLIPEncoder', timeout_ready=3000)

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

f = Flow().add(uses='jinahub+docker://AudioCLIPEncoder', timeout_ready=3000)

# Load data
x_audio, sample_rate = librosa.load('./data/sample.wav')
doc_array = DocumentArray([Document(blob=x_audio)])

with f:
    resp = f.post(on='test', inputs=doc_array, return_results=True)
    
print(f'{resp}')
```





### Inputs 

`Document` with `blob` of containing loaded audio.

### Returns

`Document` with `embedding` fields filled with an `ndarray` of the shape `embedding_dim` with `dtype=nfloat32`.


## 🔍️ Reference
- [AudioCLIP paper](https://arxiv.org/abs/2106.13043)
- [AudioCLIP code](https://github.com/AndreyGuzhov/AudioCLIP)
