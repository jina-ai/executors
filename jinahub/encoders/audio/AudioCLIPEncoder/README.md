
# ✨ AudioCLIPEncoder 

**AudioCLIPEncoder** is a class that wraps the [AudioCLIP](https://github.com/AndreyGuzhov/AudioCLIP) model for generating embeddings for audio data. 

**Table of Contents**

- [✨ AudioCLIPEncoder](#-audioclipencoder)
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

Run the provided bash script `scripts/download_model.sh` to download the pretrained model.

## 🚀 Usages

### 🚚 Via JinaHub

#### using docker images
Use the prebuilt images from JinaHub in your Python code: 

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

#### using source code
Use the source code from JinaHub in your Python code:

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
