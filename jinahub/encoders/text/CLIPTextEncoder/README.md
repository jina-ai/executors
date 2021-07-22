# ✨ CLIPTextEncoder

 **CLIPTextEncoder** is a class that wraps the text embedding functionality from the **CLIP** model.

The **CLIP** model was originally proposed in  [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020).

`CLIPTextEncoder` encodes data from a `np.ndarray` of strings and returns a `np.ndarray` of floating point values.

- Input shape: `BatchSize `

- Output shape: `BatchSize x EmbeddingDimension`

<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**

- [🌱 Prerequisites](#-prerequisites)
- [🚀 Usages](#-usages)
- [🎉️ Example](#%EF%B8%8F-example)
- [🔍️ Reference](#%EF%B8%8F-reference)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

## 🌱 Prerequisites

No prerequisites are required to run this executor.

## 🚀 Usages

### 🚚 Via JinaHub

Use the prebuilt images from JinaHub in your python codes, 

```python
from jina import Flow
	
f = Flow().add(
        uses='jinahub+docker://CLIPTextEncoder',
        volumes='/your_home_folder/.cache/clip:/root/.cache/clip'
	)
```

or in the `.yml` config.
	
```yaml
jtype: Flow
pods:
  - name: encoder
    uses: 'jinahub+docker://CLIPTextEncoder'
    volumes: '/your_home_folder/.cache/clip:/root/.cache/clip'
```


### 📦️ Via Pypi

1. Install the `jinahub-text-clip-text-encoder`

	```bash
	pip install git+https://github.com/jina-ai/executor-text-clip-text-encoder.git
	```

1. Use `jinahub-text-clip-text-encoder` in your code

	```python
	from jinahub.encoder.clip_text import CLIPTextEncoder
	from jina import Flow
	
	f = Flow().add(uses=CLIPTextEncoder)
	```


### 🐳 Via Docker

1. Clone the repo and build the docker image

	```shell
	git clone https://github.com/jina-ai/executor-text-clip-text-encoder.git
	cd executor-text-CLIP
	docker build -t jinahub-clip-text .
	```

2. Use `jinahub-clip-text` in your code

	```python
	from jina import Flow
	
	f = Flow().add(
	        uses='docker://jinahub-clip-text:latest',
	        volumes='/your_home_folder/.cache/clip:/root/.cache/clip'
		)
	```
	


## 🎉️ Example 


```python
from jina import Flow, Document
import numpy as np
	
f = Flow().add(
        uses='jinahub+docker://CLIPTextEncoder',
        volumes='/your_home_folder/.cache/clip:/root/.cache/clip'
	)
	
def check_emb(resp):
    for doc in resp.data.docs:
        if doc.emb:
            assert doc.emb.shape == (512,)
	
with f:
	f.post(
	    on='/foo', 
	    inputs=Document(text='your text'), 
	    on_done=check_emb
	)
	    
```


### Inputs 

[Documents](https://github.com/jina-ai/jina/blob/master/.github/2.0/cookbooks/Document.md) with the [`text`](https://github.com/jina-ai/jina/blob/master/.github/2.0/cookbooks/Document.md#document-attributes) attribute.

### Returns

[Documents](https://github.com/jina-ai/jina/blob/master/.github/2.0/cookbooks/Document.md) with the `embedding` attribute filled with an `ndarray` of the shape `512` with `dtype=float32`.



## 🔍️ Reference

- [CLIP blog post](https://openai.com/blog/clip/)
- [CLIP paper](https://arxiv.org/abs/2103.00020)
- [CLIP GitHub repository](https://github.com/openai/CLIP)
