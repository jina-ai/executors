# ✨ ImageTFEncoder

`ImageTFEncoder` encodes ``Document`` content from a ndarray, potentially BatchSize x (Height x Width x Channel) into a ndarray of `BatchSize * d`. Internally, :class:`ImageTFEncoder` wraps the models from `tensorflow.keras.applications`. https://keras.io/applications/


<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**

- [🌱 Prerequisites](#-prerequisites)
- [🚀 Usages](#-usages)
- [🎉️ Example](#%EF%B8%8F-example)
- [🔍️ Reference](#%EF%B8%8F-reference)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

## 🌱 Prerequisites

Some conditions to fulfill before running the executor

## 🚀 Usages

To install the dependencies locally run 
```
pip install . 
pip install -r tests/requirements.txt
```
To verify the installation works:
```
pytest tests
```

### 🚚 Via JinaHub

#### using docker images
Use the prebuilt images from JinaHub in your python codes, 

```python
from jina import Flow
	
f = Flow().add(uses='jinahub+docker://ImageTFEncoder')
```

or in the `.yml` config.
	
```yaml
jtype: Flow
pods:
  - name: encoder
    uses: 'jinahub+docker://ImageTFEncoder'
```

#### using source codes
Use the source codes from JinaHub in your python codes,

```python
from jina import Flow
	
f = Flow().add(uses='jinahub://ImageTFEncoder')
```

or in the `.yml` config.

```yaml
jtype: Flow
pods:
  - name: encoder
    uses: 'jinahub://ImageTFEncoder'
```


### 📦️ Via Pypi

1. Install the `executor-image-tf-encoder` package.

	```bash
	pip install git+https://github.com/jina-ai/executor-image-tf-encoder
	```

1. Use `jinahub-executor-image-tf-encoder` in your code

	```python
	from jina import Flow
	from jinahub.encoder.image_tf_encoder import ImageTFEncoder
	
	f = Flow().add(uses=ImageTFEncoder)
	```


### 🐳 Via Docker

1. Clone the repo and build the docker image

	```shell
	git clone https://github.com/jina-ai/executor-image-tf-encoder
	cd executor-image-tf-encoder
	docker build -t executor-image-tf-encoder .
	```

1. Use `executor-image-tf-encoder` in your codes

	```python
	from jina import Flow
	
	f = Flow().add(uses='docker://executor-image-tf-encoder:latest')
	```
	

## 🎉️ Example 


```python
from jina import Flow, Document

f = Flow().add(uses='jinahub+docker://ImageTFEncoder')

with f:
    resp = f.post(on='foo', inputs=Document(), return_results=True)
	print(f'{resp}')
```

### Inputs 

`Document` with `blob` of the shape `256`.

### Returns

`Document` with `embedding` fields filled with an `ndarray` of the shape `embedding_dim` (=128, by default) with `dtype=nfloat32`.
