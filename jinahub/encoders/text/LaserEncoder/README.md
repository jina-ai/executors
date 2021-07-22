# LaserEncoder

**LaserEncoder** is a encoder based on Facebook Research's LASER (Language-Agnostic SEntence Representations) to compute multilingual sentence embeddings.

It encodes `Document` content from an 1d array of string in size `B` into an ndarray in size `B x D`.

<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**

- [🌱 Prerequisites](#-prerequisites)
- [🚀 Usages](#-usages)
- [🎉️ Example](#%EF%B8%8F-example)
- [🔍️ Reference](#%EF%B8%8F-reference)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

## 🌱 Prerequisites

```bash
python -m laserembeddings download-models
```

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
Use the prebuilt images from JinaHub in your python codes. The input language can be configured with `language`. The full list of possible values can be found at [LASER](https://github.com/facebookresearch/LASER#supported-languages) with the language code ([ISO 639-1](https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes)) 

```python
from jina import Flow
	
f = Flow().add(uses='jinahub+docker://laser-encoder', override_with={'language': 'en'})
```

or in the `.yml` config.
	
```yaml
jtype: Flow
pods:
  - name: encoder
    uses: 'jinahub+docker://laser-encoder'
```

#### using source codes
Use the source codes from JinaHub in your python codes,

```python
from jina import Flow
	
f = Flow().add(uses='jinahub://laser-encoder')
```

or in the `.yml` config.

```yaml
jtype: Flow
pods:
  - name: encoder
    uses: 'jinahub://laser-encoder'
```


### 📦️ Via Pypi

1. Install the package.

	```bash
	pip install git+https://github.com/jina-ai/executor-text-laser-encoder.git
	```

1. Use `LaserEncoder` in your code

	```python
	from jina import Flow
	from jinahub.encoder.laser_encoder import LaserEncoder
	
	f = Flow().add(uses=LaserEncoder)
	```


### 🐳 Via Docker

1. Clone the repo and build the docker image

	```shell
	git clone https://github.com/jina-ai/executor-text-laser-encoder.git
	cd executor-text-laser-encoder
	docker build -t executor-text-laser-encoder .
	```

1. Use `executor-text-laser-encoder` in your codes

	```python
	from jina import Flow
	
	f = Flow().add(uses='docker://executor-text-laser-encoder:latest')
	```
 
## 🎉 Example:

Here is an example usage of the **LaserEncoder**.

```python
from jina import Flow, Document
f = Flow().add(uses='jinahub+docker://LaserEncoder')
with f:
    resp = f.post(on='foo', inputs=Document(text='hello Jina'), return_results=True)
```

### Inputs 

`Document` with `text` to be encoded.

### Returns

`Document` with `embedding` fields filled with an `ndarray`  with `dtype=nfloat32`.
