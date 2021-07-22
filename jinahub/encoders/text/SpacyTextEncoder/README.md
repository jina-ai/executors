# ✨ SpacyTextEncoder

**SpacyTextEncoder** is a class that encodes text with spaCy models.

<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**

- [🌱 Prerequisites](#-prerequisites)
- [🚀 Usages](#-usages)
- [🎉️ Example](#%EF%B8%8F-example)
- [🔍️ Reference](#%EF%B8%8F-reference)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

## 🌱 Prerequisites

None
## 🚀 Usages

### 🚚 Via JinaHub

#### using docker images
Use the prebuilt images from JinaHub in your python codes, 

```python
from jina import Flow
	
f = Flow().add(uses='jinahub+docker://SpacyTextEncoder')
```

or in the `.yml` config.
	
```yaml
jtype: Flow
pods:
  - name: encoder
    uses: 'jinahub+docker://SpacyTextEncoder'
```

#### using source codes
Use the source codes from JinaHub in your python codes,

```python
from jina import Flow
	
f = Flow().add(uses='jinahub://SpacyTextEncoder')
```

or in the `.yml` config.

```yaml
jtype: Flow
pods:
  - name: encoder
    uses: 'jinahub://SpacyTextEncoder'
```


### 📦️ Via Pypi

1. Install the `jinahub-spacy-text-encoder` package.

	```bash
	pip install git+https://github.com/jina-ai/executor-text-spacy-encoder.git
	```

1. Use `jinahub-spacy-text-encoder` in your code

	```python
	from jina import Flow
	from jinahub.encoder.spacy_text_encoder import SpacyTextEncoder
	
	f = Flow().add(uses=SpacyTextEncoder)
	```


### 🐳 Via Docker

1. Clone the repo and build the docker image

	```shell
	git clone https://github.com/jina-ai/executor-text-spacy-encoder.git
	cd executor-text-spacy-encoder
	docker build -t executor-text-spacy-encoder-image .
	```

1. Use `executor-text-spacy-encoder-image` in your codes

	```python
	from jina import Flow
	
	f = Flow().add(uses='docker://executor-text-spacy-encoder-image:latest')
	```
	

## 🎉️ Example 


```python
from jina import Flow, Document

f = Flow().add(uses='jinahub+docker://SpacyTextEncoder')

with f:
    resp = f.post(on='foo', inputs=Document(), return_results=True)
	print(f'{resp}')
```


#### Inputs 

`Document` with `text`.

#### Returns

`Document` with `embedding` field filled with spacy vector.

## 🔍️ Reference
- https://spacy.io/models/
- https://spacy.io/usage/models
