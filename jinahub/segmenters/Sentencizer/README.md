# ✨ Sentencizer

**Sentencizer** is a class that splits texts into sentences.

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
Use the prebuilt images from JinaHub in your Python code: 

```python
from jina import Flow
	
f = Flow().add(uses='jinahub+docker://Sentencizer')
```

or in the `.yml` config.
	
```yaml
jtype: Flow
pods:
  - name: sentencizer
    uses: 'jinahub+docker://Sentencizer'
```

#### using source codes
Use the source codes from JinaHub in your Python code:

```python
from jina import Flow
	
f = Flow().add(uses='jinahub://Sentencizer')
```

or in the `.yml` config.

```yaml
jtype: Flow
pods:
  - name: sentencizer
    uses: 'jinahub://Sentencizer'
```
<details>

### 📦️ Via Pypi

1. Install the `executors` package.

	```bash
	pip install git+https://github.com/jina-ai/executors
	```

1. Use `Sentencizer` in your code

	```python
	from jina import Flow
	from jinahub.crafters.Sentencizer.sentencizer import Sentencizer
	
	f = Flow().add(uses=Sentencizer)
	```


### 🐳 Via Docker

1. Clone the repo and build the docker image

	```shell
	git clone https://github.com/jina-ai/executors
	cd executors/jinahub/crafters/Sentencizer
	docker build -t sentencizer .
	```

1. Use `sentencizer` in your code

	```python
	from jina import Flow
	
	f = Flow().add(uses='docker://sentencizer:latest')
	```
	
</details>

## 🎉️ Example 

```python
from jina import Flow, Document

f = Flow().add(uses='jinahub+docker://Sentencizer')

with f:
    resp = f.post(on='foo', inputs=Document(text='Hello. World.'), return_results=True)
    print(f'{resp}')
```

#### Inputs 

`Document` with `text` containing two sentences split by a dot `.`, namely `Hello. World.`.

#### Returns

`Document` with two `chunks` Documents. The first chunk contains `text='Hello.'`, the second chunk contains `text='World.'`

## 🔍️ Reference
- Used in the multires-lyrics-search example in: https://github.com/jina-ai/examples

