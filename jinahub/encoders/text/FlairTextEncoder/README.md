# FlairTextEncoder

`FlairTextEncoder` encodes `Document` content from an array of string in size `B` into a ndarray in size `B x D`.
 
Internally, `FlairTextEncoder` wraps the DocumentPoolEmbeddings from Flair.

## 🌱 Prerequisites

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
	
f = Flow().add(uses='jinahub+docker://FlairTextEncoder')
```

or in the `.yml` config.
	
```yaml
jtype: Flow
pods:
  - name: encoder
    uses: 'jinahub+docker://FlairTextEncoder'
```

#### using source codes
Use the source codes from JinaHub in your python codes,

```python
from jina import Flow
	
f = Flow().add(uses='jinahub://FlairTextEncoder')
```

or in the `.yml` config.

```yaml
jtype: Flow
pods:
  - name: encoder
    uses: 'jinahub://FlairTextEncoder'
```


### 📦️ Via Pypi

1. Install the package.

	```bash
	pip install git+https://github.com/jina-ai//executor-text-flair-encoder.git
	```

1. Use `FlairTextEncoder` in your code

	```python
	from jina import Flow
	from jinahub.encoder.flair_text import FlairTextEncoder
	
	f = Flow().add(uses=FlairTextEncoder)
	```


### 🐳 Via Docker

1. Clone the repo and build the docker image

	```shell
	git clone https://github.com/jina-ai/executor-text-flair-encoder.git
	cd executor-text-flair-encoder
	docker build -t executor-text-flair-encoder .
	```

1. Use `executor-text-flair-encoder` in your codes

	```python
	from jina import Flow
	
	f = Flow().add(uses='docker://executor-text-flair-encoder:latest')
	```
 
## 🎉 Example:

Here is an example usage of the **FlairTextEncoder**.

```python
from jina import Flow, Document
f = Flow().add(uses='jinahub+docker://FlairTextEncoder')
with f:
    resp = f.post(on='foo', inputs=Document(text='hello Jina'), return_results=True)
```

### Inputs 

`Document` with `text` to be encoded.

### Returns

`Document` with `embedding` fields filled with an `ndarray`  with `dtype=nfloat32`.
