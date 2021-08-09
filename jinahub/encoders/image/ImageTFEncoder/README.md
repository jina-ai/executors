# ✨  ImageTFEncoder

`ImageTFEncoder` encodes ``Document`` content from a ndarray, potentially BatchSize x (Height x Width x Channel) into a ndarray of `BatchSize * d`. Internally, :class:`ImageTFEncoder` wraps the models from `tensorflow.keras.applications`. https://keras.io/applications/

**Table of Contents**

- [🌱 Prerequisites](#-prerequisites)
- [🚀 Usages](#-usages)
- [🎉️ Example](#-example)
- [🔍️ Reference](#-reference)


## 🌱 Prerequisites

> These are only needed if you download the source code and directly use the class. Not needed if you use the Jina Hub method below.

In case you want to install the dependencies locally run 
```
pip install -r requirements.txt
```

## 🚀 Usages

### 🚚 Via JinaHub

#### using docker images
Use the prebuilt images from JinaHub in your Python code: 

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

#### using source code
Use the source code from JinaHub in your Python code:

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

## 🔍️ Reference
