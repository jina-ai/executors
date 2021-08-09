# ✨  Executor Sentence Encoder 

**TransformerSentenceEncoder** wraps the [Sentence Transformer](https://www.sbert.net/docs)
library into an `Jina` executor. 


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
	
f = Flow().add(uses='jinahub+docker://TransformerSentenceEncoder')
```

or in the `.yml` config.
	
```yaml
jtype: Flow
pods:
  - name: encoder
    uses: 'jinahub+docker://TransformerSentenceEncoder'
```

#### using source code
Use the source code from JinaHub in your Python code:

```python
from jina import Flow
	
f = Flow().add(uses='jinahub://TransformerSentenceEncoder')
```

or in the `.yml` config.

```yaml
jtype: Flow
pods:
  - name: encoder
    uses: 'jinahub://TransformerSentenceEncoder'
```


## 🎉️ Example 

```python
from jina import Flow, Document

f = Flow().add(uses='docker://executor-sentence-transformer:latest')

with f:
    resp = f.post(on='foo', inputs=Document(), return_results=True)
	print(f'{resp}')
```

### Inputs 

`Document` with `text` sentences.

### Returns

`Document` with `embedding` fields filled with an `ndarray` of the shape `embedding_dim` (depends on the used model) with `dtype=nfloat32`.


## 🔍️ Reference
- [Sentence Transformer Library](https://www.sbert.net/docs)
