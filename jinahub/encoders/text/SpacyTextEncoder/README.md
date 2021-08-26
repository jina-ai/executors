# ✨ SpacyTextEncoder 

**SpacyTextEncoder** is a class that encodes text with spaCy models.


**Table of Contents**

- [✨ SpacyTextEncoder](#-spacytextencoder)
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
	
f = Flow().add(uses='jinahub+docker://SpacyTextEncoder')
```

or in the `.yml` config.
	
```yaml
jtype: Flow
pods:
  - name: encoder
    uses: 'jinahub+docker://SpacyTextEncoder'
```

#### using source code
Use the source code from JinaHub in your Python code:

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
