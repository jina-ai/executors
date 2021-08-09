# ✨  MinRanker

**MinRanker** is a class aggregates the score of the matched doc from the matched chunks.

<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**

- [🌱 Prerequisites](#-prerequisites)
- [🚀 Usages](#-usages)
- [🎉️ Example](#%EF%B8%8F-example)
- [🔍️ Reference](#%EF%B8%8F-reference)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

## 🌱 Prerequisites

> These are only needed if you download the source code and directly use the class. Not needed if you use the Jina Hub method below.  

There are no requirements for running this executor.

## 🚀 Usages

### 🚚 Via JinaHub

#### using docker images
Use the prebuilt images from JinaHub in your Python code: 

```python
from jina import Flow
	
f = Flow().add(uses='jinahub+docker://MinRanker')
```

or in the `.yml` config.
	
```yaml
jtype: Flow
pods:
  - name: encoder
    uses: 'jinahub+docker://MinRanker'
```

#### using source code
Use the source code from JinaHub in your Python code:

```python
from jina import Flow
	
f = Flow().add(uses='jinahub://MinRanker')
```

or in the `.yml` config.

```yaml
jtype: Flow
pods:
  - name: encoder
    uses: 'jinahub://MinRanker'
```
	

## 🎉️ Example 

```python
from jina import Flow, DocumentArray, Document
import random

document_array = DocumentArray()
document = Document(tags={'query_size': 35, 'query_price': 31, 'query_brand': 1})
for i in range(0, 10):
    chunk = Document()
    for j in range(0, 10):
        match = Document(
            tags={
                'level': 'chunk',
            }
        )
        match.scores['cosine'] = random.random()
        match.parent_id = i
        chunk.matches.append(match)
    document.chunks.append(chunk)

document_array.extend([document])

f = Flow().add(uses='jinahub://MinRanker', override_with={'metric': 'cosine'})

with f:
    resp = f.post(on='/search', inputs=document_array, return_results=True)
    print(f'{resp[0].data.docs[0].matches}')

```


## 🔍️ Reference
- See the [multires lyrics search example](https://github.com/jina-ai/examples/tree/master/multires-lyrics-search) for example usage
