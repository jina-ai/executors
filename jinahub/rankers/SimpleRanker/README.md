# ✨ SimpleRanker 

**SimpleRanker** is a class aggregates the score of the matched doc from the matched chunks.

<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**

- [✨ SimpleRanker](#-simpleranker)
  - [🌱 Prerequisites](#-prerequisites)
  - [🚀 Usages](#-usages)
    - [🚚 Via JinaHub](#-via-jinahub)
      - [using docker images](#using-docker-images)
      - [using source code](#using-source-code)
  - [🎉️ Example](#️-example)
  - [🔍️ Reference](#️-reference)

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
	
f = Flow().add(uses='jinahub+docker://SimpleRanker')
```

or in the `.yml` config.
	
```yaml
jtype: Flow
pods:
  - name: encoder
    uses: 'jinahub+docker://SimpleRanker'
    uses_with:
      metric: 'cosine'
      ranking: 'min'    # Other options are 'max', 'mean_max', 'mean_min'
```

#### using source code
Use the source code from JinaHub in your Python code:

```python
from jina import Flow
	
f = Flow().add(uses='jinahub://SimpleRanker')
```

or in the `.yml` config.

```yaml
jtype: Flow
pods:
  - name: encoder
    uses: 'jinahub://SimpleRanker'
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

f = Flow().add(uses='jinahub://SimpleRanker', uses_with={'metric': 'cosine'})

with f:
    resp = f.post(on='/search', inputs=document_array, return_results=True)
    print(f'{resp[0].data.docs[0].matches}')

```


## 🔍️ Reference
- See the [multires lyrics search example](https://github.com/jina-ai/examples/tree/master/multires-lyrics-search) for example usage
