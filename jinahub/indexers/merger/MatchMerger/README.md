# ✨ MatchMerger

**MatchMerger** Merges the results of shards by appending all matches. Assume you have 20 shards and use `top-k=10`, you will get 200 results in the merger.
The `MatchMerger` is used in the `uses_after` attribute when adding an `Executor` to the `Flow`.

**Table of Contents**

- [🌱 Prerequisites](#-prerequisites)
- [🚀 Usages](#-usages)
- [🎉️ Example](#-example)
- [🔍️ Reference](#-reference)


## 🌱 Prerequisites

> These are only needed if you download the source code and directly use the class. Not needed if you use the Jina Hub method below.

- This Executor works on Python 3.7 and 3.8. 
- Make sure to install the [requirements](requirements.txt)

## 🚀 Usages

Check [tests](tests) for an example on how to use it.

### 🚚 Via JinaHub

#### using docker images
Use the prebuilt images from JinaHub in your Python code: 

```python
from jina import Flow
	
f = Flow().add(uses='jinahub+docker://AnnoySearcher')
```

or in the `.yml` config.
	
```yaml
jtype: Flow
pods:
  - name: indexer
    uses: 'jinahub+docker://AnnoySearcher'
```

#### using source code
Use the source code from JinaHub in your code

```python
from jina import Flow
	
f = Flow().add(uses='jinahub://AnnoySearcher')
```

or in the `.yml` config.

```yaml
jtype: Flow
pods:
  - name: indexer
    uses: 'jinahub://AnnoySearcher'
```


## 🎉️ Example 

```python
from jina import Flow, Document

f = Flow().add(uses='jinahub+docker://AnnoySearcher')

with f:
    resp = f.post(on='/search', inputs=Document(), return_results=True)
    print(f'{resp}')
```

### Inputs 

`Document` with `.docs_matrix` of type `List[DocumentArray]`. It contains all the results calculated from the shards.

### Returns

`DocumentArray` which contains all the documents the user searched for including the combined matches from all the shards.

