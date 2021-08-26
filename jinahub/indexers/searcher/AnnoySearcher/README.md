# ✨ AnnoySearcher 

**AnnoySearcher** is an Annoy-powered vector-based similarity searcher. Annoy stands for "Approximate Nearest Neighbors Oh Yeah", and is a C++ library with Python bindings to search for points in space that are close to a given query point. It also creates large read-only file-based data structures that are mmapped into memory so that many processes may share the same data.

For more information, refer to the GitHub repo for [Spotify's Annoy](https://github.com/spotify/annoy).

**Table of Contents**

- [✨ AnnoySearcher](#-annoysearcher)
  - [🌱 Prerequisites](#-prerequisites)
  - [🚀 Usages](#-usages)
    - [Loading data](#loading-data)
    - [🚚 Via JinaHub](#-via-jinahub)
      - [using docker images](#using-docker-images)
      - [using source code](#using-source-code)
  - [🎉️ Example](#️-example)
    - [Inputs](#inputs)
    - [Returns](#returns)
  - [🔍️ Reference](#️-reference)


## 🌱 Prerequisites

> These are only needed if you download the source code and directly use the class. Not needed if you use the Jina Hub method below.

- This Executor works on Python 3.7 and 3.8. 
- Make sure to install the [requirements](requirements.txt)

## 🚀 Usages

Check [tests](tests) for an example on how to use it.

### Loading data

Since this is a "Searcher"-type Executor, it does not _index_ new data. Rather they are write-once classes, which take as data source a `dump_path`. 

This can be provided in different ways:

- in the YAML definition
  
```yaml
jtype: AnnoySearcher
with:
    dump_path: /tmp/your_dump_location
...
```

- from the `Flow.rolling_update` method. See [README](../../../../README.md).

The folder needs to contain the data exported from your Indexer. Again, see [README](../../../../README.md).

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

`Document` with `.embedding` the same shape as the `Documents` it has stored.

### Returns

Attaches matches to the Documents sent as inputs, with the id of the match, and its embedding. For retrieving the full metadata (original text or image blob), use a [key-value searcher](./../../keyvalue).


## 🔍️ Reference

- https://github.com/spotify/annoy
