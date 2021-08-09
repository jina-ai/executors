# ✨  SimpleIndexer

**SimpleIndexer** is a Jina indexer, using the [DocumentArrayMemmap](https://github.com/jina-ai/jina/blob/master/jina/types/arrays/memmap.py) class as a storage system.

`DocumentArrayMemmap` stores the entire `Document` object, both vectors and metadata. It is also memory efficient, since it uses the [memmap module](https://docs.python.org/3.7/library/mmap.html) 

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

### 🚚 Via JinaHub

#### using docker images

Use the prebuilt images from JinaHub in your Python code: 

```python
from jina import Flow
	
f = Flow().add(uses='jinahub+docker://SimpleIndexer')
```

or in the `.yml` config.
	
```yaml
jtype: Flow
pods:
  - name: indexer
    uses: 'jinahub+docker://SimpleIndexer'
```

#### using source code
Use the source code from JinaHub in your Python code:

```python
from jina import Flow
	
f = Flow().add(uses='jinahub://SimpleIndexer')
```

or in the `.yml` config.

```yaml
jtype: Flow
pods:
  - name: indexer
    uses: 'jinahub://SimpleIndexer'
```

## 🎉️ Example 

```python
from jina import Flow, Document

f = Flow().add(uses='jinahub://SimpleIndexer')

with f:
    resp = f.post(on='/index', inputs=Document(), return_results=True)
    print(f'{resp}')
```

Parameters:

- `index_file_name`: the name of the folder where the memmaped data will be, under the workspace

### Inputs 

`Document`, with any data. It is stored in a `DocumentArrayMemmap`

### Returns

Nothing

## 🔍️ Reference
