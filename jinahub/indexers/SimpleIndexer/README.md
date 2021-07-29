# ✨ SimpleIndexer

**SimpleIndexer** is a Jina indexer, using the [DocumentArrayMemmap](https://github.com/jina-ai/jina/blob/master/jina/types/arrays/memmap.py) class as a storage system.

`DocumentArrayMemmap` stores the entire `Document` object, both vectors and metadata. It is also memory efficient, since it uses the [memmap module](https://docs.python.org/3.7/library/mmap.html) 

<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**

- [🌱 Prerequisites](#-prerequisites)
- [🚀 Usages](#-usages)
- [🎉️ Example](#%EF%B8%8F-example)
- [🔍️ Reference](#%EF%B8%8F-reference)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

## 🌱 Prerequisites

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
<details>

### 📦️ Via Pypi

1. Install the `executor-indexers` package.

	```bash
	pip install git+https://github.com/jina-ai/executor-indexers/
	```

1. Use `executor-indexers` in your code

   ```python
   from jina import Flow
   from jinahub.indexers.SimpleIndexer import SimpleIndexer
   
   f = Flow().add(uses=SimpleIndexer)
   ```


### 🐳 Via Docker

1. Clone the repo and build the docker image

	```shell
	git clone https://github.com/jina-ai/executor-indexers/
	cd jinahub/simple/SimpleIndexer
	docker build -t simple-indexer-image .
	```

1. Use `simple-indexer-image` in your codes

	```python
	from jina import Flow
	
	f = Flow().add(uses='docker://simple-indexer-image:latest')
	```
	
</details>

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


