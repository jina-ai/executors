# ✨ MongoStorage

**MongoStorage** is a Jina indexer, using [mongodb](https://www.mongodb.com/) as a backend. 

`mongodb` is a no-sql document data model storage system. It's naturally supports JSON and provide an expressive query language allows developers to learn and use. 

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

Additionally, you will need a running MongoDB instnace. This can be a local instance, a Docker image, or a virtual machine in the cloud. Make sure you have the credentials and connection parameters.

You can start one in a Docker container, like so: 

```bash
docker run --name mongo-storage  -p 127.0.0.1:27017:27017/tcp -d mongo:latest
```

## 🚀 Usages

### 🚚 Via JinaHub

#### using docker images

Use the prebuilt images from JinaHub in your Python code: 

```python
from jina import Flow
	
f = Flow().add(uses='jinahub+docker://MongoDBStorage')
```

or in the `.yml` config.
	
```yaml
jtype: Flow
pods:
  - name: indexer
    uses: 'jinahub+docker://MongoDBStorage'
```

#### using source codes
Use the source codes from JinaHub in your python codes,

```python
from jina import Flow
	
f = Flow().add(uses='jinahub://MongoDBStorage')
```

or in the `.yml` config.

```yaml
jtype: Flow
pods:
  - name: indexer
    uses: 'jinahub://MongoDBStorage'
```


### 📦️ Via Pypi

1. Install the `executor-indexers` package.

	```bash
	pip install git+https://github.com/jina-ai/executor-indexers/
	```

1. Use `executor-indexers` in your code

   ```python
   from jina import Flow
   from jinahub.indexers.storage.MongoDBStorage import MongoDBStorage
   
   f = Flow().add(uses= MongoDBStorage)
   ```


### 🐳 Via Docker

1. Clone the repo and build the docker image

	```shell
	git clone https://github.com/jina-ai/executor-indexers/
	cd jinahub/indexers/storage/MongoDBStorage
	docker build -t mongo-image .
	```

1. Use `mongo-image` in your codes

	```python
	from jina import Flow
	
	f = Flow().add(uses='docker://mongo-image:latest')
	```
	

## 🎉️ Example 

```python
from jina import Flow, Document

f = Flow().add(uses='jinahub://MongoDBStorage')

with f:
    resp = f.post(on='/index', inputs=Document(), return_results=True)
    print(f'{resp}')
```

Parameters:

- `default_traversal_paths`: the default traversal paths for the `DocumentArray` in a request. Can be overridden with `parameters={'traversal_paths': ..}` 

Check [tests](tests/test_mongodb.py) for more usage scenarios.


### Inputs 

`Document`, with any data. It is stored in full, in bytes.

### Returns

Nothing
