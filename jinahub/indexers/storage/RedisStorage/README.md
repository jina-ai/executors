# ✨ RedisStorage

**RedisStorage** is Indexer wrapper around the redis server. Redis is an open source (BSD licensed), in-memory data structure store, used as a database, cache, and message broker. You can read more about it here: https://redis.io


**Table of Contents**

- [🌱 Prerequisites](#-prerequisites)
- [🚀 Usages](#-usages)
- [🎉️ Example](#-example)
- [🔍️ Reference](#-reference)


## 🌱 Prerequisites

> These are only needed if you download the source code and directly use the class. Not needed if you use the Jina Hub method below.

- This Executor works on Python 3.7 and 3.8. 
- Make sure to install the [requirements](requirements.txt)

Additionally, you will need a running redis server. This can be a local instance, a Docker image, or a virtual machine in the cloud. To connect to redis, you need the hostname and the port. 

You can start one in a Docker container, like so: 

```bash
docker run -p 127.0.0.1:6379:6379/tcp -d redis
```

📕 **Note on docker network for macOS users**:  
If you run both the database and the `RedisStorage` docker container on the same machine 
localhost in the `RedisStorage` resolves to a separate network created by Docker which cannot see the database running on the host network.  
Use `host.docker.internal` to access localhost on the host machine.  
You can pass this parameter to the `RedisStorage` storage by using `override_with={'hostname': 'host.docker.internal''}` when
calling the `flow.add(...)` function.

## 🚀 Usages

This indexer does not allow indexing two documents with the same `ID` and will issue a warning. It also does not allow updating a document by a non-existing ID and will issue a warning.

### 🚚 Via JinaHub

#### using docker images
Use the prebuilt images from JinaHub in your Python code: 

```python
from jina import Flow
	
f = Flow().add(uses='jinahub+docker://RedisStorage')
```

or in the `.yml` config.
	
```yaml
jtype: Flow
pods:
  - name: indexer
    uses: 'jinahub+docker://RedisStorage'
```

#### using source code
Use the source code from JinaHub in your Python code:

```python
from jina import Flow
	
f = Flow().add(uses='jinahub://RedisStorage')
```

or in the `.yml` config.

```yaml
jtype: Flow
pods:
  - name: indexer
    uses: 'jinahub://RedisStorage'
```


## 🎉️ Example 


```python
from jina import Flow, Document

f = Flow().add(uses='jinahub://RedisStorage')

with f:
    resp = f.post(on='/index', inputs=Document(), return_results=True)
    print(f'{resp}')
```

### Inputs 

Any type of `Document`.

### Returns

Nothing. The `Documents` are stored.

## 🔍️ Reference

- https://redis.io
