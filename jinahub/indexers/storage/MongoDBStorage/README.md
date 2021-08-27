# MongoStorage

**MongoStorage** is a Jina indexer, using [mongodb](https://www.mongodb.com/) as a backend. 

`mongodb` is a no-sql document data model storage system. It supports JSON and provides an expressive query language for developers.

The class constructer receive the following parameters:

1. `host`: the mongodb instance host address, default value is `localhost`.
2. `port`: the mongodb instance port, default value is 27017.
3. `username`: the username of the instance, optional, default value is `None`.
4. `password`: the password of the instance, optional, default value is `None`.
5. `database`: the database name.
6. `collection`: the collection name.




## Prerequisites

Additionally, you will need a running MongoDB instnace. This can be a local instance, a Docker image, or a virtual machine in the cloud. Make sure you have the credentials and connection parameters.

You can start one in a Docker container, like so: 

```bash
docker run --name mongo-storage  -p 127.0.0.1:27017:27017/tcp -d mongo:latest
```

## Usages

### Via JinaHub

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

## Example 

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
