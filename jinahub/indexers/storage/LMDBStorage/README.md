# LMDBStorage

**LMDBStorage** is a Jina indexer, using [lmdb](https://lmdb.readthedocs.io/en/release/) as a backend. 

`lmdb` is a disk-based key-value storage system. It is quite performant. The test `test_lmdb_crud` in `tests/` ran with 100k docs in 1m 3secs



## Prerequisites


📕 **Note on docker network for macOS users**:  
If you run both the database and the `LMDBStorage` docker container on the same machine 
localhost in the `LMDBStorage` resolves to a separate network created by Docker which cannot see the database running on the host network.  
Use `host.docker.internal` to access localhost on the host machine.  
You can pass this parameter to the `LMDBStorage` storage by using `uses_with={'hostname': 'host.docker.internal''}` when
calling the `flow.add(...)` function.

## Usages

### Via JinaHub

#### using docker images

Use the prebuilt images from JinaHub in your Python code: 

```python
from jina import Flow
	
f = Flow().add(uses='jinahub+docker://LMDBStorage')
```

or in the `.yml` config.
	
```yaml
jtype: Flow
pods:
  - name: indexer
    uses: 'jinahub+docker://LMDBStorage'
```

#### using source code
Use the source code from JinaHub in your Python code:

```python
from jina import Flow
	
f = Flow().add(uses='jinahub://LMDBStorage')
```

or in the `.yml` config.

```yaml
jtype: Flow
pods:
  - name: indexer
    uses: 'jinahub://LMDBStorage'
```

## Example 

```python
from jina import Flow, Document

f = Flow().add(uses='jinahub://LMDBStorage')

with f:
    resp = f.post(on='/index', inputs=Document(), return_results=True)
    print(f'{resp}')
```

Parameters:

- `map_size`: maximum size of the database on disk
- `default_traversal_paths`: the default traversal paths for the `DocumentArray` in a request. Can be overridden with `parameters={'traversal_paths': ..}` 

Check [tests](tests/test_lmdb.py) for more usage scenarios.


### Inputs 

`Document`, with any data. It is stored in full, in bytes.

### Returns

Nothing

## Reference
- https://lmdb.readthedocs.io/en/release/
