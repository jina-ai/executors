# LMDBStorage

**LMDBStorage** is a Jina indexer, using [lmdb](https://lmdb.readthedocs.io/en/release/) as a backend. 

`lmdb` is a disk-based key-value storage system. It is quite performant. The test `test_lmdb_crud` in `tests/` ran with 100k docs in 1m 3secs



## Prerequisites

📕 **Note on docker network for macOS users**:  
If you run both the database and the `LMDBStorage` docker container on the same machine 
localhost in the `LMDBStorage` resolves to a separate network created by Docker which cannot see the database running on the host network.  
Use `host.docker.internal` to access localhost on the host machine.  
You can pass this parameter to the `LMDBStorage` storage by 
using `uses_with={'hostname': 'host.docker.internal''}` when
calling the `flow.add(...)` function.


## Reference
- https://lmdb.readthedocs.io/en/release/

<!-- version=v0.2 -->
