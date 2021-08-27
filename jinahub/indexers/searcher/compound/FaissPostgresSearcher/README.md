# FaissPostgresSearcher

**FaissPostgresSearcher** is a compound Searcher Executor for Jina, made up of [FaissSearcher](../../FaissSearcher) for performing similarity search on the embeddings, and of [PostgresSearcher](../../keyvalue/PostgresSearcher) for retrieving the metadata of the Documents. 




Additionally, you will need a running PostgreSQL database. This can be a local instance, a Docker image, or a virtual machine in the cloud. Make sure you have the credentials and connection parameters.

You can start one in a Docker container, like so: 

```bash
docker run -e POSTGRES_PASSWORD=123456  -p 127.0.0.1:5432:5432/tcp postgres:13.2 
```

## Usages

Check [integration tests](../../../../../tests/integration/psql_dump_reload) for an example on how to use it.

### Loading data

Since this is a "Searcher"-type Executor, it does not _index_ new data. Rather they are write-once classes, which take as data source a `dump_path`. 

This can be provided in different ways:

- in the YAML definition
  
```yaml
jtype: FaissPostgresSearcher
with:
    dump_path: /tmp/your_dump_location
...
```

- from the `Flow.rolling_update` method. See [README](../../../../../README.md).

The folder needs to contain the data exported from your Indexer. Again, see [README](../../../../../README.md).

### Via JinaHub

#### using docker images
Use the prebuilt images from JinaHub in your Python code: 

```python
from jina import Flow
	
f = Flow().add(uses='jinahub+docker://FaissPostgresSearcher')
```

or in the `.yml` config.
	
```yaml
jtype: Flow
pods:
  - name: indexer
    uses: 'jinahub+docker://FaissPostgresSearcher'
```

#### using source code
Use the source code from JinaHub in your code

```python
from jina import Flow
	
f = Flow().add(uses='jinahub://FaissPostgresSearcher')
```

or in the `.yml` config.

```yaml
jtype: Flow
pods:
  - name: indexer
    uses: 'jinahub://FaissPostgresSearcher'
```


## Example 


```python
from jina import Flow, Document

f = Flow().add(uses='jinahub+docker://FaissPostgresSearcher')

with f:
    resp = f.post(on='/search', inputs=Document(), return_results=True)
    print(f'{resp}')
```

### Inputs 

`Document` with `.embedding` the same shape as the `Documents` stored in the `FaissSearcher`. The ids of the `Documents` stored in `FaissSearcher` need to exist in the `PostgresSearcher`. Otherwise you will not get back the original metadata. 

### Returns

The FaissSearcher attaches matches to the Documents sent as inputs, with the id of the match, and its embedding.
Then, the PostgresSearcher retrieves the full metadata (original text or image blob) and attaches those to the Document.
You receive back the full Document.



