# FaissLMDBSearcher

**FaissLMDBSearcher** is a compound Searcher Executor for Jina, made up of [FaissSearcher](../../FaissSearcher) for performing similarity search on the embeddings, and of [FileSearcher](../../keyvalue/FileSearcher) for retrieving the metadata of the Documents. 





## Usage 

Check [integration tests](../../../../../tests/integration/lmdb_dump_reload) for an example on how to use it.

### Loading data

Since this is a "Searcher"-type Executor, it does not _index_ new data. Rather they are write-once classes, which take as data source a `dump_path`. 

This can be provided in different ways:

- in the YAML definition
  
```yaml
jtype: FaissLMDBSearcher
with:
    dump_path: /tmp/your_dump_location
...
```

- from the `Flow.rolling_update` method. See [docs](https://docs.jina.ai/fundamentals/executor/indexers/).

The folder needs to contain the data exported from your Indexer. Again, see [docs](https://docs.jina.ai/fundamentals/executor/indexers/).




```python
from jina import Flow, Document

f = Flow().add(uses='jinahub+docker://FaissLMDBSearcher')

with f:
    resp = f.post(on='/search', inputs=Document(), return_results=True)
    print(f'{resp}')
```

### Inputs 

`Document` with `.embedding` the same shape as the `Documents` stored in the `FaissSearcher`. The ids of the `Documents` stored in `FaissSearcher` need to exist in the `FileSearcher`. Otherwise you will not get back the original metadata. 

### Returns

The FaissSearcher attaches matches to the Documents sent as inputs, with the id of the match, and its embedding.
Then, the FileSearcher retrieves the full metadata (original text or image blob) and attaches those to the Document.
You receive back the full Document.

