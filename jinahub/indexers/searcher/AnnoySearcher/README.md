# AnnoySearcher

**AnnoySearcher** is an Annoy-powered vector-based similarity searcher. Annoy stands for "Approximate Nearest Neighbors Oh Yeah", and is a C++ library with Python bindings to search for points in space that are close to a given query point. It also creates large read-only file-based data structures that are mmapped into memory so that many processes may share the same data.

For more information, refer to the GitHub repo for [Spotify's Annoy](https://github.com/spotify/annoy).



## Usage

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

- from the `Flow.rolling_update` method. See [docs](https://docs.jina.ai/fundamentals/executor/indexers/).

The folder needs to contain the data exported from your Indexer. Again, see [docs](https://docs.jina.ai/fundamentals/executor/indexers/).



## Usage 

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


## Reference

- https://github.com/spotify/annoy
