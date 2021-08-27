# HnswlibSearcher

**HnswlibSearcher** is a Hnswlib-powered vector Searcher.

Hnswlib is a fast approximate nearest neighbor search library and clustering of dense vectors.






## Usages

Check [tests](tests) for an example on how to use it.

### Loading data

Since this is a "Searcher"-type Executor, it does not _index_ new data. Rather they are write-once classes, which take as data source a `dump_path`. 

This can be provided in different ways:

- in the YAML definition
  
```yaml
jtype: HnswlibSearcher
with:
    dump_path: /tmp/your_dump_location
...
```

- from the `Flow.rolling_update` method. See [README](../../../../README.md).

The folder needs to contain the data exported from your Indexer. Again, see [README](../../../../README.md).

### Via JinaHub

#### using docker images
Use the prebuilt images from JinaHub in your Python code: 

```python
from jina import Flow
	
f = Flow().add(uses='jinahub+docker://HnswlibSearcher')
```

or in the `.yml` config.
	
```yaml
jtype: Flow
pods:
  - name: indexer
    uses: 'jinahub+docker://HnswlibSearcher'
```

#### using source code
Use the source code from JinaHub in your code

```python
from jina import Flow
	
f = Flow().add(uses='jinahub://HnswlibSearcher')
```

or in the `.yml` config.

```yaml
jtype: Flow
pods:
  - name: indexer
    uses: 'jinahub://HnswlibSearcher'
```

## Example 


```python
import numpy as np
from jina import Flow, Document

f = Flow().add(uses='jinahub+docker://HnswlibSearcher')

with f:
    resp = f.post(on='/search', inputs=Document(embedding=np.array([1,2,3])), return_results=True)
    print(f'{resp}')
```

### Inputs 

`Document` with `.embedding` the same shape as the `Documents` it has stored.

### Returns

Attaches matches to the Documents sent as inputs, with the id of the match, and its embedding. For retrieving the full metadata (original text or image blob), use a [key-value searcher](./../../keyvalue).


## Reference

- [Hnswlib](https://github.com/nmslib/hnswlib)
