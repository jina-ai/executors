# HnswlibSearcher

**HnswlibSearcher** is a vector searcher and indexer, based on the `hnswlib` library.

It uses the state-of-the-art HNSW aproximate nearest neighbors algorithm to find matches for query documents. The main advantage of this searcher (compared to searchers like FAISS) is that it does not require training, and has native support for incremental indexing.

This indexer has full support for CRUD operations, although only soft delete is possible.

## Usage

## Index and search

This example shows a common usage pattern where we first index some documents, and then
perform search on the index. 

Note that to achieved the desired tradeoff between index and query
time on one hand, and search accuracy on the other, you will need to "finetune" the
index parameters. For more information on that, see [hnswlib documentation](https://github.com/nmslib/hnswlib/blob/master/ALGO_PARAMS.md).

```python
import numpy as np
from jina import Document, Flow


def generate_docs(num_docs):
    def _gen_fn():
        for _ in range(num_docs):
            yield Document(embedding=np.random.rand(512))

    return _gen_fn


f = Flow().add(
    uses='jinahub+docker://HnswlibSearcher', uses_with={'dim': 512, 'top_k': 100}
)
with f:
    # Index 10k docs
    f.index(generate_docs(10_000), show_progress=True, request_size=100)

    # Check that we have 10k docs in index
    status = f.post('/status', return_results=True)
    num_docs = int(status[0].data.docs[0].tags['current_indexed'])
    print(f'Indexed {num_docs} documents')

    # Search for some docs
    f.search(generate_docs(1000), show_progress=True)
```


## Save and load

This example shows how to save (dump) the index, and then re-create the executor based
on the saved index.

```python
import numpy as np
from jina import Document, Flow


def generate_docs(num_docs):
    def _gen_fn():
        for _ in range(num_docs):
            yield Document(embedding=np.random.rand(512))

    return _gen_fn


f = Flow().add(
    uses='jinahub+docker://HnswlibSearcher', uses_with={'dim': 512, 'top_k': 100}
)
with f:
    # Index 10k docs and save index
    f.index(generate_docs(10_000), show_progress=True, request_size=100)
    f.post('/dump', parameters={'dump_path': '.'})

# Create new flow so that Hnswlibsearcher loads dumped files on start
f = Flow().add(
    uses='jinahub+docker://HnswlibSearcher', 
    uses_with={'dim': 512, 'top_k': 100, 'dump_path': '.'}
)
with f:
    # Check that index was properly re-built
    status = f.post('/status', return_results=True)
    num_docs = int(status[0].data.docs[0].tags['current_indexed'])
    print(f'{num_docs} documents in index')
```

## Reference

- [Hnswlib](https://github.com/nmslib/hnswlib)

<!-- version=v0.2 -->
