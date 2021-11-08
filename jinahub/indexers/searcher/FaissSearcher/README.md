# FaissSearcher

**FaissSearcher** wraps Faiss-powered vector Searcher.

**FaissSearch** is defined as a vector searcher,
These usually implement a form of similarity search,
based on the embeddings created by the encoders you have chosen in your Flow.
Vector searcher is meant to be used together with a **Storage**.
To understand Jina's storage-search workflow,
please read the documentation [here](https://docs.jina.ai/advanced/experimental/indexers/).

[Faiss](https://github.com/facebookresearch/faiss) is a library for efficient similarity 
search and clustering of dense vectors.


## Usage

### Training

To use a trainable Faiss indexer (e.g., _IVF_, _PQ_ based),
we can first train the indexer with pure `faiss` api:

```python
import numpy as np
from jina import Flow
import faiss

train_data = np.array(np.random.random([10240, 256]), dtype=np.float32)

trained_index_file = 'faiss.index'

faiss_index = faiss.index_factory(256, 'IVF64,SQ8', faiss.METRIC_INNER_PRODUCT)
faiss_index.train(train_data)
faiss.write_index(faiss_index, trained_index_file)

f = Flow().add(
    uses='jinahub://FaissSearcher',
    timeout_ready=-1,
    uses_with={
      'index_key': 'IVF64,SQ8',
      'trained_index_file': 'faiss.index',
    },
)
```

Then, we can directly use the trained indexer with providing `trained_index_file`:

```python
from jina import Flow

f = Flow().add(
    uses='jinahub://FaissSearcher',
    timeout_ready=-1,
    uses_with={
      'index_key': 'IVF10_HNSW32,PQ64',
      'trained_index_file': 'faiss.index',
      'dump_path': '/path/to/dump_file'
    },
)
```

### Loading data

Since this is a "Searcher"-type Executor, it does not _index_ new data.
Rather they are write-once classes, which take as data source a `dump_path`.
Then we can perform search operations on the loaded data.
Check out the 
[documentation](https://docs.jina.ai/advanced/experimental/indexers/#indexing-vs-searching-operations) 
for more details

## Reference

- [Facebook Research's Faiss](https://github.com/facebookresearch/faiss)


<!-- version=v0.4 -->
