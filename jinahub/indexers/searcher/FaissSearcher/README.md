# FaissSearcher

**FaissSearcher** wraps Faiss-powered vector Searcher.

**FaissSearch** is defined as a vector searcher,
These usually implement a form of similarity search,
based on the embeddings created by the encoders you have chosen in your Flow.
Vector searcher is meant to be used together with a **Storage**.
To understand Jina's storage-search workflow,
please read the documentation [here](https://docs.jina.ai/advanced/experimental/indexers/).

[Faiss](https://github.com/facebookresearch/faiss) is a library for efficient similarity search and clustering of dense vectors.
It contains algorithms that search in sets of vectors of any size, up to ones that possibly do not fit in RAM.
It also contains supporting code for evaluation and parameter tuning.
Faiss is written in C++ with complete wrappers for Python/numpy.
Some of the most useful algorithms are implemented on the GPU.
It is developed by Facebook AI Research.


## Usage

### Loading data

Since this is a "Searcher"-type Executor, it does not _index_ new data.
Rather they are write-once classes, which take as data source a `dump_path`. 
Then we can perform search operations on the loaded data.

#### Loading data with Flow


```python
import numpy as np
from jina import Flow, Document

f = Flow().add(uses='jinahub+docker://FaissSearcher', uses_with={'dump_path': '/tmp/your_dump_location'})

with f:
    resp = f.post(on='/search', inputs=Document(embedding=np.array([1,2,3])), return_results=True)
    print(resp)
```

#### Loading data with `rolling_update`

To load data from the `Flow.rolling_update` method. See [docs](https://docs.jina.ai/advanced/experimental/indexers/).

### Training

To use a trainable Faiss indexer (e.g., _IVF_, _PQ_ based),
we can first train the indexer with the data from `train_data_file`:

```python
import numpy as np
from jina import Flow

train_data_file = 'train.npy'
train_data = np.array(np.random.random([10240, 256]), dtype=np.float32)
np.save(train_data_file, train_data)

f = Flow().add(
    uses='jinahub://FaissSearcher',
    timeout_ready=-1,
    uses_with={
      'index_key': 'IVF10_HNSW32,PQ64',
      'trained_index_file': 'faiss.index',
    },
)

with f:
    # the trained index will be dumped to "faiss.index"
    f.post(on='/train', parameters={'train_data_file': train_data_file})
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

## Reference

- [Facebook Research's Faiss](https://github.com/facebookresearch/faiss)
