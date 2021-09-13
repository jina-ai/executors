# SimpleIndexer

**SimpleIndexer** is a Jina indexer, using the [DocumentArrayMemmap](https://github.com/jina-ai/jina/blob/master/jina/types/arrays/memmap.py) class as a storage system.

`DocumentArrayMemmap` stores the entire `Document` object, both vectors and metadata. It is also memory efficient, since it uses the [memmap module](https://docs.python.org/3.7/library/mmap.html) 




## Usage


```python
from jina import Flow, Document

f = Flow().add(uses='jinahub://SimpleIndexer')

with f:
    resp = f.post(on='/index', inputs=Document(), return_results=True)
    print(f'{resp}')
```

Parameters:

- `index_file_name`: the name of the folder where the memmaped data will be, under the workspace

### Inputs 

`Document`, with any data. It is stored in a `DocumentArrayMemmap`

### Returns

Nothing


