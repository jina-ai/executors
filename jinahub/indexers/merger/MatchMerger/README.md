# MatchMerger

**MatchMerger** Merges the results of shards by appending all matches. Assume you have 20 shards and use `top-k=10`, you will get 200 results in the merger.
The `MatchMerger` is used in the `uses_after` attribute when adding an `Executor` to the `Flow`.




## Usage 

```python
from jina import Flow, Document

f = Flow().add(
    uses='jinahub+docker://SimpleIndexer', 
    shards=10,
    uses_after='jinahub+docker://MatchMerger'
)

with f:
    resp = f.post(on='/search', inputs=Document(), return_results=True)
    print(f'{resp}')
```

### Inputs 

`Document` with `.docs_matrix` of type `List[DocumentArray]`. It contains all the results calculated from the shards.

### Returns

`DocumentArray` which contains all the documents the user searched for including the combined matches from all the shards.

