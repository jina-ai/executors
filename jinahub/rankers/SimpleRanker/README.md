# SimpleRanker

**SimpleRanker** is a class aggregates the score of the matched doc from the matched chunks.


## Usage 

```python
from jina import Flow, DocumentArray, Document
import random

document_array = DocumentArray()
document = Document(tags={'query_size': 35, 'query_price': 31, 'query_brand': 1})
for i in range(0, 10):
    chunk = Document()
    for j in range(0, 10):
        match = Document(
            tags={
                'level': 'chunk',
            }
        )
        match.scores['cosine'] = random.random()
        match.parent_id = i
        chunk.matches.append(match)
    document.chunks.append(chunk)

document_array.extend([document])

f = Flow().add(uses='jinahub://SimpleRanker', uses_with={'metric': 'cosine'})

with f:
    resp = f.post(on='/search', inputs=document_array, return_results=True)
    print(f'{resp[0].data.docs[0].matches}')

```


## Reference
- See the [multires lyrics search example](https://github.com/jina-ai/examples/tree/master/multires-lyrics-search) for example usage
