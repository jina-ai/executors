# DocCache

**DocCache** is an Executor that can cache documents that it has seen before, by different combination of fields (or just one field). It then removes the Document that has the same combination of values in those fields from the DocumentArray, so it will not reach the following Executors in your Flow. 

This is useful for continuously indexing Documents, and not having to worry about indexing the same Document twice.

## Notes
The Executor only removes Documents in the `/index` endpoint. In the other endpoints, operations are done by the Document `id`.



## Usage 

In a Flow:

```python
from jina import Flow, DocumentArray, Document


docs = DocumentArray([
    Document(id=1, content='🐯'),
    Document(id=2, content='🐯'),
    Document(id=3, content='🐻'),
])

f = Flow(return_results=True).add(uses='jinahub+docker://DocCache')

with f:
    response = f.post(on='/index', inputs=docs, return_results=True)

    assert len(response[0].data.docs) == 2  # the duplicated Document is removed from the request
    assert set([doc.id for doc in response[0].data.docs]) == set(['1', '3'])

    docs_to_update = DocumentArray([
        Document(id=2, content='🐼')
    ])

    response = f.post(on='/update', inputs=docs_to_update, return_results=True)
    assert len(response[0].data.docs) == 1  # the Document with `id=2` is no longer duplicated.

    response = f.post(on='/index', inputs=docs[-1], return_results=True)
    assert len(response[0].data.docs) == 0  # the Document has been cached
    f.post(on='/delete', inputs=docs[-1])
    response = f.post(on='/index', inputs=docs[-1], return_results=True)
    assert len(response[0].data.docs) == 1  # the Document is cached again after the deletion
```

## Initialization
`fields` is the one or more [attributes of Document](https://github.com/jina-ai/jina/blob/master/.github/2.0/cookbooks/Document.md#document-attributes).
The value must be a tuple of strings (e.g. `[text, tags__author]`). The default value is `('content_hash', )`
