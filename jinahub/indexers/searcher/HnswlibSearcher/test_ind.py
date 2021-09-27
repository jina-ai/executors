import numpy as np
from jina import Document, DocumentArray
from hnswlib_searcher import HnswlibSearcher

index = HnswlibSearcher(dim=20)

# Index
docs_to_index = DocumentArray(
    [
        Document(id='a', embedding=np.ones((20,))),
        Document(id='b', embedding=np.ones((20,)) * 2),
        Document(id='c', embedding=np.ones((20,)) * 3),
        Document(id='d', embedding=np.ones((20,)) * 4),
        Document(id='e', embedding=np.ones((20,)) * 5),
    ]
)
index.index(docs_to_index)
print(index._ids_to_inds)  # 5 items


# Update
d3 = docs_to_index[2]
d3.embedding = np.ones((20,)) * 3.3
index.update(DocumentArray([d3]))

_s = DocumentArray([d3])
index.search(_s, {'top_k': 1})
for m in _s[0].matches:
    print(m.scores['l2'].value, m.id)

# Search
search_docs = DocumentArray([Document(embedding=np.ones((20,)) * 1.9)])
index.search(search_docs)
for m in search_docs[0].matches:
    print(m.scores['l2'], m.id)

# Delete
index.delete({'ids': ['c']})
print(index._ids_to_inds)
_s = DocumentArray([d3])
_s[0].pop('matches')
index.search(_s, {'top_k': 1})
for m in _s[0].matches:
    print(m.scores['l2'].value, m.id)

# Save
index.dump({'dump_path': '.'})

# Clear
index.clear()
print(index._ids_to_inds)
print(index._index.element_count)
_s[0].pop('matches')
index.search(_s, {'top_k': 1})
print(_s[0].matches)

# Load
index = HnswlibSearcher(dim=20, dump_path='.')
print(index._ids_to_inds)
print(index._index.element_count)
