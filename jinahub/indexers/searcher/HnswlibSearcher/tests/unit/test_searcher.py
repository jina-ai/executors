from pathlib import Path

import numpy as np
import pytest
from jina import Document, DocumentArray, Executor
from hnswlib_searcher import HnswlibSearcher

_DIM = 10


def test_config():
    ex = Executor.load_config(str(Path(__file__).parents[2] / 'config.yml'))
    assert ex.metric == 'cosine'


def test_empty_search():
    index = HnswlibSearcher(dim=_DIM)
    da = DocumentArray([Document(embedding=np.random.normal(size=(_DIM,)))])

    index.search(da, {})
    assert len(da[0].matches) == 0


def test_index():
    NUM_DOCS = 1000
    index = HnswlibSearcher(dim=_DIM)
    embeddings = np.random.normal(size=(NUM_DOCS, _DIM))
    da1 = DocumentArray([Document(embedding=emb) for emb in embeddings])
    da2 = DocumentArray([Document(embedding=emb) for emb in embeddings])

    index.index(da1, {})
    assert len(index._ids_to_inds) == NUM_DOCS
    assert index._index.element_count == NUM_DOCS

    index.index(da2, {})
    assert len(index._ids_to_inds) == 2 * NUM_DOCS
    assert index._index.element_count == 2 * NUM_DOCS


def test_index_wrong_dim():
    index = HnswlibSearcher(dim=10)
    embeddings = np.random.normal(size=(2, 11))
    da1 = DocumentArray([Document(embedding=emb) for emb in embeddings])

    with pytest.raises(ValueError, match='Attempted to index'):
        index.index(da1, {})


@pytest.mark.parametrize('metric', ['cosine', 'l2', 'ip'])
@pytest.mark.parametrize('top_k', [5, 10])
def test_search_basic(metric: str, top_k: int):
    index = HnswlibSearcher(dim=_DIM, metric=metric, top_k=top_k)
    embeddings_ind = np.random.normal(size=(1000, _DIM))
    embeddings_search = np.random.normal(size=(10, _DIM))
    da_index = DocumentArray([Document(embedding=emb) for emb in embeddings_ind])
    da_search = DocumentArray([Document(embedding=emb) for emb in embeddings_search])

    index.index(da_index, {})
    index.search(da_search, {})

    indexed_ids = da_index.get_attributes('id')

    for d in da_search:
        ms = d.matches
        scores = [m.scores[metric].value for m in ms]
        assert len(ms) == top_k
        assert sorted(scores) == scores
        for m in ms:
            assert m.id in indexed_ids


def test_search_quality():
    pass


def test_search_wrong_dim():
    pass


def test_update():
    pass


def test_update_wrong_dim():
    pass


def test_delete():
    pass


def test_clear():
    pass


def test_dump():
    pass


def test_dump_no_path():
    pass


def test_dump_load():
    pass
